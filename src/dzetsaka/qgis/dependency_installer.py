"""Dependency installation flow extracted from plugin runtime."""

from __future__ import annotations

import contextlib
import os
import tempfile

from qgis.core import QgsApplication
from qgis.PyQt.QtWidgets import QMessageBox

from dzetsaka.qgis.dependency_catalog import FULL_DEPENDENCY_BUNDLE

_SCIENTIFIC_STACK_PACKAGES = ("numpy", "scipy", "pandas")


def _build_runtime_constraints_file(plugin_logger) -> tuple[str | None, list[str]]:
    """Build a pip constraints file pinned to the live runtime scientific stack."""
    try:
        try:
            from importlib import metadata as importlib_metadata
        except ImportError:
            import importlib_metadata  # type: ignore

        pinned_packages = []  # type: list[str]
        for pkg in _SCIENTIFIC_STACK_PACKAGES:
            version = None
            try:
                version = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                # QGIS-bundled packages can be importable without package metadata.
                pass
            except Exception as metadata_err:
                plugin_logger.warning(f"Could not read {pkg} version from package metadata: {metadata_err!s}")

            if not version:
                with contextlib.suppress(Exception):
                    module = __import__(pkg)
                    version = getattr(module, "__version__", None)

            if version:
                pinned_packages.append(f"{pkg}=={version}")

        if not pinned_packages:
            return None, []

        constraints_text = "\n".join(pinned_packages) + "\n"
        fd, runtime_constraints_file = tempfile.mkstemp(
            prefix="dzetsaka_pip_constraints_",
            suffix=".txt",
        )
        os.close(fd)
        with open(runtime_constraints_file, "w", encoding="utf-8") as f:
            f.write(constraints_text)

        plugin_logger.info(
            f"Using runtime pip constraints to keep scientific stack stable: {', '.join(pinned_packages)}",
        )
        return runtime_constraints_file, ["-c", runtime_constraints_file]
    except Exception as constraints_err:
        plugin_logger.warning(f"Could not prepare runtime pip constraints: {constraints_err!s}")
        return None, []


def try_install_dependencies(plugin, missing_deps):
    """Experimental feature to auto-install missing dependencies.

    Parameters
    ----------
    missing_deps : list
        List of missing dependency names

    Returns
    -------
    bool
        True if installation succeeded, False otherwise

    """
    if not missing_deps:
        return True

    # Normalize and deduplicate dependency names to avoid repeated install attempts.
    normalized_missing_deps = []
    seen = set()
    for dep in missing_deps:
        dep_norm = str(dep).strip()
        if not dep_norm:
            continue
        dep_key = dep_norm.lower()
        if dep_key in seen:
            continue
        seen.add(dep_key)
        normalized_missing_deps.append(dep_norm)
    requested_deps = normalized_missing_deps
    missing_deps = FULL_DEPENDENCY_BUNDLE.copy()
    plugin.log.info(
        f"All-dependencies install mode enabled. Requested={requested_deps!r}, full bundle target={missing_deps!r}",
    )

    from qgis.PyQt.QtCore import QEventLoop, QProcess

    from dzetsaka.ui.install_progress_dialog import InstallProgressDialog

    # Build conservative constraints from the live runtime to avoid breaking
    # core scientific stack packages in embedded QGIS envs.
    runtime_constraints_file, runtime_constraint_args = _build_runtime_constraints_file(plugin.log)

    # Package installation using QProcess for responsive UI
    def install_package(package, progress_dialog, extra_args=None):
        import glob
        import os
        import shutil
        import sys

        def run_command(cmd, description):
            """Run a command using QProcess and return (success, output)."""
            try:
                plugin.log.info(f"Trying {description}: {' '.join(cmd)}")
                progress_dialog.append_output(f"\n$ {' '.join(cmd)}\n")

                process = QProcess()
                process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

                # Connect signals for live output
                output_lines = []

                def handle_output():
                    data = process.readAllStandardOutput()
                    text = bytes(data).decode("utf-8", errors="replace")
                    if text:
                        output_lines.append(text)
                        progress_dialog.append_output(text)
                        plugin.log.info(f"  {text.strip()}")

                process.readyReadStandardOutput.connect(handle_output)

                # Start the process
                process.start(cmd[0], cmd[1:])

                # Wait for process to finish while keeping UI responsive
                loop = QEventLoop()
                process.finished.connect(loop.quit)

                # Check for cancellation
                def check_cancel():
                    if progress_dialog.was_cancelled():
                        process.terminate()
                        process.waitForFinished(3000)
                        if process.state() == QProcess.ProcessState.Running:
                            process.kill()
                        loop.quit()

                from qgis.PyQt.QtCore import QTimer

                cancel_timer = QTimer()
                cancel_timer.timeout.connect(check_cancel)
                cancel_timer.start(100)  # Check every 100ms

                # Hard timeout to avoid indefinite hangs that can look like a frozen QGIS UI.
                timeout_ms = 900000
                timed_out = {"value": False}

                def handle_timeout():
                    timed_out["value"] = True
                    progress_dialog.append_output(
                        f"\n⚠ Command timed out after {int(timeout_ms / 1000)}s, terminating...\n",
                    )
                    process.terminate()
                    process.waitForFinished(3000)
                    if process.state() == QProcess.ProcessState.Running:
                        process.kill()
                    loop.quit()

                timeout_timer = QTimer()
                timeout_timer.setSingleShot(True)
                timeout_timer.timeout.connect(handle_timeout)
                timeout_timer.start(timeout_ms)

                if not process.waitForStarted(5000):
                    cancel_timer.stop()
                    timeout_timer.stop()
                    plugin.log.error(f"Failed to start process: {cmd[0]}")
                    return False, "Failed to start process"

                if hasattr(loop, "exec"):
                    loop.exec()
                else:
                    loop.exec_()
                cancel_timer.stop()
                timeout_timer.stop()

                # Check if cancelled
                if progress_dialog.was_cancelled():
                    return False, "Cancelled by user"
                if timed_out["value"]:
                    return False, "Timed out while waiting for dependency installer process"

                # Read any remaining output
                handle_output()

                exit_code = process.exitCode()
                exit_status = process.exitStatus()
                output_text = "".join(output_lines)

                # Debug logging
                plugin.log.info(f"Process exit code: {exit_code}, exit status: {exit_status}")
                progress_dialog.append_output(f"\nExit code: {exit_code}, Exit status: {exit_status}\n")

                # Check both exit status (normal exit) and exit code (0 = success)
                # exitStatus() returns 0 for NormalExit, 1 for CrashExit
                try:
                    normal_exit = exit_status == QProcess.ExitStatus.NormalExit
                except AttributeError:
                    # Qt5 compatibility - enum value is 0 for NormalExit
                    normal_exit = exit_status == 0

                success = normal_exit and exit_code == 0
                plugin.log.info(f"Process success: {success} (normal_exit={normal_exit}, exit_code={exit_code})")
                return success, output_text

            except Exception as e:
                plugin.log.error(f"Error running command: {e}")
                return False, str(e)

        # Build candidate Python launchers (QGIS on Windows often has no plain "python" in PATH).
        def find_python_candidates():
            candidates = []
            qgis_macos_python_dir = None

            def _add(path):
                if path and path not in candidates:
                    candidates.append(path)

            def _looks_like_python_launcher(path):
                if path in {"python", "python3", "py"}:
                    return True
                base = os.path.basename(str(path)).lower()
                if base in {"python.exe", "python3.exe", "py.exe", "python", "python3"}:
                    return True
                return base.startswith("python")

            if sys.platform == "darwin":
                with contextlib.suppress(Exception):
                    app_path = QgsApplication.applicationFilePath() or ""
                    if app_path:
                        qgis_macos_python_dir = os.path.dirname(app_path)
                if qgis_macos_python_dir:
                    runtime_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
                    _add(os.path.join(qgis_macos_python_dir, f"python{runtime_mm}"))
                    _add(os.path.join(qgis_macos_python_dir, "python3"))
                    _add(os.path.join(qgis_macos_python_dir, "python"))

            if sys.executable:
                _add(sys.executable)
                _add(os.path.join(os.path.dirname(sys.executable), "python.exe"))
                _add(os.path.join(os.path.dirname(sys.executable), "python3.exe"))
            _add(os.path.join(sys.prefix, "python.exe"))
            _add(os.path.join(sys.base_prefix, "python.exe"))
            _add(os.path.join(os.path.dirname(os.__file__), "..", "python.exe"))

            osgeo_root = os.environ.get("OSGEO4W_ROOT", "")
            if osgeo_root:
                _add(os.path.join(osgeo_root, "bin", "python3.exe"))
                _add(os.path.join(osgeo_root, "bin", "python.exe"))
                for py in glob.glob(os.path.join(osgeo_root, "apps", "Python*", "python.exe")):
                    _add(py)

            # Legacy heuristic kept as a fallback.
            for path in sys.path:
                _add(os.path.join(path, "python.exe"))
                _add(os.path.join(path, "python3.exe"))

            for cmd in ("python", "python3", "py"):
                found = shutil.which(cmd)
                if found:
                    _add(found)

            filtered = []
            for c in candidates:
                if not ((os.path.isabs(c) and os.path.isfile(c)) or c in {"python", "python3", "py"}):
                    continue
                if not _looks_like_python_launcher(c):
                    continue
                filtered.append(c)
            return filtered, qgis_macos_python_dir

        def validate_python_candidates(raw_candidates, qgis_macos_python_dir=None):
            validated = []
            expected_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
            for py in raw_candidates:
                probe_cmd = [
                    py,
                    "-c",
                    (
                        "import sys;"
                        "print('DZETSAKA_PY_OK|%d.%d|%s' % (sys.version_info[0], sys.version_info[1], sys.executable))"
                    ),
                ]
                success, output = run_command(probe_cmd, f"python probe via {py}")
                marker_line = ""
                if success:
                    for line in (output or "").splitlines():
                        if line.startswith("DZETSAKA_PY_OK|"):
                            marker_line = line.strip()
                            break
                if not marker_line:
                    plugin.log.warning(f"Skipping non-python launcher candidate: {py}")
                    progress_dialog.append_output(f"⚠ Skipping non-python launcher: {py}\n")
                    continue

                parts = marker_line.split("|", 2)
                candidate_mm = parts[1] if len(parts) >= 2 else ""
                if candidate_mm != expected_mm:
                    plugin.log.warning(
                        f"Skipping python candidate with mismatched version ({candidate_mm} != {expected_mm}): {py}",
                    )
                    progress_dialog.append_output(
                        f"⚠ Skipping python candidate (version {candidate_mm}, expected {expected_mm}): {py}\n",
                    )
                    continue
                validated.append(py)

            if sys.platform == "darwin" and qgis_macos_python_dir:
                qgis_prefix = os.path.abspath(qgis_macos_python_dir) + os.sep
                qgis_validated = [
                    py for py in validated if os.path.isabs(py) and os.path.abspath(py).startswith(qgis_prefix)
                ]
                if qgis_validated:
                    return qgis_validated
            return validated

        def ensure_pip_available(py):
            """Bootstrap pip via get-pip.py if missing for *py*."""
            probe_cmd = [py, "-m", "pip", "--version"]
            success, _ = run_command(probe_cmd, f"pip probe via {py}")
            if success:
                return
            plugin.log.info(f"pip not found for {py}, bootstrapping via get-pip.py")
            progress_dialog.append_output(f"\n⚠ pip not found for {py}, downloading get-pip.py...\n")
            try:
                import tempfile
                import urllib.request

                fd, get_pip_path = tempfile.mkstemp(suffix=".py", prefix="dzetsaka_get_pip_")
                os.close(fd)
                try:
                    urllib.request.urlretrieve(  # nosec B310
                        "https://bootstrap.pypa.io/get-pip.py",
                        get_pip_path,
                    )
                    bootstrap_cmd = [py, get_pip_path, "--user", "--break-system-packages"]
                    ok, out = run_command(bootstrap_cmd, f"get-pip.py via {py}")
                    if ok:
                        plugin.log.info(f"Successfully bootstrapped pip for {py}")
                        progress_dialog.append_output(f"✓ pip bootstrapped for {py}\n")
                    else:
                        plugin.log.warning(f"get-pip.py failed for {py}")
                        progress_dialog.append_output(f"✗ get-pip.py failed for {py}\n")
                finally:
                    with contextlib.suppress(Exception):
                        os.remove(get_pip_path)
            except Exception as exc:
                plugin.log.warning(f"pip bootstrap error: {exc!s}")

        raw_candidates, qgis_macos_python_dir = find_python_candidates()
        python_candidates = validate_python_candidates(raw_candidates, qgis_macos_python_dir)
        plugin.log.info(f"Installing {package}. Python candidates: {python_candidates!r}")
        progress_dialog.append_output(
            "Python candidates: " + (", ".join(python_candidates) if python_candidates else "<none found>"),
        )

        # Ensure pip is available (bootstraps via get-pip.py if missing)
        for py in python_candidates:
            ensure_pip_available(py)

        # Check for cancellation before starting
        if progress_dialog.was_cancelled():
            return False

        pip_args = [
            "-m",
            "pip",
            "install",
            package,
            "--user",
            "--no-input",
            "--disable-pip-version-check",
            "--prefer-binary",
            "--break-system-packages",
        ]
        if runtime_constraint_args:
            pip_args.extend(runtime_constraint_args)
        if extra_args:
            pip_args.extend(extra_args)

        attempts = []
        for py in python_candidates:
            attempts.append(([py, *pip_args], f"pip module via {py}"))
            attempts.append(([py, "-m", "ensurepip", "--user"], f"ensurepip via {py}"))
            attempts.append(([py, *pip_args], f"pip module after ensurepip via {py}"))

        # Shell launcher fallbacks (common on Windows/QGIS environments).
        has_qgis_macos_candidate = (
            sys.platform == "darwin"
            and qgis_macos_python_dir
            and any(
                os.path.isabs(py) and os.path.abspath(py).startswith(os.path.abspath(qgis_macos_python_dir) + os.sep)
                for py in python_candidates
            )
        )
        if not has_qgis_macos_candidate:
            attempts.extend(
                [
                    (["py", "-3", *pip_args], "pip via py -3"),
                    (["py", *pip_args], "pip via py"),
                    (["python3", *pip_args], "pip via python3"),
                    (["python", *pip_args], "pip via python"),
                ],
            )

        last_output = ""
        for cmd, description in attempts:
            # Avoid running ensurepip unless pip is really missing for that attempt.
            if "ensurepip" in cmd and (
                "No module named pip" not in last_output and "pip is not installed" not in last_output.lower()
            ):
                continue
            success, output = run_command(cmd, description)
            last_output = output or ""
            if success and "ensurepip" not in cmd:
                plugin.log.info(f"Successfully installed {package} using: {' '.join(cmd)}")
                progress_dialog.append_output(f"\n✓ install_package returning True for {package}\n")
                return True

        plugin.log.warning(f"Installation failed for {package} after all launcher attempts")
        progress_dialog.append_output(f"\n✗ install_package returning False for {package}\n")

        # All methods failed
        plugin.log.error(
            f"Could not install {package}. Please install manually:\n"
            "  pip3 install --user {package}\n"
            "  Then restart QGIS.",
        )
        return False

    def install_package_bundle(packages, progress_dialog, extra_args=None):
        """Install multiple packages in one pip invocation."""
        import glob
        import os
        import re
        import shutil
        import sys

        targets = [str(p).strip() for p in packages if str(p).strip()]
        if not targets:
            return True
        target_norms = [t.lower().replace("_", "-") for t in targets]

        completed_targets = set()
        collecting_targets = set()

        def _normalize_package_token(token):
            value = str(token or "").strip().lower().replace("_", "-")
            value = re.split(r"[<>=!~\[\],;]", value)[0]
            value = value.strip()
            if not value:
                return ""
            # pip often prints installed package with version suffix (e.g., "xgboost-2.1.2")
            m = re.match(r"^([a-z0-9\-]+?)-\d", value)
            return m.group(1) if m else value

        def _match_target(token):
            norm = _normalize_package_token(token)
            if not norm:
                return None
            for i, target in enumerate(target_norms):
                if norm == target or norm.startswith(target):
                    return i
            return None

        def run_command(cmd, description):
            """Run a command using QProcess and return (success, output)."""
            try:
                plugin.log.info(f"Trying {description}: {' '.join(cmd)}")
                progress_dialog.append_output(f"\n$ {' '.join(cmd)}\n")

                process = QProcess()
                process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

                output_lines = []

                def handle_output():
                    data = process.readAllStandardOutput()
                    text = bytes(data).decode("utf-8", errors="replace")
                    if text:
                        output_lines.append(text)
                        progress_dialog.append_output(text)
                        plugin.log.info(f"  {text.strip()}")
                        for raw_line in text.splitlines():
                            line = raw_line.strip()
                            if not line:
                                continue
                            # Phase-aware progress parsing from pip logs.
                            if line.startswith("Collecting "):
                                token = line[len("Collecting ") :].split()[0]
                                idx = _match_target(token)
                                if idx is not None:
                                    collecting_targets.add(targets[idx])
                                    progress_dialog.status_label.setText(
                                        f"Installing bundle... detected {len(collecting_targets)}/{len(targets)} packages",
                                    )
                            elif "Successfully installed" in line:
                                suffix = line.split("Successfully installed", 1)[1]
                                for token in suffix.split():
                                    idx = _match_target(token)
                                    if idx is None:
                                        continue
                                    target_name = targets[idx]
                                    if target_name in completed_targets:
                                        continue
                                    completed_targets.add(target_name)
                                    progress_dialog.set_current_package(target_name, idx)
                                    progress_dialog.mark_package_complete()
                                    progress_dialog.status_label.setText(
                                        f"Installing bundle... completed {len(completed_targets)}/{len(targets)}",
                                    )

                process.readyReadStandardOutput.connect(handle_output)
                process.start(cmd[0], cmd[1:])

                loop = QEventLoop()
                process.finished.connect(loop.quit)

                def check_cancel():
                    if progress_dialog.was_cancelled():
                        process.terminate()
                        process.waitForFinished(3000)
                        if process.state() == QProcess.ProcessState.Running:
                            process.kill()
                        loop.quit()

                from qgis.PyQt.QtCore import QTimer

                cancel_timer = QTimer()
                cancel_timer.timeout.connect(check_cancel)
                cancel_timer.start(100)

                timeout_ms = 900000
                timed_out = {"value": False}

                def handle_timeout():
                    timed_out["value"] = True
                    progress_dialog.append_output(
                        f"\n⚠ Command timed out after {int(timeout_ms / 1000)}s, terminating...\n",
                    )
                    process.terminate()
                    process.waitForFinished(3000)
                    if process.state() == QProcess.ProcessState.Running:
                        process.kill()
                    loop.quit()

                timeout_timer = QTimer()
                timeout_timer.setSingleShot(True)
                timeout_timer.timeout.connect(handle_timeout)
                timeout_timer.start(timeout_ms)

                if not process.waitForStarted(5000):
                    cancel_timer.stop()
                    timeout_timer.stop()
                    plugin.log.error(f"Failed to start process: {cmd[0]}")
                    return False, "Failed to start process"

                if hasattr(loop, "exec"):
                    loop.exec()
                else:
                    loop.exec_()
                cancel_timer.stop()
                timeout_timer.stop()

                if progress_dialog.was_cancelled():
                    return False, "Cancelled by user"
                if timed_out["value"]:
                    return False, "Timed out while waiting for dependency installer process"

                handle_output()

                exit_code = process.exitCode()
                exit_status = process.exitStatus()
                output_text = "".join(output_lines)

                plugin.log.info(f"Process exit code: {exit_code}, exit status: {exit_status}")
                progress_dialog.append_output(f"\nExit code: {exit_code}, Exit status: {exit_status}\n")

                try:
                    normal_exit = exit_status == QProcess.ExitStatus.NormalExit
                except AttributeError:
                    normal_exit = exit_status == 0

                success = normal_exit and exit_code == 0
                plugin.log.info(f"Process success: {success} (normal_exit={normal_exit}, exit_code={exit_code})")
                return success, output_text
            except Exception as e:
                plugin.log.error(f"Error running command: {e}")
                return False, str(e)

        def find_python_candidates():
            candidates = []
            qgis_macos_python_dir = None

            def _add(path):
                if path and path not in candidates:
                    candidates.append(path)

            def _looks_like_python_launcher(path):
                if path in {"python", "python3", "py"}:
                    return True
                base = os.path.basename(str(path)).lower()
                if base in {"python.exe", "python3.exe", "py.exe", "python", "python3"}:
                    return True
                return base.startswith("python")

            if sys.platform == "darwin":
                with contextlib.suppress(Exception):
                    app_path = QgsApplication.applicationFilePath() or ""
                    if app_path:
                        qgis_macos_python_dir = os.path.dirname(app_path)
                if qgis_macos_python_dir:
                    runtime_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
                    _add(os.path.join(qgis_macos_python_dir, f"python{runtime_mm}"))
                    _add(os.path.join(qgis_macos_python_dir, "python3"))
                    _add(os.path.join(qgis_macos_python_dir, "python"))

            if sys.executable:
                _add(sys.executable)
                _add(os.path.join(os.path.dirname(sys.executable), "python.exe"))
                _add(os.path.join(os.path.dirname(sys.executable), "python3.exe"))
            _add(os.path.join(sys.prefix, "python.exe"))
            _add(os.path.join(sys.base_prefix, "python.exe"))
            _add(os.path.join(os.path.dirname(os.__file__), "..", "python.exe"))

            osgeo_root = os.environ.get("OSGEO4W_ROOT", "")
            if osgeo_root:
                _add(os.path.join(osgeo_root, "bin", "python3.exe"))
                _add(os.path.join(osgeo_root, "bin", "python.exe"))
                for py in glob.glob(os.path.join(osgeo_root, "apps", "Python*", "python.exe")):
                    _add(py)

            for path in sys.path:
                _add(os.path.join(path, "python.exe"))
                _add(os.path.join(path, "python3.exe"))

            for cmd in ("python", "python3", "py"):
                found = shutil.which(cmd)
                if found:
                    _add(found)

            filtered = []
            for c in candidates:
                if not ((os.path.isabs(c) and os.path.isfile(c)) or c in {"python", "python3", "py"}):
                    continue
                if not _looks_like_python_launcher(c):
                    continue
                filtered.append(c)
            return filtered, qgis_macos_python_dir

        def validate_python_candidates(raw_candidates, qgis_macos_python_dir=None):
            validated = []
            expected_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
            for py in raw_candidates:
                probe_cmd = [
                    py,
                    "-c",
                    (
                        "import sys;"
                        "print('DZETSAKA_PY_OK|%d.%d|%s' % (sys.version_info[0], sys.version_info[1], sys.executable))"
                    ),
                ]
                success, output = run_command(probe_cmd, f"python probe via {py}")
                marker_line = ""
                if success:
                    for line in (output or "").splitlines():
                        if line.startswith("DZETSAKA_PY_OK|"):
                            marker_line = line.strip()
                            break
                if not marker_line:
                    plugin.log.warning(f"Skipping non-python launcher candidate: {py}")
                    progress_dialog.append_output(f"⚠ Skipping non-python launcher: {py}\n")
                    continue

                parts = marker_line.split("|", 2)
                candidate_mm = parts[1] if len(parts) >= 2 else ""
                if candidate_mm != expected_mm:
                    plugin.log.warning(
                        f"Skipping python candidate with mismatched version ({candidate_mm} != {expected_mm}): {py}",
                    )
                    progress_dialog.append_output(
                        f"⚠ Skipping python candidate (version {candidate_mm}, expected {expected_mm}): {py}\n",
                    )
                    continue
                validated.append(py)

            if sys.platform == "darwin" and qgis_macos_python_dir:
                qgis_prefix = os.path.abspath(qgis_macos_python_dir) + os.sep
                qgis_validated = [
                    py for py in validated if os.path.isabs(py) and os.path.abspath(py).startswith(qgis_prefix)
                ]
                if qgis_validated:
                    return qgis_validated
            return validated

        def ensure_pip_available(py):
            """Bootstrap pip via get-pip.py if missing for *py*."""
            probe_cmd = [py, "-m", "pip", "--version"]
            success, _ = run_command(probe_cmd, f"pip probe via {py}")
            if success:
                return
            plugin.log.info(f"pip not found for {py}, bootstrapping via get-pip.py")
            progress_dialog.append_output(f"\n⚠ pip not found for {py}, downloading get-pip.py...\n")
            try:
                import tempfile
                import urllib.request

                fd, get_pip_path = tempfile.mkstemp(suffix=".py", prefix="dzetsaka_get_pip_")
                os.close(fd)
                try:
                    urllib.request.urlretrieve(  # nosec B310
                        "https://bootstrap.pypa.io/get-pip.py",
                        get_pip_path,
                    )
                    bootstrap_cmd = [py, get_pip_path, "--user", "--break-system-packages"]
                    ok, _ = run_command(bootstrap_cmd, f"get-pip.py via {py}")
                    if ok:
                        plugin.log.info(f"Successfully bootstrapped pip for {py}")
                        progress_dialog.append_output(f"✓ pip bootstrapped for {py}\n")
                    else:
                        plugin.log.warning(f"get-pip.py failed for {py}")
                        progress_dialog.append_output(f"✗ get-pip.py failed for {py}\n")
                finally:
                    with contextlib.suppress(Exception):
                        os.remove(get_pip_path)
            except Exception as exc:
                plugin.log.warning(f"pip bootstrap error: {exc!s}")

        raw_candidates, qgis_macos_python_dir = find_python_candidates()
        python_candidates = validate_python_candidates(raw_candidates, qgis_macos_python_dir)
        plugin.log.info(f"Installing bundle {targets}. Python candidates: {python_candidates!r}")
        progress_dialog.append_output(
            "Python candidates: " + (", ".join(python_candidates) if python_candidates else "<none found>"),
        )

        # Ensure pip is available (bootstraps via get-pip.py if missing)
        for py in python_candidates:
            ensure_pip_available(py)

        if progress_dialog.was_cancelled():
            return False

        pip_args = [
            "-m",
            "pip",
            "install",
            *targets,
            "--user",
            "--no-input",
            "--disable-pip-version-check",
            "--prefer-binary",
            "--break-system-packages",
        ]
        if runtime_constraint_args:
            pip_args.extend(runtime_constraint_args)
        if extra_args:
            pip_args.extend(extra_args)

        attempts = []
        for py in python_candidates:
            attempts.append(([py, *pip_args], f"pip bundle via {py}"))
            attempts.append(([py, "-m", "ensurepip", "--user"], f"ensurepip via {py}"))
            attempts.append(([py, *pip_args], f"pip bundle after ensurepip via {py}"))

        has_qgis_macos_candidate = (
            sys.platform == "darwin"
            and qgis_macos_python_dir
            and any(
                os.path.isabs(py) and os.path.abspath(py).startswith(os.path.abspath(qgis_macos_python_dir) + os.sep)
                for py in python_candidates
            )
        )
        if not has_qgis_macos_candidate:
            attempts.extend(
                [
                    (["py", "-3", *pip_args], "pip bundle via py -3"),
                    (["py", *pip_args], "pip bundle via py"),
                    (["python3", *pip_args], "pip bundle via python3"),
                    (["python", *pip_args], "pip bundle via python"),
                ],
            )

        last_output = ""
        for cmd, description in attempts:
            if "ensurepip" in cmd and (
                "No module named pip" not in last_output and "pip is not installed" not in last_output.lower()
            ):
                continue
            success, output = run_command(cmd, description)
            last_output = output or ""
            if success and "ensurepip" not in cmd:
                plugin.log.info(f"Successfully installed bundle using: {' '.join(cmd)}")
                progress_dialog.append_output("\n✓ install_package_bundle returning True\n")
                return True

        plugin.log.warning("Bundle installation failed after all launcher attempts")
        progress_dialog.append_output("\n✗ install_package_bundle returning False\n")
        return False

    def _is_importable(module_name):
        import importlib.util

        return importlib.util.find_spec(module_name) is not None

    # Mapping of dependency names to pip packages
    pip_packages = {
        "scikit-learn": "scikit-learn",
        "sklearn": "scikit-learn",
        "Sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "optuna": "optuna",
        "shap": "shap",
        "seaborn": "seaborn",
        "imbalanced-learn": "imbalanced-learn",
        "imblearn": "imbalanced-learn",
        "XGBoost": "xgboost",
        "CatBoost": "catboost",
        "Optuna": "optuna",
    }

    progress_labels = {
        "scikit-learn": "scikit-learn (core algorithms: RF, SVM, KNN, ET, GBC, LR, NB, MLP)",
        "xgboost": "XGBoost algorithm",
        "catboost": "CatBoost algorithm",
        "optuna": "Optuna (hyperparameter optimization)",
        "shap": "SHAP (explainability)",
        "seaborn": "seaborn (report heatmaps)",
        "imbalanced-learn": "imbalanced-learn (SMOTE / class imbalance tools)",
    }

    base_imports = {
        "scikit-learn": "sklearn",
        "sklearn": "sklearn",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "optuna": "optuna",
        "shap": "shap",
        "seaborn": "seaborn",
        "imbalanced-learn": "imblearn",
    }

    def _is_dependency_usable(package_name):
        """Dependency-specific usability check (not just importability)."""
        if package_name == "scikit-learn":
            ok, details = plugin._check_sklearn_usable()
            if not ok:
                plugin.log.warning(f"scikit-learn present but unusable: {details}")
            return ok
        base_import = base_imports.get(package_name, package_name)
        return _is_importable(base_import)

    try:
        # Show custom progress dialog with live output
        progress = InstallProgressDialog(parent=plugin, total_packages=len(missing_deps))
        progress.show()

        progress.append_output(
            "Installing full dependency bundle. First try: one pip command for all missing packages.\n",
        )

        success_count = 0
        package_order = []
        for dep in missing_deps:
            dep_key = dep.strip()
            package_name = pip_packages.get(dep_key, pip_packages.get(dep_key.lower(), dep_key.lower()))
            package_name = pip_packages.get(package_name, package_name)
            if package_name not in package_order:
                package_order.append(package_name)

        unresolved_packages = [pkg for pkg in package_order if not _is_dependency_usable(pkg)]
        bundle_progress_captured = False
        if unresolved_packages:
            progress.status_label.setText("Phase 1/3: Resolve and prepare installer...")
            progress.append_output("Bundle install attempt for: " + ", ".join(unresolved_packages) + "\n")
            progress.status_label.setText("Phase 2/3: Installing dependency bundle...")
            bundle_ok = install_package_bundle(unresolved_packages, progress)
            if not bundle_ok:
                progress.append_output("Bundle install failed. Falling back to per-package installation.\n")
            else:
                bundle_progress_captured = True
                progress.append_output("Bundle install completed. Verifying each dependency...\n")
                progress.status_label.setText("Phase 3/3: Verifying installed dependencies...")

        for i, dep in enumerate(missing_deps):
            if progress.was_cancelled():
                progress.mark_complete(success=False)
                QMessageBox.warning(
                    plugin,
                    "Installation Cancelled",
                    "Dependency installation was cancelled.",
                    QMessageBox.StandardButton.Ok,
                )
                progress.close()
                return False

            # Get the pip package name
            dep_key = dep.strip()
            package_name = pip_packages.get(dep_key, pip_packages.get(dep_key.lower(), dep_key.lower()))
            package_name = pip_packages.get(package_name, package_name)

            display_name = progress_labels.get(package_name, package_name)
            if not bundle_progress_captured:
                progress.set_current_package(display_name, i)
            else:
                progress.status_label.setText(f"Phase 3/3: Verifying {display_name}... ({i + 1}/{len(missing_deps)})")

            targets = [package_name]
            # Deduplicate while preserving order
            seen = set()
            targets = [t for t in targets if not (t in seen or seen.add(t))]

            try:
                dep_installed = False
                if _is_dependency_usable(package_name):
                    dep_installed = True
                    success_count += 1
                    plugin.log.info(f"{package_name} already available; skipping install.")
                    progress.append_output(f"✓ {package_name} already installed\n")

                for target in targets:
                    if progress.was_cancelled():
                        break

                    if dep_installed and target == package_name:
                        continue
                    # Try direct pip installation first (preferred method)
                    plugin.log.info(f"Attempting to install {target} using direct pip...")

                    install_result = install_package(target, progress)
                    plugin.log.info(f"install_package({target}) returned: {install_result}")
                    if install_result:
                        plugin.log.info(f"Successfully installed {target}")
                        progress.append_output(f"✓ {target} installed successfully\n")
                        # Trust pip exit code for success. Freshly --user-installed
                        # packages may not be importable in the running process
                        # (Python needs a restart to pick up new ~/.local/ paths).
                        # Best-effort import check for version logging only.
                        import importlib

                        try:
                            importlib.invalidate_caches()
                            import_target = base_imports.get(target, target)
                            imported = importlib.import_module(import_target)
                            if hasattr(imported, "__version__"):
                                progress.append_output(f"  Version: {imported.__version__}\n")
                        except ImportError:
                            progress.append_output(
                                f"  (will be available after QGIS restart)\n",
                            )
                        if target == package_name and not dep_installed:
                            success_count += 1
                            dep_installed = True
                    else:
                        plugin.log.warning(f"Direct pip installation failed for {target}")
                        progress.append_output(f"✗ Failed to install {target}\n")

            except Exception as e:
                import traceback

                plugin.log.error(f"Error installing {package_name}: {e!s}")
                plugin.log.error(f"Traceback: {traceback.format_exc()}")
                progress.append_output(f"✗ Error: {e!s}\n")
                progress.append_output(f"✗ Traceback: {traceback.format_exc()}\n")

            if not bundle_progress_captured:
                progress.mark_package_complete()

        progress.mark_complete(success=(success_count == len(missing_deps)))

        if success_count == len(missing_deps):
            return True
        plugin._show_github_issue_popup(
            error_title="Installation Incomplete",
            error_type="Dependency Installation Error",
            error_message=(
                f"Only {success_count} of {len(missing_deps)} dependencies were installed successfully.\n"
                "Manual fallback examples:\n"
                "  pip install --user scikit-learn\n"
                "  pip install --user xgboost catboost optuna shap imbalanced-learn\n"
            ),
            context=f"Missing dependencies requested: {', '.join(missing_deps)}",
        )
        progress.close()
        return False

    except Exception as e:
        plugin.log.error(f"Error during dependency installation: {e!s}")
        plugin._show_github_issue_popup(
            error_title="Installation Error",
            error_type=type(e).__name__,
            error_message=str(e),
            context=f"Dependency installation flow for: {', '.join(missing_deps)}",
        )
        return False
    finally:
        if runtime_constraints_file:
            with contextlib.suppress(Exception):
                os.remove(runtime_constraints_file)


def try_install_dependencies_async(plugin, missing_deps, on_complete=None):
    """Install dependencies using QgsTask (non-blocking, better QGIS integration).

    This is the recommended approach that follows QGIS best practices by running
    installation in a background task without blocking the UI or using event loops.

    Parameters
    ----------
    plugin : DzetsakaGUI
        Plugin instance
    missing_deps : list
        List of missing dependency names
    on_complete : callable, optional
        Callback function(success: bool) called when installation finishes

    Returns
    -------
    None
        Installation runs asynchronously, result returned via callback

    """
    if not missing_deps:
        if on_complete:
            on_complete(True)
        return

    from dzetsaka.qgis.dependency_install_task import DependencyInstallTask

    # Normalize and deduplicate dependency names
    normalized_missing_deps = []
    seen = set()
    for dep in missing_deps:
        dep_norm = str(dep).strip()
        if not dep_norm:
            continue
        dep_key = dep_norm.lower()
        if dep_key in seen:
            continue
        seen.add(dep_key)
        normalized_missing_deps.append(dep_norm)

    requested_deps = normalized_missing_deps
    packages = FULL_DEPENDENCY_BUNDLE.copy()
    plugin.log.info(
        f"All-dependencies install mode enabled (async). Requested={requested_deps!r}, full bundle target={packages!r}",
    )

    # Build conservative constraints from the live runtime
    runtime_constraints_file, _ = _build_runtime_constraints_file(plugin.log)

    # Mapping of dependency names to pip packages
    pip_packages = {
        "scikit-learn": "scikit-learn",
        "sklearn": "scikit-learn",
        "Sklearn": "scikit-learn",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "optuna": "optuna",
        "shap": "shap",
        "seaborn": "seaborn",
        "imbalanced-learn": "imbalanced-learn",
        "imblearn": "imbalanced-learn",
        "XGBoost": "xgboost",
        "CatBoost": "catboost",
        "Optuna": "optuna",
    }

    # Convert to pip package names
    package_order = []
    for dep in packages:
        dep_key = dep.strip()
        package_name = pip_packages.get(dep_key, pip_packages.get(dep_key.lower(), dep_key.lower()))
        package_name = pip_packages.get(package_name, package_name)
        if package_name not in package_order:
            package_order.append(package_name)

    def on_task_finished(success: bool):
        """Handle task completion in main thread (called from finished())."""
        # Clean up constraints file
        if runtime_constraints_file:
            with contextlib.suppress(Exception):
                os.remove(runtime_constraints_file)

        # Show result to user
        if success:
            QMessageBox.information(
                plugin.iface.mainWindow(),
                "Installation Successful",
                "Dependencies installed successfully!\n\n"
                "<b>Important:</b> Please restart QGIS to load the new libraries.\n\n"
                "After restarting, you can use all dzetsaka features including "
                "XGBoost, CatBoost, Optuna optimization, and SHAP explainability.",
                QMessageBox.StandardButton.Ok,
            )
        elif task.isCanceled():
            QMessageBox.warning(
                plugin.iface.mainWindow(),
                "Installation Cancelled",
                "Dependency installation was cancelled.",
                QMessageBox.StandardButton.Ok,
            )
        else:
            plugin._show_github_issue_popup(
                error_title="Installation Incomplete",
                error_type="Dependency Installation Error",
                error_message=(
                    f"Only {task.success_count} of {len(package_order)} dependencies were installed successfully.\n"
                    "Manual fallback examples:\n"
                    "  pip install --user scikit-learn\n"
                    "  pip install --user xgboost catboost optuna shap imbalanced-learn\n"
                ),
                context=f"Missing dependencies requested: {', '.join(missing_deps)}",
            )

        # Call user callback
        if on_complete:
            on_complete(success)

    # Create and submit task — use on_finished callback instead of
    # taskCompleted/taskTerminated signals.  The callback is invoked from
    # finished() which QGIS guarantees runs on the main thread, avoiding
    # macOS AppKit crashes (see issue #48).
    task = DependencyInstallTask(
        description="dzetsaka: Installing Python dependencies",
        packages=package_order,
        plugin_logger=plugin.log,
        runtime_constraints=runtime_constraints_file,
        on_finished=on_task_finished,
    )

    # Submit to task manager
    QgsApplication.taskManager().addTask(task)
    plugin.log.info(f"Dependency installation task submitted: {len(package_order)} packages")
