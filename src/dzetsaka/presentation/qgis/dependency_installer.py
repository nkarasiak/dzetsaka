"""Dependency installation flow extracted from plugin runtime."""

from __future__ import annotations

import contextlib
import os
import tempfile

from qgis.PyQt.QtWidgets import QMessageBox


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
    FULL_DEPENDENCY_BUNDLE = [
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "optuna",
        "shap",
        "imbalanced-learn",
    ]

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
        "All-dependencies install mode enabled (sequential pip commands). "
        f"Requested={requested_deps!r}, installing full bundle one-by-one={missing_deps!r}"
    )

    from qgis.PyQt.QtCore import QEventLoop, QProcess

    from dzetsaka.ui.install_progress_dialog import InstallProgressDialog

    # Build conservative constraints from the live runtime to avoid breaking
    # core scientific stack packages in embedded QGIS envs.
    runtime_constraints_file = None
    runtime_constraint_args = []
    try:
        try:
            from importlib import metadata as importlib_metadata
        except ImportError:
            import importlib_metadata  # type: ignore

        pinned_packages = []  # type: list[str]
        for pkg in ("numpy", "scipy", "pandas"):
            try:
                version = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                continue
            if version:
                pinned_packages.append(f"{pkg}=={version}")

        if pinned_packages:
            constraints_text = "\n".join(pinned_packages) + "\n"
            fd, runtime_constraints_file = tempfile.mkstemp(
                prefix="dzetsaka_pip_constraints_",
                suffix=".txt",
            )
            os.close(fd)
            with open(runtime_constraints_file, "w", encoding="utf-8") as f:
                f.write(constraints_text)
            runtime_constraint_args = ["-c", runtime_constraints_file]
            plugin.log.info(
                "Using runtime pip constraints to keep scientific stack stable: "
                f"{', '.join(pinned_packages)}"
            )
    except Exception as constraints_err:
        plugin.log.warning(f"Could not prepare runtime pip constraints: {constraints_err!s}")

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
                        f"\n⚠ Command timed out after {int(timeout_ms / 1000)}s, terminating...\n"
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

            def _add(path):
                if path and path not in candidates:
                    candidates.append(path)

            def _looks_like_python_launcher(path):
                if path in {"python", "python3", "py"}:
                    return True
                base = os.path.basename(str(path)).lower()
                return base in {"python.exe", "python3.exe", "py.exe"}

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
            return filtered

        def validate_python_candidates(raw_candidates):
            validated = []
            for py in raw_candidates:
                probe_cmd = [py, "-c", "print('DZETSAKA_PY_OK')"]
                success, output = run_command(probe_cmd, f"python probe via {py}")
                if success and "DZETSAKA_PY_OK" in (output or ""):
                    validated.append(py)
                else:
                    plugin.log.warning(f"Skipping non-python launcher candidate: {py}")
                    progress_dialog.append_output(f"⚠ Skipping non-python launcher: {py}\n")
            return validated

        python_candidates = validate_python_candidates(find_python_candidates())
        plugin.log.info(f"Installing {package}. Python candidates: {python_candidates!r}")
        progress_dialog.append_output(
            "Python candidates: " + (", ".join(python_candidates) if python_candidates else "<none found>")
        )

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
        attempts.extend(
            [
                (["py", "-3", *pip_args], "pip via py -3"),
                (["py", *pip_args], "pip via py"),
                (["python3", *pip_args], "pip via python3"),
                (["python", *pip_args], "pip via python"),
            ]
        )

        last_output = ""
        for cmd, description in attempts:
            # Avoid running ensurepip unless pip is really missing for that attempt.
            if "ensurepip" in cmd:
                if "No module named pip" not in last_output and "pip is not installed" not in last_output.lower():
                    continue
            success, output = run_command(cmd, description)
            last_output = output or ""
            if success and "ensurepip" not in cmd:
                plugin.log.info(f"Successfully installed {package} using: {' '.join(cmd)}")
                progress_dialog.append_output(f"\n✓ install_package returning True for {package}\n")
                return True

        plugin.log.warning(f"Installation failed for {package} after all launcher attempts")
        progress_dialog.append_output(f"\n✗ install_package returning False for {package}\n")

        # Method 3: On Linux, try apt as final fallback
        if sys.platform.startswith("linux"):
            apt_packages = {
                "scikit-learn": "python3-sklearn",
                "xgboost": "python3-xgboost",
                "lightgbm": "python3-lightgbm",
                "catboost": "python3-catboost",
            }
            apt_pkg = apt_packages.get(package.lower())

            if apt_pkg:
                apt_path = "/usr/bin/apt"
                if os.path.exists(apt_path):
                    plugin.log.info(f"Trying system package manager (apt install {apt_pkg})...")
                    progress_dialog.append_output("\n⚠ Trying system package manager...\n")
                    success, output = run_command(
                        ["pkexec", apt_path, "install", "-y", apt_pkg],
                        "apt via pkexec",
                    )
                    if success:
                        plugin.log.info(f"Successfully installed {apt_pkg} via apt")
                        return True
                    plugin.log.warning(f"apt install failed: {output}")

        # All methods failed
        plugin.log.error(
            f"Could not install {package}. Please install manually:\n"
            "  Option 1 - Install pip first:\n"
            "    sudo apt install python3-pip\n"
            "    pip3 install --user scikit-learn\n"
            "  Option 2 - Install via apt directly:\n"
            "    sudo apt install python3-sklearn\n"
            "  Then restart QGIS."
        )
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
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "optuna": "optuna",
        "shap": "shap",
        "imbalanced-learn": "imbalanced-learn",
        "imblearn": "imbalanced-learn",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "CatBoost": "catboost",
        "Optuna": "optuna",
    }

    progress_labels = {
        "scikit-learn": "scikit-learn (core algorithms: RF, SVM, KNN, ET, GBC, LR, NB, MLP)",
        "xgboost": "XGBoost algorithm",
        "lightgbm": "LightGBM algorithm",
        "catboost": "CatBoost algorithm",
        "optuna": "Optuna (hyperparameter optimization)",
        "shap": "SHAP (explainability)",
        "imbalanced-learn": "imbalanced-learn (SMOTE / class imbalance tools)",
    }

    base_imports = {
        "scikit-learn": "sklearn",
        "sklearn": "sklearn",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "optuna": "optuna",
        "shap": "shap",
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
        progress = InstallProgressDialog(parent=self, total_packages=len(missing_deps))
        progress.show()

        progress.append_output(
            "Installing full dependency bundle with sequential commands "
            "(one package per pip invocation).\n"
        )

        success_count = 0

        for i, dep in enumerate(missing_deps):
            if progress.was_cancelled():
                progress.mark_complete(success=False)
                QMessageBox.warning(
                    self,
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
            progress.set_current_package(display_name, i)

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
                        # Try to import to verify installation (after clearing import cache).
                        # Do not mark success unless the module is actually importable.
                        import importlib

                        try:
                            importlib.invalidate_caches()
                            import_target = base_imports.get(target, target)
                            imported = importlib.import_module(import_target)
                            if target == "scikit-learn":
                                sklearn_ok, sklearn_details = plugin._check_sklearn_usable()
                                if not sklearn_ok:
                                    plugin.log.warning(
                                        "scikit-learn install command succeeded but runtime check failed: "
                                        f"{sklearn_details}"
                                    )
                                    progress.append_output(
                                        "✗ scikit-learn is still unusable after install "
                                        f"({sklearn_details})\n"
                                    )
                                    continue
                                progress.append_output(f"  Verified: {sklearn_details}\n")
                            elif hasattr(imported, "__version__"):
                                plugin.log.info(f"Verified {import_target} import: {imported.__version__}")
                                progress.append_output(f"  Version: {imported.__version__}\n")
                            else:
                                plugin.log.info(f"Verified {import_target} import.")
                            plugin.log.info(
                                f"Checking condition: target={target}, package_name={package_name}, "
                                f"dep_installed={dep_installed}"
                            )
                            if target == package_name and not dep_installed:
                                success_count += 1
                                dep_installed = True
                                plugin.log.info(f"SUCCESS! Incremented success_count to {success_count}")
                                progress.append_output(f"✓ success_count = {success_count}\n")
                        except ImportError as import_error:
                            plugin.log.warning(
                                f"Package {target} install command succeeded but import failed: {import_error}"
                            )
                            progress.append_output("✗ Package not importable after install attempt\n")
                    else:
                        plugin.log.warning(f"Direct pip installation failed for {target}")
                        progress.append_output(f"✗ Failed to install {target}\n")

            except Exception as e:
                import traceback

                plugin.log.error(f"Error installing {package_name}: {e!s}")
                plugin.log.error(f"Traceback: {traceback.format_exc()}")
                progress.append_output(f"✗ Error: {e!s}\n")
                progress.append_output(f"✗ Traceback: {traceback.format_exc()}\n")

            progress.mark_package_complete()

        progress.mark_complete(success=(success_count == len(missing_deps)))

        if success_count == len(missing_deps):
            progress.close()
            return True
        else:
            plugin._show_github_issue_popup(
                error_title="Installation Incomplete",
                error_type="Dependency Installation Error",
                error_message=(
                    f"Only {success_count} of {len(missing_deps)} dependencies were installed successfully.\n"
                    "Manual fallback examples:\n"
                    "  pip install --user scikit-learn\n"
                    "  pip install --user xgboost lightgbm catboost optuna shap imbalanced-learn\n"
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

