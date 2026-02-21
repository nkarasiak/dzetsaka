"""Background task for installing Python dependencies."""

from __future__ import annotations

import contextlib
import glob
import os
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Callable

from qgis.core import QgsTask


class DependencyInstallTask(QgsTask):
    """QGIS background task for installing Python packages without blocking UI."""

    def __init__(
        self,
        description: str,
        packages: list[str],
        plugin_logger,
        runtime_constraints: str | None = None,
        on_finished: Callable[[bool], None] | None = None,
    ):
        """Initialize the dependency installation task.

        Parameters
        ----------
        description : str
            Task description shown in QGIS task manager
        packages : List[str]
            List of package names to install
        plugin_logger : QgisLogger
            Logger instance from the plugin
        runtime_constraints : Optional[str]
            Path to pip constraints file (for pinning numpy/scipy/pandas)

        """
        super().__init__(description)
        self.packages = packages
        self.log = plugin_logger
        self.runtime_constraints = runtime_constraints
        self.on_finished_callback = on_finished
        self.output_lines = []
        self.success_count = 0
        self.error_message = ""

    def run(self) -> bool:
        """Execute package installation in background thread."""
        try:
            self.log.info(f"Starting background installation of packages: {', '.join(self.packages)}")
            self.setProgress(1)

            # Build runtime constraints if needed
            runtime_constraint_args = []
            if self.runtime_constraints:
                runtime_constraint_args = ["-c", self.runtime_constraints]
                self.log.info(f"Using runtime constraints: {self.runtime_constraints}")

            # Try to install bundle first
            self.setProgress(10)
            bundle_success = self._install_package_bundle(self.packages, runtime_constraint_args)

            if bundle_success:
                self.log.info("Bundle installation succeeded")
                self.success_count = len(self.packages)
                self.setProgress(100)
                return True

            # Fallback to per-package installation
            self.log.info("Bundle install failed, trying per-package installation")
            total_packages = len(self.packages)

            for i, package in enumerate(self.packages):
                if self.isCanceled():
                    self.log.info("Installation cancelled by user")
                    return False

                progress = 10 + int((i / total_packages) * 80)
                self.setProgress(progress)

                if self._install_single_package(package, runtime_constraint_args):
                    self.success_count += 1
                    self.log.info(f"Successfully installed {package}")
                else:
                    self.log.warning(f"Failed to install {package}")

            self.setProgress(100)
            return self.success_count == len(self.packages)

        except Exception as exc:
            self.log.error(f"Dependency installation error: {exc!s}")
            self.error_message = str(exc)
            return False

    def finished(self, result: bool) -> None:
        """Called when task completes (runs in main thread)."""
        if result:
            self.log.info(f"Successfully installed {self.success_count}/{len(self.packages)} packages")
        elif self.isCanceled():
            self.log.info("Installation was cancelled")
        else:
            self.log.error(f"Installation failed. Installed {self.success_count}/{len(self.packages)} packages")

        if self.on_finished_callback is not None:
            self.on_finished_callback(result)

    def _find_python_executables(self) -> list[str]:
        """Find candidate Python executables for pip installation."""
        candidates = []
        qgis_macos_python_dir = None

        def _add(path):
            if path and path not in candidates:
                candidates.append(path)

        def _looks_like_python_launcher(path):
            base = os.path.basename(str(path)).lower()
            if base in {"python.exe", "python3.exe", "py.exe", "python", "python3"}:
                return True
            return base.startswith("python")

        if sys.platform == "darwin":
            try:
                from qgis.core import QgsApplication

                app_path = QgsApplication.applicationFilePath() or ""
                if app_path:
                    qgis_macos_python_dir = os.path.dirname(app_path)
            except Exception:
                qgis_macos_python_dir = None
            if qgis_macos_python_dir:
                runtime_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
                _add(os.path.join(qgis_macos_python_dir, f"python{runtime_mm}"))
                _add(os.path.join(qgis_macos_python_dir, "python3"))
                _add(os.path.join(qgis_macos_python_dir, "python"))

        # Primary candidate: sys.executable
        if sys.executable:
            _add(sys.executable)
            _add(os.path.join(os.path.dirname(sys.executable), "python.exe"))
            _add(os.path.join(os.path.dirname(sys.executable), "python3.exe"))

        # Python prefix locations
        _add(os.path.join(sys.prefix, "python.exe"))
        _add(os.path.join(sys.base_prefix, "python.exe"))

        # OSGEO4W root (Windows QGIS)
        osgeo_root = os.environ.get("OSGEO4W_ROOT", "")
        if osgeo_root:
            _add(os.path.join(osgeo_root, "bin", "python3.exe"))
            _add(os.path.join(osgeo_root, "bin", "python.exe"))
            for py in glob.glob(os.path.join(osgeo_root, "apps", "Python*", "python.exe")):
                _add(py)

        # Shell launcher fallbacks
        for cmd in ("python3", "python", "py"):
            found = shutil.which(cmd)
            if found:
                _add(found)

        # Validate executables
        validated = []
        expected_mm = f"{sys.version_info.major}.{sys.version_info.minor}"
        for py in candidates:
            if not os.path.isfile(py):
                continue
            if not _looks_like_python_launcher(py):
                continue
            # Quick validation check
            try:
                result = subprocess.run(  # nosec B603
                    [
                        py,
                        "-c",
                        (
                            "import sys;"
                            "print('DZETSAKA_PY_OK|%d.%d|%s' % (sys.version_info[0], sys.version_info[1], "
                            "sys.executable))"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode != 0:
                    continue
                marker_line = ""
                for line in (result.stdout or "").splitlines():
                    if line.startswith("DZETSAKA_PY_OK|"):
                        marker_line = line.strip()
                        break
                if not marker_line:
                    continue
                parts = marker_line.split("|", 2)
                candidate_mm = parts[1] if len(parts) >= 2 else ""
                if candidate_mm != expected_mm:
                    self.log.warning(
                        f"Skipping python candidate with mismatched version ({candidate_mm} != {expected_mm}): {py}",
                    )
                    continue
                validated.append(py)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        if sys.platform == "darwin" and qgis_macos_python_dir:
            qgis_prefix = os.path.abspath(qgis_macos_python_dir) + os.sep
            qgis_validated = [
                py for py in validated if os.path.isabs(py) and os.path.abspath(py).startswith(qgis_prefix)
            ]
            if qgis_validated:
                return qgis_validated
        return validated

    def _ensure_pip(self, py: str) -> bool:
        """Bootstrap pip via get-pip.py if it is missing for *py*."""
        try:
            probe = subprocess.run(  # nosec B603
                [py, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if probe.returncode == 0:
                return True
        except Exception:
            pass

        self.log.info(f"pip not found for {py}, attempting bootstrap via get-pip.py")
        try:
            import urllib.request

            get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
            import tempfile

            fd, get_pip_path = tempfile.mkstemp(suffix=".py", prefix="dzetsaka_get_pip_")
            os.close(fd)
            try:
                urllib.request.urlretrieve(get_pip_url, get_pip_path)  # nosec B310
                result = subprocess.run(  # nosec B603
                    [py, get_pip_path, "--user", "--break-system-packages"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode == 0:
                    self.log.info(f"Successfully bootstrapped pip for {py}")
                    return True
                self.log.warning(f"get-pip.py failed (exit {result.returncode}): {result.stderr[:300]}")
            finally:
                with contextlib.suppress(Exception):
                    os.remove(get_pip_path)
        except Exception as exc:
            self.log.warning(f"pip bootstrap failed: {exc!s}")
        return False

    def _install_package_bundle(self, packages: list[str], constraint_args: list[str]) -> bool:
        """Try to install all packages in a single pip command."""
        python_exes = self._find_python_executables()
        if not python_exes:
            self.log.warning("No Python executables found for installation")
            return False

        # Ensure pip is available (bootstraps via get-pip.py if missing)
        for py in python_exes:
            self._ensure_pip(py)

        pip_args = [
            "-m",
            "pip",
            "install",
            *packages,
            "--user",
            "--no-input",
            "--disable-pip-version-check",
            "--prefer-binary",
            "--break-system-packages",
        ]
        if constraint_args:
            pip_args.extend(constraint_args)

        # Try each Python executable
        for py in python_exes:
            if self.isCanceled():
                return False

            try:
                cmd = [py, *pip_args]
                self.log.info(f"Trying bundle install: {' '.join(cmd)}")

                result = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=900,  # 15 minute timeout
                    check=False,
                )

                if result.returncode == 0:
                    self.log.info("Bundle installation succeeded")
                    return True
                self.log.warning(f"Bundle install failed with exit code {result.returncode}")
                self.log.warning(f"stderr: {result.stderr[:500]}")

            except subprocess.TimeoutExpired:
                self.log.warning(f"Bundle install timed out with {py}")
            except Exception as exc:
                self.log.warning(f"Bundle install failed: {exc!s}")

        return False

    def _install_single_package(self, package: str, constraint_args: list[str]) -> bool:
        """Try to install a single package."""
        python_exes = self._find_python_executables()
        if not python_exes:
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
        if constraint_args:
            pip_args.extend(constraint_args)

        for py in python_exes:
            if self.isCanceled():
                return False

            try:
                cmd = [py, *pip_args]
                result = subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per package
                    check=False,
                )

                if result.returncode == 0:
                    return True

            except subprocess.TimeoutExpired:
                self.log.warning(f"Install of {package} timed out")
            except Exception as exc:
                self.log.warning(f"Install of {package} failed: {exc!s}")

        return False

    def get_output(self) -> str:
        """Get installation output logs."""
        return "\n".join(self.output_lines)
