"""Background task for installing Python dependencies."""

from __future__ import annotations

import contextlib
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple

from qgis.core import QgsTask


class DependencyInstallTask(QgsTask):
    """QGIS background task for installing Python packages without blocking UI."""

    def __init__(
        self,
        description: str,
        packages: List[str],
        plugin_logger,
        runtime_constraints: Optional[str] = None,
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

    def _find_python_executables(self) -> List[str]:
        """Find candidate Python executables for pip installation."""
        candidates = []

        def _add(path):
            if path and path not in candidates:
                candidates.append(path)

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
        for py in candidates:
            if not os.path.isfile(py):
                continue
            # Quick validation check
            try:
                result = subprocess.run(
                    [py, "-c", "print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and "OK" in result.stdout:
                    validated.append(py)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        return validated

    def _install_package_bundle(self, packages: List[str], constraint_args: List[str]) -> bool:
        """Try to install all packages in a single pip command."""
        python_exes = self._find_python_executables()
        if not python_exes:
            self.log.warning("No Python executables found for installation")
            return False

        pip_args = [
            "-m",
            "pip",
            "install",
            *packages,
            "--user",
            "--no-input",
            "--disable-pip-version-check",
            "--prefer-binary",
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

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=900,  # 15 minute timeout
                    check=False,
                )

                if result.returncode == 0:
                    self.log.info("Bundle installation succeeded")
                    return True
                else:
                    self.log.warning(f"Bundle install failed with exit code {result.returncode}")
                    self.log.warning(f"stderr: {result.stderr[:500]}")

            except subprocess.TimeoutExpired:
                self.log.warning(f"Bundle install timed out with {py}")
            except Exception as exc:
                self.log.warning(f"Bundle install failed: {exc!s}")

        return False

    def _install_single_package(self, package: str, constraint_args: List[str]) -> bool:
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
        ]
        if constraint_args:
            pip_args.extend(constraint_args)

        for py in python_exes:
            if self.isCanceled():
                return False

            try:
                cmd = [py, *pip_args]
                result = subprocess.run(
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
