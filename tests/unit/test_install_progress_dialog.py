"""Unit tests for InstallProgressDialog."""

import pytest

# Skip all tests if QGIS is not available
pytest.importorskip("qgis")


class TestInstallProgressDialog:
    """Test the InstallProgressDialog UI component."""

    def test_dialog_creation(self, qtbot):
        """Test that the dialog can be created."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=3)
        qtbot.addWidget(dialog)

        assert dialog.windowTitle() == "Installing Dependencies"
        assert dialog.total_packages == 3
        assert dialog.current_package == 0
        assert not dialog.was_cancelled()

    def test_set_current_package(self, qtbot):
        """Test setting the current package updates UI."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=3)
        qtbot.addWidget(dialog)

        dialog.set_current_package("xgboost", 1)

        assert dialog.current_package == 1
        assert "xgboost" in dialog.status_label.text()
        assert "(2/3)" in dialog.status_label.text()
        assert dialog.progress_bar.value() == 1

    def test_append_output(self, qtbot):
        """Test appending output to the text area."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=1)
        qtbot.addWidget(dialog)

        dialog.append_output("Installing package...\n")
        dialog.append_output("Download complete\n")

        text = dialog.output_text.toPlainText()
        assert "Installing package..." in text
        assert "Download complete" in text

    def test_mark_package_complete(self, qtbot):
        """Test marking a package as complete increments progress."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=3)
        qtbot.addWidget(dialog)

        dialog.set_current_package("numpy", 0)
        dialog.mark_package_complete()

        assert dialog.progress_bar.value() == 1

    def test_mark_complete_success(self, qtbot):
        """Test marking installation as complete (success)."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=2)
        qtbot.addWidget(dialog)

        dialog.mark_complete(success=True)

        assert "complete" in dialog.status_label.text().lower()
        assert dialog.cancel_button.text() == "Close"
        assert dialog.progress_bar.value() == 2

    def test_mark_complete_failure(self, qtbot):
        """Test marking installation as complete (failure)."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=2)
        qtbot.addWidget(dialog)

        dialog.mark_complete(success=False)

        assert "failed" in dialog.status_label.text().lower() or "cancelled" in dialog.status_label.text().lower()
        assert dialog.cancel_button.text() == "Close"

    def test_cancel_button_click(self, qtbot):
        """Test clicking the cancel button sets the cancelled flag."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=1)
        qtbot.addWidget(dialog)

        assert not dialog.was_cancelled()

        qtbot.mouseClick(dialog.cancel_button, Qt.LeftButton)

        assert dialog.was_cancelled()
        assert "cancelling" in dialog.status_label.text().lower()
        assert not dialog.cancel_button.isEnabled()

    def test_close_button_after_complete(self, qtbot):
        """Test that Close button closes the dialog."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=1)
        qtbot.addWidget(dialog)

        dialog.mark_complete(success=True)
        assert dialog.cancel_button.text() == "Close"

        # Clicking Close should call accept()
        with qtbot.waitSignal(dialog.accepted, timeout=1000):
            qtbot.mouseClick(dialog.cancel_button, Qt.LeftButton)

    def test_details_toggle_resizes_dialog(self, qtbot):
        """Dialog should grow when details are shown, then shrink when hidden."""
        from ui.install_progress_dialog import InstallProgressDialog

        dialog = InstallProgressDialog(parent=None, total_packages=1)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)

        collapsed_height = dialog.height()
        qtbot.mouseClick(dialog.details_button, Qt.LeftButton)
        qtbot.wait(50)
        expanded_height = dialog.height()

        qtbot.mouseClick(dialog.details_button, Qt.LeftButton)
        qtbot.wait(50)
        collapsed_again_height = dialog.height()

        assert expanded_height > collapsed_height
        assert collapsed_again_height < expanded_height


# This test requires QGIS, so it will be skipped if QGIS is not available
try:
    import pytestqt
    from qgis.PyQt.QtCore import Qt

    # Import for fixture usage
    pytest_plugins = ["pytestqt"]
except ImportError:
    # Skip all tests if PyQt or pytest-qt is not available
    pytestmark = pytest.mark.skip("PyQt/pytest-qt not available")
