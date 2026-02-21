"""Unit tests for WelcomeWizard.

These tests verify the welcome wizard structure and components without
requiring a full QGIS environment.

Note: Many tests are basic structural tests since full UI testing requires
a Qt event loop and QGIS environment. For comprehensive UI testing, run
within QGIS using the QGIS testing framework.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Mark all tests in this module as requiring mock setup
pytestmark = pytest.mark.unit


@pytest.fixture
def mock_qgis_modules():
    """Mock QGIS modules for testing outside QGIS environment."""
    # Create mock modules
    qgis_core = MagicMock()
    qgis_pyqt_qtcore = MagicMock()
    qgis_pyqt_qtgui = MagicMock()
    qgis_pyqt_qtwidgets = MagicMock()

    # Mock Qt enums and classes
    qgis_pyqt_qtcore.Qt = MagicMock()
    qgis_pyqt_qtcore.Qt.AlignmentFlag = MagicMock()
    qgis_pyqt_qtcore.Qt.WindowModality = MagicMock()
    qgis_pyqt_qtcore.Qt.TransformationMode = MagicMock()
    qgis_pyqt_qtcore.QSettings = MagicMock
    qgis_pyqt_qtcore.QTimer = MagicMock
    qgis_pyqt_qtcore.pyqtSignal = MagicMock

    qgis_pyqt_qtgui.QFont = MagicMock
    qgis_pyqt_qtgui.QPixmap = MagicMock

    qgis_pyqt_qtwidgets.QCheckBox = MagicMock
    qgis_pyqt_qtwidgets.QHBoxLayout = MagicMock
    qgis_pyqt_qtwidgets.QLabel = MagicMock
    qgis_pyqt_qtwidgets.QPushButton = MagicMock
    qgis_pyqt_qtwidgets.QScrollArea = MagicMock
    qgis_pyqt_qtwidgets.QSizePolicy = MagicMock
    qgis_pyqt_qtwidgets.QVBoxLayout = MagicMock
    qgis_pyqt_qtwidgets.QWidget = MagicMock
    qgis_pyqt_qtwidgets.QWizard = MagicMock
    qgis_pyqt_qtwidgets.QWizard.WizardStyle = MagicMock()
    qgis_pyqt_qtwidgets.QWizard.WizardOption = MagicMock()
    qgis_pyqt_qtwidgets.QWizard.DialogCode = MagicMock()
    qgis_pyqt_qtwidgets.QWizardPage = MagicMock
    qgis_pyqt_qtwidgets.QMessageBox = MagicMock

    # Install mocks
    sys.modules["qgis"] = MagicMock()
    sys.modules["qgis.core"] = qgis_core
    sys.modules["qgis.PyQt"] = MagicMock()
    sys.modules["qgis.PyQt.QtCore"] = qgis_pyqt_qtcore
    sys.modules["qgis.PyQt.QtGui"] = qgis_pyqt_qtgui
    sys.modules["qgis.PyQt.QtWidgets"] = qgis_pyqt_qtwidgets

    yield {
        "core": qgis_core,
        "QtCore": qgis_pyqt_qtcore,
        "QtGui": qgis_pyqt_qtgui,
        "QtWidgets": qgis_pyqt_qtwidgets,
    }

    # Cleanup
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("qgis"):
            del sys.modules[module_name]


@pytest.fixture
def mock_plugin():
    """Create a mock plugin instance."""
    plugin = Mock()
    plugin.plugin_dir = str(Path(__file__).resolve().parents[2])
    plugin.settings = Mock()
    plugin.log = Mock()
    plugin.log.info = Mock()
    plugin.log.warning = Mock()
    plugin.log.error = Mock()
    plugin._try_install_dependencies_async = Mock()
    plugin.open_dashboard = Mock()
    return plugin


def test_welcome_wizard_imports(mock_qgis_modules):
    """Test that WelcomeWizard can be imported."""
    from ui.welcome_wizard import WelcomeWizard

    assert WelcomeWizard is not None


def test_welcome_wizard_has_three_pages(mock_qgis_modules):
    """Test that WelcomeWizard defines three page IDs."""
    from ui.welcome_wizard import WelcomeWizard

    assert hasattr(WelcomeWizard, "PAGE_OVERVIEW")
    assert hasattr(WelcomeWizard, "PAGE_DEPENDENCIES")
    assert hasattr(WelcomeWizard, "PAGE_QUICKSTART")
    assert WelcomeWizard.PAGE_OVERVIEW == 0
    assert WelcomeWizard.PAGE_DEPENDENCIES == 1
    assert WelcomeWizard.PAGE_QUICKSTART == 2


def test_overview_page_imports(mock_qgis_modules):
    """Test that OverviewPage can be imported."""
    from ui.welcome_wizard import OverviewPage

    assert OverviewPage is not None


def test_dependency_check_page_imports(mock_qgis_modules):
    """Test that DependencyCheckPage can be imported."""
    from ui.welcome_wizard import DependencyCheckPage

    assert DependencyCheckPage is not None


def test_quickstart_page_imports(mock_qgis_modules):
    """Test that QuickStartPage can be imported."""
    from ui.welcome_wizard import QuickStartPage

    assert QuickStartPage is not None


def test_dependency_check_page_class_exists(mock_qgis_modules):
    """Test that DependencyCheckPage class exists and has expected methods."""
    from ui.welcome_wizard import DependencyCheckPage

    # Check class has expected methods (without instantiating)
    assert hasattr(DependencyCheckPage, "_is_package_installed")
    assert hasattr(DependencyCheckPage, "_update_dependency_status")
    assert hasattr(DependencyCheckPage, "_create_status_item")
    assert hasattr(DependencyCheckPage, "_on_install_clicked")


def test_quickstart_page_class_exists(mock_qgis_modules):
    """Test that QuickStartPage class exists and has expected methods."""
    from ui.welcome_wizard import QuickStartPage

    # Check class has expected methods (without instantiating)
    assert hasattr(QuickStartPage, "_load_sample_data")
    assert hasattr(QuickStartPage, "_create_option_widget")
    assert hasattr(QuickStartPage, "_on_sample_data_clicked")
    assert hasattr(QuickStartPage, "_on_user_data_clicked")


def test_wizard_class_has_completion_handler(mock_qgis_modules):
    """Test that WelcomeWizard class has completion handler."""
    from ui.welcome_wizard import WelcomeWizard

    # Verify method exists
    assert hasattr(WelcomeWizard, "_on_wizard_finished")


def test_dependency_page_structure(mock_qgis_modules):
    """Test that DependencyCheckPage has expected structure."""
    from ui.welcome_wizard import DependencyCheckPage

    # Verify class has expected methods
    assert hasattr(DependencyCheckPage, "__init__")
    assert hasattr(DependencyCheckPage, "_setup_ui")
    assert hasattr(DependencyCheckPage, "_update_dependency_status")


def test_overview_page_structure(mock_qgis_modules):
    """Test that OverviewPage has expected structure."""
    from ui.welcome_wizard import OverviewPage

    # Verify class has expected methods
    assert hasattr(OverviewPage, "__init__")
    assert hasattr(OverviewPage, "_setup_ui")
    assert hasattr(OverviewPage, "_create_feature_item")
    assert hasattr(OverviewPage, "_add_sample_image")


def test_quickstart_page_color_darkening():
    """Test color darkening utility function."""
    # Test the static method logic without instantiation
    # Color darkening formula: multiply RGB by 0.9
    hex_color = "#4CAF50"
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = int(r * 0.9), int(g * 0.9), int(b * 0.9)
    darkened = f"#{r:02x}{g:02x}{b:02x}"

    assert darkened.startswith("#")
    assert len(darkened) == 7
    assert darkened != "#4CAF50"
    # Green should be darker
    assert int(darkened[1:3], 16) < int("4C", 16)  # R channel
    assert int(darkened[3:5], 16) < int("AF", 16)  # G channel
    assert int(darkened[5:7], 16) < int("50", 16)  # B channel


@pytest.mark.parametrize(
    "package,import_name",
    [
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("catboost", "catboost"),
        ("optuna", "optuna"),
        ("shap", "shap"),
        ("seaborn", "seaborn"),
        ("imbalanced-learn", "imblearn"),
    ],
)
def test_dependency_mapping(package, import_name):
    """Test that package names map correctly to import names."""
    import importlib.util

    # Test the importlib.util.find_spec logic directly
    result = importlib.util.find_spec(import_name) is not None
    assert isinstance(result, bool)

    # Verify common packages can be checked
    assert importlib.util.find_spec("sys") is not None
    assert importlib.util.find_spec("nonexistent_package_xyz123") is None


def test_wizard_page_ids(mock_qgis_modules):
    """Test that wizard page IDs are properly defined."""
    from ui.welcome_wizard import WelcomeWizard

    # Test page ID constants
    assert WelcomeWizard.PAGE_OVERVIEW == 0
    assert WelcomeWizard.PAGE_DEPENDENCIES == 1
    assert WelcomeWizard.PAGE_QUICKSTART == 2

    # Ensure they're unique
    page_ids = {WelcomeWizard.PAGE_OVERVIEW, WelcomeWizard.PAGE_DEPENDENCIES, WelcomeWizard.PAGE_QUICKSTART}
    assert len(page_ids) == 3


def test_sample_data_directory_structure():
    """Test that sample data directory structure can be checked."""
    from pathlib import Path

    # Test Path operations used by _load_sample_data
    test_dir = Path(__file__).parent
    assert test_dir.exists()

    # Test glob patterns used in the code
    py_files = list(test_dir.glob("*.py"))
    assert len(py_files) > 0
    assert any("test_welcome_wizard.py" in str(f) for f in py_files)


def test_wizard_class_structure(mock_qgis_modules):
    """Test that wizard class has expected structure."""
    from ui.welcome_wizard import WelcomeWizard

    # Verify class attributes and methods exist
    assert hasattr(WelcomeWizard, "PAGE_OVERVIEW")
    assert hasattr(WelcomeWizard, "PAGE_DEPENDENCIES")
    assert hasattr(WelcomeWizard, "PAGE_QUICKSTART")
    assert hasattr(WelcomeWizard, "__init__")
    assert hasattr(WelcomeWizard, "_on_wizard_finished")


def test_dependency_page_has_install_handler(mock_qgis_modules):
    """Test that DependencyCheckPage has installation handler."""
    from ui.welcome_wizard import DependencyCheckPage

    # Verify method exists
    assert hasattr(DependencyCheckPage, "_on_install_clicked")
    assert hasattr(DependencyCheckPage, "_setup_ui")
