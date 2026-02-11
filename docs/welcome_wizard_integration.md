# Welcome Wizard Integration Guide

This document explains how to integrate the new `WelcomeWizard` into the dzetsaka plugin.

## Overview

The `WelcomeWizard` (`ui/welcome_wizard.py`) provides a comprehensive first-run experience with three pages:

1. **OverviewPage**: Feature showcase and introduction
2. **DependencyCheckPage**: Dependency status and optional installation
3. **QuickStartPage**: Sample data loading or direct dashboard access

## File Structure

```
ui/
├── welcome_wizard.py          # New wizard implementation
├── install_progress_dialog.py # Used by async dependency installation
└── dashboard_widget.py         # Main dashboard (opened after wizard)
```

## Integration Steps

### 1. Import the Wizard

Add to `src/dzetsaka/qgis/runtime_bootstrap.py` or `src/dzetsaka/qgis/ui_init.py`:

```python
from dzetsaka.ui.welcome_wizard import WelcomeWizard
```

### 2. Check if Wizard Should Be Shown

In `src/dzetsaka/qgis/runtime_bootstrap.py::initialize_runtime_state()`:

```python
def initialize_runtime_state(gui, iface) -> None:
    """Initialize DzetsakaGUI runtime state."""
    gui.iface = iface
    register_qgis_logging()
    gui.log = QgisLogger(tag="Dzetsaka")

    QDialog.__init__(gui)
    gui.settings = QSettings()
    gui.loadConfig()

    # ... existing code ...

    # Check if welcome wizard should be shown
    welcome_completed = gui.settings.value("/dzetsaka/welcomeCompleted", False, bool)
    gui._show_welcome_wizard = not welcome_completed and gui.firstInstallation
    gui._open_dashboard_on_init = welcome_completed  # Only auto-open dashboard if wizard was completed
```

### 3. Show Wizard at Appropriate Time

In `src/dzetsaka/qgis/ui_init.py::init_gui()`:

```python
def init_gui(plugin):
    """Initialize the plugin GUI components."""
    # ... existing code to register provider, add menu items, toolbar icons ...

    # Show welcome wizard on first run, otherwise open dashboard
    if plugin._show_welcome_wizard:
        from dzetsaka.ui.welcome_wizard import WelcomeWizard

        def show_welcome():
            wizard = WelcomeWizard(plugin, parent=plugin.iface.mainWindow())
            wizard.show()
            wizard.raise_()
            wizard.activateWindow()

        # Delay slightly to ensure QGIS is fully loaded
        QTimer.singleShot(800, show_welcome)
        plugin._show_welcome_wizard = False
    elif plugin._open_dashboard_on_init:
        plugin._open_dashboard_on_init = False
        QTimer.singleShot(1200, plugin.open_dashboard)
```

### 4. Add Manual Trigger Option (Optional)

To allow users to manually open the wizard again (e.g., for testing or re-onboarding):

In `src/dzetsaka/qgis/plugin_runtime.py`:

```python
def show_welcome_wizard(self):
    """Show the welcome wizard (can be called manually)."""
    from dzetsaka.ui.welcome_wizard import WelcomeWizard

    wizard = WelcomeWizard(self, parent=self.iface.mainWindow())
    wizard.show()
    wizard.raise_()
    wizard.activateWindow()
```

Then add a menu item in `ui_init.py`:

```python
# Add welcome wizard menu item
action = plugin._add_action(
    icon_path,
    text=plugin.tr("Show Welcome Wizard"),
    callback=plugin.show_welcome_wizard,
    parent=plugin.iface.mainWindow(),
)
plugin.iface.addPluginToMenu(plugin.menu, action)
plugin.actions.append(action)
```

## Key Features

### Dependency Installation

The wizard uses the plugin's async dependency installation:

```python
self.plugin._try_install_dependencies_async(
    FULL_DEPENDENCY_BUNDLE,
    on_installation_complete
)
```

This ensures:
- Non-blocking UI (follows QGIS best practices)
- Progress feedback
- Proper error handling
- Can be cancelled

### Sample Data Loading

The wizard can automatically load sample data from `data/sample/`:

- Looks for `.tif`/`.tiff` raster files
- Looks for `.shp`/`.gpkg`/`.geojson`/`.geoparquet*` vector files
- Adds them to the current QGIS project
- Gracefully handles missing files

### Settings Persistence

The wizard sets a QSettings flag when completed:

```python
self.plugin.settings.setValue("/dzetsaka/welcomeCompleted", True)
```

This prevents showing the wizard on subsequent plugin loads.

To reset (for testing):

```python
from qgis.PyQt.QtCore import QSettings
settings = QSettings()
settings.remove("/dzetsaka/welcomeCompleted")
```

## User Experience Flow

### First-Time User

1. Install dzetsaka plugin
2. Plugin loads → `welcomeCompleted = False` → Wizard opens automatically
3. User sees feature overview → clicks "Next"
4. User sees dependency status → optionally installs → clicks "Next"
5. User chooses "Try Sample Data" or "Use My Data"
6. Wizard closes, `welcomeCompleted = True` is saved
7. Dashboard opens (with sample data loaded if chosen)

### Returning User

1. Plugin loads → `welcomeCompleted = True` → Dashboard opens directly
2. No wizard shown (unless manually triggered)

## Testing Checklist

- [ ] Wizard opens on first plugin load
- [ ] All three pages display correctly
- [ ] Dependency status shows correct installed/not installed state
- [ ] "Install Full Bundle" button triggers async installation
- [ ] Installation progress shown without blocking UI
- [ ] "Try Sample Data" loads sample files (if available)
- [ ] "Use My Data" opens dashboard without loading data
- [ ] `welcomeCompleted` flag prevents re-showing wizard
- [ ] Manual wizard trigger works (if implemented)
- [ ] Wizard can be cancelled at any page
- [ ] Sample images display correctly (if available)

## Customization Options

### Adding Sample Images

Place classification result images in one of these locations:
- `docs/images/classification_example.png`
- `docs/classification_result.png`
- `images/sample.png`

The wizard will automatically detect and display the first found image.

### Modifying Feature Highlights

Edit the `features` list in `OverviewPage._setup_ui()`:

```python
features = [
    ("emoji", "Title", "Description"),
    # Add more features...
]
```

### Changing Dependency Descriptions

Edit the `dependency_info` dict in `DependencyCheckPage._update_dependency_status()`:

```python
dependency_info = {
    "package-name": ("import_name", "User-friendly description"),
    # Add more dependencies...
}
```

## Architecture Notes

### Class Hierarchy

```
WelcomeWizard (QWizard)
├── OverviewPage (QWizardPage)
├── DependencyCheckPage (QWizardPage)
└── QuickStartPage (QWizardPage)
```

### Key Dependencies

- **QWizard**: Modern wizard style with automatic navigation
- **QSettings**: Persistent storage of completion state
- **plugin instance**: Access to logger, settings, dependency installer
- **QTimer**: Delayed execution for smooth UX

### Signal Flow

```
WelcomeWizard.finished
    → _on_wizard_finished()
        → settings.setValue("/dzetsaka/welcomeCompleted", True)

DependencyCheckPage.install_button.clicked
    → _on_install_clicked()
        → plugin._try_install_dependencies_async()
            → on_installation_complete()
                → _update_dependency_status()

QuickStartPage buttons.clicked
    → _on_sample_data_clicked() or _on_user_data_clicked()
        → wizard.accept()
            → QTimer.singleShot(plugin.open_dashboard)
```

## Troubleshooting

### Wizard Doesn't Open

Check:
1. `gui._show_welcome_wizard` is set correctly in `runtime_bootstrap.py`
2. `welcomeCompleted` setting is False (check with QSettings)
3. QTimer delay is sufficient for QGIS to initialize

### Dependencies Show Wrong Status

The wizard checks with `importlib.util.find_spec()`. This may differ from runtime checks if:
- Package is installed but broken
- Import path issues
- Version incompatibilities

Consider using `plugin._check_sklearn_usable()` for more robust checks.

### Sample Data Won't Load

Check:
1. Files exist in `data/sample/` directory
2. File extensions match glob patterns
3. Files are valid GDAL/OGR formats
4. Permissions allow reading

## Future Enhancements

Potential improvements:

1. **Interactive Tutorial**: Add a fourth page with interactive walkthrough
2. **Video Demos**: Embed video tutorials or links
3. **User Preferences**: Allow customizing default algorithm, validation method, etc.
4. **Plugin Tour**: Highlight dashboard features with overlay tooltips
5. **News Feed**: Show latest dzetsaka updates/features
6. **Testimonials**: Include user success stories
7. **Benchmark Data**: Allow downloading standard datasets for benchmarking

## See Also

- `CLAUDE.md` - Overall plugin architecture
- `src/dzetsaka/qgis/dependency_installer.py` - Dependency installation implementation
- `ui/install_progress_dialog.py` - Installation progress UI
- `src/dzetsaka/qgis/runtime_bootstrap.py` - Plugin initialization
