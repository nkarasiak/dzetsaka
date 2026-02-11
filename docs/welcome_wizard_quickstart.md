# Welcome Wizard Quick Start

## 5-Minute Integration

### 1. Add to Runtime Bootstrap

File: `src/dzetsaka/qgis/runtime_bootstrap.py`

```python
def initialize_runtime_state(gui, iface) -> None:
    """Initialize DzetsakaGUI runtime state."""
    # ... existing code ...

    # Add these lines before gui._open_dashboard_on_init
    welcome_completed = gui.settings.value("/dzetsaka/welcomeCompleted", False, bool)
    gui._show_welcome_wizard = not welcome_completed and gui.firstInstallation
    gui._open_dashboard_on_init = welcome_completed  # Only if wizard done
```

### 2. Add to UI Initialization

File: `src/dzetsaka/qgis/ui_init.py`

```python
def init_gui(plugin):
    """Initialize the plugin GUI components."""
    # ... existing menu/toolbar setup ...

    # Replace existing auto-open logic with:
    if plugin._show_welcome_wizard:
        from dzetsaka.ui.welcome_wizard import WelcomeWizard

        def show_welcome():
            wizard = WelcomeWizard(plugin, parent=plugin.iface.mainWindow())
            wizard.show()
            wizard.raise_()
            wizard.activateWindow()

        QTimer.singleShot(800, show_welcome)
        plugin._show_welcome_wizard = False
    elif plugin._open_dashboard_on_init:
        plugin._open_dashboard_on_init = False
        QTimer.singleShot(1200, plugin.open_dashboard)
```

### 3. Test

Reset and test:

```python
from qgis.PyQt.QtCore import QSettings
QSettings().remove("/dzetsaka/welcomeCompleted")
# Reload plugin
```

## Optional: Manual Trigger

Add to `src/dzetsaka/qgis/plugin_runtime.py`:

```python
def show_welcome_wizard(self):
    """Show the welcome wizard."""
    from dzetsaka.ui.welcome_wizard import WelcomeWizard
    wizard = WelcomeWizard(self, parent=self.iface.mainWindow())
    wizard.show()
```

Add menu item in `ui_init.py`:

```python
action = plugin._add_action(
    icon_path,
    text=plugin.tr("Show Welcome Wizard"),
    callback=plugin.show_welcome_wizard,
    parent=plugin.iface.mainWindow(),
)
plugin.iface.addPluginToMenu(plugin.menu, action)
```

## Customization

### Change Feature List

Edit `ui/welcome_wizard.py`, `OverviewPage._setup_ui()`:

```python
features = [
    ("🤖", "Your Title", "Your description"),
    # Add more...
]
```

### Change Dependencies

Edit `src/dzetsaka/qgis/dependency_catalog.py`:

```python
FULL_DEPENDENCY_BUNDLE = [
    "scikit-learn",
    "your-package",
    # ...
]
```

Then update descriptions in `ui/welcome_wizard.py`, `DependencyCheckPage._update_dependency_status()`:

```python
dependency_info = {
    "your-package": ("import_name", "Description"),
}
```

### Add Sample Images

Place image at:
- `docs/images/classification_example.png` (preferred)
- `docs/classification_result.png`
- `images/sample.png`

Wizard auto-detects and displays first found.

## Troubleshooting

**Wizard doesn't appear:**
- Check `QSettings().value("/dzetsaka/welcomeCompleted")` is False
- Check `gui.firstInstallation` is True
- Check `gui._show_welcome_wizard` is set in runtime_bootstrap

**Dependencies show wrong status:**
- Wizard uses `importlib.util.find_spec(module_name)`
- For sklearn, uses plugin's `_check_sklearn_usable()` for robust check

**Sample data won't load:**
- Verify files in `data/sample/` directory
- Check file extensions match glob patterns
- Ensure valid GDAL/OGR formats

## File Locations

```
ui/
└── welcome_wizard.py          # Main implementation (648 lines)

src/dzetsaka/qgis/
├── runtime_bootstrap.py       # Add: _show_welcome_wizard flag
├── ui_init.py                 # Add: Wizard trigger logic
├── plugin_runtime.py          # Optional: Manual trigger
└── dependency_catalog.py      # Dependency list

docs/
├── welcome_wizard_integration.md    # Full guide (400+ lines)
└── welcome_wizard_quickstart.md     # This file

tests/unit/
└── test_welcome_wizard.py     # Unit tests (23 tests)
```

## Key Classes

- **WelcomeWizard**: Main wizard (QWizard)
  - PAGE_OVERVIEW = 0
  - PAGE_DEPENDENCIES = 1
  - PAGE_QUICKSTART = 2

- **OverviewPage**: Feature showcase
- **DependencyCheckPage**: Install dependencies
- **QuickStartPage**: Sample data or user data

## Settings

- **Key**: `/dzetsaka/welcomeCompleted`
- **Type**: bool
- **Default**: False
- **Set by**: Wizard on completion

## Dependencies

Wizard checks 8 packages:

1. scikit-learn → sklearn
2. xgboost → xgboost
3. lightgbm → lightgbm
4. catboost → catboost
5. optuna → optuna
6. shap → shap
7. seaborn → seaborn
8. imbalanced-learn → imblearn

## User Flow

```
First Time:
Install Plugin → Wizard Opens → Page 1 → Page 2 → Page 3 → Dashboard

Returning:
Open QGIS → Dashboard Opens Directly

Manual:
Menu → "Show Welcome Wizard" → Wizard Opens
```

## Testing Commands

```bash
# Syntax check
python -c "with open('ui/welcome_wizard.py', 'r', encoding='utf-8') as f: \
    compile(f.read(), 'welcome_wizard.py', 'exec')"

# Unit tests (after fixing guided_workflow_widget.py syntax error)
pytest tests/unit/test_welcome_wizard.py -v

# Reset for testing
python -c "from qgis.PyQt.QtCore import QSettings; \
    QSettings().remove('/dzetsaka/welcomeCompleted')"
```

## Support

- Full documentation: `docs/welcome_wizard_integration.md`
- Implementation: `ui/welcome_wizard.py`
- Tests: `tests/unit/test_welcome_wizard.py`
- Summary: `WELCOME_WIZARD_SUMMARY.md`
