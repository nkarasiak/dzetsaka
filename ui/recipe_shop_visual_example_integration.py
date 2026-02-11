"""Example integration code for visual recipe shop.

This file shows exactly how to add the visual recipe shop to the existing
guided_workflow_widget.py without breaking existing functionality.

Copy the relevant sections into guided_workflow_widget.py.
"""

# ==============================================================================
# STEP 1: Add import at the top of guided_workflow_widget.py
# ==============================================================================

# Add this with the other UI imports (around line 25-50)
try:
    from ui.recipe_shop_visual import show_visual_recipe_shop, VisualRecipeShopDialog
    _VISUAL_RECIPE_SHOP_AVAILABLE = True
except ImportError:
    _VISUAL_RECIPE_SHOP_AVAILABLE = False


# ==============================================================================
# STEP 2: Add helper method to QuickClassificationPanel class
# ==============================================================================

# Add this method to the QuickClassificationPanel class (around line 5000-5500)
def _open_visual_recipe_shop(self):
    """Open the visual recipe shop dialog.

    Modern card-based UI for browsing and selecting recipes.
    Falls back to old dialog if visual shop is not available.
    """
    if not _VISUAL_RECIPE_SHOP_AVAILABLE:
        QMessageBox.warning(
            self,
            "Visual Recipe Shop",
            "Visual recipe shop is not available. Using standard recipe selection.",
        )
        return

    # Load recipes (user + builtin)
    settings = QSettings()
    recipes = load_recipes(settings)

    if not recipes:
        QMessageBox.information(
            self,
            "No Recipes",
            "No recipes found. Please create a recipe first.",
        )
        return

    # Check dependency availability
    available_deps = check_dependency_availability()

    # Show visual recipe shop
    selected_recipe = show_visual_recipe_shop(recipes, available_deps, self)

    if selected_recipe:
        # Apply the selected recipe
        self._apply_selected_recipe(selected_recipe)

        # Update current recipe label if it exists
        if hasattr(self, "_current_recipe_label"):
            recipe_name = selected_recipe.get("name", "Custom")
            self._current_recipe_label.setText(f"📋 {recipe_name}")

        # Show confirmation
        recipe_name = selected_recipe.get("name", "Unknown Recipe")
        QMessageBox.information(
            self,
            "Recipe Applied",
            f"Successfully applied recipe: {recipe_name}\n\n"
            "All parameters have been updated to match the recipe configuration.",
        )


# ==============================================================================
# STEP 3: Add button to UI (Option A - Replace existing button)
# ==============================================================================

# In QuickClassificationPanel.__init__() method, find the recipe-related UI
# (around line 5100-5300) and replace or add:

# Original code might look like:
#     recipe_combo = QComboBox()
#     recipe_combo.addItem("Select recipe...")
#     layout.addWidget(recipe_combo)

# Replace with:
def _create_recipe_ui_section(self):
    """Create recipe selection UI section with visual shop button."""
    recipe_group = QGroupBox("Recipe / Preset Configuration")
    recipe_layout = QVBoxLayout(recipe_group)

    # Info label
    info_label = QLabel(
        "Recipes are pre-configured workflows for common classification tasks. "
        "Browse the recipe shop to find workflows optimized for your use case."
    )
    info_label.setWordWrap(True)
    info_label.setStyleSheet("color: #666666; font-size: 9pt;")
    recipe_layout.addWidget(info_label)

    # Current recipe indicator
    current_recipe_layout = QHBoxLayout()
    current_recipe_layout.addWidget(QLabel("Current:"))

    self._current_recipe_label = QLabel("None (Custom configuration)")
    self._current_recipe_label.setStyleSheet("color: #0066cc; font-weight: 500;")
    current_recipe_layout.addWidget(self._current_recipe_label, 1)
    recipe_layout.addLayout(current_recipe_layout)

    # Buttons row
    buttons_layout = QHBoxLayout()

    # Visual recipe shop button (primary action)
    if _VISUAL_RECIPE_SHOP_AVAILABLE:
        visual_shop_btn = QPushButton("🏪 Browse Recipe Shop")
        visual_shop_btn.setMinimumHeight(40)
        visual_shop_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
            QPushButton:pressed {
                background-color: #1565c0;
            }
        """)
        visual_shop_btn.clicked.connect(self._open_visual_recipe_shop)
        buttons_layout.addWidget(visual_shop_btn, 2)

    # Save current config as recipe button
    save_recipe_btn = QPushButton("💾 Save Current as Recipe")
    save_recipe_btn.setMinimumHeight(40)
    save_recipe_btn.clicked.connect(self._save_current_as_recipe)
    buttons_layout.addWidget(save_recipe_btn, 1)

    # Clear recipe button
    clear_btn = QPushButton("✖ Clear")
    clear_btn.setMinimumHeight(40)
    clear_btn.clicked.connect(self._clear_recipe)
    buttons_layout.addWidget(clear_btn, 0)

    recipe_layout.addLayout(buttons_layout)

    return recipe_group


# ==============================================================================
# STEP 4: Add supporting methods
# ==============================================================================

def _clear_recipe(self):
    """Clear current recipe and reset to defaults."""
    reply = QMessageBox.question(
        self,
        "Clear Recipe",
        "This will reset all parameters to default values. Continue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )

    if reply == QMessageBox.StandardButton.Yes:
        self._current_recipe_label.setText("None (Custom configuration)")
        # Reset UI to defaults
        # (Add your default reset logic here)


def _save_current_as_recipe(self):
    """Save current configuration as a new recipe."""
    from qgis.PyQt.QtWidgets import QInputDialog

    # Ask for recipe name
    recipe_name, ok = QInputDialog.getText(
        self,
        "Save Recipe",
        "Enter a name for this recipe:",
        QLineEdit.EchoMode.Normal,
        "My Custom Recipe",
    )

    if not ok or not recipe_name.strip():
        return

    # Collect current configuration
    current_config = self._collect_current_config()
    current_config["name"] = recipe_name.strip()

    # Mark as user recipe (not template)
    if "metadata" not in current_config:
        current_config["metadata"] = {}
    current_config["metadata"]["is_template"] = False
    current_config["metadata"]["category"] = "custom"

    # Upgrade to v2 schema
    current_config = _upgrade_recipe_to_v2(current_config)

    # Save to settings
    settings = QSettings()
    recipes = load_recipes(settings)
    recipes.append(current_config)

    # Save back to settings
    recipes_json = json.dumps({"recipes": recipes}, indent=2)
    settings.setValue("/dzetsaka/recipes", recipes_json)

    # Update UI
    self._current_recipe_label.setText(f"📋 {recipe_name}")

    # Show confirmation
    QMessageBox.information(
        self,
        "Recipe Saved",
        f"Recipe '{recipe_name}' has been saved to 'My Recipes'.\n\n"
        "You can find it in the Recipe Shop under the 'My Recipes' tab.",
    )

    # Emit update signal if available
    if hasattr(self, "recipesUpdated"):
        self.recipesUpdated.emit()


def _collect_current_config(self):
    """Collect current UI configuration as a recipe dict.

    Returns:
        Recipe dictionary with current settings
    """
    # This is a template - adapt to your actual UI structure
    config = {
        "version": 1,
        "schema_version": 2,
        "name": "Untitled Recipe",
        "description": "Custom configuration",
        "classifier": {
            "code": self.algorithm_combo.currentData() if hasattr(self, "algorithm_combo") else "GMM",
            "name": self.algorithm_combo.currentText() if hasattr(self, "algorithm_combo") else "GMM",
        },
        "validation": {
            "split_percent": self.split_spinbox.value() if hasattr(self, "split_spinbox") else 70,
            "cv_mode": "RANDOM_SPLIT",
        },
        "extraParam": {
            "USE_OPTUNA": self.optuna_checkbox.isChecked() if hasattr(self, "optuna_checkbox") else False,
            "COMPUTE_SHAP": self.shap_checkbox.isChecked() if hasattr(self, "shap_checkbox") else False,
            "USE_SMOTE": self.smote_checkbox.isChecked() if hasattr(self, "smote_checkbox") else False,
            "USE_CLASS_WEIGHTS": self.weights_checkbox.isChecked() if hasattr(self, "weights_checkbox") else False,
            "GENERATE_REPORT_BUNDLE": True,
        },
        "postprocess": {
            "save_model": True,
            "confusion_matrix": True,
            "confidence_map": False,
        },
    }

    return config


# ==============================================================================
# STEP 5: Update __init__ to use new UI
# ==============================================================================

# In QuickClassificationPanel.__init__() or similar, add the recipe section:

def __init__(self, parent=None):
    super().__init__(parent)

    # ... existing initialization code ...

    # Add recipe UI section
    recipe_section = self._create_recipe_ui_section()
    self.main_layout.addWidget(recipe_section)

    # ... rest of initialization ...


# ==============================================================================
# STEP 6: Optional - Add keyboard shortcut
# ==============================================================================

# In ClassificationDashboardDock.__init__() or plugin init:
def _setup_shortcuts(self):
    """Setup keyboard shortcuts."""
    from qgis.PyQt.QtGui import QKeySequence
    from qgis.PyQt.QtWidgets import QShortcut

    # Ctrl+R to open recipe shop
    recipe_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
    recipe_shortcut.activated.connect(self._open_visual_recipe_shop)

    # Ctrl+S to save current as recipe
    save_shortcut = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
    save_shortcut.activated.connect(self._save_current_as_recipe)


# ==============================================================================
# STEP 7: Optional - Add toolbar button
# ==============================================================================

# In plugin_runtime.py or main plugin class:
def _add_recipe_shop_action(self):
    """Add recipe shop to toolbar and menu."""
    from qgis.PyQt.QtGui import QIcon

    icon_path = os.path.join(os.path.dirname(__file__), "icons", "recipe_shop.png")
    if not os.path.exists(icon_path):
        icon_path = None

    action = QAction(
        QIcon(icon_path) if icon_path else QIcon(),
        "Recipe Shop",
        self.iface.mainWindow(),
    )
    action.setToolTip("Open dzetsaka Recipe Shop")
    action.triggered.connect(self._open_recipe_shop_from_toolbar)

    # Add to toolbar
    self.toolbar.addAction(action)

    # Add to menu
    self.iface.addPluginToMenu("dzetsaka", action)

    return action


def _open_recipe_shop_from_toolbar(self):
    """Open recipe shop from toolbar/menu action."""
    # Get the dashboard widget
    if hasattr(self, "dashboard_dock"):
        panel = self.dashboard_dock.quick_panel
        if hasattr(panel, "_open_visual_recipe_shop"):
            panel._open_visual_recipe_shop()
            return

    # Fallback: Show standalone
    if _VISUAL_RECIPE_SHOP_AVAILABLE:
        settings = QSettings()
        recipes = load_recipes(settings)
        available_deps = check_dependency_availability()

        selected_recipe = show_visual_recipe_shop(
            recipes,
            available_deps,
            self.iface.mainWindow(),
        )

        if selected_recipe:
            QMessageBox.information(
                self.iface.mainWindow(),
                "Recipe Selected",
                f"Selected: {selected_recipe.get('name')}\n\n"
                "Open the dzetsaka dashboard to apply this recipe.",
            )


# ==============================================================================
# USAGE SUMMARY
# ==============================================================================

"""
To integrate the visual recipe shop:

1. Copy the import statement to the top of guided_workflow_widget.py
2. Add the _open_visual_recipe_shop method to QuickClassificationPanel
3. Replace or supplement existing recipe UI with the new button
4. Add the supporting methods (_clear_recipe, _save_current_as_recipe, etc.)
5. Test in QGIS

The visual shop will automatically:
- Load all recipes (builtin + user)
- Check dependencies and show warnings
- Show recipes as beautiful cards
- Apply selected recipe to UI
- Save/load user recipes

Backward compatibility:
- Old combo box can coexist with new button
- Falls back gracefully if visual shop import fails
- Works with existing recipe data structures
"""
