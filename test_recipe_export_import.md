# Recipe Import/Export Feature Test

## Feature Overview

Added recipe import/export functionality to the `QuickClassificationPanel` in `ui/guided_workflow_widget.py`.

## Changes Made

### 1. UI Components (Lines 4112-4138)
- **Export Button**: QToolButton labeled "Export..." next to recipe combo
- **Import Button**: QToolButton labeled "Import..." next to recipe combo
- Both buttons have tooltips explaining their purpose
- Icons from QGIS resource system (save.svg and open.svg)

### 2. Export Method `_export_recipe()` (Lines 4565-4611)
**Functionality:**
- Gets current recipe from combo dropdown or builds from current UI state
- Opens QFileDialog for saving (.dzrecipe or .json extension)
- Writes recipe to JSON file with pretty formatting (indent=2)
- Shows success message with file path

**Workflow:**
1. User clicks "Export..." button
2. If a recipe is selected, uses that recipe
3. If no recipe or "Add custom" is selected, builds recipe from current UI state
4. Prompts user to select save location
5. Saves recipe as JSON file
6. Shows confirmation message

### 3. Import Method `_import_recipe()` (Lines 4612-4756)
**Functionality:**
- Opens QFileDialog to select .dzrecipe, .json, or any file
- Reads and parses JSON content
- Handles multiple JSON formats (single recipe, array, wrapped "recipes" object)
- Validates recipe schema (checks for required fields)
- Normalizes recipe (adds missing fields, upgrades to v2 if needed)
- Checks dependencies using `validate_recipe_dependencies()`
- Prompts to install missing dependencies if needed
- Checks for name conflicts and allows overwriting or renaming
- Adds recipe to QSettings storage
- Refreshes recipe combo to show imported recipe

**Workflow:**
1. User clicks "Import..." button
2. Selects recipe file
3. System validates recipe format and schema
4. If dependencies missing, prompts to install (with async installer support)
5. If recipe name exists, asks to overwrite or rename
6. Adds to recipe list and saves to QSettings
7. Shows success message

## Schema Validation

The import validates:
- Recipe must have "name" field (defaults to "Imported Recipe")
- Recipe must have "classifier" field with valid structure
- Recipe must have "extraParam" field (defaults to {})
- Uses `normalize_recipe()` to add missing fields and upgrade to v2 schema
- Uses `validate_recipe_dependencies()` to check for missing packages

## Dependency Installation

If a recipe requires missing dependencies:
- Shows dialog listing missing packages
- Offers to install full dependency bundle
- Uses `_try_install_dependencies()` from installer if available
- Falls back to manual instruction if installer not available
- Prompts to restart QGIS after installation

## File Formats Supported

### Export
- `.dzrecipe` (recommended, custom extension)
- `.json` (standard JSON)

### Import
- `.dzrecipe` files
- `.json` files
- Any file with JSON content
- Supports wrapped formats:
  - Single recipe object: `{...recipe...}`
  - Recipe array: `[{...recipe1...}, {...recipe2...}]`
  - Wrapped object: `{"recipes": [{...recipe1...}]}`

## Testing

### Manual Test Steps

**Export Test:**
1. Open QGIS with dzetsaka plugin
2. Open Classification Dashboard
3. Select a recipe from dropdown
4. Click "Export..." button
5. Choose save location and filename
6. Verify file is created and contains valid JSON

**Import Test:**
1. Open QGIS with dzetsaka plugin
2. Open Classification Dashboard
3. Click "Import..." button
4. Select a .dzrecipe or .json file
5. If dependencies missing, accept installation prompt
6. Verify recipe appears in dropdown
7. Verify recipe applies correctly

**Edge Cases:**
- Import recipe with same name (test overwrite/rename)
- Import recipe with missing dependencies
- Import invalid JSON file
- Import recipe without required fields
- Export with no recipe selected (should use current UI state)

## Code Quality

- Passes `ruff check` linting (All checks passed!)
- Passes `ast.parse` syntax validation
- Follows project conventions:
  - Type hints in comments
  - Google-style docstrings
  - 120 character line length
  - Proper error handling

## Integration Points

- Uses existing `normalize_recipe()` function for schema normalization
- Uses existing `validate_recipe_dependencies()` for dependency checking
- Uses existing `save_recipes()` and `load_recipes()` for persistence
- Uses existing `_build_recipe_seed_from_quick_state()` for current state export
- Uses existing `_refresh_recipe_combo()` to update UI after import
- Integrates with existing dependency installer (`_try_install_dependencies`)

## File Location

`C:\Users\nicar\git\dzetsaka\ui\guided_workflow_widget.py`

**Lines:**
- Export/Import buttons: 4112-4138
- _export_recipe method: 4565-4611
- _import_recipe method: 4612-4756
- _emit_config method: 4756+ (unchanged, methods inserted before it)
