# dzetsaka Code Style and Conventions

## File Organization
- **Main plugin file**: dzetsaka.py (GUI class and main interface)
- **Processing algorithms**: processing/ directory (individual algorithm files)
- **Core scripts**: scripts/ directory (shared functionality)
- **UI files**: ui/ directory (PyQt UI definitions)

## Python Style
- **Naming**: Mixed convention (some camelCase, some snake_case)
- **Classes**: CamelCase (e.g., `dzetsakaGUI`, `progressBar`)
- **Functions**: camelCase (e.g., `initPredict`, `runMagic`)
- **Variables**: Mixed case
- **Constants**: UPPER_CASE (e.g., `INPUT_RASTER`, `OUTPUT_MODEL`)

## Documentation
- **Docstrings**: Limited usage, some functions have brief descriptions
- **Comments**: Sparse, mainly for complex logic sections
- **Type hints**: Not consistently used

## Error Handling
- Basic try/except blocks for critical operations
- Error messages displayed through QGIS message bar
- Progress feedback through custom progressBar class

## Code Patterns
- Heavy use of GDAL for raster operations
- Object-oriented design for main classes
- Processing algorithms inherit from QgsProcessingAlgorithm
- Resource management through Qt resource system

## Notable Issues in Current Codebase
- Mixed coding styles throughout
- Limited error handling in some areas
- Float/int type inconsistencies causing Qt errors
- Some unused imports and variables