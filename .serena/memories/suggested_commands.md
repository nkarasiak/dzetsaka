# Suggested Shell Commands for dzetsaka Development

## Git Commands
- `git status` - Check current status of repository
- `git add .` - Stage all changes
- `git commit -m "message"` - Commit changes with message
- `git push origin main` - Push changes to remote repository
- `git pull origin main` - Pull latest changes from remote

## Python Testing
- `python -c "import sklearn; print('scikit-learn available')"` - Test scikit-learn import
- `python -c "import scipy; print('scipy available')"` - Test scipy import
- `python -c "import joblib; print('joblib available')"` - Test joblib import

## File Operations (Windows)
- `dir` - List directory contents
- `cd path` - Change directory
- `copy source dest` - Copy files
- `del filename` - Delete file
- `type filename` - Display file contents
- `find /i "text" filename` - Search for text in file

## QGIS Plugin Development
- Copy plugin to QGIS plugins directory: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
- Restart QGIS to reload plugin changes
- Check QGIS Python console for error messages

## Debugging
- Use QGIS Python console to test imports
- Check QGIS message log for detailed error information
- Enable debug logging in QGIS settings for detailed output

## Build/Package
- `python -m py_compile filename.py` - Check Python syntax
- Create zip file for plugin distribution
- Test plugin in fresh QGIS installation