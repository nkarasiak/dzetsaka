# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2025-07-12

### Added
- New environment initialization command
- Comprehensive error handling improvements
- Enhanced documentation across the codebase

### Fixed
- Improve error handling and add comprehensive documentation
- Resolve multiple critical issues in classification and processing
- Fixed various typos in codebase

### Changed
- **BREAKING CHANGE**: Major version bump due to significant architectural improvements
- Enhanced stability and reliability of classification algorithms
- Improved processing workflows and error handling

## [3.70] - 2024-XX-XX

### Fixed
- Fix bug with new gdal import from osgeo

## [3.7] - 2024-XX-XX

### Fixed
- Fix bug #31

## [3.64] - 2024-XX-XX

### Added
- Add closing filter in the processing toolbox
- Median and closing filter functionality

## [3.63] - 2024-XX-XX

### Fixed
- Fix bug in train algorithm (split was percent of train not of validation)

## [3.62] - 2024-XX-XX

### Fixed
- Fix bug when loading cursor was not removed after unsuccessful learning

## [3.61] - 2024-XX-XX

### Fixed
- Fix bug #19 with self.addAlgorithm(alg)

## [3.6] - 2024-XX-XX

### Added
- Add confidence map in processing
- Add median filter and shannon entropy
- Move dzetsaka icons to extension toolbar

### Fixed
- Fix bug with GMM confidence map

## [3.5.1] - 2023-XX-XX

### Fixed
- Fix bug in algorithm with vector files

## [3.5] - 2023-XX-XX

### Fixed
- Fix bug with gpkg files in train algorithm provider

### Changed
- Update to install scikit-learn on windows

## [3.4.8] - 2023-XX-XX

### Fixed
- Fix bug when classes >44

## [3.4.7] - 2023-XX-XX

### Added
- Support more than 255 classes to predict

## [3.4.6] - 2023-XX-XX

### Fixed
- Fixes in processing

## [3.4.5] - 2023-XX-XX

### Fixed
- Fix bug #17, if model is loaded, do not search for a vector file

## [3.4.4] - 2023-XX-XX

### Added
- Add version modification

## [3.4.3] - 2023-XX-XX

### Added
- Create LICENSE

## [3.4.2] - 2023-XX-XX

### Changed
- Version update

## [3.4.1] - 2023-XX-XX

### Changed
- Version update

## [3.4] - 2023-XX-XX

### Added
- Welcome help functionality

## [3.3.1] - 2023-XX-XX

### Changed
- Use gdal instead of osgeo.gdal

## [3.3] - 2023-XX-XX

### Fixed
- Fix bugs with processing toolbox

### Changed
- Various improvements

## [3.2] - 2023-XX-XX

### Changed
- Beta version with experimental processing providers

## [3.1] - 2023-XX-XX

### Added
- Experimental processing providers

## [3.0.2] - 2022-XX-XX

### Added
- Progress bar for GUI

### Fixed
- Try correcting closing dock

## [3.0.1] - 2022-XX-XX

### Fixed
- Minor fixes

## [3.0.0] - 2022-XX-XX

### Changed
- **BREAKING CHANGE**: Major version release - dzetsaka v3.0
- Replace scipy by numpy
- Major fixes and new tools for python3

### Added
- New processing functions
- Confirmation box if different projection
- Manage push message
- Rewrite feedback & resample SITS

## [2.5.1] - 2021-XX-XX

### Fixed
- Correct bug with confusion matrix (if split<100%)

## [2.5] - 2021-XX-XX

### Fixed
- v2.4.5 verification step and bug split

## [2.4.4] - 2021-XX-XX

### Changed
- Adapt for model_selection in sklearn >= 0.18
- KFold v.18 & .20

## [2.4.3] - 2021-XX-XX

### Changed
- Sklearn validation updates

## [2.4.2] - 2021-XX-XX

### Changed
- Version update

## [2.4.1] - 2021-XX-XX

### Added
- Split train/validation functionality

## [2.4] - 2021-XX-XX

### Added
- DTW (Dynamic Time Warping) functionality

## [2.3.1] - 2020-XX-XX

### Changed
- Version update

## [2.3] - 2020-XX-XX

### Changed
- Version update

## [2.2] - 2020-XX-XX

### Added
- Confidence functionality
- Saved directory feature

## [2.1.2] - 2020-XX-XX

### Fixed
- Correct bug saving classification
- Bug with output file

## [2.1.1] - 2020-XX-XX

### Changed
- Version update

## [2.1] - 2020-XX-XX

### Added
- Processing support

## [2.0.3] - 2019-XX-XX

### Changed
- Version update

## [2.0.2] - 2019-XX-XX

### Fixed
- Bug with loading model

## [2.0.1] - 2019-XX-XX

### Changed
- Version update

## [2.0] - 2019-XX-XX

### Changed
- **BREAKING CHANGE**: Major version release

## [1.2] - 2018-XX-XX

### Changed
- Version update

## [1.1] - 2018-XX-XX

### Added
- Processing error management

## [1.0.1] - 2018-XX-XX

### Added
- Mask support

## [1.0] - 2018-XX-XX

### Added
- Verification process
- Initial stable release

## [0.1] - 2018-XX-XX

### Added
- Initial release of dzetsaka classification tool