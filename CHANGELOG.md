# Change Log

All notable changes to the "mlir-inc-previewer" extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add support to remove unrelated preview blocks based on conditional compilation

## [0.0.4] - 2026-01-19

### Added

- Add comprehensive test suite using Jest in GitHub Actions workflow
- Support dev* branch pattern for better development workflow
- Add MIT license headers to all source files

### Fixed

- Fix CHANGELOG dates and add new features section
- Fix icon path from `icon.png` to `docs/icon.png`
- Fix example GIF in README

## [0.0.3] - 2026-01-12

### Added

- Add support for collapsing previews if the cursor is inside a preview block
- Add comment for original include line after expanding preview
- Add 'expandAll' command to expand all previews

### Changed

- Change the help showing method to directly opening the README file
- Refactored the code
- Replace Usage Example GIF in README and help

## [0.0.2] - 2026-01-12

### Added

- Support comprehensive C/C++ file extensions including:
  - Standard: .c, .cpp, .cxx, .h, .hpp, .hxx
  - Variants: .cc, .cp, .c++, .hh, .hp, .h++
  - Inline/template: .inl, .inc, .ipp, .tcc, .tpp
  - Definitions: .def
  - CUDA: .cu, .cuh
- Register 'cleanAndSave' command for keyboard shortcut functionality
- Update context menu groups to include cleanAndSave command at position 3
- Refine activation conditions to check both editor language ID and file extension
- Add Usage Example GIF in README and help

### Fixed

- Fix incorrect regex pattern that previously matched any language containing 'c'

## [0.0.1] - 2026-01-11

### Added

- Initial release
