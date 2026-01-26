# Change Log

All notable changes to the "mlir-inc-previewer" extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fix bug of previewing help README file

### Changed

- Change !NOTE to NOTE in README.md

## [0.0.7] - 2026-01-25

### Changed

- Update README.md
- Change the tile of one command from `MLIR Inc: Navigate Next Preview` to `MLIR Inc: Navigate to Next Preview`

## [0.0.6] - 2026-01-21

### Added

- Add test-release.sh script

### Changed

- Wrapped inserted preview content with `// clang-format off/on` markers to prevent automatic formatting from modifying the inserted text and making it difficult to remove during preview collapse.
- Merged several edit actions into a single `await editor.edit()` call to reduce VS Code UI update latency and improve perceived responsiveness.

## [0.0.5] - 2026-01-20

### Added

- Enable removing unrelated preview blocks based on the current file's conditional compilation when expanding all previews

### Changed

- Reorder expandAllPreview from reverse to forward iteration

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
