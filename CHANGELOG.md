# Change Log

All notable changes to the "mlir-inc-previewer" extension will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### CHANGED

- Replace Usage Example GIF in README and help

## [0.0.2] - 2025-05-12

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

## [0.0.1] - 2025-05-11

### Added

- Initial release
