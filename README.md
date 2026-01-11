# MLIR Inc Previewer - VS Code Extension

## 📖 Overview

[MLIR Inc Previewer](https://github.com/ConvolutedDog/mlir-inc-previewer) is a VS Code extension designed for MLIR developers to quickly preview and manage `.inc` file content. It allows you to preview included `.inc` file contents without leaving your current file.

![Usage Example GIF](https://github.com/ConvolutedDog/mlir-inc-previewer/blob/gif/Usage.gif)

## ✨ Core Features

#### 🔄 Intelligent Expand/Collapse

- **One-click Expansion**: Press `Ctrl+Shift+U` near `#include "xxx.inc"` statements to view .inc file content
- **One-click Collapse**: Press the same shortcut again or use right-click menu to collapse preview content
- **Smart Detection**: Automatically detects .inc include statements near cursor position

#### 🧹 Preview Management

- **Status Bar Display**: Shows real-time count of un-cleaned previews in current file
- **Batch Cleanup**: Click status bar or use right-click menu to clean all previews at once
- **Clean and Save**: Combined command to clean previews and save file in one operation

#### 🧭 Navigation

- **Navigate to Next**: Jump to the next preview block in the file
- **Context Menu**: All commands available via right-click in C/C++ files
- **Command Palette**: Access all features via VS Code command palette

## 🚀 Quick Start

1. Open a C/C++ file containing `.inc` include statements
2. Place cursor near `#include "xxx.inc"`
3. Use one of these methods:
   - Press `Ctrl+Shift+U`
   - Right-click -> MLIR menu
   - Command palette -> Search "MLIR"

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+U` | Toggle .inc preview |
| `Ctrl+S` | Clean all previews and save file |

## 📋 Requirements

- [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) extension for accurate .inc file navigation

## 📋 Available Commands

- **MLIR Inc: Expand/Collapse Preview** - Toggle .inc content display
- **MLIR Inc: Clean All Preview Content** - Remove all preview blocks
- **MLIR Inc: Navigate Next Preview** - Jump to next preview block
- **MLIR Inc: Show Help** - Display help documentation

## 🔧 How It Works

1. When you press `Ctrl+Shift+U` near a `.inc` include line:
   - Extension finds the target .inc file using VS Code's definition provider
   - Reads the .inc file content
   - Inserts it below the include line with special markers
   - Updates status bar with preview count

2. Preview blocks are marked with special comments:

   ```cpp
   /// --- [MLIR_INC_PREVIEW_START] ---
   /// .inc file content here
   /// --- [MLIR_INC_PREVIEW_END] ---
   ```

## 📥 Installation

1. Open VS Code
2. Go to Extensions view (`Ctrl+Shift+X`)
3. Search for "MLIR Inc Previewer"
4. Click Install

## 🤝 Contributing

This extension is open for contributions. Please submit issues and pull requests on the GitHub repository.
