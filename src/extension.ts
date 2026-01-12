import * as fs from 'fs';
import * as path from 'path';
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  // =========================================================================
  // TAG DEFINITIONS
  // =========================================================================
  // These tags are used to mark the beginning and end of inserted .inc content.
  // They allow us to identify and later remove the preview blocks.
  const BEGIN_TAG = '/// --- [MLIR_INC_PREVIEW_START] ---';
  const END_TAG = '/// --- [MLIR_INC_PREVIEW_END] ---';

  // =========================================================================
  // STATUS BAR ITEM SETUP
  // =========================================================================
  // Create a status bar item that shows the current preview count and allows
  // quick cleanup. Position it on the right side with priority 100.
  const myStatusBarItem =
      vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);

  // =========================================================================
  // STATUS BAR UPDATE FUNCTION
  // =========================================================================
  /**
   * Updates the status bar item based on the current editor state.
   *
   * Design rationale:
   * - Shows preview count when previews exist (with warning color)
   * - Shows clean state when no previews exist
   * - Updates automatically on editor/content changes
   * - Provides visual feedback about the current file state
   */
  function updateStatusBarItem() {
    const editor = vscode.window.activeTextEditor;

    if (!editor) {
      // No active editor: show default state
      myStatusBarItem.text = `$(check) MLIR Inc Preview: Clean`;
      myStatusBarItem.tooltip = 'MLIR Inc Preview: No .inc previews to clean';
      return;
    }

    const doc = editor.document;
    const previewCount = detectPreviewBlocks(doc.getText());

    if (previewCount > 0) {
      // Preview blocks exist: show count with warning indication
      myStatusBarItem.text = `$(trash) MLIR Inc Preview: ${
          previewCount} Preview${previewCount > 1 ? 's' : ''}`;
      myStatusBarItem.tooltip = `MLIR Inc Preview: Click to clean ${
          previewCount} .inc preview${previewCount > 1 ? 's' : ''}`;

      // Add warning background color to draw attention
      try {
        myStatusBarItem.backgroundColor =
            new vscode.ThemeColor('statusBarItem.warningBackground');
      } catch (e) {
        // Theme color might not exist in all themes
      }
    } else {
      // No preview blocks: show clean state
      myStatusBarItem.text = `$(check) MLIR Inc Preview: Clean`;
      myStatusBarItem.tooltip = 'MLIR Inc Preview: No .inc previews to clean';

      // Reset to default background
      myStatusBarItem.backgroundColor = undefined;
    }
  }

  // =========================================================================
  // PREVIEW DETECTION FUNCTIONS
  // =========================================================================
  /**
   * Detects the number of preview blocks in the given text.
   *
   * Algorithm:
   * 1. Split text into lines
   * 2. Track when we enter a preview block (BEGIN_TAG)
   * 3. Track when we exit a preview block (END_TAG)
   * 4. Count each complete block
   *
   * This handles nested blocks correctly by using a state machine approach.
   */
  function detectPreviewBlocks(text: string): number {
    let previewCount = 0;
    let isInsideRegion = false;
    const lines = text.split('\n');

    for (const line of lines) {
      if (line.includes(BEGIN_TAG)) {
        if (!isInsideRegion) {
          previewCount++;
          isInsideRegion = true;
        }
      } else if (line.includes(END_TAG)) {
        isInsideRegion = false;
      }
    }

    return previewCount;
  }

  /**
   * Checks if a line of text is an #include statement for a .inc file.
   *
   * Parsing logic:
   * 1. Remove inline comments (everything after //)
   * 2. Match #include pattern with quotes or angle brackets
   * 3. Check if the included path contains ".inc"
   */
  function isIncIncludeLine(lineText: string): boolean {
    // Must be a preprocessor directive
    if (!lineText.includes('#')) {
      return false;
    }

    // Remove comments to avoid false positives
    const lineWithoutComment = lineText.split('//')[0].trim();

    // Match #include pattern
    const includeMatch = lineWithoutComment.match(/^#include\s+["<](.+?)[">]/);
    if (!includeMatch) {
      return false;
    }

    // Check if it's a .inc file
    const includePath = includeMatch[1];
    return includePath.toLowerCase().includes('.inc');
  }

  /**
   * Finds the nearest .inc include line around the current cursor position.
   *
   * Search strategy:
   * 1. Check current line first
   * 2. Search up to 5 lines above
   * 3. Search up to 5 lines below
   *
   * This allows the command to work even when the cursor is near but not
   * exactly on the #include line.
   */
  function findIncIncludeLine(
      doc: vscode.TextDocument, currentLine: number): number {
    // Check current line first
    if (isIncIncludeLine(doc.lineAt(currentLine).text)) {
      return currentLine;
    }

    // Search upward (more likely to find the include)
    for (let offset = 1; offset <= 5; offset++) {
      const lineIndex = currentLine - offset;
      if (lineIndex >= 0 && isIncIncludeLine(doc.lineAt(lineIndex).text)) {
        return lineIndex;
      }
    }

    // Search downward
    for (let offset = 1; offset <= 5; offset++) {
      const lineIndex = currentLine + offset;
      if (lineIndex < doc.lineCount &&
          isIncIncludeLine(doc.lineAt(lineIndex).text)) {
        return lineIndex;
      }
    }

    return -1;  // Not found
  }

  // =========================================================================
  // STATUS BAR INITIALIZATION AND EVENT LISTENERS
  // =========================================================================
  // Initialize status bar with cleanup command
  myStatusBarItem.command = 'mlir-inc-previewer.cleanAll';
  myStatusBarItem.show();
  context.subscriptions.push(myStatusBarItem);

  // Update status bar when active editor changes
  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(() => {
    updateStatusBarItem();
  }));

  // Update status bar when document content changes
  context.subscriptions.push(
      vscode.workspace.onDidChangeTextDocument((event) => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor && event.document === activeEditor.document) {
          updateStatusBarItem();
        }
      }));

  // Initial update
  updateStatusBarItem();

  // =========================================================================
  // COMMAND 1: TOGGLE PREVIEW (Alt+U)
  // =========================================================================
  /**
   * Main command for toggling .inc file previews.
   *
   * Two modes:
   * 1. COLLAPSE: If next line has BEGIN_TAG, find END_TAG and delete the block
   * 2. EXPAND: Use VS Code's definition provider to find the .inc file,
   *    read it, and insert its content with BEGIN/END tags
   *
   * Key design decisions:
   * - Uses VS Code's built-in definition provider for reliable file resolution
   * - Only works near .inc include lines to prevent accidental expansions
   * - Preserves cursor position for better UX
   */
  let openDisposable = vscode.commands.registerCommand(
      'mlir-inc-previewer.openSide', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const doc = editor.document;
        const currentLine = editor.selection.active.line;

        // Find the nearest .inc include line
        const incIncludeLine = findIncIncludeLine(doc, currentLine);

        if (incIncludeLine === -1) {
          vscode.window.showWarningMessage(
              'MLIR Inc Preview: No .inc include statement found near cursor');
          return;
        }

        // ==================== COLLAPSE LOGIC ====================
        // Check if the next line already has an expanded preview
        const nextLineIdx = Math.min(incIncludeLine + 1, doc.lineCount - 1);
        if (doc.lineAt(nextLineIdx).text.includes(BEGIN_TAG)) {
          let endLineIdx = -1;
          // Find the matching END_TAG
          for (let i = nextLineIdx; i < doc.lineCount; i++) {
            if (doc.lineAt(i).text.includes(END_TAG)) {
              endLineIdx = i;
              break;
            }
          }
          if (endLineIdx !== -1) {
            // Delete the entire preview block
            await editor.edit(editBuilder => {
              editBuilder.delete(
                  new vscode.Range(nextLineIdx, 0, endLineIdx + 1, 0));
            });
            updateStatusBarItem();
            return;
          }
        }

        // ==================== EXPAND LOGIC ====================
        try {
          // Move cursor to the include line for accurate definition lookup
          const targetPosition = new vscode.Position(incIncludeLine, 0);
          editor.selection =
              new vscode.Selection(targetPosition, targetPosition);

          // Use VS Code's definition provider to find the .inc file
          // This leverages the editor's existing intelligence about include
          // paths
          const locations = await vscode.commands.executeCommand<any>(
              'vscode.executeDefinitionProvider', doc.uri, targetPosition);
          if (locations && locations.length > 0) {
            const target = Array.isArray(locations) ? locations[0] : locations;
            const targetUri: vscode.Uri = target.uri || target.targetUri;

            // Verify it's a .inc file
            if (targetUri.fsPath.endsWith('.inc') ||
                targetUri.fsPath.includes('.inc')) {
              try {
                // Read the .inc file content
                const content = fs.readFileSync(targetUri.fsPath, 'utf8');
                const insertText = `${BEGIN_TAG}\n/// MLIR Inc File: ${
                    targetUri.fsPath}\n${content}\n${END_TAG}\n`;

                // Insert the content after the include line
                await editor.edit(editBuilder => {
                  editBuilder.insert(
                      new vscode.Position(incIncludeLine + 1, 0), insertText);
                });

                updateStatusBarItem();
                vscode.window.setStatusBarMessage(
                    `$(check) MLIR Inc Preview: Expanded ${targetUri.fsPath}`,
                    2000);
              } catch (readError: any) {
                vscode.window.showErrorMessage(
                    `MLIR Inc Preview: Cannot read file: ${
                        targetUri.fsPath} - ${readError.message}`);
              }
            } else {
              vscode.window.showInformationMessage(
                  'MLIR Inc Preview: Target is not a .inc file');
            }
          } else {
            vscode.window.showInformationMessage(
                'MLIR Inc Preview: No .inc file definition found');
          }
        } catch (e: any) {
          vscode.window.showErrorMessage(
              'MLIR Inc Preview: Cannot expand: ' + (e.message || e));
        }
      });

  // =========================================================================
  // COMMAND 2: CLEAN ALL PREVIEWS (Status bar click)
  // =========================================================================
  /**
   * Cleans all preview blocks in the current file.
   *
   * Algorithm:
   * 1. Scan all lines to find BEGIN_TAG and END_TAG pairs
   * 2. Collect deletion ranges for each complete block
   * 3. Delete from bottom to top to maintain correct line numbers
   *
   * Performance considerations:
   * - Linear scan O(n) where n is line count
   * - Batch edit operation for efficiency
   * - Bottom-up deletion prevents line number shifting issues
   */
  let cleanDisposable = vscode.commands.registerCommand(
      'mlir-inc-previewer.cleanAll', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const doc = editor.document;
        let deleteRanges: vscode.Range[] = [];
        let isInsideRegion = false;
        let startLine = -1;

        // Scan all lines to find preview blocks
        for (let i = 0; i < doc.lineCount; i++) {
          const lineText = doc.lineAt(i).text;

          if (lineText.includes(BEGIN_TAG)) {
            isInsideRegion = true;
            startLine = i;
          } else if (lineText.includes(END_TAG) && isInsideRegion) {
            // Found a complete block: from startLine to current line
            // (inclusive)
            deleteRanges.push(new vscode.Range(startLine, 0, i + 1, 0));
            isInsideRegion = false;
          }
        }

        // Execute deletion if blocks were found
        if (deleteRanges.length > 0) {
          await editor.edit(editBuilder => {
            // Delete from bottom to top to maintain correct indices
            for (let i = deleteRanges.length - 1; i >= 0; i--) {
              editBuilder.delete(deleteRanges[i]);
            }
          }, {undoStopBefore: false, undoStopAfter: false});

          const message =
              `MLIR Inc Preview: Cleaned ${deleteRanges.length} preview${
                  deleteRanges.length > 1 ? 's' : ''}`;
          vscode.window.setStatusBarMessage(`$(check) ${message}`, 3000);
          updateStatusBarItem();
        } else {
          const message = 'MLIR Inc Preview: No preview blocks found';
          vscode.window.setStatusBarMessage(`$(check) ${message}`, 3000);
        }
      });

  // =========================================================================
  // COMMAND 3: CLEAN AND SAVE
  // =========================================================================
  /**
   * Combination command: clean all previews then save the file.
   *
   * Workflow:
   * 1. Record current state (for comparison)
   * 2. Execute cleanAll command
   * 3. Wait briefly for editor update
   * 4. Save document
   * 5. Show appropriate feedback
   *
   * This provides a safe way to ensure files are saved without temporary
   * preview content.
   */
  let cleanSaveDisposable = vscode.commands.registerCommand(
      'mlir-inc-previewer.cleanAndSave', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const document = editor.document;
        const originalContent = document.getText();

        // Record state before cleaning
        const originalPreviewCount = detectPreviewBlocks(originalContent);

        // Execute the already-tested cleanAll command
        await vscode.commands.executeCommand('mlir-inc-previewer.cleanAll');

        // Brief delay to ensure editor updates
        await new Promise(resolve => setTimeout(resolve, 50));

        // Check what changed
        const newContent = document.getText();
        const newPreviewCount = detectPreviewBlocks(newContent);
        const previewsCleaned = originalPreviewCount - newPreviewCount;

        // Save the cleaned document
        await editor.document.save();

        // Provide appropriate feedback
        if (previewsCleaned > 0) {
          vscode.window.setStatusBarMessage(
              `$(check) Cleaned ${previewsCleaned} preview${
                  previewsCleaned > 1 ? 's' : ''} and saved`,
              3000);
        } else {
          vscode.window.setStatusBarMessage(
              '$(check) MLIR Inc Preview: No preview blocks found to clean, file saved',
              3000);
        }

        updateStatusBarItem();
      });

  // =========================================================================
  // COMMAND 4: NAVIGATE TO NEXT PREVIEW
  // =========================================================================
  /**
   * Navigates to the next preview block in the file.
   *
   * Useful for:
   * - Quick navigation through multiple previews
   * - Reviewing expanded content
   * - Manual cleanup of specific blocks
   */
  let NavigateNextDisposable = vscode.commands.registerCommand(
      'mlir-inc-previewer.navigateNext', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const doc = editor.document;
        const currentLine = editor.selection.active.line;
        let nextPreviewStart = -1;

        // Find the first BEGIN_TAG AFTER current position (currentLine + 1)
        for (let i = currentLine + 1; i < doc.lineCount; i++) {
          if (doc.lineAt(i).text.includes(BEGIN_TAG)) {
            nextPreviewStart = i;
            break;
          }
        }

        // If not found after current position, search from beginning
        if (nextPreviewStart === -1) {
          for (let i = 0; i < currentLine; i++) {
            if (doc.lineAt(i).text.includes(BEGIN_TAG)) {
              nextPreviewStart = i;
              break;
            }
          }
        }

        if (nextPreviewStart !== -1) {
          // Jump to the preview block
          const position = new vscode.Position(nextPreviewStart, 0);
          editor.selection = new vscode.Selection(position, position);
          editor.revealRange(new vscode.Range(position, position));

          vscode.window.setStatusBarMessage(
              `$(check) MLIR Inc Preview: Jumped to preview (line ${
                  nextPreviewStart + 1})`,
              2000);
        } else {
          vscode.window.setStatusBarMessage(
              '$(check) MLIR Inc Preview: No more preview blocks found', 2000);
        }
      });

  // =========================================================================
  // REGISTER ALL COMMANDS
  // =========================================================================
  context.subscriptions.push(
      openDisposable, cleanDisposable, cleanSaveDisposable,
      NavigateNextDisposable);

  // =========================================================================
  // HELPER COMMAND: SHOW HELP
  // =========================================================================
  context.subscriptions.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.showHelp', async () => {
        try {
          const content = `# MLIR Inc Previewer - VS Code Extension

## 📖 Overview

[MLIR Inc Previewer](https://github.com/ConvolutedDog/mlir-inc-previewer) is a VS Code extension designed for MLIR developers to quickly preview and manage \`.inc\` file content. It allows you to preview included \`.inc\` file contents without leaving your current file.

<div align="center">
  <img src="https://private-user-images.githubusercontent.com/102723346/534348420-7977b6bf-3b07-4d52-9dd7-767974c0a2b1.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjgyMDAxMTAsIm5iZiI6MTc2ODE5OTgxMCwicGF0aCI6Ii8xMDI3MjMzNDYvNTM0MzQ4NDIwLTc5NzdiNmJmLTNiMDctNGQ1Mi05ZGQ3LTc2Nzk3NGMwYTJiMS5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTEyJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExMlQwNjM2NTBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wNWYzMmVjZGUwMTI5NGVmYjQ5MDg3NWY2MzczMWU4ZWM2MTU3ZjdmMDdhNDZiYzFhYjM4NzhkNGU5OGY0NzE5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.Nh49F9JcgukDDe6VUUqFCBuvVvkyXON6uv-Hph_RkQU" alt="Usage Example" width="800">
  <br>
  <strong>MLIR Inc Previewer Usgae Example</strong>
  <br>
</div>

## ✨ Core Features

#### 🔄 Intelligent Expand/Collapse

- **One-click Expansion**: Press \`Ctrl+Shift+U\` near \`#include "xxx.inc"\` statements to view .inc file content
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

1. Open a C/C++ file containing \`.inc\` include statements
2. Place cursor near \`#include "xxx.inc"\`
3. Use one of these methods:
   - Press \`Ctrl+Shift+U\`
   - Right-click -> MLIR menu
   - Command palette -> Search "MLIR"

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| \`Ctrl+Shift+U\` | Toggle .inc preview |
| \`Ctrl+S\` | Clean all previews and save file |

## 📋 Requirements

- [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) extension for accurate .inc file navigation

## 📋 Available Commands

- **MLIR Inc: Expand/Collapse Preview** - Toggle .inc content display
- **MLIR Inc: Clean All Preview Content** - Remove all preview blocks
- **MLIR Inc: Navigate Next Preview** - Jump to next preview block
- **MLIR Inc: Show Help** - Display help documentation

## 🔧 How It Works

1. When you press \`Ctrl+Shift+U\` near a \`.inc\` include line:
   - Extension finds the target .inc file using VS Code's definition provider
   - Reads the .inc file content
   - Inserts it below the include line with special markers
   - Updates status bar with preview count

2. Preview blocks are marked with special comments:

   \`\`\`cpp
   /// --- [MLIR_INC_PREVIEW_START] ---
   /// .inc file content here
   /// --- [MLIR_INC_PREVIEW_END] ---
   \`\`\`

## 📥 Installation

1. Open VS Code
2. Go to Extensions view (\`Ctrl+Shift+X\`)
3. Search for "MLIR Inc Previewer"
4. Click Install

## 🤝 Contributing

This extension is open for contributions. Please submit issues and pull requests on the GitHub repository.
`;

          const uri = vscode.Uri.parse(
              'untitled:' +
              'MLIR_Inc_Previewer_Help.md');
          const doc = await vscode.workspace.openTextDocument(uri);
          const edit = new vscode.WorkspaceEdit();
          edit.insert(uri, new vscode.Position(0, 0), content);
          await vscode.workspace.applyEdit(edit);
          await vscode.commands.executeCommand('markdown.showPreview', uri);
        } catch (error) {
          vscode.window.showErrorMessage(`Failed to show help: ${error}`);
        }
      }));

  // =========================================================================
  // ACTIVATION MESSAGE
  // =========================================================================
  console.log('MLIR Inc Preview extension activated');
}

export function deactivate() {
  // Clean up resources
  console.log('MLIR Inc Preview extension deactivated');
}
