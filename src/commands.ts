// Copyright (c) 2026 Jianchao Yang
// Licensed under the MIT License - see the LICENSE file for details.

import * as vscode from 'vscode';

import {showHelpHelper} from './helpers';
import {PreviewManager} from './previewManager';

/**
 * Register commands for the extension
 */
export function registerCommands(context: vscode.ExtensionContext):
    vscode.Disposable[] {
  const disposables: vscode.Disposable[] = [];

  // Command 1: Expand/Collapse Preview
  disposables.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.openSide', async () => {
        await PreviewManager.togglePreview();
      }));

  // Command 2: Expand All Preview
  disposables.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.expandAll', async () => {
        await PreviewManager.expandAllPreview();
      }));

  // Command 3: Clean All Preview Content
  disposables.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.cleanAll', async () => {
        await PreviewManager.cleanAllPreviews();
      }));

  // Command 4: Clean and Save
  disposables.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.cleanAndSave', async () => {
        await PreviewManager.cleanAndSave();
      }));

  // Command 5: Navigate to Next Preview
  disposables.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.navigateNext', async () => {
        await PreviewManager.navigateToNextPreview();
      }));

  // Command 6: Show Help
  disposables.push(vscode.commands.registerCommand(
      'mlir-inc-previewer.showHelp', async () => {
        await showHelpHelper.showHelp();
      }));

  return disposables;
}
