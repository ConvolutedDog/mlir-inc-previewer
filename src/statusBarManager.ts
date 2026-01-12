import * as vscode from 'vscode';

import {detectPreviewBlocksFromDoc} from './helpers';

/**
 * Manages the status bar item that shows the number of .inc previews.
 */
export class StatusBarManager {
  private statusBarItem: vscode.StatusBarItem;

  constructor() {
    this.statusBarItem =
        vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    // Click to clean all previews
    this.statusBarItem.command = 'mlir-inc-previewer.cleanAll';
  }

  /**
   * Returns the status bar item.
   */
  public getStatusBarItem(): vscode.StatusBarItem {
    return this.statusBarItem;
  }

  /**
   * Updates the status bar with the number of .inc previews.
   */
  public updateStatusBar(): void {
    const editor = vscode.window.activeTextEditor;

    if (!editor) {
      this.setCleanState();
      return;
    }

    const doc = editor.document;
    const previewCount = detectPreviewBlocksFromDoc(doc);

    if (previewCount > 0) {
      this.setWarningState(previewCount);
    } else {
      this.setCleanState();
    }
  }

  /**
   * Sets the status bar to show the number of .inc previews.
   */
  private setWarningState(previewCount: number): void {
    this.statusBarItem.text = `$(trash) MLIR Inc Preview: ${
        previewCount} Preview${previewCount > 1 ? 's' : ''}`;
    this.statusBarItem.tooltip = `MLIR Inc Preview: Click to clean ${
        previewCount} .inc preview${previewCount > 1 ? 's' : ''}`;

    try {
      this.statusBarItem.backgroundColor =
          new vscode.ThemeColor('statusBarItem.warningBackground');
    } catch (e) {
      // Theme color might not exist in all themes
    }
  }

  /**
   * Sets the status bar to show that there are no .inc previews.
   */
  private setCleanState(): void {
    this.statusBarItem.text = `$(check) MLIR Inc Preview: Clean`;
    this.statusBarItem.tooltip = 'MLIR Inc Preview: No .inc previews to clean';
    this.statusBarItem.backgroundColor = undefined;
  }

  /**
   * Disposes the status bar item.
   */
  public dispose(): void {
    this.statusBarItem.dispose();
  }
}
