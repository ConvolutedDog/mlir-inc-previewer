// Copyright (c) 2026 Jianchao Yang
// Licensed under the MIT License - see the LICENSE file for details.

import * as vscode from 'vscode';

import {registerCommands} from './commands';
import {StatusBarManager} from './statusBarManager';

let statusBarManager: StatusBarManager;

/**
 * Activates the extension.
 */
export function activate(context: vscode.ExtensionContext) {
  // Initialize status bar manager
  statusBarManager = new StatusBarManager();
  statusBarManager.getStatusBarItem().show();

  // Register commands
  const commandDisposables = registerCommands(context);
  context.subscriptions.push(...commandDisposables);

  // Register event listeners
  context.subscriptions.push(
      vscode.window.onDidChangeActiveTextEditor(() => {
        statusBarManager.updateStatusBar();
      }),

      vscode.workspace.onDidChangeTextDocument((event) => {
        const activeEditor = vscode.window.activeTextEditor;
        if (activeEditor && event.document === activeEditor.document) {
          statusBarManager.updateStatusBar();
        }
      }));

  // Initial status bar update
  statusBarManager.updateStatusBar();

  // Add status bar to subscriptions
  context.subscriptions.push(statusBarManager.getStatusBarItem());

  console.log('MLIR Inc Preview extension activated');
}

export function deactivate() {
  if (statusBarManager) {
    statusBarManager.dispose();
  }
  console.log('MLIR Inc Preview extension deactivated');
}
