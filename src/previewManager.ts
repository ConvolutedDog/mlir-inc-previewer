// Copyright (c) 2026 Jianchao Yang
// Licensed under the MIT License - see the LICENSE file for details.

import {assert} from 'console';
import * as fs from 'fs';
import * as vscode from 'vscode';

import {detectPreviewBlocksFromDoc, findAllCommentedIncludeLines, findAllIncIncludeLine, findAllPreviewBlocks, findCommentedIncIncludeLine, findIncIncludeLine} from './helpers';
import {MacroStateManager} from './macrosManager';
import {BEGIN_COMMENT_LINE, BEGIN_TAG, END_COMMENT_LINE, END_TAG, ifRemoveUnrelatedPreviewBlocks} from './types';

/**
 * Preview manager for .inc files.
 */
export class PreviewManager {
  /**
   * Toggles preview for .inc file at current cursor position.
   */
  public static togglePreview = async(): Promise<void> => {
    const editor: vscode.TextEditor|undefined = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc: vscode.TextDocument = editor.document;
    const currentLine: number = editor.selection.active.line;

    // If found a commented include line, reverse it
    const commentedIncIncludeLine: number =
        findCommentedIncIncludeLine(doc, currentLine);

    // Check for collapse
    if (commentedIncIncludeLine !== -1) {
      // TODO: If the cursor is in a preview block range, we can also collapse.
      if (await PreviewManager.tryCollapsePreview(
              editor, doc, commentedIncIncludeLine)) {
        return;
      } else {
        vscode.window.showErrorMessage(
            'MLIR Inc Preview: Found commented include line but not successfully collapsed');
      }
    } else {
      const incIncludeLine: number = findIncIncludeLine(doc, currentLine);
      if (incIncludeLine !== -1) {
        // Otherwise expand
        await PreviewManager.expandPreview(editor, doc, incIncludeLine);
      } else {
        vscode.window.showWarningMessage(
            'MLIR Inc Preview: No .inc include statement found near cursor');
        return;
      }
    }
  };

  /**
   * Attempts to collapse an existing preview.
   *
   * IMPORTANT:
   * We must avoid doing multiple editor.edit() calls based on old line indices.
   * Otherwise, once we delete clang-format lines, the preview block line
   * numbers shift and we may delete wrong ranges.
   */
  private static tryCollapsePreview = async(
      editor: vscode.TextEditor, doc: vscode.TextDocument,
      incIncludeLine: number): Promise<boolean> => {
    // Find the BEGIN_TAG line after the include line (do NOT assume it is the
    // immediate next line, because we inserted "// clang-format on" in
    // between).
    let beginLineIdx = -1;
    for (let i = incIncludeLine + 1; i < doc.lineCount; i++) {
      if (doc.lineAt(i).text.includes(BEGIN_TAG)) {
        beginLineIdx = i;
        break;
      }
    }
    if (beginLineIdx === -1) {
      vscode.window.showWarningMessage(
          `MLIR Inc Preview: No matching ${END_TAG} found for preview`);
      return false;
    }

    // Find the matching END_TAG line after BEGIN_TAG.
    let endLineIdx = -1;
    for (let i = beginLineIdx; i < doc.lineCount; i++) {
      if (doc.lineAt(i).text.includes(END_TAG)) {
        endLineIdx = i;
        break;
      }
    }
    if (endLineIdx === -1) {
      vscode.window.showWarningMessage(
          `MLIR Inc Preview: No matching ${END_TAG} found for preview`);
      return false;
    }

    // Perform ALL operations in ONE edit() call to avoid stale indices:
    // 1) Uncomment the include line
    // 2) Remove clang-format off/on lines around it
    // 3) Delete the preview block
    await editor.edit((editBuilder) => {
      // 1) Uncomment the include line
      const commentedLine = doc.lineAt(incIncludeLine);
      const commentedText = commentedLine.text;
      const trimmedText = commentedText.trim();

      if (trimmedText.endsWith(END_COMMENT_LINE) &&
          trimmedText.startsWith(BEGIN_COMMENT_LINE)) {
        let originalText = commentedText;

        // Remove the leading BEGIN_COMMENT_LINE
        if (originalText.startsWith(BEGIN_COMMENT_LINE)) {
          originalText = originalText.substring(BEGIN_COMMENT_LINE.length);
        }

        // Remove the ending END_COMMENT_LINE
        if (originalText.endsWith(END_COMMENT_LINE)) {
          originalText = originalText.substring(
              0, originalText.length - END_COMMENT_LINE.length);
        }

        editBuilder.replace(
            new vscode.Range(
                incIncludeLine, 0, incIncludeLine, commentedText.length),
            originalText.trim());
      }

      // Track how many lines we delete before the preview block, so we can
      // adjust begin/end indices safely.
      let shiftBeforePreview = 0;

      // 2) Remove "// clang-format off" line (usually above include). We only
      // delete if the exact line is a clang-format directive.
      if (incIncludeLine - 1 >= 0) {
        const offLineIdx = incIncludeLine - 1;
        const offLine = doc.lineAt(offLineIdx);
        if (offLine.text.trim().startsWith('// clang-format off')) {
          editBuilder.delete(
              new vscode.Range(offLineIdx, 0, offLineIdx + 1, 0));
          shiftBeforePreview += 1;
        }
      }

      // 3) Remove "// clang-format on" line.
      if (incIncludeLine + 1 < beginLineIdx) {
        const offLineIdx = incIncludeLine + 1;
        const offLine = doc.lineAt(offLineIdx);
        if (offLine.text.trim().startsWith('// clang-format on')) {
          editBuilder.delete(
              new vscode.Range(offLineIdx, 0, offLineIdx + 1, 0));
          shiftBeforePreview += 1;
        }
      }

      // 4) Delete the preview block from BEGIN_TAG to END_TAG (inclusive)
      editBuilder.delete(new vscode.Range(beginLineIdx, 0, endLineIdx + 1, 0));
    });

    return true;
  };

  /**
   * Expands a new preview for .inc file.
   */
  private static expandPreview = async(
      editor: vscode.TextEditor, doc: vscode.TextDocument,
      incIncludeLine: number): Promise<void> => {
    try {
      const targetPosition = new vscode.Position(incIncludeLine, 0);
      editor.selection = new vscode.Selection(targetPosition, targetPosition);

      const locations = await vscode.commands.executeCommand<any>(
          'vscode.executeDefinitionProvider', doc.uri, targetPosition);

      if (locations && locations.length > 0) {
        const target = Array.isArray(locations) ? locations[0] : locations;
        const targetUri: vscode.Uri = target.uri || target.targetUri;

        if (targetUri.fsPath.endsWith('.inc') ||
            targetUri.fsPath.includes('.inc')) {
          if (ifRemoveUnrelatedPreviewBlocks) {
            // 1. Process the macro state from the beginning of the document to
            // incIncludeLine
            const updatedDoc = editor.document;
            const macroManager = new MacroStateManager();
            for (let i = 0; i < incIncludeLine; i++) {
              const lineText = updatedDoc.lineAt(i).text;
              const trimmed = lineText.trim();
              macroManager.processLine(trimmed, i);
            }

            // 2. Read the .inc file content
            const originalContent = fs.readFileSync(targetUri.fsPath, 'utf8');
            const incLines = originalContent.split('\n');
            const processedLines: string[] = [];

            // 3. Process each line of the .inc file, applying the current macro
            // state
            for (let i = 0; i < incLines.length; i++) {
              const line = incLines[i];
              const trimmed = line.trim();
              macroManager.processLine(trimmed, i + incIncludeLine);

              if (trimmed.startsWith('#if') || trimmed.startsWith('#ifndef') ||
                  trimmed.startsWith('#if defined') ||
                  trimmed.startsWith('#if !defined') ||
                  trimmed.startsWith('#elif') || trimmed.startsWith('#else') ||
                  trimmed.startsWith('#endif')) {
                processedLines.push(line);
                continue;
              }

              if (macroManager.isCurrentLineActive()) {
                processedLines.push(line);
              }
            }

            // 4. Create the processed content
            const processedContent = processedLines.join('\n');
            const insertText =
                `${BEGIN_TAG}\n// clang-format off\n/// MLIR Inc File: ${
                    targetUri.fsPath}\n// clang-format on\n${
                    processedContent}\n${END_TAG}\n`;

            // 5. Insert the processed content
            await editor.edit((editBuilder) => {
              editBuilder.insert(
                  new vscode.Position(incIncludeLine + 1, 0), insertText);
            });

            await editor.edit((editBuilder) => {
              // 6. Comment the original line
              const originalLine = doc.lineAt(incIncludeLine);
              const originalText = originalLine.text;

              if (!originalText.trim().startsWith('//')) {
                // TODO: There may be a bug about clang-format.
                const commentedText =
                    BEGIN_COMMENT_LINE + originalText + END_COMMENT_LINE;
                editBuilder.replace(
                    new vscode.Range(
                        incIncludeLine, 0, incIncludeLine, originalText.length),
                    commentedText);
              }

              // 7. Add clang-format off and clang-format on lines
              editBuilder.insert(
                  new vscode.Position(incIncludeLine, 0),
                  '// clang-format off\n');
              editBuilder.insert(
                  new vscode.Position(incIncludeLine + 1, 0),
                  '// clang-format on\n');
            });
          } else {
            // 1. Read the .inc file content
            const content = fs.readFileSync(targetUri.fsPath, 'utf8');
            const insertText = `${BEGIN_TAG}\n/// MLIR Inc File: ${
                targetUri.fsPath}\n${content}\n${END_TAG}\n`;

            // 2. Insert the included content
            await editor.edit((editBuilder) => {
              editBuilder.insert(
                  new vscode.Position(incIncludeLine + 1, 0), insertText);
            });

            await editor.edit((editBuilder) => {
              // 3. Comment the original line
              const originalLine = doc.lineAt(incIncludeLine);
              const originalText = originalLine.text;

              if (!originalText.trim().startsWith('//')) {
                // TODO: There may be a bug about clang-format.
                const commentedText =
                    BEGIN_COMMENT_LINE + originalText + END_COMMENT_LINE;
                editBuilder.replace(
                    new vscode.Range(
                        incIncludeLine, 0, incIncludeLine, originalText.length),
                    commentedText);
              }

              // 4. Add clang-format off and clang-format on lines
              editBuilder.insert(
                  new vscode.Position(incIncludeLine, 0),
                  '// clang-format off\n');
              editBuilder.insert(
                  new vscode.Position(incIncludeLine + 1, 0),
                  '// clang-format on\n');
            });
          }

          vscode.window.setStatusBarMessage(
              `$(check) MLIR Inc Preview: Expanded ${targetUri.fsPath}`, 2000);
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
  };

  /**
   * Expands all preview blocks in the current file.
   */
  public static expandAllPreview = async(): Promise<void> => {
    const editor: vscode.TextEditor|undefined = vscode.window.activeTextEditor;
    if (!editor) return;

    // Reget document and line number
    let currentDoc = editor.document;
    let allIncLines = findAllIncIncludeLine(currentDoc);

    // Expand each .inc file
    while (allIncLines.length > 0) {
      const incIncludeLine = allIncLines[0];
      await PreviewManager.expandPreview(editor, currentDoc, incIncludeLine);
      // Reget document and line number caused by inserting content
      currentDoc = editor.document;
      allIncLines = findAllIncIncludeLine(currentDoc);
    }
  };

  /**
   * Cleans all preview blocks in the current file.
   *
   * IMPORTANT:
   * When deleting lines, always process from bottom to top so line indices for
   * earlier items do not shift unexpectedly.
   */
  public static cleanAllPreviews = async(): Promise<void> => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    // Always recompute from the latest document snapshot.
    let doc = editor.document;

    const commentedIncludeLines = findAllCommentedIncludeLines(doc);

    if (commentedIncludeLines.length > 0) {
      // Process from bottom to top to avoid shifting line indices.
      for (let idx = commentedIncludeLines.length - 1; idx >= 0; idx--) {
        // Refresh document snapshot each iteration to avoid stale reads.
        doc = editor.document;

        const commentedIncludeLine = commentedIncludeLines[idx];
        const incIncludeLine = commentedIncludeLine.start.line;

        // Reverse the commented include line and remove clang-format lines.
        await editor.edit((editBuilder) => {
          const commentedLine = doc.lineAt(incIncludeLine);
          const commentedText = commentedLine.text;
          const trimmedText = commentedText.trim();

          if (trimmedText.endsWith(END_COMMENT_LINE) &&
              trimmedText.startsWith(BEGIN_COMMENT_LINE)) {
            let originalText = commentedText;

            // Remove the leading BEGIN_COMMENT_LINE
            if (originalText.startsWith(BEGIN_COMMENT_LINE)) {
              originalText = originalText.substring(BEGIN_COMMENT_LINE.length);
            }

            // Remove the ending END_COMMENT_LINE
            if (originalText.endsWith(END_COMMENT_LINE)) {
              originalText = originalText.substring(
                  0, originalText.length - END_COMMENT_LINE.length);
            }

            editBuilder.replace(
                new vscode.Range(
                    incIncludeLine, 0, incIncludeLine, commentedText.length),
                originalText.trim());

            // Remove the clang-format off lines (usually the previous line)
            if (incIncludeLine - 1 >= 0) {
              const clangformatoffLine = doc.lineAt(incIncludeLine - 1);
              if (clangformatoffLine.text.trim().startsWith(
                      '// clang-format off')) {
                editBuilder.delete(
                    new vscode.Range(incIncludeLine - 1, 0, incIncludeLine, 0));
              }
            }

            // Remove the clang-format on line (often between include and
            // BEGIN_TAG) Here we just check the next line, consistent with
            // original logic.
            if (incIncludeLine + 1 < doc.lineCount) {
              const clangformatonLine = doc.lineAt(incIncludeLine + 1);
              if (clangformatonLine.text.trim().startsWith(
                      '// clang-format on')) {
                editBuilder.delete(new vscode.Range(
                    incIncludeLine + 1, 0, incIncludeLine + 2, 0));
              }
            }
          }
        });
      }
    }

    // Refresh doc snapshot after all edits above.
    doc = editor.document;

    const deleteRanges = findAllPreviewBlocks(doc);

    if (deleteRanges.length > 0) {
      await editor.edit((editBuilder) => {
        // Delete from bottom to top to maintain correct indices.
        for (let i = deleteRanges.length - 1; i >= 0; i--) {
          editBuilder.delete(deleteRanges[i]);
        }
      }, {undoStopBefore: false, undoStopAfter: false});

      const message = `MLIR Inc Preview: Cleaned ${
          deleteRanges.length} preview${deleteRanges.length > 1 ? 's' : ''}`;
      vscode.window.setStatusBarMessage(`$(check) ${message}`, 3000);
    } else {
      const message = 'MLIR Inc Preview: No preview blocks found';
      vscode.window.setStatusBarMessage(`$(check) ${message}`, 3000);
    }
  };

  /**
   * Navigates to the next preview block.
   */
  public static navigateToNextPreview = async(): Promise<void> => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc = editor.document;
    const currentLine = editor.selection.active.line;
    let nextPreviewStart = -1;

    // Find the first BEGIN_TAG AFTER current position
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
  };

  /**
   * Cleans previews and saves the file.
   */
  public static cleanAndSave = async(): Promise<void> => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const document: vscode.TextDocument = editor.document;
    const originalPreviewCount: number = detectPreviewBlocksFromDoc(document);

    await PreviewManager.cleanAllPreviews();

    // Brief delay to ensure editor updates
    await new Promise((resolve) => setTimeout(resolve, 50));

    const newPreviewCount = detectPreviewBlocksFromDoc(document);
    if (newPreviewCount !== 0) {
      vscode.window.showErrorMessage(
          'MLIR Inc Preview: Failed to clean all previews.');
      return;
    }

    const previewsCleaned = originalPreviewCount - newPreviewCount;

    await editor.document.save();

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
  };
}
