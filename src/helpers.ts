import * as vscode from 'vscode';

import {BEGIN_COMMENT_LINE, BEGIN_TAG, END_COMMENT_LINE, END_TAG, ifPreviewReadme, SearchIncLineRange} from './types';

/**
 * Detects the number of preview blocks in the given document.
 */
export function detectPreviewBlocksFromDoc(doc: vscode.TextDocument): number {
  let previewCount = 0;
  let isInsideRegion = false;
  for (let i = 0; i < doc.lineCount; i++) {
    const lineText = doc.lineAt(i).text;

    if (lineText.includes(BEGIN_TAG)) {
      previewCount++;
      isInsideRegion = true;
    } else if (lineText.includes(END_TAG) && isInsideRegion) {
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
export function isIncIncludeLine(lineText: string): boolean {
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
 * Checks if a line of text is an #include statement for a .inc file.
 */
export function isCommentedIncIncludeLine(lineText: string): boolean {
  // Must be a preprocessor directive
  if (!lineText.startsWith(BEGIN_COMMENT_LINE) ||
      !lineText.endsWith(END_COMMENT_LINE)) {
    return false;
  }

  // Get the line without comments
  let originalText = lineText;
  // Remove the leading BEGIN_COMMENT_LINE
  originalText = originalText.substring(BEGIN_COMMENT_LINE.length);
  // Remove the ending END_COMMENT_LINE
  originalText =
      originalText.substring(0, originalText.length - END_COMMENT_LINE.length);

  return isIncIncludeLine(originalText);
}

/**
 * Finds the nearest .inc include line around the current cursor position.
 *
 * This allows the command to work even when the cursor is near but not
 * exactly on the #include line.
 */
export function findIncIncludeLine(
    doc: vscode.TextDocument, currentLine: number): number {
  // Check current line first
  if (isIncIncludeLine(doc.lineAt(currentLine).text)) {
    return currentLine;
  }

  // Search upward (more likely to find the include)
  for (let offset = 1; offset <= SearchIncLineRange; offset++) {
    const lineIndex = currentLine - offset;
    if (lineIndex >= 0 && isIncIncludeLine(doc.lineAt(lineIndex).text)) {
      return lineIndex;
    }
  }

  // Search downward
  for (let offset = 1; offset <= SearchIncLineRange; offset++) {
    const lineIndex = currentLine + offset;
    if (lineIndex < doc.lineCount &&
        isIncIncludeLine(doc.lineAt(lineIndex).text)) {
      return lineIndex;
    }
  }

  return -1;  // Not found
}

/**
 * Finds the nearest and commented .inc include line around the current
 * cursor position.
 *
 * This allows the command to work even when the cursor is near but not
 * exactly on the #include line.
 */
export function findCommentedIncIncludeLine(
    doc: vscode.TextDocument, currentLine: number): number {
  // Check current line first
  if (isCommentedIncIncludeLine(doc.lineAt(currentLine).text)) {
    return currentLine;
  }

  // Search upward (more likely to find the include)
  for (let offset = 1; offset <= SearchIncLineRange; offset++) {
    const lineIndex = currentLine - offset;
    if (lineIndex >= 0 &&
        isCommentedIncIncludeLine(doc.lineAt(lineIndex).text)) {
      return lineIndex;
    }
  }

  // Search downward
  for (let offset = 1; offset <= SearchIncLineRange; offset++) {
    const lineIndex = currentLine + offset;
    if (lineIndex < doc.lineCount &&
        isCommentedIncIncludeLine(doc.lineAt(lineIndex).text)) {
      return lineIndex;
    }
  }

  // Check if the cursor is in a expanded preview block
  for (let i = currentLine; i >= 0; i--) {
    const lineText = doc.lineAt(i).text;
    if (lineText === END_TAG) break;
    if (lineText === BEGIN_TAG) {
      for (let j = i; j >= 0; j--) {
        const lineText = doc.lineAt(j).text;
        if (lineText === END_TAG) break;
        if (isCommentedIncIncludeLine(lineText)) return j;
      }
      break;
    }
  }

  return -1;  // Not found
}

/**
 * Finds all .inc include lines in the document.
 */
export function findAllIncIncludeLine(doc: vscode.TextDocument): number[] {
  const incIncludeLines: number[] = [];
  for (let i = 0; i < doc.lineCount; i++) {
    if (isIncIncludeLine(doc.lineAt(i).text)) {
      incIncludeLines.push(i);
    }
  }
  return incIncludeLines;
}

/**
 * Finds all preview blocks in the document.
 */
export function findAllPreviewBlocks(doc: vscode.TextDocument): vscode.Range[] {
  const deleteRanges: vscode.Range[] = [];
  let isInsideRegion = false;
  let startLine = -1;

  for (let i = 0; i < doc.lineCount; i++) {
    const lineText = doc.lineAt(i).text;

    if (lineText.includes(BEGIN_TAG)) {
      isInsideRegion = true;
      startLine = i;
    } else if (lineText.includes(END_TAG) && isInsideRegion) {
      deleteRanges.push(new vscode.Range(startLine, 0, i + 1, 0));
      isInsideRegion = false;
    }
  }

  return deleteRanges;
}

export function findAllCommentedIncludeLines(doc: vscode.TextDocument):
    vscode.Range[] {
  const commentedRanges: vscode.Range[] = [];
  for (let i = 0; i < doc.lineCount; i++) {
    const lineText = doc.lineAt(i).text;
    if (isCommentedIncIncludeLine(lineText)) {
      commentedRanges.push(new vscode.Range(i, 0, i + 1, 0));
    }
  }
  return commentedRanges;
}

/**
 * Helper to show the help documentation.
 */
export class showHelpHelper {
  public static async showHelp(): Promise<void> {
    try {
      // Get the root Uri of the extension
      const extension =
          vscode.extensions.getExtension('yangjianchao16.mlir-inc-previewer');
      if (!extension) {
        vscode.window.showErrorMessage('Extension not found');
        return;
      }

      // Open the README.md file
      const readmeUri =
          vscode.Uri.joinPath(extension.extensionUri, 'README.md');
      const document = await vscode.workspace.openTextDocument(readmeUri);
      await vscode.window.showTextDocument(document, {
        viewColumn: vscode.ViewColumn.One,
        preview: true,
        preserveFocus: true
      });

      // Open the preview
      if (ifPreviewReadme) {
        await vscode.commands.executeCommand(
            'markdown.showPreview', document.uri);
      }
    } catch (error) {
      vscode.window
          .showInformationMessage(
              'MLIR Inc Previewer: Check README.md for documentation',
              'Open GitHub')
          .then(selection => {
            if (selection === 'Open GitHub') {
              vscode.env.openExternal(vscode.Uri.parse(
                  'https://github.com/ConvolutedDog/mlir-inc-previewer'));
            }
          });
    }
  }
}
