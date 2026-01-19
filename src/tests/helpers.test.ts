import * as vscode from 'vscode';

import {detectPreviewBlocksFromDoc, findAllCommentedIncludeLines, findAllIncIncludeLine, findAllPreviewBlocks, findCommentedIncIncludeLine, findIncIncludeLine, isCommentedIncIncludeLine, isIncIncludeLine, showHelpHelper} from '../helpers';
import {BEGIN_COMMENT_LINE, END_COMMENT_LINE} from '../types';

const createMockDocument = (lines: string[]): vscode.TextDocument => {
  return {
    lineCount: lines.length,
    lineAt: (line: number) => ({
      text: lines[line],
      range: new vscode.Range(line, 0, line, lines[line].length),
      rangeIncludingLineBreak: new vscode.Range(line, 0, line + 1, 0),
      firstNonWhitespaceCharacterIndex: 0,
      isEmptyOrWhitespace: lines[line].trim().length === 0
    }),
    getText: () => lines.join('\n'),
    uri: vscode.Uri.file('test.cpp'),
    fileName: 'test.cpp',
    isUntitled: false,
    languageId: 'cpp',
    version: 1,
    isDirty: false,
    isClosed: false,
    save: jest.fn(),
    eol: vscode.EndOfLine.LF,
    offsetAt: jest.fn(),
    positionAt: jest.fn()
  } as any;
};

describe('Utils Functions', () => {
  describe('detectPreviewBlocksFromDoc', () => {
    const testCases = [
      {
        name: 'should return 0 for document without preview blocks',
        lines: ['#include "test.h"', 'int main() {', '  return 0;', '}'],
        expected: 0
      },
      {
        name: 'should detect single preview block',
        lines: [
          '#include "test.inc"', '/// --- [MLIR_INC_PREVIEW_START] ---',
          'void foo() {}', '/// --- [MLIR_INC_PREVIEW_END] ---',
          'int main() { return 0; }'
        ],
        expected: 1
      },
      {
        name: 'should detect multiple preview blocks',
        lines: [
          '/// --- [MLIR_INC_PREVIEW_START] ---', 'block 1',
          '/// --- [MLIR_INC_PREVIEW_END] ---', 'regular code',
          '/// --- [MLIR_INC_PREVIEW_START] ---', 'block 2',
          '/// --- [MLIR_INC_PREVIEW_END] ---'
        ],
        expected: 2
      },
      {
        name: 'should handle nested tags (treat as separate blocks)',
        lines: [
          '/// --- [MLIR_INC_PREVIEW_START] ---',
          '/// --- [MLIR_INC_PREVIEW_START] ---', 'inner',
          '/// --- [MLIR_INC_PREVIEW_END] ---', 'outer',
          '/// --- [MLIR_INC_PREVIEW_END] ---'
        ],
        expected: 2
      }
    ];

    test.each(testCases)('$name', ({name, lines, expected}) => {
      const doc = createMockDocument(lines);
      const result = detectPreviewBlocksFromDoc(doc);
      console.log(`\n=== Test: ${name} ===`);
      console.log(`Lines: ${JSON.stringify(lines)}`);
      console.log(`Expected: ${expected}, Got: ${result}`);
      expect(result).toBe(expected);
    });
  });

  describe('isIncIncludeLine', () => {
    const testCases = [
      {
        name: 'should return true for #include with .inc file',
        line: '#include "test.inc"',
        expected: true
      },
      {
        name: 'should return true for #include with .inc in angle brackets',
        line: '#include <path/test.inc>',
        expected: true
      },
      {
        name: 'should return true for #include with .INC (case insensitive)',
        line: '#include "test.INC"',
        expected: true
      },
      {
        name: 'should return false for #include without .inc',
        line: '#include "test.h"',
        expected: false
      },
      {
        name: 'should return false for non-#include lines',
        line: 'int x = 0;',
        expected: false
      },
      {
        name: 'should handle #include with inline comment',
        line: '#include "test.inc"  // This is a comment',
        expected: true
      },
      {
        name: 'should handle .inc in path',
        line: '#include "path/to/my.inc/file.h"',
        expected: true
      },
      {
        name: 'should return false for #define',
        line: '#define DEBUG 1',
        expected: false
      },
      {
        name: 'should return true for #include with spaces',
        line: '  #include  "test.inc"  ',
        expected: true
      }
    ];

    test.each(testCases)('$name', ({name, line, expected}) => {
      const result = isIncIncludeLine(line);
      console.log(`\n=== Test: ${name} ===`);
      console.log(`Line: "${line}"`);
      console.log(`Expected: ${expected}, Got: ${result}`);
      expect(result).toBe(expected);
    });
  });

  describe('isCommentedIncIncludeLine', () => {
    const testCases = [
      {
        name: 'should return true for commented #include .inc',
        line: `${BEGIN_COMMENT_LINE}#include "test.inc"${END_COMMENT_LINE}`,
        expected: true
      },
      {
        name:
            'should return true for commented #include .inc with extra spaces',
        line: `${BEGIN_COMMENT_LINE}#include "test.inc"${END_COMMENT_LINE}`,
        expected: true
      },
      {
        name: 'should return true for commented #include with angle brackets',
        line:
            `${BEGIN_COMMENT_LINE}#include <path/test.inc>${END_COMMENT_LINE}`,
        expected: true
      },
      {
        name: 'should return false for uncommented #include .inc',
        line: '#include "test.inc"',
        expected: false
      },
      {
        name: 'should return false for commented non-#include line',
        line: `${BEGIN_COMMENT_LINE}int x = 0;${END_COMMENT_LINE}`,
        expected: false
      },
      {
        name: 'should return false for commented #include without .inc',
        line: `${BEGIN_COMMENT_LINE}#include "test.h"${END_COMMENT_LINE}`,
        expected: false
      },
      {
        name: 'should return false for line missing end comment tag',
        line: `${BEGIN_COMMENT_LINE}#include "test.inc"`,
        expected: false
      },
      {
        name: 'should return false for line missing begin comment tag',
        line: `#include "test.inc"${END_COMMENT_LINE}`,
        expected: false
      },
      {
        name: 'should return false for line with wrong comment tags',
        line: `// #include "test.inc"`,
        expected: false
      },
      {
        name: 'should handle .inc in path',
        line: `${BEGIN_COMMENT_LINE}#include "path/to/my.inc/file.h" ${
            END_COMMENT_LINE}`,
        expected: true
      }
    ];

    test.each(testCases)('$name', ({name, line, expected}) => {
      const result = isCommentedIncIncludeLine(line);
      console.log(`\n=== Test: ${name} ===`);
      console.log(`Line: "${line}"`);
      console.log(`Expected: ${expected}, Got: ${result}`);
      expect(result).toBe(expected);
    });
  });

  describe('findIncIncludeLine', () => {
    const testCases = [
      {
        name: 'should find #include on current line',
        lines: ['int x = 0;', '#include "test.inc"', 'int y = 1;'],
        currentLine: 1,
        expected: 1
      },
      {
        name: 'should find #include above cursor',
        lines: ['#include "test.inc"', 'int x = 0;', 'int y = 1;'],
        currentLine: 2,
        expected: 0
      },
      {
        name: 'should find #include below cursor',
        lines: ['int x = 0;', 'int y = 1;', '#include "test.inc"'],
        currentLine: 0,
        expected: 2
      },
      {
        name: 'should return -1 if no #include found',
        lines: ['int x = 0;', 'int y = 1;', 'int z = 2;'],
        currentLine: 1,
        expected: -1
      },
      {
        name: 'should prefer current line over nearby lines',
        lines: [
          '#include "test1.inc"', '#include "test2.inc"', '#include "test3.inc"'
        ],
        currentLine: 1,
        expected: 1
      }
    ];

    test.each(testCases)('$name', ({name, lines, currentLine, expected}) => {
      const doc = createMockDocument(lines);
      const result = findIncIncludeLine(doc, currentLine);
      console.log(`\n=== Test: ${name} ===`);
      console.log(
          `Current line: ${currentLine}, Lines: ${JSON.stringify(lines)}`);
      console.log(`Expected: ${expected}, Got: ${result}`);
      expect(result).toBe(expected);
    });
  });

  describe('findCommentedIncIncludeLine', () => {
    const testCases = [
      {
        name: 'should find commented #include on current line',
        lines: [
          'int x = 0;',
          `${BEGIN_COMMENT_LINE}#include "test.inc"${END_COMMENT_LINE}`,
          'int y = 1;'
        ],
        currentLine: 1,
        expected: 1
      },
      {
        name: 'should find commented #include above cursor',
        lines: [
          `${BEGIN_COMMENT_LINE}#include "test.inc"${END_COMMENT_LINE}`,
          'int x = 0;', 'int y = 1;'
        ],
        currentLine: 2,
        expected: 0
      },
      {
        name: 'should find commented #include below cursor',
        lines: [
          'int x = 0;', 'int y = 1;',
          `${BEGIN_COMMENT_LINE}#include "test.inc"${END_COMMENT_LINE}`
        ],
        currentLine: 0,
        expected: 2
      },
      {
        name: 'should return -1 if no commented #include found',
        lines: [
          'int x = 0;',
          '#include "test.inc"',  // Not commented
          'int y = 1;'
        ],
        currentLine: 1,
        expected: -1
      },
      {
        name: 'should handle mixed commented and uncommented includes',
        lines: [
          `${BEGIN_COMMENT_LINE}#include "test1.inc"${END_COMMENT_LINE}`,
          '#include "test2.inc"',  // Not commented
          `${BEGIN_COMMENT_LINE}#include "test3.inc"${END_COMMENT_LINE}`
        ],
        currentLine: 1,
        expected: 0  // Search upwards first
      },
      {
        name: 'should find commented #include within preview block',
        lines: [
          '/// --- [MLIR_INC_PREVIEW_START] ---',
          `${BEGIN_COMMENT_LINE}#include "test.inc"${END_COMMENT_LINE}`,
          'expanded content', '/// --- [MLIR_INC_PREVIEW_END] ---',
          'regular code'
        ],
        currentLine: 2,
        expected: 1
      },
      {
        name:
            'should find commented #include outside preview block when inside block',
        lines: [
          `${BEGIN_COMMENT_LINE}#include "test1.inc"${END_COMMENT_LINE}`,
          '/// --- [MLIR_INC_PREVIEW_START] ---', 'expanded content',
          '/// --- [MLIR_INC_PREVIEW_END] ---'
        ],
        currentLine: 2,
        expected: 0
      }
    ];

    test.each(testCases)('$name', ({name, lines, currentLine, expected}) => {
      const doc = createMockDocument(lines);
      const result = findCommentedIncIncludeLine(doc, currentLine);
      console.log(`\n=== Test: ${name} ===`);
      console.log(
          `Current line: ${currentLine}, Lines: ${JSON.stringify(lines)}`);
      console.log(`Expected: ${expected}, Got: ${result}`);
      expect(result).toBe(expected);
    });
  });

  describe('findAllIncIncludeLine', () => {
    const testCases = [
      {
        name: 'should find all #include .inc lines',
        lines: [
          '#include "test1.inc"', 'int x = 0;', '#include "test2.inc"',
          '#include "test.h"', '#include <path/test3.inc>'
        ],
        expected: [0, 2, 4]
      },
      {
        name: 'should return empty array for no .inc includes',
        lines: ['#include "test.h"', 'int main() {}'],
        expected: []
      },
      {name: 'should handle empty document', lines: [], expected: []}
    ];

    test.each(testCases)('$name', ({name, lines, expected}) => {
      const doc = createMockDocument(lines);
      const result = findAllIncIncludeLine(doc);
      console.log(`\n=== Test: ${name} ===`);
      console.log(`Lines: ${JSON.stringify(lines)}`);
      console.log(`Expected: ${JSON.stringify(expected)}, Got: ${
          JSON.stringify(result)}`);
      expect(result).toEqual(expected);
    });
  });

  describe('findAllPreviewBlocks', () => {
    test('should find all preview block ranges', () => {
      const lines = [
        'regular code', '/// --- [MLIR_INC_PREVIEW_START] ---',
        'preview content 1', '/// --- [MLIR_INC_PREVIEW_END] ---', 'more code',
        '/// --- [MLIR_INC_PREVIEW_START] ---', 'preview content 2',
        '/// --- [MLIR_INC_PREVIEW_END] ---', 'end'
      ];

      const doc = createMockDocument(lines);
      const result = findAllPreviewBlocks(doc);

      console.log('\n=== Test: should find all preview block ranges ===');
      console.log(`Lines: ${JSON.stringify(lines)}`);
      console.log(`Found ${result.length} blocks`);

      expect(result).toHaveLength(2);
    });

    test('should handle document without preview blocks', () => {
      const lines = ['regular code', 'no preview blocks here'];

      const doc = createMockDocument(lines);
      const result = findAllPreviewBlocks(doc);

      expect(result).toHaveLength(0);
    });
  });

  describe('findAllCommentedIncludeLines', () => {
    const testCases = [
      {
        name: 'should find all commented include lines',
        lines: [
          'regular code',
          `${BEGIN_COMMENT_LINE}#include "test1.inc"${END_COMMENT_LINE}`,
          `${BEGIN_COMMENT_LINE}#include "test2.inc"${END_COMMENT_LINE}`,
          `${BEGIN_COMMENT_LINE}#include "test.h"${END_COMMENT_LINE}`,
          '#include "test3.inc"',  // Not commented
          'end'
        ],
        expected: [1, 2]
      },
      {
        name: 'should return empty array for no commented includes',
        lines: [
          '#include "test1.inc"',  // Not commented
          'int x = 0;', '// This is a regular comment'
        ],
        expected: []
      },
      {
        name: 'should handle mixed includes',
        lines: [
          '#include "test1.inc"',
          `${BEGIN_COMMENT_LINE}#include "test2.inc"${END_COMMENT_LINE}`,
          '#include "test.h"',
          `${BEGIN_COMMENT_LINE}#include <path/test3.inc>${END_COMMENT_LINE}`
        ],
        expected: [1, 3]
      },
      {
        name: 'should handle lines with extra spaces',
        lines: [
          `${BEGIN_COMMENT_LINE} #include  "test.inc"  ${END_COMMENT_LINE}`,
          `${BEGIN_COMMENT_LINE}#include "test2.inc"${END_COMMENT_LINE}`
        ],
        expected: [0, 1]
      }
    ];

    test.each(testCases)('$name', ({name, lines, expected}) => {
      const doc = createMockDocument(lines);
      const result = findAllCommentedIncludeLines(doc);
      console.log(`\n=== Test: ${name} ===`);
      console.log(`Lines: ${JSON.stringify(lines)}`);
      console.log(`Expected: ${JSON.stringify(expected)}, Got: ${
          JSON.stringify(result)}`);
      expect(result).toHaveLength(expected.length);
    });
  });

  describe('showHelpHelper', () => {
    test('showHelp should not throw', async () => {
      // Simulate vscode.extensions.getExtension
      const mockExtension = {extensionUri: vscode.Uri.file('/mock/path')};

      const originalGetExtension = vscode.extensions.getExtension;
      (vscode.extensions as any).getExtension =
          jest.fn().mockReturnValue(mockExtension);

      try {
        // showHelp may attempt to open a file, so we just ensure it doesn't
        // throw
        await expect(showHelpHelper.showHelp()).resolves.not.toThrow();
      } finally {
        // Restore the original function
        (vscode.extensions as any).getExtension = originalGetExtension;
      }
    });
  });
});
