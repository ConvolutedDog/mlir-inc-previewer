import {MacroStateManager} from '../macrosManager';

describe('MacroStateManager', () => {
  describe('Basic functionality', () => {
    const testCases = [
      {
        name: 'should start with no macros defined',
        setup: (manager: MacroStateManager) => {},
        expectedActive: true,
      },
      {
        name: 'should handle #define',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define DEBUG', 0);
        },
        expectedActive: true,
      },
      {
        name: 'should handle #undef',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define DEBUG', 0);
          manager.processLine('#undef DEBUG', 1);
        },
        expectedActive: true,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('#ifdef conditions', () => {
    const testCases = [
      {
        name: 'should be active when macro is defined',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define DEBUG', 0);
          manager.processLine('#ifdef DEBUG', 1);
        },
        expectedActive: true,
      },
      {
        name: 'should be inactive when macro is not defined',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#ifdef DEBUG', 0);
        },
        expectedActive: false,
      },
      {
        name: 'should handle code inside #ifdef block',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#ifdef DEBUG', 0);
          manager.processLine('int x = 1;', 1);
        },
        expectedActive: false,
      },
      {
        name: 'should handle code inside #ifdef block',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define DEBUG', 0);
          manager.processLine('#ifdef DEBUG', 1);
          manager.processLine('int x = 1;', 2);
        },
        expectedActive: true,
      },
      {
        name: 'should be active after #endif',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define DEBUG', 0);
          manager.processLine('#ifdef DEBUG', 1);
          manager.processLine('#endif', 2);
        },
        expectedActive: true,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('#ifndef conditions', () => {
    const testCases = [
      {
        name: 'should be active when macro is not defined',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#ifndef RELEASE', 0);
        },
        expectedActive: true,
      },
      {
        name: 'should be inactive when macro is defined',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define RELEASE', 0);
          manager.processLine('#ifndef RELEASE', 1);
        },
        expectedActive: false,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('#if defined() syntax', () => {
    const testCases = [
      {
        name: 'should handle #if defined(MACRO)',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define OPTION_A', 0);
          manager.processLine('#if defined(OPTION_A)', 1);
        },
        expectedActive: true,
      },
      {
        name: 'should handle #if !defined(MACRO) - undefined',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#if !defined(OPTION_A)', 0);
        },
        expectedActive: true,
      },
      {
        name: 'should handle #if !defined(MACRO) - defined',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define OPTION_A', 0);
          manager.processLine('#if !defined(OPTION_A)', 1);
        },
        expectedActive: false,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('Nested conditions', () => {
    const testCases = [
      {
        name: 'should handle simple nesting',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define FEATURE_A', 0);
          manager.processLine('#define FEATURE_B', 1);
          manager.processLine('#ifdef FEATURE_A', 2);
          manager.processLine('#ifdef FEATURE_B', 3);
          manager.processLine('#endif', 4);
          manager.processLine('#endif', 5);
        },
        expectedActive: true,
      },
      {
        name: 'should handle mixed conditions',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define A', 0);
          manager.processLine('#if defined(A)', 1);
          manager.processLine('#ifndef B', 2);
          manager.processLine('int x = 1;', 3);
          manager.processLine('#endif', 4);
          manager.processLine('#endif', 5);
        },
        expectedActive: true,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('#else branches', () => {
    const testCases = [
      {
        name: 'should handle #else when #ifdef is false',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define VERSION_2', 0);
          manager.processLine('#ifdef VERSION_1', 1);
          manager.processLine('#else', 2);
          manager.processLine('#endif', 3);
        },
        expectedActive: true,
      },
      {
        name: 'should handle #else when #ifdef is true',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define VERSION_1', 0);
          manager.processLine('#ifdef VERSION_1', 1);
          manager.processLine('#else', 2);
          manager.processLine('#endif', 3);
        },
        expectedActive: true,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('#elif handling', () => {
    const testCases = [
      {
        name: 'should handle #elif chain',
        setup: (manager: MacroStateManager) => {
          manager.processLine('#define LEVEL_2', 0);
          manager.processLine('#if defined(LEVEL_1)', 1);
          manager.processLine('#elif defined(LEVEL_2)', 2);
          manager.processLine('#else', 3);
          manager.processLine('#endif', 4);
        },
        expectedActive: true,
      },
    ];

    test.each(testCases)('$name', ({name, setup, expectedActive}) => {
      const manager = new MacroStateManager();
      setup(manager);
      console.log(`\n=== Test: ${name} ===\n` + manager.getStatusInfo());
      expect(manager.isCurrentLineActive()).toBe(expectedActive);
    });
  });

  describe('Code inclusion test', () => {
    test('should correctly include/exclude code based on macros', () => {
      const manager = new MacroStateManager();
      const code = `
#define DEBUG
#define FEATURE_X

#ifdef DEBUG
    log("Debug mode");
    #ifdef FEATURE_X
        enableFeatureX();
    #endif
#else
    log("Release mode");
#endif

#ifndef TEST_MODE
    initProduction();
#endif
`.split('\n');

      const includedLines: string[] = [];

      code.forEach((line, index) => {
        manager.processLine(line, index);
        if (manager.isCurrentLineActive() && line.trim() &&
            !line.trim().startsWith('#') && line.trim()) {
          includedLines.push(line.trim());
        }
      });

      console.log(
          `\n=== Test: Code inclusion test ===\n` + manager.getStatusInfo());
      expect(includedLines).toHaveLength(3);
      expect(includedLines[0]).toBe('log("Debug mode");');
      expect(includedLines[1]).toBe('enableFeatureX();');
      expect(includedLines[2]).toBe('initProduction();');
    });
  });
});
