// Copyright (c) 2026 Jianchao Yang
// Licensed under the MIT License - see the LICENSE file for details.

/**
 * Manages macro definitions and undefinitions
 */
export class MacroStateManager {
  private definedMacros = new Set<string>();
  private macroStack: Array<{
    macroName: string; conditionType: 'ifdef' | 'ifndef' | 'if' | 'if!';
    isActive: boolean;
    lineNumber: number;
  }> = [];

  /**
   * Process a line of code and update macro state
   */
  public processLine(line: string, lineNumber: number): void {
    const trimmed = line.trim();

    // #define
    const defineMatch = trimmed.match(/^#define\s+(\w+)/);
    if (defineMatch) {
      this.definedMacros.add(defineMatch[1]);
      return;
    }

    // #undef
    const undefMatch = trimmed.match(/^#undef\s+(\w+)/);
    if (undefMatch) {
      if (this.definedMacros.has(undefMatch[1]))
        this.definedMacros.delete(undefMatch[1]);
      return;
    }

    // #ifdef
    const ifdefMatch = trimmed.match(/^#ifdef\s+(\w+)/);
    if (ifdefMatch) {
      const macroName = ifdefMatch[1];
      const isActive = this.definedMacros.has(macroName);
      this.macroStack.push(
          {macroName, conditionType: 'ifdef', isActive, lineNumber});
      return;
    }

    // #endif
    if (trimmed.startsWith('#endif')) {
      if (this.macroStack.length > 0) {
        this.macroStack.pop();
      }
      return;
    }

    // #ifndef
    const ifndefMatch = trimmed.match(/^#ifndef\s+(\w+)/);
    if (ifndefMatch) {
      const macroName = ifndefMatch[1];
      const isActive = !this.definedMacros.has(macroName);
      this.macroStack.push(
          {macroName, conditionType: 'ifndef', isActive, lineNumber});
      return;
    }

    // #if defined(X)
    const ifDefinedMatch = trimmed.match(/^#if\s+defined\s*\(\s*(\w+)\s*\)/);
    if (ifDefinedMatch) {
      const macroName = ifDefinedMatch[1];
      const isActive = this.definedMacros.has(macroName);
      this.macroStack.push(
          {macroName, conditionType: 'if', isActive, lineNumber});
      return;
    }

    // #if !defined(X)
    const ifNotDefinedMatch =
        trimmed.match(/^#if\s*!\s*defined\s*\(\s*(\w+)\s*\)/);
    if (ifNotDefinedMatch) {
      const macroName = ifNotDefinedMatch[1];
      const isActive = !this.definedMacros.has(macroName);
      this.macroStack.push(
          {macroName, conditionType: 'if!', isActive, lineNumber});
      return;
    }

    // #else - Reverse the last block's status
    if (trimmed.startsWith('#else')) {
      if (this.macroStack.length > 0) {
        const lastBlock = this.macroStack[this.macroStack.length - 1];
        lastBlock.isActive = !lastBlock.isActive;
      }
      return;
    }

    // #elif
    if (trimmed.startsWith('#elif')) {
      if (this.macroStack.length > 0) {
        this.macroStack[this.macroStack.length - 1].isActive = false;
      }
      return;
    }
  }

  /**
   * Check if the current line is within an active block
   */
  public isCurrentLineActive(): boolean {
    // If there are no condition blocks or all blocks are active
    return this.macroStack.length === 0 ||
        this.macroStack.every(block => block.isActive);
  }

  /**
   * Get current macro state information (for debugging purposes)
   */
  public getStatusInfo(): string {
    const lines: string[] = [];

    lines.push(`Defined macros: ${
        Array.from(this.definedMacros).join(', ') || '(none)'}`);
    lines.push(`Active blocks: ${this.isCurrentLineActive() ? 'Yes' : 'No'}`);

    if (this.macroStack.length > 0) {
      lines.push('Condition stack:');
      this.macroStack.forEach((block, index) => {
        lines.push(
            `  ${index + 1}. ${block.conditionType} ${block.macroName} (line ${
                block.lineNumber +
                1}): ${block.isActive ? 'Active' : 'Inactive'}`);
      });
    }

    return lines.join('\n');
  }

  /**
   * Get defined macros
   */
  public getDefinedMacros(): Set<string> {
    return this.definedMacros;
  }
}
