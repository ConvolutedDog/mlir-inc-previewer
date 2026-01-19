export const window = {
  createOutputChannel: jest.fn(),
  showInformationMessage: jest.fn(),
  showErrorMessage: jest.fn(),
  showWarningMessage: jest.fn(),
  setStatusBarMessage: jest.fn(),
  showTextDocument: jest.fn(),
  activeTextEditor: null,
  visibleTextEditors: []
};

export const workspace = {
  openTextDocument: jest.fn(),
  getConfiguration: jest.fn(),
  onDidChangeTextDocument: jest.fn()
};

export const commands = {
  executeCommand: jest.fn()
};

export const extensions = {
  getExtension: jest.fn()
};

export const env = {
  openExternal: jest.fn()
};

export const Uri = {
  file: (path: string) => ({path, scheme: 'file', fsPath: path}),
  joinPath: (base: any, ...pathSegments: string[]) => ({
    path: base.path + '/' + pathSegments.join('/'),
    fsPath: base.fsPath + '/' + pathSegments.join('/')
  }),
  parse: jest.fn()
};

export const Range = class {
  constructor(public start: any, public end: any) {}
};

export const Position = class {
  constructor(public line: number, public character: number) {}
};

export const EndOfLine = {
  LF: 1,
  CRLF: 2
};

export const ViewColumn = {
  One: 1,
  Two: 2,
  Three: 3
};

export default {
  window,
  workspace,
  commands,
  extensions,
  env,
  Uri,
  Range,
  Position,
  EndOfLine,
  ViewColumn
};
