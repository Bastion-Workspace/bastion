import { EditorView, keymap, lineNumbers } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import {
  defaultKeymap,
  history,
  historyKeymap,
  indentWithTab,
} from '@codemirror/commands';
import {
  bracketMatching,
  defaultHighlightStyle,
  foldGutter,
  indentOnInput,
  syntaxHighlighting,
} from '@codemirror/language';
import { markdown } from '@codemirror/lang-markdown';
import { javascript } from '@codemirror/lang-javascript';
import { json } from '@codemirror/lang-json';
import { python } from '@codemirror/lang-python';
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';

const codeSpaceBaseTheme = (dark) =>
  EditorView.baseTheme({
    '&': {
      fontSize: '13px',
      backgroundColor: dark ? '#1e1e1e' : '#fff',
    },
    '.cm-editor': {
      backgroundColor: dark ? '#1e1e1e' : '#fff',
    },
    '.cm-scroller': {
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
      backgroundColor: dark ? '#1e1e1e' : '#fff',
    },
    '.cm-content': {
      caretColor: dark ? '#e0e0e0' : '#111',
    },
    '.cm-gutters': {
      backgroundColor: dark ? '#252526' : '#f5f5f5',
      color: dark ? '#858585' : '#666',
      borderRight: `1px solid ${dark ? '#333' : '#e0e0e0'}`,
    },
    '.cm-activeLineGutter': {
      backgroundColor: dark ? '#2a2d2e' : '#e8e8e8',
    },
  });

/**
 * @param {string} filename — basename or path for extension detection
 * @param {boolean} darkMode
 * @returns {import('@codemirror/state').Extension[]}
 */
export function buildCodeSpaceEditorExtensions(filename, darkMode) {
  const lower = (filename || '').toLowerCase();
  const ext = lower.includes('.') ? lower.replace(/^.*\./, '') : '';

  /** @type {import('@codemirror/state').Extension[]} */
  const lang = [];
  switch (ext) {
    case 'js':
    case 'mjs':
    case 'cjs':
      lang.push(javascript({ jsx: false, typescript: false }));
      break;
    case 'jsx':
      lang.push(javascript({ jsx: true, typescript: false }));
      break;
    case 'ts':
      lang.push(javascript({ jsx: false, typescript: true }));
      break;
    case 'tsx':
      lang.push(javascript({ jsx: true, typescript: true }));
      break;
    case 'json':
    case 'jsonc':
      lang.push(json());
      break;
    case 'py':
    case 'pyw':
      lang.push(python());
      break;
    case 'css':
      lang.push(css());
      break;
    case 'html':
    case 'htm':
      lang.push(html());
      break;
    case 'md':
    case 'markdown':
      lang.push(markdown());
      break;
    default:
      break;
  }

  return [
    lineNumbers(),
    history(),
    foldGutter(),
    indentOnInput(),
    bracketMatching(),
    syntaxHighlighting(defaultHighlightStyle, { fallback: true }),
    keymap.of([indentWithTab, ...defaultKeymap, ...historyKeymap]),
    codeSpaceBaseTheme(darkMode),
    ...lang,
    EditorState.tabSize.of(2),
  ];
}
