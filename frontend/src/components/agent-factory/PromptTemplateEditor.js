/**
 * CodeMirror-based prompt template editor for Agent Factory.
 * - Syntax highlighting for {var} and {{#var}}...{{/var}}
 * - Autocomplete on { for runtime variables, step.field, playbook inputs
 * - LaTeX-style conditional block: {{# triggers completion that inserts {{#key}}\n  \n{{/key}} with cursor in the middle
 */
import React, { useMemo, useCallback, useState, useRef, useEffect } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView, Decoration, ViewPlugin, keymap } from '@codemirror/view';
import { history, defaultKeymap } from '@codemirror/commands';
import { autocompletion, completionKeymap, snippet } from '@codemirror/autocomplete';
import { Box } from '@mui/material';
import { useTheme } from '../../contexts/ThemeContext';
import { PROMPT_VARIABLES, PROMPT_VARIABLE_GROUPS, DYNAMIC_VARIABLE_PATTERNS } from '../../utils/promptVariableManifest';
import { getOutputFieldsForStep } from '../../utils/agentFactoryTypeWiring';

const PROMPT_EDITOR_HEIGHT_KEY = 'agent-factory-prompt-editor-height';
const DEFAULT_HEIGHT_PX = 220;
const MIN_HEIGHT_PX = 120;
const MAX_HEIGHT_PX = 480;

function getStepsWithOutputs(step) {
  if (!step) return [];
  if (step.step_type === 'parallel' && Array.isArray(step.parallel_steps)) {
    return step.parallel_steps.map((s) => ({ step: s }));
  }
  if (step.step_type === 'branch') {
    const then = (step.then_steps || []).map((s) => ({ step: s }));
    const el = (step.else_steps || []).map((s) => ({ step: s }));
    return [...then, ...el];
  }
  return [{ step }];
}

const promptVarDeco = Decoration.mark({ class: 'cm-promptVar' });
const promptConditionalDeco = Decoration.mark({ class: 'cm-promptConditional' });

const promptHighlightPlugin = ViewPlugin.fromClass(
  class {
    constructor(view) {
      this.decorations = this.buildDecorations(view);
    }
    update(update) {
      if (update.docChanged) this.decorations = this.buildDecorations(update.view);
    }
    buildDecorations(view) {
      const doc = view.state.doc;
      const decorations = [];
      const text = doc.toString();
      const singleRegex = /\{([^}]+)\}/g;
      let m;
      while ((m = singleRegex.exec(text)) !== null) {
        if (!m[0].startsWith('{{')) {
          decorations.push(promptVarDeco.range(m.index, m.index + m[0].length));
        }
      }
      const openRegex = /\{\{#([^}]+)\}\}/g;
      while ((m = openRegex.exec(text)) !== null) {
        const varName = m[1].trim();
        const closeTag = `{{/${varName}}}`;
        const closeIdx = text.indexOf(closeTag, m.index + m[0].length);
        if (closeIdx !== -1) {
          decorations.push(promptConditionalDeco.range(m.index, m.index + m[0].length));
          decorations.push(promptConditionalDeco.range(closeIdx, closeIdx + closeTag.length));
        } else {
          decorations.push(promptConditionalDeco.range(m.index, m.index + m[0].length));
        }
      }
      return Decoration.set(decorations, true);
    }
  },
  { decorations: (v) => v.decorations }
);

const promptTheme = EditorView.baseTheme({
  '.cm-promptVar': {
    color: 'var(--cm-prompt-var-color, #1976d2)',
    fontWeight: 500,
  },
  '.cm-promptConditional': {
    color: 'var(--cm-prompt-conditional-color, #2e7d32)',
    fontWeight: 500,
  },
});

function createPromptEditorTheme(darkMode) {
  const bg = darkMode ? '#1e1e1e' : '#ffffff';
  const fg = darkMode ? '#d4d4d4' : '#212121';
  const promptVarColor = darkMode ? '#82b1ff' : '#1976d2';
  const promptConditionalColor = darkMode ? '#81c784' : '#2e7d32';
  const selectionBg = darkMode ? '#264f78' : '#b3d7ff';
  const cursorColor = darkMode ? '#ffffff' : '#000000';
  const tooltipBg = darkMode ? '#2d2d2d' : '#ffffff';
  const tooltipFg = darkMode ? '#d4d4d4' : '#212121';
  const tooltipBorder = darkMode ? '#404040' : '#e0e0e0';
  const tooltipSelectedBg = darkMode ? '#264f78' : '#e3f2fd';
  const tooltipSelectedFg = darkMode ? '#ffffff' : '#1976d2';
  return EditorView.baseTheme({
    '&': { backgroundColor: bg, color: fg },
    '.cm-editor': { backgroundColor: bg, color: fg },
    '.cm-scroller': { backgroundColor: bg, color: fg },
    '.cm-content': {
      fontFamily: 'monospace',
      fontSize: '14px',
      lineHeight: '1.5',
      wordBreak: 'break-word',
      overflowWrap: 'anywhere',
      backgroundColor: bg,
      color: fg,
    },
    '&.cm-focused .cm-content': { backgroundColor: bg },
    '.cm-selectionBackground, ::selection': { backgroundColor: selectionBg },
    '.cm-cursor': { borderLeftColor: cursorColor },
    '.cm-line': { caretColor: cursorColor },
    '.cm-promptVar': { color: promptVarColor, fontWeight: 500 },
    '.cm-promptConditional': { color: promptConditionalColor, fontWeight: 500 },
    '.cm-tooltip': {
      backgroundColor: tooltipBg,
      color: tooltipFg,
      border: `1px solid ${tooltipBorder}`,
      borderRadius: 4,
    },
    '.cm-tooltip-autocomplete': {
      backgroundColor: tooltipBg,
      '& > ul': {
        backgroundColor: tooltipBg,
        color: tooltipFg,
      },
      '& > ul > li': {
        color: tooltipFg,
      },
      '& > ul > li[aria-selected]': {
        backgroundColor: tooltipSelectedBg,
        color: tooltipSelectedFg,
      },
    },
    '.cm-tooltip .cm-completionLabel': { color: tooltipFg },
    '.cm-tooltip .cm-completionDetail': { color: darkMode ? '#9e9e9e' : '#757575' },
  });
}

function createPromptAutocomplete(upstreamSteps = [], playbookInputs = [], actionsByName = {}) {
  return autocompletion({
    override: [
      (context) => {
        const line = context.state.doc.lineAt(context.pos);
        const lineStart = line.from;
        const before = context.state.sliceDoc(lineStart, context.pos);
        const matchConditional = before.match(/\{\{#([^}]*)$/);
        if (matchConditional) {
          const prefix = matchConditional[1].toLowerCase();
          const from = context.pos - matchConditional[0].length;
          const options = PROMPT_VARIABLES.filter(
            (v) => v.key.toLowerCase().startsWith(prefix) || prefix === ''
          ).map((variable) => ({
            label: `{{#${variable.key}}}...{{/${variable.key}}}`,
            type: 'conditional',
            detail: PROMPT_VARIABLE_GROUPS[variable.group] || variable.group,
            info: variable.description,
            apply: snippet('{{#' + variable.key + '}}\n  $0\n{{/' + variable.key + '}}'),
          }));
          if (options.length) return { from, options };
          return null;
        }
        const matchVar = before.match(/\{([^}]*)$/);
        if (!matchVar) return null;
        const varPrefix = matchVar[1].toLowerCase();
        const from = context.pos - matchVar[0].length;
        const options = [];
        const seen = new Set();
        const add = (value, label, detail, info) => {
          const key = value + label;
          if (seen.has(key)) return;
          seen.add(key);
          const matchLabel = (label || value).toLowerCase();
          if (varPrefix && !matchLabel.startsWith(varPrefix) && !value.toLowerCase().startsWith(varPrefix)) return;
          options.push({ label: label || value, type: 'text', detail: detail || '', info, apply: value });
        };
        for (const v of PROMPT_VARIABLES) {
          const value = `{${v.key}}`;
          add(value, v.key, PROMPT_VARIABLE_GROUPS[v.group] || v.group, v.description);
        }
        for (const p of DYNAMIC_VARIABLE_PATTERNS) {
          const value = `{${p.key}}`;
          add(value, p.key, p.label, p.description);
        }
        if (upstreamSteps.length && actionsByName && typeof actionsByName === 'object') {
          for (const s of upstreamSteps) {
            for (const { step } of getStepsWithOutputs(s)) {
              const key = step.output_key || step.name || step.action || '';
              if (!key) continue;
              const fields = getOutputFieldsForStep(step, actionsByName);
              for (const f of fields) {
                const value = `{${key}.${f.name}}`;
                add(value, `${key}.${f.name}`, 'Upstream step', null);
              }
            }
          }
        }
        if (Array.isArray(playbookInputs) && playbookInputs.length) {
          for (const p of playbookInputs) {
            const name = typeof p === 'object' && p.name != null ? p.name : String(p);
            const value = `{${name}}`;
            add(value, name, 'Playbook input', null);
          }
        }
        if (options.length) return { from, options };
        return null;
      },
    ],
  });
}

export default function PromptTemplateEditor({
  value = '',
  onChange,
  label = 'Prompt template',
  minLines = 3,
  upstreamSteps = [],
  playbookInputs = [],
  actionsByName = {},
  placeholder = 'Use {step_name.field} for upstream values. Type { for variables.',
}) {
  const { darkMode } = useTheme();
  const minHeightPx = Math.max(MIN_HEIGHT_PX, Math.round(minLines * 1.5 * 16));
  const [editorHeight, setEditorHeight] = useState(() => {
    try {
      const saved = localStorage.getItem(PROMPT_EDITOR_HEIGHT_KEY);
      if (saved != null) {
        const n = parseInt(saved, 10);
        if (!Number.isNaN(n) && n >= minHeightPx && n <= MAX_HEIGHT_PX) return n;
      }
    } catch (_) {}
    return DEFAULT_HEIGHT_PX;
  });
  const heightRef = useRef(editorHeight);
  useEffect(() => {
    heightRef.current = editorHeight;
  }, [editorHeight]);
  useEffect(() => {
    if (editorHeight < minHeightPx) setEditorHeight(minHeightPx);
  }, [minHeightPx, editorHeight]);

  const extensions = useMemo(() => {
    return [
      history(),
      keymap.of([...defaultKeymap, ...completionKeymap]),
      promptHighlightPlugin,
      createPromptEditorTheme(darkMode),
      createPromptAutocomplete(upstreamSteps, playbookInputs, actionsByName),
      EditorView.lineWrapping,
    ];
  }, [darkMode, upstreamSteps, playbookInputs, actionsByName]);

  const handleChange = useCallback(
    (val) => {
      if (typeof onChange === 'function') onChange(val);
    },
    [onChange]
  );

  const handleResizeStart = useCallback(
    (e) => {
      e.preventDefault();
      const startY = e.clientY ?? e.touches?.[0]?.clientY ?? 0;
      const startH = heightRef.current;
      const onMove = (ev) => {
        const y = ev.clientY ?? ev.touches?.[0]?.clientY;
        if (y == null) return;
        const delta = y - startY;
        const next = Math.round(Math.min(MAX_HEIGHT_PX, Math.max(minHeightPx, startH + delta)));
        setEditorHeight(next);
        try {
          localStorage.setItem(PROMPT_EDITOR_HEIGHT_KEY, String(next));
        } catch (_) {}
      };
      const onEnd = () => {
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onEnd);
        window.removeEventListener('touchmove', onMove, { passive: true });
        window.removeEventListener('touchend', onEnd);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onEnd);
      window.addEventListener('touchmove', onMove, { passive: true });
      window.addEventListener('touchend', onEnd);
    },
    [minHeightPx]
  );

  return (
    <Box sx={{ mb: 2, position: 'relative' }}>
      <CodeMirror
        value={value}
        height={`${editorHeight}px`}
        minHeight={`${minHeightPx}px`}
        onChange={handleChange}
        extensions={extensions}
        placeholder={placeholder}
        basicSetup={false}
        data-label={label}
        style={{ border: '1px solid', borderColor: 'divider', borderRadius: 1 }}
      />
      <Box
        role="separator"
        aria-label="Resize prompt editor"
        onMouseDown={handleResizeStart}
        onTouchStart={handleResizeStart}
        sx={{
          position: 'absolute',
          left: 0,
          right: 0,
          bottom: 0,
          height: 12,
          cursor: 'row-resize',
          zIndex: 5,
          touchAction: 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          '&:hover': { bgcolor: 'action.hover' },
          '&::after': {
            content: '""',
            width: 32,
            height: 3,
            borderRadius: 1.5,
            bgcolor: 'divider',
            opacity: 0.8,
          },
        }}
      />
    </Box>
  );
}
