import React, { useMemo, useEffect, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView, keymap, Decoration, ViewPlugin } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { markdown, markdownLanguage } from '@codemirror/lang-markdown';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { useEditor } from '../contexts/EditorContext';
import { useTheme } from '../contexts/ThemeContext';
import { parseFrontmatter as parseMarkdownFrontmatter } from '../utils/frontmatterUtils';
import { Box, TextField, Button, Tooltip, Drawer, IconButton, Typography, Stack, Switch, FormControlLabel } from '@mui/material';
import { Add, Delete } from '@mui/icons-material';
import { createGhostTextExtension } from './editor/extensions/ghostTextExtension';
import { createInlineEditSuggestionsExtension } from './editor/extensions/inlineEditSuggestionsExtension';
import { createLiveEditDiffExtension } from './editor/extensions/liveEditDiffExtension';
import { editorSuggestionService } from '../services/editor/EditorSuggestionService';

const createMdTheme = (darkMode) => EditorView.baseTheme({
  '.cm-content': { 
    fontFamily: 'monospace', 
    fontSize: '14px', 
    lineHeight: '1.5', 
    wordBreak: 'break-word', 
    overflowWrap: 'anywhere',
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121'
  },
  '.cm-gutters': {
    backgroundColor: darkMode ? '#1e1e1e' : '#f5f5f5',
    color: darkMode ? '#858585' : '#999999',
    border: 'none'
  },
  '.cm-activeLineGutter': {
    backgroundColor: darkMode ? '#2d2d2d' : '#e8f2ff'
  },
  '.cm-activeLine': {
    backgroundColor: darkMode ? '#2d2d2d' : '#f0f8ff'
  },
  '.cm-selectionBackground, ::selection': {
    backgroundColor: darkMode ? '#264f78' : '#b3d7ff'
  },
  '.cm-cursor': {
    borderLeftColor: darkMode ? '#ffffff' : '#000000'
  },
  '&.cm-focused .cm-selectionBackground, &.cm-focused ::selection': {
    backgroundColor: darkMode ? '#264f78' : '#b3d7ff'
  },
  '.cm-line.cm-fm-hidden': { display: 'none' },
  '.cm-line': {
    caretColor: darkMode ? '#ffffff' : '#000000'
  },
  // Markdown syntax highlighting adjustments for dark mode
  '.cm-content .cm-meta': {
    color: darkMode ? '#808080' : '#999999'
  },
  '.cm-content .cm-header': {
    color: darkMode ? '#e0e0e0' : '#000000',
    fontWeight: 'bold'
  },
  '.cm-content .cm-strong': {
    color: darkMode ? '#e0e0e0' : '#000000',
    fontWeight: 'bold'
  },
  '.cm-content .cm-emphasis': {
    color: darkMode ? '#e0e0e0' : '#000000',
    fontStyle: 'italic'
  },
  '.cm-content .cm-link': {
    color: darkMode ? '#90caf9' : '#1976d2'
  },
  '.cm-content .cm-url': {
    color: darkMode ? '#90caf9' : '#1976d2'
  }
});

function parseFrontmatter(text) {
  try {
    const trimmed = text.startsWith('\ufeff') ? text.slice(1) : text;
    if (!trimmed.startsWith('---\n')) return { data: {}, lists: {}, order: [], raw: '', body: text };
    const end = trimmed.indexOf('\n---', 4);
    if (end === -1) return { data: {}, lists: {}, order: [], raw: '', body: text };
    const yaml = trimmed.slice(4, end).replace(/\r/g, '');
    const body = trimmed.slice(end + 4).replace(/^\n/, '');
    const data = {};
    const lists = {};
    const order = [];
    const lines = yaml.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const m = line.match(/^([A-Za-z0-9_\-]+):\s*(.*)$/);
      if (m) {
        const k = m[1].trim();
        const v = m[2];
        order.push(k);
        if (v && v.trim().length > 0) {
          data[k] = String(v).trim();
        } else {
          const items = [];
          let j = i + 1;
          while (j < lines.length) {
            const ln = lines[j];
            if (/^\s*-\s+/.test(ln)) {
              items.push(ln.replace(/^\s*-\s+/, ''));
              j++;
            } else if (/^\s+$/.test(ln)) {
              j++;
            } else {
              break;
            }
          }
          if (items.length > 0) {
            lists[k] = items;
            i = j - 1;
          } else {
            data[k] = '';
          }
        }
      }
    }
    return { data, lists, order, raw: yaml, body };
  } catch (e) {
    return { data: {}, lists: {}, order: [], raw: '', body: text };
  }
}

function mergeFrontmatter(originalYaml, scalarUpdates, listUpdates, keyOrder) {
  const lines = (originalYaml || '').split('\n');
  const blocks = {};
  const order = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^([A-Za-z0-9_\-]+):\s*(.*)$/);
    if (m) {
      const k = m[1].trim();
      const block = [lines[i]];
      let j = i + 1;
      while (j < lines.length && !/^[A-Za-z0-9_\-]+:\s*/.test(lines[j])) {
        block.push(lines[j]);
        j++;
      }
      blocks[k] = block;
      order.push(k);
      i = j - 1;
    }
  }
  const nextOrder = [...new Set([...(keyOrder || []), ...order])];
  for (const [k, v] of Object.entries(scalarUpdates || {})) {
    const kv = String(v ?? '').trim();
    if (kv.length === 0) continue;
    blocks[k] = [`${k}: ${kv}`];
    if (!nextOrder.includes(k)) nextOrder.push(k);
  }
  for (const [k, arr] of Object.entries(listUpdates || {})) {
    const items = Array.isArray(arr) ? arr.filter(s => String(s).trim().length > 0) : [];
    if (items.length === 0) continue;
    const block = [`${k}:`];
    for (const it of items) block.push(`  - ${String(it)}`);
    blocks[k] = block;
    if (!nextOrder.includes(k)) nextOrder.push(k);
  }
  const rebuilt = [];
  for (const k of nextOrder) {
    if (blocks[k] && blocks[k].length) {
      rebuilt.push(...blocks[k]);
    }
  }
  return rebuilt.length ? `---\n${rebuilt.join('\n')}\n---\n` : '';
}

function buildFrontmatter(data) {
  const lines = Object.entries(data)
    .filter(([_, v]) => v !== undefined && v !== null && String(v).length > 0)
    .map(([k, v]) => `${k}: ${v}`);
  if (lines.length === 0) return '';
  return `---\n${lines.join('\n')}\n---\n`;
}

export default function MarkdownCMEditor({ value, onChange, filename, canonicalPath }) {
  const { darkMode } = useTheme();
  const [suggestionsEnabled, setSuggestionsEnabled] = useState(() => {
    try {
      const saved = localStorage.getItem('editorPredictiveSuggestionsEnabled');
      return saved !== null ? JSON.parse(saved) : false;
    } catch {
      return false;
    }
  });

  // Persist suggestions preference to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('editorPredictiveSuggestionsEnabled', JSON.stringify(suggestionsEnabled));
    } catch (error) {
      console.error('Failed to save predictive suggestions preference:', error);
    }
  }, [suggestionsEnabled]);

  // Removed floating Accept UI state; Tab or clicking ghost text suffices
  const ghostExt = useMemo(() => suggestionsEnabled ? createGhostTextExtension(async ({ prefix, suffix, position, signal, frontmatter, filename: fn }) => {
    try {
      const { suggestion } = await editorSuggestionService.suggest({
        prefix,
        suffix,
        filename: fn,
        language: 'markdown',
        cursorOffset: position,
        frontmatter,
        maxChars: 80,
        signal
      });
      return suggestion || '';
    } catch { return ''; }
  }, { debounceMs: 350 }) : [], [suggestionsEnabled]);
  const mdTheme = useMemo(() => createMdTheme(darkMode), [darkMode]);
  const inlineEditExt = useMemo(() => createInlineEditSuggestionsExtension(), []);
  const liveEditDiffExt = useMemo(() => {
    const ext = createLiveEditDiffExtension();
    console.log('🔍 MarkdownCMEditor: Created liveEditDiffExt:', ext?.length, 'items');
    // Test event listener to verify events are firing
    const testListener = (e) => {
      console.log('🔍 TEST: MarkdownCMEditor received editorOperationsLive event:', e.detail);
    };
    window.addEventListener('editorOperationsLive', testListener);
    // Clean up after 30 seconds
    setTimeout(() => {
      window.removeEventListener('editorOperationsLive', testListener);
    }, 30000);
    return ext;
  }, []);
  const extensions = useMemo(() => [
    history(),
    keymap.of([...defaultKeymap, ...historyKeymap]),
    markdown({ base: markdownLanguage }),
    EditorView.lineWrapping,
    mdTheme,
    inlineEditExt,
    liveEditDiffExt
  ], [mdTheme, inlineEditExt, liveEditDiffExt]);

  const { setEditorState } = useEditor();
  const [fmOpen, setFmOpen] = useState(false);
  const { data: initialData, lists: initialLists, raw: initialRaw, order: initialOrder, body: initialBody } = useMemo(() => {
    const parsed = parseFrontmatter((value || '').replace(/\r\n/g, '\n'));
    console.log('🎯 ROOSEVELT FRONTMATTER DEBUG:');
    console.log('  📄 Value (first 100 chars):', value?.slice(0, 100) + (value?.length > 100 ? '...' : ''));
    console.log('  📋 Parsed object:', parsed);
    console.log('  🏷️ Data keys:', Object.keys(parsed.data || {}));
    console.log('  🏷️ Data values:', parsed.data);
    console.log('  ✅ Has data:', Object.keys(parsed.data || {}).length > 0);
    return parsed;
  }, [value]);
  const baseTitle = useMemo(() => (filename ? String(filename).replace(/\.[^.]+$/, '') : ''), [filename]);
  const [fmEntries, setFmEntries] = useState(() => {
    const entries = Object.entries(initialData).map(([k, v]) => ({ key: k, value: String(v ?? '') }));
    // Ensure title exists for new files
    if (!entries.find(e => e.key === 'title') && baseTitle) {
      entries.unshift({ key: 'title', value: baseTitle });
    }
    console.log('🎯 ROOSEVELT FRONTMATTER ENTRIES INIT:');
    console.log('  🏷️ Initial data:', initialData);
    console.log('  📝 Entries array:', entries);
    console.log('  📄 Base title:', baseTitle);
    return entries;
  });
  const [fmListEntries, setFmListEntries] = useState(() => {
    const obj = {};
    Object.entries(initialLists || {}).forEach(([k, arr]) => {
      obj[k] = (arr || []).join('\n');
    });
    return obj;
  });
  const [fmRaw, setFmRaw] = useState(initialRaw || '');
  const [fmOrder, setFmOrder] = useState(initialOrder || []);

  useEffect(() => {
    const entries = Object.entries(initialData).map(([k, v]) => ({ key: k, value: String(v ?? '') }));
    if (!entries.find(e => e.key === 'title') && baseTitle) {
      entries.unshift({ key: 'title', value: baseTitle });
    }
    console.log('🎯 ROOSEVELT FRONTMATTER ENTRIES UPDATE:');
    console.log('  🏷️ Initial data:', initialData);
    console.log('  📝 Entries array:', entries);
    console.log('  📄 Base title:', baseTitle);
    console.log('  📋 Initial lists:', initialLists);
    console.log('  📄 Initial raw:', initialRaw);
    setFmEntries(entries);
    const obj = {};
    Object.entries(initialLists || {}).forEach(([k, arr]) => { obj[k] = (arr || []).join('\n'); });
    setFmListEntries(obj);
    setFmRaw(initialRaw || '');
    setFmOrder(initialOrder || []);
  }, [initialData, initialLists, initialRaw, initialOrder, baseTitle]);

  // Show frontmatter in the editor (no hiding)
  const frontmatterHider = useMemo(() => {
    return ViewPlugin.fromClass(class {
      constructor(view) {
        this.decorations = Decoration.none;
      }
      update(update) {
        this.decorations = Decoration.none;
      }
    }, { decorations: v => v.decorations });
  }, []);

  // Publish active editor state to context on mount/update (debounced localStorage write for ChatSidebar consumption)
  useEffect(() => {
    const fullText = (value || '').replace(/\r\n/g, '\n');
    const { data, lists } = parseFrontmatter(fullText);
    
    // **ROOSEVELT'S FRONTMATTER MERGE**: Merge data and lists so array fields (files, components, etc.) are included!
    const mergedFrontmatter = { ...data, ...lists };
    
    // **ROOSEVELT'S CANONICAL PATH**: Backend provides the canonical path - we just pass it through!
    const payload = {
      isEditable: true,
      filename: filename || 'untitled.md',
      language: 'markdown',
      content: fullText,
      contentLength: fullText.length,
      frontmatter: mergedFrontmatter,
      // Cursor info populated by onUpdate plugin below; default unknowns here
      cursorOffset: -1,
      selectionStart: -1,
      selectionEnd: -1,
      canonicalPath: canonicalPath || null, // **ROOSEVELT: Backend-provided canonical path!**
    };
    setEditorState(payload);
    // Debounced localStorage update to make active_editor available to ChatSidebarContext
    const handle = setTimeout(() => {
      try {
        localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
      } catch {}
    }, 400);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, filename, canonicalPath]);

  // Removed floating Accept listener; no longer needed

  // Global test listener for editorOperationsLive (temporary debug)
  useEffect(() => {
    const globalTestListener = (e) => {
      console.log('🔍 GLOBAL TEST: editorOperationsLive event received:', {
        type: e.type,
        detail: e.detail,
        operationsCount: e.detail?.operations?.length
      });
    };
    window.addEventListener('editorOperationsLive', globalTestListener);
    return () => {
      window.removeEventListener('editorOperationsLive', globalTestListener);
    };
  }, []);

  // ROOSEVELT'S EDITOR OPS APPLY: Listen for editor operations and apply to content with optimistic concurrency
  useEffect(() => {
    // Simple undo stack for apply batches
    if (!window.__editorUndoStack) window.__editorUndoStack = [];
    if (!window.__pushEditorUndo) {
      window.__pushEditorUndo = (text) => {
        try {
          const arr = window.__editorUndoStack;
          arr.push({ text, ts: Date.now() });
          if (arr.length > 50) arr.shift();
        } catch {}
      };
    }
    if (!window.__undoEditorOnce) {
      window.__undoEditorOnce = () => {
        try {
          const arr = window.__editorUndoStack;
          const last = arr.pop();
          if (last && typeof last.text === 'string') {
            onChange && onChange(last.text);
            return true;
          }
          return false;
        } catch { return false; }
      };
    }

    function applyOperations(e) {
      try {
        const detail = e.detail || {};
        const operations = Array.isArray(detail.operations) ? detail.operations : [];
        if (!operations.length) return;
        const current = (value || '').replace(/\r\n/g, '\n');
        // Push undo snapshot before applying batch
        window.__pushEditorUndo(current);
        // Verify pre_hash for each operation (if provided)
        function sliceHash(s) {
          // lightweight consistent hash for UI (not cryptographic, backend uses SHA-256)
          let h = 0;
          for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
          return h.toString(16);
        }
        // Prepare transforms; apply from highest index to lowest to keep offsets stable
        const ops = operations.slice().sort((a, b) => (b.start || 0) - (a.start || 0));
        let nextText = current;
        for (const op of ops) {
          const start = Math.max(0, Math.min(nextText.length, Number(op.start || 0)));
          const end = Math.max(start, Math.min(nextText.length, Number(op.end || start)));
          const before = nextText.slice(0, start);
          const mid = nextText.slice(start, end);
          const after = nextText.slice(end);
          // pre_hash check if present
          if (op.pre_hash && op.pre_hash.length > 0) {
            const ph = sliceHash(mid);
            if (ph !== op.pre_hash) {
              console.warn('⚠️ Pre-hash mismatch, skipping operation to avoid conflict:', { start, end });
              continue;
            }
          }
          if (op.op_type === 'delete_range') {
            nextText = before + after;
          } else { // replace_range default
            const text = typeof op.text === 'string' ? op.text : '';
            nextText = before + text + after;
          }
        }
        if (nextText !== current) {
          onChange && onChange(nextText);
        }
      } catch (err) {
        console.error('❌ Failed to apply editor operations:', err);
      }
    }
    window.addEventListener('codexApplyEditorOps', applyOperations);
    
    // Handle live edit acceptance (single operation from inline diff)
    function handleLiveEditAccepted(e) {
      try {
        const { operationId, operation } = e.detail || {};
        if (!operation) return;
        
        // Apply the single operation
        const current = (value || '').replace(/\r\n/g, '\n');
        window.__pushEditorUndo(current);
        
        const ops = [operation];
        applyOperations({ detail: { operations: ops } });
        
        // Remove from pending diffs
        window.dispatchEvent(new CustomEvent('removeLiveDiff', { 
          detail: { operationId } 
        }));
      } catch (err) {
        console.error('Failed to handle live edit acceptance:', err);
      }
    }
    
    // Handle live edit rejection (just remove visualization)
    function handleLiveEditRejected(e) {
      try {
        const { operationId } = e.detail || {};
        if (!operationId) return;
        
        // Just remove visualization
        window.dispatchEvent(new CustomEvent('removeLiveDiff', { 
          detail: { operationId } 
        }));
      } catch (err) {
        console.error('Failed to handle live edit rejection:', err);
      }
    }
    
    window.addEventListener('liveEditAccepted', handleLiveEditAccepted);
    window.addEventListener('liveEditRejected', handleLiveEditRejected);
    
    // Provide current editor content on request for diff previews
    function provideEditorContent() {
      try {
        const current = (value || '').replace(/\r\n/g, '\n');
        window.dispatchEvent(new CustomEvent('codexProvideEditorContent', { detail: { content: current } }));
      } catch {}
    }
    window.addEventListener('codexRequestEditorContent', provideEditorContent);
    
    return () => {
      window.removeEventListener('codexApplyEditorOps', applyOperations);
      window.removeEventListener('liveEditAccepted', handleLiveEditAccepted);
      window.removeEventListener('liveEditRejected', handleLiveEditRejected);
    };
  }, [value, onChange]);

  const applyFrontmatter = () => {
    const fullText = (value || '').replace(/\r\n/g, '\n');
    const parsed = parseFrontmatter(fullText);
    const nextData = {};
    fmEntries.forEach(({ key, value }) => {
      const k = String(key || '').trim();
      if (!k) return;
      nextData[k] = String(value ?? '').trim();
    });
    // Ensure title exists
    if (!nextData.title && baseTitle) {
      nextData.title = baseTitle;
    }
    const listUpdates = {};
    Object.entries(fmListEntries || {}).forEach(([k, txt]) => {
      const key = String(k || '').trim();
      if (!key) return;
      const items = String(txt || '').split('\n').map(s => s.trim()).filter(Boolean);
      if (items.length) listUpdates[key] = items;
    });
    const fmBlock = mergeFrontmatter(parsed.raw || fmRaw || '', nextData, listUpdates, parsed.order || fmOrder || []);
    const next = `${fmBlock}${parsed.body}`;
    onChange && onChange(next);
  };

  const addEntry = () => setFmEntries(prev => [...prev, { key: '', value: '' }]);
  const removeEntry = (idx) => setFmEntries(prev => prev.filter((_, i) => i !== idx));
  const updateEntry = (idx, field, val) => setFmEntries(prev => prev.map((e, i) => i === idx ? { ...e, [field]: val } : e));

  return (
    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <FormControlLabel control={<Switch size="small" checked={suggestionsEnabled} onChange={(e) => setSuggestionsEnabled(!!e.target.checked)} />} label={<Typography variant="caption">Predictive Suggestions</Typography>} />
        <Tooltip title="Edit frontmatter (stored invisibly at top as YAML)">
          <Button variant="outlined" size="small" onClick={() => setFmOpen(true)}>
            Frontmatter
          </Button>
        </Tooltip>
      </Box>

      <Drawer anchor="right" open={fmOpen} onClose={() => setFmOpen(false)} ModalProps={{ keepMounted: true }}>
        <Box sx={{ width: 360, p: 2, maxHeight: '100vh', overflow: 'auto' }} role="presentation">
          <Typography variant="h6" sx={{ mb: 1 }}>Frontmatter</Typography>
          <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
            Key: value pairs. This panel writes YAML at the very top of the file.
          </Typography>

          <Stack spacing={1} sx={{ mb: 1 }}>
            {fmEntries.map((entry, idx) => (
              <Box key={idx} sx={{ display: 'flex', gap: 1 }}>
                <TextField
                  size="small"
                  label="Field"
                  value={entry.key}
                  onChange={(e) => updateEntry(idx, 'key', e.target.value)}
                  sx={{ flex: 1 }}
                />
                <TextField
                  size="small"
                  label="Value"
                  value={entry.value}
                  onChange={(e) => updateEntry(idx, 'value', e.target.value)}
                  sx={{ flex: 1 }}
                />
                <IconButton size="small" onClick={() => removeEntry(idx)} aria-label="remove">
                  <Delete fontSize="small" />
                </IconButton>
              </Box>
            ))}
            {Object.entries(fmListEntries).map(([k, v]) => (
              <Box key={`list-${k}`} sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <TextField
                  size="small"
                  label="List Field"
                  value={k}
                  onChange={(e) => {
                    const newKey = e.target.value;
                    setFmListEntries(prev => {
                      const next = { ...prev };
                      next[newKey] = next[k];
                      if (newKey !== k) delete next[k];
                      return next;
                    });
                  }}
                />
                <TextField
                  size="small"
                  label="Items (one per line)"
                  value={v}
                  onChange={(e) => setFmListEntries(prev => ({ ...prev, [k]: e.target.value }))}
                  multiline
                  minRows={3}
                />
              </Box>
            ))}
          </Stack>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button variant="text" size="small" startIcon={<Add />} onClick={addEntry}>Add field</Button>
            <Button variant="text" size="small" startIcon={<Add />} onClick={() => setFmListEntries(prev => ({ ...prev, 'characters': (prev['characters'] || '') }))}>Add list</Button>
          </Box>

          <Box sx={{ mt: 2, display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
            <Button variant="outlined" size="small" onClick={() => setFmOpen(false)}>Close</Button>
            <Button variant="contained" size="small" onClick={() => { applyFrontmatter(); setFmOpen(false); }}>Apply</Button>
          </Box>
        </Box>
      </Drawer>
      {/* Memoize update listener to avoid reconfiguring extensions on every render */}
      {useMemo(() => (
        <CodeMirror
        value={value}
        height="100%"
        basicSetup={true}
        extensions={[...extensions, frontmatterHider, ...(ghostExt || []), EditorView.updateListener.of((update) => {
          try {
            if (!update.view) return;
            const sel = update.state.selection.main;
            const cursorOffset = sel.head;
            const selectionStart = sel.from;
            const selectionEnd = sel.to;
            const docText = update.state.doc.toString();
            const { data, lists } = parseFrontmatter(docText);
            
            // **ROOSEVELT'S FRONTMATTER MERGE**: Merge data and lists so array fields (files, components, etc.) are included!
            const mergedFrontmatter = { ...data, ...lists };
            
            // **ROOSEVELT: Use backend-provided canonical path**
            const payload = {
              isEditable: true,
              filename: filename || 'untitled.md',
              language: 'markdown',
              content: docText,
              contentLength: docText.length,
              frontmatter: mergedFrontmatter,
              cursorOffset,
              selectionStart,
              selectionEnd,
              canonicalPath: canonicalPath || null, // **ROOSEVELT: Backend-provided canonical path!**
            };
            setEditorState(payload);
            // throttle writes
            if (!window.__editor_ctx_write_ts || Date.now() - window.__editor_ctx_write_ts > 500) {
              window.__editor_ctx_write_ts = Date.now();
              localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
            }
          } catch {}
        })]}
        onChange={(val) => onChange && onChange(val)}
        style={{ height: '60vh' }}
      />), [value, filename, canonicalPath, extensions, frontmatterHider, ghostExt, liveEditDiffExt, setEditorState])}
      {/* Removed floating Accept/Dismiss UI */}
      {/* Undo bar */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
        <Button size="small" variant="text" onClick={() => { try { if (!window.__undoEditorOnce || !window.__undoEditorOnce()) { /* no-op */ } } catch {} }}>Undo Apply</Button>
      </Box>
    </Box>
  );
}


