import React, { useMemo, useEffect, useLayoutEffect, useState, useRef, forwardRef, useImperativeHandle } from 'react';
import CodeMirror, { ExternalChange } from '@uiw/react-codemirror';
import { EditorView, keymap, Decoration, ViewPlugin } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { markdown, markdownLanguage } from '@codemirror/lang-markdown';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { yCollab, yUndoManagerKeymap } from 'y-codemirror.next';
import { searchKeymap, getSearchQuery, setSearchQuery, openSearchPanel, closeSearchPanel } from '@codemirror/search';
import { useEditor } from '../contexts/EditorContext';
import { useTheme } from '../contexts/ThemeContext';
import { ACCENT_PALETTES } from '../theme/themeConfig';
import { parseFrontmatter as parseMarkdownFrontmatter } from '../utils/frontmatterUtils';
import { Box, TextField, Button, Tooltip, IconButton, Typography, Stack, Switch, FormControlLabel, Menu, MenuItem, ListItemIcon, ListItemText } from '@mui/material';
import { Add, Delete, ArrowUpward, ArrowDownward, ArrowDropDown, Article, TableChart } from '@mui/icons-material';
import { createGhostTextExtension } from './editor/extensions/ghostTextExtension';
import { createInlineEditSuggestionsExtension } from './editor/extensions/inlineEditSuggestionsExtension';
import { createLiveEditDiffExtension, getLiveEditDiffPlugin } from './editor/extensions/liveEditDiffExtension';
import ProposalSplitView from './editor/ProposalSplitView';
import { editorSuggestionService } from '../services/editor/EditorSuggestionService';
import { documentDiffStore } from '../services/documentDiffStore';
import OrgFileLinkDialog from './OrgFileLinkDialog';
import DictationButton from './editor/DictationButton';
import MarkdownTableEditor from './editor/MarkdownTableEditor';
import ResizableRightDrawer from './editor/ResizableRightDrawer';
import { detectTableAtCursor, parseMarkdownTable, emptyTableModel } from '../utils/markdownTableUtils';
import { devLog } from '../utils/devConsole';

const createMdTheme = (darkMode, accentId = 'blue') => {
  const palette = ACCENT_PALETTES[accentId] || ACCENT_PALETTES.blue;
  const accent = darkMode ? palette.dark : palette.light;
  const primaryMain = accent?.primary?.main ?? (darkMode ? '#90caf9' : '#1976d2');
  const selectionBg = darkMode ? '#264f78' : '#b3d7ff';
  return EditorView.baseTheme({
  '&': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121',
  },
  '.cm-editor': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121',
  },
  '.cm-scroller': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121',
  },
  '.cm-content': { 
    fontFamily: 'monospace', 
    fontSize: '14px', 
    lineHeight: '1.5', 
    wordBreak: 'break-word', 
    overflowWrap: 'anywhere',
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
    color: darkMode ? '#d4d4d4' : '#212121'
  },
  '.cm-focused': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '&.cm-focused': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '.cm-editor.cm-focused': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
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
    backgroundColor: selectionBg
  },
  '.cm-cursor': {
    borderLeftColor: darkMode ? '#ffffff' : '#000000'
  },
  '&.cm-focused .cm-selectionBackground, &.cm-focused ::selection': {
    backgroundColor: selectionBg
  },
  '.cm-line.cm-fm-hidden': { display: 'none' },
  '.cm-line': {
    caretColor: darkMode ? '#ffffff' : '#000000'
  },
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
    color: primaryMain
  },
  '.cm-content .cm-url': {
    color: primaryMain
  }
});
};

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

const MarkdownCMEditor = forwardRef(({
  value,
  onChange,
  filename,
  canonicalPath,
  documentId,
  initialScrollPosition = 0,
  restoreScrollAfterRefresh = null,
  onScrollRestored,
  onScrollChange,
  onCurrentSectionChange,
  menuContainerEl = null,
  isCollaborative = false,
  ytext = null,
  awareness = null,
  undoManager = null,
  onCollabDocChange = null,
}, ref) => {
  const { darkMode, accentId } = useTheme();
  const themeSignature = `${darkMode ? 'dark' : 'light'}-${accentId}`;
  const [themeRemountNonce, setThemeRemountNonce] = useState(0);
  const hasMountedRef = useRef(false);
  
  const [diffCount, setDiffCount] = useState(0);
  const [showSplitView, setShowSplitView] = useState(false);

  // Track diff count for UI visibility
  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
      return;
    }
    // Force CodeMirror remount so mode/accent updates are applied immediately.
    setThemeRemountNonce((prev) => prev + 1);
  }, [themeSignature]);

  useEffect(() => {
    if (!documentId) {
      setDiffCount(0);
      return;
    }
    
    const updateCount = () => {
      const count = documentDiffStore.getDiffCount(documentId);
      devLog(`📊 MarkdownCMEditor: Diff count for ${documentId} is ${count}`);
      setDiffCount(count);
    };
    
    updateCount();
    documentDiffStore.subscribe(updateCount);
    return () => documentDiffStore.unsubscribe(updateCount);
  }, [documentId]);

  // Refs for scroll position preservation
  const editorViewRef = useRef(null);
  const savedScrollPosRef = useRef(null);
  const shouldRestoreScrollRef = useRef(false);
  const scrollCallbackTimeoutRef = useRef(null);
  const hasRestoredInitialScrollRef = useRef(false);
  const [fileLinkDialogOpen, setFileLinkDialogOpen] = useState(false);
  const [toolsMenuAnchorEl, setToolsMenuAnchorEl] = useState(null);
  const [tableEditorOpen, setTableEditorOpen] = useState(false);
  const [tableEditorModel, setTableEditorModel] = useState(() => emptyTableModel());
  const [tableEditorIsEditing, setTableEditorIsEditing] = useState(false);
  const tableReplaceRangeRef = useRef(null);

  const [collabDocText, setCollabDocText] = useState('');
  useLayoutEffect(() => {
    if (!isCollaborative || !ytext) {
      setCollabDocText('');
      return undefined;
    }
    const bump = () => setCollabDocText(ytext.toString());
    bump();
    ytext.observe(bump);
    return () => {
      ytext.unobserve(bump);
    };
  }, [isCollaborative, ytext]);

  const insertTextAtCursor = React.useCallback((text) => {
    const view = editorViewRef.current;
    if (!view) return;
    const state = view.state;
    const selection = state.selection.main;
    const from = selection.from;
    const to = selection.to;
    view.dispatch({
      changes: { from, to, insert: text },
      selection: { anchor: from + text.length },
    });
  }, []);

  const handleFileLinkSelect = React.useCallback((linkText) => {
    insertTextAtCursor(linkText);
  }, [insertTextAtCursor]);

  const handleOpenTableEditor = React.useCallback(() => {
    const view = editorViewRef.current;
    tableReplaceRangeRef.current = null;
    if (!view) {
      setTableEditorModel(emptyTableModel());
      setTableEditorIsEditing(false);
      setTableEditorOpen(true);
      return;
    }
    const doc = view.state.doc;
    const pos = view.state.selection.main.head;
    const detected = detectTableAtCursor(doc, pos);
    if (detected) {
      const parsed = parseMarkdownTable(detected.text);
      if (parsed) {
        setTableEditorModel(parsed);
        setTableEditorIsEditing(true);
        tableReplaceRangeRef.current = { from: detected.from, to: detected.to };
      } else {
        setTableEditorModel(emptyTableModel());
        setTableEditorIsEditing(false);
      }
    } else {
      setTableEditorModel(emptyTableModel());
      setTableEditorIsEditing(false);
    }
    setTableEditorOpen(true);
  }, []);

  // Expose scrollToLine and getScrollPosition to parent via ref
  useImperativeHandle(ref, () => ({
    getScrollPosition: () => {
      if (!editorViewRef.current?.scrollDOM) return 0;
      return editorViewRef.current.scrollDOM.scrollTop;
    },
    getSelectedText: () => {
      const view = editorViewRef.current;
      if (!view) return null;
      const selection = view.state.selection.main;
      if (selection.from === selection.to) return null;
      return view.state.sliceDoc(selection.from, selection.to);
    },
    scrollToLine: (lineNum) => {
      if (!editorViewRef.current || !lineNum || lineNum < 1) return;
      try {
        const view = editorViewRef.current;
        const line = view.state.doc.line(lineNum);
        const pos = line.from;
        
        view.dispatch({
          effects: EditorView.scrollIntoView(pos, { y: 'start', yMargin: 100 })
        });
        
        // Add brief highlight effect
        const lineElement = view.domAtPos(pos).node.parentElement?.closest('.cm-line');
        if (lineElement) {
          lineElement.style.backgroundColor = '#fff3cd';
          setTimeout(() => {
            lineElement.style.backgroundColor = '';
          }, 1000);
        }
      } catch (err) {
        console.error('Error scrolling to line:', err);
      }
    }
  }));
  
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
        maxChars: 300,
        signal
      });
      return suggestion || '';
    } catch { return ''; }
  }, { debounceMs: 350 }) : [], [suggestionsEnabled]);
  const mdTheme = useMemo(() => createMdTheme(darkMode, accentId), [darkMode, accentId]);
  const inlineEditExt = useMemo(() => createInlineEditSuggestionsExtension(), []);
  
  // ✅ documentId comes from props, passed down from DocumentViewer
  const liveEditDiffExt = useMemo(() => {
    if (!documentId) {
      devLog('🔍 MarkdownCMEditor: No documentId, skipping liveEditDiffExt');
      return [];
    }
    
    const ext = createLiveEditDiffExtension(documentId);
    devLog('🔍 MarkdownCMEditor: Created liveEditDiffExt for document:', documentId, 'items:', ext?.length);
    return ext;
  }, [documentId]);
  // Persistent search configuration
  const persistentSearchExt = useMemo(() => {
    // ViewPlugin to keep search panel open and restore search term
    const persistentSearchPlugin = ViewPlugin.fromClass(class {
      constructor(view) {
        this.view = view;
        this.lastSearch = localStorage.getItem('editor_last_search') || '';
        this.panelElement = null;
        
        // Set up observer to monitor search panel
        this.observer = new MutationObserver(() => {
          this.handlePanelChange();
        });
        
        // Start observing after a short delay to let panel render
        setTimeout(() => {
          this.panelElement = view.dom.querySelector('.cm-search');
          if (this.panelElement) {
            this.observer.observe(this.panelElement, {
              attributes: true,
              attributeFilter: ['style', 'class']
            });
            // Prevent panel from closing when clicking editor
            this.setupClickHandler();
          }
        }, 200);
      }
      
      setupClickHandler() {
        // Override default behavior: keep panel open when clicking editor
        const editorContent = this.view.contentDOM;
        if (editorContent) {
          const handleMouseDown = (e) => {
            const panel = this.view.dom.querySelector('.cm-search');
            if (panel && !panel.contains(e.target)) {
              // Click in editor - prevent panel from closing
              // CodeMirror closes panel on focus loss, so we'll reopen it
              setTimeout(() => {
                const currentPanel = this.view.dom.querySelector('.cm-search');
                if (!currentPanel || currentPanel.style.display === 'none') {
                  // Panel was closed, reopen it
                  openSearchPanel(this.view);
                  // Restore search term
                  if (this.lastSearch) {
                    const query = getSearchQuery(this.view.state);
                    if (query) {
                      this.view.dispatch({
                        effects: setSearchQuery(this.view.state, {
                          search: this.lastSearch,
                          caseSensitive: query.caseSensitive || false,
                          literal: query.literal || false,
                          regexp: query.regexp || false,
                          wholeWord: query.wholeWord || false
                        })
                      });
                    }
                  }
                }
              }, 10);
            }
          };
          editorContent.addEventListener('mousedown', handleMouseDown, true);
          this.clickHandler = handleMouseDown;
        }
      }
      
      handlePanelChange() {
        const panel = this.view.dom.querySelector('.cm-search');
        if (panel && this.panelElement !== panel) {
          this.panelElement = panel;
          this.setupClickHandler();
        }
      }
      
      update(update) {
        // Save search term when it changes
        const query = getSearchQuery(update.state);
        if (query && query.search) {
          this.lastSearch = query.search;
          localStorage.setItem('editor_last_search', query.search);
        }
      }
      
      destroy() {
        if (this.observer) {
          this.observer.disconnect();
        }
        if (this.clickHandler && this.view.contentDOM) {
          this.view.contentDOM.removeEventListener('mousedown', this.clickHandler, true);
        }
      }
    });
    
    // Custom keymap for Ctrl+F that restores last search
    const persistentSearchKeymap = keymap.of([
      {
        key: 'Mod-f',
        run: (view) => {
          const lastSearch = localStorage.getItem('editor_last_search') || '';
          openSearchPanel(view);
          // Restore last search term after opening panel
          if (lastSearch) {
            setTimeout(() => {
              const query = getSearchQuery(view.state);
              if (query) {
                view.dispatch({
                  effects: setSearchQuery(view.state, {
                    search: lastSearch,
                    caseSensitive: query.caseSensitive || false,
                    literal: query.literal || false,
                    regexp: query.regexp || false,
                    wholeWord: query.wholeWord || false
                  })
                });
              }
            }, 150);
          }
          return true;
        }
      }
    ]);
    
    return [
      persistentSearchKeymap,
      persistentSearchPlugin
    ];
  }, []);

  const fileLinkKeymap = useMemo(() => keymap.of([
    { key: 'Mod-Alt-l', run: () => { setFileLinkDialogOpen(true); return true; } }
  ]), []);

  const tableEditorKeymap = useMemo(() => keymap.of([
    { key: 'Mod-Shift-t', run: () => { handleOpenTableEditor(); return true; } }
  ]), [handleOpenTableEditor]);

  const extensions = useMemo(() => {
    const hist = isCollaborative ? [] : [history()];
    // y-codemirror.next exports yUndoManagerKeymap as a KeyBinding[] (not a factory); yCollab wires UndoManager via facet.
    const undoKeys =
      isCollaborative && ytext && awareness && undoManager
        ? yUndoManagerKeymap
        : historyKeymap;
    const collabExts =
      isCollaborative && ytext && awareness && undoManager
        ? [yCollab(ytext, awareness, { undoManager })]
        : [];
    return [
      ...hist,
      keymap.of([...defaultKeymap, ...undoKeys, ...searchKeymap]),
      fileLinkKeymap,
      tableEditorKeymap,
      markdown({ base: markdownLanguage }),
      EditorView.lineWrapping,
      mdTheme,
      inlineEditExt,
      liveEditDiffExt,
      ...persistentSearchExt,
      ...collabExts,
    ];
  }, [
    mdTheme,
    inlineEditExt,
    liveEditDiffExt,
    persistentSearchExt,
    fileLinkKeymap,
    tableEditorKeymap,
    isCollaborative,
    ytext,
    awareness,
    undoManager,
  ]);

  const { setEditorState } = useEditor();
  const [fmOpen, setFmOpen] = useState(false);

  const fmSourceLine = isCollaborative ? collabDocText : value;
  const { data: initialData, lists: initialLists, raw: initialRaw, order: initialOrder, body: initialBody } = useMemo(() => {
    const parsed = parseFrontmatter((fmSourceLine || '').replace(/\r\n/g, '\n'));
    // Debug logging removed for performance - fires on every keystroke
    return parsed;
  }, [fmSourceLine]);
  const baseTitle = useMemo(() => (filename ? String(filename).replace(/\.[^.]+$/, '') : ''), [filename]);
  const [fmEntries, setFmEntries] = useState(() => {
    const entries = Object.entries(initialData).map(([k, v]) => ({ key: k, value: String(v ?? '') }));
    // Ensure title exists for new files
    if (!entries.find(e => e.key === 'title') && baseTitle) {
      entries.unshift({ key: 'title', value: baseTitle });
    }
    // Debug logging removed for performance
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
    // Debug logging removed for performance - fires on every change
    setFmEntries(entries);
    const obj = {};
    Object.entries(initialLists || {}).forEach(([k, arr]) => { obj[k] = (arr || []).join('\n'); });
    setFmListEntries(obj);
    setFmRaw(initialRaw || '');
    setFmOrder(initialOrder || []);
  }, [initialData, initialLists, initialRaw, initialOrder, baseTitle]);

  // Restore scroll position after value changes (from diff accept/reject, or from full-document refresh after Accept All)
  const effectiveEditorText = isCollaborative ? collabDocText : value;
  const prevValueForScrollRestoreRef = useRef(effectiveEditorText);
  useEffect(() => {
    if (shouldRestoreScrollRef.current && savedScrollPosRef.current !== null && editorViewRef.current) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        if (editorViewRef.current && savedScrollPosRef.current !== null) {
          const scrollDOM = editorViewRef.current.scrollDOM;
          if (scrollDOM) {
            scrollDOM.scrollTop = savedScrollPosRef.current;
            devLog('📜 Restored scroll position:', savedScrollPosRef.current);
            shouldRestoreScrollRef.current = false;
            savedScrollPosRef.current = null;
          }
        }
      });
      return;
    }
    // Full-document refresh (e.g. Accept All): parent set restoreScrollAfterRefresh; restore when content updates
    const valueChanged = prevValueForScrollRestoreRef.current !== effectiveEditorText;
    prevValueForScrollRestoreRef.current = effectiveEditorText;
    if (valueChanged && typeof restoreScrollAfterRefresh === 'number' && restoreScrollAfterRefresh > 0 && editorViewRef.current) {
      requestAnimationFrame(() => {
        if (editorViewRef.current && editorViewRef.current.scrollDOM) {
          editorViewRef.current.scrollDOM.scrollTop = restoreScrollAfterRefresh;
          devLog('📜 Restored scroll after refresh:', restoreScrollAfterRefresh);
          if (typeof onScrollRestored === 'function') onScrollRestored();
        }
      });
    }
  }, [effectiveEditorText, restoreScrollAfterRefresh, onScrollRestored]);
  
  // Restore initial scroll position on mount (for tab switching and page reload).
  // Editor view ref is set asynchronously by CodeMirror, so retry until the view exists.
  useEffect(() => {
    if (hasRestoredInitialScrollRef.current || initialScrollPosition <= 0) return;

    const tryRestore = () => {
      if (!editorViewRef.current?.scrollDOM) return false;
      editorViewRef.current.scrollDOM.scrollTop = initialScrollPosition;
      hasRestoredInitialScrollRef.current = true;
      return true;
    };

    let rafId = null;
    let t1 = null;
    let t2 = null;

    rafId = requestAnimationFrame(() => {
      if (tryRestore()) return;
      t1 = setTimeout(() => {
        if (tryRestore()) return;
        t2 = setTimeout(tryRestore, 100);
      }, 50);
    });

    return () => {
      if (rafId != null) cancelAnimationFrame(rafId);
      if (t1 != null) clearTimeout(t1);
      if (t2 != null) clearTimeout(t2);
    };
  }, [initialScrollPosition]);
  
  // Track scroll changes and notify parent (debounced)
  useEffect(() => {
    if (!editorViewRef.current || !onScrollChange) return;
    
    const scrollDOM = editorViewRef.current.scrollDOM;
    if (!scrollDOM) return;
    
    const handleScroll = () => {
      // Clear any pending callback
      if (scrollCallbackTimeoutRef.current) {
        clearTimeout(scrollCallbackTimeoutRef.current);
      }
      
      // Debounce scroll position updates (300ms)
      scrollCallbackTimeoutRef.current = setTimeout(() => {
        if (scrollDOM && onScrollChange) {
          const scrollTop = scrollDOM.scrollTop;
          onScrollChange(scrollTop);
        }
      }, 300);
    };
    
    scrollDOM.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      scrollDOM.removeEventListener('scroll', handleScroll);
      if (scrollCallbackTimeoutRef.current) {
        clearTimeout(scrollCallbackTimeoutRef.current);
      }
    };
  }, [onScrollChange]);
  
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

  // Track last documentId to detect document switches
  const lastDocumentIdRef = React.useRef(null);
  
  // Helper function to parse frontmatter and update cache (on-demand only)
  const refreshEditorCacheWithFrontmatter = React.useCallback((content = null) => {
    const fullText = content !== null
      ? String(content).replace(/\r\n/g, '\n')
      : (isCollaborative ? (collabDocText || '') : (value || '')).replace(/\r\n/g, '\n');
    const { data, lists } = parseFrontmatter(fullText);
    const mergedFrontmatter = { ...data, ...lists };
    
    // Update window cache
    window.__last_editor_frontmatter = mergedFrontmatter;
    window.__last_editor_content = fullText;
    
    // CRITICAL FIX: Get actual cursor position from CodeMirror view
    let cursorOffset = -1;
    let selectionStart = -1;
    let selectionEnd = -1;
    
    if (editorViewRef.current) {
      try {
        const sel = editorViewRef.current.state.selection.main;
        cursorOffset = sel.head;
        selectionStart = sel.from;
        selectionEnd = sel.to;
        
        devLog('📍 CURSOR POSITION DEBUG: Cache refresh captured cursor at:', {
          cursorOffset,
          selectionStart,
          selectionEnd,
          documentLength: fullText.length,
          isAtEndOfFile: cursorOffset === fullText.length,
          filename: filename
        });
      } catch (e) {
        // If we can't get cursor position, fall back to -1
        console.warn('Failed to get cursor position from editor view:', e);
      }
    } else {
      console.warn('⚠️ No editorViewRef available when refreshing cache - cursor position will be -1');
    }
    
    // Set context state
    const payload = {
      isEditable: true,
      filename: filename || 'untitled.md',
      language: 'markdown',
      content: fullText,
      contentLength: fullText.length,
      frontmatter: mergedFrontmatter,
      cursorOffset: cursorOffset,
      selectionStart: selectionStart,
      selectionEnd: selectionEnd,
      canonicalPath: canonicalPath || null,
      documentId: documentId || null,
    };
    
    setEditorState(payload);
    
    // Write to localStorage for chat to read
    try {
      localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
    } catch {}
    
    return mergedFrontmatter;
  }, [value, collabDocText, isCollaborative, filename, canonicalPath, documentId, setEditorState]);

  const handleApplyTableMarkdown = React.useCallback((md) => {
    const view = editorViewRef.current;
    if (!view) return;
    const range = tableReplaceRangeRef.current;
    // Mark as external so @uiw/react-codemirror does not treat this as debounced typing (which can
    // queue a stale value sync and revert the edit). Collaborative mode uses onChange={() => {}},
    // so we must push the new text to the parent explicitly.
    const externalAnno = [ExternalChange.of(true)];
    if (range && typeof range.from === 'number' && typeof range.to === 'number') {
      view.dispatch({
        changes: { from: range.from, to: range.to, insert: md },
        selection: { anchor: range.from + md.length },
        annotations: externalAnno,
      });
    } else {
      const pos = view.state.selection.main.head;
      let insert = md;
      if (pos > 0) {
        const chBefore = view.state.doc.sliceString(pos - 1, pos);
        if (chBefore !== '\n') insert = `\n${insert}`;
      }
      if (!insert.endsWith('\n')) insert += '\n';
      view.dispatch({
        changes: { from: pos, to: pos, insert },
        selection: { anchor: pos + insert.length },
        annotations: externalAnno,
      });
    }
    tableReplaceRangeRef.current = null;
    const next = view.state.doc.toString();
    if (isCollaborative && typeof onCollabDocChange === 'function') {
      onCollabDocChange(next);
    } else if (onChange) {
      onChange(next);
    }
    queueMicrotask(() => {
      try {
        refreshEditorCacheWithFrontmatter(next);
      } catch {
        /* ignore */
      }
    });
  }, [isCollaborative, onChange, onCollabDocChange, refreshEditorCacheWithFrontmatter]);

  // Set editor state when document changes (NOT on every value change)
  // This ensures frontmatter is correct when switching tabs
  useEffect(() => {
    const documentChanged = lastDocumentIdRef.current !== documentId;
    
    // Only update if document changed
    if (!documentChanged) {
      return;
    }
    
    // Update ref to track current document
    lastDocumentIdRef.current = documentId;
    
    // Parse frontmatter and update cache when document changes
    refreshEditorCacheWithFrontmatter();
    
    // Cleanup on unmount - clear editor state
    return () => {
      if (documentId !== lastDocumentIdRef.current) {
        // Only clear if we're actually unmounting (document changed)
        setEditorState({
          isEditable: false,
          filename: null,
          language: null,
          content: null,
          contentLength: 0,
          frontmatter: null,
          cursorOffset: -1,
          selectionStart: -1,
          selectionEnd: -1,
          canonicalPath: null,
          documentId: null,
        });
      }
    };
    // Only run when documentId/filename/canonicalPath changes, NOT on value changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename, canonicalPath, documentId, refreshEditorCacheWithFrontmatter]);

  // Watch for content changes and update cache (content only, no frontmatter parsing)
  // Debounced to avoid excessive updates during typing
  useEffect(() => {
    if (isCollaborative) return undefined;
    const fullText = (value || '').replace(/\r\n/g, '\n');
    const cachedContent = window.__last_editor_content || '';
    const cachedFrontmatter = window.__last_editor_frontmatter || {};
    
    // Only update if content changed (don't parse frontmatter during typing)
    if (fullText !== cachedContent) {
      // Debounce updates to avoid excessive writes during typing (500ms)
      if (window.__editor_content_update_timer) {
        clearTimeout(window.__editor_content_update_timer);
      }
      
      window.__editor_content_update_timer = setTimeout(() => {
        // Update content cache (keep existing frontmatter - don't reparse)
        window.__last_editor_content = fullText;
        
        // Update React context and localStorage with current content but cached frontmatter
        const payload = {
          isEditable: true,
          filename: filename || 'untitled.md',
          language: 'markdown',
          content: fullText,
          contentLength: fullText.length,
          frontmatter: cachedFrontmatter, // Use cached frontmatter, not re-parsed
          cursorOffset: -1,
          selectionStart: -1,
          selectionEnd: -1,
          canonicalPath: canonicalPath || null,
          documentId: documentId || null,
        };
        
        setEditorState(payload);
        
        // Update localStorage cache so ChatSidebar picks up content changes
        try {
          localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
        } catch (e) {
          console.error('Failed to update editor_ctx_cache:', e);
        }
      }, 500); // 500ms debounce for content updates
    }
    
    // Cleanup: clear timeout on unmount or when deps change
    return () => {
      if (window.__editor_content_update_timer) {
        clearTimeout(window.__editor_content_update_timer);
      }
    };
  }, [isCollaborative, value, filename, canonicalPath, documentId, setEditorState]);

  // Removed floating Accept listener; no longer needed

  // Listen for cache refresh requests (e.g., from chat before sending message)
  useEffect(() => {
    const handleRefreshCache = () => {
      // Refresh cache with fresh frontmatter parsing before chat reads it
      refreshEditorCacheWithFrontmatter();
      
      // Dispatch event to notify that cache refresh is complete
      window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
    };
    
    window.addEventListener('refreshEditorCache', handleRefreshCache);
    return () => {
      window.removeEventListener('refreshEditorCache', handleRefreshCache);
    };
  }, [refreshEditorCacheWithFrontmatter]);

  // Clean up decorations on unmount to prevent duplicates
  useEffect(() => {
    return () => {
      // Clear any pending decoration updates
      if (window.__decorationCleanupTimeout) {
        clearTimeout(window.__decorationCleanupTimeout);
        window.__decorationCleanupTimeout = null;
      }
    };
  }, []);

  // Editor operations apply: Listen for editor operations and apply via CodeMirror transactions
  useEffect(() => {

    function applyOperations(e) {
      try {
        const detail = e.detail || {};
        const operations = Array.isArray(detail.operations) ? detail.operations : [];
        if (!operations.length) return;
        
        const view = editorViewRef.current;
        if (!view) {
          console.warn('⚠️ No editor view available for applying operations');
          return;
        }
        
        const doc = view.state.doc;
        const docText = doc.toString();
        
        // Verify pre_hash for each operation (if provided)
        function sliceHash(s) {
          // lightweight consistent hash for UI (not cryptographic, backend uses SHA-256)
          let h = 0;
          for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
          return h.toString(16);
        }
        
        // Prepare transforms; apply from highest index to lowest to keep offsets stable
        // For operations with the same start position (text chunks), sort by chunk_index ascending
        const ops = operations.slice().sort((a, b) => {
          const startDiff = (b.start || 0) - (a.start || 0);
          if (startDiff !== 0) return startDiff; // Different positions: highest first
          
          // Same position: check if they're text chunks (have is_text_chunk and chunk_index)
          const aIsChunk = a.is_text_chunk && a.chunk_index !== undefined;
          const bIsChunk = b.is_text_chunk && b.chunk_index !== undefined;
          
          if (aIsChunk && bIsChunk) {
            // Both are chunks: sort by chunk_index ascending (chunk 0, then 1, then 2...)
            return (a.chunk_index || 0) - (b.chunk_index || 0);
          }
          
          return 0; // Keep original order for non-chunks at same position
        });
        
        // Build ChangeSet from operations
        const changes = [];
        let hasValidChanges = false;
        
        for (const op of ops) {
          const start = Math.max(0, Math.min(doc.length, Number(op.start || 0)));
          const end = Math.max(start, Math.min(doc.length, Number(op.end || start)));
          
          // pre_hash check if present (skip for insert operations where start == end)
          if (op.pre_hash && op.pre_hash.length > 0 && start !== end) {
            const currentSlice = docText.slice(start, end);
            const ph = sliceHash(currentSlice);
            if (ph !== op.pre_hash) {
              console.warn('⚠️ Pre-hash mismatch, skipping operation to avoid conflict:', { start, end });
              continue;
            }
          }
          
          let newText = '';
          if (op.op_type === 'delete_range') {
            newText = '';
          } else if (op.op_type === 'insert_after_heading' || op.op_type === 'insert_after') {
            // Insert operation: start === end, insert text at that position
            newText = typeof op.text === 'string' ? op.text : '';
          } else { // replace_range default
            newText = typeof op.text === 'string' ? op.text : '';
          }
          
          // Only add change if it's different from current content
          const currentSlice = docText.slice(start, end);
          if (currentSlice !== newText) {
            changes.push({ from: start, to: end, insert: newText });
            hasValidChanges = true;
          }
        }
        
        if (hasValidChanges && changes.length > 0) {
          devLog('✅ Applying operations via CodeMirror transactions:', { 
            originalLength: doc.length, 
            operationsCount: operations.length,
            changesCount: changes.length
          });
          
          // Save scroll position before applying changes
          if (view.scrollDOM) {
            savedScrollPosRef.current = view.scrollDOM.scrollTop;
            shouldRestoreScrollRef.current = true;
            devLog('💾 Saved scroll position:', savedScrollPosRef.current);
          }
          
          // Dispatch as single transaction using the changes array directly
          // This integrates edits into native undo/redo history
          view.dispatch({
            changes: changes,
            userEvent: 'agent-edit'
          });
          
          // Get the new document text after transaction
          const nextText = view.state.doc.toString();
          
          if (onChange) {
            onChange(nextText);
            try {
              refreshEditorCacheWithFrontmatter(nextText);
            } catch (err) {
              console.error('Failed to update cache after operation apply:', err);
            }
            if (detail.emitLiveEditApplied && documentId) {
              window.dispatchEvent(new CustomEvent('liveEditApplied', {
                detail: { documentId, content: nextText }
              }));
            }
          } else if (isCollaborative && typeof onCollabDocChange === 'function') {
            try {
              refreshEditorCacheWithFrontmatter(nextText);
            } catch (err) {
              console.error('Failed to update cache after operation apply:', err);
            }
            onCollabDocChange(nextText);
            if (detail.emitLiveEditApplied && documentId) {
              window.dispatchEvent(new CustomEvent('liveEditApplied', {
                detail: { documentId, content: nextText }
              }));
            }
          } else {
            console.warn('⚠️ onChange callback is not defined');
          }
        } else {
          console.warn('⚠️ Operations did not produce valid changes', { 
            operationsCount: operations.length,
            operations: operations.map(op => ({ 
              op_type: op.op_type, 
              start: op.start, 
              end: op.end,
              textLength: op.text?.length,
              textPreview: op.text?.substring(0, 30)
            })),
            docLength: doc.length
          });
        }
      } catch (err) {
        console.error('❌ Failed to apply editor operations:', err);
      }
    }
    window.addEventListener('codexApplyEditorOps', applyOperations);
    
    // Provide current editor content on request for diff previews
    function provideEditorContent() {
      try {
        const fromView = editorViewRef.current?.state?.doc?.toString?.();
        const current = (fromView ?? (isCollaborative ? (collabDocText || '') : (value || ''))).replace(/\r\n/g, '\n');
        window.dispatchEvent(new CustomEvent('codexProvideEditorContent', { detail: { content: current } }));
      } catch { /* ignore */ }
    }
    window.addEventListener('codexRequestEditorContent', provideEditorContent);
    
    return () => {
      window.removeEventListener('codexApplyEditorOps', applyOperations);
      window.removeEventListener('codexRequestEditorContent', provideEditorContent);
    };
  }, [value, onChange, isCollaborative, onCollabDocChange, collabDocText, refreshEditorCacheWithFrontmatter, documentId]);

  const applyFrontmatter = () => {
    const fullText = (isCollaborative ? (collabDocText || '') : (value || '')).replace(/\r\n/g, '\n');
    const parsed = parseFrontmatter(fullText);
    const nextData = {};
    fmEntries.forEach(({ key, value: entryVal }) => {
      const k = String(key || '').trim();
      if (!k) return;
      nextData[k] = String(entryVal ?? '').trim();
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
    if (isCollaborative && ytext) {
      ytext.doc.transact(() => {
        ytext.delete(0, ytext.length);
        ytext.insert(0, next);
      });
      if (typeof onCollabDocChange === 'function') onCollabDocChange(next);
    } else if (onChange) {
      onChange(next);
    }
    setTimeout(() => {
      refreshEditorCacheWithFrontmatter(next);
    }, 0);
  };

  const addEntry = () => setFmEntries(prev => [...prev, { key: '', value: '' }]);
  const removeEntry = (idx) => setFmEntries(prev => prev.filter((_, i) => i !== idx));
  const updateEntry = (idx, field, val) => setFmEntries(prev => prev.map((e, i) => i === idx ? { ...e, [field]: val } : e));

  // Memoize CodeMirror to avoid reconfiguring extensions on every render. Must be called unconditionally (rules of hooks).
  const codeMirrorEditor = useMemo(() => (
    <CodeMirror
      key={`${documentId || 'no-doc'}-${themeSignature}-${themeRemountNonce}${isCollaborative ? '-collab' : ''}`}
      value={isCollaborative ? (collabDocText || value || '') : (value || '')}
      height="100%"
      basicSetup={false}
      extensions={[...extensions, frontmatterHider, ...(ghostExt || []), EditorView.updateListener.of((update) => {
        try {
          if (!update.view) return;
          if (!editorViewRef.current) editorViewRef.current = update.view;
          const sel = update.state.selection.main;
          const cursorOffset = sel.head;
          const selectionStart = sel.from;
          const selectionEnd = sel.to;
          const docText = update.state.doc.toString();
          if (isCollaborative && update.docChanged && typeof onCollabDocChange === 'function') {
            if (window.__collab_preview_timer) clearTimeout(window.__collab_preview_timer);
            const view = update.view;
            window.__collab_preview_timer = setTimeout(() => {
              const txt = view.state.doc.toString();
              onCollabDocChange(txt);
              try {
                refreshEditorCacheWithFrontmatter(txt);
              } catch { /* ignore */ }
            }, 120);
          }
          if (onCurrentSectionChange && (update.selectionSet || update.docChanged)) {
            onCurrentSectionChange(docText, cursorOffset);
          }
          const mergedFrontmatter = window.__last_editor_frontmatter || {};
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
            canonicalPath: canonicalPath || null,
            documentId: documentId || null,
          };
          if (!window.__editor_ctx_write_ts || Date.now() - window.__editor_ctx_write_ts > 500) {
            window.__editor_ctx_write_ts = Date.now();
            localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
          }
        } catch { /* ignore */ }
      })]}
      onChange={isCollaborative ? (() => {}) : ((val) => onChange && onChange(val))}
      style={{ height: '100%' }}
    />
  ), [value, collabDocText, filename, canonicalPath, extensions, frontmatterHider, ghostExt, liveEditDiffExt, setEditorState, onCurrentSectionChange, documentId, darkMode, accentId, themeSignature, themeRemountNonce, isCollaborative, onCollabDocChange, refreshEditorCacheWithFrontmatter]);

  return (
    <Box sx={{ bgcolor: 'background.paper', p: 1, borderRadius: 1, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
        <FormControlLabel control={<Switch size="small" checked={suggestionsEnabled} onChange={(e) => setSuggestionsEnabled(!!e.target.checked)} />} label={<Typography variant="caption">Predictive Suggestions</Typography>} />
        <Box sx={{ display: 'flex', gap: 1 }}>
          <DictationButton insertText={insertTextAtCursor} />
          <Tooltip title="Frontmatter, markdown table (Mod+Shift+T), and more">
            <Button
              variant="outlined"
              size="small"
              endIcon={<ArrowDropDown />}
              onClick={(e) => setToolsMenuAnchorEl(e.currentTarget)}
            >
              Tools
            </Button>
          </Tooltip>
          <Menu
            anchorEl={toolsMenuAnchorEl}
            open={Boolean(toolsMenuAnchorEl)}
            onClose={() => setToolsMenuAnchorEl(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
            PaperProps={{ sx: { minWidth: 200 } }}
            // When the editor is inside a Fullscreen API element, portals to document.body
            // can render outside the fullscreen tree. Mount the menu inside the provided container.
            container={menuContainerEl || undefined}
          >
            <MenuItem
              onClick={() => {
                setToolsMenuAnchorEl(null);
                refreshEditorCacheWithFrontmatter();
                setFmOpen(true);
              }}
            >
              <ListItemIcon>
                <Article fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Frontmatter" primaryTypographyProps={{ fontSize: '0.875rem' }} />
            </MenuItem>
            <MenuItem
              onClick={() => {
                setToolsMenuAnchorEl(null);
                handleOpenTableEditor();
              }}
            >
              <ListItemIcon>
                <TableChart fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary={
                  editorViewRef.current &&
                  detectTableAtCursor(
                    editorViewRef.current.state.doc,
                    editorViewRef.current.state.selection.main.head
                  )
                    ? 'Edit table'
                    : 'Insert table'
                }
                primaryTypographyProps={{ fontSize: '0.875rem' }}
              />
            </MenuItem>
          </Menu>
        </Box>
      </Box>

      <ResizableRightDrawer
        open={fmOpen}
        onClose={() => setFmOpen(false)}
        defaultWidth={360}
        minWidth={280}
        storageKey="markdownEditor_frontmatterDrawerWidth"
        ModalProps={{ keepMounted: true }}
      >
        <Box sx={{ p: 2, boxSizing: 'border-box' }} role="presentation">
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
      </ResizableRightDrawer>
      <MarkdownTableEditor
        open={tableEditorOpen}
        onClose={() => setTableEditorOpen(false)}
        initialModel={tableEditorModel}
        isEditing={tableEditorIsEditing}
        onApply={handleApplyTableMarkdown}
      />
      <Box sx={{ flex: 1, minHeight: 0 }}>
      {showSplitView && documentId ? (
        <ProposalSplitView
          content={value}
          operations={documentDiffStore.getDiffs(documentId)?.operations ?? []}
          onAcceptAll={() => {
            try {
              const plugin = getLiveEditDiffPlugin(documentId);
              if (plugin?.acceptAllOperations) plugin.acceptAllOperations();
              setShowSplitView(false);
            } catch (err) {
              console.error('Failed to accept all:', err);
            }
          }}
          onRejectAll={() => {
            try {
              const plugin = getLiveEditDiffPlugin(documentId);
              if (plugin?.rejectAllOperations) plugin.rejectAllOperations();
              setShowSplitView(false);
            } catch (err) {
              console.error('Failed to reject all:', err);
            }
          }}
          onClose={() => setShowSplitView(false)}
          height="100%"
        />
      ) : (
        codeMirrorEditor
      )}
      </Box>
      {/* Removed floating Accept/Dismiss UI */}
      {/* Diff navigation and batch operations */}
      {diffCount > 0 && !showSplitView && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1, mt: 1 }}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button 
              size="small" 
              variant="outlined" 
              onClick={() => setShowSplitView(true)}
            >
              Compare
            </Button>
            <Button 
              size="small" 
              variant="outlined" 
              startIcon={<ArrowUpward />}
              onClick={() => {
                try {
                  const plugin = documentId ? getLiveEditDiffPlugin(documentId) : null;
                  const view = editorViewRef.current;
                  if (plugin && view) {
                    const cursorPos = view.state.selection.main.head;
                    const prevDiff = plugin.findPreviousDiff(cursorPos);
                    if (prevDiff) {
                      plugin.jumpToPosition(prevDiff.position);
                    } else {
                      devLog('No previous diff found');
                    }
                  } else {
                    console.warn('⚠️ Cannot navigate: plugin or view not available');
                  }
                } catch (err) {
                  console.error('Failed to jump to previous diff:', err);
                }
              }}
            >
              Previous Edit
            </Button>
            <Button 
              size="small" 
              variant="outlined" 
              startIcon={<ArrowDownward />}
              onClick={() => {
                try {
                  const plugin = documentId ? getLiveEditDiffPlugin(documentId) : null;
                  const view = editorViewRef.current;
                  if (plugin && view) {
                    const cursorPos = view.state.selection.main.head;
                    const nextDiff = plugin.findNextDiff(cursorPos);
                    if (nextDiff) {
                      plugin.jumpToPosition(nextDiff.position);
                    } else {
                      devLog('No next diff found');
                    }
                  } else {
                    console.warn('⚠️ Cannot navigate: plugin or view not available');
                  }
                } catch (err) {
                  console.error('Failed to jump to next diff:', err);
                }
              }}
            >
              Next Edit
            </Button>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button 
              size="small" 
              variant="outlined" 
              color="success"
              onClick={() => {
                try {
                  const plugin = documentId ? getLiveEditDiffPlugin(documentId) : null;
                  if (plugin && plugin.acceptAllOperations) {
                    plugin.acceptAllOperations();
                  } else {
                    console.warn('⚠️ Cannot accept all: plugin not available');
                  }
                } catch (err) {
                  console.error('Failed to accept all operations:', err);
                }
              }}
            >
              Accept All
            </Button>
            <Button 
              size="small" 
              variant="outlined" 
              color="error"
              onClick={() => {
                try {
                  const plugin = documentId ? getLiveEditDiffPlugin(documentId) : null;
                  if (plugin && plugin.rejectAllOperations) {
                    plugin.rejectAllOperations();
                  } else {
                    console.warn('⚠️ Cannot reject all: plugin not available');
                  }
                } catch (err) {
                  console.error('Failed to reject all operations:', err);
                }
              }}
            >
              Reject All
            </Button>
          </Box>
        </Box>
      )}
      <OrgFileLinkDialog
        open={fileLinkDialogOpen}
        onClose={() => setFileLinkDialogOpen(false)}
        onSelect={handleFileLinkSelect}
        currentDocumentPath={canonicalPath}
        linkFormat="markdown"
      />
    </Box>
  );
});

MarkdownCMEditor.displayName = 'MarkdownCMEditor';

export default MarkdownCMEditor;


