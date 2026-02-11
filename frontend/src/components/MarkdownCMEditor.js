import React, { useMemo, useEffect, useState, useRef, forwardRef, useImperativeHandle } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView, keymap, Decoration, ViewPlugin } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { markdown, markdownLanguage } from '@codemirror/lang-markdown';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { searchKeymap, getSearchQuery, setSearchQuery, openSearchPanel, closeSearchPanel } from '@codemirror/search';
import { useEditor } from '../contexts/EditorContext';
import { useTheme } from '../contexts/ThemeContext';
import { parseFrontmatter as parseMarkdownFrontmatter } from '../utils/frontmatterUtils';
import { Box, TextField, Button, Tooltip, Drawer, IconButton, Typography, Stack, Switch, FormControlLabel } from '@mui/material';
import { Add, Delete, ArrowUpward, ArrowDownward } from '@mui/icons-material';
import { createGhostTextExtension } from './editor/extensions/ghostTextExtension';
import { createInlineEditSuggestionsExtension } from './editor/extensions/inlineEditSuggestionsExtension';
import { createLiveEditDiffExtension, getLiveEditDiffPlugin } from './editor/extensions/liveEditDiffExtension';
import { editorSuggestionService } from '../services/editor/EditorSuggestionService';
import { documentDiffStore } from '../services/documentDiffStore';
import OrgFileLinkDialog from './OrgFileLinkDialog';

const createMdTheme = (darkMode) => EditorView.baseTheme({
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

const MarkdownCMEditor = forwardRef(({ value, onChange, filename, canonicalPath, documentId, initialScrollPosition = 0, onScrollChange, onCurrentSectionChange }, ref) => {
  const { darkMode } = useTheme();
  
  const [diffCount, setDiffCount] = useState(0);

  // Track diff count for UI visibility
  useEffect(() => {
    if (!documentId) {
      setDiffCount(0);
      return;
    }
    
    const updateCount = () => {
      const count = documentDiffStore.getDiffCount(documentId);
      console.log(`📊 MarkdownCMEditor: Diff count for ${documentId} is ${count}`);
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

  // Expose scrollToLine method to parent via ref
  useImperativeHandle(ref, () => ({
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
  const mdTheme = useMemo(() => createMdTheme(darkMode), [darkMode]);
  const inlineEditExt = useMemo(() => createInlineEditSuggestionsExtension(), []);
  
  // ✅ documentId comes from props, passed down from DocumentViewer
  const liveEditDiffExt = useMemo(() => {
    if (!documentId) {
      console.log('🔍 MarkdownCMEditor: No documentId, skipping liveEditDiffExt');
      return [];
    }
    
    const ext = createLiveEditDiffExtension(documentId);
    console.log('🔍 MarkdownCMEditor: Created liveEditDiffExt for document:', documentId, 'items:', ext?.length);
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

  const extensions = useMemo(() => [
    history(),
    keymap.of([...defaultKeymap, ...historyKeymap, ...searchKeymap]),
    fileLinkKeymap,
    markdown({ base: markdownLanguage }),
    EditorView.lineWrapping,
    mdTheme,
    inlineEditExt,
    liveEditDiffExt,
    ...persistentSearchExt
  ], [mdTheme, inlineEditExt, liveEditDiffExt, persistentSearchExt, fileLinkKeymap]);

  const { setEditorState } = useEditor();
  const [fmOpen, setFmOpen] = useState(false);

  const { data: initialData, lists: initialLists, raw: initialRaw, order: initialOrder, body: initialBody } = useMemo(() => {
    const parsed = parseFrontmatter((value || '').replace(/\r\n/g, '\n'));
    // Debug logging removed for performance - fires on every keystroke
    return parsed;
  }, [value]);
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

  // Restore scroll position after value changes (from diff accept/reject)
  useEffect(() => {
    if (shouldRestoreScrollRef.current && savedScrollPosRef.current !== null && editorViewRef.current) {
      // Use requestAnimationFrame to ensure DOM has updated
      requestAnimationFrame(() => {
        if (editorViewRef.current && savedScrollPosRef.current !== null) {
          const scrollDOM = editorViewRef.current.scrollDOM;
          if (scrollDOM) {
            scrollDOM.scrollTop = savedScrollPosRef.current;
            console.log('📜 Restored scroll position:', savedScrollPosRef.current);
            shouldRestoreScrollRef.current = false;
            savedScrollPosRef.current = null;
          }
        }
      });
    }
  }, [value]);
  
  // Restore initial scroll position on mount (for tab switching and page reload)
  useEffect(() => {
    if (!hasRestoredInitialScrollRef.current && editorViewRef.current && initialScrollPosition > 0) {
      requestAnimationFrame(() => {
        if (editorViewRef.current) {
          const scrollDOM = editorViewRef.current.scrollDOM;
          if (scrollDOM) {
            scrollDOM.scrollTop = initialScrollPosition;
            console.log('📜 Restored initial scroll position:', initialScrollPosition);
            hasRestoredInitialScrollRef.current = true;
          }
        }
      });
    }
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
    const fullText = content !== null ? content : (value || '').replace(/\r\n/g, '\n');
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
        
        console.log('📍 CURSOR POSITION DEBUG: Cache refresh captured cursor at:', {
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
  }, [value, filename, canonicalPath, documentId, setEditorState]);

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
  }, [value, filename, canonicalPath, documentId, setEditorState]);

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
        
        const current = (value || '').replace(/\r\n/g, '\n');
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
          console.log('✅ Applying operations via CodeMirror transactions:', { 
            originalLength: doc.length, 
            operationsCount: operations.length,
            changesCount: changes.length
          });
          
          // Save scroll position before applying changes
          if (view.scrollDOM) {
            savedScrollPosRef.current = view.scrollDOM.scrollTop;
            shouldRestoreScrollRef.current = true;
            console.log('💾 Saved scroll position:', savedScrollPosRef.current);
          }
          
          // Dispatch as single transaction using the changes array directly
          // This integrates edits into native undo/redo history
          view.dispatch({
            changes: changes,
            userEvent: 'agent-edit'
          });
          
          // Get the new document text after transaction
          const nextText = view.state.doc.toString();
          
          // Update React state via onChange to keep it in sync
          if (onChange) {
            onChange(nextText);
            
            // CRITICAL: Force immediate cache update after operations (bypass debounce)
            // This ensures chat messages sent immediately after accepting diffs use fresh content
            // Parse frontmatter here since operations may have changed it
            try {
              refreshEditorCacheWithFrontmatter(nextText);
            } catch (err) {
              console.error('Failed to update cache after operation apply:', err);
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
    
    // Handle live edit acceptance (single operation from inline diff)
    function handleLiveEditAccepted(e) {
      try {
        const { operationId, operation } = e.detail || {};
        if (!operation) {
          console.warn('⚠️ No operation in liveEditAccepted event');
          return;
        }
        
        console.log('✅ Accepting live edit:', { 
          operationId, 
          operation: { 
            op_type: operation.op_type,
            start: operation.start,
            end: operation.end,
            textLength: operation.text?.length,
            textPreview: operation.text?.substring(0, 50) + '...' 
          } 
        });
        
        // Ensure operation has all required fields
        const normalizedOp = {
          op_type: operation.op_type || 'replace_range',
          start: Number(operation.start || 0),
          end: Number(operation.end !== undefined ? operation.end : operation.start || 0),
          text: operation.text || ''
        };
        
        console.log('✅ Normalized operation for apply:', normalizedOp);
        
        const ops = [normalizedOp];
        applyOperations({ detail: { operations: ops } });
        
        // Remove from pending diffs (handled by plugin, but dispatch for consistency)
        window.dispatchEvent(new CustomEvent('removeLiveDiff', { 
          detail: { operationId } 
        }));
      } catch (err) {
        console.error('❌ Failed to handle live edit acceptance:', err);
      }
    }
    
    // Handle live edit rejection (just remove visualization)
    function handleLiveEditRejected(e) {
      try {
        const { operationId } = e.detail || {};
        if (!operationId) return;
        
        // Save scroll position before removing decoration (DOM changes can cause scroll jump)
        let savedScrollPos = null;
        if (editorViewRef.current) {
          const scrollDOM = editorViewRef.current.scrollDOM;
          if (scrollDOM) {
            savedScrollPos = scrollDOM.scrollTop;
            console.log('💾 Saved scroll position for reject:', savedScrollPos);
          }
        }
        
        // Remove visualization
        window.dispatchEvent(new CustomEvent('removeLiveDiff', { 
          detail: { operationId } 
        }));
        
        // Restore scroll position after decoration removal
        // Use double RAF to ensure decoration removal has completed
        if (savedScrollPos !== null && editorViewRef.current) {
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              if (editorViewRef.current) {
                const scrollDOM = editorViewRef.current.scrollDOM;
                if (scrollDOM) {
                  scrollDOM.scrollTop = savedScrollPos;
                  console.log('📜 Restored scroll position after reject:', savedScrollPos);
                }
              }
            });
          });
        }
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
    
    // Refresh cache with fresh frontmatter after applying changes
    setTimeout(() => {
      refreshEditorCacheWithFrontmatter(next);
    }, 0);
  };

  const addEntry = () => setFmEntries(prev => [...prev, { key: '', value: '' }]);
  const removeEntry = (idx) => setFmEntries(prev => prev.filter((_, i) => i !== idx));
  const updateEntry = (idx, field, val) => setFmEntries(prev => prev.map((e, i) => i === idx ? { ...e, [field]: val } : e));

  return (
    <Box sx={{ bgcolor: darkMode ? '#1e1e1e' : '#ffffff', p: 2, borderRadius: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <FormControlLabel control={<Switch size="small" checked={suggestionsEnabled} onChange={(e) => setSuggestionsEnabled(!!e.target.checked)} />} label={<Typography variant="caption">Predictive Suggestions</Typography>} />
        <Tooltip title="Edit frontmatter (stored invisibly at top as YAML)">
          <Button variant="outlined" size="small" onClick={() => {
            // Refresh frontmatter cache when opening drawer
            refreshEditorCacheWithFrontmatter();
            setFmOpen(true);
          }}>
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
      {/* Memoize update listener to avoid reconfiguring extensions on every render. Key includes darkMode so editor remounts when theme changes (CodeMirror applies theme at mount only). */}
        {useMemo(() => (
        <CodeMirror
        key={`${documentId || 'no-doc'}-${darkMode ? 'dark' : 'light'}`}
        value={value}
        height="100%"
        basicSetup={false}
        extensions={[...extensions, frontmatterHider, ...(ghostExt || []), EditorView.updateListener.of((update) => {
          try {
            if (!update.view) return;
            
            // Capture editor view reference for scroll position management
            if (!editorViewRef.current) {
              editorViewRef.current = update.view;
            }
            
            const sel = update.state.selection.main;
            const cursorOffset = sel.head;
            const selectionStart = sel.from;
            const selectionEnd = sel.to;
            const docText = update.state.doc.toString();
            
            // Update current section if callback provided
            if (onCurrentSectionChange && (update.selectionSet || update.docChanged)) {
              onCurrentSectionChange(docText, cursorOffset);
            }
            
            // **CRITICAL FIX**: NEVER re-parse frontmatter after mount!
            // The mount effect already parsed it correctly. Re-parsing during typing often fails
            // because CodeMirror update events fire before document is fully initialized.
            // Always use the cached frontmatter from mount.
            const needsContentUpdate = update.docChanged;
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
              documentId: documentId || null, // ✅ CRITICAL: Include documentId for diff persistence
            };
            
            // **PERFORMANCE FIX**: Don't update React context during typing!
            // ChatSidebar only checks if editor is open (already set on mount via useEffect)
            // When sending messages, chat reads from localStorage, not React context
            // So we ONLY write to localStorage, not to React state
            
            // Throttle localStorage writes for chat to read when sending messages
            if (!window.__editor_ctx_write_ts || Date.now() - window.__editor_ctx_write_ts > 500) {
              window.__editor_ctx_write_ts = Date.now();
              localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
            }
          } catch {}
        })]}
        onChange={(val) => onChange && onChange(val)}
        style={{ height: '60vh' }}
      />), [value, filename, canonicalPath, extensions, frontmatterHider, ghostExt, liveEditDiffExt, setEditorState, onCurrentSectionChange, documentId, darkMode])}
      {/* Removed floating Accept/Dismiss UI */}
      {/* Diff navigation and batch operations */}
      {diffCount > 0 && (
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 1, mt: 1 }}>
          <Box sx={{ display: 'flex', gap: 1 }}>
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
                      console.log('No previous diff found');
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
                      console.log('No next diff found');
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


