import React, { useMemo, useRef, useEffect, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView, keymap, ViewPlugin } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { searchKeymap, getSearchQuery, setSearchQuery, openSearchPanel, closeSearchPanel } from '@codemirror/search';
import { useTheme } from '../contexts/ThemeContext';
import { useEditor } from '../contexts/EditorContext';
import { Box, Button } from '@mui/material';
import { ArrowUpward, ArrowDownward } from '@mui/icons-material';
import OrgFileLinkDialog from './OrgFileLinkDialog';
import DictationButton from './editor/DictationButton';
import { createLiveEditDiffExtension, getLiveEditDiffPlugin } from './editor/extensions/liveEditDiffExtension';
import { documentDiffStore } from '../services/documentDiffStore';
import { orgDecorationsPlugin, orgFoldService, createOrgTabKeymap, createBaseTheme, codeFolding, createContentIndentationPlugin, createFoldStatePersistencePlugin } from './OrgEditorPlugins';
import apiService from '../services/apiService';
import yaml from 'js-yaml';

// Memoize OrgCMEditor to prevent re-renders from parent context updates. darkMode prop ensures re-render when theme toggles (memo compares props only).
const OrgCMEditor = React.memo(React.forwardRef(({ value, onChange, scrollToLine = null, scrollToHeading = null, initialScrollPosition = 0, onScrollChange, canonicalPath, filename, documentId, folderId, onCurrentSectionChange, darkMode: darkModeProp }, ref) => {
  const { darkMode: darkModeContext, accentId } = useTheme();
  const darkMode = darkModeProp !== undefined ? darkModeProp : darkModeContext;
  const themeSignature = `${darkMode ? 'dark' : 'light'}-${accentId}`;
  const [themeRemountNonce, setThemeRemountNonce] = useState(0);
  const hasMountedRef = useRef(false);
  const { setEditorState } = useEditor() || { setEditorState: () => {} };
  const editorRef = useRef(null);
  const [currentHeadingLine, setCurrentHeadingLine] = useState(null);
  const scrollCallbackTimeoutRef = useRef(null);
  const hasRestoredInitialScrollRef = useRef(false);
  const [fileLinkDialogOpen, setFileLinkDialogOpen] = useState(false);
  const [indentContentToHeading, setIndentContentToHeading] = useState(false);
  const [diffCount, setDiffCount] = useState(0);
  const savedScrollPosRef = useRef(null);
  const shouldRestoreScrollRef = useRef(false);

  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
      return;
    }
    // Force CodeMirror remount so mode/accent updates are applied immediately.
    setThemeRemountNonce((prev) => prev + 1);
  }, [themeSignature]);
  
  // Method to insert text at cursor position
  const insertTextAtCursor = (text) => {
    if (!editorRef.current?.view) return;
    const view = editorRef.current.view;
    const state = view.state;
    const selection = state.selection.main;
    const from = selection.from;
    const to = selection.to;
    
    // Insert text at cursor position
    view.dispatch({
      changes: { from, to, insert: text },
      selection: { anchor: from + text.length }
    });
  };

  // Handle file link selection
  const handleFileLinkSelect = (linkText) => {
    insertTextAtCursor(linkText);
  };

  // Load org settings on mount
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const response = await apiService.get('/api/org/settings');
        if (response.success && response.settings) {
          setIndentContentToHeading(response.settings.display_preferences?.indent_content_to_heading ?? false);
        }
      } catch (err) {
        console.error('Failed to load org settings:', err);
      }
    };
    loadSettings();
  }, []);

  // Track diff count for UI visibility
  useEffect(() => {
    if (!documentId) {
      setDiffCount(0);
      return;
    }
    const updateCount = () => setDiffCount(documentDiffStore.getDiffCount(documentId));
    updateCount();
    documentDiffStore.subscribe(updateCount);
    return () => documentDiffStore.unsubscribe(updateCount);
  }, [documentId]);

  // Expose editor methods to parent via ref
  React.useImperativeHandle(ref, () => ({
    getCurrentLine: () => {
      if (!editorRef.current?.view) return null;
      const view = editorRef.current.view;
      const cursorPos = view.state.selection.main.head;
      const line = view.state.doc.lineAt(cursorPos);
      return line.number;
    },
    getCurrentHeading: () => {
      if (!editorRef.current?.view) return null;
      const view = editorRef.current.view;
      const cursorPos = view.state.selection.main.head;
      const currentLine = view.state.doc.lineAt(cursorPos).number;
      
      // Search backwards from current line to find the heading
      for (let i = currentLine; i >= 1; i--) {
        const line = view.state.doc.line(i);
        const lineText = view.state.sliceDoc(line.from, line.to);
        const headMatch = lineText.match(/^\*+\s+(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)?\s*(.*)$/i);
        if (headMatch) {
          return headMatch[2]?.trim() || lineText.trim();
        }
      }
      return 'Current entry';
    },
    getScrollPosition: () => {
      if (!editorRef.current?.view) return 0;
      const scrollDOM = editorRef.current.view.scrollDOM;
      return scrollDOM ? scrollDOM.scrollTop : 0;
    },
    getSelectedText: () => {
      if (!editorRef.current?.view) return null;
      const view = editorRef.current.view;
      const selection = view.state.selection.main;
      if (selection.from === selection.to) return null;
      return view.state.sliceDoc(selection.from, selection.to);
    },
    scrollToLine: (lineNum) => {
      if (!editorRef.current?.view || !lineNum || lineNum < 1) return;
      try {
        const view = editorRef.current.view;
        const line = view.state.doc.line(lineNum);
        const pos = line.from;
        
        view.dispatch({
          effects: EditorView.scrollIntoView(pos, { y: 'start', yMargin: 100 })
        });
        
        // Add brief highlight effect
        const lineElement = view.domAtPos(pos).node.parentElement?.closest('.cm-line');
        if (lineElement) {
          // Remove previous highlights
          document.querySelectorAll('.org-current-heading').forEach(el => {
            el.classList.remove('org-current-heading');
          });
          
          // Add persistent highlight
          lineElement.classList.add('org-current-heading');
          
          // Flash yellow briefly
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

  const baseTheme = useMemo(() => createBaseTheme(darkMode, accentId), [darkMode, accentId]);
  const contentIndentationPlugin = useMemo(() => createContentIndentationPlugin(indentContentToHeading), [indentContentToHeading]);
  const foldStatePersistencePlugin = useMemo(() => createFoldStatePersistencePlugin(documentId), [documentId]);
  const liveEditDiffExt = useMemo(() => {
    if (!documentId) return [];
    return createLiveEditDiffExtension(documentId);
  }, [documentId]);

  // Extension to track current section
  const currentSectionTracker = useMemo(() => {
    if (!onCurrentSectionChange) return [];
    
    return EditorView.updateListener.of((update) => {
      if (!update.view) return;
      try {
        if (update.selectionSet || update.docChanged) {
          const state = update.state;
          const selection = state.selection.main;
          const cursorOffset = selection.head;
          const docText = state.doc.toString();
          onCurrentSectionChange(docText, cursorOffset);
        }
      } catch (err) {
        console.error('Error tracking current section:', err);
      }
    });
  }, [onCurrentSectionChange]);
  
  const fileLinkKeymap = useMemo(() => keymap.of([
    { key: 'Mod-Alt-l', run: () => { setFileLinkDialogOpen(true); return true; } }
  ]), []);

  const extensions = useMemo(() => [
    history(),
    keymap.of([...defaultKeymap, ...historyKeymap, ...searchKeymap]),
    keymap.of(createOrgTabKeymap()),
    fileLinkKeymap,
    orgDecorationsPlugin,
    codeFolding(),
    orgFoldService,
    EditorView.lineWrapping,
    baseTheme,
    contentIndentationPlugin,
    foldStatePersistencePlugin,
    ...liveEditDiffExt,
    ...persistentSearchExt,
    ...(currentSectionTracker ? [currentSectionTracker] : [])
  ], [baseTheme, contentIndentationPlugin, foldStatePersistencePlugin, persistentSearchExt, currentSectionTracker, fileLinkKeymap, liveEditDiffExt]);

  // Restore scroll position after value changes (from diff accept/reject)
  useEffect(() => {
    if (shouldRestoreScrollRef.current && savedScrollPosRef.current !== null && editorRef.current?.view) {
      requestAnimationFrame(() => {
        if (editorRef.current?.view && savedScrollPosRef.current !== null) {
          editorRef.current.view.scrollDOM.scrollTop = savedScrollPosRef.current;
          shouldRestoreScrollRef.current = false;
          savedScrollPosRef.current = null;
        }
      });
    }
  }, [value]);

  // Scroll to line or heading when editor is ready
  useEffect(() => {
    if (!editorRef.current || !value) return;
    
    const scrollTimeout = setTimeout(() => {
      try {
        const view = editorRef.current.view;
        if (!view) return;
        
        if (scrollToHeading) {
          // Find line with matching heading
          console.log('Scrolling org editor to heading:', scrollToHeading);
          const text = view.state.doc.toString();
          const lines = text.split('\n');
          const headingLower = scrollToHeading.toLowerCase().trim();
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const headMatch = line.match(/^\*+\s+(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)?\s*(.*)$/i);
            if (headMatch) {
              const headingText = headMatch[2].trim().toLowerCase();
              if (headingText === headingLower || line.toLowerCase().includes(headingLower)) {
                // Found the heading! Scroll to this line
                const lineNum = i + 1;
                const pos = view.state.doc.line(lineNum).from;
                view.dispatch({
                  effects: EditorView.scrollIntoView(pos, { y: 'start', yMargin: 100 })
                });
                
                // Set current heading for persistent highlighting
                setCurrentHeadingLine(lineNum);
                
                // Add persistent highlight class and flash yellow
                const lineElement = view.domAtPos(pos).node.parentElement?.closest('.cm-line');
                if (lineElement) {
                  // Remove previous highlights
                  document.querySelectorAll('.org-current-heading').forEach(el => {
                    el.classList.remove('org-current-heading');
                  });
                  
                  // Add persistent highlight
                  lineElement.classList.add('org-current-heading');
                  
                  // Flash yellow briefly
                  lineElement.style.backgroundColor = '#fff3cd';
                  setTimeout(() => {
                    lineElement.style.backgroundColor = '';
                  }, 1000);
                }
                console.log('Scrolled to heading at line', lineNum);
                return;
              }
            }
          }
          console.warn('⚠️ Heading not found in editor:', scrollToHeading);
        } else if (scrollToLine !== null && scrollToLine > 0) {
          // Scroll to specific line number
          console.log('Scrolling org editor to line:', scrollToLine);
          const lineCount = view.state.doc.lines;
          const targetLine = Math.min(scrollToLine, lineCount);
          
          if (targetLine > 0) {
            const pos = view.state.doc.line(targetLine).from;
            view.dispatch({
              effects: EditorView.scrollIntoView(pos, { y: 'center', yMargin: 100 })
            });
            console.log('Scrolled to line', targetLine);
          }
        }
      } catch (err) {
        console.error('❌ Failed to scroll editor:', err);
      }
    }, 300);
    
    return () => clearTimeout(scrollTimeout);
  }, [value, scrollToLine, scrollToHeading]);
  
  // Restore initial scroll position on mount (for tab switching and page reload)
  // Also check localStorage for persistence across sessions
  useEffect(() => {
    if (!hasRestoredInitialScrollRef.current && editorRef.current) {
      // Wait for editor to be fully initialized
      const restoreTimeout = setTimeout(() => {
        if (editorRef.current?.view) {
          const view = editorRef.current.view;
          const scrollDOM = view.scrollDOM;
          if (scrollDOM) {
            // Priority: 1) initialScrollPosition prop, 2) localStorage, 3) 0
            let scrollPos = initialScrollPosition;
            
            // If no initial scroll position provided, try localStorage
            if (scrollPos === 0 && documentId) {
              try {
                const storageKey = `org_scroll_position_${documentId}`;
                const saved = localStorage.getItem(storageKey);
                if (saved) {
                  scrollPos = parseInt(saved, 10) || 0;
                  console.log(`📜 Restored scroll position from localStorage for document ${documentId}:`, scrollPos);
                }
              } catch (err) {
                console.error('Failed to load scroll position from localStorage:', err);
              }
            } else if (scrollPos > 0) {
              console.log('📜 Restored initial org scroll position:', scrollPos);
            }
            
            if (scrollPos > 0) {
              scrollDOM.scrollTop = scrollPos;
            }
            
            hasRestoredInitialScrollRef.current = true;
          }
        }
      }, 100);
      
      return () => clearTimeout(restoreTimeout);
    }
  }, [initialScrollPosition, documentId]);
  
  // Track scroll changes and notify parent (debounced)
  // Also save to localStorage for persistence across sessions
  useEffect(() => {
    if (!editorRef.current || !onScrollChange) return;
    
    const checkAndAttach = () => {
      if (editorRef.current?.view) {
        const scrollDOM = editorRef.current.view.scrollDOM;
        if (scrollDOM) {
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
                
                // Also save to localStorage for persistence across sessions
                if (documentId) {
                  try {
                    const storageKey = `org_scroll_position_${documentId}`;
                    localStorage.setItem(storageKey, scrollTop.toString());
                  } catch (err) {
                    console.error('Failed to save scroll position to localStorage:', err);
                  }
                }
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
        }
      }
      return null;
    };
    
    // Editor might not be ready immediately, try after a short delay
    const timer = setTimeout(() => {
      const cleanup = checkAndAttach();
      if (cleanup) {
        return cleanup;
      }
    }, 100);
    
    return () => {
      clearTimeout(timer);
      if (scrollCallbackTimeoutRef.current) {
        clearTimeout(scrollCallbackTimeoutRef.current);
      }
    };
  }, [onScrollChange, documentId]);

  // Helper function to parse frontmatter (extracted for reuse)
  const parseFrontmatter = React.useCallback((text) => {
    const fmRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
    const match = text.match(fmRegex);
    if (match) {
      try {
        const data = yaml.load(match[1]) || {};
        return { data, content: match[2] };
      } catch (e) {
        return { data: {}, content: text };
      }
    }
    return { data: {}, content: text };
  }, []);

  // Refresh editor cache with current content from CodeMirror document (for ChatSidebar)
  // This reads directly from the editor's document state, not the value prop
  const refreshEditorCacheWithContent = React.useCallback(() => {
    // Check if editor is mounted and available
    if (!editorRef.current?.view || !filename) {
      console.warn('⚠️ Cannot refresh org editor cache: editor not available', {
        hasEditorRef: !!editorRef.current,
        hasView: !!editorRef.current?.view,
        filename: filename
      });
      // Still dispatch event so ChatSidebar doesn't hang
      window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
      return;
    }

    try {
      const view = editorRef.current.view;
      const state = view.state;
      
      if (!state || !state.doc) {
        console.warn('⚠️ Cannot refresh org editor cache: invalid editor state');
        window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
        return;
      }
      
      // Read current content directly from CodeMirror document (most up-to-date)
      const fullText = state.doc.toString().replace(/\r\n/g, '\n');
      
      if (!fullText) {
        console.warn('⚠️ Cannot refresh org editor cache: empty document');
        window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
        return;
      }
      
      // Parse frontmatter if present
      const parsed = parseFrontmatter(fullText);
      const mergedFrontmatter = { ...(parsed.data || {}) };
      
      // Get current cursor position
      const selection = state.selection.main;
      const cursorOffset = selection.head;
      const selectionStart = selection.from;
      const selectionEnd = selection.to;
      
      // Create payload with current editor state
      const payload = {
        isEditable: true,
        filename: filename || 'untitled.org',
        language: 'org',
        content: fullText,
        contentLength: fullText.length,
        frontmatter: mergedFrontmatter,
        cursorOffset: cursorOffset,
        selectionStart: selectionStart,
        selectionEnd: selectionEnd,
        canonicalPath: canonicalPath || null,
        documentId: documentId || null,
        folderId: folderId || null,
      };
      
      // Update React context state
      setEditorState(payload);
      
      // Update localStorage cache immediately (no debounce - this is for ChatSidebar)
      try {
        localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
        console.log('✅ Org editor cache refreshed (for ChatSidebar):', {
          contentLength: fullText.length,
          filename: payload.filename,
          cursorOffset: cursorOffset,
          lastChars: fullText.slice(-100) // Log last 100 chars to verify content
        });
        
        // Dispatch event to notify that cache refresh is complete
        window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
      } catch (e) {
        console.error('Failed to update editor_ctx_cache from refreshEditorCacheWithContent:', e);
        // Still dispatch event even on error (fallback timeout will handle it)
        window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
      }
    } catch (error) {
      console.error('Error refreshing org editor cache:', error);
      // Still dispatch event even on error
      window.dispatchEvent(new CustomEvent('editorCacheRefreshed'));
    }
  }, [filename, canonicalPath, documentId, folderId, setEditorState, parseFrontmatter]);

  // Listen for cache refresh requests (e.g., from ChatSidebar before sending message)
  useEffect(() => {
    const handleRefreshCache = () => {
      // Refresh cache with fresh content from CodeMirror document before chat reads it
      refreshEditorCacheWithContent();
    };
    
    window.addEventListener('refreshEditorCache', handleRefreshCache);
    return () => {
      window.removeEventListener('refreshEditorCache', handleRefreshCache);
    };
  }, [refreshEditorCacheWithContent]);

  // Editor operations apply: listen for codexApplyEditorOps, liveEditAccepted, liveEditRejected, codexRequestEditorContent
  useEffect(() => {
    function sliceHash(s) {
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
      return h.toString(16);
    }

    function applyOperations(e) {
      try {
        const detail = e.detail || {};
        const operations = Array.isArray(detail.operations) ? detail.operations : [];
        if (!operations.length) return;

        const view = editorRef.current?.view;
        if (!view) return;

        const doc = view.state.doc;
        const docText = doc.toString();

        const ops = operations.slice().sort((a, b) => {
          const startDiff = (b.start || 0) - (a.start || 0);
          if (startDiff !== 0) return startDiff;
          const aIsChunk = a.is_text_chunk && a.chunk_index !== undefined;
          const bIsChunk = b.is_text_chunk && b.chunk_index !== undefined;
          if (aIsChunk && bIsChunk) return (a.chunk_index || 0) - (b.chunk_index || 0);
          return 0;
        });

        const changes = [];
        let hasValidChanges = false;

        for (const op of ops) {
          const start = Math.max(0, Math.min(doc.length, Number(op.start || 0)));
          const end = Math.max(start, Math.min(doc.length, Number(op.end || start)));

          if (op.pre_hash && op.pre_hash.length > 0 && start !== end) {
            const currentSlice = docText.slice(start, end);
            const ph = sliceHash(currentSlice);
            if (ph !== op.pre_hash) continue;
          }

          let newText = '';
          if (op.op_type === 'delete_range') {
            newText = '';
          } else if (op.op_type === 'insert_after_heading' || op.op_type === 'insert_after') {
            newText = typeof op.text === 'string' ? op.text : '';
          } else {
            newText = typeof op.text === 'string' ? op.text : '';
          }

          const currentSlice = docText.slice(start, end);
          if (currentSlice !== newText) {
            changes.push({ from: start, to: end, insert: newText });
            hasValidChanges = true;
          }
        }

        if (hasValidChanges && changes.length > 0) {
          if (view.scrollDOM) {
            savedScrollPosRef.current = view.scrollDOM.scrollTop;
            shouldRestoreScrollRef.current = true;
          }
          view.dispatch({
            changes,
            userEvent: 'agent-edit'
          });
          const nextText = view.state.doc.toString();
          if (onChange) {
            onChange(nextText);
            try {
              refreshEditorCacheWithContent();
            } catch (err) {
              console.error('Failed to update cache after operation apply:', err);
            }
          }
        }
      } catch (err) {
        console.error('Failed to apply editor operations:', err);
      }
    }

    function handleLiveEditAccepted(e) {
      try {
        const { operationId, operation } = e.detail || {};
        if (!operation) return;
        const normalizedOp = {
          op_type: operation.op_type || 'replace_range',
          start: Number(operation.start || 0),
          end: Number(operation.end !== undefined ? operation.end : operation.start || 0),
          text: operation.text || ''
        };
        applyOperations({ detail: { operations: [normalizedOp] } });
        window.dispatchEvent(new CustomEvent('removeLiveDiff', { detail: { operationId } }));
      } catch (err) {
        console.error('Failed to handle live edit acceptance:', err);
      }
    }

    function handleLiveEditRejected(e) {
      try {
        const { operationId } = e.detail || {};
        if (!operationId) return;
        let savedScrollPos = null;
        if (editorRef.current?.view?.scrollDOM) {
          savedScrollPos = editorRef.current.view.scrollDOM.scrollTop;
        }
        window.dispatchEvent(new CustomEvent('removeLiveDiff', { detail: { operationId } }));
        if (savedScrollPos !== null && editorRef.current?.view) {
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              if (editorRef.current?.view?.scrollDOM) {
                editorRef.current.view.scrollDOM.scrollTop = savedScrollPos;
              }
            });
          });
        }
      } catch (err) {
        console.error('Failed to handle live edit rejection:', err);
      }
    }

    function provideEditorContent() {
      try {
        const current = (value || '').replace(/\r\n/g, '\n');
        window.dispatchEvent(new CustomEvent('codexProvideEditorContent', { detail: { content: current } }));
      } catch {}
    }

    window.addEventListener('codexApplyEditorOps', applyOperations);
    window.addEventListener('liveEditAccepted', handleLiveEditAccepted);
    window.addEventListener('liveEditRejected', handleLiveEditRejected);
    window.addEventListener('codexRequestEditorContent', provideEditorContent);
    return () => {
      window.removeEventListener('codexApplyEditorOps', applyOperations);
      window.removeEventListener('liveEditAccepted', handleLiveEditAccepted);
      window.removeEventListener('liveEditRejected', handleLiveEditRejected);
      window.removeEventListener('codexRequestEditorContent', provideEditorContent);
    };
  }, [value, onChange, refreshEditorCacheWithContent]);

  // Update editor context ONLY on mount and tab switch (NOT during typing)
  // The cache will be refreshed on-demand when ChatSidebar requests it via refreshEditorCache event
  useEffect(() => {
    if (!value || !filename) return;
    
    const fullText = (value || '').replace(/\r\n/g, '\n');
    
    const parsed = parseFrontmatter(fullText);
    const mergedFrontmatter = { ...(parsed.data || {}) };
    
    // Get cursor position from editor if available
    let cursorOffset = -1;
    let selectionStart = -1;
    let selectionEnd = -1;
    if (editorRef.current?.view) {
      const state = editorRef.current.view.state;
      const selection = state.selection.main;
      cursorOffset = selection.head;
      selectionStart = selection.from;
      selectionEnd = selection.to;
    }
    
    const payload = {
      isEditable: true,
      filename: filename || 'untitled.org',
      language: 'org',
      content: fullText,
      contentLength: fullText.length,
      frontmatter: mergedFrontmatter,
      cursorOffset: cursorOffset,
      selectionStart: selectionStart,
      selectionEnd: selectionEnd,
      canonicalPath: canonicalPath || null,
      documentId: documentId || null,
      folderId: folderId || null,
    };
    
    setEditorState(payload);
    
    // Update localStorage cache (only on mount/tab switch, not during typing)
    try {
      localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
      console.log('✅ Org editor cache updated (tab switch):', {
        contentLength: fullText.length,
        filename: payload.filename,
        cursorOffset: cursorOffset
      });
    } catch (e) {
      console.error('Failed to update editor_ctx_cache from OrgCMEditor:', e);
    }
    
    // CHANGED: Only update on mount/tab switch (filename, documentId, canonicalPath changes)
    // NOT on value changes during typing - cache will be refreshed via refreshEditorCache event
  }, [filename, canonicalPath, documentId, folderId, setEditorState, parseFrontmatter]);

  return (
    <>
      <Box sx={{ bgcolor: 'background.paper', p: 1, borderRadius: 1, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 0.5 }}>
          <DictationButton insertText={insertTextAtCursor} />
        </Box>
        <CodeMirror
          key={`${documentId || 'no-doc'}-${themeSignature}-${themeRemountNonce}`}
          ref={editorRef}
          value={value}
          height="100%"
          style={{ height: '100%', minHeight: 0 }}
          basicSetup={false}
          extensions={extensions}
          onChange={(val) => {
            onChange && onChange(val);
            // Update editor context immediately on change
            if (editorRef.current?.view && filename) {
              const state = editorRef.current.view.state;
              const selection = state.selection.main;
              const fullText = (val || '').replace(/\r\n/g, '\n');

              const parsed = parseFrontmatter(fullText);
              const mergedFrontmatter = { ...(parsed.data || {}) };
              
              const payload = {
                isEditable: true,
                filename: filename || 'untitled.org',
                language: 'org',
                content: fullText,
                contentLength: fullText.length,
                frontmatter: mergedFrontmatter,
                cursorOffset: selection.head,
                selectionStart: selection.from,
                selectionEnd: selection.to,
                canonicalPath: canonicalPath || null,
                documentId: documentId || null,
                folderId: folderId || null,
              };
              
              setEditorState(payload);
              
              // Update localStorage cache
              try {
                localStorage.setItem('editor_ctx_cache', JSON.stringify(payload));
              } catch (e) {
                console.error('Failed to update editor_ctx_cache:', e);
              }
            }
          }}
        />
        </Box>
      </Box>
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
                  const view = editorRef.current?.view;
                  if (plugin && view) {
                    const cursorPos = view.state.selection.main.head;
                    const prevDiff = plugin.findPreviousDiff(cursorPos);
                    if (prevDiff) plugin.jumpToPosition(prevDiff.position);
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
                  const view = editorRef.current?.view;
                  if (plugin && view) {
                    const cursorPos = view.state.selection.main.head;
                    const nextDiff = plugin.findNextDiff(cursorPos);
                    if (nextDiff) plugin.jumpToPosition(nextDiff.position);
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
                  if (plugin?.acceptAllOperations) plugin.acceptAllOperations();
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
                  if (plugin?.rejectAllOperations) plugin.rejectAllOperations();
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
      />
    </>
  );
}), (prevProps, nextProps) => {
  // Custom comparison: only re-render if these specific props change. darkMode included so theme toggle triggers re-render.
  return (
    prevProps.value === nextProps.value &&
    prevProps.scrollToLine === nextProps.scrollToLine &&
    prevProps.scrollToHeading === nextProps.scrollToHeading &&
    prevProps.initialScrollPosition === nextProps.initialScrollPosition &&
    prevProps.canonicalPath === nextProps.canonicalPath &&
    prevProps.filename === nextProps.filename &&
    prevProps.documentId === nextProps.documentId &&
    prevProps.folderId === nextProps.folderId &&
    prevProps.onCurrentSectionChange === nextProps.onCurrentSectionChange &&
    prevProps.darkMode === nextProps.darkMode
  );
  // Note: onChange, onScrollChange, and onCurrentSectionChange are callback functions - we assume they're stable
});

export default OrgCMEditor;

