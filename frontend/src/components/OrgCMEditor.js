import React, { useMemo, useRef, useEffect, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView, keymap, ViewPlugin } from '@codemirror/view';
import { EditorState } from '@codemirror/state';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { searchKeymap, getSearchQuery, setSearchQuery, openSearchPanel, closeSearchPanel } from '@codemirror/search';
import { useTheme } from '../contexts/ThemeContext';
import { useEditor } from '../contexts/EditorContext';
import { Box, IconButton, Tooltip } from '@mui/material';
import { HelpOutline } from '@mui/icons-material';
import OrgFileLinkDialog from './OrgFileLinkDialog';
import { orgDecorationsPlugin, orgFoldService, createOrgTabKeymap, createBaseTheme, codeFolding, createContentIndentationPlugin, createFoldStatePersistencePlugin } from './OrgEditorPlugins';
import apiService from '../services/apiService';

// Memoize OrgCMEditor to prevent re-renders from parent context updates. darkMode prop ensures re-render when theme toggles (memo compares props only).
const OrgCMEditor = React.memo(React.forwardRef(({ value, onChange, scrollToLine = null, scrollToHeading = null, initialScrollPosition = 0, onScrollChange, canonicalPath, filename, documentId, folderId, onCurrentSectionChange, darkMode: darkModeProp }, ref) => {
  const { darkMode: darkModeContext } = useTheme();
  const darkMode = darkModeProp !== undefined ? darkModeProp : darkModeContext;
  const { setEditorState } = useEditor() || { setEditorState: () => {} };
  const editorRef = useRef(null);
  const [currentHeadingLine, setCurrentHeadingLine] = useState(null);
  const scrollCallbackTimeoutRef = useRef(null);
  const hasRestoredInitialScrollRef = useRef(false);
  const [fileLinkDialogOpen, setFileLinkDialogOpen] = useState(false);
  const [indentContentToHeading, setIndentContentToHeading] = useState(false);
  
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

  const baseTheme = useMemo(() => createBaseTheme(darkMode), [darkMode]);
  const contentIndentationPlugin = useMemo(() => createContentIndentationPlugin(indentContentToHeading), [indentContentToHeading]);
  const foldStatePersistencePlugin = useMemo(() => createFoldStatePersistencePlugin(documentId), [documentId]);
  
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
    ...persistentSearchExt,
    ...(currentSectionTracker ? [currentSectionTracker] : [])
  ], [baseTheme, contentIndentationPlugin, foldStatePersistencePlugin, persistentSearchExt, currentSectionTracker, fileLinkKeymap]);

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
        // Use dynamic import to avoid bundling yaml parser if not needed
        const yaml = require('js-yaml');
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
      <Box sx={{ bgcolor: darkMode ? '#1e1e1e' : '#ffffff', p: 2, borderRadius: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
          <Tooltip title="Org-Mode Help: Headings (*, **, ***), TODO states (TODO/NEXT/WAITING/HOLD; DONE/CANCELLED), Checkboxes (- [ ] item, - [x] done), Progress indicators ([n/m] or [n%] in headings), Properties (:PROPERTIES: ... :END:). Keyboard Shortcuts: Ctrl+Shift+H to fold/unfold headers, Ctrl+Alt+L to insert file link, Ctrl+Shift+T to toggle checkbox, Ctrl+Alt+H to fold all, Ctrl+Alt+Shift+H to unfold all.">
            <IconButton 
              size="small"
              onClick={() => alert('Org-Mode Help\n\nHeadings: *, **, ***\nTODO states: TODO/NEXT/WAITING/HOLD; DONE/CANCELLED\nCheckboxes: - [ ] item, - [x] done, - [-] partially done\nProgress indicators: Add [n/m] or [n%] to headings\nProperties: :PROPERTIES: ... :END:\n\nKeyboard Shortcuts:\nCtrl+Shift+H: Fold/unfold current header\nCtrl+Alt+L: Insert file link\nCtrl+Shift+T: Toggle checkbox at cursor\nCtrl+Alt+H: Fold all headings\nCtrl+Alt+Shift+H: Unfold all headings\n\nFeatures:\n- Parent checkboxes auto-update when children change\n- Progress indicators auto-update when checkboxes toggle\n- Enable "Indent Content to Heading Level" in Settings for visual indentation\n\nFolding (Emacs compatible):\n- #+STARTUP: overview/content/showall/show2levels/etc.\n- :VISIBILITY: folded in PROPERTIES drawer\n- Fold state persists between tab switches and sessions')}
            >
              <HelpOutline fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
        <CodeMirror
          key={darkMode ? 'dark' : 'light'}
          ref={editorRef}
          value={value}
          height="50vh"
          basicSetup={false}
          extensions={extensions}
          onChange={(val) => {
            onChange && onChange(val);
            // Update editor context immediately on change
            if (editorRef.current?.view && filename) {
              const state = editorRef.current.view.state;
              const selection = state.selection.main;
              const fullText = (val || '').replace(/\r\n/g, '\n');
              
              // Parse frontmatter if present
              const parseFrontmatter = (text) => {
                const fmRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
                const match = text.match(fmRegex);
                if (match) {
                  try {
                    const yaml = require('js-yaml');
                    const data = yaml.load(match[1]) || {};
                    return { data, content: match[2] };
                  } catch (e) {
                    return { data: {}, content: text };
                  }
                }
                return { data: {}, content: text };
              };
              
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

