import React, { useMemo, useRef, useEffect, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { EditorView, Decoration, keymap, ViewPlugin } from '@codemirror/view';
import { EditorState, StateField, StateEffect } from '@codemirror/state';
import { history, defaultKeymap, historyKeymap } from '@codemirror/commands';
import { foldGutter, foldKeymap } from '@codemirror/language';
import { useTheme } from '../contexts/ThemeContext';
import { Box, IconButton, Tooltip } from '@mui/material';
import { HelpOutline } from '@mui/icons-material';

const TODO_STATES = ['TODO', 'NEXT', 'STARTED', 'WAITING', 'HOLD'];
const DONE_STATES = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];

function buildOrgDecorationsFromView(view) {
  const { doc } = view.state;
  const decos = [];
  const text = doc.toString();
  const lines = text.split('\n');
  let pos = 0;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const start = pos;
    const end = pos + line.length;
    const head = line.match(/^(\*+)\s+(.*)$/);
    if (head) {
      const level = head[1].length;
      decos.push(Decoration.line({ class: `org-heading org-level-${level}` }).range(start));
    }
    const cbMatch = line.match(/^(\s*[-+*]\s+)\[( |x|X|-)\]/);
    if (cbMatch) {
      decos.push(Decoration.mark({ class: 'org-checkbox' }).range(start + cbMatch[1].length, start + cbMatch[0].length));
    }
    pos = end + 1;
  }
  return EditorView.decorations.of(Decoration.set(decos, true));
}

const orgDecorationsPlugin = ViewPlugin.fromClass(class {
  constructor(view) {
    this.decorations = Decoration.none;
    this.updateDecos(view);
  }
  update(update) {
    if (update.docChanged || update.viewportChanged) {
      this.updateDecos(update.view);
    }
  }
  updateDecos(view) {
    const { doc } = view.state;
    const decos = [];
    const text = doc.toString();
    const lines = text.split('\n');
    let pos = 0;
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const start = pos;
      const end = pos + line.length;
      const head = line.match(/^(\*+)\s+(.*)$/);
      if (head) {
        const level = head[1].length;
        decos.push(Decoration.line({ class: `org-heading org-level-${level}` }).range(start));
        // Highlight TODO/DONE keyword at start of headline text
        const rest = head[2] || '';
        const first = (rest.split(/\s+/)[0] || '').toUpperCase();
        if (first) {
          const todoStart = start + head[1].length + 1; // stars + space
          const todoEnd = todoStart + first.length;
          if (TODO_STATES.includes(first)) {
            decos.push(Decoration.mark({ class: 'org-todo-mark' }).range(todoStart, todoEnd));
          } else if (DONE_STATES.includes(first)) {
            decos.push(Decoration.mark({ class: 'org-done-mark' }).range(todoStart, todoEnd));
          }
        }
      }
      const cbMatch = line.match(/^(\s*[-+*]\s+)\[( |x|X|-)\]/);
      if (cbMatch) {
        decos.push(Decoration.mark({ class: 'org-checkbox' }).range(start + cbMatch[1].length, start + cbMatch[0].length));
      }
      // ROOSEVELT'S LINK DECORATIONS: Highlight org-mode links [[link]] and [[link][description]]
      const linkRegex = /\[\[([^\]]+)\](?:\[([^\]]+)\])?\]/g;
      let linkMatch;
      while ((linkMatch = linkRegex.exec(line)) !== null) {
        const linkStart = start + linkMatch.index;
        const linkEnd = linkStart + linkMatch[0].length;
        decos.push(Decoration.mark({ class: 'org-link' }).range(linkStart, linkEnd));
      }
      pos = end + 1;
    }
    this.decorations = Decoration.set(decos, true);
  }
}, { decorations: v => v.decorations });

const createBaseTheme = (darkMode) => EditorView.baseTheme({
  '&': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '.cm-editor': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '.cm-scroller': {
    backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
  },
  '.cm-content': { 
    fontFamily: 'monospace', 
    fontSize: '14px', 
    lineHeight: '1.5',
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
  '.cm-line': {
    caretColor: darkMode ? '#ffffff' : '#000000'
  },
  '.cm-line.org-heading': { fontWeight: '600' },
  '.cm-line.org-level-1': { fontSize: '18px', paddingTop: '6px', paddingBottom: '2px', paddingLeft: '0px' },
  '.cm-line.org-level-2': { fontSize: '16px', paddingTop: '4px', paddingBottom: '2px', paddingLeft: '12px' },
  '.cm-line.org-level-3': { fontSize: '15px', paddingTop: '2px', paddingBottom: '1px', paddingLeft: '24px' },
  '.cm-line.org-level-4': { fontSize: '14px', paddingLeft: '36px' },
  '.cm-line.org-current-heading': { 
    backgroundColor: darkMode ? '#264f78' : '#e3f2fd', 
    borderLeft: darkMode ? '3px solid #90caf9' : '3px solid #1976d2' 
  },
  '.org-checkbox': { 
    backgroundColor: darkMode ? '#424242' : '#e5e7eb', 
    borderRadius: '2px',
    color: darkMode ? '#b3b3b3' : '#212121'
  },
  '.org-todo-mark': { color: darkMode ? '#f44336' : '#c62828', fontWeight: '700' },
  '.org-done-mark': { color: darkMode ? '#66bb6a' : '#2e7d32', fontWeight: '700' },
  '.org-link': { 
    color: darkMode ? '#90caf9' : '#1976d2', 
    textDecoration: 'underline', 
    cursor: 'pointer' 
  },
  '.cm-foldGutter': { width: '16px' },
  '.cm-foldPlaceholder': { 
    backgroundColor: darkMode ? '#424242' : '#eee', 
    border: darkMode ? '1px solid #616161' : '1px solid #ddd', 
    color: darkMode ? '#b3b3b3' : '#888',
    borderRadius: '3px',
    padding: '0 4px',
    fontFamily: 'monospace'
  }
});

const OrgCMEditor = React.forwardRef(({ value, onChange, scrollToLine = null, scrollToHeading = null, initialScrollPosition = 0, onScrollChange }, ref) => {
  const { darkMode } = useTheme();
  const editorRef = useRef(null);
  const [currentHeadingLine, setCurrentHeadingLine] = useState(null);
  const scrollCallbackTimeoutRef = useRef(null);
  const hasRestoredInitialScrollRef = useRef(false);
  
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
    }
  }));
  
  const baseTheme = useMemo(() => createBaseTheme(darkMode), [darkMode]);
  const extensions = useMemo(() => [
    history(),
    keymap.of([...defaultKeymap, ...historyKeymap, ...foldKeymap]),
    orgDecorationsPlugin,
    foldGutter(),
    baseTheme
  ], [baseTheme]);

  // Scroll to line or heading when editor is ready
  useEffect(() => {
    if (!editorRef.current || !value) return;
    
    const scrollTimeout = setTimeout(() => {
      try {
        const view = editorRef.current.view;
        if (!view) return;
        
        if (scrollToHeading) {
          // Find line with matching heading
          console.log('📍 ROOSEVELT: Scrolling org editor to heading:', scrollToHeading);
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
                console.log('✅ ROOSEVELT: Scrolled to heading at line', lineNum);
                return;
              }
            }
          }
          console.warn('⚠️ Heading not found in editor:', scrollToHeading);
        } else if (scrollToLine !== null && scrollToLine > 0) {
          // Scroll to specific line number
          console.log('📍 ROOSEVELT: Scrolling org editor to line:', scrollToLine);
          const lineCount = view.state.doc.lines;
          const targetLine = Math.min(scrollToLine, lineCount);
          
          if (targetLine > 0) {
            const pos = view.state.doc.line(targetLine).from;
            view.dispatch({
              effects: EditorView.scrollIntoView(pos, { y: 'center', yMargin: 100 })
            });
            console.log('✅ ROOSEVELT: Scrolled to line', targetLine);
          }
        }
      } catch (err) {
        console.error('❌ Failed to scroll editor:', err);
      }
    }, 300);
    
    return () => clearTimeout(scrollTimeout);
  }, [value, scrollToLine, scrollToHeading]);
  
  // Restore initial scroll position on mount (for tab switching and page reload)
  useEffect(() => {
    if (!hasRestoredInitialScrollRef.current && editorRef.current && initialScrollPosition > 0) {
      // Wait for editor to be fully initialized
      const restoreTimeout = setTimeout(() => {
        if (editorRef.current?.view) {
          const view = editorRef.current.view;
          const scrollDOM = view.scrollDOM;
          if (scrollDOM) {
            scrollDOM.scrollTop = initialScrollPosition;
            console.log('📜 Restored initial org scroll position:', initialScrollPosition);
            hasRestoredInitialScrollRef.current = true;
          }
        }
      }, 100);
      
      return () => clearTimeout(restoreTimeout);
    }
  }, [initialScrollPosition]);
  
  // Track scroll changes and notify parent (debounced)
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
  }, [onScrollChange]);

  return (
    <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
        <Tooltip title="Org-Mode Help: Headings (*, **, ***), TODO states (TODO/NEXT/WAITING/HOLD; DONE/CANCELLED), Checkboxes (- [ ] item, - [x] done), Properties (:PROPERTIES: ... :END:)">
          <IconButton 
            size="small"
            onClick={() => alert('Org-Mode Help\n\nHeadings: *, **, ***\nTODO states: TODO/NEXT/WAITING/HOLD; DONE/CANCELLED\nCheckboxes: - [ ] item, - [x] done\nProperties: :PROPERTIES: ... :END:')}
          >
            <HelpOutline fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      <CodeMirror
        ref={editorRef}
        value={value}
        height="50vh"
        basicSetup={false}
        extensions={extensions}
        onChange={(val) => onChange && onChange(val)}
      />
    </Box>
  );
});

export default OrgCMEditor;

