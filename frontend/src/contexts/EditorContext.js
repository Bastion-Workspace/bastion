import React, { createContext, useContext, useState, useMemo, useEffect } from 'react';
import { useLocation } from 'react-router-dom';

// EditorContext holds the active editable tab state for the current page view
// It is intentionally minimal and only populated when an editor is active

const EditorContext = createContext(null);

export const EditorProvider = ({ children }) => {
  const location = useLocation();
  const [editorState, setEditorState] = useState({
    isEditable: false,
    filename: null,
    language: null,
    content: null,
    contentLength: 0,
    frontmatter: null, // { title, type, ... }
    cursorOffset: -1,
    selectionStart: -1,
    selectionEnd: -1,
  });

  // **ROOSEVELT'S NAVIGATION CLEANUP**: Clear editor state when leaving documents page
  useEffect(() => {
    const onDocumentsPage = location.pathname.startsWith('/documents');
    
    // Clear editor state when NOT on documents page to prevent checkbox from hanging around
    if (!onDocumentsPage && editorState.isEditable) {
      console.log('🧹 ROOSEVELT: Leaving documents page - clearing editor state');
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
      });
    }
  }, [location.pathname, editorState.isEditable]);

  const value = useMemo(() => ({ editorState, setEditorState }), [editorState]);

  return (
    <EditorContext.Provider value={value}>
      {children}
    </EditorContext.Provider>
  );
};

export const useEditor = () => {
  const ctx = useContext(EditorContext);
  if (!ctx) {
    throw new Error('useEditor must be used within an EditorProvider');
  }
  return ctx;
};


