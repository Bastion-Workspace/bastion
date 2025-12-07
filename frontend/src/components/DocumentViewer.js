import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  TextField
} from '@mui/material';
import {
  Description,
  CalendarToday,
  Person,
  Category,
  Tag,
  OpenInNew,
  Download,
  Edit,
  Save,
  Visibility,
  FileDownload,
  Schedule
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import apiService from '../services/apiService';
import exportService from '../services/exportService';
import { Dialog, DialogTitle, DialogContent, DialogActions, FormGroup, FormControlLabel, Checkbox, Button, Stack } from '@mui/material';
import OrgRenderer from './OrgRenderer';
import OrgCMEditor from './OrgCMEditor';
import MarkdownCMEditor from './MarkdownCMEditor';
import OrgRefileDialog from './OrgRefileDialog';
import OrgArchiveDialog from './OrgArchiveDialog';
import OrgTagDialog from './OrgTagDialog';
import PDFDocumentViewer from './PDFDocumentViewer';
import AudioPlayer from './AudioPlayer';
import { useEditor } from '../contexts/EditorContext';
import { parseFrontmatter } from '../utils/frontmatterUtils';

const DocumentViewer = ({ documentId, onClose, scrollToLine = null, scrollToHeading = null }) => {
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [saving, setSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const { setEditorState } = useEditor();
  const [exportOpen, setExportOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [epubTitle, setEpubTitle] = useState('');
  const [epubAuthor, setEpubAuthor] = useState('');
  const [epubLanguage, setEpubLanguage] = useState('en');
  const [includeToc, setIncludeToc] = useState(true);
  const [includeCover, setIncludeCover] = useState(true);
  const [splitOnHeadings, setSplitOnHeadings] = useState(true);
  const [splitLevels, setSplitLevels] = useState([1, 2]);
  const contentBoxRef = React.useRef(null);
  const [backlinks, setBacklinks] = useState([]);
  const [loadingBacklinks, setLoadingBacklinks] = useState(false);
  const [refileDialogOpen, setRefileDialogOpen] = useState(false);
  const [refileSourceFile, setRefileSourceFile] = useState('');
  const [refileSourceLine, setRefileSourceLine] = useState(null);
  const [refileSourceHeading, setRefileSourceHeading] = useState('');
  const [archiveDialogOpen, setArchiveDialogOpen] = useState(false);
  const [archiveSourceFile, setArchiveSourceFile] = useState('');
  const [archiveSourceLine, setArchiveSourceLine] = useState(null);
  const [archiveSourceHeading, setArchiveSourceHeading] = useState('');
  const [activeClock, setActiveClock] = useState(null);
  const [checkingClock, setCheckingClock] = useState(true);
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [tagSourceLine, setTagSourceLine] = useState(null);
  const [tagSourceHeading, setTagSourceHeading] = useState('');
  const orgEditorRef = React.useRef(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editingFilename, setEditingFilename] = useState(false);
  const [editedTitle, setEditedTitle] = useState('');
  const [editedFilename, setEditedFilename] = useState('');
  const [updatingMetadata, setUpdatingMetadata] = useState(false);
  const [externalUpdateNotification, setExternalUpdateNotification] = useState(null);

  // Helper functions for unsaved content persistence
  const getUnsavedContentKey = (docId) => `unsaved_content_${docId}`;
  
  const getUnsavedContent = (docId) => {
    try {
      const key = getUnsavedContentKey(docId);
      const saved = localStorage.getItem(key);
      return saved ? saved : null;
    } catch (e) {
      console.error('Failed to get unsaved content:', e);
      return null;
    }
  };
  
  const saveUnsavedContent = (docId, content) => {
    try {
      const key = getUnsavedContentKey(docId);
      if (content !== null && content !== undefined) {
        localStorage.setItem(key, content);
      }
    } catch (e) {
      console.error('Failed to save unsaved content:', e);
    }
  };
  
  const clearUnsavedContent = (docId) => {
    try {
      const key = getUnsavedContentKey(docId);
      localStorage.removeItem(key);
    } catch (e) {
      console.error('Failed to clear unsaved content:', e);
    }
  };

  // Fetch document content (can be called multiple times for refresh)
  const fetchDocument = React.useCallback(async (forceRefresh = false, preserveEditMode = false) => {
    try {
      setLoading(true);
      setError(null);
      
      // Check for unsaved content first (unless forcing refresh)
      if (!forceRefresh) {
        const unsavedContent = getUnsavedContent(documentId);
        if (unsavedContent !== null) {
          // We have unsaved content, but we still need document metadata
          // Fetch metadata only, then use unsaved content
          const response = await apiService.getDocumentContent(documentId);
          const docData = {
            ...response.metadata,
            content: unsavedContent, // Use unsaved content instead of fetched
            chunk_count: response.chunk_count,
            total_length: response.total_length
          };
          setDocument(docData);
          setEditContent(unsavedContent);
          
          // Auto-enter edit mode if applicable (or preserve if already editing)
          const fname = (docData.filename || '').toLowerCase();
          const isUserOwned = !!docData.user_id || docData.collection_type === 'user';
          if (preserveEditMode && isEditing) {
            // Preserve current edit mode
            setIsEditing(true);
          } else if (isUserOwned && (fname.endsWith('.md') || fname.endsWith('.org'))) {
            setIsEditing(true);
            setShowPreview(false);
          }
          
          setLoading(false);
          return;
        }
      }
      
      // No unsaved content or forcing refresh - fetch from API
      const response = await apiService.getDocumentContent(documentId);
      
      const docData = {
        ...response.metadata,
        content: response.content,
        chunk_count: response.chunk_count,
        total_length: response.total_length
      };
      setDocument(docData);
      setEditContent(response.content || '');
      
      // Clear any stale unsaved content when we fetch fresh content
      if (forceRefresh) {
        clearUnsavedContent(documentId);
      }
      
      // Auto-enter edit mode for user-created MD/ORG documents (or preserve if already editing)
      const fname = (docData.filename || '').toLowerCase();
      const isUserOwned = !!docData.user_id || docData.collection_type === 'user';
      
      if (preserveEditMode && isEditing) {
        // Preserve current edit mode when force refreshing (don't change isEditing state)
        console.log('🔄 Preserving edit mode during force refresh');
        // isEditing state will remain unchanged
      } else {
        // Normal auto-enter logic
        console.log('🎯 ROOSEVELT EDIT MODE DEBUG:', {
          filename: docData.filename,
          fname,
          user_id: docData.user_id,
          collection_type: docData.collection_type,
          isUserOwned,
          shouldEdit: isUserOwned && (fname.endsWith('.md') || fname.endsWith('.org'))
        });
        if (isUserOwned && (fname.endsWith('.md') || fname.endsWith('.org'))) {
          console.log('🎯 ROOSEVELT: ENTERING EDIT MODE!');
          setIsEditing(true);
          setShowPreview(false);
        } else if (!preserveEditMode) {
          // If not a user-owned editable file and not preserving, exit edit mode
          setIsEditing(false);
        }
      }
      
    } catch (err) {
      console.error('Failed to fetch document:', err);
      setError('Failed to load document content');
    } finally {
      setLoading(false);
    }
  }, [documentId, isEditing]);

  // Initial document load
  useEffect(() => {
    if (documentId) {
      fetchDocument(false);
    }
  }, [documentId, fetchDocument]);

  // Save unsaved content whenever editContent changes (debounced)
  useEffect(() => {
    if (!documentId || !isEditing || !editContent) return;
    
    // Only save if content differs from saved document content
    const savedContent = document?.content || '';
    if (editContent === savedContent) {
      // Content matches saved version, clear unsaved content
      clearUnsavedContent(documentId);
      return;
    }
    
    // Debounce saving unsaved content
    const timeoutId = setTimeout(() => {
      saveUnsavedContent(documentId, editContent);
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [documentId, isEditing, editContent, document?.content]);

  // WebSocket listener for real-time document updates
  useEffect(() => {
    if (!documentId) return;

    const token = apiService.getToken();
    if (!token) {
      console.error('❌ No authentication token available for document WebSocket');
      return;
    }

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/folders?token=${encodeURIComponent(token)}`;
    let ws = null;

    try {
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('📡 DocumentViewer: Connected to updates WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          
          // Listen for updates to THIS document
          if (update.type === 'document_status_update' && update.document_id === documentId) {
            console.log('🔄 Document updated, refreshing content:', update);
            
            // Always refresh when status is 'completed' (agent finished updating)
            // This ensures users see agent changes in real-time, even if they have unsaved content
            if (update.status === 'completed') {
              console.log('🔄 Auto-refreshing document content (status: completed - agent finished work)...');
              
              // Check if user has unsaved changes
              const hasUnsaved = getUnsavedContent(documentId) !== null;
              if (hasUnsaved) {
                console.log('⚠️ Document has unsaved changes, but refreshing due to agent completion');
                // Show notification that file was updated externally
                setExternalUpdateNotification({
                  message: 'File was updated by agent. Your unsaved changes were overwritten.',
                  timestamp: Date.now()
                });
                // Clear notification after 5 seconds
                setTimeout(() => setExternalUpdateNotification(null), 5000);
              }
              
              // Clear unsaved content since agent update takes precedence
              clearUnsavedContent(documentId);
              
              // Force refresh to get latest content from server, preserving edit mode if user was editing
              fetchDocument(true, true); // preserveEditMode = true
            } else if (update.status === 'processing') {
              // Document is being processed - don't refresh yet, wait for 'completed'
              console.log('⏳ Document is being processed, waiting for completion...');
            } else if (!isEditing) {
              // If not editing and status changed, refresh
              console.log('🔄 Auto-refreshing document content (not editing, status changed)...');
              fetchDocument(true);
            } else {
              // User is editing and status is not completed - don't interrupt them
              console.log('⏸️ User is editing and status is not completed, skipping auto-refresh');
            }
          }
          
          // Listen for document edit proposals
          if (update.type === 'document_edit_proposal' && update.document_id === documentId) {
            console.log('📝 Received edit proposal for this document:', update);
            handleEditProposal(update);
          }
        } catch (err) {
          console.error('❌ Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('❌ DocumentViewer WebSocket error:', err);
      };

      ws.onclose = () => {
        console.log('📡 DocumentViewer: WebSocket connection closed');
      };

    } catch (err) {
      console.error('❌ Failed to establish WebSocket connection:', err);
    }

    // Cleanup on unmount
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [documentId, isEditing, fetchDocument]);

  // Handle edit proposals - convert to inline suggestions
  const handleEditProposal = React.useCallback(async (proposal) => {
    if (!proposal || !isEditing) return; // Only show suggestions in edit mode
    
    try {
      // Fetch current document content to find text positions
      const currentContent = editContent || document?.content || '';
      
      // Convert proposal to inline suggestions
      if (proposal.edit_type === 'content' && proposal.content_edit) {
        const contentEdit = proposal.content_edit;
        
        if (contentEdit.edit_mode === 'append') {
          // Append: show suggestion at end of document
          const from = currentContent.length;
          const to = currentContent.length;
          const original = '';
          const suggested = contentEdit.content;
          
          // Dispatch suggestion event
          window.dispatchEvent(new CustomEvent('inlineEditSuggestion', {
            detail: {
              suggestionId: `suggestion-${proposal.proposal_id}-append`,
              from,
              to,
              original,
              suggested,
              proposalId: proposal.proposal_id,
              onAccept: async (proposalId) => {
                // Apply the edit
                await apiService.applyDocumentEditProposal(proposalId);
              },
              onReject: async (proposalId) => {
                // Reject the edit (could mark as rejected in backend)
                console.log('Rejected proposal:', proposalId);
              }
            }
          }));
        } else if (contentEdit.edit_mode === 'replace') {
          // Replace: find the section to replace
          // For now, show as append if we can't find exact match
          const from = 0;
          const to = currentContent.length;
          const original = currentContent;
          const suggested = contentEdit.content;
          
          window.dispatchEvent(new CustomEvent('inlineEditSuggestion', {
            detail: {
              suggestionId: `suggestion-${proposal.proposal_id}-replace`,
              from,
              to,
              original,
              suggested,
              proposalId: proposal.proposal_id,
              onAccept: async (proposalId) => {
                await apiService.applyDocumentEditProposal(proposalId);
              },
              onReject: async (proposalId) => {
                console.log('Rejected proposal:', proposalId);
              }
            }
          }));
        }
      } else if (proposal.edit_type === 'operations' && proposal.operations) {
        // Operation-based edits: create suggestions for each operation
        proposal.operations.forEach((op, idx) => {
          const suggestionId = `suggestion-${proposal.proposal_id}-op-${idx}`;
          const from = op.start || 0;
          const to = op.end || 0;
          const original = op.original_text || currentContent.slice(from, to);
          const suggested = op.text || '';
          
          window.dispatchEvent(new CustomEvent('inlineEditSuggestion', {
            detail: {
              suggestionId,
              from,
              to,
              original,
              suggested,
              proposalId: proposal.proposal_id,
              onAccept: async (proposalId) => {
                await apiService.applyDocumentEditProposal(proposalId, [idx]);
              },
              onReject: async (proposalId) => {
                console.log('Rejected operation:', proposalId, idx);
              }
            }
          }));
        });
      }
    } catch (err) {
      console.error('❌ Error handling edit proposal:', err);
    }
  }, [documentId, isEditing, editContent, document]);

  // Handle scrolling to specific line or heading
  useEffect(() => {
    if (!document || loading) return;

    // Give the DOM time to render
    const scrollTimeout = setTimeout(() => {
      if (scrollToHeading && contentBoxRef.current) {
        // Scroll to heading in preview pane (for non-edit mode)
        // Edit mode scrolling is handled by OrgCMEditor
        console.log('📍 ROOSEVELT: Scrolling preview to heading:', scrollToHeading);
        try {
          const headingText = scrollToHeading.toLowerCase().trim();
          const allHeadings = contentBoxRef.current.querySelectorAll('[id^="org-heading-"]');
          
          for (const heading of allHeadings) {
            const titleElement = heading.querySelector('span[style*="fontWeight: 600"]');
            if (titleElement && titleElement.textContent.toLowerCase().trim() === headingText) {
              heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
              // Highlight briefly
              const originalBg = heading.style.backgroundColor;
              heading.style.backgroundColor = '#fff3cd';
              setTimeout(() => {
                heading.style.backgroundColor = originalBg;
              }, 1500);
              return;
            }
          }
          console.warn('⚠️ Heading not found in preview:', scrollToHeading);
        } catch (err) {
          console.error('❌ Error scrolling preview:', err);
        }
      } else if (scrollToLine !== null && contentBoxRef.current) {
        // Scroll to specific line number (approximate)
        console.log('📍 ROOSEVELT: Scrolling to line:', scrollToLine);
        const contentBox = contentBoxRef.current;
        const lineHeight = 20; // Approximate line height
        const targetY = scrollToLine * lineHeight;
        contentBox.scrollTo({ top: targetY, behavior: 'smooth' });
      }
    }, 300);

    return () => clearTimeout(scrollTimeout);
  }, [document, loading, scrollToLine, scrollToHeading]);

  // Fetch backlinks for org files
  useEffect(() => {
    const fetchBacklinks = async () => {
      if (!document || !document.filename) return;
      
      const fname = document.filename.toLowerCase();
      if (!fname.endsWith('.org')) return;
      
      try {
        setLoadingBacklinks(true);
        console.log('🔗 ROOSEVELT: Fetching backlinks for', document.filename);
        
        const response = await apiService.get(`/api/org/backlinks?filename=${encodeURIComponent(document.filename)}`);
        
        if (response.success && response.backlinks) {
          setBacklinks(response.backlinks);
          console.log(`✅ ROOSEVELT: Found ${response.backlinks.length} backlinks`);
        }
      } catch (err) {
        console.error('Failed to fetch backlinks:', err);
        // Fail silently - backlinks are supplementary information
      } finally {
        setLoadingBacklinks(false);
      }
    };
    
    fetchBacklinks();
  }, [document]);

  // **ROOSEVELT REFILE HOTKEY!** Ctrl+Shift+M to refile current heading
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check for Ctrl+Shift+M (or Cmd+Shift+M on Mac)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'M') {
        event.preventDefault();
        
        const fname = (document?.filename || '').toLowerCase();
        if (!fname.endsWith('.org')) return;
        
        // Open refile dialog with current position
        console.log('📦 ROOSEVELT: Refile hotkey triggered!');
        
        // Get current cursor position from editor (dynamic!)
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        console.log('📦 ROOSEVELT: Current cursor at line:', currentLine, 'heading:', currentHeading);
        
        // Get relative file path from document
        // Try to construct proper path with folder
        let filePath = document.filename;
        
        // If we have folder info, include it
        if (document.folder_id && document.folder_name) {
          filePath = `${document.folder_name}/${document.filename}`;
        } else {
          // Default to OrgMode folder for org files
          filePath = `OrgMode/${document.filename}`;
        }
        
        console.log('📦 ROOSEVELT: Refile source file path:', filePath);
        
        setRefileSourceFile(filePath);
        setRefileSourceLine(currentLine);
        setRefileSourceHeading(currentHeading);
        setRefileDialogOpen(true);
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading]);

  // **ROOSEVELT ARCHIVE HOTKEY!** Ctrl+Shift+A to archive current heading
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check for Ctrl+Shift+A (or Cmd+Shift+A on Mac)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'A') {
        event.preventDefault();
        
        const fname = (document?.filename || '').toLowerCase();
        if (!fname.endsWith('.org')) return;
        
        // Open archive dialog with current position
        console.log('📦 ROOSEVELT: Archive hotkey triggered!');
        
        // Get current cursor position from editor (dynamic!)
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        console.log('📦 ROOSEVELT: Current cursor at line:', currentLine, 'heading:', currentHeading);
        
        // Get relative file path from document
        let filePath = document.filename;
        
        // If we have folder info, include it
        if (document.folder_id && document.folder_name) {
          filePath = `${document.folder_name}/${document.filename}`;
        } else {
          // Default to OrgMode folder for org files
          filePath = `OrgMode/${document.filename}`;
        }
        
        console.log('📦 ROOSEVELT: Archive source file path:', filePath);
        
        setArchiveSourceFile(filePath);
        setArchiveSourceLine(currentLine);
        setArchiveSourceHeading(currentHeading);
        setArchiveDialogOpen(true);
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading]);

  // **ROOSEVELT CLOCKING HOTKEYS!** Ctrl+Shift+I (clock in) and Ctrl+Shift+O (clock out)
  useEffect(() => {
    const handleKeyDown = async (event) => {
      const fname = (document?.filename || '').toLowerCase();
      if (!fname.endsWith('.org')) return;
      
      // Clock In: Ctrl+Shift+I
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'I') {
        event.preventDefault();
        
        console.log('⏰ ROOSEVELT: Clock in hotkey triggered!');
        
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        let filePath = document.filename;
        if (document.folder_id && document.folder_name) {
          filePath = `${document.folder_name}/${document.filename}`;
        } else {
          filePath = `OrgMode/${document.filename}`;
        }
        
        try {
          const response = await apiService.post('/api/org/clock/in', {
            file_path: filePath,
            line_number: currentLine,
            heading: currentHeading
          });
          
          if (response.success) {
            console.log('✅ Clocked in:', currentHeading);
            alert(`⏰ Clocked in to:\n${currentHeading}`);
            setActiveClock(response);
          } else {
            alert(`⚠️ ${response.message}`);
          }
        } catch (err) {
          console.error('❌ Clock in failed:', err);
          alert(`❌ Clock in failed: ${err.message}`);
        }
      }
      
      // Clock Out: Ctrl+Shift+O
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'O') {
        event.preventDefault();
        
        console.log('⏰ ROOSEVELT: Clock out hotkey triggered!');
        
        try {
          const response = await apiService.post('/api/org/clock/out');
          
          if (response.success) {
            console.log('✅ Clocked out:', response.duration_display);
            alert(`⏰ Clocked out!\n\nDuration: ${response.duration_display}\nTask: ${response.heading}`);
            setActiveClock(null);
            // Refresh file to show LOGBOOK entry
            fetchDocument();
          } else {
            alert(`⚠️ ${response.message}`);
          }
        } catch (err) {
          console.error('❌ Clock out failed:', err);
          alert(`❌ Clock out failed: ${err.message}`);
        }
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading, fetchDocument]);

  // Check for active clock on mount
  useEffect(() => {
    const checkActiveClock = async () => {
      try {
        const response = await apiService.get('/api/org/clock/active');
        if (response.success && response.active_clock) {
          setActiveClock(response.active_clock);
        }
      } catch (err) {
        console.error('❌ Failed to check active clock:', err);
      } finally {
        setCheckingClock(false);
      }
    };
    
    checkActiveClock();
  }, []);

  // Tag hotkey - Ctrl+Shift+E to add tags to current heading
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check for Ctrl+Shift+E (or Cmd+Shift+E on Mac)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'E') {
        event.preventDefault();
        
        const fname = (document?.filename || '').toLowerCase();
        if (!fname.endsWith('.org')) return;
        
        console.log('🏷️ Tag hotkey triggered!');
        
        // Get current cursor position from editor
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        console.log('🏷️ Current cursor at line:', currentLine, 'heading:', currentHeading);
        
        setTagSourceLine(currentLine);
        setTagSourceHeading(currentHeading);
        setTagDialogOpen(true);
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading]);
  
  // Handle refile dialog close
  const handleRefileClose = (result) => {
    setRefileDialogOpen(false);
    
    if (result?.success) {
      // Refresh the document after successful refile
      console.log('✅ ROOSEVELT: Refile completed, refreshing...');
      fetchDocument();
    }
  };

  // Handle archive dialog close
  const handleArchiveClose = (result) => {
    setArchiveDialogOpen(false);
    
    if (result?.success) {
      // Refresh the document after successful archive
      console.log('✅ Archive completed, refreshing...');
      fetchDocument();
    }
  };

  // Handle tag dialog close
  const handleTagClose = () => {
    setTagDialogOpen(false);
    // Note: Tag dialog handles refresh internally via window.location.reload()
  };

  // Publish editor context (must be declared before any early returns)
  useEffect(() => {
    const fname = (document?.filename || '').toLowerCase();
    if (isEditing && fname.endsWith('.md')) {
      // **BULLY!** Parse frontmatter for proper editor context
      const parsed = parseFrontmatter(editContent || '');
      
      // **ROOSEVELT'S FRONTMATTER MERGE**: Merge data and lists so array fields (files, components, etc.) are included!
      const mergedFrontmatter = { ...(parsed.data || {}), ...(parsed.lists || {}) };
      
      setEditorState((prev) => ({
        ...prev,
        isEditable: true,
        filename: document?.filename || null,
        language: 'markdown',
        content: editContent,
        contentLength: (editContent || '').length,
        frontmatter: mergedFrontmatter,
        canonicalPath: document?.canonical_path || null
      }));
    } else {
      setEditorState({
        isEditable: false,
        filename: null,
        language: null,
        content: null,
        contentLength: 0,
        frontmatter: null
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isEditing, editContent, document?.filename]);

  // **ROOSEVELT'S CLEANUP**: Clear editor state when component unmounts (tab closes)
  useEffect(() => {
    return () => {
      console.log('🧹 ROOSEVELT: DocumentViewer unmounting - clearing editor state');
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
      // Also clear the localStorage cache to prevent stale data
      try {
        localStorage.removeItem('editor_ctx_cache');
      } catch (e) {
        console.error('Failed to clear editor_ctx_cache:', e);
      }
    };
  }, [setEditorState]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Handle title click to start editing
  const handleTitleClick = () => {
    setEditedTitle(document.title || document.filename);
    setEditingTitle(true);
  };

  // Handle filename click to start editing
  const handleFilenameClick = () => {
    setEditedFilename(document.filename);
    setEditingFilename(true);
  };

  // Save title changes
  const handleSaveTitle = async () => {
    if (!editedTitle.trim() || editedTitle === (document.title || document.filename)) {
      setEditingTitle(false);
      return;
    }

    try {
      setUpdatingMetadata(true);
      await apiService.updateDocumentMetadata(documentId, { title: editedTitle.trim() });
      console.log('Updated document title');
      
      // Refresh document to get updated data
      await fetchDocument();
      setEditingTitle(false);
    } catch (err) {
      console.error('Failed to update title:', err);
      alert(`Failed to update title: ${err.message}`);
    } finally {
      setUpdatingMetadata(false);
    }
  };

  // Save filename changes
  const handleSaveFilename = async () => {
    if (!editedFilename.trim() || editedFilename === document.filename) {
      setEditingFilename(false);
      return;
    }

    try {
      setUpdatingMetadata(true);
      await apiService.renameDocument(documentId, editedFilename.trim());
      console.log('Renamed document file');
      
      // Refresh document to get updated data
      await fetchDocument();
      setEditingFilename(false);
    } catch (err) {
      console.error('Failed to rename file:', err);
      alert(`Failed to rename file: ${err.message}`);
    } finally {
      setUpdatingMetadata(false);
    }
  };

  // Handle escape key to cancel editing
  const handleKeyDown = (e, type) => {
    if (e.key === 'Escape') {
      if (type === 'title') {
        setEditingTitle(false);
      } else {
        setEditingFilename(false);
      }
    } else if (e.key === 'Enter') {
      if (type === 'title') {
        handleSaveTitle();
      } else {
        handleSaveFilename();
      }
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!document) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">No document selected</Alert>
      </Box>
    );
  }

  const fnameLower = (document.filename || '').toLowerCase();

  // **BULLY!** PDF files get special full-screen viewer treatment!
  if (fnameLower.endsWith('.pdf')) {
    return (
      <PDFDocumentViewer 
        documentId={documentId}
        filename={document.filename}
      />
    );
  }

  // Audio files get audio player treatment
  const audioExtensions = ['.mp3', '.aac', '.wav', '.flac', '.ogg', '.m4a', '.wma', '.opus'];
  const isAudioFile = audioExtensions.some(ext => fnameLower.endsWith(ext));
  
  if (isAudioFile && document) {
    const audioUrl = `/api/documents/${documentId}/file`;
    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
          <Typography variant="h6">{document.filename || document.title || 'Audio File'}</Typography>
          {document.title && document.title !== document.filename && (
            <Typography variant="body2" color="text.secondary">{document.title}</Typography>
          )}
        </Box>
        <Box sx={{ flex: 1, p: 3, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Box sx={{ width: '100%', maxWidth: '800px' }}>
            <AudioPlayer
              src={audioUrl}
              filename={document.filename}
            />
          </Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', overflow: 'hidden' }}>
      {/* Single scroll area inside the viewer */}
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* External Update Notification */}
        {externalUpdateNotification && (
          <Alert 
            severity="info" 
            onClose={() => setExternalUpdateNotification(null)}
            sx={{ 
              mx: 2, 
              mt: 1, 
              mb: 0,
              '& .MuiAlert-message': {
                fontSize: '0.875rem'
              }
            }}
          >
            {externalUpdateNotification.message}
          </Alert>
        )}
        
        {/* Compact Header */}
        <Box sx={{ px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
          <Box sx={{ minWidth: 0, overflow: 'hidden', flex: 1 }}>
            {/* Editable Title */}
            {editingTitle ? (
              <TextField
                value={editedTitle}
                onChange={(e) => setEditedTitle(e.target.value)}
                onBlur={handleSaveTitle}
                onKeyDown={(e) => handleKeyDown(e, 'title')}
                autoFocus
                fullWidth
                size="small"
                disabled={updatingMetadata}
                sx={{ mb: 0.5 }}
                placeholder="Enter title..."
              />
            ) : (
              <Typography 
                variant="subtitle1" 
                onClick={handleTitleClick}
                sx={{ 
                  fontWeight: 600, 
                  mb: 0, 
                  whiteSpace: 'nowrap', 
                  overflow: 'hidden', 
                  textOverflow: 'ellipsis',
                  cursor: 'pointer',
                  '&:hover': {
                    backgroundColor: 'action.hover',
                    borderRadius: 1,
                    px: 0.5
                  }
                }}
                title="Click to edit title"
              >
                {document.title || document.filename}
              </Typography>
            )}
            
            {/* Editable Filename */}
            {editingFilename ? (
              <TextField
                value={editedFilename}
                onChange={(e) => setEditedFilename(e.target.value)}
                onBlur={handleSaveFilename}
                onKeyDown={(e) => handleKeyDown(e, 'filename')}
                autoFocus
                fullWidth
                size="small"
                disabled={updatingMetadata}
                placeholder="Enter filename..."
              />
            ) : (
              <Typography 
                variant="caption" 
                color="text.secondary" 
                onClick={handleFilenameClick}
                sx={{ 
                  whiteSpace: 'nowrap', 
                  overflow: 'hidden', 
                  textOverflow: 'ellipsis',
                  cursor: 'pointer',
                  display: 'block',
                  '&:hover': {
                    backgroundColor: 'action.hover',
                    borderRadius: 1,
                    px: 0.5
                  }
                }}
                title="Click to rename file"
              >
                {document.filename}
              </Typography>
            )}
          </Box>

          {/* **ROOSEVELT ACTIVE CLOCK INDICATOR!** */}
          {activeClock && (
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 1, 
              px: 2, 
              py: 0.5, 
              bgcolor: 'primary.main', 
              color: 'primary.contrastText',
              borderRadius: 1,
              fontSize: '0.875rem'
            }}>
              <Schedule fontSize="small" />
              <Box>
                <Typography variant="caption" sx={{ display: 'block', fontWeight: 600, color: 'inherit' }}>
                  ⏰ Clocked In
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', fontSize: '0.75rem', color: 'inherit', opacity: 0.9 }}>
                  {activeClock.heading} ({activeClock.elapsed_display || '0:00'})
                </Typography>
              </Box>
            </Box>
          )}

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
            {(document.filename && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org'))) && (
              !isEditing ? (
                <Tooltip title="Edit">
                  <IconButton size="small" onClick={() => {
                    setIsEditing(true);
                    const fname = (document.filename || '').toLowerCase();
                    if (fname.endsWith('.md') || fname.endsWith('.org')) {
                      setShowPreview(false);
                    }
                  }}>
                    <Edit fontSize="small" />
                  </IconButton>
                </Tooltip>
              ) : (
                <Tooltip title={saving ? 'Saving...' : 'Save'}>
                  <span>
                    <IconButton size="small" onClick={async () => {
                      if (saving) return;
                      try {
                        setSaving(true);
                        
                        // **ROOSEVELT RECURRING TASKS!** Check for TODO->DONE changes
                        const isOrgFile = (document?.filename || '').toLowerCase().endsWith('.org');
                        let recurringHandled = false;
                        
                        if (isOrgFile && document.content) {
                          // Find lines that changed from TODO to DONE
                          const oldLines = document.content.split('\n');
                          const newLines = editContent.split('\n');
                          
                          for (let i = 0; i < Math.min(oldLines.length, newLines.length); i++) {
                            const oldLine = oldLines[i];
                            const newLine = newLines[i];
                            
                            // Check if line changed from TODO to DONE
                            const wasTodo = /^\*+\s+(TODO|NEXT|STARTED|WAITING)\s+/.test(oldLine);
                            const isDone = /^\*+\s+(DONE|CANCELED|CANCELLED)\s+/.test(newLine);
                            
                            if (wasTodo && isDone) {
                              console.log('🔁 ROOSEVELT: Detected TODO->DONE at line', i + 1);
                              
                              // Build file path
                              let filePath = document.filename;
                              if (document.folder_id && document.folder_name) {
                                filePath = `${document.folder_name}/${document.filename}`;
                              } else {
                                filePath = `OrgMode/${document.filename}`;
                              }
                              
                              try {
                                // Call recurring API
                                const response = await apiService.post('/api/org/recurring/complete', {
                                  file_path: filePath,
                                  line_number: i + 1
                                });
                                
                                if (response.success && response.is_recurring) {
                                  console.log('✅ Recurring task handled:', response.message);
                                  recurringHandled = true;
                                  
                                  // Show notification
                                  alert(`🔁 Recurring Task!\n\n${response.message}\n\nFile will be refreshed.`);
                                }
                              } catch (err) {
                                console.error('❌ Failed to handle recurring task:', err);
                              }
                            }
                          }
                        }
                        
                        // Save content
                        await apiService.updateDocumentContent(document.document_id, editContent);
                        setDocument((prev) => prev ? { ...prev, content: editContent } : prev);
                        
                        // Clear unsaved content after successful save
                        clearUnsavedContent(documentId);
                        
                        // If recurring task was handled, refresh to get updated content
                        if (recurringHandled) {
                          setTimeout(() => {
                            fetchDocument(true); // Force refresh to get updated content
                          }, 500);
                        }
                        
                        // BULLY! Don't automatically switch out of edit mode - let the user continue working!
                      } catch (e) {
                        console.error('Save failed', e);
                        alert('Save failed');
                      } finally {
                        setSaving(false);
                      }
                    }} disabled={saving}>
                      <Save fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              )
            )}
            {isEditing && document.filename && fnameLower.endsWith('.md') && (
              <Tooltip title="Export as EPUB">
                <IconButton size="small" onClick={() => {
                  setEpubTitle((document.title || document.filename || '').replace(/\.[^.]+$/, ''));
                  setEpubAuthor(document.author || '');
                  setExportOpen(true);
                }}>
                  <FileDownload fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            {isEditing && (
              <Tooltip title={showPreview ? 'Hide Preview' : 'Show Preview'}>
                <IconButton size="small" onClick={() => setShowPreview((p) => !p)}>
                  <Visibility fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title="Open in new tab">
              <IconButton size="small">
                <OpenInNew fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Download">
              <IconButton size="small">
                <Download fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Content area (single scroll) */}
        <Box ref={contentBoxRef} sx={{ flex: 1, minHeight: 0, overflow: 'auto', p: 2, backgroundColor: 'background.default' }}>
          <Paper variant="outlined" sx={{ p: 2, backgroundColor: 'grey.50' }}>
            {isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? (
              fnameLower.endsWith('.org') ? (
                showPreview ? (
                  // Split view for OrgMode: Editor + Preview
                  <Box sx={{ display: 'flex', gap: 2, height: '70vh' }}>
                    <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                        Edit Mode
                      </Typography>
                      <Box sx={{ p: 1, height: 'calc(100% - 40px)', overflow: 'auto' }}>
                        <OrgCMEditor 
                          ref={orgEditorRef}
                          value={editContent} 
                          onChange={setEditContent}
                          scrollToLine={scrollToLine}
                          scrollToHeading={scrollToHeading}
                        />
                      </Box>
                    </Box>
                    <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                        Preview
                      </Typography>
                      <Box sx={{ p: 1, height: 'calc(100% - 40px)', overflow: 'auto' }}>
                        <OrgRenderer 
                          content={editContent}
                          onNavigate={async (navInfo) => {
                            if (navInfo.type === 'file') {
                              console.log('🔗 ROOSEVELT: Navigating to file:', navInfo.path);
                              alert(`File navigation coming soon!\nTarget: ${navInfo.path}`);
                            } else if (navInfo.type === 'id') {
                              console.log('🔗 ROOSEVELT: Navigating to ID:', navInfo.id);
                              alert(`ID navigation coming soon!\nTarget ID: ${navInfo.id}`);
                            }
                          }}
                        />
                      </Box>
                    </Box>
                  </Box>
                ) : (
                  // Editor only for OrgMode
                  <OrgCMEditor 
                    ref={orgEditorRef}
                    value={editContent} 
                    onChange={setEditContent}
                    scrollToLine={scrollToLine}
                    scrollToHeading={scrollToHeading}
                  />
                )
              ) : fnameLower.endsWith('.md') ? (
                <MarkdownCMEditor 
                  value={editContent} 
                  onChange={setEditContent} 
                  filename={document.filename}
                  canonicalPath={document.canonical_path}
                />
              ) : (
                <TextField
                  multiline
                  fullWidth
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  sx={{ 
                    '& .MuiInputBase-root': { 
                      fontFamily: 'monospace', 
                      fontSize: 14, 
                      lineHeight: 1.5,
                      minHeight: '60vh',
                      alignItems: 'flex-start'
                    }
                  }}
                />
              )
            ) : document.filename && fnameLower.endsWith('.md') ? (
              <Box sx={{ 
                '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1, fontWeight: 'bold' },
                '& p': { mb: 1.5, lineHeight: 1.6 },
                '& img': { maxWidth: '100%', height: 'auto', borderRadius: 1, my: 2 },
                '& a': { color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } },
                '& blockquote': { borderLeft: 3, borderColor: 'primary.main', pl: 2, ml: 0, my: 2, backgroundColor: 'grey.100', py: 1, pr: 2 },
                '& code': { backgroundColor: 'grey.200', px: 0.5, py: 0.25, borderRadius: 0.5, fontFamily: 'monospace', fontSize: '0.875em' },
                '& pre': { backgroundColor: 'grey.200', p: 2, borderRadius: 1, overflow: 'auto', '& code': { backgroundColor: 'transparent', p: 0 } },
                '& ul, & ol': { pl: 3, mb: 1.5 },
                '& li': { mb: 0.5 },
                '& strong': { fontWeight: 'bold' },
                '& em': { fontStyle: 'italic' }
              }}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}
                >
                  {document.content}
                </ReactMarkdown>
              </Box>
            ) : document.filename && document.filename.endsWith('.org') ? (
              <>
                <Box sx={{ p: 1 }}>
                  <OrgRenderer 
                    content={document.content}
                    onNavigate={async (navInfo) => {
                      if (navInfo.type === 'file') {
                        // Handle file links - search for document by filename
                        try {
                          console.log('🔗 ROOSEVELT: Navigating to file:', navInfo.path);
                          // TODO: Implement file navigation by searching for document
                          // For now, just log the intent
                          alert(`File navigation coming soon!\nTarget: ${navInfo.path}\n\nThis will search for and open the document.`);
                        } catch (err) {
                          console.error('Failed to navigate to file:', err);
                        }
                      } else if (navInfo.type === 'id') {
                        // Handle ID-based links
                        console.log('🔗 ROOSEVELT: Navigating to ID:', navInfo.id);
                        alert(`ID navigation coming soon!\nTarget ID: ${navInfo.id}\n\nThis will navigate to the heading with this ID property.`);
                      }
                    }}
                  />
                </Box>
                
                {/* Backlinks Section */}
                {!isEditing && backlinks.length > 0 && (
                  <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' }}>
                    <Typography variant="h6" sx={{ mb: 1, fontSize: '1rem', fontWeight: 600, color: 'primary.main' }}>
                      🔗 Backlinks ({backlinks.length})
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary', fontStyle: 'italic' }}>
                      Files that link to this document
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {backlinks.map((backlink, index) => (
                        <Paper 
                          key={index}
                          elevation={1}
                          sx={{ 
                            p: 1.5, 
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': { 
                              backgroundColor: '#e3f2fd',
                              transform: 'translateX(4px)',
                              boxShadow: 2
                            }
                          }}
                          onClick={async () => {
                            console.log('🔗 ROOSEVELT: Navigating to backlink:', backlink.filename);
                            try {
                              // Look up document by filename
                              const response = await apiService.get(`/api/org/lookup-document?filename=${encodeURIComponent(backlink.filename)}`);
                              if (response.success && response.document) {
                                // Open the document (this will require parent component support)
                                alert(`Navigation coming soon!\nWill open: ${backlink.filename}\n\nDocument ID: ${response.document.document_id}`);
                              }
                            } catch (err) {
                              console.error('Failed to navigate to backlink:', err);
                              alert(`Failed to navigate to ${backlink.filename}`);
                            }
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                            <Description fontSize="small" sx={{ color: 'primary.main' }} />
                            <Typography variant="body1" sx={{ fontWeight: 500, color: 'primary.main' }}>
                              {backlink.filename}
                            </Typography>
                            {backlink.link_count > 1 && (
                              <Chip 
                                label={`${backlink.link_count} links`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem' }}
                              />
                            )}
                          </Box>
                          <Typography variant="body2" sx={{ color: 'text.secondary', fontSize: '0.85rem', pl: 3.5 }}>
                            {backlink.context}
                          </Typography>
                        </Paper>
                      ))}
                    </Box>
                  </Box>
                )}
              </>
            ) : (
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                {document.content}
              </Typography>
            )}
          </Paper>
        </Box>
      </Box>
      {/* EPUB Export Dialog */}
      <Dialog open={exportOpen} onClose={() => setExportOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Export as EPUB</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label="Title" value={epubTitle} onChange={(e) => setEpubTitle(e.target.value)} fullWidth />
            <TextField label="Author" value={epubAuthor} onChange={(e) => setEpubAuthor(e.target.value)} fullWidth />
            <TextField label="Language" value={epubLanguage} onChange={(e) => setEpubLanguage(e.target.value)} fullWidth />
            <FormGroup>
              <FormControlLabel control={<Checkbox checked={includeToc} onChange={(e) => setIncludeToc(e.target.checked)} />} label="Include Table of Contents" />
              <FormControlLabel control={<Checkbox checked={includeCover} onChange={(e) => setIncludeCover(e.target.checked)} />} label="Include Cover (frontmatter 'cover' or first image)" />
              <FormControlLabel control={<Checkbox checked={splitOnHeadings} onChange={(e) => setSplitOnHeadings(e.target.checked)} />} label="Split on Headings (H1/H2 by default)" />
            </FormGroup>
            {/* Simple split level toggles H1-H6 */}
            <Stack direction="row" spacing={1}>
              {[1,2,3,4,5,6].map((lvl) => (
                <FormControlLabel key={lvl} control={<Checkbox checked={splitLevels.includes(lvl)} onChange={(e) => {
                  const checked = e.target.checked;
                  setSplitLevels((prev) => checked ? Array.from(new Set([...prev, lvl])).sort((a,b)=>a-b) : prev.filter(v => v !== lvl));
                }} />} label={`H${lvl}`} />
              ))}
            </Stack>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportOpen(false)} disabled={exporting}>Cancel</Button>
          <Button variant="contained" onClick={async () => {
            try {
              setExporting(true);
              await exportService.exportMarkdownAsEpub(editContent || document.content || '', {
                includeToc,
                includeCover,
                splitOnHeadings,
                splitOnHeadingLevels: splitLevels,
                metadata: { title: epubTitle || 'Untitled', author: epubAuthor || 'Unknown Author', language: epubLanguage || 'en' },
                headingAlignments: {},
              });
              setExportOpen(false);
            } catch (e) {
              alert('EPUB export failed');
            } finally {
              setExporting(false);
            }
          }} disabled={exporting}>{exporting ? 'Exporting...' : 'Export EPUB'}</Button>
        </DialogActions>
      </Dialog>

      {/* Org Refile Dialog */}
      <OrgRefileDialog
        open={refileDialogOpen}
        onClose={handleRefileClose}
        sourceFile={refileSourceFile}
        sourceLine={refileSourceLine}
        sourceHeading={refileSourceHeading}
      />

      {/* Org Archive Dialog */}
      <OrgArchiveDialog
        open={archiveDialogOpen}
        onClose={handleArchiveClose}
        sourceFile={archiveSourceFile}
        sourceLine={archiveSourceLine}
        sourceHeading={archiveSourceHeading}
        onArchiveComplete={handleArchiveClose}
      />

      {/* Org Tag Dialog */}
      <OrgTagDialog
        open={tagDialogOpen}
        onClose={handleTagClose}
        document={document}
        lineNumber={tagSourceLine}
        currentHeading={tagSourceHeading}
      />
    </Box>
  );
};

export default DocumentViewer;
