import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  IconButton,
  Typography,
  Paper,
  Divider,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  ChevronLeft,
  ChevronRight,
  History,
  Close,
  FileDownload,
  Fullscreen,
  FullscreenExit,
  Edit,
  ToggleOn,
  ToggleOff,
} from '@mui/icons-material';
import ChatMessagesArea from './chat/ChatMessagesArea';
import ChatInputArea from './chat/ChatInputArea';
import { useEditor } from '../contexts/EditorContext';
import { useTheme } from '../contexts/ThemeContext';
import FloatingHistoryWindow from './FloatingHistoryWindow';
import SplitResizeHandle from './common/SplitResizeHandle';
import { useChatSidebar } from '../contexts/ChatSidebarContext';
import { useQuery, useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import exportService from '../services/exportService';

const ChatSidebar = () => {
  const {
    isCollapsed,
    sidebarWidth,
    toggleSidebar,
    setSidebarWidth,
    isFullWidth,
    setIsFullWidth,
    isResizing,
    setIsResizing,
    currentConversationId,
    selectConversation,
    createNewConversation,
    messages,
  } = useChatSidebar();
  const { editorState } = useEditor();
  const { darkMode } = useTheme();
  const editorOpen = !!editorState?.isEditable;
  const { 
    editorPreference, 
    handleEditorPreferenceChange
  } = useChatSidebar();

  const currentDocumentId = editorState?.documentId ?? null;
  const hasPendingEditsForCurrentFile = useMemo(() => {
    if (!currentDocumentId) return false;
    return messages.some(
      (m) =>
        Array.isArray(m.editor_operations) &&
        m.editor_operations.length > 0 &&
        (m.editor_document_id === currentDocumentId || m.editor_document_id == null)
    );
  }, [messages, currentDocumentId]);

  const [historyWindowOpen, setHistoryWindowOpen] = useState(false);
  const [tempWidth, setTempWidth] = useState(sidebarWidth); // Local state for resize
  const sidebarRef = useRef(null);
  
  // Use refs to store stable references to avoid stale closures
  const isResizingRef = useRef(false);
  const tempWidthRef = useRef(sidebarWidth);
  const setSidebarWidthRef = useRef(null); // Initialize as null to avoid function reference issues

  // Update refs when state changes
  useEffect(() => {
    isResizingRef.current = isResizing;
  }, [isResizing]);

  useEffect(() => {
    tempWidthRef.current = tempWidth;
  }, [tempWidth]);

  useEffect(() => {
    // Ensure setSidebarWidth is a function before assigning to ref
    if (typeof setSidebarWidth === 'function') {
      setSidebarWidthRef.current = setSidebarWidth;
    }
  }, [setSidebarWidth]);

  // Get conversation data for dynamic title
  const { data: conversationData, isLoading: conversationTitleLoading } = useQuery(
    ['conversation', currentConversationId],
    () => currentConversationId ? apiService.getConversation(currentConversationId) : null,
    {
      enabled: !!currentConversationId,
      refetchOnWindowFocus: false,
      staleTime: 0, // Always allow cache updates from streaming to show immediately
    }
  );

  // Inline title editing state
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [tempTitle, setTempTitle] = useState('');
  const titleInputRef = useRef(null);

  useEffect(() => {
    if (isEditingTitle && titleInputRef.current) {
      titleInputRef.current.focus();
      titleInputRef.current.select();
    }
  }, [isEditingTitle]);

  const queryClient = useQueryClient();

  const handleStartEditTitle = () => {
    if (!currentConversationId) return;
    const current = conversationData?.conversation?.title || 'Chat';
    setTempTitle(current);
    setIsEditingTitle(true);
  };

  const commitTitleUpdate = async (newTitle) => {
    if (!currentConversationId) {
      setIsEditingTitle(false);
      return;
    }
    const trimmed = (newTitle || '').trim();
    // If empty, keep previous title and just exit edit mode
    if (trimmed.length === 0) {
      setIsEditingTitle(false);
      return;
    }
    // Optimistic update
    const prevData = conversationData;
    queryClient.setQueryData(['conversation', currentConversationId], (old) => {
      if (!old?.conversation) return old;
      return { ...old, conversation: { ...old.conversation, title: trimmed } };
    });
    try {
      await apiService.updateConversation(currentConversationId, trimmed, { title: trimmed });
      queryClient.invalidateQueries(['conversations']);
      queryClient.invalidateQueries(['conversation', currentConversationId]);
    } catch (error) {
      console.error('Failed to rename conversation:', error);
      // Revert on error
      queryClient.setQueryData(['conversation', currentConversationId], prevData);
      alert(error?.response?.data?.detail || error.message || 'Failed to rename conversation');
    } finally {
      setIsEditingTitle(false);
    }
  };

  const handleTitleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitTitleUpdate(tempTitle);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setIsEditingTitle(false);
    }
  };

  // ROOSEVELT'S IMPROVED TITLE DISPLAY: Better feedback during conversation restoration
  const conversationTitle = (() => {
    if (!currentConversationId) return 'Chat';
    if (conversationTitleLoading && !conversationData) return 'Loading conversation...';
    return conversationData?.conversation?.title || 'Chat';
  })();

  // Stable event handlers using refs to avoid stale closures
  const handleMouseMove = useCallback((e) => {
    if (!isResizingRef.current) return;
    
    // Calculate new width based on mouse position from right edge of screen
    const newWidth = window.innerWidth - e.clientX;
    const minWidth = 280;
    const maxWidth = Math.min(800, window.innerWidth * 0.5); // 50% of page width, max 800px
    
    console.log('🔄 Mouse move - clientX:', e.clientX, 'window width:', window.innerWidth, 'new width:', newWidth);
    
    if (newWidth >= minWidth && newWidth <= maxWidth) {
      console.log('🔄 Resizing to:', newWidth);
      setTempWidth(newWidth);
      // Update context width in real-time so sidebar follows mouse cursor
      if (setSidebarWidthRef.current && typeof setSidebarWidthRef.current === 'function') {
        setSidebarWidthRef.current(newWidth);
      } else if (typeof setSidebarWidth === 'function') {
        setSidebarWidth(newWidth);
      }
    } else {
      console.log('🔄 Width out of bounds - min:', minWidth, 'max:', maxWidth);
    }
  }, [setSidebarWidth]); // Add setSidebarWidth for fallback

  const handleMouseUp = useCallback(() => {
    console.log('🔄 Resize ended - final tempWidth:', tempWidthRef.current);
    
    // Stop resizing first to prevent further mouse move events
    setIsResizing(false);
    
    // Width is already updated in real-time during drag, but persist to localStorage on mouse up
    // This ensures we don't write to localStorage excessively during drag
    try {
      console.log('🔄 Persisting final sidebar width to localStorage:', tempWidthRef.current);
      localStorage.setItem('chatSidebarWidth', JSON.stringify(tempWidthRef.current));
    } catch (error) {
      console.error('❌ Error persisting sidebar width:', error);
    }
  }, []); // No dependencies needed since width is already updated in real-time

  useEffect(() => {
    if (isResizing) {
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
      return () => {
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isResizing]);

  // Safety mechanism: if resize state gets stuck, force cleanup after 5 seconds
  useEffect(() => {
    if (isResizing) {
      const timeout = setTimeout(() => {
        console.warn('⚠️ Resize state stuck for 5 seconds, forcing cleanup');
        setIsResizing(false);
      }, 5000);
      
      return () => clearTimeout(timeout);
    }
  }, [isResizing]);

  // Handle resize functionality
  useEffect(() => {
    if (isResizing) {
      console.log('🔄 Adding resize event listeners');
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      
      return () => {
        console.log('🔄 Removing resize event listeners');
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
    
    // Return empty cleanup function when not resizing
    return () => {};
  }, [isResizing, handleMouseMove, handleMouseUp]);
  
  // Update tempWidth when sidebarWidth changes (for initial load)
  useEffect(() => {
    console.log('🔄 Context sidebarWidth changed to:', sidebarWidth);
    setTempWidth(sidebarWidth);
  }, [sidebarWidth]);

  // Cleanup resize event listeners on unmount
  useEffect(() => {
    return () => {
      console.log('🔄 Component unmounting - cleaning up resize state');
      setIsResizing(false);
      // Also remove any lingering event listeners
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [handleMouseMove, handleMouseUp]);

  const handleResizeStart = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('🔄 Resize started at clientX:', e.clientX);
    setTempWidth(sidebarWidth); // Initialize temp width with current width
    setIsResizing(true);
  }, [sidebarWidth]);

  const handleFullWidthToggle = () => {
    setIsFullWidth(!isFullWidth);
  };

  const handleSelectConversation = (conversationId) => {
    selectConversation(conversationId);
    // Only close history window if we're actually selecting a conversation (not clearing it)
    if (conversationId) {
      setHistoryWindowOpen(false);
    }
  };

  const handleNewChat = () => {
    createNewConversation();
    setHistoryWindowOpen(false);
  };

  const handleClearCurrentConversation = () => {
    selectConversation(null);
    // Don't close history window when clearing conversation
  };

  const handleExportConversation = async () => {
    if (!currentConversationId || !conversationData?.conversation) {
      console.warn('No conversation to export');
      return;
    }

    try {
      await exportService.exportConversation(conversationData.conversation, 'pdf');
    } catch (error) {
      console.error('Failed to export conversation:', error);
    }
  };

  if (isCollapsed) {
    return null; // Let parent handle collapsed state
  }

  // Calculate sidebar width based on full-width mode and resize state
  // Always use tempWidth when resizing to prevent jumping
  const currentWidth = isFullWidth ? '100vw' : `${tempWidth}px`;

  return (
    <Box
      ref={sidebarRef}
      sx={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        backgroundColor: 'background.paper',
        userSelect: isResizing ? 'none' : 'auto', // Prevent text selection during resize
      }}
    >
      {/* Header */}
      <Box
        sx={{
          flexShrink: 0,
          boxSizing: 'border-box',
          minHeight: 44,
          height: 44,
          px: 2,
          py: 0,
          borderBottom: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'background.paper',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Tooltip title={conversationTitle} placement="bottom">
          {isEditingTitle ? (
            <input
              ref={titleInputRef}
              value={tempTitle}
              onChange={(e) => setTempTitle(e.target.value)}
              onBlur={() => commitTitleUpdate(tempTitle)}
              onKeyDown={handleTitleKeyDown}
              style={{
                fontSize: '0.875rem',
                fontWeight: 500,
                maxWidth: isFullWidth ? 'calc(100vw - 200px)' : sidebarWidth - 120,
                border: '1px solid var(--mui-palette-divider)',
                borderRadius: 4,
                padding: '4px 6px',
                outline: 'none'
              }}
            />
          ) : (
            <Typography 
              variant="subtitle2" 
              onClick={handleStartEditTitle}
              sx={{ 
                fontSize: '0.875rem',
                fontWeight: 500,
                maxWidth: isFullWidth ? 'calc(100vw - 200px)' : sidebarWidth - 120, // Leave space for buttons
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                cursor: currentConversationId ? 'text' : 'default',
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}
            >
              {conversationTitleLoading && currentConversationId && (
                <CircularProgress size={16} />
              )}
              {conversationTitle}
            </Typography>
          )}
        </Tooltip>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
          {editorOpen && (
            <Tooltip
              title={
                hasPendingEditsForCurrentFile
                  ? 'Pending edits for this file (saved in conversation)'
                  : editorPreference === 'prefer'
                    ? 'Prefer Editor (on)'
                    : 'Prefer Editor (off)'
              }
            >
              <IconButton
                onClick={() => handleEditorPreferenceChange(editorPreference === 'prefer' ? 'ignore' : 'prefer')}
                size="small"
                color={editorPreference === 'prefer' || hasPendingEditsForCurrentFile ? 'primary' : 'default'}
                sx={{
                  width: 32,
                  height: 32,
                  '& .MuiSvgIcon-root': { fontSize: '1.1rem' },
                }}
              >
                {editorPreference === 'prefer' ? <ToggleOn /> : <ToggleOff />}
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Chats">
            <IconButton 
              onClick={() => setHistoryWindowOpen(!historyWindowOpen)}
              size="small"
              color={historyWindowOpen ? 'primary' : 'default'}
              sx={{ width: 32, height: 32 }}
            >
              <History />
            </IconButton>
          </Tooltip>
          
          {currentConversationId && (
            <Tooltip title="Export Conversation">
              <IconButton 
                onClick={handleExportConversation}
                size="small"
                color="default"
                sx={{ width: 32, height: 32 }}
              >
                <FileDownload />
              </IconButton>
            </Tooltip>
          )}
          
          <Tooltip title={isFullWidth ? "Exit Full Width" : "Full Width"}>
            <IconButton 
              onClick={handleFullWidthToggle}
              size="small"
              color={isFullWidth ? 'primary' : 'default'}
              sx={{ width: 32, height: 32 }}
            >
              {isFullWidth ? <FullscreenExit /> : <Fullscreen />}
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Collapse Chat">
            <IconButton onClick={toggleSidebar} size="small" sx={{ width: 32, height: 32 }}>
              <ChevronRight />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Messages Area - darkMode prop ensures re-render when theme toggles (React.memo blocks context-only updates) */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden', backgroundColor: 'background.default' }}>
        <ChatMessagesArea darkMode={darkMode} />
      </Box>

      {/* Input Area */}
      <Box sx={{ borderTop: '1px solid', borderColor: 'divider' }}>
        <ChatInputArea />
      </Box>

      {!isFullWidth && (
        <SplitResizeHandle
          edge="leading"
          isResizing={isResizing}
          onMouseDown={handleResizeStart}
          sx={{ zIndex: 1000 }}
        />
      )}

      {/* Floating History Window */}
      {historyWindowOpen && (
        <FloatingHistoryWindow 
          onClose={() => setHistoryWindowOpen(false)}
          onSelectConversation={handleSelectConversation}
          onNewChat={handleNewChat}
          onClearCurrentConversation={handleClearCurrentConversation}
          activeConversationId={currentConversationId}
          anchorEl={sidebarRef.current}
          anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
          transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        />
      )}
    </Box>
  );
};

export default ChatSidebar; 