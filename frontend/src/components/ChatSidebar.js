import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  IconButton,
  Typography,
  Paper,
  Divider,
  Tooltip,
  CircularProgress,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
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
  Search,
  MoreVert,
  Add,
} from '@mui/icons-material';
import ChatMessagesArea from './chat/ChatMessagesArea';
import ChatInputArea from './chat/ChatInputArea';
import ArtifactDrawerPanel from './chat/ArtifactDrawerPanel';
import { artifactTypeIcon } from './chat/artifactTypeIcons';
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
    activeArtifact,
    setActiveArtifact,
    artifactHistory,
    revertArtifact,
    artifactCollapsed,
    setArtifactCollapsed,
    editorPreference,
    handleEditorPreferenceChange,
  } = useChatSidebar();
  const { editorState } = useEditor();
  const { darkMode } = useTheme();
  const editorOpen = !!(editorState?.isEditable || editorState?.content);

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
  const [moreMenuAnchor, setMoreMenuAnchor] = useState(null);
  const [tempWidth, setTempWidth] = useState(sidebarWidth); // Local state for resize
  const [artifactPanelWidth, setArtifactPanelWidth] = useState(320);
  const [isArtifactResizing, setIsArtifactResizing] = useState(false);
  const artifactDragRef = useRef({ startX: 0, startW: 320 });
  const artifactSavedSidebarWidthRef = useRef(null);
  const sidebarWidthRef = useRef(sidebarWidth);
  const sidebarRef = useRef(null);
  const chatMessagesAreaRef = useRef(null);
  
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
    sidebarWidthRef.current = sidebarWidth;
  }, [sidebarWidth]);

  useEffect(() => {
    if (activeArtifact && !artifactCollapsed) {
      if (artifactSavedSidebarWidthRef.current === null) {
        const prevWidth = sidebarWidthRef.current;
        artifactSavedSidebarWidthRef.current = prevWidth;
        if (!isFullWidth) {
          const target = Math.min(Math.floor(window.innerWidth * 0.65), 1000);
          const newWidth = Math.max(prevWidth, Math.max(280, target));
          const extraSpace = newWidth - prevWidth;
          setArtifactPanelWidth(Math.max(320, extraSpace));
          setSidebarWidth(newWidth);
        } else {
          setArtifactPanelWidth(Math.max(320, Math.floor(sidebarWidthRef.current * 0.5)));
        }
      }
    } else {
      if (artifactSavedSidebarWidthRef.current !== null) {
        if (!isFullWidth) {
          setSidebarWidth(artifactSavedSidebarWidthRef.current);
        }
        artifactSavedSidebarWidthRef.current = null;
      }
      if (!activeArtifact) {
        setArtifactPanelWidth(320);
      }
    }
  }, [activeArtifact, artifactCollapsed, isFullWidth, setSidebarWidth]);

  useEffect(() => {
    if (!isArtifactResizing) return undefined;
    const onMove = (e) => {
      const el = sidebarRef.current;
      if (!el) return;
      const total = el.clientWidth;
      const dw = e.clientX - artifactDragRef.current.startX;
      const nw = artifactDragRef.current.startW + dw;
      const clamped = Math.max(200, Math.min(nw, total - 200));
      setArtifactPanelWidth(clamped);
    };
    const onUp = () => setIsArtifactResizing(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [isArtifactResizing]);

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

  const handleArtifactResizeStart = useCallback((e) => {
    e.preventDefault();
    artifactDragRef.current = { startX: e.clientX, startW: artifactPanelWidth };
    setIsArtifactResizing(true);
  }, [artifactPanelWidth]);

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

  const closeMoreMenu = () => setMoreMenuAnchor(null);

  const handleMoreFindInChat = () => {
    closeMoreMenu();
    chatMessagesAreaRef.current?.openConversationSearch?.();
  };

  const handleMoreExport = () => {
    closeMoreMenu();
    handleExportConversation();
  };

  const handleMoreFullWidth = () => {
    closeMoreMenu();
    handleFullWidthToggle();
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
                maxWidth: isFullWidth ? 'calc(100vw - 200px)' : sidebarWidth - 160,
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
                maxWidth: isFullWidth ? 'calc(100vw - 200px)' : sidebarWidth - 160, // Leave space for buttons
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
          <Tooltip title="New chat">
            <IconButton
              onClick={handleNewChat}
              size="small"
              color="primary"
              aria-label="New chat"
              sx={{ width: 32, height: 32, mr: 0.75 }}
            >
              <Add fontSize="small" />
            </IconButton>
          </Tooltip>
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

          <Tooltip title="More options">
            <IconButton
              size="small"
              onClick={(e) => setMoreMenuAnchor(e.currentTarget)}
              aria-label="More chat options"
              aria-haspopup="true"
              aria-expanded={Boolean(moreMenuAnchor)}
              sx={{ width: 32, height: 32 }}
            >
              <MoreVert />
            </IconButton>
          </Tooltip>
          <Menu
            anchorEl={moreMenuAnchor}
            open={Boolean(moreMenuAnchor)}
            onClose={closeMoreMenu}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem onClick={handleMoreFindInChat} disabled={!currentConversationId} dense>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Search fontSize="small" />
              </ListItemIcon>
              <ListItemText>Find in chat</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleMoreExport} disabled={!currentConversationId} dense>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <FileDownload fontSize="small" />
              </ListItemIcon>
              <ListItemText>Export conversation</ListItemText>
            </MenuItem>
            <MenuItem onClick={handleMoreFullWidth} dense>
              <ListItemIcon sx={{ minWidth: 36 }}>
                {isFullWidth ? <FullscreenExit fontSize="small" /> : <Fullscreen fontSize="small" />}
              </ListItemIcon>
              <ListItemText>{isFullWidth ? 'Exit full width' : 'Full width'}</ListItemText>
            </MenuItem>
          </Menu>

          <Tooltip title="Collapse Chat">
            <IconButton onClick={toggleSidebar} size="small" sx={{ width: 32, height: 32 }}>
              <ChevronRight />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Box
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'row',
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {activeArtifact && artifactCollapsed && (
          <Box
            onClick={() => setArtifactCollapsed(false)}
            sx={{
              width: 40,
              minWidth: 40,
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              py: 0.5,
              gap: 0.5,
              borderRight: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
              minHeight: 0,
              cursor: 'pointer',
            }}
            aria-label="Expand artifact panel"
          >
            <Tooltip title="Expand panel">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setArtifactCollapsed(false);
                }}
                aria-label="Expand artifact panel"
              >
                <ChevronRight fontSize="small" />
              </IconButton>
            </Tooltip>
            <Box sx={{ py: 0.5, flexShrink: 0 }}>{artifactTypeIcon(activeArtifact.artifact_type)}</Box>
            <Box
              sx={{
                flex: 1,
                minHeight: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                px: 0.25,
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  writingMode: 'vertical-rl',
                  textOrientation: 'mixed',
                  maxHeight: 'min(240px, 40vh)',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  fontWeight: 600,
                  color: 'text.secondary',
                }}
                title={activeArtifact.title || 'Artifact'}
              >
                {activeArtifact.title || 'Artifact'}
              </Typography>
            </Box>
            <Tooltip title="Close artifact">
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  setActiveArtifact(null);
                }}
                aria-label="Close artifact"
              >
                <Close fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        )}
        {activeArtifact && !artifactCollapsed && (
          <Box
            sx={{
              width: artifactPanelWidth,
              minWidth: 200,
              maxWidth: '85%',
              flexShrink: 0,
              display: 'flex',
              flexDirection: 'column',
              minHeight: 0,
              position: 'relative',
            }}
          >
            <ArtifactDrawerPanel
              artifact={activeArtifact}
              artifactHistory={artifactHistory}
              onRevert={revertArtifact}
              onClose={() => setActiveArtifact(null)}
              onCollapse={() => setArtifactCollapsed(true)}
              conversationId={currentConversationId || null}
            />
            <SplitResizeHandle
              edge="trailing"
              isResizing={isArtifactResizing}
              onMouseDown={handleArtifactResizeStart}
            />
          </Box>
        )}
        <Box
          sx={{
            flex: 1,
            minWidth: 0,
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
          }}
        >
          <Box sx={{ flexGrow: 1, overflow: 'hidden', backgroundColor: 'background.default' }}>
            <ChatMessagesArea ref={chatMessagesAreaRef} darkMode={darkMode} />
          </Box>
          <Box sx={{ borderTop: '1px solid', borderColor: 'divider' }}>
            <ChatInputArea />
          </Box>
        </Box>
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