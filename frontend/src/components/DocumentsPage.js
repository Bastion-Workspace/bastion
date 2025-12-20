import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  IconButton,
  Divider
} from '@mui/material';
import {
  ChevronRight,
  DragIndicator
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import FileTreeSidebar from './FileTreeSidebar';
import TabbedContentManager from './TabbedContentManager';
import RSSFeedManager from './RSSFeedManager';
import { useAuth } from '../contexts/AuthContext';
import apiService from '../services/apiService';

const DocumentsPage = () => {
  const { user } = useAuth();
  
  // State for sidebar management with persistence
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    // ROOSEVELT: Load sidebar collapsed state from localStorage
    try {
      const saved = localStorage.getItem('sidebarCollapsed');
      return saved ? JSON.parse(saved) : false;
    } catch (error) {
      console.error('Failed to load sidebar collapsed state:', error);
      return false;
    }
  });
  
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    // ROOSEVELT: Load sidebar width from localStorage
    try {
      const saved = localStorage.getItem('sidebarWidth');
      return saved ? parseInt(saved, 10) : 280;
    } catch (error) {
      console.error('Failed to load sidebar width:', error);
      return 280;
    }
  });
  
  const [selectedFolderId, setSelectedFolderId] = useState(() => {
    // ROOSEVELT: Load selected folder from localStorage
    try {
      const saved = localStorage.getItem('selectedFolderId');
      return saved && saved !== 'null' ? saved : null;
    } catch (error) {
      console.error('Failed to load selected folder:', error);
      return null;
    }
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [showFeedManager, setShowFeedManager] = useState(false);
  const [rssFeedContext, setRSSFeedContext] = useState(null);
  const tabbedContentRef = useRef();
  const [isResizing, setIsResizing] = useState(false);
  
  // Min and max sidebar widths
  const MIN_SIDEBAR_WIDTH = 240;
  const MAX_SIDEBAR_WIDTH = 500;
  
  // Mobile detection and responsive behavior
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window !== 'undefined') {
      return window.matchMedia && window.matchMedia('(max-width: 900px)').matches;
    }
    return false;
  });

  // Monitor screen size changes
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.matchMedia && window.matchMedia('(max-width: 900px)').matches;
      setIsMobile(mobile);
      
      // Auto-collapse sidebar on mobile, restore on desktop
      if (mobile && !sidebarCollapsed) {
        setSidebarCollapsed(true);
      }
    };

    checkMobile();
    
    if (window.matchMedia) {
      const mediaQuery = window.matchMedia('(max-width: 900px)');
      if (mediaQuery.addEventListener) {
        mediaQuery.addEventListener('change', checkMobile);
        return () => mediaQuery.removeEventListener('change', checkMobile);
      } else {
        // Fallback for older browsers
        mediaQuery.addListener(checkMobile);
        return () => mediaQuery.removeListener(checkMobile);
      }
    }
  }, []);
  
  // Clear selected folder if it doesn't exist (handles database wipes)
  useEffect(() => {
    if (selectedFolderId) {
      // Check if the folder still exists by trying to fetch its contents
      apiService.getFolderContents(selectedFolderId)
        .catch(() => {
          // Folder doesn't exist, clear the selection and cached data
          console.log('Selected folder no longer exists, clearing selection and cached data');
          setSelectedFolderId(null);
          
          // Clear any cached folder data from localStorage
          try {
            localStorage.removeItem('expandedFolders');
            localStorage.removeItem('rss-tabs');
            localStorage.removeItem('rss-active-tab');
            localStorage.removeItem('selectedFolderId');
            localStorage.removeItem('chatSidebarCurrentConversation');
            localStorage.removeItem('chatSidebarCurrentConversation');
            // Keep sidebar state - that's intentional across database wipes
          } catch (error) {
            console.error('Failed to clear cached folder data:', error);
          }
        });
    }
  }, [selectedFolderId]);
  
  // One-time cleanup on mount to handle database wipes
  useEffect(() => {
    // Check if we have any cached folder data and if the database is fresh
    const hasCachedData = localStorage.getItem('expandedFolders') || 
                          localStorage.getItem('rss-tabs') || 
                          localStorage.getItem('rss-active-tab');
    
    if (hasCachedData) {
      // Try to fetch the folder tree to see if the database is fresh
      apiService.getFolderTree()
        .then((folderTree) => {
          // If we get a fresh folder tree, clear any cached folder data that might be stale
          if (folderTree && folderTree.length === 0) {
            console.log('Fresh database detected, clearing cached folder data');
            localStorage.removeItem('expandedFolders');
            localStorage.removeItem('rss-tabs');
            localStorage.removeItem('rss-active-tab');
            localStorage.removeItem('selectedFolderId');
            // NOTE: Don't clear chatSidebarCurrentConversation - conversations are stored separately from documents
          }
        })
        .catch(() => {
          // If we can't fetch the folder tree, clear cached data to be safe
          console.log('Cannot fetch folder tree, clearing cached data');
          localStorage.removeItem('expandedFolders');
          localStorage.removeItem('rss-tabs');
          localStorage.removeItem('rss-active-tab');
          localStorage.removeItem('selectedFolderId');
        });
    }
  }, []);
  
  // ROOSEVELT: Persist sidebar state changes to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('sidebarCollapsed', JSON.stringify(sidebarCollapsed));
      console.log('💾 Persisted sidebar collapsed state:', sidebarCollapsed);
    } catch (error) {
      console.error('Failed to persist sidebar collapsed state:', error);
    }
  }, [sidebarCollapsed]);
  
  useEffect(() => {
    try {
      localStorage.setItem('sidebarWidth', sidebarWidth.toString());
      console.log('💾 Persisted sidebar width:', sidebarWidth);
    } catch (error) {
      console.error('Failed to persist sidebar width:', error);
    }
  }, [sidebarWidth]);
  
  useEffect(() => {
    try {
      if (selectedFolderId) {
        localStorage.setItem('selectedFolderId', selectedFolderId);
        console.log('💾 Persisted selected folder:', selectedFolderId);
      } else {
        localStorage.removeItem('selectedFolderId');
        console.log('💾 Cleared selected folder from persistence');
      }
    } catch (error) {
      console.error('Failed to persist selected folder:', error);
    }
  }, [selectedFolderId]);
  
  // Handlers
  const toggleSidebar = useCallback(() => {
    setSidebarCollapsed(prev => !prev);
  }, []);
  
  // ROOSEVELT: Sidebar resize functionality
  const handleResizeStart = useCallback((e) => {
    e.preventDefault();
    setIsResizing(true);
  }, []);
  
  const handleResizeMove = useCallback((e) => {
    if (!isResizing) return;
    
    const newWidth = e.clientX;
    if (newWidth >= MIN_SIDEBAR_WIDTH && newWidth <= MAX_SIDEBAR_WIDTH) {
      setSidebarWidth(newWidth);
    }
  }, [isResizing, MIN_SIDEBAR_WIDTH, MAX_SIDEBAR_WIDTH]);
  
  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
  }, []);
  
  // Add global mouse event listeners for resize
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleResizeMove);
      document.addEventListener('mouseup', handleResizeEnd);
      document.body.style.cursor = 'ew-resize';
      document.body.style.userSelect = 'none';
      
      return () => {
        document.removeEventListener('mousemove', handleResizeMove);
        document.removeEventListener('mouseup', handleResizeEnd);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isResizing, handleResizeMove, handleResizeEnd]);
  
  const handleFolderSelect = useCallback((folderId) => {
    setSelectedFolderId(folderId);
    setSelectedFile(null); // Clear file selection when folder changes
  }, []);
  
  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file);
    // Open file in the tabbed content manager
    if (tabbedContentRef.current && tabbedContentRef.current.openDocument) {
      tabbedContentRef.current.openDocument(file.document_id, file.filename || file.title);
    }
  }, []);

  const handleRSSFeedClick = useCallback((feedId, feedName) => {
    // Open the feed in the tabbed content manager
    if (tabbedContentRef.current && tabbedContentRef.current.openRSSFeed) {
      tabbedContentRef.current.openRSSFeed(feedId, feedName);
    }
  }, []);

  const handleAddRSSFeed = useCallback((context = null) => {
    setShowFeedManager(true);
    // Store the context for the RSS feed manager
    if (context) {
      setRSSFeedContext(context);
    }
  }, []);

  // ROOSEVELT: Expose TabbedContentManager ref globally for FileTreeSidebar to access
  useEffect(() => {
    window.tabbedContentManagerRef = tabbedContentRef.current;
    return () => {
      window.tabbedContentManagerRef = null;
    };
  }, [tabbedContentRef.current]);

  const handleFeedAdded = useCallback((newFeed) => {
    setShowFeedManager(false);
    // The FileTreeSidebar will automatically refresh its feed list
    console.log('New feed added:', newFeed);
  }, []);

  // Bridge: listen for sidebar request to open News headlines tab
  useEffect(() => {
    const handler = () => {
      try { tabbedContentRef.current?.openNewsHeadlines?.(); } catch {}
    };
    window.addEventListener('openNewsHeadlines', handler);
    return () => window.removeEventListener('openNewsHeadlines', handler);
  }, []);

  return (
    <Box sx={{ 
      display: 'flex', 
      height: { xs: 'calc(var(--appvh, 100vh) - 59px - 32px)', md: 'calc(100dvh - 59px - 32px)' },
      overflow: 'hidden',
      paddingBottom: 'env(safe-area-inset-bottom)'
    }}>
      {/* Left Sidebar - File Tree with RSS Integration */}
      {!sidebarCollapsed && (
        <Box
          sx={{
            position: isMobile ? 'fixed' : 'relative',
            width: isMobile ? { xs: '280px', sm: `${Math.min(sidebarWidth, 320)}px` } : `${sidebarWidth}px`,
            maxWidth: isMobile ? '85vw' : 'none',
            left: 0,
            top: 0,
            bottom: 0,
            zIndex: isMobile ? 1300 : 'auto',
            backgroundColor: 'background.paper',
            borderRight: '1px solid',
            borderColor: 'divider',
            display: 'flex',
            flexDirection: 'column',
            flexShrink: 0,
            boxShadow: isMobile ? '4px 0 12px rgba(0,0,0,0.15)' : '2px 0 8px rgba(0,0,0,0.1)',
            transition: 'width 0.2s ease-in-out, transform 0.2s ease-in-out',
            overflow: 'hidden',
            height: '100%'
          }}
        >
          <FileTreeSidebar
            selectedFolderId={selectedFolderId}
            onFolderSelect={handleFolderSelect}
            onFileSelect={handleFileSelect}
            onRSSFeedClick={handleRSSFeedClick}
            onAddRSSFeed={handleAddRSSFeed}
            width={sidebarWidth}
            isCollapsed={sidebarCollapsed}
            onToggleCollapse={toggleSidebar}
          />
          
          {/* ROOSEVELT: Resize Handle - Hidden on mobile */}
          {!isMobile && (
            <Box
              onMouseDown={handleResizeStart}
              sx={{
                position: 'absolute',
                right: -2,
                top: 0,
                bottom: 0,
                width: 4,
                cursor: 'ew-resize',
                backgroundColor: isResizing ? 'primary.main' : 'transparent',
                '&:hover': {
                  backgroundColor: 'primary.light',
                },
                zIndex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
            <DragIndicator 
              sx={{ 
                color: 'text.secondary', 
                fontSize: 16,
                transform: 'rotate(90deg)',
                opacity: isResizing ? 1 : 0,
                transition: 'opacity 0.2s'
              }} 
            />
          </Box>
          )}
        </Box>
      )}

      {/* Sidebar collapsed toggle button - floats when collapsed */}
      {sidebarCollapsed && (
        <Box sx={{
          position: 'fixed',
          left: 0,
          top: '50%',
          transform: 'translateY(-50%)',
          zIndex: 1200,
          width: 'auto',
          height: 'auto',
          backgroundColor: 'transparent'
        }}>
          <IconButton
            onClick={toggleSidebar}
            size="small"
            sx={{
              backgroundColor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
              borderLeft: 'none',
              borderRadius: '0 8px 8px 0',
              boxShadow: '0 2px 6px rgba(0,0,0,0.1)'
            }}
          >
            <ChevronRight />
          </IconButton>
        </Box>
      )}
      
      {/* Main Content Area - Tabbed Content Manager */}
      <Box sx={{ 
        flexGrow: 1, 
        display: 'flex', 
        flexDirection: 'column',
        position: 'relative',
        height: '100%',
        width: '100%',
        maxWidth: '100vw',
        overflow: 'hidden'
      }}>
        {/* Tabbed Content Area */}
        <Box sx={{ 
          flexGrow: 1, 
          overflow: 'hidden',
          backgroundColor: 'background.default'
        }}>
          <TabbedContentManager ref={tabbedContentRef} />
        </Box>
      </Box>

      {/* RSS Feed Manager Modal */}
      <RSSFeedManager
        isOpen={showFeedManager}
        onClose={() => {
          setShowFeedManager(false);
          setRSSFeedContext(null); // Clear context when closing
        }}
        onFeedAdded={handleFeedAdded}
        feedContext={rssFeedContext}
      />
    </Box>
  );
};

export default DocumentsPage;
