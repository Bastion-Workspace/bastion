import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation, Navigate } from 'react-router-dom';
import { Container, Box, IconButton, Tooltip, SwipeableDrawer } from '@mui/material';
import { ChevronLeft } from '@mui/icons-material';
import { motion } from 'framer-motion';
import { QueryClient, QueryClientProvider } from 'react-query';
import NewsPage from './components/NewsPage';
import NewsDetailPage from './components/NewsDetailPage';
import { AuthProvider } from './contexts/AuthContext';
import { CapabilitiesProvider } from './contexts/CapabilitiesContext';
import { EditorProvider } from './contexts/EditorContext';
import { ModelProvider } from './contexts/ModelContext';
import { ChatSidebarProvider, useChatSidebar } from './contexts/ChatSidebarContext';
import { MessagingProvider } from './contexts/MessagingContext';
import { TeamProvider } from './contexts/TeamContext';
import { MusicProvider } from './contexts/MediaContext';
import ModelConfigurationNotification from './components/ModelConfigurationNotification';
import Navigation from './components/Navigation';
import ChatSidebar from './components/ChatSidebar';
import MessagingDrawer from './components/messaging/MessagingDrawer';
import LoginPage from './components/LoginPage';
import ProtectedRoute from './components/ProtectedRoute';
import HomePage from './components/HomePage';
import DocumentsPage from './components/DocumentsPage';
import ChatPage from './components/ChatPage';
import SettingsPage from './components/SettingsPage';
import TeamsPage from './components/teams/TeamsPage';
import TeamDetailPage from './components/teams/TeamDetailPage';
import OrgQuickCapture from './components/OrgQuickCapture';
import StatusBar from './components/StatusBar';
import MediaPage from './components/MediaPage';

import PDFTextLayerEditor from './components/PDFTextLayerEditor';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Main content component that uses the chat sidebar context
const MainContent = () => {
  const location = useLocation();
  const isDocumentsRoute = location.pathname.startsWith('/documents');
  const isMediaRoute = location.pathname.startsWith('/media') || location.pathname.startsWith('/music');
  const { isCollapsed, sidebarWidth, isFullWidth, isResizing, toggleSidebar } = useChatSidebar();
  const isMobile = /Mobi|Android/i.test(navigator.userAgent);
  
  // Quick Capture state and hotkey listener
  const [captureOpen, setCaptureOpen] = useState(false);
  
  // **ROOSEVELT'S EDITOR CONTEXT CLEANUP**: Clear stale editor context when navigating away from Documents
  useEffect(() => {
    if (!isDocumentsRoute) {
      // Not on Documents page - clear the editor context cache so it doesn't interfere with other pages
      try {
        localStorage.removeItem('editor_ctx_cache');
        console.log('🧹 Cleared editor context cache (not on Documents page)');
      } catch (e) {
        console.warn('Failed to clear editor context cache:', e);
      }
    }
  }, [isDocumentsRoute]);
  
  useEffect(() => {
    // Global hotkey listener for Ctrl+Shift+C
    const handleKeyDown = (e) => {
      // Ctrl+Shift+C (or Cmd+Shift+C on Mac)
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        setCaptureOpen(true);
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  return (
    <Box sx={{ 
      display: 'flex', 
      height: { xs: 'calc(var(--appvh, 100vh) - 59px - 32px)', md: 'calc(100dvh - 59px - 32px)' },
      position: 'relative',
      paddingBottom: 'env(safe-area-inset-bottom)',
      width: '100%',
      maxWidth: '100vw',
      overflow: 'hidden'
    }}>
      {/* Main Content Area - Responsive to chat sidebar */}
      <Box sx={{ 
        flexGrow: 1, 
        overflow: 'hidden',
        transition: isResizing ? 'none' : 'margin-right 0.3s ease-in-out',
        marginRight: isCollapsed ? 0 : (isFullWidth ? '100vw' : `${sidebarWidth}px`),
        minWidth: 0, // Allow content to shrink below its natural size
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        maxWidth: { xs: '100vw', md: 'none' }
      }}>
        <Container 
          maxWidth={isDocumentsRoute || isMediaRoute ? false : 'xl'} 
          disableGutters={isDocumentsRoute || isMediaRoute} 
          sx={{ 
            mt: isDocumentsRoute || isMediaRoute ? 0 : 4, 
            mb: isDocumentsRoute || isMediaRoute ? 0 : 4, 
            px: isDocumentsRoute || isMediaRoute ? 0 : undefined,
            flex: 1,
            overflow: isDocumentsRoute || isMediaRoute ? 'hidden' : 'auto',
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, width: '100%' }}
          >
            <Routes>
              <Route path="/" element={<Navigate to="/documents" replace />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/news" element={<NewsPage />} />
              <Route path="/news/:newsId" element={<NewsDetailPage />} />

              <Route path="/chat" element={<ChatPage />} />
              <Route path="/teams" element={<TeamsPage />} />
              <Route path="/teams/:teamId" element={<TeamDetailPage />} />
              <Route path="/pdf-text-editor/:documentId" element={<PDFTextLayerEditor />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/media" element={<MediaPage />} />
              <Route path="/music" element={<MediaPage />} />
            </Routes>
          </motion.div>
        </Container>
      </Box>
      
      {/* Chat Sidebar - Fixed position on the right */}
      <Box sx={{ display: { xs: 'none', md: isCollapsed ? 'none' : 'flex' } }}>
        {/* Desktop fixed chat sidebar */}
        <Box sx={{
          position: 'fixed',
          right: 0,
          top: '61px',
          height: { xs: 'calc(var(--appvh, 100vh) - 61px - 32px)', md: 'calc(100dvh - 61px - 32px)' },
          width: isCollapsed ? 0 : (isFullWidth ? '100vw' : `${sidebarWidth}px`),
          backgroundColor: 'background.paper',
          borderLeft: '1px solid',
          borderColor: 'divider',
          zIndex: 1200,
          boxShadow: 'none',
          transition: isResizing ? 'none' : 'width 0.3s ease-in-out',
          overflow: 'hidden',
          display: isCollapsed ? 'none' : 'flex',
          flexDirection: 'column',
        }}>
          <ChatSidebar />
        </Box>
      </Box>
      
      {/* Mobile Chat Drawer */}
      <SwipeableDrawer
        anchor="right"
        open={!isCollapsed && isMobile}
        onOpen={toggleSidebar}
        onClose={toggleSidebar}
        disableSwipeToOpen={false}
        ModalProps={{ keepMounted: true }}
        PaperProps={{ sx: { width: '100vw', paddingTop: 'env(safe-area-inset-top)', paddingBottom: 'env(safe-area-inset-bottom)' } }}
      >
        <ChatSidebar />
      </SwipeableDrawer>

      {/* Collapsed Chat Sidebar Toggle Button */}
      {isCollapsed && (
        <Box sx={{
          position: 'fixed',
          right: 0,
          top: '50%',
          transform: 'translateY(-50%)',
          zIndex: 1300,
          width: 'auto',
          height: 'auto',
          backgroundColor: 'transparent',
          display: 'block'
        }}>
          <Tooltip title="Open Chat">
            <IconButton
              onClick={toggleSidebar}
              sx={{
                backgroundColor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider',
                borderRight: 'none',
                borderRadius: '8px 0 0 8px',
                boxShadow: '-2px 0 8px rgba(0,0,0,0.1)',
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ChevronLeft />
            </IconButton>
          </Tooltip>
        </Box>
      )}
      
      {/* Org Quick Capture Modal - Available anywhere via Ctrl+Shift+C */}
      <OrgQuickCapture 
        open={captureOpen} 
        onClose={() => setCaptureOpen(false)} 
      />
    </Box>
  );
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <CapabilitiesProvider>
        <ModelProvider>
        <ChatSidebarProvider>
          <MessagingProvider>
          <TeamProvider>
          <MusicProvider>
          <EditorProvider>
          <div className="App">
            <Routes>
              {/* Public route */}
              <Route path="/login" element={<LoginPage />} />
              
              {/* Protected routes */}
              <Route path="/*" element={
                <ProtectedRoute>
                  <Navigation />
                  <MainContent />
                  <StatusBar />
                  <MessagingDrawer />
                  <ModelConfigurationNotification />
                </ProtectedRoute>
              } />
            </Routes>
          </div>
          </EditorProvider>
          </MusicProvider>
          </TeamProvider>
          </MessagingProvider>
        </ChatSidebarProvider>
        </ModelProvider>
        </CapabilitiesProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
