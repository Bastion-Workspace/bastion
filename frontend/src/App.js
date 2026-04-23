import React, { useState, useEffect } from 'react';
import { Routes, Route, useLocation, Navigate, useParams } from 'react-router-dom';
import { Container, Box, IconButton, Tooltip, SwipeableDrawer, useTheme } from '@mui/material';
import { ChevronRight } from '@mui/icons-material';
import { motion } from 'framer-motion';
import { QueryClient, QueryClientProvider, useQuery } from 'react-query';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import AuthQueryCacheBoundary from './components/AuthQueryCacheBoundary';
import { VoiceAvailabilityProvider } from './contexts/VoiceAvailabilityContext';
import { CapabilitiesProvider } from './contexts/CapabilitiesContext';
import { EditorProvider } from './contexts/EditorContext';
import { ModelProvider } from './contexts/ModelContext';
import { ChatSidebarProvider, useChatSidebar } from './contexts/ChatSidebarContext';
import { MessagingProvider } from './contexts/MessagingContext';
import { NotificationProvider } from './contexts/NotificationContext';
import { TeamExecutionProvider } from './contexts/TeamExecutionContext';
import { TeamProvider } from './contexts/TeamContext';
import { MusicProvider } from './contexts/MediaContext';
import { VideoProvider } from './contexts/VideoContext';
import { ControlPaneProvider } from './contexts/ControlPaneContext';
import { ArtifactInstanceProvider } from './contexts/ArtifactInstanceContext';
import { ImageLightboxProvider } from './components/common/ImageLightbox';
import { LearningProvider } from './contexts/LearningContext';
import LearningQuizOverlay from './components/LearningQuizOverlay';
import ModelConfigurationNotification from './components/ModelConfigurationNotification';
import Navigation from './components/Navigation';
import ChatSidebar from './components/ChatSidebar';
import MessagingDrawer from './components/messaging/MessagingDrawer';
import LoginPage from './components/LoginPage';
import ProtectedRoute from './components/ProtectedRoute';
import HomeDashboardPage from './components/HomeDashboardPage';
import DocumentsPage from './components/DocumentsPage';
import CodeSpacesPage from './components/CodeSpacesPage';
import ChatPage from './components/ChatPage';
import SettingsPage from './components/SettingsPage';
import ControlPanesPage from './components/ControlPanesPage';
import TeamsPage from './components/teams/TeamsPage';
import TeamDetailPage from './components/teams/TeamDetailPage';
import OrgQuickCapture from './components/OrgQuickCapture';
import JournalDayModal from './components/JournalDayModal';
import StatusBar from './components/StatusBar';
import MediaPage from './components/MediaPage';
import MapPage from './components/maps/MapPage';
import AgentFactoryPage from './components/AgentFactoryPage';
import AgentDashboardPage from './components/AgentDashboardPage';
import GamesPage from './components/games/GamesPage';
import PDFTextLayerEditor from './components/PDFTextLayerEditor';
import PublicArtifactPage from './components/PublicArtifactPage';
import AppWallpaperLayer from './components/AppWallpaperLayer';
import apiService from './services/apiService';
import { UI_WALLPAPER_QUERY_KEY } from './config/uiWallpaperBuiltins';
import {
  isUiWallpaperConfigActive,
  mainWorkspaceWallpaperTintBg,
} from './theme/wallpaperPaneSx';

function LegacyAgentLineRedirect() {
  const { lineId } = useParams();
  return <Navigate to={`/agent-factory/line/${lineId}`} replace />;
}

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
  const theme = useTheme();
  const { isAuthenticated, loading: authLoading } = useAuth();
  const { data: uiWallpaperData } = useQuery(
    [UI_WALLPAPER_QUERY_KEY],
    () => apiService.settings.getUserUiWallpaper(),
    { enabled: isAuthenticated && !authLoading, staleTime: 60_000 }
  );
  const mainWorkspaceTint = mainWorkspaceWallpaperTintBg(
    theme,
    isUiWallpaperConfigActive(uiWallpaperData?.config)
  );

  const location = useLocation();
  const isDocumentsRoute = location.pathname.startsWith('/documents');
  const isCodeSpacesRoute = location.pathname.startsWith('/code-spaces');
  const isMediaRoute = location.pathname.startsWith('/media') || location.pathname.startsWith('/music');
  const isAgentFactoryRoute = location.pathname.startsWith('/agent-factory');
  const isFullWidthRoute = isDocumentsRoute || isCodeSpacesRoute || isMediaRoute || isAgentFactoryRoute;
  /** Home has no doc sidebar / tab chrome; same full-bleed shell as full-width routes so wallpaper tint reaches edges and scrolls with content. */
  const isHomeRoute =
    location.pathname === '/home' || location.pathname.startsWith('/home/');
  /** Teams list + team detail: no doc chrome; tint to edges and full scroll like Home. */
  const isTeamsRoute =
    location.pathname === '/teams' || location.pathname.startsWith('/teams/');
  /** Settings: full-column tint to nav and status bar (inner page supplies horizontal padding). */
  const isSettingsRoute = location.pathname.startsWith('/settings');
  /** Control Panes: same full-column tint as Settings. */
  const isControlPanesRoute = location.pathname.startsWith('/control-panes');
  const isMainWorkspaceFullBleed =
    isFullWidthRoute ||
    isHomeRoute ||
    isTeamsRoute ||
    isSettingsRoute ||
    isControlPanesRoute;
  const { isCollapsed, sidebarWidth, isFullWidth, isResizing, toggleSidebar } = useChatSidebar();
  const isMobile = /Mobi|Android/i.test(navigator.userAgent);
  
  // Quick Capture and Journal state and hotkey listeners
  const [captureOpen, setCaptureOpen] = useState(false);
  const [journalOpen, setJournalOpen] = useState(false);
  
  // Clear editor context cache when navigating away from Documents so Chat (or other
  // pages) don't send stale document context. On /documents, TabbedContentManager and
  // DocumentViewer keep or clear editor_ctx_cache based on the active tab and viewer state.
  useEffect(() => {
    if (!isDocumentsRoute) {
      try {
        localStorage.removeItem('editor_ctx_cache');
      } catch (e) {
        console.warn('Failed to clear editor context cache:', e);
      }
    }
  }, [isDocumentsRoute]);
  
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        setCaptureOpen(true);
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'J') {
        e.preventDefault();
        setJournalOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    const openFromUi = () => setCaptureOpen(true);
    window.addEventListener('openQuickCapture', openFromUi);
    return () => window.removeEventListener('openQuickCapture', openFromUi);
  }, []);

  return (
    <Box sx={{ 
      display: 'flex', 
      height: { xs: 'calc(var(--appvh, 100vh) - var(--app-nav-height, 59px) - 32px)', md: 'calc(100dvh - var(--app-nav-height, 59px) - 32px)' },
      position: 'relative',
      zIndex: 1,
      paddingBottom: 'env(safe-area-inset-bottom)',
      width: '100%',
      maxWidth: '100%',
      overflow: 'hidden'
    }}>
      {/* Main Content Area - Responsive to chat sidebar */}
      <Box sx={{ 
        flexGrow: 1, 
        // Do not use overflow:hidden here: it can prevent the MuiContainer scrollbar from receiving
        // drag/hit-testing in some browsers (nested under framer-motion). Outer shell still clips.
        overflow: 'visible',
        transition: isResizing ? 'none' : 'margin-right 0.3s ease-in-out',
        marginRight: isCollapsed ? 0 : (isFullWidth ? '100vw' : `${sidebarWidth}px`),
        // When chat is open, inset so the fixed sidebar strip does not overlap the scrollbar.
        // When chat closed: 10px clears the outer overflow:hidden edge so scrollbar hover/drag stay reliable.
        // Chat panel is margin-reserved; no extra gutter so the seam aligns with the split handle.
        pr: isMobile ? 0 : (!isCollapsed ? 0 : '10px'),
        minWidth: 0, // Allow content to shrink below its natural size
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        maxWidth: { xs: '100%', md: 'none' }
      }}>
        <Container 
          maxWidth={isMainWorkspaceFullBleed ? false : 'xl'} 
          disableGutters={isMainWorkspaceFullBleed} 
          sx={{ 
            mt: isMainWorkspaceFullBleed ? 0 : 4, 
            mb: isMainWorkspaceFullBleed ? 0 : 4, 
            px: isMainWorkspaceFullBleed ? 0 : undefined,
            flex: 1,
            // Full-width + Home: scroll here (single root) — nested overflow:auto inside framer-motion
            // caused native scrollbars to not receive pointer events in some browsers.
            ...(isMainWorkspaceFullBleed
              ? { overflowY: 'auto', overflowX: 'hidden' }
              : { overflow: 'auto' }),
            // Avoid scrollbar-gutter: stable here — on Chromium/WebKit it can desync native
            // scrollbar hit-testing from the painted thumb vs custom ::-webkit-scrollbar styles.
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            // Tint on the scroll container so it covers full width/height including long scrolls (not clipped to motion.div).
            ...(mainWorkspaceTint ? { backgroundColor: mainWorkspaceTint } : {}),
          }}
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{
              display: 'flex',
              flexDirection: 'column',
              flex: 1,
              minHeight: 0,
              width: '100%',
            }}
          >
            <Routes>
              <Route path="/" element={<Navigate to="/documents" replace />} />
              <Route path="/home" element={<HomeDashboardPage />} />
              <Route path="/home/:dashboardId" element={<HomeDashboardPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/code-spaces" element={<CodeSpacesPage />} />

              <Route path="/chat" element={<ChatPage />} />
              <Route path="/teams" element={<TeamsPage />} />
              <Route path="/teams/:teamId" element={<TeamDetailPage />} />
              <Route path="/pdf-text-editor/:documentId" element={<PDFTextLayerEditor />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/control-panes" element={<ControlPanesPage />} />
              <Route path="/contacts" element={<Navigate to="/documents" replace />} />
              <Route path="/media" element={<MediaPage />} />
              <Route path="/music" element={<MediaPage />} />
              <Route path="/map" element={
                <ProtectedRoute>
                  <MapPage />
                </ProtectedRoute>
              } />
              <Route path="/games" element={<GamesPage />} />
              <Route path="/games/:gameId" element={<GamesPage />} />
              <Route path="/agent-dashboard" element={<AgentDashboardPage />} />
              <Route path="/agent-factory" element={<AgentFactoryPage />} />
              <Route path="/agent-factory/lines" element={<Navigate to="/agent-factory" replace />} />
              <Route path="/agent-factory/lines/new" element={<Navigate to="/agent-factory" replace />} />
              <Route path="/agent-factory/lines/:lineId/*" element={<LegacyAgentLineRedirect />} />
              <Route path="/agent-factory/line/:id" element={<AgentFactoryPage />} />
              <Route path="/agent-factory/agent/:id" element={<AgentFactoryPage />} />
              <Route path="/agent-factory/playbook/:id" element={<AgentFactoryPage />} />
              <Route path="/agent-factory/skill/:skillId" element={<AgentFactoryPage />} />
              <Route path="/agent-factory/datasource/:id" element={<AgentFactoryPage />} />
              <Route path="/agent-factory/:id" element={<AgentFactoryPage />} />
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
          top: 'var(--app-nav-height, 59px)',
          height: { xs: 'calc(var(--appvh, 100vh) - var(--app-nav-height, 59px) - 32px)', md: 'calc(100dvh - var(--app-nav-height, 59px) - 32px)' },
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

      {/* Collapsed chat: main pr (10px) + scrollbar (~8px) + safe area */}
      {isCollapsed && (
        <Box
          sx={{
            position: 'fixed',
            top: '50%',
            transform: 'translateY(-50%)',
            right: {
              xs: 'max(12px, calc(8px + env(safe-area-inset-right, 0px)))',
              md: 'calc(22px + env(safe-area-inset-right, 0px))',
            },
            zIndex: 1300,
            pointerEvents: 'none',
          }}
        >
          <Tooltip title="Open Chat">
            <IconButton
              size="small"
              onClick={toggleSidebar}
              aria-label="Open chat"
              sx={{
                pointerEvents: 'auto',
                backgroundColor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1,
                boxShadow: 1,
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ChevronRight fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      )}
      
      {/* Org Quick Capture Modal - Available anywhere via Ctrl+Shift+C */}
      <OrgQuickCapture 
        open={captureOpen} 
        onClose={() => setCaptureOpen(false)} 
      />
      {/* Journal for the day - Ctrl+Shift+J */}
      <JournalDayModal 
        open={journalOpen} 
        onClose={() => setJournalOpen(false)} 
      />
    </Box>
  );
};

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider queryClient={queryClient}>
        <AuthQueryCacheBoundary />
        <VoiceAvailabilityProvider>
        <CapabilitiesProvider>
        <ModelProvider>
        <ChatSidebarProvider>
          <NotificationProvider>
          <TeamExecutionProvider>
          <MessagingProvider>
          <TeamProvider>
          <MusicProvider>
          <VideoProvider>
          <ControlPaneProvider>
          <ArtifactInstanceProvider>
          <EditorProvider>
          <ImageLightboxProvider>
          <LearningProvider>
          <div
            className="App"
            style={{
              position: 'relative',
              zIndex: 1,
              minHeight: '100%',
              backgroundColor: 'transparent',
            }}
          >
            <AppWallpaperLayer />
            <Routes>
              {/* Public route */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/shared/artifact/:shareToken" element={<PublicArtifactPage />} />

              {/* Protected routes */}
              <Route path="/*" element={
                <ProtectedRoute>
                  <>
                    <Navigation />
                    <MainContent />
                    <StatusBar />
                    <MessagingDrawer />
                    <ModelConfigurationNotification />
                    <LearningQuizOverlay />
                  </>
                </ProtectedRoute>
              } />
            </Routes>
          </div>
          </LearningProvider>
          </ImageLightboxProvider>
          </EditorProvider>
          </ArtifactInstanceProvider>
          </ControlPaneProvider>
          </VideoProvider>
          </MusicProvider>
          </TeamProvider>
          </MessagingProvider>
          </TeamExecutionProvider>
          </NotificationProvider>
        </ChatSidebarProvider>
        </ModelProvider>
        </CapabilitiesProvider>
        </VoiceAvailabilityProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
