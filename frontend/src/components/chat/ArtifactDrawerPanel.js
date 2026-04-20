import React, { useState, useCallback, useEffect } from 'react';
import { useQueryClient } from 'react-query';
import {
  Box,
  Typography,
  IconButton,
  Tooltip,
  Button,
  Chip,
  Alert,
  Dialog,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  TextField,
  DialogTitle,
  DialogContent,
  DialogActions,
  Stack,
} from '@mui/material';
import {
  Close,
  ContentCopy,
  Code,
  Fullscreen,
  ChevronLeft,
  PushPin,
  IosShare,
  GetApp,
  KeyboardArrowDown,
} from '@mui/icons-material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight, materialDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ArtifactRenderer from './ArtifactRenderer';
import savedArtifactService from '../../services/savedArtifactService';
import { useTheme } from '../../contexts/ThemeContext';
import PinToDashboardDialog from './PinToDashboardDialog';

const langForHighlight = (artifact) => {
  const hint = (artifact?.language || '').toLowerCase();
  if (hint && hint !== 'mermaid') return hint;
  const t = (artifact?.artifact_type || '').toLowerCase();
  if (t === 'mermaid') return 'markdown';
  if (t === 'html' || t === 'chart') return 'html';
  if (t === 'react') return 'jsx';
  if (t === 'svg') return 'markup';
  return 'text';
};

/**
 * Left column in chat sidebar when an artifact is open: toolbar + renderer or code view.
 */
const ArtifactDrawerPanel = ({
  artifact,
  onClose,
  artifactHistory = [],
  onRevert,
  onCollapse,
  conversationId = null,
  messageId = null,
}) => {
  const { darkMode } = useTheme();
  const queryClient = useQueryClient();
  const [showCode, setShowCode] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [savingLibrary, setSavingLibrary] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [versionMenuAnchor, setVersionMenuAnchor] = useState(null);
  const [libraryMenuAnchor, setLibraryMenuAnchor] = useState(null);
  const [savedLibraryId, setSavedLibraryId] = useState(null);
  const [pinOpen, setPinOpen] = useState(false);
  const [shareOpen, setShareOpen] = useState(false);
  const [sharePayload, setSharePayload] = useState(null);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    setSavedLibraryId(null);
  }, [artifact?.code, artifact?.title, artifact?.artifact_type]);

  useEffect(() => {
    if (!feedback) return undefined;
    const t = setTimeout(() => setFeedback(null), 4000);
    return () => clearTimeout(t);
  }, [feedback]);

  const handleCopy = useCallback(() => {
    const text = artifact?.code ?? '';
    if (!text) {
      setFeedback({ severity: 'error', text: 'Nothing to copy' });
      return;
    }
    if (!navigator.clipboard?.writeText) {
      setFeedback({ severity: 'error', text: 'Clipboard not available' });
      return;
    }
    void navigator.clipboard.writeText(text).then(
      () => setFeedback({ severity: 'success', text: 'Copied to clipboard' }),
      () => setFeedback({ severity: 'error', text: 'Copy failed' })
    );
  }, [artifact]);

  const handleSaveToLibrary = useCallback(async () => {
    if (!artifact?.code) return;
    const t = (artifact.artifact_type || '').toLowerCase();
    const allowed = ['html', 'mermaid', 'chart', 'svg', 'react'];
    if (!allowed.includes(t)) {
      setFeedback({ severity: 'error', text: 'This artifact type cannot be saved to the library' });
      return;
    }
    setSavingLibrary(true);
    try {
      const created = await savedArtifactService.create({
        title: artifact.title || 'Artifact',
        artifact_type: t,
        code: artifact.code,
        language: artifact.language || null,
        source_conversation_id: conversationId || null,
        source_message_id: messageId || null,
      });
      setSavedLibraryId(created.id);
      queryClient.invalidateQueries(['savedArtifactsList']);
      setFeedback({ severity: 'success', text: 'Saved to artifact library' });
    } catch (e) {
      const d = e?.response?.data?.detail;
      setFeedback({
        severity: 'error',
        text: typeof d === 'string' ? d : e?.message || 'Save to library failed',
      });
    } finally {
      setSavingLibrary(false);
    }
  }, [artifact, conversationId, messageId, queryClient]);

  const handleShare = useCallback(async () => {
    if (!savedLibraryId) return;
    try {
      const res = await savedArtifactService.share(savedLibraryId);
      setSharePayload(res);
      setShareOpen(true);
    } catch (e) {
      const d = e?.response?.data?.detail;
      setFeedback({
        severity: 'error',
        text: typeof d === 'string' ? d : e?.message || 'Share failed',
      });
    }
  }, [savedLibraryId]);

  const handleExport = useCallback(async () => {
    if (!savedLibraryId) return;
    setExporting(true);
    try {
      await savedArtifactService.downloadExport(savedLibraryId);
      setFeedback({ severity: 'success', text: 'Download started' });
    } catch (e) {
      setFeedback({ severity: 'error', text: e?.message || 'Export failed' });
    } finally {
      setExporting(false);
    }
  }, [savedLibraryId]);

  const copyText = (label, text) => {
    if (!text || !navigator.clipboard?.writeText) {
      setFeedback({ severity: 'error', text: 'Clipboard not available' });
      return;
    }
    void navigator.clipboard.writeText(text).then(
      () => setFeedback({ severity: 'success', text: `${label} copied` }),
      () => setFeedback({ severity: 'error', text: 'Copy failed' })
    );
  };

  if (!artifact) return null;

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: 0,
        borderRight: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      <Box
        sx={{
          flexShrink: 0,
          px: 1,
          py: 0.75,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          flexWrap: 'wrap',
        }}
      >
        <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }} noWrap title={artifact.title}>
          {artifact.title || 'Artifact'}
        </Typography>
        <Chip size="small" label={String(artifact.artifact_type || '').toUpperCase()} sx={{ height: 22 }} />
        {artifactHistory.length > 0 && onRevert && (
          <>
            <Chip
              size="small"
              label={`v${artifactHistory.length + 1}`}
              onClick={(e) => setVersionMenuAnchor(e.currentTarget)}
              sx={{ height: 22, cursor: 'pointer' }}
              variant="outlined"
            />
            <Menu
              anchorEl={versionMenuAnchor}
              open={Boolean(versionMenuAnchor)}
              onClose={() => setVersionMenuAnchor(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            >
              {artifactHistory.map((h, i) => (
                <MenuItem
                  key={`${i}-${h.title || ''}-${h.code?.slice?.(0, 20) || ''}`}
                  onClick={() => {
                    onRevert(i);
                    setVersionMenuAnchor(null);
                  }}
                >
                  {h.title || `Version ${i + 1}`} ({String(h.artifact_type || '').toUpperCase()})
                </MenuItem>
              ))}
            </Menu>
          </>
        )}
        <Tooltip title="Full screen">
          <IconButton size="small" onClick={() => setFullscreen(true)} aria-label="Full screen artifact">
            <Fullscreen fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title={showCode ? 'Show preview' : 'View code'}>
          <IconButton size="small" onClick={() => setShowCode((v) => !v)} aria-label="Toggle code">
            <Code fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Copy code">
          <IconButton size="small" onClick={handleCopy} aria-label="Copy code">
            <ContentCopy fontSize="small" />
          </IconButton>
        </Tooltip>
        <Button
          size="small"
          variant="outlined"
          disabled={!artifact.code || savingLibrary}
          onClick={() => void handleSaveToLibrary()}
          sx={{ textTransform: 'none', minWidth: 0, px: 1, py: 0.25 }}
        >
          {savingLibrary ? 'Saving…' : 'Save to library'}
        </Button>
        {savedLibraryId ? (
          <>
            <Button
              size="small"
              variant="outlined"
              color="success"
              endIcon={<KeyboardArrowDown sx={{ fontSize: 18 }} />}
              onClick={(e) => setLibraryMenuAnchor(e.currentTarget)}
              aria-haspopup="menu"
              aria-expanded={Boolean(libraryMenuAnchor)}
              aria-controls={libraryMenuAnchor ? 'artifact-library-menu' : undefined}
              sx={{ textTransform: 'none', minWidth: 0, px: 1, py: 0.25 }}
            >
              Library
            </Button>
            <Menu
              id="artifact-library-menu"
              anchorEl={libraryMenuAnchor}
              open={Boolean(libraryMenuAnchor)}
              onClose={() => setLibraryMenuAnchor(null)}
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'right' }}
            >
              <MenuItem
                onClick={() => {
                  setLibraryMenuAnchor(null);
                  setPinOpen(true);
                }}
              >
                <ListItemIcon>
                  <PushPin fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Pin to home dashboard" />
              </MenuItem>
              <MenuItem
                onClick={() => {
                  setLibraryMenuAnchor(null);
                  void handleShare();
                }}
              >
                <ListItemIcon>
                  <IosShare fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Share link…" />
              </MenuItem>
              <MenuItem
                disabled={exporting}
                onClick={() => {
                  setLibraryMenuAnchor(null);
                  void handleExport();
                }}
              >
                <ListItemIcon>
                  <GetApp fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Download standalone HTML" />
              </MenuItem>
            </Menu>
          </>
        ) : null}
        {typeof onCollapse === 'function' && (
          <Tooltip title="Collapse panel">
            <IconButton
              size="small"
              onClick={onCollapse}
              aria-label="Collapse artifact panel"
            >
              <ChevronLeft fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
        <Tooltip title="Close">
          <IconButton size="small" onClick={onClose} aria-label="Close artifact">
            <Close fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {feedback && (
          <Alert severity={feedback.severity} onClose={() => setFeedback(null)} sx={{ borderRadius: 0 }}>
            {feedback.text}
          </Alert>
        )}
        {showCode ? (
          <Box sx={{ p: 1, flex: 1, minHeight: 0, overflow: 'auto' }}>
            <SyntaxHighlighter
              language={langForHighlight(artifact)}
              style={darkMode ? materialDark : materialLight}
              customStyle={{ margin: 0, fontSize: '0.75rem' }}
              showLineNumbers
            >
              {artifact.code || ''}
            </SyntaxHighlighter>
          </Box>
        ) : (
          <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ArtifactRenderer artifact={artifact} height="100%" />
          </Box>
        )}
      </Box>

      <Dialog fullScreen open={fullscreen} onClose={() => setFullscreen(false)}>
        <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
          <Box
            sx={{
              p: 1,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderBottom: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
            }}
          >
            <Typography variant="h6" sx={{ ml: 2 }}>
              {artifact.title || 'Artifact'}
            </Typography>
            <IconButton onClick={() => setFullscreen(false)} aria-label="Exit full screen">
              <Close />
            </IconButton>
          </Box>
          <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ArtifactRenderer artifact={artifact} height="100%" scaleMode="viewport" />
          </Box>
        </Box>
      </Dialog>

      <PinToDashboardDialog
        open={pinOpen}
        onClose={() => setPinOpen(false)}
        artifactId={savedLibraryId}
      />

      <Dialog open={shareOpen} onClose={() => setShareOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Share links</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Anyone with these links can view this artifact. Revoke sharing from the artifact library in
            the document sidebar if needed.
          </Typography>
          {sharePayload ? (
            <Stack spacing={2}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  View page
                </Typography>
                <TextField value={sharePayload.public_url || ''} fullWidth size="small" InputProps={{ readOnly: true }} />
                <Button size="small" onClick={() => copyText('Link', sharePayload.public_url)}>
                  Copy
                </Button>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Embed (iframe)
                </Typography>
                <TextField value={sharePayload.embed_url || ''} fullWidth size="small" InputProps={{ readOnly: true }} />
                <Button size="small" onClick={() => copyText('Embed URL', sharePayload.embed_url)}>
                  Copy
                </Button>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  API (JSON)
                </Typography>
                <TextField value={sharePayload.api_url || ''} fullWidth size="small" InputProps={{ readOnly: true }} />
                <Button size="small" onClick={() => copyText('API URL', sharePayload.api_url)}>
                  Copy
                </Button>
              </Box>
            </Stack>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ArtifactDrawerPanel;
