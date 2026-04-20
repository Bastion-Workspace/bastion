import React, { useState, useCallback, useEffect, useMemo } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Tooltip,
  Button,
  Chip,
  Alert,
  Dialog,
  TextField,
  DialogTitle,
  DialogContent,
  DialogActions,
  Stack,
  CircularProgress,
} from '@mui/material';
import {
  Close,
  ContentCopy,
  Code,
  Fullscreen,
  IosShare,
  GetApp,
  Edit,
  Check,
  PushPin,
} from '@mui/icons-material';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { materialLight, materialDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import ArtifactRenderer from './chat/ArtifactRenderer';
import savedArtifactService from '../services/savedArtifactService';
import { useTheme } from '../contexts/ThemeContext';
import PinToDashboardDialog from './chat/PinToDashboardDialog';

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
 * Full-tab viewer for a saved artifact (opened from the artifact library sidebar).
 */
const ArtifactViewerTab = ({ artifactId, onClose }) => {
  const { darkMode } = useTheme();
  const queryClient = useQueryClient();
  const [showCode, setShowCode] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [shareOpen, setShareOpen] = useState(false);
  const [sharePayload, setSharePayload] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [pinOpen, setPinOpen] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');

  const { data, isLoading, error, refetch } = useQuery(
    ['savedArtifactFull', artifactId],
    () => savedArtifactService.get(artifactId),
    {
      enabled: Boolean(artifactId),
      staleTime: 60 * 1000,
    }
  );

  const artifact = useMemo(() => {
    if (!data) return null;
    return {
      artifact_type: data.artifact_type,
      title: data.title,
      code: data.code,
      language: data.language || undefined,
    };
  }, [data]);

  useEffect(() => {
    if (data?.title != null) setTitleDraft(data.title);
  }, [data?.title]);

  useEffect(() => {
    if (!feedback) return undefined;
    const t = setTimeout(() => setFeedback(null), 4000);
    return () => clearTimeout(t);
  }, [feedback]);

  const updateTabTitleInChrome = useCallback(
    (newTitle) => {
      if (typeof window !== 'undefined' && window.tabbedContentManagerRef?.updateArtifactTabTitle) {
        window.tabbedContentManagerRef.updateArtifactTabTitle(artifactId, newTitle);
      }
    },
    [artifactId]
  );

  const patchTitleMutation = useMutation(
    async (title) => {
      return savedArtifactService.patch(artifactId, { title });
    },
    {
      onSuccess: (res) => {
        queryClient.invalidateQueries(['savedArtifactsList']);
        queryClient.invalidateQueries(['savedArtifactFull', artifactId]);
        const next = res?.title || titleDraft;
        updateTabTitleInChrome(next);
        setEditingTitle(false);
        setFeedback({ severity: 'success', text: 'Title updated' });
      },
      onError: (e) => {
        const d = e?.response?.data?.detail;
        setFeedback({
          severity: 'error',
          text: typeof d === 'string' ? d : e?.message || 'Rename failed',
        });
      },
    }
  );

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

  const handleShare = useCallback(async () => {
    try {
      const res = await savedArtifactService.share(artifactId);
      setSharePayload(res);
      setShareOpen(true);
      queryClient.invalidateQueries(['savedArtifactsList']);
    } catch (e) {
      const d = e?.response?.data?.detail;
      setFeedback({
        severity: 'error',
        text: typeof d === 'string' ? d : e?.message || 'Share failed',
      });
    }
  }, [artifactId, queryClient]);

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      await savedArtifactService.downloadExport(artifactId);
      setFeedback({ severity: 'success', text: 'Download started' });
    } catch (e) {
      setFeedback({ severity: 'error', text: e?.message || 'Export failed' });
    } finally {
      setExporting(false);
    }
  }, [artifactId]);

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

  const commitTitle = () => {
    const t = (titleDraft || '').trim();
    if (!t || t === data?.title) {
      setEditingTitle(false);
      setTitleDraft(data?.title || '');
      return;
    }
    patchTitleMutation.mutate(t);
  };

  if (!artifactId) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="text.secondary">No artifact selected.</Typography>
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%" minHeight={200}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !data || !artifact) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">
          Could not load this artifact. It may have been deleted.
        </Typography>
        <Button sx={{ mt: 1 }} onClick={() => refetch()}>
          Retry
        </Button>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: 0,
        bgcolor: 'background.default',
      }}
    >
      <Box
        sx={{
          flexShrink: 0,
          px: 1.5,
          py: 0.75,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          flexWrap: 'nowrap',
          bgcolor: 'background.paper',
          minWidth: 0,
          overflowX: 'auto',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            flex: 1,
            minWidth: 0,
            overflow: 'hidden',
          }}
        >
          {editingTitle ? (
            <>
              <TextField
                size="small"
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitTitle();
                  if (e.key === 'Escape') {
                    setEditingTitle(false);
                    setTitleDraft(data.title);
                  }
                }}
                sx={{ flex: 1, minWidth: 120, maxWidth: 360 }}
                autoFocus
              />
              <Tooltip title="Save title">
                <IconButton size="small" onClick={commitTitle} disabled={patchTitleMutation.isLoading}>
                  <Check fontSize="small" />
                </IconButton>
              </Tooltip>
            </>
          ) : (
            <>
              <Typography variant="subtitle2" sx={{ flex: 1, minWidth: 0 }} noWrap title={data.title}>
                {data.title || 'Artifact'}
              </Typography>
              <Tooltip title="Rename">
                <IconButton
                  size="small"
                  onClick={() => {
                    setTitleDraft(data.title);
                    setEditingTitle(true);
                  }}
                  aria-label="Rename artifact"
                  sx={{ flexShrink: 0 }}
                >
                  <Edit fontSize="small" />
                </IconButton>
              </Tooltip>
            </>
          )}
          <Chip
            size="small"
            label={String(artifact.artifact_type || '').toUpperCase()}
            sx={{ height: 22, flexShrink: 0 }}
          />
        </Box>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            flexShrink: 0,
            flexWrap: 'wrap',
            justifyContent: 'flex-end',
          }}
        >
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
            startIcon={<PushPin fontSize="small" />}
            onClick={() => setPinOpen(true)}
            sx={{ textTransform: 'none' }}
          >
            Pin
          </Button>
          <Button
            size="small"
            variant="outlined"
            startIcon={<IosShare fontSize="small" />}
            onClick={() => void handleShare()}
            sx={{ textTransform: 'none' }}
          >
            Share
          </Button>
          <Button
            size="small"
            variant="outlined"
            disabled={exporting}
            startIcon={<GetApp fontSize="small" />}
            onClick={() => void handleExport()}
            sx={{ textTransform: 'none' }}
          >
            Export
          </Button>
          {typeof onClose === 'function' && (
            <Tooltip title="Close tab">
              <IconButton size="small" onClick={onClose} aria-label="Close">
                <Close fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
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
              customStyle={{ margin: 0, fontSize: '0.8125rem' }}
              showLineNumbers
            >
              {artifact.code || ''}
            </SyntaxHighlighter>
          </Box>
        ) : (
          <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', p: 1 }}>
            <ArtifactRenderer artifact={artifact} artifactId={artifactId} height="100%" />
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
              {data.title || 'Artifact'}
            </Typography>
            <IconButton onClick={() => setFullscreen(false)} aria-label="Exit full screen">
              <Close />
            </IconButton>
          </Box>
          <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <ArtifactRenderer artifact={artifact} artifactId={artifactId} height="100%" scaleMode="viewport" />
          </Box>
        </Box>
      </Dialog>

      <PinToDashboardDialog open={pinOpen} onClose={() => setPinOpen(false)} artifactId={artifactId} />

      <Dialog open={shareOpen} onClose={() => setShareOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Share links</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Anyone with these links can view this artifact. Revoke sharing from the artifact library
            sidebar if needed.
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

export default ArtifactViewerTab;
