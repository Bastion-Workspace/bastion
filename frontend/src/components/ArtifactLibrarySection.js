import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Tooltip,
  Collapse,
  Divider,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Stack,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Delete as DeleteIcon,
  Share as ShareIcon,
  LinkOff as LinkOffIcon,
  LibraryBooks as LibraryBooksIcon,
} from '@mui/icons-material';
import { useQuery, useQueryClient } from 'react-query';
import savedArtifactService from '../services/savedArtifactService';

/**
 * Collapsible sidebar section listing saved chat artifacts (library).
 * Opens artifacts in the main tabbed area via onArtifactClick.
 */
const ArtifactLibrarySection = ({ onArtifactClick }) => {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(() => {
    try {
      const saved = localStorage.getItem('artifactLibraryExpanded');
      return saved !== null ? JSON.parse(saved) : true;
    } catch {
      return true;
    }
  });
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [sharePayload, setSharePayload] = useState(null);
  const [shareTargetId, setShareTargetId] = useState(null);

  useEffect(() => {
    localStorage.setItem('artifactLibraryExpanded', JSON.stringify(expanded));
  }, [expanded]);

  const { data, isLoading, error } = useQuery(
    ['savedArtifactsList'],
    () => savedArtifactService.list(),
    { staleTime: 60 * 1000 }
  );

  const artifacts = data?.artifacts || [];

  const handleShare = async (artifact, e) => {
    e.stopPropagation();
    setShareTargetId(artifact.id);
    try {
      const res = await savedArtifactService.share(artifact.id);
      setSharePayload(res);
      setShareDialogOpen(true);
      queryClient.invalidateQueries(['savedArtifactsList']);
    } catch (err) {
      console.error('Share failed:', err);
    }
  };

  const handleUnshare = async (artifact, e) => {
    e.stopPropagation();
    if (!window.confirm('Revoke public sharing for this artifact?')) return;
    try {
      await savedArtifactService.unshare(artifact.id);
      queryClient.invalidateQueries(['savedArtifactsList']);
      if (shareDialogOpen && shareTargetId === artifact.id) {
        setShareDialogOpen(false);
        setSharePayload(null);
      }
    } catch (err) {
      console.error('Unshare failed:', err);
    }
  };

  const handleDelete = async (artifact, e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete "${artifact.title}" from your artifact library?`)) return;
    try {
      await savedArtifactService.delete(artifact.id);
      queryClient.invalidateQueries(['savedArtifactsList']);
      queryClient.invalidateQueries({ queryKey: ['savedArtifactFull', artifact.id] });
    } catch (err) {
      console.error('Delete failed:', err);
    }
  };

  const copyText = (label, text) => {
    if (!text || !navigator.clipboard?.writeText) return;
    void navigator.clipboard.writeText(text).then(
      () => {},
      () => {}
    );
  };

  if (isLoading || error || artifacts.length === 0) {
    return null;
  }

  return (
    <>
      <Divider sx={{ my: 0.5 }} />
      <Box>
        <Box sx={{ px: 2, pb: 0.5 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              mb: 1,
              gap: 0.5,
            }}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                flex: 1,
                minWidth: 0,
                cursor: 'pointer',
                '&:hover': { backgroundColor: 'action.hover' },
                borderRadius: 1,
                p: 0.5,
              }}
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 600,
                  color: 'text.secondary',
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  ml: 0.5,
                }}
              >
                Artifact library
              </Typography>
            </Box>
          </Box>
        </Box>

        <Collapse in={expanded}>
          <List dense sx={{ py: 0 }}>
            {artifacts.map((a) => (
              <ListItem
                key={a.id}
                disablePadding
                secondaryAction={
                  <Box sx={{ display: 'flex', alignItems: 'center', mr: 0.5 }}>
                    <Tooltip title={a.is_public ? 'Sharing options' : 'Share'}>
                      <IconButton edge="end" size="small" onClick={(e) => handleShare(a, e)}>
                        <ShareIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    {a.is_public ? (
                      <Tooltip title="Revoke sharing">
                        <IconButton edge="end" size="small" onClick={(e) => handleUnshare(a, e)}>
                          <LinkOffIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    ) : null}
                    <Tooltip title="Delete">
                      <IconButton edge="end" size="small" onClick={(e) => handleDelete(a, e)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                }
              >
                <ListItemButton
                  onClick={() => onArtifactClick?.(a)}
                  sx={{ pr: 14, borderRadius: 1 }}
                >
                  <ListItemIcon sx={{ minWidth: 32, alignSelf: 'flex-start', mt: 0.5 }}>
                    <LibraryBooksIcon fontSize="small" color="action" />
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Typography variant="body2" noWrap title={a.title}>
                        {a.title}
                      </Typography>
                    }
                    secondary={
                      <Chip
                        component="span"
                        size="small"
                        label={String(a.artifact_type || '').toUpperCase()}
                        sx={{ height: 20, mt: 0.25, fontSize: '0.65rem' }}
                      />
                    }
                    secondaryTypographyProps={{ component: 'div' }}
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Collapse>
      </Box>

      <Dialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Share links</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Anyone with these links can view this artifact. Use Revoke sharing in the sidebar to
            disable access.
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
          <Button onClick={() => setShareDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default ArtifactLibrarySection;
