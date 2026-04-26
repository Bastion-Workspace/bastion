/**
 * Device tokens (Bastion Local Proxy): list, create, show one-time token, revoke.
 * Shown under Settings → Connections.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Chip,
} from '@mui/material';
import LinkOffIcon from '@mui/icons-material/LinkOff';
import AddIcon from '@mui/icons-material/Add';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import RefreshIcon from '@mui/icons-material/Refresh';
import DevicesIcon from '@mui/icons-material/Devices';
import apiService from '../services/apiService';

const POLL_MS = 12000;

const formatDate = (iso) => {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
};

const DeviceTokensSettings = () => {
  const [tokens, setTokens] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [revokingId, setRevokingId] = useState(null);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [deviceName, setDeviceName] = useState('');
  const [creating, setCreating] = useState(false);
  const [createdResult, setCreatedResult] = useState(null);
  const [tokenCopied, setTokenCopied] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  const loadTokens = async ({ silent } = {}) => {
    try {
      setError(null);
      if (silent) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      const res = await apiService.get('/api/settings/device-tokens');
      setTokens(res.tokens || []);
    } catch (err) {
      setError(err.message || 'Failed to load device tokens');
      setTokens([]);
    } finally {
      if (silent) {
        setRefreshing(false);
      } else {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    loadTokens({ silent: false });
  }, []);

  useEffect(() => {
    const id = setInterval(() => loadTokens({ silent: true }), POLL_MS);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const onVis = () => {
      if (document.visibilityState === 'visible') {
        loadTokens({ silent: true });
      }
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, []);

  const deviceSecondaryText = (t) => {
    const name = t.device_name || '';
    if (t.connected) {
      const parts = ['Connected now'];
      if (t.live_device_id && t.live_device_id !== name) {
        parts.push(`Device ID: ${t.live_device_id}`);
      }
      return parts.join(' · ');
    }
    if (t.last_connected_at) {
      return `Offline · Last connected: ${formatDate(t.last_connected_at)}`;
    }
    return `Offline · Created: ${formatDate(t.created_at)} · Not connected yet`;
  };

  const handleAddDevice = async () => {
    const name = (deviceName || '').trim();
    if (!name) return;
    try {
      setCreating(true);
      setError(null);
      setCreatedResult(null);
      const res = await apiService.post('/api/settings/device-tokens', { device_name: name });
      setCreatedResult({
        device_name: res.device_name,
        token: res.token,
        message: res.message || 'Copy the token now; it will not be shown again.',
      });
      setDeviceName('');
      await loadTokens({ silent: true });
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to create token');
    } finally {
      setCreating(false);
    }
  };

  const handleCloseAddDialog = () => {
    setAddDialogOpen(false);
    setDeviceName('');
    setCreatedResult(null);
    setTokenCopied(false);
    setError(null);
  };

  const handleCopyToken = async () => {
    if (!createdResult?.token) return;
    try {
      await navigator.clipboard.writeText(createdResult.token);
      setTokenCopied(true);
      setTimeout(() => setTokenCopied(false), 2000);
    } catch {
      setError('Could not copy to clipboard');
    }
  };

  const handleRevoke = async (tokenId) => {
    try {
      setRevokingId(tokenId);
      setError(null);
      await apiService.delete(`/api/settings/device-tokens/${encodeURIComponent(tokenId)}`);
      await loadTokens({ silent: true });
    } catch (err) {
      setError(err.message || 'Failed to revoke token');
    } finally {
      setRevokingId(null);
    }
  };

  const activeTokens = tokens.filter((t) => !t.revoked);
  const revokedTokens = tokens.filter((t) => t.revoked);

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <DevicesIcon /> Local proxy (devices)
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Add devices running the Bastion Local Proxy daemon so they can connect to this workspace. Create a token here, then paste it in the daemon's Settings. Each token is shown only once.
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setAddDialogOpen(true)}
            >
              Add device
            </Button>
            <Button
              variant="outlined"
              startIcon={refreshing ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
              onClick={() => loadTokens({ silent: true })}
              disabled={loading || refreshing}
            >
              Refresh status
            </Button>
            {refreshing && !loading && (
              <Typography variant="caption" color="text.secondary">
                Updating…
              </Typography>
            )}
          </Box>

          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading devices…</Typography>
            </Box>
          ) : activeTokens.length === 0 && revokedTokens.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No devices yet. Click "Add device" and paste the token into the Local Proxy daemon.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {activeTokens.map((t) => (
                <ListItem
                  key={t.id}
                  secondaryAction={
                    <IconButton
                      edge="end"
                      aria-label="Revoke"
                      onClick={() => handleRevoke(t.id)}
                      disabled={revokingId === t.id}
                    >
                      {revokingId === t.id ? (
                        <CircularProgress size={20} />
                      ) : (
                        <LinkOffIcon />
                      )}
                    </IconButton>
                  }
                >
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Typography component="span" variant="body1">
                          {t.device_name || 'Unnamed device'}
                        </Typography>
                        <Chip
                          size="small"
                          label={t.connected ? 'Online' : 'Offline'}
                          color={t.connected ? 'success' : 'default'}
                          variant={t.connected ? 'filled' : 'outlined'}
                        />
                      </Box>
                    }
                    secondary={deviceSecondaryText(t)}
                  />
                </ListItem>
              ))}
              {revokedTokens.length > 0 && (
                <>
                  <Typography variant="caption" color="text.secondary" sx={{ px: 2, pt: 1 }}>
                    Revoked
                  </Typography>
                  {revokedTokens.map((t) => (
                    <ListItem key={t.id} disabled>
                      <ListItemText
                        primary={t.device_name || 'Unnamed device'}
                        secondary={`Revoked · Created ${formatDate(t.created_at)}`}
                      />
                    </ListItem>
                  ))}
                </>
              )}
            </List>
          )}
        </CardContent>
      </Card>

      <Dialog open={addDialogOpen} onClose={handleCloseAddDialog} maxWidth="sm" fullWidth>
        <DialogTitle>{createdResult ? 'Token created' : 'Add device'}</DialogTitle>
        <DialogContent>
          {!createdResult ? (
            <TextField
              autoFocus
              margin="dense"
              label="Device name"
              fullWidth
              value={deviceName}
              onChange={(e) => setDeviceName(e.target.value)}
              placeholder="e.g. My laptop"
              helperText="A label to identify this device in the list"
            />
          ) : (
            <Box>
              <Typography variant="body2" color="text.secondary" paragraph>
                {createdResult.message}
              </Typography>
              <Box
                sx={{
                  bgcolor: 'action.hover',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  fontSize: '0.85rem',
                  wordBreak: 'break-all',
                  mb: 2,
                }}
              >
                {createdResult.token}
              </Box>
              <Button
                variant="outlined"
                startIcon={<ContentCopyIcon />}
                onClick={handleCopyToken}
                fullWidth
              >
                {tokenCopied ? 'Copied' : 'Copy token'}
              </Button>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                In the Local Proxy daemon, open Settings → paste this token and (if asked) the Device ID shown there. Save and connect.
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          {createdResult ? (
            <Button onClick={handleCloseAddDialog} variant="contained">
              Done
            </Button>
          ) : (
            <>
              <Button onClick={handleCloseAddDialog}>Cancel</Button>
              <Button
                variant="contained"
                onClick={handleAddDevice}
                disabled={creating || !deviceName.trim()}
              >
                {creating ? 'Creating…' : 'Create token'}
              </Button>
            </>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DeviceTokensSettings;
