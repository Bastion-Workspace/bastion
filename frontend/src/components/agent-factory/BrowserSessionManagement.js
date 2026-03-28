/**
 * Session management UI: list saved browser sessions, delete, and re-authenticate.
 * Used in Settings > Browser Sessions.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Delete, Refresh, Lock } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import BrowserAuthCapture from './BrowserAuthCapture';

export default function BrowserSessionManagement() {
  const queryClient = useQueryClient();
  const [reAuthDomain, setReAuthDomain] = useState(null);
  const [reAuthLoginUrl, setReAuthLoginUrl] = useState('');
  const [reAuthSessionId, setReAuthSessionId] = useState(null);
  const [reAuthPendingAuth, setReAuthPendingAuth] = useState(null);
  const [reAuthError, setReAuthError] = useState(null);

  const { data, isLoading } = useQuery(
    'browserAuthSessions',
    () => apiService.agentFactory.browserAuthListSessions(),
    { staleTime: 30000 }
  );

  const deleteMutation = useMutation(
    (domain) => apiService.agentFactory.browserAuthDeleteSession(domain),
    {
      onSuccess: () => queryClient.invalidateQueries('browserAuthSessions'),
    }
  );

  const sessions = data?.sessions ?? [];

  const handleReAuthClick = (session) => {
    setReAuthDomain(session.site_domain);
    setReAuthLoginUrl(session.site_domain ? `https://${session.site_domain}/login` : '');
    setReAuthSessionId(null);
    setReAuthPendingAuth(null);
    setReAuthError(null);
  };

  const handleReAuthStart = async () => {
    if (!reAuthDomain || !reAuthLoginUrl.trim()) return;
    setReAuthError(null);
    try {
      const res = await apiService.agentFactory.browserAuthStartSession({
        site_domain: reAuthDomain,
        login_url: reAuthLoginUrl.trim(),
      });
      if (res?.session_id) {
        setReAuthSessionId(res.session_id);
        setReAuthPendingAuth({
          session_id: res.session_id,
          site_domain: reAuthDomain,
          prompt: 'Log in in the browser, then click "I\'m Logged In".',
        });
      } else {
        setReAuthError(res?.detail || 'Failed to start session');
      }
    } catch (err) {
      setReAuthError(err?.response?.data?.detail || err?.message || 'Failed to start session');
    }
  };

  const handleReAuthComplete = async () => {
    if (!reAuthSessionId || !reAuthDomain) return;
    try {
      await apiService.agentFactory.browserAuthCapture(reAuthSessionId, {
        site_domain: reAuthDomain,
      });
      await apiService.agentFactory.browserAuthCloseSession(reAuthSessionId);
      queryClient.invalidateQueries('browserAuthSessions');
      setReAuthDomain(null);
      setReAuthSessionId(null);
      setReAuthPendingAuth(null);
      setReAuthLoginUrl('');
    } catch (err) {
      setReAuthError(err?.response?.data?.detail || err?.message || 'Capture failed');
    }
  };

  const handleReAuthCancel = async () => {
    if (reAuthSessionId) {
      try {
        await apiService.agentFactory.browserAuthCloseSession(reAuthSessionId);
      } catch (_) {}
    }
    setReAuthDomain(null);
    setReAuthSessionId(null);
    setReAuthPendingAuth(null);
    setReAuthLoginUrl('');
    setReAuthError(null);
  };

  const formatDate = (iso) => {
    if (!iso) return '—';
    try {
      const d = new Date(iso);
      return d.toLocaleString();
    } catch {
      return iso;
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Lock /> Browser sessions
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Saved login sessions for playbook browser steps. Delete to remove; use Re-authenticate to log in again and update the saved session.
      </Typography>

      {isLoading ? (
        <Box display="flex" alignItems="center" gap={2} p={2}>
          <CircularProgress size={24} />
          <Typography variant="body2">Loading sessions…</Typography>
        </Box>
      ) : sessions.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Typography color="text.secondary">No saved browser sessions.</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Sessions are created when you complete a browser login during a playbook run or re-authenticate here.
          </Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Site</TableCell>
                <TableCell>Last used</TableCell>
                <TableCell align="center">Status</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {sessions.map((s) => (
                <TableRow key={s.site_domain}>
                  <TableCell>{s.site_domain}</TableCell>
                  <TableCell>{formatDate(s.last_used_at)}</TableCell>
                  <TableCell align="center">
                    <Chip
                      label={s.is_valid ? 'Valid' : 'Invalid'}
                      size="small"
                      color={s.is_valid ? 'success' : 'default'}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">
                    <IconButton
                      size="small"
                      title="Re-authenticate"
                      onClick={() => handleReAuthClick(s)}
                    >
                      <Refresh />
                    </IconButton>
                    <IconButton
                      size="small"
                      title="Delete session"
                      onClick={() => deleteMutation.mutate(s.site_domain)}
                      disabled={deleteMutation.isLoading}
                    >
                      <Delete />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      <Dialog
        open={reAuthDomain != null}
        onClose={handleReAuthCancel}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Re-authenticate: {reAuthDomain}
        </DialogTitle>
        <DialogContent>
          {reAuthError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setReAuthError(null)}>
              {reAuthError}
            </Alert>
          )}
          {!reAuthPendingAuth ? (
            <Box pt={1}>
              <TextField
                fullWidth
                label="Login URL"
                value={reAuthLoginUrl}
                onChange={(e) => setReAuthLoginUrl(e.target.value)}
                placeholder="https://example.com/login"
                margin="normal"
              />
              <DialogActions sx={{ px: 0, pt: 2 }}>
                <Button onClick={handleReAuthCancel}>Cancel</Button>
                <Button variant="contained" onClick={handleReAuthStart}>
                  Open browser
                </Button>
              </DialogActions>
            </Box>
          ) : (
            <BrowserAuthCapture
              pendingAuth={reAuthPendingAuth}
              onComplete={handleReAuthComplete}
              onCancel={handleReAuthCancel}
            />
          )}
        </DialogContent>
        {reAuthPendingAuth && (
          <DialogActions>
            <Button onClick={handleReAuthCancel}>Cancel</Button>
          </DialogActions>
        )}
      </Dialog>
    </Box>
  );
}
