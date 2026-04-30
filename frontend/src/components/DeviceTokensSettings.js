/**
 * Device tokens (Bastion Local Proxy): list, create, show one-time token, revoke,
 * and per-device capability + shell policy (Settings gear).
 */

import React, { useState, useEffect, useCallback } from 'react';
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
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  FormControlLabel,
  Switch,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
} from '@mui/material';
import LinkOffIcon from '@mui/icons-material/LinkOff';
import AddIcon from '@mui/icons-material/Add';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import RefreshIcon from '@mui/icons-material/Refresh';
import DevicesIcon from '@mui/icons-material/Devices';
import SettingsIcon from '@mui/icons-material/Settings';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
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

const linesToList = (s) => {
  if (!s || !String(s).trim()) return [];
  return String(s)
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean);
};

const emptyCapForm = () => ({
  enabled: true,
  allowed_paths: '',
  denied_paths: '',
  allowed_commands: '',
  denied_patterns: '',
});

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

  const [proxyCaps, setProxyCaps] = useState([]);
  const [policyDialogToken, setPolicyDialogToken] = useState(null);
  const [policyTab, setPolicyTab] = useState(0);
  const [capForm, setCapForm] = useState({});
  const [shellRules, setShellRules] = useState([]);
  const [policyLoading, setPolicyLoading] = useState(false);
  const [policySaving, setPolicySaving] = useState(false);
  const [policyError, setPolicyError] = useState(null);
  const [ruleForm, setRuleForm] = useState({
    pattern: '',
    match_mode: 'prefix',
    action: 'require_approval',
    priority: 50,
    label: '',
  });
  const [ruleSaving, setRuleSaving] = useState(false);

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

  const openPolicyDialog = useCallback(async (tokenRow) => {
    setPolicyDialogToken(tokenRow);
    setPolicyTab(0);
    setPolicyError(null);
    setPolicyLoading(true);
    setRuleForm({
      pattern: '',
      match_mode: 'prefix',
      action: 'require_approval',
      priority: 50,
      label: '',
    });
    try {
      let caps = proxyCaps;
      if (!caps.length) {
        const cr = await apiService.get('/api/settings/proxy-capabilities');
        caps = cr.capabilities || [];
        setProxyCaps(caps);
      }
      const [polRes, rulesRes] = await Promise.all([
        apiService.get(
          `/api/settings/device-tokens/${encodeURIComponent(tokenRow.id)}/policy`
        ),
        apiService.get('/api/settings/shell-policy/rules'),
      ]);
      const pol = polRes.capabilities || {};
      const form = {};
      caps.forEach((c) => {
        const p = pol[c.id] || {};
        form[c.id] = {
          enabled: p.enabled !== false,
          allowed_paths: (p.allowed_paths || []).join('\n'),
          denied_paths: (p.denied_paths || []).join('\n'),
          allowed_commands: (p.allowed_commands || []).join('\n'),
          denied_patterns: (p.denied_patterns || []).join('\n'),
        };
      });
      setCapForm(form);
      setShellRules(rulesRes.rules || []);
    } catch (e) {
      setPolicyError(e.message || 'Failed to load policy');
      setCapForm({});
      setShellRules([]);
    } finally {
      setPolicyLoading(false);
    }
  }, [proxyCaps]);

  const closePolicyDialog = () => {
    setPolicyDialogToken(null);
    setPolicyError(null);
    setPolicyTab(0);
  };

  const saveCapabilitiesPolicy = async () => {
    if (!policyDialogToken || !proxyCaps.length) return;
    setPolicySaving(true);
    setPolicyError(null);
    try {
      const capsPayload = {};
      proxyCaps.forEach((c) => {
        const st = capForm[c.id] || emptyCapForm();
        capsPayload[c.id] = {
          enabled: !!st.enabled,
          allowed_paths: linesToList(st.allowed_paths),
          denied_paths: linesToList(st.denied_paths),
          allowed_commands: linesToList(st.allowed_commands),
          denied_patterns: linesToList(st.denied_patterns),
        };
      });
      await apiService.patch(
        `/api/settings/device-tokens/${encodeURIComponent(policyDialogToken.id)}/policy`,
        { capabilities: capsPayload }
      );
      await loadTokens({ silent: true });
    } catch (e) {
      setPolicyError(e.response?.data?.detail || e.message || 'Save failed');
    } finally {
      setPolicySaving(false);
    }
  };

  const reloadShellRules = async () => {
    const rulesRes = await apiService.get('/api/settings/shell-policy/rules');
    setShellRules(rulesRes.rules || []);
  };

  const addShellRule = async () => {
    const pattern = (ruleForm.pattern || '').trim();
    if (!pattern) return;
    setRuleSaving(true);
    setPolicyError(null);
    try {
      await apiService.post('/api/settings/shell-policy/rules', {
        pattern,
        match_mode: ruleForm.match_mode,
        action: ruleForm.action,
        priority: Number(ruleForm.priority) || 50,
        label: (ruleForm.label || '').trim() || null,
      });
      setRuleForm({
        pattern: '',
        match_mode: 'prefix',
        action: 'require_approval',
        priority: 50,
        label: '',
      });
      await reloadShellRules();
    } catch (e) {
      setPolicyError(e.response?.data?.detail || e.message || 'Failed to add rule');
    } finally {
      setRuleSaving(false);
    }
  };

  const deleteShellRule = async (ruleId) => {
    setPolicyError(null);
    try {
      await apiService.delete(
        `/api/settings/shell-policy/rules/${encodeURIComponent(ruleId)}`
      );
      await reloadShellRules();
    } catch (e) {
      setPolicyError(e.response?.data?.detail || e.message || 'Failed to delete rule');
    }
  };

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

  const updateCapField = (capId, field, value) => {
    setCapForm((prev) => ({
      ...prev,
      [capId]: { ...(prev[capId] || emptyCapForm()), [field]: value },
    }));
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <DevicesIcon /> Local proxy (devices)
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Add devices running the Bastion Local Proxy daemon so they can connect to this workspace. Create a token here, then paste it in the daemon&apos;s Settings. Use the gear icon to set which capabilities are advertised and shell command rules (allow / deny / require approval).
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
              No devices yet. Click &quot;Add device&quot; and paste the token into the Local Proxy daemon.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {activeTokens.map((t) => (
                <ListItem
                  key={t.id}
                  secondaryAction={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <IconButton
                        edge="end"
                        aria-label="Proxy policy"
                        onClick={() => openPolicyDialog(t)}
                        sx={{ mr: 0.5 }}
                      >
                        <SettingsIcon />
                      </IconButton>
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
                    </Box>
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
                In the Local Proxy daemon, open Settings, paste this token, and save and connect.
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

      <Dialog
        open={Boolean(policyDialogToken)}
        onClose={closePolicyDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Proxy policy
          {policyDialogToken ? ` — ${policyDialogToken.device_name || policyDialogToken.id}` : ''}
        </DialogTitle>
        <DialogContent dividers>
          {policyError && (
            <Alert severity="error" sx={{ mb: 2 }} onClose={() => setPolicyError(null)}>
              {policyError}
            </Alert>
          )}
          {!policyDialogToken ? null : policyLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : (
            <>
              <Tabs value={policyTab} onChange={(_, v) => setPolicyTab(v)} sx={{ mb: 2 }}>
                <Tab label="Capabilities" />
                <Tab label="Shell command rules" />
              </Tabs>
              {policyTab === 0 && (
                <Box>
                  {!policyDialogToken.connected && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      Device is offline. Saved policy will apply when the daemon reconnects.
                    </Alert>
                  )}
                  {proxyCaps.map((c) => {
                    const st = capForm[c.id] || emptyCapForm();
                    const sup = c.supports || {};
                    return (
                      <Box key={c.id} sx={{ mb: 2, pb: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
                        <FormControlLabel
                          control={
                            <Switch
                              checked={!!st.enabled}
                              onChange={(e) => updateCapField(c.id, 'enabled', e.target.checked)}
                            />
                          }
                          label={<Typography fontWeight={600}>{c.label}</Typography>}
                        />
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                          {c.description}
                        </Typography>
                        {sup.path_policy ? (
                          <>
                            <TextField
                              size="small"
                              fullWidth
                              margin="dense"
                              label="Allowed paths (one per line)"
                              value={st.allowed_paths}
                              onChange={(e) => updateCapField(c.id, 'allowed_paths', e.target.value)}
                              multiline
                              minRows={2}
                            />
                            <TextField
                              size="small"
                              fullWidth
                              margin="dense"
                              label="Denied path substrings (one per line)"
                              value={st.denied_paths}
                              onChange={(e) => updateCapField(c.id, 'denied_paths', e.target.value)}
                              multiline
                              minRows={2}
                            />
                          </>
                        ) : null}
                        {sup.command_policy ? (
                          <>
                            <TextField
                              size="small"
                              fullWidth
                              margin="dense"
                              label="Allowed command binaries (one per line; empty = any)"
                              value={st.allowed_commands}
                              onChange={(e) => updateCapField(c.id, 'allowed_commands', e.target.value)}
                              multiline
                              minRows={2}
                            />
                            <TextField
                              size="small"
                              fullWidth
                              margin="dense"
                              label="Denied substrings (one per line)"
                              value={st.denied_patterns}
                              onChange={(e) => updateCapField(c.id, 'denied_patterns', e.target.value)}
                              multiline
                              minRows={2}
                            />
                          </>
                        ) : null}
                      </Box>
                    );
                  })}
                </Box>
              )}
              {policyTab === 1 && (
                <Box>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    Rules apply to local shell execution in agents (first match wins). Empty list means all commands are allowed unless a rule matches. Use &quot;require approval&quot; to prompt in chat before running.
                  </Alert>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Pattern</TableCell>
                        <TableCell>Match</TableCell>
                        <TableCell>Action</TableCell>
                        <TableCell align="right"> </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {(shellRules || []).map((r) => (
                        <TableRow key={r.id}>
                          <TableCell>{r.pattern}</TableCell>
                          <TableCell>{r.match_mode}</TableCell>
                          <TableCell>{r.action}</TableCell>
                          <TableCell align="right">
                            <IconButton
                              size="small"
                              aria-label="Delete rule"
                              onClick={() => deleteShellRule(r.id)}
                            >
                              <DeleteOutlineIcon fontSize="small" />
                            </IconButton>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                    Add rule
                  </Typography>
                  <TextField
                    size="small"
                    fullWidth
                    margin="dense"
                    label="Pattern (e.g. git or rm)"
                    value={ruleForm.pattern}
                    onChange={(e) => setRuleForm((f) => ({ ...f, pattern: e.target.value }))}
                  />
                  <FormControl size="small" fullWidth margin="dense">
                    <InputLabel>Match mode</InputLabel>
                    <Select
                      label="Match mode"
                      value={ruleForm.match_mode}
                      onChange={(e) =>
                        setRuleForm((f) => ({ ...f, match_mode: e.target.value }))
                      }
                    >
                      <MenuItem value="prefix">prefix (first token)</MenuItem>
                      <MenuItem value="contains">contains</MenuItem>
                      <MenuItem value="glob">glob (full command)</MenuItem>
                    </Select>
                  </FormControl>
                  <FormControl size="small" fullWidth margin="dense">
                    <InputLabel>Action</InputLabel>
                    <Select
                      label="Action"
                      value={ruleForm.action}
                      onChange={(e) => setRuleForm((f) => ({ ...f, action: e.target.value }))}
                    >
                      <MenuItem value="allow">allow</MenuItem>
                      <MenuItem value="deny">deny</MenuItem>
                      <MenuItem value="require_approval">require approval</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    size="small"
                    fullWidth
                    margin="dense"
                    type="number"
                    label="Priority (lower runs first)"
                    value={ruleForm.priority}
                    onChange={(e) =>
                      setRuleForm((f) => ({ ...f, priority: e.target.value }))
                    }
                  />
                  <Button
                    variant="outlined"
                    onClick={addShellRule}
                    disabled={ruleSaving || !(ruleForm.pattern || '').trim()}
                    sx={{ mt: 1 }}
                  >
                    {ruleSaving ? 'Adding…' : 'Add rule'}
                  </Button>
                </Box>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closePolicyDialog}>Close</Button>
          {policyTab === 0 && (
            <Button
              variant="contained"
              onClick={saveCapabilitiesPolicy}
              disabled={policySaving || policyLoading || !proxyCaps.length}
            >
              {policySaving ? 'Saving…' : 'Save capabilities'}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DeviceTokensSettings;
