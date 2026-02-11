/**
 * External Connections Settings - OAuth email (Office 365), messaging bots (Telegram, Discord), and admin system email.
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
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Link,
  Grid,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import LinkOffIcon from '@mui/icons-material/LinkOff';
import EmailIcon from '@mui/icons-material/Email';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';

const ExternalConnectionsSettings = () => {
  const { user } = useAuth();
  const [connections, setConnections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connecting, setConnecting] = useState(false);
  const [refreshingId, setRefreshingId] = useState(null);
  const [disconnectingId, setDisconnectingId] = useState(null);

  const [systemEmail, setSystemEmail] = useState({ connection_id: null, connection: null });
  const [systemEmailLoading, setSystemEmailLoading] = useState(false);
  const [systemEmailSaving, setSystemEmailSaving] = useState(false);
  const [systemEmailSelect, setSystemEmailSelect] = useState('');
  const [testEmailSending, setTestEmailSending] = useState(false);
  const [testEmailResult, setTestEmailResult] = useState(null);

  const [smtpLoading, setSmtpLoading] = useState(false);
  const [smtpSaving, setSmtpSaving] = useState(false);
  const [smtpTestSending, setSmtpTestSending] = useState(false);
  const [smtpTestResult, setSmtpTestResult] = useState(null);
  const [smtpEnabled, setSmtpEnabled] = useState(false);
  const [smtpHost, setSmtpHost] = useState('');
  const [smtpPort, setSmtpPort] = useState(587);
  const [smtpUser, setSmtpUser] = useState('');
  const [smtpPassword, setSmtpPassword] = useState('');
  const [smtpFromEmail, setSmtpFromEmail] = useState('');
  const [smtpFromName, setSmtpFromName] = useState('');
  const [smtpUseTls, setSmtpUseTls] = useState(true);
  const [smtpPasswordSet, setSmtpPasswordSet] = useState(false);
  const [smtpTestToEmail, setSmtpTestToEmail] = useState('');

  const [telegramDialogOpen, setTelegramDialogOpen] = useState(false);
  const [discordDialogOpen, setDiscordDialogOpen] = useState(false);
  const [telegramToken, setTelegramToken] = useState('');
  const [discordToken, setDiscordToken] = useState('');
  const [botConnecting, setBotConnecting] = useState(false);
  const [botStatuses, setBotStatuses] = useState({});

  const loadConnections = async () => {
    try {
      setError(null);
      const res = await apiService.get('/api/connections');
      setConnections(res.connections || []);
    } catch (err) {
      setError(err.message || 'Failed to load connections');
      setConnections([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConnections();
  }, []);

  const loadSystemEmail = async () => {
    if (user?.role !== 'admin') return;
    try {
      setSystemEmailLoading(true);
      const res = await apiService.get('/api/admin/system-email');
      setSystemEmail({
        connection_id: res.connection_id ?? null,
        connection: res.connection ?? null,
      });
      setSystemEmailSelect(res.connection_id != null ? String(res.connection_id) : '');
    } catch (err) {
      setSystemEmail({ connection_id: null, connection: null });
      setSystemEmailSelect('');
    } finally {
      setSystemEmailLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === 'admin') loadSystemEmail();
  }, [user?.role]);

  const loadSmtpSettings = async () => {
    if (user?.role !== 'admin') return;
    try {
      setSmtpLoading(true);
      const res = await apiService.get('/api/admin/smtp-settings');
      setSmtpEnabled(res.enabled ?? false);
      setSmtpHost(res.host ?? '');
      setSmtpPort(res.port ?? 587);
      setSmtpUser(res.user ?? '');
      setSmtpFromEmail(res.from_email ?? '');
      setSmtpFromName(res.from_name ?? '');
      setSmtpUseTls(res.use_tls ?? true);
      setSmtpPasswordSet(res.password_set ?? false);
      setSmtpPassword('');
    } catch {
      setSmtpEnabled(false);
      setSmtpHost('');
      setSmtpPort(587);
      setSmtpUser('');
      setSmtpFromEmail('');
      setSmtpFromName('');
      setSmtpUseTls(true);
      setSmtpPasswordSet(false);
    } finally {
      setSmtpLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === 'admin') loadSmtpSettings();
  }, [user?.role]);

  const handleSaveSmtp = async () => {
    if (user?.role !== 'admin') return;
    try {
      setSmtpSaving(true);
      setError(null);
      setSmtpTestResult(null);
      const body = {
        enabled: smtpEnabled,
        host: smtpHost,
        port: smtpPort,
        user: smtpUser,
        use_tls: smtpUseTls,
        from_email: smtpFromEmail,
        from_name: smtpFromName,
      };
      if (smtpPassword) body.password = smtpPassword;
      await apiService.put('/api/admin/smtp-settings', body);
      setSmtpPassword('');
      await loadSmtpSettings();
    } catch (err) {
      setError(err.message || 'Failed to save SMTP settings');
    } finally {
      setSmtpSaving(false);
    }
  };

  const handleTestSmtp = async () => {
    if (user?.role !== 'admin') return;
    try {
      setSmtpTestSending(true);
      setSmtpTestResult(null);
      setError(null);
      const body = smtpTestToEmail.trim() ? { to_email: smtpTestToEmail.trim() } : {};
      const res = await apiService.post('/api/admin/smtp-settings/test', body);
      setSmtpTestResult({ success: true, message: res.message || 'Test email sent!' });
    } catch (err) {
      const detail = err.response?.data?.detail;
      const message = Array.isArray(detail) ? detail[0] : detail || err.message || 'Test failed.';
      setSmtpTestResult({ success: false, message });
    } finally {
      setSmtpTestSending(false);
    }
  };

  const handleConnectOffice365 = async () => {
    try {
      setConnecting(true);
      setError(null);
      const res = await apiService.get('/api/oauth/microsoft/authorize');
      if (res?.url) {
        window.location.href = res.url;
        return;
      }
      setError('No authorization URL returned');
    } catch (err) {
      setError(err.message || 'Failed to start OAuth');
    } finally {
      setConnecting(false);
    }
  };

  const handleDisconnect = async (connectionId) => {
    try {
      setDisconnectingId(connectionId);
      setError(null);
      await apiService.delete(`/api/connections/${connectionId}`);
      await loadConnections();
      if (user?.role === 'admin' && systemEmail.connection_id === connectionId) {
        setSystemEmail({ connection_id: null, connection: null });
        setSystemEmailSelect('');
      }
    } catch (err) {
      setError(err.message || 'Failed to disconnect');
    } finally {
      setDisconnectingId(null);
    }
  };

  const handleRefresh = async (connectionId) => {
    try {
      setRefreshingId(connectionId);
      setError(null);
      await apiService.post(`/api/connections/${connectionId}/refresh`);
      await loadConnections();
    } catch (err) {
      setError(err.message || 'Failed to refresh token');
    } finally {
      setRefreshingId(null);
    }
  };

  const handleSetSystemEmail = async () => {
    if (user?.role !== 'admin') return;
    const id = systemEmailSelect === '' || systemEmailSelect === 'none' ? null : parseInt(systemEmailSelect, 10);
    try {
      setSystemEmailSaving(true);
      setError(null);
      setTestEmailResult(null);
      await apiService.put('/api/admin/system-email', { connection_id: id });
      await loadSystemEmail();
    } catch (err) {
      setError(err.message || 'Failed to set system email');
    } finally {
      setSystemEmailSaving(false);
    }
  };

  const handleTestSystemEmail = async () => {
    if (user?.role !== 'admin') return;
    try {
      setTestEmailSending(true);
      setTestEmailResult(null);
      setError(null);
      const res = await apiService.post('/api/admin/system-email/test');
      setTestEmailResult({ success: true, message: res.message || 'Test email sent!' });
    } catch (err) {
      setTestEmailResult({ success: false, message: err.message || 'Test failed.' });
    } finally {
      setTestEmailSending(false);
    }
  };

  const emailConnections = connections.filter((c) => c.connection_type === 'email');
  const chatBotConnections = connections.filter((c) => c.connection_type === 'chat_bot');

  const loadBotStatus = async (connectionId) => {
    try {
      const res = await apiService.get(`/api/connections/${connectionId}/bot-status`);
      setBotStatuses((prev) => ({ ...prev, [connectionId]: res }));
    } catch {
      setBotStatuses((prev) => ({ ...prev, [connectionId]: { status: 'error', bot_username: '', error: 'Failed to load' } }));
    }
  };

  useEffect(() => {
    chatBotConnections.forEach((c) => loadBotStatus(c.id));
  }, [connections]);

  const handleConnectTelegram = async () => {
    if (!telegramToken.trim()) return;
    try {
      setBotConnecting(true);
      setError(null);
      await apiService.post('/api/connections/telegram', { bot_token: telegramToken.trim() });
      setTelegramToken('');
      setTelegramDialogOpen(false);
      await loadConnections();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect Telegram bot');
    } finally {
      setBotConnecting(false);
    }
  };

  const handleConnectDiscord = async () => {
    if (!discordToken.trim()) return;
    try {
      setBotConnecting(true);
      setError(null);
      await apiService.post('/api/connections/discord', { bot_token: discordToken.trim() });
      setDiscordToken('');
      setDiscordDialogOpen(false);
      await loadConnections();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect Discord bot');
    } finally {
      setBotConnecting(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <EmailIcon /> Email connections
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect your email (Office 365) to read and send mail via the assistant.
      </Typography>

      {error && (
        <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Button
            variant="contained"
            onClick={handleConnectOffice365}
            disabled={connecting}
            startIcon={connecting ? <CircularProgress size={20} /> : null}
          >
            {connecting ? 'Redirecting…' : 'Connect Office 365'}
          </Button>

          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading connections…</Typography>
            </Box>
          ) : emailConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No email connections yet. Connect Office 365 above.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {emailConnections.map((conn) => (
                <ListItem
                  key={conn.id}
                  secondaryAction={
                    <>
                      <IconButton
                        edge="end"
                        aria-label="Refresh"
                        onClick={() => handleRefresh(conn.id)}
                        disabled={refreshingId === conn.id}
                      >
                        {refreshingId === conn.id ? (
                          <CircularProgress size={20} />
                        ) : (
                          <RefreshIcon />
                        )}
                      </IconButton>
                      <IconButton
                        edge="end"
                        aria-label="Disconnect"
                        onClick={() => handleDisconnect(conn.id)}
                        disabled={disconnectingId === conn.id}
                      >
                        {disconnectingId === conn.id ? (
                          <CircularProgress size={20} />
                        ) : (
                          <LinkOffIcon />
                        )}
                      </IconButton>
                    </>
                  }
                >
                  <ListItemText
                    primary={conn.display_name || conn.account_identifier}
                    secondary={`${conn.provider} · ${conn.account_identifier}${conn.connection_status ? ` · ${conn.connection_status}` : ''}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Divider sx={{ my: 2 }} />
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SmartToyIcon /> Messaging bots
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect a Telegram or Discord bot to chat with the AI assistant from those platforms. Conversations appear in the web UI.
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Button
            variant="contained"
            onClick={() => setTelegramDialogOpen(true)}
            disabled={botConnecting}
            sx={{ mr: 1 }}
          >
            Connect Telegram Bot
          </Button>
          <Button
            variant="contained"
            onClick={() => setDiscordDialogOpen(true)}
            disabled={botConnecting}
          >
            Connect Discord Bot
          </Button>
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : chatBotConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No messaging bots connected. Create a bot in Telegram (BotFather) or Discord (Developer Portal), then paste the token above.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {chatBotConnections.map((conn) => (
                <ListItem
                  key={conn.id}
                  secondaryAction={
                    <IconButton
                      edge="end"
                      aria-label="Disconnect"
                      onClick={() => handleDisconnect(conn.id)}
                      disabled={disconnectingId === conn.id}
                    >
                      {disconnectingId === conn.id ? <CircularProgress size={20} /> : <LinkOffIcon />}
                    </IconButton>
                  }
                >
                  <ListItemText
                    primary={conn.display_name || conn.account_identifier || `${conn.provider} bot`}
                    secondary={`${conn.provider} · @${conn.account_identifier} · ${(botStatuses[conn.id]?.status) || '…'}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Dialog open={telegramDialogOpen} onClose={() => !botConnecting && setTelegramDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect Telegram Bot</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Create a bot via <Link href="https://t.me/BotFather" target="_blank" rel="noopener noreferrer">BotFather</Link>, then paste the bot token here.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Bot token"
            type="password"
            fullWidth
            value={telegramToken}
            onChange={(e) => setTelegramToken(e.target.value)}
            placeholder="123456789:ABCdefGHI..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTelegramDialogOpen(false)} disabled={botConnecting}>Cancel</Button>
          <Button onClick={handleConnectTelegram} variant="contained" disabled={botConnecting || !telegramToken.trim()}>
            {botConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={discordDialogOpen} onClose={() => !botConnecting && setDiscordDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect Discord Bot</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Create an application and bot in the <Link href="https://discord.com/developers/applications" target="_blank" rel="noopener noreferrer">Discord Developer Portal</Link>, then paste the bot token here.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Bot token"
            type="password"
            fullWidth
            value={discordToken}
            onChange={(e) => setDiscordToken(e.target.value)}
            placeholder="..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDiscordDialogOpen(false)} disabled={botConnecting}>Cancel</Button>
          <Button onClick={handleConnectDiscord} variant="contained" disabled={botConnecting || !discordToken.trim()}>
            {botConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

      {user?.role === 'admin' && (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="h6" gutterBottom>
            System outbound email
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Account used when the system sends emails on behalf of the app (e.g. agent notifications). Configure SMTP (any provider) or use a connected Microsoft account.
          </Typography>

          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>SMTP (recommended)</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Use any SMTP server (Office 365, Gmail, SendGrid, etc.). No OAuth required.
              </Typography>
              {smtpLoading ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={24} />
                  <Typography variant="body2">Loading…</Typography>
                </Box>
              ) : (
                <>
                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        size="small"
                        fullWidth
                        label="SMTP host"
                        value={smtpHost}
                        onChange={(e) => setSmtpHost(e.target.value)}
                        placeholder="smtp.office365.com"
                      />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                      <TextField
                        size="small"
                        fullWidth
                        type="number"
                        label="Port"
                        value={smtpPort}
                        onChange={(e) => setSmtpPort(Number(e.target.value) || 587)}
                      />
                    </Grid>
                    <Grid item xs={12} sm={3}>
                      <FormControlLabel
                        control={<Checkbox checked={smtpUseTls} onChange={(e) => setSmtpUseTls(e.target.checked)} />}
                        label="Use TLS"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        size="small"
                        fullWidth
                        label="Username"
                        value={smtpUser}
                        onChange={(e) => setSmtpUser(e.target.value)}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        size="small"
                        fullWidth
                        type="password"
                        label="Password"
                        value={smtpPassword}
                        onChange={(e) => setSmtpPassword(e.target.value)}
                        placeholder={smtpPasswordSet ? '•••••••• (leave blank to keep)' : ''}
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        size="small"
                        fullWidth
                        label="From email"
                        value={smtpFromEmail}
                        onChange={(e) => setSmtpFromEmail(e.target.value)}
                        placeholder="noreply@example.com"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        size="small"
                        fullWidth
                        label="From name"
                        value={smtpFromName}
                        onChange={(e) => setSmtpFromName(e.target.value)}
                        placeholder="Bastion AI"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={<Checkbox checked={smtpEnabled} onChange={(e) => setSmtpEnabled(e.target.checked)} />}
                        label="Enable SMTP for system outbound email"
                      />
                    </Grid>
                    <Grid item xs={12} sm={6}>
                      <TextField
                        size="small"
                        fullWidth
                        type="email"
                        label="Test email address (optional)"
                        value={smtpTestToEmail}
                        onChange={(e) => setSmtpTestToEmail(e.target.value)}
                        placeholder={user?.email || 'e.g. you@example.com'}
                        helperText="Leave blank to use your profile email"
                      />
                    </Grid>
                  </Grid>
                  <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Button variant="contained" onClick={handleSaveSmtp} disabled={smtpSaving}>
                      {smtpSaving ? 'Saving…' : 'Save SMTP'}
                    </Button>
                    <Button variant="outlined" onClick={handleTestSmtp} disabled={smtpTestSending || !smtpHost}>
                      {smtpTestSending ? 'Sending…' : 'Send test email'}
                    </Button>
                  </Box>
                  {smtpTestResult && (
                    <Alert severity={smtpTestResult.success ? 'success' : 'error'} sx={{ mt: 2 }}>
                      {smtpTestResult.message}
                    </Alert>
                  )}
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>Or use Microsoft account (OAuth)</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Use a connected Office 365 account for system email. Requires Microsoft OAuth to be configured.
              </Typography>
              {systemEmailLoading ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={24} />
                  <Typography variant="body2">Loading…</Typography>
                </Box>
              ) : (
                <>
                  {systemEmail.connection ? (
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      Current: {systemEmail.connection.display_name || systemEmail.connection.account_identifier} (
                      {systemEmail.connection.account_identifier})
                    </Typography>
                  ) : (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      No Microsoft connection designated.
                    </Typography>
                  )}
                  <FormControl size="small" sx={{ minWidth: 280, mr: 1 }}>
                    <InputLabel>Designate connection</InputLabel>
                    <Select
                      value={systemEmailSelect}
                      label="Designate connection"
                      onChange={(e) => setSystemEmailSelect(e.target.value)}
                    >
                      <MenuItem value="">None</MenuItem>
                      {emailConnections.map((conn) => (
                        <MenuItem key={conn.id} value={String(conn.id)}>
                          {conn.display_name || conn.account_identifier} ({conn.account_identifier})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Button
                    variant="outlined"
                    onClick={handleSetSystemEmail}
                    disabled={systemEmailSaving}
                    sx={{ mr: 1 }}
                  >
                    {systemEmailSaving ? 'Saving…' : 'Save'}
                  </Button>
                  {systemEmail.connection_id != null && (
                    <Button
                      variant="outlined"
                      onClick={handleTestSystemEmail}
                      disabled={testEmailSending}
                    >
                      {testEmailSending ? 'Sending…' : 'Send test email'}
                    </Button>
                  )}
                  {testEmailResult && (
                    <Alert severity={testEmailResult.success ? 'success' : 'error'} sx={{ mt: 2 }}>
                      {testEmailResult.message}
                    </Alert>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
};

export default ExternalConnectionsSettings;
