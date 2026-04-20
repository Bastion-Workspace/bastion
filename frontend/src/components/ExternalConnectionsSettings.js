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
  Collapse,
  ListItemIcon,
  Tooltip,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import LinkOffIcon from '@mui/icons-material/LinkOff';
import EmailIcon from '@mui/icons-material/Email';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import ExtensionIcon from '@mui/icons-material/Extension';
import HubIcon from '@mui/icons-material/Hub';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';
import DeviceTokensSettings from './DeviceTokensSettings';

const M365_SERVICE_OPTIONS = [
  { key: 'email', label: 'Email' },
  { key: 'calendar', label: 'Calendar' },
  { key: 'contacts', label: 'Contacts' },
  { key: 'todo', label: 'To Do' },
  { key: 'files', label: 'OneDrive' },
  { key: 'onenote', label: 'OneNote' },
  { key: 'planner', label: 'Planner' },
  { key: 'devops', label: 'Azure DevOps' },
];

const DEFAULT_M365_SERVICES = ['email', 'calendar', 'contacts'];

const ExternalConnectionsSettings = () => {
  const { user } = useAuth();
  const [connections, setConnections] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connecting, setConnecting] = useState(false);
  const [githubConnecting, setGithubConnecting] = useState(false);
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
  const [slackDialogOpen, setSlackDialogOpen] = useState(false);
  const [smsDialogOpen, setSmsDialogOpen] = useState(false);
  const [telegramToken, setTelegramToken] = useState('');
  const [discordToken, setDiscordToken] = useState('');
  const [slackBotToken, setSlackBotToken] = useState('');
  const [slackAppToken, setSlackAppToken] = useState('');
  const [smsAccountSid, setSmsAccountSid] = useState('');
  const [smsAuthToken, setSmsAuthToken] = useState('');
  const [smsFromNumber, setSmsFromNumber] = useState('');
  const [botConnecting, setBotConnecting] = useState(false);
  const [botStatuses, setBotStatuses] = useState({});

  const [imapSmtpDialogOpen, setImapSmtpDialogOpen] = useState(false);
  const [imapSmtpConnecting, setImapSmtpConnecting] = useState(false);
  const [connImapHost, setConnImapHost] = useState('');
  const [connImapPort, setConnImapPort] = useState(993);
  const [connImapSsl, setConnImapSsl] = useState(true);
  const [connSmtpHost, setConnSmtpHost] = useState('');
  const [connSmtpPort, setConnSmtpPort] = useState(587);
  const [connSmtpTls, setConnSmtpTls] = useState(true);
  const [connImapSmtpUsername, setConnImapSmtpUsername] = useState('');
  const [connImapPassword, setConnImapPassword] = useState('');
  const [connSmtpPassword, setConnSmtpPassword] = useState('');
  const [connImapSmtpDisplayName, setConnImapSmtpDisplayName] = useState('');

  const [caldavDialogOpen, setCaldavDialogOpen] = useState(false);
  const [caldavConnecting, setCaldavConnecting] = useState(false);
  const [caldavUrl, setCaldavUrl] = useState('');
  const [caldavUsername, setCaldavUsername] = useState('');
  const [caldavPassword, setCaldavPassword] = useState('');
  const [caldavDisplayName, setCaldavDisplayName] = useState('');

  const [giteaDialogOpen, setGiteaDialogOpen] = useState(false);
  const [giteaConnecting, setGiteaConnecting] = useState(false);
  const [giteaApiBaseUrl, setGiteaApiBaseUrl] = useState('');
  const [giteaPat, setGiteaPat] = useState('');
  const [giteaDisplayName, setGiteaDisplayName] = useState('');

  const [mcpServers, setMcpServers] = useState([]);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpDialogOpen, setMcpDialogOpen] = useState(false);
  const [mcpSaving, setMcpSaving] = useState(false);
  const [mcpName, setMcpName] = useState('');
  const [mcpTransport, setMcpTransport] = useState('sse');
  const [mcpUrl, setMcpUrl] = useState('');
  const [mcpCommand, setMcpCommand] = useState('');
  const [mcpArgsText, setMcpArgsText] = useState('');
  const [mcpHeadersJson, setMcpHeadersJson] = useState('{}');
  const [mcpEnvJson, setMcpEnvJson] = useState('{}');
  const [mcpEditingId, setMcpEditingId] = useState(null);
  const [mcpBusyId, setMcpBusyId] = useState(null);

  const [m365ConnectDialogOpen, setM365ConnectDialogOpen] = useState(false);
  const [m365ConnectSelections, setM365ConnectSelections] = useState(() => new Set(DEFAULT_M365_SERVICES));
  const [m365ConnServices, setM365ConnServices] = useState({});
  const [m365ServicesLoading, setM365ServicesLoading] = useState(false);
  const [m365SavingId, setM365SavingId] = useState(null);
  /** When true, the M365 service checkboxes for that connection id are shown (collapsed by default). */
  const [m365ServicesExpanded, setM365ServicesExpanded] = useState({});
  const [devopsOrgNames, setDevopsOrgNames] = useState({});
  const [devopsOrgSaving, setDevopsOrgSaving] = useState(null);

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

  useEffect(() => {
    const ms = connections.filter((c) => c.connection_type === 'email' && c.provider === 'microsoft');
    if (ms.length === 0) {
      setM365ConnServices({});
      return undefined;
    }
    let cancelled = false;
    (async () => {
      setM365ServicesLoading(true);
      const next = {};
      for (const c of ms) {
        try {
          const r = await apiService.get(`/api/connections/${c.id}/microsoft/services`);
          if (!cancelled) {
            next[c.id] = {
              enabled: Array.isArray(r.enabled_services) ? [...r.enabled_services] : [],
              available: Array.isArray(r.available_services) ? r.available_services : [],
            };
          }
        } catch {
          if (!cancelled) {
            next[c.id] = { enabled: [...DEFAULT_M365_SERVICES], available: M365_SERVICE_OPTIONS.map((o) => o.key), error: true };
          }
        }
      }
      if (!cancelled) {
        setM365ConnServices(next);
        setM365ServicesLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [connections]);

  useEffect(() => {
    const ms = connections.filter((c) => c.connection_type === 'email' && c.provider === 'microsoft');
    const next = {};
    for (const c of ms) {
      next[c.id] = c.devops_organization || '';
    }
    setDevopsOrgNames(next);
  }, [connections]);

  const saveDevopsOrg = async (connId) => {
    try {
      setDevopsOrgSaving(connId);
      setError(null);
      await apiService.post(`/api/connections/${connId}/microsoft/devops-config`, {
        organization: devopsOrgNames[connId] || '',
      });
      await loadConnections();
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : err.message || 'Failed to save DevOps organization');
    } finally {
      setDevopsOrgSaving(null);
    }
  };

  const loadMcpServers = async () => {
    try {
      setMcpLoading(true);
      const data = await apiService.agentFactory.listMcpServers();
      setMcpServers(Array.isArray(data) ? data : []);
    } catch (err) {
      setMcpServers([]);
    } finally {
      setMcpLoading(false);
    }
  };

  useEffect(() => {
    loadMcpServers();
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

  const handleOpenM365ConnectDialog = () => {
    setM365ConnectSelections(new Set(DEFAULT_M365_SERVICES));
    setM365ConnectDialogOpen(true);
  };

  const handleConfirmM365Connect = async () => {
    try {
      setConnecting(true);
      setError(null);
      const services = Array.from(m365ConnectSelections).filter(Boolean).join(',') || DEFAULT_M365_SERVICES.join(',');
      const res = await apiService.get('/api/oauth/microsoft/authorize', {
        params: { services },
      });
      if (res?.url) {
        setM365ConnectDialogOpen(false);
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

  const toggleM365ConnService = (connId, key) => {
    setM365ConnServices((prev) => {
      const cur = prev[connId];
      if (!cur) return prev;
      const enabled = new Set(cur.enabled || []);
      if (enabled.has(key)) enabled.delete(key);
      else enabled.add(key);
      return { ...prev, [connId]: { ...cur, enabled: Array.from(enabled) } };
    });
  };

  const saveMicrosoftServices = async (connId) => {
    const row = m365ConnServices[connId];
    if (!row?.enabled?.length) {
      setError('Select at least one Microsoft 365 service.');
      return;
    }
    try {
      setM365SavingId(connId);
      setError(null);
      const res = await apiService.post(`/api/connections/${connId}/microsoft/update-services`, {
        services: row.enabled,
      });
      if (res?.reauth_required && res?.authorize_url) {
        window.location.href = res.authorize_url;
        return;
      }
      await loadConnections();
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : err.message || 'Failed to update services');
    } finally {
      setM365SavingId(null);
    }
  };

  const handleConnectGitHub = async () => {
    try {
      setGithubConnecting(true);
      setError(null);
      const res = await apiService.get('/api/oauth/github/authorize');
      if (res?.url) {
        window.location.href = res.url;
        return;
      }
      setError('No authorization URL returned');
    } catch (err) {
      const detail = err.response?.data?.detail;
      const message = typeof detail === 'string' ? detail : err.message || 'Failed to start GitHub OAuth';
      setError(message);
    } finally {
      setGithubConnecting(false);
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
  const calendarConnections = connections.filter((c) => c.connection_type === 'calendar');
  const githubConnections = connections.filter(
    (c) => c.connection_type === 'code_platform' && c.provider === 'github',
  );
  const giteaConnections = connections.filter(
    (c) => c.connection_type === 'code_platform' && c.provider === 'gitea',
  );
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

  const handleConnectSlack = async () => {
    if (!slackBotToken.trim() || !slackAppToken.trim()) return;
    try {
      setBotConnecting(true);
      setError(null);
      await apiService.post('/api/connections/slack', {
        bot_token: slackBotToken.trim(),
        app_token: slackAppToken.trim(),
      });
      setSlackBotToken('');
      setSlackAppToken('');
      setSlackDialogOpen(false);
      await loadConnections();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect Slack bot');
    } finally {
      setBotConnecting(false);
    }
  };

  const handleConnectSMS = async () => {
    if (!smsAccountSid.trim() || !smsAuthToken.trim() || !smsFromNumber.trim()) return;
    try {
      setBotConnecting(true);
      setError(null);
      await apiService.post('/api/connections/sms', {
        account_sid: smsAccountSid.trim(),
        auth_token: smsAuthToken.trim(),
        from_number: smsFromNumber.trim(),
      });
      setSmsAccountSid('');
      setSmsAuthToken('');
      setSmsFromNumber('');
      setSmsDialogOpen(false);
      await loadConnections();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect SMS');
    } finally {
      setBotConnecting(false);
    }
  };

  const handleConnectImapSmtp = async () => {
    if (!connImapHost.trim() || !connSmtpHost.trim() || !connImapSmtpUsername.trim() || !connImapPassword || !connSmtpPassword) return;
    try {
      setImapSmtpConnecting(true);
      setError(null);
      await apiService.post('/api/connections/imap-smtp', {
        imap_host: connImapHost.trim(),
        imap_port: connImapPort,
        imap_ssl: connImapSsl,
        smtp_host: connSmtpHost.trim(),
        smtp_port: connSmtpPort,
        smtp_tls: connSmtpTls,
        username: connImapSmtpUsername.trim(),
        imap_password: connImapPassword,
        smtp_password: connSmtpPassword,
        display_name: connImapSmtpDisplayName.trim() || undefined,
      });
      setConnImapHost('');
      setConnImapPort(993);
      setConnImapSsl(true);
      setConnSmtpHost('');
      setConnSmtpPort(587);
      setConnSmtpTls(true);
      setConnImapSmtpUsername('');
      setConnImapPassword('');
      setConnSmtpPassword('');
      setConnImapSmtpDisplayName('');
      setImapSmtpDialogOpen(false);
      await loadConnections();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect IMAP/SMTP');
    } finally {
      setImapSmtpConnecting(false);
    }
  };

  const handleConnectCaldav = async () => {
    if (!caldavUrl.trim() || !caldavUsername.trim() || !caldavPassword) return;
    try {
      setCaldavConnecting(true);
      setError(null);
      await apiService.post('/api/connections/caldav', {
        url: caldavUrl.trim(),
        username: caldavUsername.trim(),
        password: caldavPassword,
        display_name: caldavDisplayName.trim() || undefined,
      });
      setCaldavUrl('');
      setCaldavUsername('');
      setCaldavPassword('');
      setCaldavDisplayName('');
      setCaldavDialogOpen(false);
      await loadConnections();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to connect CalDAV');
    } finally {
      setCaldavConnecting(false);
    }
  };

  const handleConnectGitea = async () => {
    if (!giteaApiBaseUrl.trim() || !giteaPat.trim()) return;
    try {
      setGiteaConnecting(true);
      setError(null);
      await apiService.post('/api/connections/gitea', {
        api_base_url: giteaApiBaseUrl.trim(),
        personal_access_token: giteaPat.trim(),
        display_name: giteaDisplayName.trim() || undefined,
      });
      setGiteaApiBaseUrl('');
      setGiteaPat('');
      setGiteaDisplayName('');
      setGiteaDialogOpen(false);
      await loadConnections();
    } catch (err) {
      const detail = err.response?.data?.detail;
      const message =
        typeof detail === 'string'
          ? detail
          : err.message || 'Failed to connect Gitea';
      setError(message);
    } finally {
      setGiteaConnecting(false);
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <EmailIcon /> Email connections
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect your email (Office 365 or IMAP/SMTP) to read and send mail via the assistant.
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
            onClick={handleOpenM365ConnectDialog}
            disabled={connecting}
            sx={{ mr: 1 }}
          >
            Connect Office 365
          </Button>
          <Button
            variant="contained"
            onClick={() => setImapSmtpDialogOpen(true)}
            disabled={imapSmtpConnecting}
          >
            Connect IMAP/SMTP
          </Button>

          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading connections…</Typography>
            </Box>
          ) : emailConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No email connections yet. Connect Office 365 or IMAP/SMTP above.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {emailConnections.map((conn) => (
                <ListItem
                  key={conn.id}
                  alignItems="flex-start"
                  sx={{ flexWrap: 'wrap' }}
                  secondaryAction={
                    <>
                      {conn.provider === 'microsoft' && (
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
                      )}
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
                  {conn.provider === 'microsoft' && (
                    <ListItemIcon sx={{ minWidth: 40, mt: 0.25, alignSelf: 'flex-start' }}>
                      <Tooltip
                        title={
                          m365ServicesExpanded[conn.id]
                            ? 'Hide Microsoft 365 services'
                            : 'Manage Microsoft 365 services'
                        }
                      >
                        <IconButton
                          size="small"
                          edge="start"
                          aria-expanded={Boolean(m365ServicesExpanded[conn.id])}
                          aria-controls={`m365-services-${conn.id}`}
                          aria-label="Microsoft 365 services"
                          onClick={() =>
                            setM365ServicesExpanded((prev) => ({
                              ...prev,
                              [conn.id]: !prev[conn.id],
                            }))
                          }
                        >
                          {m365ServicesExpanded[conn.id] ? (
                            <ExpandLessIcon fontSize="small" />
                          ) : (
                            <ExpandMoreIcon fontSize="small" />
                          )}
                        </IconButton>
                      </Tooltip>
                    </ListItemIcon>
                  )}
                  <ListItemText
                    sx={{ flex: '1 1 auto', minWidth: 0 }}
                    primary={conn.display_name || conn.account_identifier}
                    secondary={
                      <>
                        {`${conn.provider} · ${conn.account_identifier}${
                          conn.connection_status ? ` · ${conn.connection_status}` : ''
                        }`}
                        {conn.provider === 'microsoft' && Array.isArray(conn.enabled_services) && conn.enabled_services.length > 0 ? (
                          <>
                            <br />
                            <Typography component="span" variant="caption" color="text.secondary">
                              Enabled: {conn.enabled_services.join(', ')}
                            </Typography>
                          </>
                        ) : null}
                      </>
                    }
                  />
                  {conn.provider === 'microsoft' && (
                    <Collapse
                      in={Boolean(m365ServicesExpanded[conn.id])}
                      timeout="auto"
                      sx={{ flexBasis: '100%', width: '100%' }}
                    >
                      <Box
                        id={`m365-services-${conn.id}`}
                        sx={{ pt: 1, pb: 1, pl: { xs: 1, sm: 2 }, maxWidth: 720 }}
                      >
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                          Microsoft 365 services (save applies; extra permissions may open sign-in again)
                        </Typography>
                        {m365ServicesLoading && !m365ConnServices[conn.id] ? (
                          <CircularProgress size={22} />
                        ) : (
                          <>
                            <Grid container spacing={0.5} sx={{ maxWidth: 720 }}>
                              {M365_SERVICE_OPTIONS.map((opt) => {
                                const row = m365ConnServices[conn.id];
                                const checked = row?.enabled?.includes(opt.key) ?? false;
                                const allowed = !row?.available?.length || row.available.includes(opt.key);
                                return (
                                  <Grid item xs={6} sm={4} md={3} key={opt.key}>
                                    <FormControlLabel
                                      control={
                                        <Checkbox
                                          size="small"
                                          checked={checked}
                                          disabled={!allowed}
                                          onChange={() => toggleM365ConnService(conn.id, opt.key)}
                                        />
                                      }
                                      label={opt.label}
                                    />
                                  </Grid>
                                );
                              })}
                            </Grid>
                            {(m365ConnServices[conn.id]?.enabled || []).includes('devops') && (
                              <Box sx={{ mt: 1.5, display: 'flex', alignItems: 'center', gap: 1, maxWidth: 480 }}>
                                <TextField
                                  size="small"
                                  label="Azure DevOps Organization"
                                  placeholder="e.g. my-org"
                                  value={devopsOrgNames[conn.id] || ''}
                                  onChange={(e) =>
                                    setDevopsOrgNames((prev) => ({ ...prev, [conn.id]: e.target.value }))
                                  }
                                  sx={{ flex: 1 }}
                                  helperText="Your Azure DevOps organization name (from dev.azure.com/{org})"
                                />
                                <Button
                                  size="small"
                                  variant="outlined"
                                  onClick={() => saveDevopsOrg(conn.id)}
                                  disabled={devopsOrgSaving === conn.id}
                                >
                                  {devopsOrgSaving === conn.id ? 'Saving…' : 'Save'}
                                </Button>
                              </Box>
                            )}
                            <Button
                              size="small"
                              variant="outlined"
                              sx={{ mt: 1 }}
                              onClick={() => saveMicrosoftServices(conn.id)}
                              disabled={m365SavingId === conn.id}
                            >
                              {m365SavingId === conn.id ? 'Saving…' : 'Save services'}
                            </Button>
                          </>
                        )}
                      </Box>
                    </Collapse>
                  )}
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Dialog open={m365ConnectDialogOpen} onClose={() => !connecting && setM365ConnectDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect Office 365</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Choose which Microsoft 365 services this connection may use. You can add more later by expanding the account row and editing services.
          </Typography>
          {M365_SERVICE_OPTIONS.map((opt) => (
            <FormControlLabel
              key={opt.key}
              control={
                <Checkbox
                  checked={m365ConnectSelections.has(opt.key)}
                  onChange={() => {
                    setM365ConnectSelections((prev) => {
                      const n = new Set(prev);
                      if (n.has(opt.key)) n.delete(opt.key);
                      else n.add(opt.key);
                      return n;
                    });
                  }}
                />
              }
              label={opt.label}
            />
          ))}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setM365ConnectDialogOpen(false)} disabled={connecting}>
            Cancel
          </Button>
          <Button variant="contained" onClick={handleConfirmM365Connect} disabled={connecting || m365ConnectSelections.size === 0}>
            {connecting ? 'Redirecting…' : 'Continue to Microsoft'}
          </Button>
        </DialogActions>
      </Dialog>

      <Divider sx={{ my: 2 }} />
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <CalendarMonthIcon /> Calendar connections
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect CalDAV calendars (Google Calendar, Nextcloud, iCloud, etc.) for the Agenda view. Office 365 calendars are included when you connect Office 365 under Email.
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Button
            variant="contained"
            onClick={() => setCaldavDialogOpen(true)}
            disabled={caldavConnecting}
          >
            Connect CalDAV
          </Button>
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : calendarConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No calendar connections yet. Connect CalDAV above.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {calendarConnections.map((conn) => (
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
                    primary={conn.display_name || conn.account_identifier}
                    secondary={`${conn.provider} · ${conn.account_identifier}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Divider sx={{ my: 2 }} />
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <HubIcon /> GitHub
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect GitHub so agents can read repositories, pull requests, issues, and code via the GitHub API. Bind accounts in Agent
        Factory to grant specific agents access. Configure GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET on the server.
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Button
            variant="contained"
            onClick={handleConnectGitHub}
            disabled={githubConnecting}
            startIcon={githubConnecting ? <CircularProgress size={20} /> : null}
          >
            {githubConnecting ? 'Redirecting…' : 'Connect GitHub'}
          </Button>
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : githubConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No GitHub accounts connected yet.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {githubConnections.map((conn) => (
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
                    primary={conn.display_name || conn.account_identifier}
                    secondary={`@${conn.account_identifier}${conn.connection_status ? ` · ${conn.connection_status}` : ''}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Divider sx={{ my: 2 }} />
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <HubIcon /> Gitea
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect a self-hosted Gitea instance with a personal access token. Use your server URL (for example{' '}
        <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>https://git.example.com</Box>
        ); <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>/api/v1</Box> is added if omitted. Bind the
        Gitea tool pack in Agent Factory and select this connection.
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Button variant="contained" onClick={() => setGiteaDialogOpen(true)} disabled={giteaConnecting}>
            Connect Gitea
          </Button>
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : giteaConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No Gitea instances connected yet.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {giteaConnections.map((conn) => (
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
                    primary={conn.display_name || conn.account_identifier}
                    secondary={`${conn.account_identifier}${conn.connection_status ? ` · ${conn.connection_status}` : ''}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Divider sx={{ my: 2 }} />
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <ExtensionIcon /> MCP servers
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Add Model Context Protocol servers (stdio, SSE, or streamable HTTP). Set environment variables as JSON for stdio servers
        (for example <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>BLENDER_HOST</Box>,{' '}
        <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.85em' }}>BLENDER_PORT</Box>
        ). Discover tools after saving, then reference the server in Agent Factory playbook tool packs.
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Button
            variant="contained"
            onClick={() => {
              setMcpEditingId(null);
              setMcpName('');
              setMcpTransport('sse');
              setMcpUrl('');
              setMcpCommand('');
              setMcpArgsText('');
              setMcpHeadersJson('{}');
              setMcpEnvJson('{}');
              setMcpDialogOpen(true);
            }}
            disabled={mcpSaving}
            sx={{ mr: 1 }}
          >
            Add MCP server
          </Button>
          {mcpLoading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : mcpServers.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No MCP servers configured.
            </Typography>
          ) : (
            <List dense sx={{ mt: 2 }}>
              {mcpServers.map((srv) => {
                const rawTools = srv.discovered_tools;
                const tools = Array.isArray(rawTools)
                  ? rawTools
                  : [];
                const toolCount = tools.length;
                return (
                  <ListItem
                    key={srv.id}
                    alignItems="flex-start"
                    secondaryAction={
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center', justifyContent: 'flex-end' }}>
                        <Button
                          size="small"
                          onClick={async () => {
                            const sid = Number(srv.id);
                            try {
                              setMcpBusyId(sid);
                              await apiService.agentFactory.testMcpServer(srv.id);
                              await loadMcpServers();
                            } catch (e) {
                              setError(e.message || 'MCP test failed');
                            } finally {
                              setMcpBusyId(null);
                            }
                          }}
                          disabled={mcpBusyId !== null && Number(mcpBusyId) === Number(srv.id)}
                        >
                          Test
                        </Button>
                        <Button
                          size="small"
                          onClick={async () => {
                            try {
                              setMcpBusyId(Number(srv.id));
                              await apiService.agentFactory.discoverMcpServer(srv.id);
                              await loadMcpServers();
                            } catch (e) {
                              setError(e.message || 'Discovery failed');
                            } finally {
                              setMcpBusyId(null);
                            }
                          }}
                          disabled={mcpBusyId !== null && Number(mcpBusyId) === Number(srv.id)}
                        >
                          Refresh tools
                        </Button>
                        <Button
                          size="small"
                          onClick={() => {
                            setMcpEditingId(srv.id);
                            setMcpName(srv.name || '');
                            setMcpTransport(srv.transport || 'sse');
                            setMcpUrl(srv.url || '');
                            setMcpCommand(srv.command || '');
                            const a = srv.args;
                            setMcpArgsText(Array.isArray(a) ? a.map(String).join('\n') : '');
                            try {
                              setMcpHeadersJson(JSON.stringify((srv.headers && typeof srv.headers === 'object') ? srv.headers : {}, null, 2));
                            } catch {
                              setMcpHeadersJson('{}');
                            }
                            try {
                              setMcpEnvJson(JSON.stringify((srv.env && typeof srv.env === 'object') ? srv.env : {}, null, 2));
                            } catch {
                              setMcpEnvJson('{}');
                            }
                            setMcpDialogOpen(true);
                          }}
                        >
                          Edit
                        </Button>
                        <IconButton
                          edge="end"
                          aria-label="Delete MCP server"
                          onClick={async () => {
                            if (!window.confirm(`Remove MCP server "${srv.name}"?`)) return;
                            try {
                              await apiService.agentFactory.deleteMcpServer(srv.id);
                              await loadMcpServers();
                            } catch (e) {
                              setError(e.message || 'Delete failed');
                            }
                          }}
                        >
                          <LinkOffIcon />
                        </IconButton>
                      </Box>
                    }
                  >
                    <ListItemText
                      primary={`${srv.name} · ${srv.transport}`}
                      secondary={
                        <>
                          {srv.url ? `${srv.url} · ` : ''}
                          {toolCount} tool(s) discovered
                          {toolCount > 0 && (
                            <Typography component="span" variant="caption" display="block" color="text.secondary">
                              {(tools || []).slice(0, 8).map((t) => (typeof t === 'object' ? t.name : t)).filter(Boolean).join(', ')}
                              {toolCount > 8 ? '…' : ''}
                            </Typography>
                          )}
                        </>
                      }
                    />
                  </ListItem>
                );
              })}
            </List>
          )}
        </CardContent>
      </Card>

      <Dialog
        open={mcpDialogOpen}
        onClose={() => !mcpSaving && setMcpDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>{mcpEditingId != null ? 'Edit MCP server' : 'Add MCP server'}</DialogTitle>
        <DialogContent>
          <TextField margin="dense" label="Name" fullWidth value={mcpName} onChange={(e) => setMcpName(e.target.value)} />
          <FormControl fullWidth margin="dense">
            <InputLabel>Transport</InputLabel>
            <Select
              label="Transport"
              value={mcpTransport}
              onChange={(e) => setMcpTransport(e.target.value)}
            >
              <MenuItem value="sse">SSE (URL)</MenuItem>
              <MenuItem value="streamable_http">Streamable HTTP (URL)</MenuItem>
              <MenuItem value="stdio">Stdio (command)</MenuItem>
            </Select>
          </FormControl>
          {(mcpTransport === 'sse' || mcpTransport === 'streamable_http') && (
            <TextField margin="dense" label="Server URL" fullWidth value={mcpUrl} onChange={(e) => setMcpUrl(e.target.value)} />
          )}
          {mcpTransport === 'stdio' && (
            <>
              <TextField margin="dense" label="Command" fullWidth value={mcpCommand} onChange={(e) => setMcpCommand(e.target.value)} placeholder="uvx" />
              <TextField
                margin="dense"
                label="Arguments (one per line)"
                fullWidth
                multiline
                minRows={2}
                value={mcpArgsText}
                onChange={(e) => setMcpArgsText(e.target.value)}
                placeholder="blender-mcp"
                helperText="One argument per line (not a JSON array)."
              />
            </>
          )}
          {(mcpTransport === 'sse' || mcpTransport === 'streamable_http') && (
            <TextField
              margin="dense"
              label="Auth headers (JSON object)"
              fullWidth
              multiline
              minRows={2}
              value={mcpHeadersJson}
              onChange={(e) => setMcpHeadersJson(e.target.value)}
            />
          )}
          <TextField
            margin="dense"
            label="Environment variables (JSON object)"
            fullWidth
            multiline
            minRows={3}
            value={mcpEnvJson}
            onChange={(e) => setMcpEnvJson(e.target.value)}
            placeholder='{"BLENDER_HOST":"host.docker.internal","BLENDER_PORT":"9876"}'
            helperText="Passed to the MCP subprocess (stdio) or merged into the process environment. Values must be strings."
          />
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              if (!mcpSaving) {
                setMcpDialogOpen(false);
                setMcpEditingId(null);
              }
            }}
            disabled={mcpSaving}
          >
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={mcpSaving || !mcpName.trim()}
            onClick={async () => {
              let headers = {};
              try {
                headers = JSON.parse(mcpHeadersJson || '{}');
                if (typeof headers !== 'object' || headers === null || Array.isArray(headers)) headers = {};
              } catch {
                setError('Headers must be a valid JSON object');
                return;
              }
              let env = {};
              try {
                env = JSON.parse(mcpEnvJson || '{}');
                if (typeof env !== 'object' || env === null || Array.isArray(env)) env = {};
                const stringEnv = {};
                Object.keys(env).forEach((k) => {
                  if (env[k] !== undefined && env[k] !== null) {
                    stringEnv[String(k)] = String(env[k]);
                  }
                });
                env = stringEnv;
              } catch {
                setError('Environment must be a valid JSON object');
                return;
              }
              const args = mcpArgsText.split('\n').map((l) => l.trim()).filter(Boolean);
              const body = {
                name: mcpName.trim(),
                transport: mcpTransport,
                url: (mcpTransport === 'sse' || mcpTransport === 'streamable_http') ? mcpUrl.trim() : null,
                command: mcpTransport === 'stdio' ? mcpCommand.trim() : null,
                args: mcpTransport === 'stdio' ? args : [],
                headers,
                env,
                is_active: true,
              };
              try {
                setMcpSaving(true);
                setError(null);
                if (mcpEditingId != null) {
                  await apiService.agentFactory.updateMcpServer(mcpEditingId, body);
                } else {
                  await apiService.agentFactory.createMcpServer(body);
                }
                setMcpDialogOpen(false);
                setMcpEditingId(null);
                await loadMcpServers();
              } catch (err) {
                setError(err.message || 'Failed to save MCP server');
              } finally {
                setMcpSaving(false);
              }
            }}
          >
            {mcpSaving ? 'Saving…' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      <Divider sx={{ my: 2 }} />
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <SmartToyIcon /> Messaging bots
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Connect a Telegram, Discord, or Slack bot to chat with the AI assistant from those platforms. Add Twilio SMS for outbound notifications. Conversations appear in the web UI.
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
            sx={{ mr: 1 }}
          >
            Connect Discord Bot
          </Button>
          <Button
            variant="contained"
            onClick={() => setSlackDialogOpen(true)}
            disabled={botConnecting}
            sx={{ mr: 1 }}
          >
            Connect Slack Bot
          </Button>
          <Button
            variant="contained"
            onClick={() => setSmsDialogOpen(true)}
            disabled={botConnecting}
            sx={{ mr: 1 }}
          >
            Connect SMS (Twilio)
          </Button>
          {loading ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2">Loading…</Typography>
            </Box>
          ) : chatBotConnections.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              No messaging bots connected. Create a bot in Telegram (BotFather), Discord (Developer Portal), or Slack (api.slack.com/apps), or add Twilio SMS credentials.
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
                    secondary={`${conn.provider} · ${conn.provider === 'sms' ? conn.account_identifier : `@${conn.account_identifier}`} · ${(botStatuses[conn.id]?.status) || '…'}`}
                  />
                </ListItem>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      <Dialog open={imapSmtpDialogOpen} onClose={() => !imapSmtpConnecting && setImapSmtpDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect IMAP/SMTP</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Add an email account via IMAP (read) and SMTP (send). Works with Gmail (app password), Fastmail, and self-hosted servers.
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={12} sm={8}>
              <TextField margin="dense" label="IMAP host" fullWidth value={connImapHost} onChange={(e) => setConnImapHost(e.target.value)} placeholder="imap.gmail.com" />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField margin="dense" label="IMAP port" type="number" fullWidth value={connImapPort} onChange={(e) => setConnImapPort(Number(e.target.value) || 993)} />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel control={<Checkbox checked={connImapSsl} onChange={(e) => setConnImapSsl(e.target.checked)} />} label="IMAP SSL" />
            </Grid>
            <Grid item xs={12} sm={8}>
              <TextField margin="dense" label="SMTP host" fullWidth value={connSmtpHost} onChange={(e) => setConnSmtpHost(e.target.value)} placeholder="smtp.gmail.com" />
            </Grid>
            <Grid item xs={12} sm={4}>
              <TextField margin="dense" label="SMTP port" type="number" fullWidth value={connSmtpPort} onChange={(e) => setConnSmtpPort(Number(e.target.value) || 587)} />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel control={<Checkbox checked={connSmtpTls} onChange={(e) => setConnSmtpTls(e.target.checked)} />} label="SMTP TLS" />
            </Grid>
            <Grid item xs={12}>
              <TextField margin="dense" label="Username" fullWidth value={connImapSmtpUsername} onChange={(e) => setConnImapSmtpUsername(e.target.value)} placeholder="you@example.com" />
            </Grid>
            <Grid item xs={12}>
              <TextField margin="dense" label="IMAP password" type="password" fullWidth value={connImapPassword} onChange={(e) => setConnImapPassword(e.target.value)} />
            </Grid>
            <Grid item xs={12}>
              <TextField margin="dense" label="SMTP password" type="password" fullWidth value={connSmtpPassword} onChange={(e) => setConnSmtpPassword(e.target.value)} placeholder="Often same as IMAP" />
            </Grid>
            <Grid item xs={12}>
              <TextField margin="dense" label="Display name (optional)" fullWidth value={connImapSmtpDisplayName} onChange={(e) => setConnImapSmtpDisplayName(e.target.value)} />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImapSmtpDialogOpen(false)} disabled={imapSmtpConnecting}>Cancel</Button>
          <Button onClick={handleConnectImapSmtp} variant="contained" disabled={imapSmtpConnecting || !connImapHost.trim() || !connSmtpHost.trim() || !connImapSmtpUsername.trim() || !connImapPassword || !connSmtpPassword}>
            {imapSmtpConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={caldavDialogOpen} onClose={() => !caldavConnecting && setCaldavDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect CalDAV</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Add a CalDAV calendar (Google Calendar, Nextcloud, iCloud). Use the CalDAV URL and your username and password (or app password).
          </Typography>
          <TextField margin="dense" label="CalDAV URL" fullWidth value={caldavUrl} onChange={(e) => setCaldavUrl(e.target.value)} placeholder="https://caldav.calendar.google.com or your server URL" />
          <TextField margin="dense" label="Username" fullWidth value={caldavUsername} onChange={(e) => setCaldavUsername(e.target.value)} />
          <TextField margin="dense" label="Password" type="password" fullWidth value={caldavPassword} onChange={(e) => setCaldavPassword(e.target.value)} />
          <TextField margin="dense" label="Display name (optional)" fullWidth value={caldavDisplayName} onChange={(e) => setCaldavDisplayName(e.target.value)} />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCaldavDialogOpen(false)} disabled={caldavConnecting}>Cancel</Button>
          <Button onClick={handleConnectCaldav} variant="contained" disabled={caldavConnecting || !caldavUrl.trim() || !caldavUsername.trim() || !caldavPassword}>
            {caldavConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={giteaDialogOpen} onClose={() => !giteaConnecting && setGiteaDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect Gitea</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Create a personal access token in Gitea (Settings → Applications). Grant at least read access to repositories as needed.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Gitea server URL"
            fullWidth
            value={giteaApiBaseUrl}
            onChange={(e) => setGiteaApiBaseUrl(e.target.value)}
            placeholder="https://git.example.com"
          />
          <TextField
            margin="dense"
            label="Personal access token"
            type="password"
            fullWidth
            value={giteaPat}
            onChange={(e) => setGiteaPat(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Display name (optional)"
            fullWidth
            value={giteaDisplayName}
            onChange={(e) => setGiteaDisplayName(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setGiteaDialogOpen(false)} disabled={giteaConnecting}>
            Cancel
          </Button>
          <Button
            onClick={handleConnectGitea}
            variant="contained"
            disabled={giteaConnecting || !giteaApiBaseUrl.trim() || !giteaPat.trim()}
          >
            {giteaConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

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

      <Dialog open={slackDialogOpen} onClose={() => !botConnecting && setSlackDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect Slack Bot</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Create a Slack app at <Link href="https://api.slack.com/apps" target="_blank" rel="noopener noreferrer">api.slack.com/apps</Link>. Enable Socket Mode and create an App-Level Token (xapp-...) with <code>connections:write</code>. Install the app to your workspace and add Bot Token Scopes: <code>chat:write</code>, <code>channels:history</code>, <code>groups:history</code>, <code>im:history</code>, <code>im:read</code>, <code>im:write</code>, <code>files:read</code>, <code>files:write</code>, <code>users:read</code>, <code>channels:read</code>. Then paste both tokens below.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Bot token (xoxb-...)"
            type="password"
            fullWidth
            value={slackBotToken}
            onChange={(e) => setSlackBotToken(e.target.value)}
            placeholder="xoxb-..."
          />
          <TextField
            margin="dense"
            label="App token (xapp-...) for Socket Mode"
            type="password"
            fullWidth
            value={slackAppToken}
            onChange={(e) => setSlackAppToken(e.target.value)}
            placeholder="xapp-..."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSlackDialogOpen(false)} disabled={botConnecting}>Cancel</Button>
          <Button onClick={handleConnectSlack} variant="contained" disabled={botConnecting || !slackBotToken.trim() || !slackAppToken.trim()}>
            {botConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={smsDialogOpen} onClose={() => !botConnecting && setSmsDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Connect SMS (Twilio)</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" paragraph>
            Get your Account SID, Auth Token, and a Twilio phone number from the{' '}
            <Link href="https://console.twilio.com" target="_blank" rel="noopener noreferrer">Twilio Console</Link>.
            SMS is outbound-only; agents can send notifications to phone numbers via the send outbound message tool.
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Account SID"
            fullWidth
            value={smsAccountSid}
            onChange={(e) => setSmsAccountSid(e.target.value)}
            placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          />
          <TextField
            margin="dense"
            label="Auth Token"
            type="password"
            fullWidth
            value={smsAuthToken}
            onChange={(e) => setSmsAuthToken(e.target.value)}
            placeholder="..."
          />
          <TextField
            margin="dense"
            label="From Number"
            fullWidth
            value={smsFromNumber}
            onChange={(e) => setSmsFromNumber(e.target.value)}
            placeholder="+15551234567"
            helperText="E.164 format (e.g. +15551234567)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSmsDialogOpen(false)} disabled={botConnecting}>Cancel</Button>
          <Button onClick={handleConnectSMS} variant="contained" disabled={botConnecting || !smsAccountSid.trim() || !smsAuthToken.trim() || !smsFromNumber.trim()}>
            {botConnecting ? 'Connecting…' : 'Connect'}
          </Button>
        </DialogActions>
      </Dialog>

      <Divider sx={{ my: 2 }} />
      <DeviceTokensSettings />

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
