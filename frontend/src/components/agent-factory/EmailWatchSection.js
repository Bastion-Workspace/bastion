/**
 * Email Watch section: which email connections this agent watches; trigger on new matching emails.
 * Persists to profile.watch_config.email_watches; backend syncs to agent_email_watches on save.
 */

import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Switch,
  TextField,
  CircularProgress,
} from '@mui/material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

export default function EmailWatchSection({ profile, onChange, compact }) {
  const { data: connectionsData, isLoading } = useQuery(
    'connections-email',
    () => apiService.get('/api/connections?connection_type=email'),
    { staleTime: 60 * 1000 }
  );
  const connections = connectionsData?.connections ?? [];

  if (!profile) return null;

  const watchConfig = profile.watch_config || {};
  const emailWatches = watchConfig.email_watches || [];

  const getWatch = (connectionId) =>
    emailWatches.find((w) => String(w.connection_id) === String(connectionId));

  const setEmailWatches = (next) => {
    onChange({
      ...profile,
      watch_config: { ...watchConfig, email_watches: next },
    });
  };

  const setWatch = (connectionId, enabled, overrides = {}) => {
    const rest = emailWatches.filter((w) => String(w.connection_id) !== String(connectionId));
    if (!enabled) {
      setEmailWatches(rest);
      return;
    }
    const existing = getWatch(connectionId);
    setEmailWatches([
      ...rest,
      {
        connection_id: typeof connectionId === 'number' ? connectionId : parseInt(connectionId, 10),
        subject_pattern: overrides.subject_pattern ?? existing?.subject_pattern ?? '',
        sender_pattern: overrides.sender_pattern ?? existing?.sender_pattern ?? '',
        folder: overrides.folder ?? existing?.folder ?? 'Inbox',
      },
    ]);
  };

  const updateWatchField = (connectionId, field, value) => {
    const existing = getWatch(connectionId);
    if (!existing) return;
    setWatch(connectionId, true, { ...existing, [field]: value });
  };

  return (
    <Box>
      {!compact && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Watch email inboxes and trigger on new messages. Filter by subject or sender.
        </Typography>
      )}
      {isLoading ? (
        <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
          <CircularProgress size={24} />
        </Box>
      ) : connections.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No email connections. Add one in Settings → External connections.
        </Typography>
      ) : (
        <List disablePadding dense>
          {connections.map((conn) => {
            const watch = getWatch(conn.id);
            const watching = !!watch;
            return (
              <ListItem key={conn.id} disablePadding sx={{ flexWrap: 'wrap', py: 0.5 }}>
                <ListItemText
                  primary={conn.display_name || conn.account_identifier || `Connection ${conn.id}`}
                  primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', width: '100%' }}>
                  <Switch
                    size="small"
                    checked={watching}
                    onChange={(e) => setWatch(conn.id, e.target.checked)}
                  />
                  {watching && (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5, width: '100%' }}>
                      <TextField
                        size="small"
                        label="Subject contains"
                        placeholder="e.g. invoice"
                        value={watch?.subject_pattern ?? ''}
                        onChange={(e) => updateWatchField(conn.id, 'subject_pattern', e.target.value)}
                        sx={{ minWidth: 150, flex: 1 }}
                      />
                      <TextField
                        size="small"
                        label="Sender contains"
                        placeholder="e.g. @acme.com"
                        value={watch?.sender_pattern ?? ''}
                        onChange={(e) => updateWatchField(conn.id, 'sender_pattern', e.target.value)}
                        sx={{ minWidth: 150, flex: 1 }}
                      />
                      <TextField
                        size="small"
                        label="Folder"
                        value={watch?.folder ?? 'Inbox'}
                        onChange={(e) => updateWatchField(conn.id, 'folder', e.target.value)}
                        sx={{ minWidth: 100 }}
                      />
                    </Box>
                  )}
                </Box>
              </ListItem>
            );
          })}
        </List>
      )}
    </Box>
  );
}

export function emailWatchSummary(profile) {
  const watches = profile?.watch_config?.email_watches || [];
  return watches.length ? `${watches.length} inbox${watches.length !== 1 ? 'es' : ''}` : '';
}
