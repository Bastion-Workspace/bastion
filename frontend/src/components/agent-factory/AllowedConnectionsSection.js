/**
 * Profile-level allowlist for external connections.
 * Empty allowed_connections = no restriction (all user connections allowed at runtime).
 * Entries are id-only: { connection_id: N }.
 */

import React, { useMemo, useCallback } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  FormControlLabel,
  Radio,
  RadioGroup,
  Checkbox,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

function isToolConnection(conn) {
  const t = (conn?.connection_type || '').trim();
  return t && t !== 'chat_bot';
}

function connectionLabel(conn) {
  return conn.display_name || conn.account_identifier || `Connection ${conn.id}`;
}

function providerGroupLabel(conn) {
  const prov = (conn.provider || '').trim();
  const label = connectionLabel(conn);
  if (prov) {
    const provDisplay = prov.charAt(0).toUpperCase() + prov.slice(1);
    return `${provDisplay} — ${label}`;
  }
  return label;
}

function toAllowEntry(conn) {
  const id = Number(conn.id);
  if (!Number.isFinite(id)) return null;
  return { connection_id: id };
}

function allowedIdSet(entries) {
  const s = new Set();
  for (const e of entries || []) {
    const id = Number(e?.connection_id ?? e?.id);
    if (Number.isFinite(id)) s.add(id);
  }
  return s;
}

export default function AllowedConnectionsSection({ profile, onChange, readOnly }) {
  const { data, isLoading, error } = useQuery(
    ['connections', 'allowedConnectionsProfile'],
    () => apiService.get('/api/connections'),
    { staleTime: 60000 }
  );

  const rawList = data?.connections ?? [];
  const toolConnections = useMemo(() => rawList.filter(isToolConnection), [rawList]);

  const allowed = Array.isArray(profile?.allowed_connections) ? profile.allowed_connections : [];
  const allowedIds = useMemo(() => allowedIdSet(allowed), [allowed]);

  const mode = !allowed.length ? 'all' : 'selected';

  const commit = useCallback(
    (nextAllowed) => {
      onChange({ ...profile, allowed_connections: nextAllowed });
    },
    [profile, onChange]
  );

  const onModeChange = (e) => {
    if (readOnly) return;
    const v = e.target.value;
    if (v === 'all') {
      commit([]);
      return;
    }
    if (v === 'selected') {
      if (!toolConnections.length) {
        commit([]);
        return;
      }
      const allEntries = toolConnections.map(toAllowEntry).filter(Boolean);
      commit(allEntries);
    }
  };

  const toggleConn = (conn, checked) => {
    if (readOnly) return;
    const cid = Number(conn.id);
    if (!Number.isFinite(cid)) return;
    const nextIds = new Set(allowedIds);
    if (checked) nextIds.add(cid);
    else nextIds.delete(cid);
    const nextAllowed = [];
    for (const c of toolConnections) {
      const id = Number(c.id);
      if (Number.isFinite(id) && nextIds.has(id)) {
        nextAllowed.push({ connection_id: id });
      }
    }
    if (nextAllowed.length === 0) {
      commit([]);
      return;
    }
    commit(nextAllowed);
  };

  if (isLoading) {
    return (
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Loading connections…
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Alert severity="warning">Could not load connections. Try again later.</Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="subtitle1" gutterBottom>
          External connections for tools
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Limit which external accounts this agent may use. Leave unrestricted to allow any
          account you connect in Settings. Playbook steps can narrow further per step.
        </Typography>

        <RadioGroup value={mode} onChange={onModeChange}>
          <FormControlLabel
            value="all"
            control={<Radio />}
            label="All External Accounts"
            disabled={readOnly}
          />
          <FormControlLabel
            value="selected"
            control={<Radio />}
            label="Specific External Accounts (select which)"
            disabled={readOnly || toolConnections.length === 0}
          />
        </RadioGroup>

        {mode === 'selected' && toolConnections.length === 0 && (
          <Alert severity="info" sx={{ mt: 1 }}>
            No tool-capable connections yet. Add them under Settings → External connections.
          </Alert>
        )}

        {mode === 'selected' && toolConnections.length > 0 && (
          <Box sx={{ mt: 1, pl: 1 }}>
            {toolConnections.map((conn) => {
              const cid = Number(conn.id);
              const checked = allowedIds.has(cid);
              return (
                <FormControlLabel
                  key={cid}
                  control={
                    <Checkbox
                      size="small"
                      checked={checked}
                      disabled={readOnly}
                      onChange={(e) => toggleConn(conn, e.target.checked)}
                    />
                  }
                  label={<Typography variant="body2">{providerGroupLabel(conn)}</Typography>}
                  sx={{ display: 'flex', ml: 0 }}
                />
              );
            })}
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
