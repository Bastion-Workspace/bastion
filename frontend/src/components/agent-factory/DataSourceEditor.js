/**
 * Standalone data connection (connector) editor: build definition and test endpoints.
 * Shows usage warning when the connector is attached to agents.
 * Uses ConnectorBuilder for identity, auth (including OAuth), endpoints, and inline test with raw JSON.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { Delete, Download } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import UsageWarningBanner from './UsageWarningBanner';
import ConnectorBuilder from './ConnectorBuilder';

const SAVE_DEBOUNCE_MS = 600;

export default function DataSourceEditor({ connectorId, onCloseEntityTab }) {
  const queryClient = useQueryClient();
  const [localConnector, setLocalConnector] = useState(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [exportLoading, setExportLoading] = useState(false);
  const saveTimeoutRef = useRef(null);
  const pendingRef = useRef(null);

  const { data: connector, isLoading: connectorLoading, error: connectorError } = useQuery(
    ['agentFactoryConnector', connectorId],
    () => apiService.agentFactory.getConnector(connectorId),
    { enabled: !!connectorId, retry: false }
  );
  const { data: usageAgents = [] } = useQuery(
    ['agentFactoryConnectorUsage', connectorId],
    () => apiService.agentFactory.getConnectorUsage(connectorId),
    { enabled: !!connectorId, retry: false }
  );

  const updateConnectorMutation = useMutation(
    (body) => apiService.agentFactory.updateConnector(connectorId, body),
    {
      onSuccess: (data) => {
        queryClient.setQueryData(['agentFactoryConnector', connectorId], data);
        queryClient.invalidateQueries('agentFactoryConnectors');
        setLocalConnector(null);
        pendingRef.current = null;
      },
    }
  );
  const deleteConnectorMutation = useMutation(
    (force) => apiService.agentFactory.deleteConnector(connectorId, force),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryConnectors');
        setDeleteOpen(false);
        onCloseEntityTab?.('datasource', connectorId);
      },
    }
  );

  useEffect(() => {
    setLocalConnector(null);
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    pendingRef.current = null;
  }, [connectorId]);

  useEffect(() => {
    return () => {
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
    };
  }, []);

  const handleChange = useCallback(
    (next) => {
      setLocalConnector(next);
      pendingRef.current = next;
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = setTimeout(() => {
        const pending = pendingRef.current;
        if (!pending || !connectorId) {
          saveTimeoutRef.current = null;
          return;
        }
        const body = {};
        if (pending.name !== undefined) body.name = pending.name;
        if (pending.description !== undefined) body.description = pending.description;
        if (pending.connector_type !== undefined) body.connector_type = pending.connector_type;
        if (pending.definition !== undefined) body.definition = pending.definition;
        if (pending.requires_auth !== undefined) body.requires_auth = pending.requires_auth;
        if (pending.auth_fields !== undefined) body.auth_fields = pending.auth_fields;
        updateConnectorMutation.mutate(body);
        saveTimeoutRef.current = null;
      }, SAVE_DEBOUNCE_MS);
    },
    [connectorId, updateConnectorMutation]
  );

  const displayConnector = localConnector ?? connector;

  if (!connectorId) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Select a data connection from the sidebar, or create one from a template or custom.
        </Typography>
      </Box>
    );
  }

  if (connectorLoading || connectorError) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        {connectorError && (
          <Alert severity="error">
            Failed to load connector. It may have been deleted.
          </Alert>
        )}
        {connectorLoading && <CircularProgress />}
      </Box>
    );
  }

  if (!displayConnector) return null;

  const isLocked = !!displayConnector.is_locked;

  const handleLockToggle = (e) => {
    const locked = e.target.checked;
    updateConnectorMutation.mutate({ is_locked: locked });
    setLocalConnector((prev) => (prev ? { ...prev, is_locked: locked } : null));
  };

  const handleExport = async () => {
    if (!connectorId || exportLoading) return;
    setExportLoading(true);
    try {
      const res = await apiService.agentFactory.exportConnectors([connectorId]);
      const text = await res.text();
      const blob = new Blob([text], { type: 'application/x-yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const connectorName = displayConnector.name || 'connector';
      const sanitizedName = connectorName.replace(/[^a-z0-9]/gi, '-').toLowerCase();
      a.download = `${sanitizedName}.yaml`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export connector failed:', err);
    } finally {
      setExportLoading(false);
    }
  };

  return (
    <Box sx={{ p: 2, overflow: 'auto', maxWidth: 720, flex: 1, minHeight: 0 }}>
      <UsageWarningBanner resourceLabel="This data connection" agents={usageAgents} />

      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap', gap: 1 }}>
        <Typography variant="h6" noWrap>
          {displayConnector.name || 'Unnamed connector'}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FormControlLabel
            control={
              <Switch
                checked={isLocked}
                onChange={handleLockToggle}
                disabled={updateConnectorMutation.isLoading}
                color="primary"
              />
            }
            label={isLocked ? 'Locked' : 'Unlocked'}
            labelPlacement="start"
          />
          <Button
            size="small"
            variant="outlined"
            startIcon={<Download />}
            onClick={handleExport}
            disabled={exportLoading}
          >
            Export
          </Button>
          <Button
            size="small"
            color="error"
            variant="outlined"
            startIcon={<Delete />}
            onClick={() => setDeleteOpen(true)}
            disabled={isLocked}
          >
            Delete
          </Button>
        </Box>
      </Box>

      <ConnectorBuilder
        connectorId={connectorId}
        connector={displayConnector}
        onChange={handleChange}
        readOnly={isLocked}
      />

      <Dialog open={deleteOpen} onClose={() => !deleteConnectorMutation.isLoading && setDeleteOpen(false)}>
        <DialogTitle>Delete connector</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete <strong>{displayConnector.name || 'this connector'}</strong>?
            {usageAgents.length > 0 && (
              <>
                {' '}
                It is attached to {usageAgents.length} agent(s). Detach first or force delete.
              </>
            )}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteOpen(false)} disabled={deleteConnectorMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteConnectorMutation.mutate(usageAgents.length > 0)}
            disabled={deleteConnectorMutation.isLoading}
          >
            {deleteConnectorMutation.isLoading ? 'Deleting…' : usageAgents.length > 0 ? 'Force delete' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
