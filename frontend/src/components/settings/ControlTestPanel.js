/**
 * Per-control test panel for the Add/Edit Control dialog.
 * When pane is saved: calls executeAction (same as status bar). When unsaved: calls testEndpoint.
 * Resolves param_source from controlValues when provided. Shows raw response and allows path selection for value_path.
 */

import React, { useState, useCallback } from 'react';
import { Box, Button, Typography, CircularProgress, Alert } from '@mui/material';
import { Science } from '@mui/icons-material';
import apiService from '../../services/apiService';
import JsonResponseViewer from '../agent-factory/JsonResponseViewer';

function buildParams(control, controlValues = {}) {
  const params = {};
  for (const src of control.param_source || []) {
    if (src.param && src.from_control_id && controlValues[src.from_control_id] !== undefined && controlValues[src.from_control_id] !== null && controlValues[src.from_control_id] !== '') {
      params[src.param] = controlValues[src.from_control_id];
    }
  }
  return params;
}

export default function ControlTestPanel({
  paneId,
  control,
  connectorId,
  connectionId,
  credentials,
  allControls = [],
  controlValues = {},
  onPathSelect,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [rawResponse, setRawResponse] = useState(null);

  const runTest = useCallback(async () => {
    if (!control?.endpoint_id) return;
    setLoading(true);
    setError('');
    setRawResponse(null);
    const params = buildParams(control, controlValues);
    try {
      if (paneId) {
        const result = await apiService.controlPanes.executeAction(paneId, control.endpoint_id, params);
        setRawResponse(result?.raw_response ?? result ?? null);
        if (result?.error) setError(result.error);
      } else {
        const body = {
          connector_id: connectorId,
          endpoint_id: control.endpoint_id,
          params,
        };
        if (connectionId != null) {
          body.connection_id = connectionId;
        } else if (credentials && Object.keys(credentials).length > 0) {
          body.credentials = credentials;
        }
        const res = await apiService.controlPanes.testEndpoint(body);
        setRawResponse(res?.raw_response ?? res ?? null);
        if (res?.error) setError(res.error);
      }
    } catch (e) {
      setError(e?.message || String(e));
      setRawResponse(null);
    } finally {
      setLoading(false);
    }
  }, [paneId, control, connectorId, connectionId, credentials, controlValues]);

  if (!control?.endpoint_id) return null;

  return (
    <Box sx={{ mt: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Test this control
      </Typography>
      <Button
        size="small"
        variant="outlined"
        startIcon={loading ? <CircularProgress size={16} /> : <Science />}
        disabled={loading || !control.endpoint_id}
        onClick={runTest}
        sx={{ mb: 1 }}
      >
        Test
      </Button>
      {paneId ? (
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
          Runs the pane endpoint with saved credentials.
        </Typography>
      ) : (
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
          Uses connector credentials from the form (no pane save required).
        </Typography>
      )}
      {error && (
        <Alert severity="error" sx={{ mb: 1 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      {rawResponse != null && (
        <JsonResponseViewer
          data={rawResponse}
          onPathSelect={onPathSelect}
          maxHeight={220}
        />
      )}
    </Box>
  );
}
