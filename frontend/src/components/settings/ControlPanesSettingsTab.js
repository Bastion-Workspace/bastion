import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  IconButton,
  Switch,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Add, Edit, Delete, Tune, Science } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import savedArtifactService from '../../services/savedArtifactService';
import JsonResponseViewer from '../agent-factory/JsonResponseViewer';
import ControlTestPanel from './ControlTestPanel';

const ICON_OPTIONS = [
  'Tune',
  'Settings',
  'VolumeUp',
  'PlayArrow',
  'TouchApp',
  'Dashboard',
  'ToggleOn',
  'SmartToy',
];

const BUTTON_ICON_OPTIONS = [
  'PlayArrow',
  'Pause',
  'Stop',
  'SkipNext',
  'SkipPrevious',
  'Close',
  'VolumeUp',
  'VolumeOff',
  'Refresh',
  'PowerSettingsNew',
  'Lightbulb',
  'Thermostat',
];

const CONTROL_TYPES = [
  { value: 'slider', label: 'Slider' },
  { value: 'dropdown', label: 'Dropdown' },
  { value: 'toggle', label: 'Toggle' },
  { value: 'button', label: 'Button' },
  { value: 'text_display', label: 'Text display' },
];

function defaultControl(type) {
  const base = {
    id: `control_${Date.now()}`,
    type,
    label: '',
    endpoint_id: '',
    param_key: '',
    refresh_endpoint_id: '',
    value_path: '',
    param_source: [],
    refresh_param_source: [],
  };
  if (type === 'slider') return { ...base, min: 0, max: 100, step: 1 };
  if (type === 'dropdown') return { ...base, options_endpoint_id: '', options: [], options_label_path: 'name', options_value_path: 'id' };
  if (type === 'button') return { ...base, icon: 'PlayArrow' };
  return base;
}

export default function ControlPanesSettingsTab() {
  const queryClient = useQueryClient();
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingPane, setEditingPane] = useState(null);
  const [form, setForm] = useState({
    name: '',
    icon: 'Tune',
    pane_type: 'connector',
    connector_id: '',
    artifact_id: '',
    artifact_popover_width: 360,
    artifact_popover_height: 400,
    credentials_encrypted: {},
    connection_id: null,
    controls: [],
    is_visible: true,
    sort_order: 0,
    refresh_interval: 0,
  });
  const [addControlOpen, setAddControlOpen] = useState(false);
  const [editingControlIndex, setEditingControlIndex] = useState(null);
  const [newControl, setNewControl] = useState(defaultControl('button'));
  const [addParamSourceParam, setAddParamSourceParam] = useState('');
  const [addParamSourceFrom, setAddParamSourceFrom] = useState('');
  const [addRefreshParamSourceParam, setAddRefreshParamSourceParam] = useState('');
  const [addRefreshParamSourceFrom, setAddRefreshParamSourceFrom] = useState('');

  const [testEndpointId, setTestEndpointId] = useState('');
  const [testParams, setTestParams] = useState('{}');
  const [testLoading, setTestLoading] = useState(false);
  const [testError, setTestError] = useState('');
  const [testRawResponse, setTestRawResponse] = useState(null);
  const [activePathTarget, setActivePathTarget] = useState(null);

  const [healthStatus, setHealthStatus] = useState({});
  const healthProbedRef = useRef(false);

  const { data: panes = [], isLoading } = useQuery(
    'controlPanes',
    () => apiService.controlPanes.listPanes()
  );
  const { data: connectors = [] } = useQuery(
    'agentFactoryConnectors',
    () => apiService.agentFactory.listConnectors(),
    { enabled: editorOpen }
  );
  const { data: savedArtifactsPayload } = useQuery(
    ['savedArtifactsList'],
    () => savedArtifactService.list(),
    { enabled: editorOpen }
  );
  const savedArtifactsList = savedArtifactsPayload?.artifacts || [];
  const [connectorIdForEndpoints, setConnectorIdForEndpoints] = useState('');
  const { data: connectorDetail } = useQuery(
    ['connector', connectorIdForEndpoints],
    () => apiService.agentFactory.getConnector(connectorIdForEndpoints),
    { enabled: !!connectorIdForEndpoints && editorOpen }
  );
  const endpointIds = connectorDetail?.definition?.endpoints
    ? Object.keys(connectorDetail.definition.endpoints)
    : [];

  useEffect(() => {
    if (editorOpen && form.connector_id && endpointIds.length > 0 && !testEndpointId) {
      setTestEndpointId(endpointIds[0]);
    }
  }, [editorOpen, form.connector_id, endpointIds, testEndpointId]);

  useEffect(() => {
    if (panes.length === 0 || healthProbedRef.current) return;
    healthProbedRef.current = true;
    const run = async () => {
      const next = {};
      panes.forEach((p) => { next[p.id] = 'loading'; });
      setHealthStatus(next);
      const results = await Promise.all(
        panes.map(async (pane) => {
          try {
            if ((pane.pane_type || 'connector') === 'artifact') {
              return { id: pane.id, status: 'ok' };
            }
            let endpointId = (pane.controls || []).find((c) => c.refresh_endpoint_id)?.refresh_endpoint_id;
            if (!endpointId) {
              if (!pane.connector_id) return { id: pane.id, status: 'error' };
              const conn = await apiService.agentFactory.getConnector(pane.connector_id);
              const ids = conn?.definition?.endpoints ? Object.keys(conn.definition.endpoints) : [];
              endpointId = ids[0];
            }
            if (!endpointId) return { id: pane.id, status: 'error' };
            const body = {
              connector_id: pane.connector_id,
              endpoint_id: endpointId,
              params: {},
            };
            if (pane.connection_id != null) body.connection_id = pane.connection_id;
            else if (pane.credentials_encrypted && Object.keys(pane.credentials_encrypted).length > 0) {
              body.credentials = pane.credentials_encrypted;
            }
            await apiService.controlPanes.testEndpoint(body);
            return { id: pane.id, status: 'ok' };
          } catch {
            return { id: pane.id, status: 'error' };
          }
        })
      );
      const nextStatus = {};
      results.forEach((r) => { nextStatus[r.id] = r.status; });
      setHealthStatus((prev) => ({ ...prev, ...nextStatus }));
    };
    run();
  }, [panes]);

  const createMutation = useMutation(
    (body) => apiService.controlPanes.createPane(body),
    { onSuccess: () => { queryClient.invalidateQueries('controlPanes'); setEditorOpen(false); resetForm(); } }
  );
  const updateMutation = useMutation(
    ({ id, body }) => apiService.controlPanes.updatePane(id, body),
    { onSuccess: () => { queryClient.invalidateQueries('controlPanes'); setEditorOpen(false); setEditingPane(null); resetForm(); } }
  );
  const deleteMutation = useMutation(
    (id) => apiService.controlPanes.deletePane(id),
    { onSuccess: () => queryClient.invalidateQueries('controlPanes') }
  );
  const visibilityMutation = useMutation(
    ({ id, isVisible }) => apiService.controlPanes.toggleVisibility(id, isVisible),
    { onSuccess: () => queryClient.invalidateQueries('controlPanes') }
  );

  function resetForm() {
    setForm({
      name: '',
      icon: 'Tune',
      pane_type: 'connector',
      connector_id: '',
      artifact_id: '',
      artifact_popover_width: 360,
      artifact_popover_height: 400,
      credentials_encrypted: {},
      connection_id: null,
      controls: [],
      is_visible: true,
      sort_order: 0,
      refresh_interval: 0,
    });
    setConnectorIdForEndpoints('');
    setTestEndpointId('');
    setTestParams('{}');
    setTestError('');
    setTestRawResponse(null);
    setActivePathTarget(null);
  }

  const handleAddPane = () => {
    setEditingPane(null);
    resetForm();
    setEditorOpen(true);
  };

  const handleEditPane = (pane) => {
    setEditingPane(pane);
    const creds = pane.credentials_encrypted;
    const credsObj =
      creds == null
        ? {}
        : typeof creds === 'object' && !Array.isArray(creds)
          ? creds
          : typeof creds === 'string'
            ? (() => {
                try {
                  return JSON.parse(creds);
                } catch {
                  return {};
                }
              })()
            : {};
    setForm({
      name: pane.name || '',
      icon: pane.icon || 'Tune',
      pane_type: pane.pane_type || 'connector',
      connector_id: pane.connector_id || '',
      artifact_id: pane.artifact_id || '',
      artifact_popover_width: pane.artifact_popover_width ?? 360,
      artifact_popover_height: pane.artifact_popover_height ?? 400,
      credentials_encrypted: credsObj,
      connection_id: pane.connection_id ?? null,
      controls: Array.isArray(pane.controls) ? pane.controls : [],
      is_visible: pane.is_visible !== false,
      sort_order: pane.sort_order || 0,
      refresh_interval: pane.refresh_interval ?? 0,
    });
    setConnectorIdForEndpoints(pane.connector_id || '');
    setEditorOpen(true);
  };

  const handleSavePane = () => {
    if (!form.name.trim()) return;
    const creds = form.credentials_encrypted;
    const credsObj =
      creds == null
        ? {}
        : typeof creds === 'object' && !Array.isArray(creds)
          ? creds
          : typeof creds === 'string'
            ? (() => {
                try {
                  return JSON.parse(creds);
                } catch {
                  return {};
                }
              })()
            : {};
    const isArtifact = (form.pane_type || 'connector') === 'artifact';
    if (isArtifact) {
      if (!form.artifact_id) return;
      const body = {
        name: form.name.trim(),
        icon: form.icon,
        pane_type: 'artifact',
        artifact_id: form.artifact_id,
        artifact_popover_width: Number(form.artifact_popover_width) || 360,
        artifact_popover_height: Number(form.artifact_popover_height) || 400,
        is_visible: form.is_visible,
        sort_order: form.sort_order,
        refresh_interval: form.refresh_interval,
      };
      if (editingPane) {
        updateMutation.mutate({ id: editingPane.id, body });
      } else {
        createMutation.mutate(body);
      }
      return;
    }
    if (!form.connector_id) return;
    const body = {
      name: form.name.trim(),
      icon: form.icon,
      pane_type: 'connector',
      connector_id: form.connector_id,
      credentials_encrypted: credsObj,
      connection_id: form.connection_id,
      controls: form.controls || [],
      is_visible: form.is_visible,
      sort_order: form.sort_order,
      refresh_interval: form.refresh_interval,
    };
    if (editingPane) {
      updateMutation.mutate({ id: editingPane.id, body });
    } else {
      createMutation.mutate(body);
    }
  };

  const handleAddControl = () => {
    setEditingControlIndex(null);
    setNewControl(defaultControl('button'));
    setAddParamSourceParam('');
    setAddParamSourceFrom('');
    setAddRefreshParamSourceParam('');
    setAddRefreshParamSourceFrom('');
    setAddControlOpen(true);
  };

  const handleEditControl = (index) => {
    const control = form.controls[index];
    if (!control) return;
    setEditingControlIndex(index);
    setNewControl({ ...defaultControl(control.type), ...control, param_source: (control.param_source && Array.isArray(control.param_source)) ? control.param_source : [], refresh_param_source: (control.refresh_param_source && Array.isArray(control.refresh_param_source)) ? control.refresh_param_source : [] });
    setAddParamSourceParam('');
    setAddParamSourceFrom('');
    setAddRefreshParamSourceParam('');
    setAddRefreshParamSourceFrom('');
    setAddControlOpen(true);
  };

  const handleSaveControl = () => {
    if (!newControl.label || !newControl.endpoint_id) return;
    if (editingControlIndex !== null) {
      setForm((prev) => {
        const next = [...(prev.controls || [])];
        next[editingControlIndex] = { ...newControl };
        return { ...prev, controls: next };
      });
      setEditingControlIndex(null);
    } else {
      setForm((prev) => ({ ...prev, controls: [...(prev.controls || []), { ...newControl }] }));
    }
    setAddControlOpen(false);
  };

  const removeControl = (index) => {
    setForm((prev) => ({
      ...prev,
      controls: prev.controls.filter((_, i) => i !== index),
    }));
  };

  const runTest = async () => {
    if (!form.connector_id || !testEndpointId) return;
    setTestLoading(true);
    setTestError('');
    setTestRawResponse(null);
    let params = {};
    try {
      params = JSON.parse(testParams.trim() || '{}');
    } catch {
      params = {};
    }
    try {
      const body = {
        connector_id: form.connector_id,
        endpoint_id: testEndpointId,
        params,
      };
      if (form.connection_id != null) {
        body.connection_id = form.connection_id;
      } else if (form.credentials_encrypted && Object.keys(form.credentials_encrypted).length > 0) {
        body.credentials = form.credentials_encrypted;
      }
      const res = await apiService.controlPanes.testEndpoint(body);
      setTestRawResponse(res?.raw_response ?? res ?? null);
      if (res?.error) setTestError(res.error);
    } catch (e) {
      setTestError(e?.message || String(e));
      setTestRawResponse(null);
    } finally {
      setTestLoading(false);
    }
  };

  const handleTestPathSelect = (path) => {
    if (activePathTarget) {
      setNewControl((c) => ({ ...c, [activePathTarget]: path }));
    }
  };

  const apiKey = form.credentials_encrypted?.api_key ?? '';

  return (
    <Box>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Control Panes
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Add custom control panes to the status bar (icons to the right of the music controls). Each pane is wired to a data connection and can show sliders, dropdowns, toggles, buttons, or text.
      </Typography>

      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
          <CircularProgress />
        </Box>
      ) : (
        <>
          <Button startIcon={<Add />} variant="outlined" onClick={handleAddPane} sx={{ mb: 2 }}>
            Add Control Pane
          </Button>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {panes.map((pane) => {
              const status = healthStatus[pane.id];
              return (
                <Card key={pane.id} variant="outlined">
                  <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Tune sx={{ color: 'text.secondary' }} />
                        {status && (
                          <Box
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              bgcolor: status === 'ok' ? 'success.main' : status === 'error' ? 'error.main' : 'action.disabled',
                              flexShrink: 0,
                            }}
                            title={status === 'ok' ? 'Connected' : status === 'error' ? 'Connection failed' : 'Checking...'}
                            aria-label={status === 'ok' ? 'Connected' : status === 'error' ? 'Connection failed' : 'Checking'}
                          />
                        )}
                        <Typography variant="subtitle1">{pane.name}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {(pane.pane_type || 'connector') === 'artifact'
                            ? `artifact ${pane.artifact_id || ''}`
                            : `(${pane.connector_name || pane.connector_id})`}
                        </Typography>
                      </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Typography variant="caption">Show in status bar</Typography>
                      <Switch
                        size="small"
                        checked={pane.is_visible !== false}
                        onChange={(e) =>
                          visibilityMutation.mutate({ id: pane.id, isVisible: e.target.checked })
                        }
                      />
                      <IconButton size="small" onClick={() => handleEditPane(pane)} aria-label="Edit">
                        <Edit fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => deleteMutation.mutate(pane.id)}
                        aria-label="Delete"
                      >
                        <Delete fontSize="small" />
                      </IconButton>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
              );
            })}
          </Box>
        </>
      )}

      <Dialog open={editorOpen} onClose={() => setEditorOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>{editingPane ? 'Edit Control Pane' : 'New Control Pane'}</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2, mt: 1 }}>
            <Box>
              <TextField
                fullWidth
                label="Name"
                value={form.name}
                onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
                sx={{ mb: 2 }}
              />
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Icon</InputLabel>
                <Select
                  value={form.icon}
                  label="Icon"
                  onChange={(e) => setForm((p) => ({ ...p, icon: e.target.value }))}
                >
                  {ICON_OPTIONS.map((opt) => (
                    <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Pane type</InputLabel>
                <Select
                  value={form.pane_type || 'connector'}
                  label="Pane type"
                  onChange={(e) => setForm((p) => ({ ...p, pane_type: e.target.value }))}
                >
                  <MenuItem value="connector">Data connector</MenuItem>
                  <MenuItem value="artifact">Saved artifact</MenuItem>
                </Select>
              </FormControl>
              {(form.pane_type || 'connector') === 'artifact' ? (
                <>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Saved artifact</InputLabel>
                    <Select
                      value={form.artifact_id}
                      label="Saved artifact"
                      onChange={(e) => setForm((p) => ({ ...p, artifact_id: e.target.value }))}
                    >
                      <MenuItem value="">
                        <em>Select…</em>
                      </MenuItem>
                      {savedArtifactsList.map((a) => (
                        <MenuItem key={a.id} value={a.id}>
                          {a.title} ({a.artifact_type})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <TextField
                    fullWidth
                    type="number"
                    label="Popover width (px)"
                    inputProps={{ min: 200, max: 1200 }}
                    value={form.artifact_popover_width}
                    onChange={(e) =>
                      setForm((p) => ({
                        ...p,
                        artifact_popover_width: Math.max(200, Math.min(1200, parseInt(e.target.value, 10) || 360)),
                      }))
                    }
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    type="number"
                    label="Popover height (px)"
                    inputProps={{ min: 120, max: 1600 }}
                    value={form.artifact_popover_height}
                    onChange={(e) =>
                      setForm((p) => ({
                        ...p,
                        artifact_popover_height: Math.max(120, Math.min(1600, parseInt(e.target.value, 10) || 400)),
                      }))
                    }
                    sx={{ mb: 2 }}
                  />
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    The artifact runs in the background while the app is open. State syncs with the same saved artifact on the home dashboard when both use the same artifact ID.
                  </Typography>
                </>
              ) : (
                <>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel>Connector</InputLabel>
                    <Select
                      value={form.connector_id}
                      label="Connector"
                      onChange={(e) => {
                        setForm((p) => ({ ...p, connector_id: e.target.value }));
                        setConnectorIdForEndpoints(e.target.value);
                        setTestEndpointId('');
                      }}
                    >
                      {connectors.map((c) => (
                        <MenuItem key={c.id} value={c.id}>{c.name}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <TextField
                    fullWidth
                    type="password"
                    label="API Key (if required by connector)"
                    value={apiKey}
                    onChange={(e) =>
                      setForm((p) => ({
                        ...p,
                        credentials_encrypted: { ...(p.credentials_encrypted || {}), api_key: e.target.value },
                      }))
                    }
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    type="number"
                    inputProps={{ min: 0, step: 1 }}
                    label="Refresh interval (seconds)"
                    helperText="0 = off; 5, 10, or 30 recommended for live state"
                    value={form.refresh_interval ?? 0}
                    onChange={(e) => setForm((p) => ({ ...p, refresh_interval: Math.max(0, parseInt(e.target.value, 10) || 0) }))}
                    sx={{ mb: 2 }}
                  />
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Controls</Typography>
                    <List dense>
                      {form.controls.map((c, idx) => {
                        const bindingSummary = (c.param_source && c.param_source.length > 0)
                          ? c.param_source
                            .map((src) => `${src.param || '?'} ← ${form.controls.find((o) => o.id === src.from_control_id)?.label || src.from_control_id || '?'}`)
                            .join(', ')
                          : '';
                        const primary = bindingSummary
                          ? `${c.type}: ${c.label || c.endpoint_id || '—'} (${bindingSummary})`
                          : `${c.type}: ${c.label || c.endpoint_id || '—'}`;
                        return (
                          <ListItem key={c.id || idx}>
                            <ListItemText primary={primary} />
                            <ListItemSecondaryAction>
                            <IconButton size="small" onClick={() => handleEditControl(idx)} aria-label="Edit control">
                              <Edit fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => removeControl(idx)} aria-label="Remove control">
                              <Delete fontSize="small" />
                            </IconButton>
                          </ListItemSecondaryAction>
                        </ListItem>
                        );
                      })}
                    </List>
                    <Button size="small" startIcon={<Add />} onClick={handleAddControl}>
                      Add Control
                    </Button>
                  </Box>
                </>
              )}
            </Box>
            {(form.pane_type || 'connector') === 'connector' ? (
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Test endpoint</Typography>
              <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                <InputLabel>Endpoint</InputLabel>
                <Select
                  value={testEndpointId}
                  label="Endpoint"
                  onChange={(e) => setTestEndpointId(e.target.value)}
                  disabled={!form.connector_id}
                >
                  {endpointIds.map((id) => (
                    <MenuItem key={id} value={id}>{id}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                fullWidth
                size="small"
                multiline
                minRows={2}
                label="Params (JSON)"
                placeholder='{"id": "room1"}'
                value={testParams}
                onChange={(e) => setTestParams(e.target.value)}
                sx={{ mb: 1 }}
              />
              <Button
                size="small"
                variant="outlined"
                startIcon={testLoading ? <CircularProgress size={16} /> : <Science />}
                disabled={testLoading || !form.connector_id || !testEndpointId}
                onClick={runTest}
                sx={{ mb: 1 }}
              >
                Test
              </Button>
              {testError && (
                <Alert severity="error" sx={{ mb: 1 }} onClose={() => setTestError('')}>
                  {testError}
                </Alert>
              )}
              <FormControl fullWidth size="small" sx={{ mb: 0.5 }}>
                <InputLabel>Apply path to</InputLabel>
                <Select
                  value={activePathTarget || ''}
                  label="Apply path to"
                  onChange={(e) => setActivePathTarget(e.target.value || null)}
                >
                  <MenuItem value="">(none)</MenuItem>
                  <MenuItem value="value_path">Value path</MenuItem>
                  <MenuItem value="options_label_path">Options label path</MenuItem>
                  <MenuItem value="options_value_path">Options value path</MenuItem>
                </Select>
              </FormControl>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                {activePathTarget ? 'Click a key in the response to fill the selected field.' : 'Select a field above, then click a key in the response.'}
              </Typography>
              {testRawResponse != null && (
                <JsonResponseViewer
                  data={testRawResponse}
                  onPathSelect={handleTestPathSelect}
                  maxHeight={300}
                />
              )}
            </Box>
            ) : null}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditorOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSavePane}
            disabled={
              !form.name.trim()
              || createMutation.isLoading
              || updateMutation.isLoading
              || ((form.pane_type || 'connector') === 'artifact'
                ? !form.artifact_id
                : !form.connector_id)
            }
          >
            {editingPane ? 'Update' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={addControlOpen} onClose={() => { setAddControlOpen(false); setEditingControlIndex(null); }} maxWidth="xs" fullWidth>
        <DialogTitle>{editingControlIndex !== null ? 'Edit Control' : 'Add Control'}</DialogTitle>
        <DialogContent>
          <FormControl fullWidth size="small" sx={{ mt: 1, mb: 2 }}>
            <InputLabel>Type</InputLabel>
            <Select
              value={newControl.type}
              label="Type"
              onChange={(e) => setNewControl(defaultControl(e.target.value))}
            >
              {CONTROL_TYPES.map((t) => (
                <MenuItem key={t.value} value={t.value}>{t.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            size="small"
            label="Label"
            value={newControl.label}
            onChange={(e) => setNewControl((c) => ({ ...c, label: e.target.value }))}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel>Endpoint</InputLabel>
            <Select
              value={newControl.endpoint_id}
              label="Endpoint"
              onChange={(e) => {
                const nextId = e.target.value;
                const nextDef = connectorDetail?.definition?.endpoints?.[nextId];
                const nextParams = (nextDef?.params || []).map((p) => p.name || p.id).filter(Boolean);
                const keepParam = nextParams.includes(newControl.param_key);
                setNewControl((c) => ({ ...c, endpoint_id: nextId, param_key: keepParam ? c.param_key : '' }));
              }}
            >
              {endpointIds.map((id) => (
                <MenuItem key={id} value={id}>{id}</MenuItem>
              ))}
            </Select>
          </FormControl>
          {(() => {
            const selectedEndpointDef = connectorDetail?.definition?.endpoints?.[newControl.endpoint_id];
            const endpointParams = selectedEndpointDef?.params || [];
            return endpointParams.length > 0 ? (
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                  Endpoint expects
                </Typography>
                <List dense disablePadding sx={{ bgcolor: 'action.hover', borderRadius: 1, px: 1, py: 0.5 }}>
                  {endpointParams.map((p, i) => (
                    <ListItem key={i} disablePadding sx={{ minHeight: 28 }}>
                      <ListItemText
                        primary={
                          <Typography variant="body2" component="span">
                            <Typography component="span" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                              {p.name || p.id || '?'}
                            </Typography>
                            {p.in && (
                              <Typography component="span" sx={{ color: 'text.secondary', ml: 0.5 }}>
                                ({p.in}{p.required ? ', required' : ''}{p.default != null ? `, default: ${p.default}` : ''})
                              </Typography>
                            )}
                            {p.description && (
                              <Typography component="span" sx={{ color: 'text.secondary', display: 'block', fontSize: '0.75rem' }}>
                                {p.description}
                              </Typography>
                            )}
                          </Typography>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            ) : null;
          })()}
          {(() => {
            const selectedEndpointDef = connectorDetail?.definition?.endpoints?.[newControl.endpoint_id];
            const endpointParamNames = (selectedEndpointDef?.params || [])
              .map((p) => p.name || p.id)
              .filter(Boolean);
            return (
              <FormControl fullWidth size="small" sx={{ mb: 2 }} disabled={!newControl.endpoint_id}>
                <InputLabel>Param key (value sent to endpoint)</InputLabel>
                <Select
                  value={newControl.param_key || ''}
                  label="Param key (value sent to endpoint)"
                  onChange={(e) => setNewControl((c) => ({ ...c, param_key: e.target.value || '' }))}
                >
                  {endpointParamNames.length === 0 ? (
                    <MenuItem value="">(No params)</MenuItem>
                  ) : (
                    endpointParamNames.map((name) => (
                      <MenuItem key={name} value={name}>{name}</MenuItem>
                    ))
                  )}
                </Select>
              </FormControl>
            );
          })()}
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
              Parameter bindings (inject another control&apos;s value when this control runs)
            </Typography>
            <List dense sx={{ py: 0 }}>
              {(newControl.param_source || []).map((src, idx) => (
                <ListItem key={idx} sx={{ py: 0, minHeight: 36 }}>
                  <ListItemText
                    primary={`${src.param || '?'} ← ${(form.controls.find((c) => c.id === src.from_control_id)?.label || src.from_control_id || '?')}`}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      size="small"
                      aria-label="Remove binding"
                      onClick={() => {
                        const next = (newControl.param_source || []).filter((_, i) => i !== idx);
                        setNewControl((c) => ({ ...c, param_source: next }));
                      }}
                    >
                      <Delete fontSize="small" />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
              <TextField
                size="small"
                label="Param name"
                placeholder="e.g. id"
                value={addParamSourceParam}
                onChange={(e) => setAddParamSourceParam(e.target.value)}
                sx={{ width: 100 }}
              />
              <FormControl size="small" sx={{ minWidth: 160 }}>
                <InputLabel>From control</InputLabel>
                <Select
                  value={addParamSourceFrom}
                  label="From control"
                  onChange={(e) => setAddParamSourceFrom(e.target.value)}
                >
                  <MenuItem value="">(control)</MenuItem>
                  {form.controls
                    .filter((c, i) => i !== editingControlIndex)
                    .map((c) => (
                      <MenuItem key={c.id} value={c.id}>
                        {c.label || c.endpoint_id || c.id}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>
              <Button
                size="small"
                variant="outlined"
                disabled={!addParamSourceParam.trim() || !addParamSourceFrom}
                onClick={() => {
                  setNewControl((c) => ({
                    ...c,
                    param_source: [...(c.param_source || []), { param: addParamSourceParam.trim(), from_control_id: addParamSourceFrom }],
                  }));
                  setAddParamSourceParam('');
                  setAddParamSourceFrom('');
                }}
              >
                Add
              </Button>
            </Box>
          </Box>
          {(() => {
            const selectedEndpointDef = connectorDetail?.definition?.endpoints?.[newControl.endpoint_id];
            const requiredParams = (selectedEndpointDef?.params || [])
              .filter((p) => p.required)
              .map((p) => p.name || p.id)
              .filter(Boolean);
            const alreadyBound = (newControl.param_source || []).map((s) => s.param);
            const missingParams = requiredParams.filter((p) => !alreadyBound.includes(p));
            const selectorControls = form.controls.filter(
              (c, i) => i !== editingControlIndex && (c.options_endpoint_id || (c.options && c.options.length > 0))
            );
            const suggestedControl = missingParams.length > 0 && selectorControls.length > 0 ? selectorControls[0] : null;
            const paramToSuggest = suggestedControl && missingParams[0];
            if (!paramToSuggest || !suggestedControl) return null;
            return (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  This endpoint requires <strong>{paramToSuggest}</strong>. Bind it to {suggestedControl.label || suggestedControl.endpoint_id || 'another control'}?
                </Typography>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => {
                    setNewControl((c) => ({
                      ...c,
                      param_source: [...(c.param_source || []), { param: paramToSuggest, from_control_id: suggestedControl.id }],
                    }));
                  }}
                >
                  Add binding
                </Button>
              </Alert>
            );
          })()}
          {(newControl.type === 'slider' || newControl.type === 'dropdown' || newControl.type === 'toggle' || newControl.type === 'text_display') && (
            <>
              <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                <InputLabel>Refresh endpoint (for current value)</InputLabel>
                <Select
                  value={newControl.refresh_endpoint_id || ''}
                  label="Refresh endpoint (for current value)"
                  onChange={(e) => setNewControl((c) => ({ ...c, refresh_endpoint_id: e.target.value || '' }))}
                >
                  <MenuItem value="">(None)</MenuItem>
                  {endpointIds.map((id) => (
                    <MenuItem key={id} value={id}>{id}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                fullWidth
                size="small"
                label="Value path (e.g. result.volume)"
                value={newControl.value_path || ''}
                onChange={(e) => setNewControl((c) => ({ ...c, value_path: e.target.value }))}
                sx={{ mb: 2 }}
              />
              {newControl.refresh_endpoint_id && (
                <>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                    Refresh parameter bindings (inject into refresh call)
                  </Typography>
                  <List dense sx={{ py: 0 }}>
                    {(newControl.refresh_param_source || []).map((src, idx) => (
                      <ListItem key={idx} sx={{ py: 0, minHeight: 36 }}>
                        <ListItemText
                          primary={`${src.param || '?'} ← ${(form.controls.find((c) => c.id === src.from_control_id)?.label || src.from_control_id || '?')}`}
                          primaryTypographyProps={{ variant: 'body2' }}
                        />
                        <ListItemSecondaryAction>
                          <IconButton
                            size="small"
                            aria-label="Remove refresh binding"
                            onClick={() => {
                              const next = (newControl.refresh_param_source || []).filter((_, i) => i !== idx);
                              setNewControl((c) => ({ ...c, refresh_param_source: next }));
                            }}
                          >
                            <Delete fontSize="small" />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                  <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
                    <TextField
                      size="small"
                      label="Param name"
                      placeholder="e.g. id"
                      value={addRefreshParamSourceParam}
                      onChange={(e) => setAddRefreshParamSourceParam(e.target.value)}
                      sx={{ width: 100 }}
                    />
                    <FormControl size="small" sx={{ minWidth: 160 }}>
                      <InputLabel>From control</InputLabel>
                      <Select
                        value={addRefreshParamSourceFrom}
                        label="From control"
                        onChange={(e) => setAddRefreshParamSourceFrom(e.target.value)}
                      >
                        <MenuItem value="">(control)</MenuItem>
                        {form.controls
                          .filter((c, i) => i !== editingControlIndex)
                          .map((c) => (
                            <MenuItem key={c.id} value={c.id}>
                              {c.label || c.endpoint_id || c.id}
                            </MenuItem>
                          ))}
                      </Select>
                    </FormControl>
                    <Button
                      size="small"
                      variant="outlined"
                      disabled={!addRefreshParamSourceParam.trim() || !addRefreshParamSourceFrom}
                      onClick={() => {
                        setNewControl((c) => ({
                          ...c,
                          refresh_param_source: [...(c.refresh_param_source || []), { param: addRefreshParamSourceParam.trim(), from_control_id: addRefreshParamSourceFrom }],
                        }));
                        setAddRefreshParamSourceParam('');
                        setAddRefreshParamSourceFrom('');
                      }}
                    >
                      Add
                    </Button>
                  </Box>
                </>
              )}
            </>
          )}
          {newControl.type === 'slider' && (
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <TextField
                type="number"
                size="small"
                label="Min"
                value={newControl.min ?? 0}
                onChange={(e) => setNewControl((c) => ({ ...c, min: Number(e.target.value) }))}
              />
              <TextField
                type="number"
                size="small"
                label="Max"
                value={newControl.max ?? 100}
                onChange={(e) => setNewControl((c) => ({ ...c, max: Number(e.target.value) }))}
              />
              <TextField
                type="number"
                size="small"
                label="Step"
                value={newControl.step ?? 1}
                onChange={(e) => setNewControl((c) => ({ ...c, step: Number(e.target.value) }))}
              />
            </Box>
          )}
          {newControl.type === 'button' && (
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Icon</InputLabel>
              <Select
                value={newControl.icon || 'PlayArrow'}
                label="Icon"
                onChange={(e) => setNewControl((c) => ({ ...c, icon: e.target.value || 'PlayArrow' }))}
              >
                {BUTTON_ICON_OPTIONS.map((opt) => (
                  <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {newControl.type === 'dropdown' && (
            <>
              <TextField
                fullWidth
                size="small"
                label="Static options (comma-separated)"
                placeholder="e.g. play, pause, stop"
                helperText="Used when no options endpoint is set; each value becomes label and value"
                value={Array.isArray(newControl.options) ? newControl.options.map((o) => (typeof o === 'string' ? o : o?.label ?? o?.value ?? '')).join(', ') : ''}
                onChange={(e) => {
                  const raw = e.target.value.trim();
                  const arr = raw ? raw.split(',').map((s) => s.trim()).filter(Boolean) : [];
                  const options = arr.map((s) => ({ label: s, value: s }));
                  setNewControl((c) => ({ ...c, options }));
                }}
                sx={{ mb: 1 }}
              />
              <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                <InputLabel>Options endpoint ID</InputLabel>
                <Select
                  value={newControl.options_endpoint_id || ''}
                  label="Options endpoint ID"
                  onChange={(e) => setNewControl((c) => ({ ...c, options_endpoint_id: e.target.value || '' }))}
                >
                  <MenuItem value="">(None)</MenuItem>
                  {endpointIds.map((id) => (
                    <MenuItem key={id} value={id}>{id}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                fullWidth
                size="small"
                label="Options label path"
                value={newControl.options_label_path || 'name'}
                onChange={(e) => setNewControl((c) => ({ ...c, options_label_path: e.target.value }))}
                sx={{ mb: 1 }}
              />
              <TextField
                fullWidth
                size="small"
                label="Options value path"
                value={newControl.options_value_path || 'id'}
                onChange={(e) => setNewControl((c) => ({ ...c, options_value_path: e.target.value }))}
                sx={{ mb: 2 }}
              />
            </>
          )}
          {newControl.endpoint_id && (
            <ControlTestPanel
              paneId={editingPane?.id}
              control={newControl}
              connectorId={form.connector_id}
              connectionId={form.connection_id}
              credentials={form.credentials_encrypted}
              allControls={form.controls}
              onPathSelect={handleTestPathSelect}
            />
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setAddControlOpen(false); setEditingControlIndex(null); }}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSaveControl}
            disabled={!newControl.label || !newControl.endpoint_id}
          >
            {editingControlIndex !== null ? 'Update' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
