/**
 * Full form for building a connector definition: identity, auth (with OAuth dropdown),
 * endpoints (path, method, params, response_list_path, pagination), and inline test per endpoint.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Card,
  CardContent,
  IconButton,
  Paper,
  CircularProgress,
  Alert,
  List,
  ListItem,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import { Add, Delete, Science, ExpandMore, ExpandLess } from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';
import JsonResponseViewer from './JsonResponseViewer';

const AUTH_TYPES = [
  { value: 'none', label: 'None' },
  { value: 'api_key', label: 'API Key' },
  { value: 'bearer', label: 'Bearer Token' },
  { value: 'oauth_connection', label: 'OAuth (External Connection)' },
];

const PAGINATION_TYPES = [
  { value: 'none', label: 'None' },
  { value: 'offset', label: 'Offset' },
  { value: 'page', label: 'Page' },
  { value: 'cursor', label: 'Cursor' },
];

function emptyDefinition() {
  return {
    base_url: '',
    auth: { type: 'none' },
    endpoints: {},
  };
}

function emptyEndpoint() {
  return {
    path: '/',
    method: 'GET',
    params: [],
    response_list_path: 'data',
    pagination: { type: 'none' },
    description: '',
  };
}

export default function ConnectorBuilder({ connectorId, connector, onChange, onTestResult, readOnly = false }) {
  const [expandedEndpoint, setExpandedEndpoint] = useState(null);
  const [testCreds, setTestCreds] = useState({});
  const [testConnectionId, setTestConnectionId] = useState('');
  const [testParams, setTestParams] = useState({});
  const [testParamsInput, setTestParamsInput] = useState('');
  const [testLoading, setTestLoading] = useState(false);
  const [testRaw, setTestRaw] = useState(null);
  const [testError, setTestError] = useState('');

  const def = connector?.definition || emptyDefinition();
  const endpoints = def.endpoints || {};
  const endpointIds = Object.keys(endpoints);

  const { data: connections = [] } = useQuery(
    'externalConnections',
    () => apiService.get('/api/connections').then((r) => r.connections || []),
    { retry: false }
  );
  const activeConnections = connections.filter((c) => c.is_active !== false);

  const updateDef = useCallback(
    (next) => {
      onChange?.({ ...connector, definition: next });
    },
    [connector, onChange]
  );

  const setBaseUrl = (v) => updateDef({ ...def, base_url: v || '' });
  const setAuth = (auth) => updateDef({ ...def, auth: auth || { type: 'none' } });
  const setEndpoints = (eps) => updateDef({ ...def, endpoints: eps });

  const addEndpoint = () => {
    const id = `endpoint_${Date.now()}`;
    setEndpoints({ ...endpoints, [id]: emptyEndpoint() });
    setExpandedEndpoint(id);
  };

  const removeEndpoint = (id) => {
    const next = { ...endpoints };
    delete next[id];
    setEndpoints(next);
    if (expandedEndpoint === id) setExpandedEndpoint(null);
  };

  const updateEndpoint = (id, updates) => {
    setEndpoints({ ...endpoints, [id]: { ...(endpoints[id] || emptyEndpoint()), ...updates } });
  };

  const runTest = async (endpointId) => {
    if (!connectorId || !endpointId) return;
    setTestLoading(true);
    setTestError('');
    setTestRaw(null);
    let params = {};
    try {
      params = JSON.parse(testParamsInput.trim() || '{}');
    } catch {
      params = typeof testParams === 'object' ? testParams : {};
    }
    try {
      const body = {
        endpoint_id: endpointId,
        params,
      };
      if (def.auth?.type === 'oauth_connection' && testConnectionId) {
        body.connection_id = parseInt(testConnectionId, 10);
      } else if (def.auth?.type === 'api_key' || def.auth?.type === 'bearer') {
        body.credentials = testCreds;
      }
      const res = await apiService.agentFactory.testConnector(connectorId, body);
      setTestRaw(res?.raw_response ?? null);
      onTestResult?.(res);
      if (res?.error) setTestError(res.error);
    } catch (e) {
      setTestError(e?.message || String(e));
      setTestRaw(null);
    } finally {
      setTestLoading(false);
    }
  };

  const handlePathSelect = (endpointId, path) => {
    updateEndpoint(endpointId, { response_list_path: path });
  };

  return (
    <Box>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
            Identity
          </Typography>
          <TextField
            fullWidth
            label="Name"
            value={connector?.name ?? ''}
            onChange={(e) => onChange?.({ ...connector, name: e.target.value })}
            placeholder="My API"
            sx={{ mb: 2 }}
            disabled={readOnly}
          />
          <TextField
            fullWidth
            label="Description"
            multiline
            minRows={2}
            value={connector?.description ?? ''}
            onChange={(e) => onChange?.({ ...connector, description: e.target.value })}
            placeholder="Optional description"
            disabled={readOnly}
          />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 0.5 }}>
            Connection
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Base URL and authentication for this connector.
          </Typography>
          <TextField
            fullWidth
            label="Base URL"
            value={def.base_url || ''}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.example.com"
            sx={{ mb: 2 }}
            disabled={readOnly}
          />
          <FormControl fullWidth sx={{ mb: 2 }} disabled={readOnly}>
            <InputLabel>Auth type</InputLabel>
            <Select
              value={def.auth?.type || 'none'}
              label="Auth type"
              onChange={(e) => setAuth({ ...def.auth, type: e.target.value })}
            >
              {AUTH_TYPES.map((t) => (
                <MenuItem key={t.value} value={t.value}>
                  {t.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {(def.auth?.type === 'api_key' || def.auth?.type === 'bearer') && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Authentication
              </Typography>
              <TextField
                fullWidth
                size="small"
                label="Credentials key name"
                placeholder="api_key"
                value={def.auth?.credentials_key || 'api_key'}
                onChange={(e) => setAuth({ ...def.auth, credentials_key: e.target.value })}
                sx={{ mb: 0.5 }}
                disabled={readOnly}
              />
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Name used when passing the value to the API (usually api_key).
              </Typography>
              {def.auth?.type === 'api_key' && (
                <>
                  <FormControl fullWidth size="small" sx={{ mb: 1 }} disabled={readOnly}>
                    <InputLabel>Send key in</InputLabel>
                    <Select
                      value={def.auth?.location || 'header'}
                      label="Send key in"
                      onChange={(e) => setAuth({ ...def.auth, location: e.target.value })}
                    >
                      <MenuItem value="header">Header</MenuItem>
                      <MenuItem value="query">Query parameter</MenuItem>
                    </Select>
                  </FormControl>
                  {(def.auth?.location || 'header') === 'header' && (
                    <TextField
                      fullWidth
                      size="small"
                      label="Header name (e.g. X-API-Key)"
                      value={def.auth?.header_name || 'X-API-Key'}
                      onChange={(e) => setAuth({ ...def.auth, header_name: e.target.value })}
                      disabled={readOnly}
                      sx={{ mb: 1 }}
                    />
                  )}
                  {(def.auth?.location || 'header') === 'query' && (
                    <TextField
                      fullWidth
                      size="small"
                      label="Query parameter name (e.g. apikey)"
                      value={def.auth?.param_name || 'apikey'}
                      onChange={(e) => setAuth({ ...def.auth, param_name: e.target.value })}
                      disabled={readOnly}
                      sx={{ mb: 1 }}
                    />
                  )}
                </>
              )}
              <TextField
                fullWidth
                size="small"
                label="API key or token"
                type="password"
                inputProps={{ 'aria-label': 'API key or token for testing' }}
                value={testCreds?.api_key ?? testCreds?.[def.auth?.credentials_key] ?? ''}
                onChange={(e) =>
                  setTestCreds({ [def.auth?.credentials_key || 'api_key']: e.target.value })
                }
                disabled={readOnly}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                Used when you run endpoint tests below. For playbook runs, set this when you add the connector to an agent (Data source binding).
              </Typography>
            </Box>
          )}
          {def.auth?.type === 'oauth_connection' && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Authentication
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                For endpoint tests, choose which External Connection to use. At runtime, the profile's data source binding selects the connection.
              </Typography>
              <FormControl fullWidth size="small" disabled={readOnly}>
                <InputLabel>External Connection (for testing)</InputLabel>
                <Select
                  value={testConnectionId}
                  label="External Connection (for testing)"
                  onChange={(e) => setTestConnectionId(e.target.value)}
                >
                  <MenuItem value="">—</MenuItem>
                  {activeConnections.map((c) => (
                    <MenuItem key={c.id} value={String(c.id)}>
                      {c.display_name || c.account_identifier || `${c.provider} (${c.id})`}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          )}
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="subtitle1" fontWeight={600}>
              Endpoints
            </Typography>
            <Button size="small" startIcon={<Add />} onClick={addEndpoint} disabled={readOnly}>
              Add endpoint
            </Button>
          </Box>
          {endpointIds.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              Add an endpoint to define a path, method, and response mapping.
            </Typography>
          )}
          {endpointIds.map((id) => {
            const ep = endpoints[id] || emptyEndpoint();
            const isExpanded = expandedEndpoint === id;
            return (
              <Paper key={id} variant="outlined" sx={{ p: 2, mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <IconButton size="small" onClick={() => setExpandedEndpoint(isExpanded ? null : id)} disabled={readOnly}>
                      {isExpanded ? <ExpandLess /> : <ExpandMore />}
                    </IconButton>
                    <Typography variant="subtitle2" fontFamily="monospace">
                      {id}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {ep.method || 'GET'} {ep.path || '/'}
                    </Typography>
                  </Box>
                  <IconButton size="small" color="error" onClick={() => removeEndpoint(id)} aria-label="Remove endpoint" disabled={readOnly}>
                    <Delete fontSize="small" />
                  </IconButton>
                </Box>
                {isExpanded && (
                  <Box sx={{ mt: 2, pl: 2 }}>
                    <TextField
                      fullWidth
                      size="small"
                      label="Endpoint ID (slug)"
                      value={id}
                      onChange={(e) => {
                        const next = { ...endpoints };
                        const val = e.target.value.trim() || id;
                        if (val !== id) {
                          next[val] = next[id];
                          delete next[id];
                          setEndpoints(next);
                          setExpandedEndpoint(val);
                        }
                      }}
                      sx={{ mb: 1 }}
                      disabled={readOnly}
                    />
                    <TextField
                      fullWidth
                      size="small"
                      label="Path (use {param} for placeholders)"
                      value={ep.path || '/'}
                      onChange={(e) => updateEndpoint(id, { path: e.target.value || '/' })}
                      placeholder="/v1/items"
                      sx={{ mb: 1 }}
                      disabled={readOnly}
                    />
                    <FormControl fullWidth size="small" sx={{ mb: 1 }} disabled={readOnly}>
                      <InputLabel>Method</InputLabel>
                      <Select
                        value={ep.method || 'GET'}
                        label="Method"
                        onChange={(e) => updateEndpoint(id, { method: e.target.value })}
                      >
                        <MenuItem value="GET">GET</MenuItem>
                        <MenuItem value="POST">POST</MenuItem>
                      </Select>
                    </FormControl>
                    <TextField
                      fullWidth
                      size="small"
                      label="Response list path (dot-notation, e.g. data.items)"
                      value={ep.response_list_path || ''}
                      onChange={(e) => updateEndpoint(id, { response_list_path: e.target.value })}
                      placeholder="data or . for root array"
                      sx={{ mb: 1 }}
                      disabled={readOnly}
                    />
                    <TextField
                      fullWidth
                      size="small"
                      label="Description"
                      value={ep.description || ''}
                      onChange={(e) => updateEndpoint(id, { description: e.target.value })}
                      sx={{ mb: 2 }}
                      disabled={readOnly}
                    />
                    <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                      Endpoint parameters
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Fixed value: sent with every request. Default: used when caller omits it. From caller: playbook or test must supply.
                    </Typography>
                    <List dense disablePadding sx={{ mb: 1 }}>
                      {(ep.params || []).map((p, idx) => {
                        const kind = (p.value !== undefined && p.value !== '') ? 'value' : (p.default !== undefined && p.default !== '') ? 'default' : 'caller';
                        return (
                          <ListItem key={idx} disablePadding sx={{ alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 0.5 }}>
                            <TextField
                              size="small"
                              label="Name"
                              placeholder="e.g. function, symbol"
                              value={p.name || ''}
                              onChange={(e) => {
                                const next = [...(ep.params || [])];
                                next[idx] = { ...next[idx], name: e.target.value };
                                updateEndpoint(id, { params: next });
                              }}
                              sx={{ width: 120 }}
                              disabled={readOnly}
                            />
                            <FormControl size="small" sx={{ minWidth: 140 }} disabled={readOnly}>
                              <InputLabel>Send as</InputLabel>
                              <Select
                                value={kind}
                                label="Send as"
                                onChange={(e) => {
                                  const next = [...(ep.params || [])];
                                  const v = e.target.value;
                                  if (v === 'value') next[idx] = { ...next[idx], value: next[idx].value || '', default: undefined };
                                  else if (v === 'default') next[idx] = { ...next[idx], default: next[idx].default ?? '', value: undefined };
                                  else next[idx] = { ...next[idx], value: undefined, default: undefined };
                                  updateEndpoint(id, { params: next });
                                }}
                              >
                                <MenuItem value="value">Fixed value</MenuItem>
                                <MenuItem value="default">Default</MenuItem>
                                <MenuItem value="caller">From caller</MenuItem>
                              </Select>
                            </FormControl>
                            {kind === 'value' && (
                              <TextField
                                size="small"
                                label="Value"
                                value={p.value || ''}
                                onChange={(e) => {
                                  const next = [...(ep.params || [])];
                                  next[idx] = { ...next[idx], value: e.target.value };
                                  updateEndpoint(id, { params: next });
                                }}
                                sx={{ width: 160 }}
                                disabled={readOnly}
                              />
                            )}
                            {(kind === 'default' || kind === 'caller') && (
                              <TextField
                                size="small"
                                label={kind === 'default' ? 'Default' : 'Default (optional)'}
                                placeholder={kind === 'caller' ? 'if omitted' : ''}
                                value={p.default ?? ''}
                                onChange={(e) => {
                                  const next = [...(ep.params || [])];
                                  next[idx] = { ...next[idx], default: kind === 'default' ? e.target.value : (e.target.value || undefined) };
                                  updateEndpoint(id, { params: next });
                                }}
                                sx={{ width: 130 }}
                                disabled={readOnly}
                              />
                            )}
                            {kind === 'caller' && (
                              <FormControlLabel
                                control={
                                  <Checkbox
                                    checked={!!p.required}
                                    onChange={(e) => {
                                      const next = [...(ep.params || [])];
                                      next[idx] = { ...next[idx], required: e.target.checked };
                                      updateEndpoint(id, { params: next });
                                    }}
                                    disabled={readOnly}
                                  />
                                }
                                label="Required"
                              />
                            )}
                            <IconButton size="small" color="error" onClick={() => { const next = (ep.params || []).filter((_, i) => i !== idx); updateEndpoint(id, { params: next }); }} disabled={readOnly} aria-label="Remove parameter">
                              <Delete fontSize="small" />
                            </IconButton>
                          </ListItem>
                        );
                      })}
                    </List>
                    <Button size="small" startIcon={<Add />} onClick={() => updateEndpoint(id, { params: [...(ep.params || []), { name: '', in: 'query' }] })} disabled={readOnly} sx={{ mb: 2 }}>
                      Add parameter
                    </Button>
                    <Typography variant="overline" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                      Test this endpoint
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                      Params from the endpoint definition (fixed value or default) are sent automatically. Add only what the caller supplies (e.g. symbol).
                    </Typography>
                    <TextField
                      fullWidth
                      size="small"
                      label="Params (JSON)"
                      placeholder='{"name": "pikachu"}'
                      value={testParamsInput}
                      onChange={(e) => {
                        setTestParamsInput(e.target.value);
                        try {
                          const v = e.target.value.trim();
                          setTestParams(v ? JSON.parse(v) : {});
                        } catch {
                          // Keep previous testParams when input is invalid partial JSON
                        }
                      }}
                      sx={{ mb: 1 }}
                      disabled={readOnly}
                    />
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={testLoading ? <CircularProgress size={16} /> : <Science />}
                      disabled={testLoading || readOnly}
                      onClick={() => runTest(id)}
                      sx={{ mb: 2 }}
                    >
                      Run test
                    </Button>
                    {testError && (
                      <Alert severity="error" sx={{ mb: 2 }} onClose={() => setTestError('')}>
                        {testError}
                      </Alert>
                    )}
                    {testRaw != null && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                          Raw response — click a key to set as response list path
                        </Typography>
                        <JsonResponseViewer
                          data={testRaw}
                          onPathSelect={(path) => handlePathSelect(id, path)}
                          maxHeight={300}
                        />
                      </Box>
                    )}
                  </Box>
                )}
              </Paper>
            );
          })}
        </CardContent>
      </Card>
    </Box>
  );
}
