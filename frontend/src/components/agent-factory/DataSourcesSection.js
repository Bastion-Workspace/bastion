/**
 * Data Sources section: list connectors attached to the profile, add from template, test connection.
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  CircularProgress,
  Alert,
  Switch,
  FormControlLabel,
} from '@mui/material';
import { Add, Delete, Science, Edit } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

export default function DataSourcesSection({ profileId }) {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [testOpen, setTestOpen] = useState(false);
  const [testSource, setTestSource] = useState(null);
  const [testEndpoint, setTestEndpoint] = useState('');
  const [testParams, setTestParams] = useState({});
  const [testResult, setTestResult] = useState('');
  const [testLoading, setTestLoading] = useState(false);

  const { data: dataSources = [], isLoading: sourcesLoading } = useQuery(
    ['agentFactoryDataSources', profileId],
    () => apiService.agentFactory.listDataSources(profileId),
    { enabled: !!profileId, retry: false }
  );

  const { data: templates = [], isLoading: templatesLoading } = useQuery(
    'agentFactoryConnectorTemplates',
    () => apiService.agentFactory.getConnectorTemplates(),
    { enabled: addOpen, retry: false }
  );

  const createFromTemplateMutation = useMutation(
    ({ profileId: id, template_name }) =>
      apiService.agentFactory.createDataSourceFromTemplate(id, { template_name }),
    {
      onSuccess: (_, { profileId: id }) => {
        queryClient.invalidateQueries(['agentFactoryDataSources', id]);
        queryClient.invalidateQueries(['agentFactoryActions', id]);
        setAddOpen(false);
        setSelectedTemplate('');
      },
    }
  );

  const updateSourceMutation = useMutation(
    ({ profileId: id, sourceId, body }) =>
      apiService.agentFactory.updateDataSource(id, sourceId, body),
    {
      onSuccess: (_, { profileId: id }) => {
        queryClient.invalidateQueries(['agentFactoryDataSources', id]);
      },
    }
  );

  const deleteSourceMutation = useMutation(
    ({ profileId: id, sourceId }) =>
      apiService.agentFactory.deleteDataSource(id, sourceId),
    {
      onSuccess: (_, { profileId: id }) => {
        queryClient.invalidateQueries(['agentFactoryDataSources', id]);
        queryClient.invalidateQueries(['agentFactoryActions', id]);
      },
    }
  );

  const handleAddFromTemplate = () => {
    if (!selectedTemplate || !profileId) return;
    createFromTemplateMutation.mutate({ profileId, template_name: selectedTemplate });
  };

  const handleTestConnection = async () => {
    if (!testSource || !profileId || !testEndpoint) return;
    setTestLoading(true);
    setTestResult('');
    try {
      const params = typeof testParams === 'object' && testParams !== null ? testParams : {};
      const res = await apiService.agentFactory.executeConnector({
        profile_id: profileId,
        connector_id: testSource.connector_id,
        endpoint_id: testEndpoint,
        params,
      });
      setTestResult(
        typeof res === 'string'
          ? res
          : res?.formatted ?? (res?.error ? `Error: ${res.error}` : JSON.stringify(res))
      );
    } catch (e) {
      setTestResult(`Error: ${e.message || String(e)}`);
    } finally {
      setTestLoading(false);
    }
  };

  const handleToggleEnabled = (source, enabled) => {
    if (!profileId) return;
    updateSourceMutation.mutate({
      profileId,
      sourceId: source.id,
      body: { is_enabled: enabled },
    });
  };

  const connectorEndpoints = testSource?.connector_endpoints ?? [];

  if (!profileId) return null;

  return (
    <>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
            Data Connections
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Add connectors from templates. Connector endpoints appear as playbook tools (connector:…).
          </Typography>
          {sourcesLoading ? (
            <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <>
              <List dense>
                {dataSources.map((ds) => (
                  <ListItem
                    key={ds.id}
                    secondaryAction={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              size="small"
                              checked={ds.is_enabled !== false}
                              onChange={(_, checked) => handleToggleEnabled(ds, checked)}
                            />
                          }
                          label="Enabled"
                          labelPlacement="start"
                          sx={{ mr: 1 }}
                        />
                        <IconButton
                          size="small"
                          aria-label="Test"
                          onClick={() => {
                            setTestSource(ds);
                            setTestEndpoint('');
                            setTestParams({});
                            setTestResult('');
                            setTestOpen(true);
                          }}
                        >
                          <Science fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          aria-label="Remove"
                          onClick={() =>
                            window.confirm('Remove this data source?') &&
                            deleteSourceMutation.mutate({ profileId, sourceId: ds.id })
                          }
                        >
                          <Delete fontSize="small" />
                        </IconButton>
                      </Box>
                    }
                  >
                    <ListItemText
                      primary={ds.connector_name || ds.connector_id}
                      secondary={ds.is_enabled !== false ? 'Enabled' : 'Disabled'}
                    />
                    <Button
                      size="small"
                      startIcon={<Edit fontSize="small" />}
                      onClick={() => navigate(`/agent-factory/datasource/${ds.connector_id}`)}
                      sx={{ mr: 0.5, textTransform: 'none' }}
                    >
                      Edit
                    </Button>
                  </ListItem>
                ))}
              </List>
              <Button
                startIcon={<Add />}
                variant="outlined"
                size="small"
                onClick={() => setAddOpen(true)}
              >
                Add from template
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      <Dialog open={addOpen} onClose={() => setAddOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add connector from template</DialogTitle>
        <DialogContent>
          {templatesLoading ? (
            <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          ) : (
            <FormControl fullWidth sx={{ mt: 1 }}>
              <InputLabel>Template</InputLabel>
              <Select
                value={selectedTemplate}
                label="Template"
                onChange={(e) => setSelectedTemplate(e.target.value)}
              >
                <MenuItem value="">—</MenuItem>
                {templates.map((t) => (
                  <MenuItem key={t.name} value={t.name}>
                    {t.name} {t.requires_auth ? '(requires auth)' : ''}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {createFromTemplateMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {createFromTemplateMutation.error?.message || 'Failed to add connector'}
            </Alert>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAddFromTemplate}
            disabled={!selectedTemplate || createFromTemplateMutation.isLoading}
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={testOpen} onClose={() => setTestOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Test connection</DialogTitle>
        <DialogContent>
          {testSource && (
            <>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {testSource.connector_name || testSource.connector_id}
              </Typography>
              <FormControl fullWidth size="small" sx={{ mt: 1, mb: 2 }}>
                <InputLabel>Endpoint</InputLabel>
                <Select
                  value={testEndpoint}
                  label="Endpoint"
                  onChange={(e) => setTestEndpoint(e.target.value)}
                >
                  <MenuItem value="">—</MenuItem>
                  {connectorEndpoints.map((ep) => (
                    <MenuItem key={ep} value={ep}>
                      {ep}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                fullWidth
                size="small"
                label="Params (JSON)"
                placeholder='{"key": "value"}'
                value={typeof testParams === 'object' ? JSON.stringify(testParams) : testParams}
                onChange={(e) => {
                  try {
                    const v = e.target.value.trim();
                    setTestParams(v ? JSON.parse(v) : {});
                  } catch {
                    setTestParams(e.target.value);
                  }
                }}
                sx={{ mb: 2 }}
              />
              {testResult !== '' && (
                <Alert severity={testResult.startsWith('Error') ? 'error' : 'info'} sx={{ mt: 1 }}>
                  <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: '0.8rem' }}>
                    {testResult}
                  </pre>
                </Alert>
              )}
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTestOpen(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={handleTestConnection}
            disabled={!testEndpoint || testLoading}
          >
            {testLoading ? <CircularProgress size={20} /> : 'Run'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
