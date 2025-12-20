/**
 * Entertainment Sync Manager
 * Component for managing Sonarr/Radarr sync configurations
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  FormControlLabel,
  Checkbox,
  Alert,
  Chip,
  Grid,
  InputLabel,
  CircularProgress
} from '@mui/material';
import entertainmentSyncService from '../services/entertainmentSyncService';

const EntertainmentSyncManager = () => {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [testingConnection, setTestingConnection] = useState(null);
  const [syncing, setSyncing] = useState(null);
  
  const [formData, setFormData] = useState({
    source_type: 'radarr',
    api_url: '',
    api_key: '',
    enabled: true,
    sync_frequency_minutes: 60
  });

  useEffect(() => {
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await entertainmentSyncService.getConfigs();
      setConfigs(data);
    } catch (err) {
      setError(err.message || 'Failed to load configurations');
      console.error('Failed to load sync configs:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    
    try {
      setLoading(true);
      await entertainmentSyncService.createConfig(formData);
      await loadConfigs();
      setShowAddForm(false);
      setFormData({
        source_type: 'radarr',
        api_url: '',
        api_key: '',
        enabled: true,
        sync_frequency_minutes: 60
      });
    } catch (err) {
      setError(err.message || 'Failed to create configuration');
      console.error('Failed to create config:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleTestConnection = async (configId) => {
    try {
      setTestingConnection(configId);
      setError(null);
      const result = await entertainmentSyncService.testConnection(configId);
      
      if (result.success) {
        alert(`Connection successful! ${result.message}`);
      } else {
        alert(`Connection failed: ${result.message}`);
      }
    } catch (err) {
      alert(`Connection test failed: ${err.message}`);
    } finally {
      setTestingConnection(null);
    }
  };

  const handleTriggerSync = async (configId) => {
    try {
      setSyncing(configId);
      setError(null);
      const result = await entertainmentSyncService.triggerSync(configId);
      
      if (result.success) {
        alert(`Sync triggered! Task ID: ${result.task_id}`);
        await loadConfigs(); // Refresh to show updated status
      } else {
        alert(`Failed to trigger sync: ${result.message}`);
      }
    } catch (err) {
      alert(`Failed to trigger sync: ${err.message}`);
    } finally {
      setSyncing(null);
    }
  };

  const handleDelete = async (configId) => {
    if (!window.confirm('Are you sure you want to delete this configuration?')) {
      return;
    }
    
    try {
      setError(null);
      await entertainmentSyncService.deleteConfig(configId);
      await loadConfigs();
    } catch (err) {
      setError(err.message || 'Failed to delete configuration');
      console.error('Failed to delete config:', err);
    }
  };

  const handleToggleEnabled = async (config) => {
    try {
      setError(null);
      await entertainmentSyncService.updateConfig(config.config_id, {
        enabled: !config.enabled
      });
      await loadConfigs();
    } catch (err) {
      setError(err.message || 'Failed to update configuration');
      console.error('Failed to update config:', err);
    }
  };

  const getStatusBadge = (status) => {
    const statusMap = {
      'success': { color: 'success', label: 'Success' },
      'failed': { color: 'error', label: 'Failed' },
      'running': { color: 'primary', label: 'Running' }
    };
    
    const statusInfo = statusMap[status] || { color: 'default', label: 'Unknown' };
    return (
      <Chip
        label={statusInfo.label}
        color={statusInfo.color}
        size="small"
      />
    );
  };

  const formatLastSync = (lastSyncAt) => {
    if (!lastSyncAt) return 'Never';
    const date = new Date(lastSyncAt);
    return date.toLocaleString();
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Entertainment Sync Configuration
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Configure Radarr and Sonarr API connections to automatically sync your media library.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Box sx={{ mb: 3 }}>
        <Button
          variant="contained"
          onClick={() => setShowAddForm(!showAddForm)}
        >
          {showAddForm ? 'Cancel' : '+ Add Configuration'}
        </Button>
      </Box>
      
      {showAddForm && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Add New Configuration
            </Typography>
            
            <Box component="form" onSubmit={handleSubmit}>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Source Type</InputLabel>
                    <Select
                      name="source_type"
                      value={formData.source_type}
                      onChange={handleInputChange}
                      label="Source Type"
                    >
                      <MenuItem value="radarr">Radarr (Movies)</MenuItem>
                      <MenuItem value="sonarr">Sonarr (TV Shows)</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="API URL"
                    name="api_url"
                    value={formData.api_url}
                    onChange={handleInputChange}
                    placeholder="http://localhost:7878"
                    required
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="API Key"
                    name="api_key"
                    type="password"
                    value={formData.api_key}
                    onChange={handleInputChange}
                    placeholder="Your API key"
                    required
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Sync Frequency (minutes)"
                    name="sync_frequency_minutes"
                    type="number"
                    value={formData.sync_frequency_minutes}
                    onChange={handleInputChange}
                    inputProps={{ min: 15 }}
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        name="enabled"
                        checked={formData.enabled}
                        onChange={handleInputChange}
                      />
                    }
                    label="Enabled"
                  />
                </Grid>
                
                <Grid item xs={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    color="success"
                    disabled={loading}
                  >
                    {loading ? 'Creating...' : 'Create Configuration'}
                  </Button>
                </Grid>
              </Grid>
            </Box>
          </CardContent>
        </Card>
      )}
      
      {loading && !showAddForm && (
        <Box display="flex" justifyContent="center" alignItems="center" p={3}>
          <CircularProgress />
        </Box>
      )}
      
      {!loading && configs.length === 0 && !showAddForm && (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="text.secondary">
            No configurations found. Add one to get started.
          </Typography>
        </Box>
      )}
      
      {configs.length > 0 && (
        <Grid container spacing={2}>
          {configs.map(config => (
            <Grid item xs={12} key={config.config_id}>
              <Card 
                sx={{ 
                  opacity: config.enabled ? 1 : 0.7,
                  backgroundColor: config.enabled ? 'background.paper' : 'action.disabledBackground'
                }}
              >
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                    <Box>
                      <Typography variant="h6">
                        {config.source_type === 'radarr' ? '🎬 Radarr' : '📺 Sonarr'}
                        {!config.enabled && ' (Disabled)'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {config.api_url}
                      </Typography>
                    </Box>
                    <Box display="flex" gap={1}>
                      <Button
                        size="small"
                        variant="outlined"
                        color={config.enabled ? 'error' : 'success'}
                        onClick={() => handleToggleEnabled(config)}
                      >
                        {config.enabled ? 'Disable' : 'Enable'}
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        color="error"
                        onClick={() => handleDelete(config.config_id)}
                      >
                        Delete
                      </Button>
                    </Box>
                  </Box>
                  
                  <Box mb={2}>
                    <Typography variant="body2">
                      Last Sync: {formatLastSync(config.last_sync_at)}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1} mt={0.5}>
                      <Typography variant="body2">Status:</Typography>
                      {getStatusBadge(config.last_sync_status)}
                    </Box>
                    <Typography variant="body2">
                      Items Synced: {config.items_synced}
                    </Typography>
                    {config.sync_error && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {config.sync_error}
                      </Alert>
                    )}
                  </Box>
                  
                  <Box display="flex" gap={1}>
                    <Button
                      variant="outlined"
                      color="info"
                      onClick={() => handleTestConnection(config.config_id)}
                      disabled={testingConnection === config.config_id}
                    >
                      {testingConnection === config.config_id ? 'Testing...' : 'Test Connection'}
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => handleTriggerSync(config.config_id)}
                      disabled={syncing === config.config_id || !config.enabled}
                    >
                      {syncing === config.config_id ? 'Syncing...' : 'Trigger Sync'}
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Box>
  );
};

export default EntertainmentSyncManager;

