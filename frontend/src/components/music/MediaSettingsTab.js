import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
  IconButton,
  Tooltip,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Save,
  Delete,
  Refresh,
  CheckCircle,
  Error as ErrorIcon,
  MusicNote,
  Add,
  Edit,
  LibraryMusic,
  Headphones,
  LiveTv,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

const MediaSettingsTab = () => {
  const queryClient = useQueryClient();
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [editingSource, setEditingSource] = useState(null);
  const [serverUrl, setServerUrl] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authType, setAuthType] = useState('password');
  const [serviceType, setServiceType] = useState('subsonic');
  const [serviceName, setServiceName] = useState('');
  const [testResult, setTestResult] = useState(null);
  const [testingServiceType, setTestingServiceType] = useState(null);

  // Fetch all configured sources
  const { data: sourcesData, isLoading: loadingSources } = useQuery(
    'mediaSources',
    () => apiService.music.getSources(),
    {
      retry: false,
      refetchOnWindowFocus: false,
    }
  );

  const sources = sourcesData?.sources || [];

  // Save config mutation
  const saveMutation = useMutation(
    (configData) => apiService.music.saveConfig(configData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('mediaSources');
        queryClient.invalidateQueries('musicConfig');
        setTestResult({ success: true, message: 'Configuration saved successfully' });
        resetForm();
        setAddDialogOpen(false);
        setEditingSource(null);
      },
      onError: (error) => {
        setTestResult({
          success: false,
          message: error.response?.data?.detail || 'Failed to save configuration',
        });
      },
    }
  );

  // Test connection mutation
  const testMutation = useMutation(
    (serviceType) => apiService.music.testConnection(serviceType),
    {
      onSuccess: (data, serviceType) => {
        // Always set result for the service type that was tested
        if (data.success) {
          setTestResult({ 
            success: true, 
            message: data.message || data.auth_method_used || 'Connection successful!',
            serviceType: serviceType 
          });
        } else {
          setTestResult({ 
            success: false, 
            message: data.error || 'Connection failed',
            serviceType: serviceType 
          });
        }
        setTestingServiceType(null);
      },
      onError: (error, serviceType) => {
        // Set error for the service type that was tested
        const errorMessage = error.response?.data?.detail || 
                            error.response?.data?.error || 
                            error.message || 
                            'Connection test failed';
        setTestResult({
          success: false,
          message: errorMessage,
          serviceType: serviceType
        });
        setTestingServiceType(null);
      },
    }
  );

  // Refresh cache mutation
  const refreshMutation = useMutation(
    (serviceType) => apiService.music.refreshCache(serviceType),
    {
      onSuccess: (data, st) => {
        queryClient.invalidateQueries('mediaSources');
        queryClient.invalidateQueries('musicConfig');
        const pl = st === 'audiobookshelf' ? 'podcasts' : 'playlists';
        setTestResult({
          success: true,
          message: `Cache refreshed: ${data.albums} albums, ${data.artists} artists, ${data.playlists} ${pl}, ${data.tracks} tracks`,
        });
      },
      onError: (error) => {
        setTestResult({
          success: false,
          message: error.response?.data?.detail || 'Failed to refresh cache',
        });
      },
    }
  );

  // Delete config mutation
  const deleteMutation = useMutation(
    (serviceType) => apiService.music.deleteConfig(serviceType),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('mediaSources');
        queryClient.invalidateQueries('musicConfig');
        setTestResult({ success: true, message: 'Configuration deleted' });
      },
      onError: (error) => {
        setTestResult({
          success: false,
          message: error.response?.data?.detail || 'Failed to delete configuration',
        });
      },
    }
  );

  const resetForm = () => {
    setServerUrl('');
    setUsername('');
    setPassword('');
    setAuthType('password');
    setServiceType('subsonic');
    setServiceName('');
    setTestResult(null);
  };

  const handleAddSource = () => {
    resetForm();
    setEditingSource(null);
    setAddDialogOpen(true);
  };

  const handleEditSource = (source) => {
    setServerUrl(source.server_url || '');
    setUsername(source.username || '');
    setPassword(''); // Don't show password
    setAuthType(source.auth_type || 'password');
    setServiceType(source.service_type || 'subsonic');
    setServiceName(source.service_name || '');
    setEditingSource(source);
    setAddDialogOpen(true);
  };

  const handleSave = () => {
    if (serviceType === 'audiobookshelf') {
      if (!password) {
        setTestResult({ success: false, message: 'Please enter API token' });
        return;
      }
      if (!serverUrl) {
        setTestResult({ success: false, message: 'Please enter server URL' });
        return;
      }
      saveMutation.mutate({
        server_url: serverUrl,
        username: 'api',
        password: password,
        auth_type: 'token',
        service_type: 'audiobookshelf',
        service_name: serviceName || undefined,
      });
    } else if (serviceType === 'emby') {
      if (!serverUrl || !username || !password) {
        setTestResult({ success: false, message: 'Please enter server URL, username, and password' });
        return;
      }
      saveMutation.mutate({
        server_url: serverUrl,
        username,
        password,
        auth_type: 'password',
        service_type: 'emby',
        service_name: serviceName || undefined,
      });
    } else {
      if (!serverUrl) {
        setTestResult({ success: false, message: 'Please enter server URL' });
        return;
      }
      if (!username || !password) {
        setTestResult({ success: false, message: 'Please fill in all fields' });
        return;
      }
      saveMutation.mutate({
        server_url: serverUrl,
        username,
        password,
        auth_type: authType,
        service_type: 'subsonic',
        service_name: serviceName || undefined,
      });
    }
  };

  const handleTest = (sourceType = null) => {
    const testType = sourceType || serviceType;
    
    // Clear previous test result when starting a new test
    setTestResult(null);
    
    // If testing from the list (sourceType provided), test existing saved config
    if (sourceType) {
      setTestingServiceType(testType);
      testMutation.mutate(testType);
      return;
    }
    
    // If testing from dialog, validate and save first, then test
    if (testType === 'audiobookshelf') {
      if (!serverUrl || !password) {
        setTestResult({ success: false, message: 'Please enter server URL and API token' });
        return;
      }
      // Save first, then test
      saveMutation.mutate(
        {
          server_url: serverUrl,
          username: 'api',
          password: password,
          auth_type: 'token',
          service_type: 'audiobookshelf',
          service_name: serviceName || undefined,
        },
        {
          onSuccess: () => {
            setTestingServiceType('audiobookshelf');
            setTestResult(null);
            testMutation.mutate('audiobookshelf');
          },
        }
      );
    } else if (testType === 'emby') {
      if (!serverUrl || !username || !password) {
        setTestResult({ success: false, message: 'Please fill in all fields before testing' });
        return;
      }
      saveMutation.mutate(
        {
          server_url: serverUrl,
          username,
          password,
          auth_type: 'password',
          service_type: 'emby',
          service_name: serviceName || undefined,
        },
        {
          onSuccess: () => {
            setTestingServiceType('emby');
            setTestResult(null);
            testMutation.mutate('emby');
          },
        }
      );
    } else {
      if (!serverUrl || !username || !password) {
        setTestResult({ success: false, message: 'Please fill in all fields before testing' });
        return;
      }
      saveMutation.mutate(
        {
          server_url: serverUrl,
          username,
          password,
          auth_type: authType,
          service_type: 'subsonic',
          service_name: serviceName || undefined,
        },
        {
          onSuccess: () => {
            setTestingServiceType('subsonic');
            setTestResult(null);
            testMutation.mutate('subsonic');
          },
        }
      );
    }
  };

  const handleRefresh = (sourceType) => {
    refreshMutation.mutate(sourceType);
  };

  const handleDelete = (sourceType) => {
    const source = sources.find(s => s.service_type === sourceType);
    const displayName = source?.service_name || sourceType;
    if (window.confirm(`Are you sure you want to delete ${displayName}? This will also clear its cache.`)) {
      deleteMutation.mutate(sourceType);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const getServiceIcon = (type) => {
    if (type === 'audiobookshelf') return <Headphones />;
    if (type === 'emby') return <LiveTv />;
    return <LibraryMusic />;
  };

  const getServiceLabel = (type) => {
    if (type === 'audiobookshelf') return 'Audiobookshelf';
    if (type === 'emby') return 'Emby';
    return 'SubSonic';
  };

  if (loadingSources) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center">
            <MusicNote sx={{ mr: 1 }} />
            <Typography variant="h6">Media Sources</Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleAddSource}
          >
            Add Source
          </Button>
        </Box>
        <Typography variant="body2" color="text.secondary" paragraph>
          Configure your media sources to stream music, audiobooks, and podcasts.
        </Typography>
      </Grid>

      {/* List of configured sources */}
      {sources.length === 0 ? (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box textAlign="center" py={4}>
                <Typography variant="body1" color="text.secondary">
                  No media sources configured. Click "Add Source" to get started.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ) : (
        sources.map((source) => (
          <Grid item xs={12} key={source.service_type}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                  <Box display="flex" alignItems="center" gap={2}>
                    {getServiceIcon(source.service_type)}
                    <Box>
                      <Typography variant="h6">
                        {source.service_name || getServiceLabel(source.service_type)}
                      </Typography>
                      <Chip
                        label={getServiceLabel(source.service_type)}
                        size="small"
                        color={
                          source.service_type === 'audiobookshelf'
                            ? 'primary'
                            : source.service_type === 'emby'
                            ? 'secondary'
                            : 'default'
                        }
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Box>
                  <Box display="flex" gap={1}>
                    <Tooltip title="Edit">
                      <IconButton size="small" onClick={() => handleEditSource(source)}>
                        <Edit />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(source.service_type)}
                        disabled={deleteMutation.isLoading}
                      >
                        <Delete />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>

                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Server: {source.server_url}
                </Typography>
                {source.service_type !== 'audiobookshelf' && (
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Username: {source.username}
                  </Typography>
                )}

                <Box display="flex" gap={2} mt={2} flexWrap="wrap">
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={testMutation.isLoading && testingServiceType === source.service_type ? <CircularProgress size={16} /> : <CheckCircle />}
                    onClick={() => handleTest(source.service_type)}
                    disabled={testMutation.isLoading && testingServiceType === source.service_type}
                  >
                    {testMutation.isLoading && testingServiceType === source.service_type ? 'Testing...' : 'Test Connection'}
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<Refresh />}
                    onClick={() => handleRefresh(source.service_type)}
                    disabled={refreshMutation.isLoading}
                  >
                    Refresh Cache
                  </Button>
                </Box>

                {testResult && testResult.serviceType === source.service_type && (
                  <Box mt={2}>
                    <Alert
                      severity={testResult.success ? 'success' : 'error'}
                      icon={testResult.success ? <CheckCircle /> : <ErrorIcon />}
                      onClose={() => {
                        setTestResult(null);
                        setTestingServiceType(null);
                      }}
                      sx={{ 
                        '& .MuiAlert-message': {
                          wordBreak: 'break-word',
                        }
                      }}
                    >
                      {testResult.message}
                    </Alert>
                  </Box>
                )}

                {source.last_sync_at && (
                  <Box mt={2}>
                    <Typography variant="body2" color="text.secondary">
                      Last sync: {formatDate(source.last_sync_at)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Albums: {source.total_albums} | Artists: {source.total_artists} | {source.service_type === 'audiobookshelf' ? 'Podcasts' : 'Playlists'}: {source.total_playlists} | Tracks: {source.total_tracks}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))
      )}

      {/* Add/Edit Source Dialog */}
      <Dialog open={addDialogOpen} onClose={() => { setAddDialogOpen(false); resetForm(); }} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingSource ? 'Edit Media Source' : 'Add Media Source'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Service Type</InputLabel>
                <Select
                  value={serviceType}
                  onChange={(e) => {
                    const v = e.target.value;
                    setServiceType(v);
                    if (v === 'audiobookshelf') {
                      setAuthType('token');
                    } else if (v === 'emby') {
                      setAuthType('password');
                    }
                  }}
                  label="Service Type"
                  disabled={!!editingSource}
                >
                  <MenuItem value="subsonic">SubSonic (Music)</MenuItem>
                  <MenuItem value="audiobookshelf">Audiobookshelf (Audiobooks & Podcasts)</MenuItem>
                  <MenuItem value="emby">Emby (Video & Music)</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Service Name (Optional)"
                value={serviceName}
                onChange={(e) => setServiceName(e.target.value)}
                placeholder={
                  serviceType === 'audiobookshelf'
                    ? 'My Audiobookshelf'
                    : serviceType === 'emby'
                    ? 'My Emby'
                    : 'My Music Server'
                }
                helperText="A friendly name to identify this source"
              />
            </Grid>

            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Server URL"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                placeholder={
                  serviceType === 'audiobookshelf'
                    ? 'https://audiobookshelf.example.com'
                    : serviceType === 'emby'
                    ? 'http://emby.local:8096'
                    : 'https://music.example.com'
                }
                helperText={`Full URL to your ${getServiceLabel(serviceType)} server`}
              />
            </Grid>

            {serviceType === 'audiobookshelf' ? (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  type="password"
                  label="API Token"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  helperText="Your Audiobookshelf API token (found in Settings → Users)"
                />
              </Grid>
            ) : serviceType === 'emby' ? (
              <>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    type="password"
                    label="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    helperText="Emby user password (AuthenticateByName)"
                  />
                </Grid>
              </>
            ) : (
              <>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Authentication Type</InputLabel>
                    <Select
                      value={authType}
                      onChange={(e) => setAuthType(e.target.value)}
                      label="Authentication Type"
                    >
                      <MenuItem value="password">Password</MenuItem>
                      <MenuItem value="token">Token</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    type="password"
                    label={authType === 'token' ? 'Token' : 'Password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    helperText={authType === 'token' ? 'SubSonic authentication token' : 'Your SubSonic password'}
                  />
                </Grid>
              </>
            )}

            {testResult && (!testResult.serviceType || testResult.serviceType === serviceType) && (
              <Grid item xs={12}>
                <Alert
                  severity={testResult.success ? 'success' : 'error'}
                  icon={testResult.success ? <CheckCircle /> : <ErrorIcon />}
                  onClose={() => setTestResult(null)}
                  sx={{ 
                    '& .MuiAlert-message': {
                      wordBreak: 'break-word',
                    }
                  }}
                >
                  {testResult.message}
                </Alert>
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setAddDialogOpen(false); resetForm(); }}>
            Cancel
          </Button>
          <Button
            variant="outlined"
            startIcon={<CheckCircle />}
            onClick={() => handleTest()}
            disabled={testMutation.isLoading || saveMutation.isLoading}
          >
            Test
          </Button>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={handleSave}
            disabled={saveMutation.isLoading}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};

export default MediaSettingsTab;
