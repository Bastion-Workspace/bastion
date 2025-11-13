/**
 * Cyber Catalog Settings Tab
 * 
 * **BULLY!** Configure your cyber data cataloging!
 * 
 * Provides UI for:
 * - Enable/disable catalog service
 * - LLM categorization toggle
 * - Folder path selection
 * - Configuration options
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Paper,
  Stack,
  Divider,
  Chip
} from '@mui/material';
import {
  FolderOpen,
  Save,
  PlayArrow,
  Stop,
  Refresh
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import apiService from '../services/apiService';

const CyberCatalogSettingsTab = () => {
  // State
  const [config, setConfig] = useState({
    enabled: false,
    enable_llm_categorization: true,
    watch_folders: [],
    max_file_size_mb: 100,
    max_llm_file_size_mb: 100,
    llm_sample_size_chars: 5000,
    llm_sampling_strategy: "smart",
    llm_sampling_threshold_mb: 1.0,
    batch_size: 10,
    enable_entity_extraction: true,
    auto_tag_by_filename: true,
    exclude_patterns: [".git", "__pycache__", ".DS_Store"]
  });
  
  const [newFolderPath, setNewFolderPath] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [cataloging, setCataloging] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [lastCatalogResult, setLastCatalogResult] = useState(null);

  // Load config on mount
  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.get('/api/cyber-catalog/config');
      
      if (response) {
        // Merge with defaults
        setConfig(prev => ({
          ...prev,
          ...response,
          watch_folders: response.watch_folders || []
        }));
      }
    } catch (err) {
      console.error('Failed to load cyber catalog config:', err);
      setError(err.message || 'Failed to load configuration');
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    try {
      setSaving(true);
      setError(null);
      setSuccess(null);
      
      const response = await apiService.post('/api/cyber-catalog/config', config);
      
      if (response.success) {
        setSuccess('Configuration saved successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(response.message || 'Failed to save configuration');
      }
    } catch (err) {
      console.error('Failed to save cyber catalog config:', err);
      setError(err.message || 'Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const startCataloging = async () => {
    if (!config.watch_folders || config.watch_folders.length === 0) {
      setError('Please add at least one folder to scan');
      return;
    }

    try {
      setCataloging(true);
      setError(null);
      setSuccess(null);
      setLastCatalogResult(null);

      // Catalog first folder (can be extended to catalog all)
      const folderPath = config.watch_folders[0];
      
      const response = await apiService.post('/api/cyber-catalog/catalog', {
        folder_path: folderPath,
        config: {
          validate_only: true,
          enable_llm_categorization: config.enable_llm_categorization,
          max_file_size_mb: config.max_file_size_mb,
          max_llm_file_size_mb: config.max_llm_file_size_mb,
          llm_sample_size_chars: config.llm_sample_size_chars,
          llm_sampling_strategy: config.llm_sampling_strategy,
          llm_sampling_threshold_mb: config.llm_sampling_threshold_mb,
          batch_size: config.batch_size,
          enable_entity_extraction: config.enable_entity_extraction,
          auto_tag_by_filename: config.auto_tag_by_filename,
          exclude_patterns: config.exclude_patterns
        }
      });

      if (response.success) {
        setLastCatalogResult(response);
        setSuccess(`Cataloging complete! ${response.entry_count} files cataloged.`);
        setTimeout(() => setSuccess(null), 5000);
      } else {
        setError(response.error || 'Cataloging failed');
      }
    } catch (err) {
      console.error('Failed to start cataloging:', err);
      setError(err.message || 'Failed to start cataloging');
    } finally {
      setCataloging(false);
    }
  };

  const addFolder = () => {
    if (newFolderPath.trim() && !config.watch_folders.includes(newFolderPath.trim())) {
      setConfig(prev => ({
        ...prev,
        watch_folders: [...prev.watch_folders, newFolderPath.trim()]
      }));
      setNewFolderPath('');
    }
  };

  const removeFolder = (folderPath) => {
    setConfig(prev => ({
      ...prev,
      watch_folders: prev.watch_folders.filter(f => f !== folderPath)
    }));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Alert severity="info" sx={{ mb: 3 }}>
        <strong>Cyber Data Catalog:</strong> Configure automated cataloging of cyber/breach data files.
        Files are cataloged to JSON for validation before database integration.
      </Alert>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Service Activation */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Box display="flex" alignItems="center">
                    <FolderOpen sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">Service Activation</Typography>
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.enabled}
                        onChange={(e) => setConfig(prev => ({ ...prev, enabled: e.target.checked }))}
                        color="primary"
                      />
                    }
                    label={config.enabled ? "Enabled" : "Disabled"}
                  />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  Enable or disable the cyber catalog service. When enabled, folders can be cataloged.
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Folder Configuration */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Watch Folders
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Folders to scan for cyber/breach data files. Files are scanned recursively.
                </Typography>

                <Stack spacing={2} sx={{ mb: 2 }}>
                  {config.watch_folders.map((folder, index) => (
                    <Paper key={index} variant="outlined" sx={{ p: 2 }}>
                      <Box display="flex" alignItems="center" justifyContent="space-between">
                        <Typography variant="body1" sx={{ fontFamily: 'monospace' }}>
                          {folder}
                        </Typography>
                        <Button
                          size="small"
                          color="error"
                          onClick={() => removeFolder(folder)}
                        >
                          Remove
                        </Button>
                      </Box>
                    </Paper>
                  ))}
                </Stack>

                <Box display="flex" gap={2}>
                  <TextField
                    fullWidth
                    label="Folder Path"
                    value={newFolderPath}
                    onChange={(e) => setNewFolderPath(e.target.value)}
                    placeholder="/path/to/breach/data"
                    disabled={!config.enabled}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        addFolder();
                      }
                    }}
                  />
                  <Button
                    variant="outlined"
                    onClick={addFolder}
                    disabled={!config.enabled || !newFolderPath.trim()}
                    startIcon={<FolderOpen />}
                  >
                    Add Folder
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* LLM Configuration */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  LLM Categorization
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Use LLM to automatically categorize and tag files.
                </Typography>

                <Stack spacing={2}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.enable_llm_categorization}
                        onChange={(e) => setConfig(prev => ({ ...prev, enable_llm_categorization: e.target.checked }))}
                        disabled={!config.enabled}
                      />
                    }
                    label="Enable LLM Categorization"
                  />

                  <TextField
                    label="Max LLM File Size (MB)"
                    type="number"
                    value={config.max_llm_file_size_mb}
                    onChange={(e) => setConfig(prev => ({ ...prev, max_llm_file_size_mb: parseInt(e.target.value) || 100 }))}
                    disabled={!config.enabled || !config.enable_llm_categorization}
                    helperText="Files larger than this skip LLM processing"
                    fullWidth
                  />

                  <TextField
                    label="LLM Sample Size (chars)"
                    type="number"
                    value={config.llm_sample_size_chars}
                    onChange={(e) => setConfig(prev => ({ ...prev, llm_sample_size_chars: parseInt(e.target.value) || 5000 }))}
                    disabled={!config.enabled || !config.enable_llm_categorization}
                    helperText="Character limit sent to LLM (default: 5000)"
                    fullWidth
                  />

                  <TextField
                    label="Sampling Threshold (MB)"
                    type="number"
                    value={config.llm_sampling_threshold_mb}
                    onChange={(e) => setConfig(prev => ({ ...prev, llm_sampling_threshold_mb: parseFloat(e.target.value) || 1.0 }))}
                    disabled={!config.enabled || !config.enable_llm_categorization}
                    helperText="Files larger than this use smart sampling"
                    fullWidth
                    inputProps={{ step: 0.1 }}
                  />
                </Stack>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Processing Configuration */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Processing Options
                </Typography>

                <Stack spacing={2}>
                  <TextField
                    label="Max File Size (MB)"
                    type="number"
                    value={config.max_file_size_mb}
                    onChange={(e) => setConfig(prev => ({ ...prev, max_file_size_mb: parseInt(e.target.value) || 100 }))}
                    disabled={!config.enabled}
                    helperText="Skip files larger than this"
                    fullWidth
                  />

                  <TextField
                    label="Batch Size"
                    type="number"
                    value={config.batch_size}
                    onChange={(e) => setConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) || 10 }))}
                    disabled={!config.enabled}
                    helperText="Files to process per batch"
                    fullWidth
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.enable_entity_extraction}
                        onChange={(e) => setConfig(prev => ({ ...prev, enable_entity_extraction: e.target.checked }))}
                        disabled={!config.enabled}
                      />
                    }
                    label="Enable Entity Extraction"
                  />

                  <FormControlLabel
                    control={
                      <Switch
                        checked={config.auto_tag_by_filename}
                        onChange={(e) => setConfig(prev => ({ ...prev, auto_tag_by_filename: e.target.checked }))}
                        disabled={!config.enabled}
                      />
                    }
                    label="Auto Tag by Filename"
                  />
                </Stack>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Actions */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" gap={2} flexWrap="wrap">
                  <Button
                    variant="contained"
                    onClick={saveConfig}
                    disabled={!config.enabled || saving}
                    startIcon={saving ? <CircularProgress size={20} /> : <Save />}
                  >
                    Save Configuration
                  </Button>

                  <Button
                    variant="contained"
                    color="success"
                    onClick={startCataloging}
                    disabled={!config.enabled || cataloging || config.watch_folders.length === 0}
                    startIcon={cataloging ? <CircularProgress size={20} /> : <PlayArrow />}
                  >
                    {cataloging ? 'Cataloging...' : 'Start Cataloging'}
                  </Button>

                  <Button
                    variant="outlined"
                    onClick={loadConfig}
                    disabled={loading}
                    startIcon={<Refresh />}
                  >
                    Reload Config
                  </Button>
                </Box>

                {lastCatalogResult && (
                  <Box mt={3}>
                    <Divider sx={{ mb: 2 }} />
                    <Typography variant="subtitle2" gutterBottom>
                      Last Catalog Result:
                    </Typography>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Stack spacing={1}>
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="body2">Files Cataloged:</Typography>
                          <Chip label={lastCatalogResult.entry_count} size="small" color="primary" />
                        </Box>
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="body2">JSON Path:</Typography>
                          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                            {lastCatalogResult.json_path}
                          </Typography>
                        </Box>
                        <Box display="flex" justifyContent="space-between">
                          <Typography variant="body2">Processing Time:</Typography>
                          <Typography variant="body2">
                            {lastCatalogResult.processing_time_seconds?.toFixed(2)}s
                          </Typography>
                        </Box>
                        {lastCatalogResult.validation && (
                          <Box>
                            <Typography variant="body2" gutterBottom>
                              Validation: {lastCatalogResult.validation.is_valid ? '✅ Valid' : '❌ Invalid'}
                            </Typography>
                            {lastCatalogResult.validation.warnings?.length > 0 && (
                              <Alert severity="warning" sx={{ mt: 1 }}>
                                {lastCatalogResult.validation.warnings.join(', ')}
                              </Alert>
                            )}
                          </Box>
                        )}
                      </Stack>
                    </Paper>
                  </Box>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  );
};

export default CyberCatalogSettingsTab;



