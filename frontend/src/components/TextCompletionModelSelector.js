import React, { useMemo, useState } from 'react';
import { Box, Typography, FormControl, InputLabel, Select, MenuItem, Chip, Alert, CircularProgress, Grid, Button, Tooltip, IconButton } from '@mui/material';
import { FlashOn, Refresh, CheckCircle } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const TextCompletionModelSelector = ({ enabledModels, modelsData, modelsLoading }) => {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState('');

  const { isLoading: loadingSetting } = useQuery(
    'textCompletionModelSetting',
    async () => {
      const res = await apiService.get('/api/models/text-completion');
      return res?.text_completion_model || '';
    },
    { onSuccess: (val) => setSelectedModel(val || '') }
  );

  const updateSetting = useMutation(
    (modelId) => apiService.post('/api/models/text-completion', { model_name: modelId }),
    { onSuccess: () => queryClient.invalidateQueries('textCompletionModelSetting') }
  );

  const enabledModelsArray = useMemo(
    () => modelsData?.models?.filter(m => enabledModels.has(m.id)) || [],
    [modelsData, enabledModels]
  );

  const recommended = useMemo(() => {
    if (!enabledModelsArray.length) return [];
    const fastHints = ['haiku', 'gpt-3.5', '7b', '8b', 'flash'];
    return enabledModelsArray.filter(m => fastHints.some(h => (m.name || '').toLowerCase().includes(h)));
  }, [enabledModelsArray]);

  if (modelsLoading || loadingSetting) {
    return (
      <Box display="flex" alignItems="center" gap={2} p={3}>
        <CircularProgress size={24} />
        <Typography>Loading text-completion models...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} md={8}>
          <FormControl fullWidth>
            <InputLabel>Text Completion Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => { setSelectedModel(e.target.value); updateSetting.mutate(e.target.value); }}
              label="Text Completion Model"
              disabled={updateSetting.isLoading}
            >
              {enabledModelsArray.length === 0 ? (
                <MenuItem disabled>No enabled models available</MenuItem>
              ) : (
                enabledModelsArray.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                      <Box>
                        <Typography variant="body2">{model.name}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {model.provider} • {model.context_length?.toLocaleString()} context
                        </Typography>
                      </Box>
                      {recommended.some(rm => rm.id === model.id) && (
                        <Chip label="Recommended" size="small" color="success" variant="outlined" />
                      )}
                    </Box>
                  </MenuItem>
                ))
              )}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box display="flex" gap={1}>
            <Tooltip title="Refresh model list">
              <IconButton size="small" onClick={() => queryClient.invalidateQueries('availableModels')}>
                <Refresh />
              </IconButton>
            </Tooltip>
            <Chip icon={<FlashOn />} label="Keep this FAST" size="small" color="primary" variant="outlined" />
          </Box>
        </Grid>
      </Grid>

      {recommended.length > 0 && (
        <Alert severity="info" sx={{ mt: 2 }}>
          <Box display="flex" alignItems="center" mb={1}>
            <CheckCircle sx={{ mr: 1, fontSize: 20 }} />
            <Typography variant="subtitle2">Recommended Fast Models</Typography>
          </Box>
          <Box display="flex" flexWrap="wrap" gap={1}>
            {recommended.map((m) => (
              <Chip key={m.id} label={m.name} size="small" onClick={() => updateSetting.mutate(m.id)} clickable />
            ))}
          </Box>
        </Alert>
      )}
    </Box>
  );
};

export default TextCompletionModelSelector;


