import React, { useEffect, useMemo, useState } from 'react';
import { Box, Typography, FormControl, InputLabel, Select, MenuItem, Chip, Alert, CircularProgress, Grid, Button } from '@mui/material';
import { Image as ImageIcon, Visibility as VisibilityIcon, Refresh } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const CONFIGS = {
  generation: {
    settingKey: 'image_generation_model',
    queryKey: 'imageGenerationModelSetting',
    label: 'Image Generation Model',
    description: 'OpenRouter model used for image generation',
    Icon: ImageIcon,
    filter: (m) => {
      if (Array.isArray(m.output_modalities) && m.output_modalities.includes('image')) return true;
      const id = (m.id || '').toLowerCase();
      const name = (m.name || '').toLowerCase();
      return id.includes('image') || id.includes('vision') || name.includes('image') || name.includes('vision') || id.includes('gemini') || name.includes('gemini');
    },
    alertText: 'This model will be used by the Image Generation Agent for all image requests.',
    loadingText: 'Loading image generation models...',
    emptyText: 'No enabled image-capable models'
  },
  analysis: {
    settingKey: 'image_analysis_model',
    queryKey: 'imageAnalysisModelSetting',
    label: 'Image Analysis Model',
    description: 'Vision model for image description and analysis',
    Icon: VisibilityIcon,
    filter: (m) => {
      if (Array.isArray(m.input_modalities) && m.input_modalities.includes('image')) return true;
      const id = (m.id || '').toLowerCase();
      const name = (m.name || '').toLowerCase();
      return id.includes('vision') || name.includes('vision') || id.includes('gemini') || name.includes('gemini') || id.includes('gpt-4o') || id.includes('claude-3');
    },
    alertText: 'This model is used for describing images (metadata overlay and chat).',
    loadingText: 'Loading image analysis models...',
    emptyText: 'No enabled vision-capable models'
  }
};

const ImageGenerationModelSelector = ({ enabledModels, modelsData, modelsLoading, variant = 'generation' }) => {
  const config = CONFIGS[variant] || CONFIGS.generation;
  const { settingKey, queryKey, label, description, Icon, filter, alertText, loadingText, emptyText } = config;
  const queryClient = useQueryClient();
  const [selectedImageModel, setSelectedImageModel] = useState('');

  const { isLoading: loadingSetting } = useQuery(
    queryKey,
    async () => {
      const cat = await apiService.settings.getSettingsByCategory('llm');
      const value = cat?.settings?.[settingKey];
      return value || '';
    },
    {
      onSuccess: (value) => setSelectedImageModel(value || '')
    }
  );

  const updateSettingMutation = useMutation(
    (modelId) => apiService.settings.setSettingValue(settingKey, {
      value: modelId,
      description,
      category: 'llm'
    }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(queryKey);
      }
    }
  );

  const enabledModelsArray = useMemo(
    () => modelsData?.models?.filter(m => enabledModels.has(m.id)) || [],
    [modelsData, enabledModels]
  );

  const imageCapable = enabledModelsArray.filter(filter);

  const handleChange = (modelId) => {
    setSelectedImageModel(modelId);
    updateSettingMutation.mutate(modelId);
  };

  if (modelsLoading || loadingSetting) {
    return (
      <Box display="flex" alignItems="center" gap={2} p={3}>
        <CircularProgress size={24} />
        <Typography>{loadingText}</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} md={8}>
          <FormControl fullWidth>
            <InputLabel>{label}</InputLabel>
            <Select
              value={selectedImageModel}
              onChange={(e) => handleChange(e.target.value)}
              label={label}
              disabled={updateSettingMutation.isLoading}
            >
              {imageCapable.length === 0 ? (
                <MenuItem disabled>{emptyText}</MenuItem>
              ) : (
                imageCapable.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                      <Box display="flex" alignItems="center" gap={1}>
                        <Icon fontSize="small" />
                        <Typography variant="body2">{model.name}</Typography>
                      </Box>
                      <Chip label={model.provider} size="small" variant="outlined" />
                    </Box>
                  </MenuItem>
                ))
              )}
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} md={4}>
          <Box display="flex" gap={1}>
            <Button size="small" variant="outlined" onClick={() => queryClient.invalidateQueries('availableModels')} startIcon={<Refresh />}>Refresh</Button>
          </Box>
        </Grid>
      </Grid>

      {selectedImageModel && (
        <Alert severity="info" sx={{ mt: 2 }}>
          {alertText}
        </Alert>
      )}

      {updateSettingMutation.isError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to update {label.toLowerCase()}.
        </Alert>
      )}
    </Box>
  );
};

export default ImageGenerationModelSelector;


