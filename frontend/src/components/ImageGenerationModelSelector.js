import React, { useEffect, useMemo, useState } from 'react';
import { Box, Typography, FormControl, InputLabel, Select, MenuItem, Chip, Alert, CircularProgress, Grid, Button } from '@mui/material';
import { Image as ImageIcon, Refresh } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const ImageGenerationModelSelector = ({ enabledModels, modelsData, modelsLoading }) => {
  const queryClient = useQueryClient();
  const [selectedImageModel, setSelectedImageModel] = useState('');

  // Load current setting
  const { isLoading: loadingSetting } = useQuery(
    'imageGenerationModelSetting',
    async () => {
      const cat = await apiService.settings.getSettingsByCategory('llm');
      // API returns {category, settings: {image_generation_model: "value"}, count}
      // settings is a flat dict where keys map directly to values
      const value = cat?.settings?.image_generation_model;
      return value || '';
    },
    {
      onSuccess: (value) => setSelectedImageModel(value || '')
    }
  );

  const updateSettingMutation = useMutation(
    (modelId) => apiService.settings.setSettingValue('image_generation_model', {
      value: modelId,
      description: 'OpenRouter model used for image generation',
      category: 'llm'
    }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('imageGenerationModelSetting');
      }
    }
  );

  const enabledModelsArray = useMemo(
    () => modelsData?.models?.filter(m => enabledModels.has(m.id)) || [],
    [modelsData, enabledModels]
  );

  const imageLikely = (m) => {
    // Prefer official output_modalities check when available
    if (Array.isArray(m.output_modalities) && m.output_modalities.includes('image')) return true;
    const id = (m.id || '').toLowerCase();
    const name = (m.name || '').toLowerCase();
    return id.includes('image') || id.includes('vision') || name.includes('image') || name.includes('vision') || id.includes('gemini') || name.includes('gemini');
  };

  const imageCapable = enabledModelsArray.filter(imageLikely);

  const handleChange = (modelId) => {
    setSelectedImageModel(modelId);
    updateSettingMutation.mutate(modelId);
  };

  if (modelsLoading || loadingSetting) {
    return (
      <Box display="flex" alignItems="center" gap={2} p={3}>
        <CircularProgress size={24} />
        <Typography>Loading image generation models...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} md={8}>
          <FormControl fullWidth>
            <InputLabel>Image Generation Model</InputLabel>
            <Select
              value={selectedImageModel}
              onChange={(e) => handleChange(e.target.value)}
              label="Image Generation Model"
              disabled={updateSettingMutation.isLoading}
            >
              {imageCapable.length === 0 ? (
                <MenuItem disabled>No enabled image-capable models</MenuItem>
              ) : (
                imageCapable.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                      <Box display="flex" alignItems="center" gap={1}>
                        <ImageIcon fontSize="small" />
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
          This model will be used by the Image Generation Agent for all image requests.
        </Alert>
      )}

      {updateSettingMutation.isError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to update image generation model.
        </Alert>
      )}
    </Box>
  );
};

export default ImageGenerationModelSelector;


