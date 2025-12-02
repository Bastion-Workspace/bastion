import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  Button,
  CircularProgress,
  Paper,
  Grid,
  Tooltip,
  IconButton
} from '@mui/material';
import { Speed, Refresh, CheckCircle, Warning } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

/**
 * Component for selecting the classification model used for intent classification
 * This model should be fast and lightweight for optimal performance
 */
const ClassificationModelSelector = ({ enabledModels, modelsData, modelsLoading }) => {
  const queryClient = useQueryClient();
  const [selectedClassificationModel, setSelectedClassificationModel] = useState('');
  const [testingModel, setTestingModel] = useState(false);

  // Fetch current classification model setting
  const { data: currentClassificationModel, isLoading: loadingCurrentModel } = useQuery(
    'classificationModel',
    () => apiService.get('/api/models/classification'),
    {
      onSuccess: (data) => {
        // Always set the effective model (what agents actually use)
        if (data?.effective_classification_model) {
          setSelectedClassificationModel(data.effective_classification_model);
        }
      },
      onError: () => {
        // Setting doesn't exist yet, that's okay
      }
    }
  );

  // Update classification model mutation
  const updateClassificationModelMutation = useMutation(
    (modelId) => apiService.post('/api/models/classification', {
      model_name: modelId
    }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('classificationModel');
      }
    }
  );

  // Test classification model mutation
  const testClassificationMutation = useMutation(
    (modelId) => apiService.post('/api/classification/test-model', {
      model_id: modelId,
      test_query: "What is artificial intelligence?"
    }),
    {
      onMutate: () => {
        setTestingModel(true);
      },
      onSettled: () => {
        setTestingModel(false);
      }
    }
  );

  // Get recommended models for classification (fast, lightweight models)
  const getRecommendedModels = () => {
    if (!modelsData?.models) return [];
    
    const fastModels = [
      'anthropic/claude-3-haiku',
      'openai/gpt-3.5-turbo',
      'meta-llama/llama-3-8b-instruct',
      'mistralai/mistral-7b-instruct',
      'google/gemini-pro'
    ];
    
    return modelsData.models.filter(model => 
      enabledModels.has(model.id) && 
      (fastModels.includes(model.id) || 
       model.name.toLowerCase().includes('haiku') ||
       model.name.toLowerCase().includes('3.5') ||
       model.name.toLowerCase().includes('7b') ||
       model.name.toLowerCase().includes('8b'))
    );
  };

  const handleModelChange = (modelId) => {
    setSelectedClassificationModel(modelId);
    updateClassificationModelMutation.mutate(modelId);
  };

  const handleTestModel = () => {
    if (selectedClassificationModel) {
      testClassificationMutation.mutate(selectedClassificationModel);
    }
  };

  const recommendedModels = getRecommendedModels();
  const enabledModelsArray = modelsData?.models?.filter(m => enabledModels.has(m.id)) || [];

  if (modelsLoading || loadingCurrentModel) {
    return (
      <Box display="flex" alignItems="center" gap={2} p={3}>
        <CircularProgress size={24} />
        <Typography>Loading classification models...</Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Model Selection */}
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={12} md={8}>
          <FormControl fullWidth>
            <InputLabel>Classification Model</InputLabel>
            <Select
              value={selectedClassificationModel}
              onChange={(e) => handleModelChange(e.target.value)}
              label="Classification Model"
              disabled={updateClassificationModelMutation.isLoading}
            >
              {enabledModelsArray.length === 0 ? (
                <MenuItem disabled>
                  No enabled models available
                </MenuItem>
              ) : (
                enabledModelsArray.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                      <Box>
                        <Typography variant="body2">
                          {model.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {model.provider} • {model.context_length?.toLocaleString()} context
                        </Typography>
                      </Box>
                      {recommendedModels.some(rm => rm.id === model.id) && (
                        <Chip 
                          label="Recommended" 
                          size="small" 
                          color="success" 
                          variant="outlined"
                        />
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
            <Button
              variant="outlined"
              onClick={handleTestModel}
              disabled={!selectedClassificationModel || testingModel}
              startIcon={testingModel ? <CircularProgress size={16} /> : <Speed />}
              size="small"
            >
              {testingModel ? 'Testing...' : 'Test Speed'}
            </Button>
            
            <Tooltip title="Refresh model list">
              <IconButton size="small" onClick={() => queryClient.invalidateQueries('availableModels')}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Grid>
      </Grid>

      {/* Recommendations */}
      {recommendedModels.length > 0 && (
        <Paper sx={{ p: 2, mt: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
          <Box display="flex" alignItems="center" mb={1}>
            <CheckCircle sx={{ mr: 1, fontSize: 20 }} />
            <Typography variant="subtitle2">
              Recommended Fast Models
            </Typography>
          </Box>
          <Box display="flex" flexWrap="wrap" gap={1}>
            {recommendedModels.map((model) => (
              <Chip
                key={model.id}
                label={model.name}
                size="small"
                variant={selectedClassificationModel === model.id ? "filled" : "outlined"}
                color="success"
                onClick={() => handleModelChange(model.id)}
                clickable
              />
            ))}
          </Box>
          <Typography variant="caption" display="block" sx={{ mt: 1 }}>
            These models are optimized for speed and work well for intent classification.
          </Typography>
        </Paper>
      )}

      {/* Current Selection Info */}
      {selectedClassificationModel && (
        <Paper sx={{
          p: 2,
          mt: 2,
          bgcolor: currentClassificationModel?.classification_model_is_fallback ? 'warning.light' : 'primary.light',
          color: currentClassificationModel?.classification_model_is_fallback ? 'warning.contrastText' : 'primary.contrastText'
        }}>
          <Box display="flex" alignItems="center" mb={1}>
            <Speed sx={{ mr: 1, fontSize: 20 }} />
            <Typography variant="subtitle2">
              Current Classification Model
              {currentClassificationModel?.classification_model_is_fallback && (
                <Chip
                  label="Using Fallback"
                  size="small"
                  color="warning"
                  sx={{ ml: 1, fontSize: '0.7rem' }}
                />
              )}
            </Typography>
          </Box>
          <Typography variant="body2">
            {modelsData?.models?.find(m => m.id === selectedClassificationModel)?.name || selectedClassificationModel}
          </Typography>
          <Typography variant="caption">
            This model will be used for all intent classification requests.
            {currentClassificationModel?.classification_model_is_fallback && (
              <Box component="span" sx={{ display: 'block', mt: 1, fontWeight: 'bold' }}>
                ⚠️ No model explicitly set - using system fallback. Consider selecting a specific model above.
              </Box>
            )}
          </Typography>
        </Paper>
      )}

      {/* Test Results */}
      {testClassificationMutation.isSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          <strong>Test Successful!</strong> 
          Classification completed in {testClassificationMutation.data?.response_time?.toFixed(2)}s
          <br />
          <Typography variant="caption">
            Detected mode: {testClassificationMutation.data?.execution_mode} 
            (confidence: {(testClassificationMutation.data?.confidence * 100)?.toFixed(1)}%)
          </Typography>
        </Alert>
      )}

      {testClassificationMutation.isError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          <strong>Test Failed:</strong> {testClassificationMutation.error?.response?.data?.detail || testClassificationMutation.error?.message}
        </Alert>
      )}

      {/* Status Messages */}
      {updateClassificationModelMutation.isSuccess && (
        <Alert severity="success" sx={{ mt: 2 }}>
          Classification model updated successfully!
        </Alert>
      )}

      {updateClassificationModelMutation.isError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          Failed to update classification model: {updateClassificationModelMutation.error?.response?.data?.detail}
        </Alert>
      )}

      {/* No Models Warning */}
      {enabledModelsArray.length === 0 && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          <Box display="flex" alignItems="center">
            <Warning sx={{ mr: 1 }} />
            <Box>
              <Typography variant="body2">
                <strong>No Enabled Models:</strong> You need to enable at least one model in the Model Management section below.
              </Typography>
              <Typography variant="caption">
                Intent classification requires an enabled model to function properly.
              </Typography>
            </Box>
          </Box>
        </Alert>
      )}

      {/* Performance Tips */}
      <Alert severity="info" sx={{ mt: 2 }}>
        <Typography variant="body2" gutterBottom>
          <strong>Performance Tips:</strong>
        </Typography>
        <Typography variant="caption" component="div">
          • Choose lightweight models (Haiku, GPT-3.5, 7B/8B models) for fastest classification<br/>
          • Avoid large models (GPT-4, Claude Opus) as they add unnecessary latency<br/>
          • Test your selected model to ensure sub-second response times<br/>
          • The classification model only determines execution mode, not the final response quality
        </Typography>
      </Alert>
    </Box>
  );
};

export default ClassificationModelSelector;
