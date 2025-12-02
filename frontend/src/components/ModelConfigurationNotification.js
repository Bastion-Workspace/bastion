import React, { useState, useEffect } from 'react';
import {
  Alert,
  Snackbar,
  Button,
  Box,
  Typography,
  Chip
} from '@mui/material';
import { Settings, Psychology } from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/apiService';

/**
 * Component that shows a notification when AI models are not explicitly configured
 * and prompts users to configure them for better performance and consistency.
 */
const ModelConfigurationNotification = () => {
  const [open, setOpen] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const navigate = useNavigate();

  // Check if models are configured
  const { data: modelData, isLoading } = useQuery(
    'modelConfigurationCheck',
    () => apiService.get('/api/models/classification'),
    {
      enabled: !dismissed, // Don't check if user dismissed
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // Check every 5 minutes
      onSuccess: (data) => {
        // Show notification if either model is using fallback
        const needsConfiguration = data?.chat_model_is_fallback || data?.classification_model_is_fallback;
        const hasBeenDismissed = localStorage.getItem('modelConfigNotificationDismissed');

        if (needsConfiguration && !hasBeenDismissed) {
          // Delay showing notification to avoid overwhelming user on first load
          setTimeout(() => setOpen(true), 3000);
        }
      }
    }
  );

  const handleDismiss = () => {
    setOpen(false);
    setDismissed(true);
    localStorage.setItem('modelConfigNotificationDismissed', 'true');
  };

  const handleConfigure = () => {
    setOpen(false);
    navigate('/settings');
    // Scroll to AI Models tab after navigation
    setTimeout(() => {
      const aiModelsTab = document.querySelector('[data-tab="ai-models"]');
      if (aiModelsTab) {
        aiModelsTab.click();
      }
    }, 500);
  };

  // Reset dismissed state if models become unconfigured again
  useEffect(() => {
    if (modelData && (modelData.chat_model_is_fallback || modelData.classification_model_is_fallback)) {
      const hasBeenDismissed = localStorage.getItem('modelConfigNotificationDismissed');
      if (!hasBeenDismissed) {
        setDismissed(false);
      }
    }
  }, [modelData]);

  if (isLoading || dismissed) {
    return null;
  }

  return (
    <Snackbar
      open={open}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      sx={{ maxWidth: 600 }}
    >
      <Alert
        severity="warning"
        variant="filled"
        sx={{ width: '100%' }}
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              color="inherit"
              size="small"
              onClick={handleConfigure}
              startIcon={<Settings />}
            >
              Configure
            </Button>
            <Button
              color="inherit"
              size="small"
              onClick={handleDismiss}
            >
              Dismiss
            </Button>
          </Box>
        }
      >
        <Box sx={{ mb: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center' }}>
            <Psychology sx={{ mr: 1, fontSize: 20 }} />
            AI Models Using Defaults
          </Typography>
        </Box>

        <Typography variant="body2" sx={{ mb: 1 }}>
          Your AI agents are currently using system fallback models instead of explicitly configured ones.
          This may result in inconsistent behavior or suboptimal performance.
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          {modelData?.chat_model_is_fallback && (
            <Chip
              label={`Chat: ${modelData?.effective_chat_model}`}
              size="small"
              color="warning"
              variant="outlined"
            />
          )}
          {modelData?.classification_model_is_fallback && (
            <Chip
              label={`Classification: ${modelData?.effective_classification_model}`}
              size="small"
              color="warning"
              variant="outlined"
            />
          )}
        </Box>

        <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
          Click "Configure" to set specific models in Settings → AI Models
        </Typography>
      </Alert>
    </Snackbar>
  );
};

export default ModelConfigurationNotification;
