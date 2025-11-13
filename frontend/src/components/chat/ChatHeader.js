import React from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Tooltip,
  Chip,
  CircularProgress,
} from '@mui/material';
import {
  Psychology,
  Clear,
  Settings,
  Menu as MenuIcon,
  SmartToy,
} from '@mui/icons-material';

const ChatHeader = ({
  sidebarCollapsed,
  onToggleSidebar,
  conversationTitle,
  enabledModels,
  currentModel,
  availableModels,
  onModelSelect,
  isSelectingModel,
  onClearChat,
  onOpenSettings,
}) => {
  return (
    <Paper elevation={1} sx={{ p: 1.5 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Box display="flex" alignItems="center" gap={1.5}>
          <IconButton 
            onClick={onToggleSidebar}
            size="small"
          >
            <MenuIcon />
          </IconButton>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Psychology color="primary" sx={{ fontSize: '1.2rem' }} />
            <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>
              {conversationTitle || "Knowledge Base Chat"}
            </Typography>
            <Chip 
              label="MCP Mode" 
              size="small" 
              color="info" 
              variant="outlined"
              sx={{ height: '24px', fontSize: '0.75rem' }}
            />
          </Box>
        </Box>

        {/* Model Selection and Actions */}
        <Box display="flex" alignItems="center" gap={1.5}>
          {/* Current Model Display & Dropdown */}
          {enabledModels?.enabled_models?.length > 0 && (
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>AI Model</InputLabel>
              <Select
                value={currentModel?.current_model || ''}
                onChange={(e) => onModelSelect(e.target.value)}
                label="AI Model"
                startAdornment={<SmartToy sx={{ mr: 1, color: 'primary.main' }} />}
                disabled={isSelectingModel}
              >
                {enabledModels.enabled_models.map((modelId) => {
                  const modelInfo = availableModels?.models?.find(m => m.id === modelId);
                  const isSelected = currentModel?.current_model === modelId;
                  return (
                    <MenuItem key={modelId} value={modelId}>
                      <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                        <Box>
                          <Typography variant="body2" sx={{ fontWeight: isSelected ? 'bold' : 'normal' }}>
                            {modelInfo?.name || modelId}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {modelInfo?.provider} • {modelInfo?.context_length?.toLocaleString()} ctx
                          </Typography>
                        </Box>
                        {isSelected && (
                          <Chip 
                            label="Active" 
                            size="small" 
                            color="success" 
                            variant="outlined"
                          />
                        )}
                      </Box>
                    </MenuItem>
                  );
                })}
              </Select>
              {isSelectingModel && (
                <Typography variant="caption" color="primary" sx={{ mt: 0.25, fontSize: '0.7rem' }}>
                  Switching model...
                </Typography>
              )}
            </FormControl>
          )}

          {/* No models enabled warning */}
          {enabledModels?.enabled_models?.length === 0 && (
            <Tooltip title="No models enabled. Go to Settings to enable models.">
              <Chip 
                label="No Models" 
                size="small" 
                color="warning" 
                variant="outlined"
                icon={<SmartToy />}
              />
            </Tooltip>
          )}

          {/* Actions */}
          <Box display="flex" gap={0.5}>
            <Tooltip title="Clear conversation">
              <IconButton onClick={onClearChat} size="small">
                <Clear />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Settings">
              <IconButton onClick={onOpenSettings} size="small">
                <Settings />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Box>
    </Paper>
  );
};

export default ChatHeader; 