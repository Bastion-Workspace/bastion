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
  // Chat dropdown excludes image generation model (used only for image creation)
  const chatModels = (enabledModels?.enabled_models || []).filter(
    (m) => m !== (enabledModels?.image_generation_model || '')
  );

  // Format cost for display (per 1M tokens by default)
  const formatCost = (cost) => {
    if (!cost) return 'Free';
    if (cost < 0.001) return `$${(cost * 1000000).toFixed(2)}`;
    if (cost < 1) return `$${(cost * 1000).toFixed(2)}`;
    return `$${cost.toFixed(3)}`;
  };

  // Format pricing string for display
  const formatPricing = (modelInfo) => {
    if (!modelInfo) return '';
    
    const parts = [];
    
    // Add context length
    if (modelInfo.context_length) {
      parts.push(`${modelInfo.context_length.toLocaleString()} ctx`);
    }
    
    // Add pricing if available
    if (modelInfo.input_cost || modelInfo.output_cost) {
      const inputPrice = modelInfo.input_cost ? formatCost(modelInfo.input_cost) : 'Free';
      const outputPrice = modelInfo.output_cost ? formatCost(modelInfo.output_cost) : 'Free';
      parts.push(`I/O: ${inputPrice} / ${outputPrice}`);
    }
    
    return parts.join(' • ');
  };

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
          {chatModels.length > 0 && (
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>AI Model</InputLabel>
              <Select
                value={currentModel?.current_model || ''}
                onChange={(e) => onModelSelect(e.target.value)}
                label="AI Model"
                disabled={isSelectingModel}
              >
                {chatModels.map((modelId) => {
                  const modelInfo = availableModels?.models?.find(m => m.id === modelId);
                  const isSelected = currentModel?.current_model === modelId;
                  const pricingInfo = formatPricing(modelInfo);
                  return (
                    <MenuItem key={modelId} value={modelId}>
                      <Box display="flex" alignItems="center" justifyContent="space-between" width="100%" sx={{ gap: 1 }}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            fontWeight: isSelected ? 'bold' : 'normal',
                            flex: 1,
                            textAlign: 'left'
                          }}
                        >
                          {modelInfo?.name || modelId}
                        </Typography>
                        <Box display="flex" alignItems="center" gap={1}>
                          {pricingInfo && (
                            <Typography 
                              variant="caption" 
                              color="text.secondary"
                              sx={{ textAlign: 'right', whiteSpace: 'nowrap' }}
                            >
                              {pricingInfo}
                            </Typography>
                          )}
                          {isSelected && (
                            <Chip 
                              label="Active" 
                              size="small" 
                              color="success" 
                              variant="outlined"
                            />
                          )}
                        </Box>
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
          {chatModels.length === 0 && (
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