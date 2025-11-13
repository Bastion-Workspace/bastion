import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Cancel as CancelIcon,
  Schedule as ScheduleIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon
} from '@mui/icons-material';

/**
 * AsyncTaskProgress Component
 * Shows progress for async orchestrator tasks with cancellation option
 */
const AsyncTaskProgress = ({ 
  message, 
  onCancel 
}) => {
  const { progress, taskId, isPending, isCancelled, isError } = message;
  
  if (!isPending && !isCancelled && !isError) {
    return null; // Regular completed message
  }
  
  const getStatusColor = () => {
    if (isCancelled) return 'warning';
    if (isError) return 'error';
    if (isPending) return 'info';
    return 'success';
  };
  
  const getStatusIcon = () => {
    if (isCancelled) return <CancelIcon fontSize="small" />;
    if (isError) return <ErrorIcon fontSize="small" />;
    if (isPending) return <ScheduleIcon fontSize="small" />;
    return <CheckIcon fontSize="small" />;
  };
  
  const getStatusText = () => {
    if (isCancelled) return 'Cancelled';
    if (isError) return 'Failed';
    if (isPending) return 'Processing';
    return 'Completed';
  };

  return (
    <Box sx={{ mt: 1, p: 2, backgroundColor: 'rgba(0,0,0,0.05)', borderRadius: 1 }}>
      {/* Status Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Chip
          icon={getStatusIcon()}
          label={getStatusText()}
          color={getStatusColor()}
          size="small"
          variant="outlined"
        />
        
        {isPending && onCancel && (
          <Tooltip title="Cancel Task">
            <IconButton
              size="small"
              onClick={() => onCancel(taskId)}
              sx={{ ml: 1 }}
            >
              <CancelIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
      
      {/* Progress Bar for Pending Tasks */}
      {isPending && progress && (
        <Box sx={{ mb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
              {progress.message || 'Processing...'}
            </Typography>
            {progress.percentage !== undefined && (
              <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                {progress.percentage}%
              </Typography>
            )}
          </Box>
          
          <LinearProgress
            variant={progress.percentage !== undefined ? "determinate" : "indeterminate"}
            value={progress.percentage || 0}
            sx={{ 
              height: 6, 
              borderRadius: 3,
              backgroundColor: 'rgba(0,0,0,0.1)'
            }}
          />
        </Box>
      )}
      
      {/* Task Details */}
      {taskId && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Task ID: {taskId.substring(0, 8)}...
        </Typography>
      )}
      
      {/* Estimated Completion */}
      {isPending && message.metadata?.estimated_completion && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
          Estimated: {message.metadata.estimated_completion}
        </Typography>
      )}
    </Box>
  );
};

export default AsyncTaskProgress;
