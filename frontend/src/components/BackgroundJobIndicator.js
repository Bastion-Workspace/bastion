import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { 
  Box, 
  Typography, 
  LinearProgress, 
  Chip, 
  IconButton, 
  Collapse,
  Alert,
  Paper,
  Button
} from '@mui/material';
import { 
  ExpandMore, 
  ExpandLess, 
  Cancel, 
  Refresh,
  CheckCircle,
  Error as ErrorIcon,
  Schedule
} from '@mui/icons-material';

/**
 * Component that displays background job progress and handles reconnection
 */
const BackgroundJobIndicator = forwardRef(({ 
  backgroundJobService, 
  conversationId, 
  onJobCompleted, 
  onJobError 
}, ref) => {
  const [ongoingJobs, setOngoingJobs] = useState([]);
  const [completedJobs, setCompletedJobs] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);

  // Expose methods to parent component
  useImperativeHandle(ref, () => ({
    connectToNewJob: (jobId) => {
      console.log(`🔗 Connecting to new job: ${jobId}`);
      
      // Add job to ongoing jobs immediately
      setOngoingJobs(prev => {
        const exists = prev.find(j => j.job_id === jobId);
        if (!exists) {
          return [...prev, { 
            job_id: jobId, 
            status: 'queued',
            progress: {},
            query: 'New job...'
          }];
        }
        return prev;
      });
      
      // Connect to job progress WebSocket immediately
      if (backgroundJobService) {
        backgroundJobService.connectToJobProgress(jobId, {
          onProgress: (jobId, progress) => {
            setOngoingJobs(prev => prev.map(j => 
              j.job_id === jobId ? { ...j, progress, job_status: progress.status || j.job_status } : j
            ));
          },
          onComplete: (jobId, result) => {
            // Move from ongoing to completed
            setOngoingJobs(prev => {
              const job = prev.find(j => j.job_id === jobId);
              if (job) {
                const completedJob = { ...job, result };
                setCompletedJobs(completed => [...completed, completedJob]);
                if (onJobCompleted) onJobCompleted(completedJob);
              }
              return prev.filter(j => j.job_id !== jobId);
            });
          },
          onError: (jobId, error) => {
            console.error(`❌ Job ${jobId} error:`, error);
            if (onJobError) onJobError(error);
          }
        });
      }
    }
  }));

  useEffect(() => {
    if (conversationId && backgroundJobService) {
      checkForOngoingJobs();
    }
  }, [conversationId, backgroundJobService]);

  const checkForOngoingJobs = async () => {
    if (!conversationId || !backgroundJobService) return;

    try {
      setLoading(true);
      
      // Check for ongoing jobs and reconnect
      await backgroundJobService.checkAndReconnectToOngoingJobs(
        conversationId,
        handleJobFound
      );
      
    } catch (error) {
      console.error('❌ Failed to check ongoing jobs:', error);
      if (onJobError) onJobError(error);
    } finally {
      setLoading(false);
    }
  };

  const handleJobFound = (job) => {
    if (job.status === 'completed') {
      // Job finished while we were away
      setCompletedJobs(prev => {
        const exists = prev.find(j => j.job_id === job.job_id);
        if (!exists) {
          if (onJobCompleted) onJobCompleted(job);
          return [...prev, job];
        }
        return prev;
      });
      
      // Remove from ongoing if it was there
      setOngoingJobs(prev => prev.filter(j => j.job_id !== job.job_id));
      
    } else if (job.status === 'running' || job.status === 'queued') {
      // Job is still ongoing
      setOngoingJobs(prev => {
        const exists = prev.find(j => j.job_id === job.job_id);
        if (!exists) {
          return [...prev, { 
            ...job, 
            progress: job.progress_data || {},
            reconnected: true 
          }];
        }
        return prev;
      });
      
      // Connect to progress updates
      backgroundJobService.connectToJobProgress(job.job_id, {
        onProgress: (jobId, progress) => {
          setOngoingJobs(prev => prev.map(j => 
            j.job_id === jobId ? { ...j, progress, job_status: progress.status || j.job_status } : j
          ));
        },
        onComplete: (jobId, result) => {
          // Move from ongoing to completed
          setOngoingJobs(prev => {
            const job = prev.find(j => j.job_id === jobId);
            if (job) {
              const completedJob = { ...job, result };
              setCompletedJobs(completed => [...completed, completedJob]);
              if (onJobCompleted) onJobCompleted(completedJob);
            }
            return prev.filter(j => j.job_id !== jobId);
          });
        },
        onError: (jobId, error) => {
          console.error(`❌ Job ${jobId} error:`, error);
          if (onJobError) onJobError(error);
        }
      });
    }
  };

  const handleCancelJob = async (jobId) => {
    try {
      await backgroundJobService.cancelJob(jobId);
      setOngoingJobs(prev => prev.filter(j => j.job_id !== jobId));
    } catch (error) {
      console.error(`❌ Failed to cancel job ${jobId}:`, error);
      if (onJobError) onJobError(error);
    }
  };

  const getJobStatusColor = (status) => {
    switch (status) {
      case 'queued': return 'default';
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'cancelled': return 'warning';
      default: return 'default';
    }
  };

  const getJobStatusIcon = (status) => {
    switch (status) {
      case 'queued': return <Schedule />;
      case 'running': return null; // Will show progress
      case 'completed': return <CheckCircle />;
      case 'failed': return <ErrorIcon />;
      case 'cancelled': return <Cancel />;
      default: return null;
    }
  };

  const formatProgress = (progress) => {
    if (!progress || typeof progress !== 'object') return '';
    
    const { current_iteration, max_iterations, current_tool, status } = progress;
    
    if (current_iteration && max_iterations) {
      const percentage = Math.round((current_iteration / max_iterations) * 100);
      return `${percentage}% (${current_iteration}/${max_iterations})`;
    }
    
    if (current_tool) {
      return `Using ${current_tool}`;
    }
    
    if (status) {
      return status.replace(/_/g, ' ');
    }
    
    return 'Processing...';
  };

  const totalJobs = ongoingJobs.length + completedJobs.length;

  if (totalJobs === 0) return null;

  return (
    <Paper 
      elevation={1} 
      sx={{ 
        mb: 2, 
        p: 2, 
        backgroundColor: '#f8f9fa',
        border: '1px solid #e9ecef'
      }}
    >
      <Box 
        display="flex" 
        alignItems="center" 
        justifyContent="space-between"
        onClick={() => setExpanded(!expanded)}
        sx={{ cursor: 'pointer' }}
      >
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="subtitle2" fontWeight={600}>
            Background Research Jobs
          </Typography>
          {loading && <LinearProgress size={20} />}
          {ongoingJobs.length > 0 && (
            <Chip 
              label={`${ongoingJobs.length} active`} 
              color="primary" 
              size="small" 
            />
          )}
          {completedJobs.length > 0 && (
            <Chip 
              label={`${completedJobs.length} completed`} 
              color="success" 
              size="small" 
            />
          )}
        </Box>
        
        <Box display="flex" alignItems="center" gap={1}>
          <IconButton size="small" onClick={(e) => { e.stopPropagation(); checkForOngoingJobs(); }}>
            <Refresh />
          </IconButton>
          <IconButton size="small">
            {expanded ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
        </Box>
      </Box>

      <Collapse in={expanded}>
        <Box mt={2}>
          {/* Ongoing Jobs */}
          {ongoingJobs.map((job) => (
            <Box key={job.job_id} mb={2}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Box display="flex" alignItems="center" justifyContent="between" mb={1}>
                  <Box flex={1}>
                    <Typography variant="body2" fontWeight={500}>
                      {job.query?.substring(0, 80)}...
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {job.reconnected && "Reconnected to ongoing job • "}
                      {new Date(job.created_at).toLocaleTimeString()}
                    </Typography>
                  </Box>
                  
                  <Box display="flex" alignItems="center" gap={1}>
                    <Chip 
                      label={job.job_status || 'running'}
                      color={getJobStatusColor(job.job_status)} 
                      size="small"
                      icon={getJobStatusIcon(job.job_status)}
                    />
                    <IconButton 
                      size="small" 
                      onClick={() => handleCancelJob(job.job_id)}
                      title="Cancel job"
                    >
                      <Cancel />
                    </IconButton>
                  </Box>
                </Box>
                
                {job.progress && (
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      {formatProgress(job.progress)}
                    </Typography>
                    {job.progress.current_iteration && job.progress.max_iterations && (
                      <LinearProgress 
                        variant="determinate" 
                        value={(job.progress.current_iteration / job.progress.max_iterations) * 100}
                        sx={{ mt: 0.5 }}
                      />
                    )}
                  </Box>
                )}
              </Paper>
            </Box>
          ))}

          {/* Completed Jobs */}
          {completedJobs.map((job) => (
            <Box key={job.job_id} mb={1}>
              <Alert 
                severity="success" 
                variant="outlined"
                action={
                  <Button 
                    size="small" 
                    onClick={() => {
                      // Scroll to the completed message in chat
                      if (onJobCompleted) onJobCompleted(job);
                    }}
                  >
                    View Result
                  </Button>
                }
              >
                <Typography variant="body2">
                  Research completed: {job.query?.substring(0, 60)}...
                </Typography>
              </Alert>
            </Box>
          ))}

          {totalJobs === 0 && (
            <Typography variant="body2" color="text.secondary" align="center">
              No background jobs for this conversation
            </Typography>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
});

export default BackgroundJobIndicator; 