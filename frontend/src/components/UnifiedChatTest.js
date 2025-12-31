import React, { useState, useEffect } from 'react';
import { useChatManagerUnified } from '../hooks/useChatManagerUnified';
import { Box, TextField, Button, Typography, Paper, List, ListItem, ListItemText, Chip, CircularProgress } from '@mui/material';
import { formatTimestamp } from '../utils/chatUtils';

/**
 * Test component for the unified background processing system
 * Demonstrates the new architecture with integrated intent classification
 */
const UnifiedChatTest = () => {
  const [conversationId] = useState(`test_conv_${Date.now()}`);
  const [sessionId] = useState(`test_session_${Date.now()}`);
  const [inputValue, setInputValue] = useState('');
  
  const {
    messages,
    isLoading,
    error,
    pendingJobs,
    sendMessage,
    clearMessages,
    cancelJob
  } = useChatManagerUnified(conversationId, sessionId);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputValue.trim()) {
      await sendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const getExecutionModeColor = (mode) => {
    switch (mode) {
      case 'chat': return 'primary';
      case 'direct': return 'secondary';
      case 'plan': return 'warning';
      case 'execute': return 'success';
      default: return 'default';
    }
  };

  const getExecutionModeIcon = (mode) => {
    switch (mode) {
      case 'chat': return '💬';
      case 'direct': return '🔍';
      case 'plan': return '📋';
      case 'execute': return '🚀';
      default: return '🔄';
    }
  };

  return (
    <Box sx={{ maxWidth: 800, margin: '0 auto', padding: 2 }}>
      <Typography variant="h4" gutterBottom>
        🎯 Unified Background Processing Test
      </Typography>
      
      <Typography variant="body2" color="text.secondary" paragraph>
        This test page demonstrates the new unified background processing system with integrated intent classification.
        All messages are automatically classified and processed in the background.
      </Typography>

      {/* Test Queries */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>Quick Test Queries</Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          <Button 
            size="small" 
            variant="outlined"
            onClick={() => setInputValue("Hello, how are you?")}
          >
            💬 Chat: "Hello, how are you?"
          </Button>
          <Button 
            size="small" 
            variant="outlined"
            onClick={() => setInputValue("What is artificial intelligence?")}
          >
            🔍 Direct: "What is artificial intelligence?"
          </Button>
          <Button 
            size="small" 
            variant="outlined"
            onClick={() => setInputValue("Research the impact of climate change on global economics")}
          >
            📋 Plan: "Research climate change economics"
          </Button>
        </Box>
      </Paper>

      {/* Pending Jobs */}
      {pendingJobs.length > 0 && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <Typography variant="h6" gutterBottom>
            🔄 Active Jobs ({pendingJobs.length})
          </Typography>
          <List dense>
            {pendingJobs.map((job) => (
              <ListItem key={job.id} sx={{ pl: 0 }}>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip 
                        label={`${getExecutionModeIcon(job.execution_mode)} ${job.execution_mode}`}
                        color={getExecutionModeColor(job.execution_mode)}
                        size="small"
                      />
                      <Typography variant="body2">
                        {job.query.substring(0, 50)}...
                      </Typography>
                      {job.progress && (
                        <CircularProgress size={16} />
                      )}
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        Confidence: {(job.confidence * 100).toFixed(1)}% | {job.reasoning}
                      </Typography>
                      {job.progress && (
                        <Typography variant="caption" display="block">
                          Progress: {job.progress.status} ({job.progress.current_iteration}/{job.progress.max_iterations})
                        </Typography>
                      )}
                    </Box>
                  }
                />
                <Button 
                  size="small" 
                  color="error"
                  onClick={() => cancelJob(job.id)}
                >
                  Cancel
                </Button>
              </ListItem>
            ))}
          </List>
        </Paper>
      )}

      {/* Error Display */}
      {error && (
        <Paper sx={{ p: 2, mb: 2, bgcolor: 'error.light' }}>
          <Typography color="error">
            ❌ Error: {error}
          </Typography>
        </Paper>
      )}

      {/* Chat Input */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <form onSubmit={handleSubmit}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Enter your message to test unified processing..."
              disabled={isLoading}
              multiline
              maxRows={3}
            />
            <Button 
              type="submit" 
              variant="contained" 
              disabled={isLoading || !inputValue.trim()}
              sx={{ minWidth: 100 }}
            >
              {isLoading ? <CircularProgress size={24} /> : 'Send'}
            </Button>
          </Box>
        </form>
        <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
          <Button size="small" onClick={clearMessages}>
            Clear Messages
          </Button>
          <Typography variant="caption" color="text.secondary">
            Conversation ID: {conversationId}
          </Typography>
        </Box>
      </Paper>

      {/* Messages */}
      <Paper sx={{ p: 2, maxHeight: 600, overflow: 'auto' }}>
        <Typography variant="h6" gutterBottom>
          💬 Messages ({messages.length})
        </Typography>
        
        {messages.length === 0 ? (
          <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
            No messages yet. Send a message to test the unified processing system!
          </Typography>
        ) : (
          <List>
            {messages.map((message, index) => (
              <ListItem key={message.id || index} sx={{ pl: 0, alignItems: 'flex-start' }}>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Chip 
                        label={message.type === 'user' ? '👤 You' : '🤖 Assistant'}
                        color={message.type === 'user' ? 'primary' : 'secondary'}
                        size="small"
                      />
                      {message.execution_mode && (
                        <Chip 
                          label={`${getExecutionModeIcon(message.execution_mode)} ${message.execution_mode}`}
                          color={getExecutionModeColor(message.execution_mode)}
                          size="small"
                          variant="outlined"
                        />
                      )}
                      {message.isPending && (
                        <Chip 
                          label="⏳ Processing"
                          color="warning"
                          size="small"
                          icon={<CircularProgress size={12} />}
                        />
                      )}
                      {message.isError && (
                        <Chip 
                          label="❌ Error"
                          color="error"
                          size="small"
                        />
                      )}
                      <Typography variant="caption" color="text.secondary">
                        {formatTimestamp(message.timestamp)}
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          whiteSpace: 'pre-wrap',
                          color: message.isError ? 'error.main' : 'text.primary'
                        }}
                      >
                        {message.content}
                      </Typography>
                      
                      {message.citations && message.citations.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" color="text.secondary">
                            📚 Citations ({message.citations.length}):
                          </Typography>
                          {message.citations.slice(0, 3).map((citation, idx) => (
                            <Chip 
                              key={idx}
                              label={citation.document_title}
                              size="small"
                              variant="outlined"
                              sx={{ ml: 0.5, mt: 0.5 }}
                            />
                          ))}
                        </Box>
                      )}
                      
                      {message.processing_time && (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                          ⏱️ Processing time: {message.processing_time.toFixed(2)}s
                        </Typography>
                      )}
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        )}
      </Paper>

      {/* System Info */}
      <Paper sx={{ p: 2, mt: 2, bgcolor: 'grey.50' }}>
        <Typography variant="caption" color="text.secondary">
          🔧 System: Unified Background Processing with Intent Classification<br/>
          📡 WebSocket: {sessionId}<br/>
          💾 Conversation: {conversationId}<br/>
          🎯 All messages are automatically classified and processed in background workers
        </Typography>
      </Paper>
    </Box>
  );
};

export default UnifiedChatTest;
