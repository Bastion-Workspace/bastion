import React from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
} from '@mui/material';
import { Send } from '@mui/icons-material';

const ChatInput = ({
  query,
  onQueryChange,
  onSendMessage,
  onKeyPress,
  isLoading,
  messages,
  textFieldRef,
  forceExecutionMode,
  onSetForceExecutionMode,
  currentConversationId,
  sessionId,
}) => {
  return (
    <Paper elevation={3} sx={{ p: 2, mt: 2, mx: 2 }}>
      <Box display="flex" gap={2} alignItems="flex-end">
        <TextField
          ref={textFieldRef}
          fullWidth
          multiline
          maxRows={4}
          placeholder="Chat naturally or ask research questions - AI will determine the best approach..."
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyPress={onKeyPress}
          disabled={isLoading && !messages.some(msg => msg.isResearchJob)}
          variant="outlined"
          size="small"
        />
        <Button
          variant="contained"
          endIcon={<Send />}
          onClick={onSendMessage}
          disabled={!query.trim() || (isLoading && !messages.some(msg => msg.isResearchJob))}
          sx={{ px: 3, py: 1.5 }}
        >
          Send
        </Button>
      </Box>
      
      {/* Execution Mode Override Buttons */}
      {!isLoading && query.trim() && (
        <Box display="flex" gap={1} alignItems="center" mt={1}>
          <Typography variant="caption" color="text.secondary">
            Override mode:
          </Typography>
          <Button
            size="small"
            variant={forceExecutionMode === 'chat' ? 'contained' : 'outlined'}
            onClick={() => onSetForceExecutionMode('chat')}
            sx={{ fontSize: '0.75rem', py: 0.25, px: 1 }}
          >
            Chat
          </Button>
          <Button
            size="small"
            variant={forceExecutionMode === 'plan' ? 'contained' : 'outlined'}
            onClick={() => onSetForceExecutionMode('plan')}
            sx={{ fontSize: '0.75rem', py: 0.25, px: 1 }}
          >
            Research
          </Button>
          <Button
            size="small"
            variant={forceExecutionMode === 'direct' ? 'contained' : 'outlined'}
            onClick={() => onSetForceExecutionMode('direct')}
            sx={{ fontSize: '0.75rem', py: 0.25, px: 1 }}
          >
            Direct
          </Button>
          {forceExecutionMode && (
            <Button
              size="small"
              onClick={() => onSetForceExecutionMode(null)}
              sx={{ fontSize: '0.75rem', py: 0.25, px: 1, minWidth: 'auto' }}
            >
              ✕
            </Button>
          )}
        </Box>
      )}
      
      {/* Session info */}
      <Typography variant="caption" color="text.secondary">
        {currentConversationId ? `Conversation: ${currentConversationId.slice(-8)}` : `Session: ${sessionId.slice(-8)}`}
      </Typography>
    </Paper>
  );
};

export default ChatInput; 