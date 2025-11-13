import React from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { Chat } from '@mui/icons-material';

const ChatPage = () => {
  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      height: '100%',
      p: 4
    }}>
      <Paper 
        elevation={2} 
        sx={{ 
          p: 4, 
          textAlign: 'center',
          maxWidth: 500
        }}
      >
        <Chat sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
        <Typography variant="h4" gutterBottom>
          Chat is Always Available
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          The chat sidebar is now available on every page. You can access your conversations, 
          send messages, and interact with AI models from anywhere in the application.
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Use the chat sidebar on the right to start a new conversation or continue an existing one.
        </Typography>
      </Paper>
    </Box>
  );
};

export default ChatPage;
