/**
 * Roosevelt's Room Chat Interface
 * Message display and input for a specific room
 * 
 * BULLY! Real-time messaging in action!
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Typography,
  Paper,
  Avatar,
  Tooltip,
} from '@mui/material';
import {
  Send,
  ArrowBack,
} from '@mui/icons-material';
import { useMessaging } from '../../contexts/MessagingContext';
import PresenceIndicator from './PresenceIndicator';

const RoomChat = () => {
  const {
    currentRoom,
    messages,
    sendMessage,
    selectRoom,
    presence,
  } = useMessaging();

  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim() || !currentRoom) return;

    try {
      await sendMessage(currentRoom.room_id, inputValue.trim());
      setInputValue('');
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatMessageTime = (timestamp) => {
    if (!timestamp) return '';
    
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  if (!currentRoom) return null;

  const otherParticipants = currentRoom.participants?.filter(
    p => p.user_id !== currentRoom.created_by
  ) || [];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Room header */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <IconButton size="small" onClick={() => selectRoom(null)}>
          <ArrowBack />
        </IconButton>
        <Box sx={{ flex: 1 }}>
          <Typography variant="subtitle1">
            {currentRoom.display_name || currentRoom.room_name}
          </Typography>
          {otherParticipants.length > 0 && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PresenceIndicator
                status={presence[otherParticipants[0].user_id]?.status || 'offline'}
                size="small"
              />
              <Typography variant="caption" color="text.secondary">
                {presence[otherParticipants[0].user_id]?.status || 'offline'}
              </Typography>
            </Box>
          )}
        </Box>
      </Box>

      {/* Messages area */}
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
      >
        {messages.length === 0 ? (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
            }}
          >
            <Typography color="text.secondary">
              No messages yet. Start the conversation!
            </Typography>
          </Box>
        ) : (
          messages.map((message) => {
            const isOwn = message.sender_id === currentRoom.created_by;
            
            return (
              <Box
                key={message.message_id}
                sx={{
                  display: 'flex',
                  justifyContent: isOwn ? 'flex-end' : 'flex-start',
                  gap: 1,
                }}
              >
                {!isOwn && (
                  <Avatar sx={{ width: 32, height: 32 }}>
                    {message.display_name?.charAt(0) || message.username?.charAt(0) || '?'}
                  </Avatar>
                )}
                <Paper
                  elevation={1}
                  sx={{
                    p: 1.5,
                    maxWidth: '70%',
                    backgroundColor: isOwn ? 'primary.main' : 'background.paper',
                    color: isOwn ? 'primary.contrastText' : 'text.primary',
                  }}
                >
                  {!isOwn && (
                    <Typography variant="caption" sx={{ fontWeight: 600, color: 'inherit' }}>
                      {message.display_name || message.username}
                    </Typography>
                  )}
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      whiteSpace: 'pre-wrap', 
                      wordBreak: 'break-word',
                      color: 'inherit'
                    }}
                  >
                    {message.content}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{
                      display: 'block',
                      mt: 0.5,
                      opacity: 0.7,
                      fontSize: '0.7rem',
                      color: 'inherit',
                    }}
                  >
                    {formatMessageTime(message.created_at)}
                  </Typography>
                </Paper>
              </Box>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input area */}
      <Box
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: 'divider',
          display: 'flex',
          gap: 1,
        }}
      >
        <TextField
          fullWidth
          multiline
          maxRows={4}
          placeholder="Type a message..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          size="small"
        />
        <Tooltip title="Send">
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={!inputValue.trim()}
          >
            <Send />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
};

export default RoomChat;

