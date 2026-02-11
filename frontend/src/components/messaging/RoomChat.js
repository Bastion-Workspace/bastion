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
  Button,
} from '@mui/material';
import {
  Send,
  ArrowBack,
  AttachFile,
  Close,
  Download,
} from '@mui/icons-material';
import { useMessaging } from '../../contexts/MessagingContext';
import PresenceIndicator from './PresenceIndicator';
import messagingService from '../../services/messagingService';
import AudioPlayer from '../AudioPlayer';
import TeamInvitationMessage from './TeamInvitationMessage';
import { formatTimestamp } from '../../utils/chatUtils';
import { useImageLightbox } from '../common/ImageLightbox';

const RoomChat = () => {
  const {
    user,
    currentRoom,
    messages,
    sendMessage,
    selectRoom,
    presence,
  } = useMessaging();

  const [inputValue, setInputValue] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const [messageAttachments, setMessageAttachments] = useState({});
  const { openLightbox } = useImageLightbox();
  const [imageBlobUrls, setImageBlobUrls] = useState({});
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textFieldRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load attachments for messages and create blob URLs
  useEffect(() => {
    const loadAttachments = async () => {
      const attachmentsMap = {};
      const blobUrlsMap = {};
      
      for (const message of messages) {
        if (message.message_id && !messageAttachments[message.message_id]) {
          try {
            const atts = await messagingService.getMessageAttachments(message.message_id);
            if (atts && atts.length > 0) {
              attachmentsMap[message.message_id] = atts;
              
              // Create blob URLs for images
              for (const att of atts) {
                if (!imageBlobUrls[att.attachment_id]) {
                  try {
                    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
                    const response = await fetch(
                      `/api/messaging/attachments/${att.attachment_id}/file`,
                      {
                        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                      }
                    );
                    if (response.ok) {
                      const blob = await response.blob();
                      const blobUrl = URL.createObjectURL(blob);
                      blobUrlsMap[att.attachment_id] = blobUrl;
                    }
                  } catch (error) {
                    console.error('Failed to load image blob:', error);
                  }
                }
              }
            }
          } catch (error) {
            console.error('Failed to load attachments:', error);
          }
        }
      }
      
      if (Object.keys(attachmentsMap).length > 0) {
        setMessageAttachments(prev => ({ ...prev, ...attachmentsMap }));
      }
      if (Object.keys(blobUrlsMap).length > 0) {
        setImageBlobUrls(prev => ({ ...prev, ...blobUrlsMap }));
      }
    };
    loadAttachments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      Object.values(imageBlobUrls).forEach(url => URL.revokeObjectURL(url));
    };
  }, [imageBlobUrls]);

  const handleSend = async () => {
    if ((!inputValue.trim() && !previewFile) || !currentRoom) return;

    try {
      // Send message first
      const message = await sendMessage(
        currentRoom.room_id,
        inputValue.trim() || (previewFile ? '📷' : ''),
        'text',
        null
      );

      // Upload attachment if preview exists
      if (previewFile && message && message.message_id) {
        try {
          const attachment = await messagingService.uploadAttachment(
            currentRoom.room_id,
            message.message_id,
            previewFile
          );
          
          console.log('✅ Attachment uploaded successfully:', attachment);
          
          // Reload attachments for this message after upload completes
          try {
            const atts = await messagingService.getMessageAttachments(message.message_id);
            console.log('📎 Reloaded attachments:', atts);
            
            if (atts && atts.length > 0) {
              // Update messageAttachments state
              setMessageAttachments(prev => ({
                ...prev,
                [message.message_id]: atts
              }));
              
              // Create blob URLs for newly uploaded images
              const blobUrlsMap = {};
              for (const att of atts) {
                if (att.mime_type?.startsWith('image/')) {
                  try {
                    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
                    const response = await fetch(
                      `/api/messaging/attachments/${att.attachment_id}/file`,
                      {
                        headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                      }
                    );
                    if (response.ok) {
                      const blob = await response.blob();
                      const blobUrl = URL.createObjectURL(blob);
                      blobUrlsMap[att.attachment_id] = blobUrl;
                      console.log('🖼️ Created blob URL for attachment:', att.attachment_id);
                    } else {
                      console.error('❌ Failed to fetch attachment file:', response.status, response.statusText);
                    }
                  } catch (error) {
                    console.error('❌ Failed to load image blob:', error);
                  }
                }
              }
              
              if (Object.keys(blobUrlsMap).length > 0) {
                setImageBlobUrls(prev => ({ ...prev, ...blobUrlsMap }));
                console.log('✅ Image blob URLs created:', Object.keys(blobUrlsMap));
              }
            } else {
              console.warn('⚠️ No attachments found after upload');
            }
          } catch (error) {
            console.error('❌ Failed to reload attachments after upload:', error);
          }
        } catch (error) {
          console.error('❌ Failed to upload attachment:', error);
          console.error('❌ Error details:', error.message, error.response);
        }
      }

      // Clear input and preview
      setInputValue('');
      setImagePreview(null);
      setPreviewFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
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

  const handlePaste = async (e) => {
    const items = e.clipboardData.items;
    for (let item of items) {
      if (item.type.indexOf('image') !== -1) {
        e.preventDefault();
        const file = item.getAsFile();
        await handleImageSelect(file);
        break;
      }
    }
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      await handleImageSelect(file);
    }
  };

  const handleImageSelect = async (file) => {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      alert('Only image files (JPG, PNG, GIF, WebP) are allowed');
      return;
    }

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setImagePreview(e.target.result);
      setPreviewFile(file);
    };
    reader.readAsDataURL(file);
  };

  const removePreview = () => {
    setImagePreview(null);
    setPreviewFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };


  if (!currentRoom) return null;

  const currentUserId = user?.user_id;
  const otherParticipants = currentRoom.participants?.filter(
    p => p.user_id !== currentUserId
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
          {!currentRoom.team_id && otherParticipants.length > 0 && (
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
            // Check if this is a team invitation message
            if (message.message_type === 'team_invitation' || message.metadata?.invitation_id) {
              return (
                <Box
                  key={message.message_id}
                  sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    my: 1
                  }}
                >
                  <TeamInvitationMessage message={message} />
                </Box>
              );
            }

            const isOwn = message.sender_id === currentUserId;
            
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
                  
                  {/* Display attachments */}
                  {messageAttachments[message.message_id]?.map((attachment) => (
                    <Box key={attachment.attachment_id} mt={1}>
                      {attachment.mime_type?.startsWith('audio/') ? (
                        <AudioPlayer
                          src={messagingService.getAttachmentUrl(attachment.attachment_id)}
                          filename={attachment.filename}
                        />
                      ) : attachment.mime_type?.startsWith('image/') ? (
                        <Box
                          component="img"
                          src={imageBlobUrls[attachment.attachment_id] || messagingService.getAttachmentUrl(attachment.attachment_id)}
                          alt={attachment.filename}
                          onClick={() => openLightbox(
                            imageBlobUrls[attachment.attachment_id] || messagingService.getAttachmentUrl(attachment.attachment_id),
                            { filename: attachment.filename }
                          )}
                          sx={{
                            maxWidth: '100%',
                            maxHeight: '300px',
                            height: 'auto',
                            borderRadius: 1,
                            cursor: 'pointer',
                            display: 'block',
                            objectFit: 'contain',
                          }}
                        />
                      ) : (
                        <Button
                          variant="outlined"
                          size="small"
                          href={messagingService.getAttachmentUrl(attachment.attachment_id)}
                          download={attachment.filename || 'attachment'}
                          startIcon={<Download />}
                        >
                          {attachment.filename || 'Download attachment'}
                        </Button>
                      )}
                    </Box>
                  ))}
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
                    {formatTimestamp(message.created_at)}
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
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
          {imagePreview && (
            <Box sx={{ position: 'relative', display: 'inline-block' }}>
              <Box
                component="img"
                src={imagePreview}
                alt="Preview"
                sx={{
                  maxWidth: '200px',
                  maxHeight: '200px',
                  borderRadius: 1,
                  border: '1px solid',
                  borderColor: 'divider',
                }}
              />
              <IconButton
                size="small"
                onClick={removePreview}
                sx={{
                  position: 'absolute',
                  top: 0,
                  right: 0,
                  backgroundColor: 'rgba(0,0,0,0.5)',
                  color: 'white',
                  '&:hover': { backgroundColor: 'rgba(0,0,0,0.7)' },
                }}
              >
                <Close fontSize="small" />
              </IconButton>
            </Box>
          )}
          <TextField
            ref={textFieldRef}
            fullWidth
            multiline
            maxRows={4}
            placeholder="Type a message or paste an image..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            onPaste={handlePaste}
            size="small"
          />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          <Tooltip title="Attach image">
            <IconButton
              size="small"
              onClick={() => fileInputRef.current?.click()}
            >
              <AttachFile fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Send">
            <IconButton
              color="primary"
              onClick={handleSend}
              disabled={!inputValue.trim() && !previewFile}
            >
              <Send fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Box>
  );
};

export default RoomChat;

