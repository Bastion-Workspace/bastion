/**
 * Chat input with @-mention autocomplete, reply-to bar, image preview, and typing indicator.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Box, TextField, IconButton, Tooltip, Typography, Paper, List, ListItemButton, ListItemText, ListItemAvatar, Avatar,
} from '@mui/material';
import { Send, AttachFile, Close, SmartToy } from '@mui/icons-material';
import messagingService from '../../services/messagingService';

const RoomChatInput = ({
  roomId,
  onSend,
  typingUsers,
  replyTo,
  onCancelReply,
  editingMessage,
  onCancelEdit,
  sendDisabled = false,
}) => {
  const [inputValue, setInputValue] = useState('');
  const [imagePreview, setImagePreview] = useState(null);
  const [previewFile, setPreviewFile] = useState(null);
  const [mentionables, setMentionables] = useState([]);
  const [mentionQuery, setMentionQuery] = useState(null);
  const [mentionAnchorIdx, setMentionAnchorIdx] = useState(-1);
  const [pendingMentions, setPendingMentions] = useState([]);

  const fileInputRef = useRef(null);
  const typingTimeoutRef = useRef(null);
  const isTypingRef = useRef(false);

  useEffect(() => {
    if (!roomId) return;
    messagingService.getMentionables(roomId).then(setMentionables).catch(() => {});
  }, [roomId]);

  useEffect(() => {
    if (editingMessage) {
      setInputValue(editingMessage.content || '');
    }
  }, [editingMessage]);

  const sendTyping = useCallback((typing) => {
    if (roomId) messagingService.sendTypingIndicator(roomId, typing);
  }, [roomId]);

  const stopTyping = useCallback(() => {
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    if (isTypingRef.current) {
      sendTyping(false);
      isTypingRef.current = false;
    }
  }, [sendTyping]);

  const handleChange = useCallback((e) => {
    const val = e.target.value;
    setInputValue(val);

    if (!isTypingRef.current) {
      isTypingRef.current = true;
      sendTyping(true);
    }
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    typingTimeoutRef.current = setTimeout(() => {
      sendTyping(false);
      isTypingRef.current = false;
    }, 2000);

    const cursor = e.target.selectionStart;
    const before = val.slice(0, cursor);
    const atMatch = before.match(/@(\w*)$/);
    if (atMatch) {
      setMentionQuery(atMatch[1].toLowerCase());
      setMentionAnchorIdx(atMatch.index);
    } else {
      setMentionQuery(null);
      setMentionAnchorIdx(-1);
    }
  }, [sendTyping]);

  const filteredMentionables = mentionQuery !== null
    ? mentionables.filter(m => {
        const name = (m.display_name || m.handle || '').toLowerCase();
        return name.includes(mentionQuery);
      }).slice(0, 8)
    : [];

  const insertMention = useCallback((mentionable) => {
    const displayName = mentionable.display_name || mentionable.handle;
    const before = inputValue.slice(0, mentionAnchorIdx);
    const after = inputValue.slice(mentionAnchorIdx + 1 + (mentionQuery?.length || 0));
    setInputValue(`${before}@${displayName} ${after}`);
    setMentionQuery(null);
    setMentionAnchorIdx(-1);

    const mention = { type: mentionable.type, display_name: displayName };
    if (mentionable.type === 'user') mention.user_id = mentionable.user_id;
    if (mentionable.type === 'agent') mention.agent_profile_id = mentionable.agent_profile_id;
    setPendingMentions(prev => [...prev, mention]);
  }, [inputValue, mentionAnchorIdx, mentionQuery]);

  const handleSend = useCallback(async () => {
    if (sendDisabled && !editingMessage) return;
    if (!inputValue.trim() && !previewFile) return;
    stopTyping();
    await onSend({
      content: inputValue.trim() || (previewFile ? '📷' : ''),
      mentions: pendingMentions.length > 0 ? pendingMentions : null,
      replyToMessageId: replyTo?.message_id || null,
      editingMessageId: editingMessage?.message_id || null,
      file: previewFile,
    });
    setInputValue('');
    setPendingMentions([]);
    setImagePreview(null);
    setPreviewFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, [inputValue, previewFile, pendingMentions, replyTo, editingMessage, onSend, stopTyping, sendDisabled]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (filteredMentionables.length > 0 && mentionQuery !== null) {
        insertMention(filteredMentionables[0]);
      } else {
        handleSend();
      }
    }
  };

  const handlePaste = async (e) => {
    for (const item of e.clipboardData.items) {
      if (item.type.indexOf('image') !== -1) {
        e.preventDefault();
        const file = item.getAsFile();
        handleImageSelect(file);
        break;
      }
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file?.type.startsWith('image/')) handleImageSelect(file);
  };

  const handleImageSelect = (file) => {
    const allowed = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowed.includes(file.type) || file.size > 10 * 1024 * 1024) return;
    const reader = new FileReader();
    reader.onload = (e) => { setImagePreview(e.target.result); setPreviewFile(file); };
    reader.readAsDataURL(file);
  };

  useEffect(() => () => {
    if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
  }, []);

  return (
    <Box>
      {typingUsers?.length > 0 && (
        <Box sx={{ px: 2, py: 0.5 }}>
          <Typography variant="caption" color="text.secondary" sx={{ fontStyle: 'italic' }}>
            {typingUsers.length === 1
              ? `${typingUsers[0].display_name || typingUsers[0].user_id} is typing...`
              : `${typingUsers.length} people are typing...`}
          </Typography>
        </Box>
      )}

      {/* Reply-to bar */}
      {replyTo && (
        <Box sx={{ px: 2, pt: 1, display: 'flex', alignItems: 'center', gap: 1, borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ borderLeft: 2, borderColor: 'primary.main', pl: 1, flex: 1, minWidth: 0 }}>
            <Typography variant="caption" sx={{ fontWeight: 600 }}>
              Replying to {replyTo.display_name || replyTo.username}
            </Typography>
            <Typography variant="caption" display="block" noWrap color="text.secondary">
              {replyTo.content?.slice(0, 100)}
            </Typography>
          </Box>
          <IconButton size="small" onClick={onCancelReply}><Close fontSize="small" /></IconButton>
        </Box>
      )}

      {/* Edit bar */}
      {editingMessage && (
        <Box sx={{ px: 2, pt: 1, display: 'flex', alignItems: 'center', gap: 1, borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ borderLeft: 2, borderColor: 'warning.main', pl: 1, flex: 1, minWidth: 0 }}>
            <Typography variant="caption" sx={{ fontWeight: 600, color: 'warning.main' }}>
              Editing message
            </Typography>
          </Box>
          <IconButton size="small" onClick={onCancelEdit}><Close fontSize="small" /></IconButton>
        </Box>
      )}

      {/* Mention autocomplete dropdown */}
      {filteredMentionables.length > 0 && (
        <Paper elevation={4} sx={{ mx: 2, mb: 0.5, maxHeight: 200, overflow: 'auto' }}>
          <List dense disablePadding>
            {filteredMentionables.map((m, i) => (
              <ListItemButton key={`${m.type}-${m.user_id || m.agent_profile_id}-${i}`} onClick={() => insertMention(m)}>
                <ListItemAvatar sx={{ minWidth: 36 }}>
                  <Avatar sx={{ width: 24, height: 24, fontSize: '0.75rem' }}>
                    {m.type === 'agent' ? <SmartToy sx={{ fontSize: 16 }} /> : (m.display_name?.charAt(0) || '?')}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={m.display_name || m.handle}
                  secondary={m.type === 'agent' ? 'Agent' : 'User'}
                  primaryTypographyProps={{ variant: 'body2' }}
                  secondaryTypographyProps={{ variant: 'caption' }}
                />
              </ListItemButton>
            ))}
          </List>
        </Paper>
      )}

      <Box sx={{ p: 2, borderTop: (replyTo || editingMessage) ? 0 : 1, borderColor: 'divider', display: 'flex', gap: 1 }}>
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
          {imagePreview && (
            <Box sx={{ position: 'relative', display: 'inline-block' }}>
              <Box component="img" src={imagePreview} alt="Preview" sx={{ maxWidth: 200, maxHeight: 200, borderRadius: 1, border: '1px solid', borderColor: 'divider' }} />
              <IconButton
                size="small"
                onClick={() => { setImagePreview(null); setPreviewFile(null); if (fileInputRef.current) fileInputRef.current.value = ''; }}
                sx={{ position: 'absolute', top: 0, right: 0, bgcolor: 'rgba(0,0,0,0.5)', color: 'white', '&:hover': { bgcolor: 'rgba(0,0,0,0.7)' } }}
              >
                <Close fontSize="small" />
              </IconButton>
            </Box>
          )}
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder={editingMessage ? 'Edit your message...' : 'Type a message... Use @ to mention'}
            value={inputValue}
            onChange={handleChange}
            onKeyPress={handleKeyPress}
            onPaste={handlePaste}
            onBlur={stopTyping}
            size="small"
            disabled={sendDisabled && !editingMessage}
          />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileSelect} />
          {!editingMessage && (
            <Tooltip title="Attach image">
              <IconButton size="small" onClick={() => fileInputRef.current?.click()} disabled={sendDisabled}>
                <AttachFile fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title={editingMessage ? 'Save edit' : 'Send'}>
            <IconButton
              color="primary"
              onClick={handleSend}
              disabled={
                (sendDisabled && !editingMessage) || (!inputValue.trim() && !previewFile)
              }
            >
              <Send fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
    </Box>
  );
};

export default React.memo(RoomChatInput);
