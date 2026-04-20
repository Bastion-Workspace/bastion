/**
 * Room Chat Interface
 * Orchestrates message display, input, reactions, search, reply, and edit.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Box,
  IconButton,
  Typography,
  Avatar,
  Popover,
  Tooltip,
  TextField,
  InputAdornment,
  Alert,
  Chip,
} from '@mui/material';
import { ArrowBack, Search, Close, Hub } from '@mui/icons-material';
import { useMessaging } from '../../contexts/MessagingContext';
import PresenceIndicator from './PresenceIndicator';
import messagingService from '../../services/messagingService';
import apiService from '../../services/apiService';
import TeamInvitationMessage from './TeamInvitationMessage';
import RoomMessageBubble from './RoomMessageBubble';
import RoomChatInput from './RoomChatInput';
import { useImageLightbox } from '../common/ImageLightbox';

const QUICK_EMOJIS = ['👍', '❤️', '😂', '😮', '😢', '🎉', '🔥', '👀'];

const RoomChat = () => {
  const {
    user,
    currentRoom,
    messages,
    sendMessage,
    editMessage,
    selectRoom,
    presence,
    federatedPresenceByPeer,
    federatedAttachmentsByMessage,
    federatedReadReceiptByRoom,
    addReaction,
    removeReaction,
    typingUsers,
  } = useMessaging();

  const [messageAttachments, setMessageAttachments] = useState({});
  const [imageBlobUrls, setImageBlobUrls] = useState({});
  const [emojiAnchor, setEmojiAnchor] = useState(null);
  const [emojiTargetMessageId, setEmojiTargetMessageId] = useState(null);
  const [replyTo, setReplyTo] = useState(null);
  const [editingMessage, setEditingMessage] = useState(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [federationRoomStatus, setFederationRoomStatus] = useState(null);
  const { openLightbox } = useImageLightbox();
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollToBottom(); }, [messages]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!currentRoom || currentRoom.room_type !== 'federated') {
        setFederationRoomStatus(null);
        return;
      }
      try {
        const s = await apiService.get(
          `/api/federation/rooms/${encodeURIComponent(currentRoom.room_id)}/status`
        );
        if (!cancelled) setFederationRoomStatus(s);
      } catch {
        if (!cancelled) setFederationRoomStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [currentRoom?.room_id, currentRoom?.room_type]);

  // Load attachments
  useEffect(() => {
    const load = async () => {
      const aMap = {}, bMap = {};
      for (const msg of messages) {
        if (msg.message_id && !messageAttachments[msg.message_id]) {
          try {
            const atts = await messagingService.getMessageAttachments(msg.message_id);
            if (atts?.length > 0) {
              aMap[msg.message_id] = atts;
              for (const att of atts) {
                if (!imageBlobUrls[att.attachment_id]) {
                  try {
                    const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
                    const resp = await fetch(`/api/messaging/attachments/${att.attachment_id}/file`, {
                      headers: token ? { Authorization: `Bearer ${token}` } : {},
                    });
                    if (resp.ok) bMap[att.attachment_id] = URL.createObjectURL(await resp.blob());
                  } catch {}
                }
              }
            }
          } catch {}
        }
      }
      for (const att of Object.values(federatedAttachmentsByMessage).flat()) {
        if (att?.attachment_id && att.mime_type?.startsWith('image/') && !imageBlobUrls[att.attachment_id]) {
          try {
            const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
            const resp = await fetch(`/api/messaging/attachments/${att.attachment_id}/file`, {
              headers: token ? { Authorization: `Bearer ${token}` } : {},
            });
            if (resp.ok) bMap[att.attachment_id] = URL.createObjectURL(await resp.blob());
          } catch {}
        }
      }
      if (Object.keys(aMap).length) setMessageAttachments(p => ({ ...p, ...aMap }));
      if (Object.keys(bMap).length) setImageBlobUrls(p => ({ ...p, ...bMap }));
    };
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, federatedAttachmentsByMessage]);

  useEffect(() => () => {
    Object.values(imageBlobUrls).forEach(url => URL.revokeObjectURL(url));
  }, [imageBlobUrls]);

  // Search
  useEffect(() => {
    if (!searchQuery.trim() || !currentRoom) { setSearchResults([]); return; }
    const t = setTimeout(async () => {
      try {
        const res = await messagingService.searchMessages(currentRoom.room_id, searchQuery);
        setSearchResults(res.results || []);
      } catch { setSearchResults([]); }
    }, 300);
    return () => clearTimeout(t);
  }, [searchQuery, currentRoom]);

  // Reactions
  const handleReactionClick = useCallback((e, msgId) => {
    setEmojiAnchor(e.currentTarget);
    setEmojiTargetMessageId(msgId);
  }, []);

  const handleEmojiSelect = useCallback(async (emoji) => {
    if (!emojiTargetMessageId || !currentRoom) return;
    const msg = messages.find(m => m.message_id === emojiTargetMessageId);
    const existing = (msg?.reactions || []).find(r => r.emoji === emoji && r.user_id === user?.user_id);
    if (existing) await removeReaction(currentRoom.room_id, existing.reaction_id);
    else await addReaction(currentRoom.room_id, emojiTargetMessageId, emoji);
    setEmojiAnchor(null);
    setEmojiTargetMessageId(null);
  }, [emojiTargetMessageId, currentRoom, messages, user, addReaction, removeReaction]);

  const handleReactionChipClick = useCallback(async (msgId, emoji, reactions) => {
    if (!currentRoom) return;
    const mine = reactions.find(r => r.emoji === emoji && r.user_id === user?.user_id);
    if (mine) await removeReaction(currentRoom.room_id, mine.reaction_id);
    else await addReaction(currentRoom.room_id, msgId, emoji);
  }, [currentRoom, user, addReaction, removeReaction]);

  // Send / edit handler from input
  const handleSend = useCallback(async ({ content, mentions, replyToMessageId, editingMessageId, file }) => {
    if (!currentRoom) return;
    const peerSt = federationRoomStatus?.federation_peer_status;
    const fedBlocked =
      currentRoom.room_type === 'federated' &&
      federationRoomStatus?.federated &&
      peerSt &&
      peerSt !== 'active';
    if (fedBlocked && !editingMessageId) return;
    if (editingMessageId) {
      await editMessage(currentRoom.room_id, editingMessageId, content);
      setEditingMessage(null);
      return;
    }
    const msg = await sendMessage(currentRoom.room_id, content, 'text', null, mentions, replyToMessageId);
    setReplyTo(null);

    if (file && msg?.message_id) {
      try {
        await messagingService.uploadAttachment(currentRoom.room_id, msg.message_id, file);
        const atts = await messagingService.getMessageAttachments(msg.message_id);
        if (atts?.length) {
          setMessageAttachments(p => ({ ...p, [msg.message_id]: atts }));
          const bMap = {};
          for (const att of atts) {
            if (att.mime_type?.startsWith('image/')) {
              try {
                const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
                const resp = await fetch(`/api/messaging/attachments/${att.attachment_id}/file`, {
                  headers: token ? { Authorization: `Bearer ${token}` } : {},
                });
                if (resp.ok) bMap[att.attachment_id] = URL.createObjectURL(await resp.blob());
              } catch {}
            }
          }
          if (Object.keys(bMap).length) setImageBlobUrls(p => ({ ...p, ...bMap }));
        }
      } catch (err) {
        console.error('Failed to upload attachment:', err);
      }
    }
  }, [currentRoom, sendMessage, editMessage, federationRoomStatus]);

  if (!currentRoom) return null;

  const currentUserId = user?.user_id;
  const otherParticipants = currentRoom.participants?.filter(p => p.user_id !== currentUserId) || [];
  const peerSt = federationRoomStatus?.federation_peer_status;
  const federationPeerInactive =
    currentRoom.room_type === 'federated' &&
    !!federationRoomStatus?.federated &&
    !!peerSt &&
    peerSt !== 'active';
  const federationPeerLabel =
    federationRoomStatus?.federation_peer_display_name ||
    federationRoomStatus?.federation_peer_url ||
    'peer';

  const fedMeta = currentRoom?.federation_metadata;
  const fedPeerId = fedMeta && typeof fedMeta === 'object' ? fedMeta.peer_id : null;
  const fedPresenceBucket = fedPeerId ? federatedPresenceByPeer[fedPeerId] : null;
  let federatedRemotePresenceStatus = 'offline';
  if (fedPresenceBucket && typeof fedPresenceBucket === 'object') {
    const statuses = Object.values(fedPresenceBucket).map((x) => x.status);
    if (statuses.includes('online')) federatedRemotePresenceStatus = 'online';
    else if (statuses.includes('away')) federatedRemotePresenceStatus = 'away';
    else if (statuses.length) federatedRemotePresenceStatus = statuses[0];
  }
  const fedRead = currentRoom?.room_id
    ? federatedReadReceiptByRoom[currentRoom.room_id]
    : null;

  const mergeMessageAttachments = (messageId) => {
    const base = messageAttachments[messageId] || [];
    const extra = federatedAttachmentsByMessage[messageId] || [];
    if (!extra.length) return base;
    const seen = new Set(base.map((a) => a.attachment_id));
    return [...base, ...extra.filter((a) => a.attachment_id && !seen.has(a.attachment_id))];
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2, borderBottom: 1, borderColor: 'divider' }}>
        <IconButton size="small" onClick={() => selectRoom(null)}><ArrowBack /></IconButton>
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            <Typography variant="subtitle1">{currentRoom.display_name || currentRoom.room_name}</Typography>
            {currentRoom.room_type === 'federated' && (
              <Chip size="small" icon={<Hub sx={{ fontSize: '16px !important' }} />} label="Federated" variant="outlined" />
            )}
          </Box>
          {!currentRoom.team_id && otherParticipants.length > 0 && currentRoom.room_type !== 'federated' && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <PresenceIndicator status={presence[otherParticipants[0].user_id]?.status || 'offline'} size="small" />
              <Typography variant="caption" color="text.secondary">
                {presence[otherParticipants[0].user_id]?.status || 'offline'}
              </Typography>
            </Box>
          )}
          {currentRoom.room_type === 'federated' && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              <PresenceIndicator status={federatedRemotePresenceStatus} size="small" />
              <Typography variant="caption" color="text.secondary">
                Remote presence: {federatedRemotePresenceStatus}
              </Typography>
              {fedRead?.last_read_at && (
                <Typography variant="caption" color="text.secondary" sx={{ ml: 0.5 }}>
                  · Last read {fedRead.user_address || 'peer'}: {new Date(fedRead.last_read_at).toLocaleString()}
                </Typography>
              )}
            </Box>
          )}
        </Box>
        <Tooltip title="Search messages">
          <IconButton size="small" onClick={() => setSearchOpen(o => !o)}>
            <Search />
          </IconButton>
        </Tooltip>
      </Box>

      {federationPeerInactive && (
        <Alert severity="warning" sx={{ borderRadius: 0 }}>
          Federation with {federationPeerLabel} is suspended. Messages cannot be delivered until the
          connection is restored.
        </Alert>
      )}

      {/* Search bar */}
      {searchOpen && (
        <Box sx={{ px: 2, py: 1, borderBottom: 1, borderColor: 'divider' }}>
          <TextField
            fullWidth size="small" placeholder="Search messages..."
            value={searchQuery} onChange={e => setSearchQuery(e.target.value)} autoFocus
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={() => { setSearchOpen(false); setSearchQuery(''); setSearchResults([]); }}>
                    <Close fontSize="small" />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          {searchResults.length > 0 && (
            <Box sx={{ maxHeight: 200, overflow: 'auto', mt: 1 }}>
              {searchResults.map(r => (
                <Box key={r.message_id} sx={{ py: 0.5, borderBottom: '1px solid', borderColor: 'divider' }}>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {r.display_name || r.username}
                  </Typography>
                  <Typography variant="body2" noWrap>{r.content}</Typography>
                </Box>
              ))}
            </Box>
          )}
        </Box>
      )}

      {/* Messages */}
      <Box sx={{ flex: 1, overflow: 'auto', p: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {messages.length === 0 ? (
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <Typography color="text.secondary">No messages yet. Start the conversation!</Typography>
          </Box>
        ) : messages.map(message => {
          if (message.message_type === 'team_invitation' || message.metadata?.invitation_id) {
            return (
              <Box key={message.message_id} sx={{ display: 'flex', justifyContent: 'center', my: 1 }}>
                <TeamInvitationMessage message={message} />
              </Box>
            );
          }
          const isOwn = message.sender_id === currentUserId;
          return (
            <Box key={message.message_id} sx={{ display: 'flex', justifyContent: isOwn ? 'flex-end' : 'flex-start', gap: 1 }}>
              {!isOwn && (
                <Avatar sx={{ width: 32, height: 32 }}>
                  {message.display_name?.charAt(0) || message.username?.charAt(0) || '?'}
                </Avatar>
              )}
              <RoomMessageBubble
                message={message}
                isOwn={isOwn}
                currentUserId={currentUserId}
                roomIsFederated={currentRoom.room_type === 'federated'}
                onReactionClick={handleReactionClick}
                onReactionChipClick={handleReactionChipClick}
                onReply={setReplyTo}
                onEdit={setEditingMessage}
                attachmentsList={mergeMessageAttachments(message.message_id)}
                imageBlobUrls={imageBlobUrls}
                openLightbox={openLightbox}
              />
            </Box>
          );
        })}
        <div ref={messagesEndRef} />
      </Box>

      {/* Emoji picker */}
      <Popover
        open={Boolean(emojiAnchor)} anchorEl={emojiAnchor}
        onClose={() => { setEmojiAnchor(null); setEmojiTargetMessageId(null); }}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Box sx={{ display: 'flex', gap: 0.5, p: 1 }}>
          {QUICK_EMOJIS.map(emoji => (
            <IconButton key={emoji} size="small" onClick={() => handleEmojiSelect(emoji)} sx={{ fontSize: '1.2rem', p: 0.5 }}>
              {emoji}
            </IconButton>
          ))}
        </Box>
      </Popover>

      {/* Input */}
      <RoomChatInput
        roomId={currentRoom.room_id}
        onSend={handleSend}
        typingUsers={typingUsers}
        replyTo={replyTo}
        onCancelReply={() => setReplyTo(null)}
        editingMessage={editingMessage}
        onCancelEdit={() => setEditingMessage(null)}
        sendDisabled={federationPeerInactive}
      />
    </Box>
  );
};

export default RoomChat;
