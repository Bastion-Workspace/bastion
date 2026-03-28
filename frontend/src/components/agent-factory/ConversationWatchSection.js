/**
 * Conversation Watch section: watch AI conversations and/or specific chat rooms; trigger on new message.
 * Persists to profile.watch_config.conversation_watches; backend syncs to agent_conversation_watches on save.
 */

import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Switch,
  CircularProgress,
} from '@mui/material';
import ForumIcon from '@mui/icons-material/Forum';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

export default function ConversationWatchSection({ profile, onChange, compact }) {
  const { data: rooms = [], isLoading: roomsLoading } = useQuery(
    'messaging-rooms',
    async () => {
      const r = await apiService.get('/api/messaging/rooms');
      return r.rooms || [];
    },
    { staleTime: 60 * 1000 }
  );

  if (!profile) return null;

  const watchConfig = profile.watch_config || {};
  const conversationWatches = watchConfig.conversation_watches || [];

  const hasAiConversationsWatch = conversationWatches.some(
    (w) => (w.watch_type || '').toLowerCase() === 'ai_conversations'
  );
  const getRoomWatch = (roomId) =>
    conversationWatches.find(
      (w) => (w.watch_type || '').toLowerCase() === 'chat_room' && String(w.room_id) === String(roomId)
    );

  const setConversationWatches = (next) => {
    onChange({
      ...profile,
      watch_config: { ...watchConfig, conversation_watches: next },
    });
  };

  const setAiConversationsWatch = (enabled) => {
    const rest = conversationWatches.filter(
      (w) => (w.watch_type || '').toLowerCase() !== 'ai_conversations'
    );
    if (enabled) setConversationWatches([...rest, { watch_type: 'ai_conversations' }]);
    else setConversationWatches(rest);
  };

  const setRoomWatch = (roomId, enabled) => {
    const rest = conversationWatches.filter(
      (w) => (w.watch_type || '').toLowerCase() !== 'chat_room' || String(w.room_id) !== String(roomId)
    );
    if (enabled) setConversationWatches([...rest, { watch_type: 'chat_room', room_id: roomId }]);
    else setConversationWatches(rest);
  };

  return (
    <Box>
      {!compact && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Trigger when new messages appear in AI conversations or chat rooms.
        </Typography>
      )}
      <List disablePadding dense>
        <ListItem disablePadding sx={{ py: 0.5 }}>
          <ListItemText
            primary="All AI conversations"
            primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
          />
          <Switch
            size="small"
            checked={hasAiConversationsWatch}
            onChange={(e) => setAiConversationsWatch(e.target.checked)}
          />
        </ListItem>
      </List>

      {rooms.length > 0 && (
        <>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, mb: 0.5, display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <ForumIcon sx={{ fontSize: 14 }} />
            Chat rooms
          </Typography>
          {roomsLoading ? (
            <Box sx={{ py: 1, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={20} />
            </Box>
          ) : (
            <List disablePadding dense>
              {rooms.map((room) => {
                const roomId = room.room_id || room.id;
                const watching = !!getRoomWatch(roomId);
                return (
                  <ListItem key={roomId} disablePadding sx={{ py: 0.25 }}>
                    <ListItemText
                      primary={room.room_name || room.name || `Room ${roomId}`}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                    <Switch
                      size="small"
                      checked={watching}
                      onChange={(e) => setRoomWatch(roomId, e.target.checked)}
                    />
                  </ListItem>
                );
              })}
            </List>
          )}
        </>
      )}
    </Box>
  );
}

export function conversationWatchSummary(profile) {
  const watches = profile?.watch_config?.conversation_watches || [];
  if (!watches.length) return '';
  const parts = [];
  if (watches.some((w) => (w.watch_type || '').toLowerCase() === 'ai_conversations')) parts.push('AI chats');
  const roomCount = watches.filter((w) => (w.watch_type || '').toLowerCase() === 'chat_room').length;
  if (roomCount) parts.push(`${roomCount} room${roomCount !== 1 ? 's' : ''}`);
  return parts.join(', ');
}
