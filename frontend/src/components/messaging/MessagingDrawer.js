/**
 * Collapsible drawer for user-to-user messaging.
 */

import React, { useState } from 'react';
import {
  Drawer,
  IconButton,
  Badge,
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Tooltip,
} from '@mui/material';
import {
  Close,
  Add,
  Fullscreen,
  FullscreenExit,
  ChatBubbleOutline,
  Hub,
} from '@mui/icons-material';
import { useMessaging } from '../../contexts/MessagingContext';
import PresenceIndicator from './PresenceIndicator';
import {
  getEffectiveDisplayStatus,
  summarizeTeamPresence,
} from '../../utils/effectivePresence';
import RoomChat from './RoomChat';
import CreateRoomModal from './CreateRoomModal';
import RoomContextMenu from './RoomContextMenu';
import RenameRoomModal from './RenameRoomModal';
import AddParticipantModal from './AddParticipantModal';

const MessagingDrawer = () => {
  const { user } = useMessaging();
  const {
    isDrawerOpen,
    toggleDrawer,
    closeDrawer,
    totalUnreadCount,
    rooms,
    selectRoom,
    currentRoomId,
    presence,
    presenceTick,
    deleteRoom,
    isMessagingFullScreen,
    toggleFullScreen,
    toggleMute,
  } = useMessaging();

  const [showCreateRoom, setShowCreateRoom] = useState(false);
  const [contextMenu, setContextMenu] = useState(null);
  const [selectedRoom, setSelectedRoom] = useState(null);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showAddParticipantModal, setShowAddParticipantModal] = useState(false);

  const formatLastMessage = (timestamp) => {
    if (!timestamp) return '';
    
    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);
      
      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString();
    } catch {
      return '';
    }
  };

  const getUserPresenceDisplay = (userId) => {
    void presenceTick;
    const raw = presence[userId];
    const status = getEffectiveDisplayStatus(raw);
    const lastSeenAt = raw?.last_seen_at ?? null;
    return { status, lastSeenAt };
  };

  const handleContextMenu = (event, room) => {
    event.preventDefault();
    setSelectedRoom(room);
    setContextMenu({
      mouseX: event.clientX,
      mouseY: event.clientY,
    });
  };

  const handleCloseContextMenu = () => {
    setContextMenu(null);
  };

  const handleRename = () => {
    setShowRenameModal(true);
  };

  const handleDelete = async () => {
    if (selectedRoom && window.confirm(`Are you sure you want to delete the room "${selectedRoom.display_name || selectedRoom.room_name || 'Unnamed Room'}"? This action cannot be undone.`)) {
      try {
        await deleteRoom(selectedRoom.room_id);
      } catch (error) {
        alert('Failed to delete room. Please try again.');
      }
    }
  };

  const handleAddParticipant = () => {
    setShowAddParticipantModal(true);
  };

  const handleToggleMute = async () => {
    if (selectedRoom) {
      const currentMuted = selectedRoom.notification_settings?.muted || false;
      try {
        await toggleMute(selectedRoom.room_id, !currentMuted);
      } catch (error) {
        alert('Failed to update notification settings. Please try again.');
      }
    }
  };

  // Room list component (reusable for both drawer and full-screen modes)
  const RoomList = ({ showHeader = true }) => (
    <>
      {showHeader && (
        <Box
          sx={{
            p: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: 1,
            borderColor: 'divider',
          }}
        >
          <Typography variant="h6">Messages</Typography>
          <Box>
            <Tooltip title="New Conversation">
              <IconButton size="small" onClick={() => setShowCreateRoom(true)}>
                <Add />
              </IconButton>
            </Tooltip>
            {!isMessagingFullScreen && (
              <Tooltip title="Full Screen">
                <IconButton size="small" onClick={toggleFullScreen}>
                  <Fullscreen />
                </IconButton>
              </Tooltip>
            )}
            {isMessagingFullScreen && (
              <Tooltip title="Exit Full Screen">
                <IconButton size="small" onClick={toggleFullScreen}>
                  <FullscreenExit />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title="Close">
              <IconButton size="small" onClick={closeDrawer}>
                <Close />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      )}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List>
          {rooms.length === 0 ? (
            <ListItem>
              <ListItemText
                primary="No conversations yet"
                secondary="Start a new conversation to get started"
              />
            </ListItem>
          ) : (
            rooms.map((room) => {
              void presenceTick;
              const currentUserId = user?.user_id;
              const otherUser = room.participants?.find(p => p.user_id !== currentUserId);
              const displayName = room.display_name || room.room_name || 'Unnamed Room';
              const isTeamRoom = !!room.team_id;
              const isFederated = room.room_type === 'federated';
              const directDisplay = otherUser ? getUserPresenceDisplay(otherUser.user_id) : { status: 'offline', lastSeenAt: null };
              const teamSummary = isTeamRoom
                ? summarizeTeamPresence(room.participants, currentUserId, presence)
                : null;
              const teamAvatarOthers = teamSummary?.others?.slice(0, 4) || [];

              return (
                <ListItemButton
                  key={room.room_id}
                  onClick={() => selectRoom(room.room_id)}
                  onContextMenu={(e) => handleContextMenu(e, room)}
                  selected={currentRoomId === room.room_id}
                >
                  <ListItemAvatar>
                    <Box sx={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                      {isTeamRoom && teamAvatarOthers.length > 0 ? (
                        <Box sx={{ display: 'flex', alignItems: 'center', pl: 0.5 }}>
                          {teamAvatarOthers.map((p, idx) => {
                            const disp = getUserPresenceDisplay(p.user_id);
                            const initial = (p.display_name || p.username || '?').charAt(0).toUpperCase();
                            return (
                              <Box
                                key={p.user_id || idx}
                                sx={{
                                  position: 'relative',
                                  ml: idx > 0 ? -1 : 0,
                                  zIndex: teamAvatarOthers.length - idx,
                                }}
                              >
                                <Avatar sx={{ width: 32, height: 32, fontSize: 14 }}>
                                  {initial}
                                </Avatar>
                                <Box
                                  sx={{
                                    position: 'absolute',
                                    bottom: 0,
                                    right: 0,
                                  }}
                                >
                                  <PresenceIndicator
                                    status={disp.status}
                                    lastSeenAt={disp.lastSeenAt}
                                    size="small"
                                    showTooltip={false}
                                  />
                                </Box>
                              </Box>
                            );
                          })}
                        </Box>
                      ) : isTeamRoom ? (
                        <Avatar>
                          {displayName.charAt(0).toUpperCase()}
                        </Avatar>
                      ) : (
                        <Box sx={{ position: 'relative' }}>
                          <Avatar>
                            {displayName.charAt(0).toUpperCase()}
                          </Avatar>
                          <Box
                            sx={{
                              position: 'absolute',
                              bottom: 0,
                              right: 0,
                            }}
                          >
                            <PresenceIndicator
                              status={directDisplay.status}
                              lastSeenAt={directDisplay.lastSeenAt}
                              size="small"
                              showTooltip={false}
                            />
                          </Box>
                        </Box>
                      )}
                    </Box>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {isFederated && (
                          <Tooltip title="Federated room">
                            <Hub sx={{ fontSize: 18, color: 'text.secondary', flexShrink: 0 }} />
                          </Tooltip>
                        )}
                        <Typography variant="body1" noWrap sx={{ flex: 1 }}>
                          {displayName}
                        </Typography>
                        {room.unread_count > 0 && (
                          <Badge badgeContent={room.unread_count} color="error" />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box component="span" sx={{ display: 'block' }}>
                        <Typography variant="caption" color="text.secondary" component="span" display="block">
                          {formatLastMessage(room.last_message_at)}
                        </Typography>
                        {isTeamRoom && teamSummary && teamSummary.memberCount > 0 && (
                          <Typography variant="caption" color="text.secondary" component="span" display="block">
                            {teamSummary.memberCount} member{teamSummary.memberCount !== 1 ? 's' : ''}
                            {' · '}
                            {teamSummary.activeCount} active
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                </ListItemButton>
              );
            })
          )}
        </List>
      </Box>
    </>
  );

  // Empty state component for when no room is selected in full-screen mode
  const EmptyChatState = () => (
    <Box
      sx={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        p: 4,
      }}
    >
      <ChatBubbleOutline sx={{ fontSize: 64, color: 'text.secondary', opacity: 0.5 }} />
      <Typography variant="h6" color="text.secondary">
        Select a conversation to start chatting
      </Typography>
    </Box>
  );

  return (
    <>
      {/* Full-screen mode */}
      {isDrawerOpen && isMessagingFullScreen && (
        <Box
          sx={{
            position: 'fixed',
            right: 0,
            top: '66px',
            height: { xs: 'calc(var(--appvh, 100vh) - 66px - 32px)', md: 'calc(100dvh - 66px - 32px)' },
            width: '100vw',
            backgroundColor: 'background.paper',
            borderLeft: '1px solid',
            borderColor: 'divider',
            zIndex: 1200,
            boxShadow: 'none',
            display: 'flex',
            flexDirection: 'row',
            overflow: 'hidden',
          }}
        >
          {/* Left panel - Room List */}
          <Box
            sx={{
              width: 350,
              display: 'flex',
              flexDirection: 'column',
              borderRight: 1,
              borderColor: 'divider',
              backgroundColor: 'background.paper',
            }}
          >
            <RoomList showHeader={true} />
          </Box>

          {/* Right panel - Chat Area */}
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {currentRoomId ? (
              <RoomChat />
            ) : (
              <EmptyChatState />
            )}
          </Box>
        </Box>
      )}

      {/* Drawer mode */}
      {isDrawerOpen && !isMessagingFullScreen && (
        <Drawer
          anchor="right"
          open={isDrawerOpen}
          onClose={closeDrawer}
          PaperProps={{
            sx: {
              width: { xs: '100%', sm: 400, md: 480 },
              display: 'flex',
              flexDirection: 'column',
              bottom: '32px',
              height: 'calc(100% - 32px)',
            },
          }}
        >
          {currentRoomId ? (
            <>
              <Box
                sx={{
                  p: 2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  borderBottom: 1,
                  borderColor: 'divider',
                }}
              >
                <Typography variant="h6">Messages</Typography>
                <Box>
                  <Tooltip title="New Conversation">
                    <IconButton size="small" onClick={() => setShowCreateRoom(true)}>
                      <Add />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Full Screen">
                    <IconButton size="small" onClick={toggleFullScreen}>
                      <Fullscreen />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Close">
                    <IconButton size="small" onClick={closeDrawer}>
                      <Close />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              <RoomChat />
            </>
          ) : (
            <RoomList showHeader={true} />
          )}
        </Drawer>
      )}

      {/* Create Room Modal */}
      <CreateRoomModal 
        open={showCreateRoom}
        onClose={() => setShowCreateRoom(false)}
      />

      {/* Room Context Menu */}
      <RoomContextMenu
        anchorPosition={contextMenu}
        open={Boolean(contextMenu)}
        onClose={handleCloseContextMenu}
        onRename={handleRename}
        onDelete={handleDelete}
        onAddParticipant={handleAddParticipant}
        onToggleMute={handleToggleMute}
        isMuted={selectedRoom?.notification_settings?.muted || false}
      />

      {/* Rename Room Modal */}
      <RenameRoomModal
        open={showRenameModal}
        onClose={() => setShowRenameModal(false)}
        room={selectedRoom}
      />

      {/* Add Participant Modal */}
      <AddParticipantModal
        open={showAddParticipantModal}
        onClose={() => setShowAddParticipantModal(false)}
        room={selectedRoom}
      />
    </>
  );
};

export default MessagingDrawer;

