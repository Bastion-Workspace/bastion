/**
 * Roosevelt's Messaging Drawer
 * Collapsible drawer for user-to-user messaging
 * 
 * BULLY! The messaging cavalry interface!
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
} from '@mui/icons-material';
import { useMessaging } from '../../contexts/MessagingContext';
import PresenceIndicator from './PresenceIndicator';
import RoomChat from './RoomChat';
import CreateRoomModal from './CreateRoomModal';
import RoomContextMenu from './RoomContextMenu';
import RenameRoomModal from './RenameRoomModal';
import AddParticipantModal from './AddParticipantModal';

const MessagingDrawer = () => {
  const {
    isDrawerOpen,
    toggleDrawer,
    closeDrawer,
    totalUnreadCount,
    rooms,
    selectRoom,
    currentRoomId,
    presence,
    deleteRoom,
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

  const getUserPresence = (userId) => {
    return presence[userId]?.status || 'offline';
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

  return (
    <>
      {/* Messaging drawer */}
      <Drawer
        anchor="right"
        open={isDrawerOpen}
        onClose={closeDrawer}
        PaperProps={{
          sx: {
            width: { xs: '100%', sm: 400, md: 480 },
            display: 'flex',
            flexDirection: 'column',
          },
        }}
      >
        {/* Header */}
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
            <Tooltip title="Close">
              <IconButton size="small" onClick={closeDrawer}>
                <Close />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Content */}
        {currentRoomId ? (
          <RoomChat />
        ) : (
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
                  const otherUser = room.participants?.find(p => p.user_id !== room.created_by);
                  const displayName = room.display_name || room.room_name || 'Unnamed Room';
                  const userStatus = otherUser ? getUserPresence(otherUser.user_id) : 'offline';
                  
                  return (
                    <ListItemButton
                      key={room.room_id}
                      onClick={() => selectRoom(room.room_id)}
                      onContextMenu={(e) => handleContextMenu(e, room)}
                      selected={currentRoomId === room.room_id}
                    >
                      <ListItemAvatar>
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
                              status={userStatus}
                              size="small"
                              showTooltip={false}
                            />
                          </Box>
                        </Box>
                      </ListItemAvatar>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body1" noWrap sx={{ flex: 1 }}>
                              {displayName}
                            </Typography>
                            {room.unread_count > 0 && (
                              <Badge badgeContent={room.unread_count} color="error" />
                            )}
                          </Box>
                        }
                        secondary={formatLastMessage(room.last_message_at)}
                      />
                    </ListItemButton>
                  );
                })
              )}
            </List>
          </Box>
        )}
      </Drawer>

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

