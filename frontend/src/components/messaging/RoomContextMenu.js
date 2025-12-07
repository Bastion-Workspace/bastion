/**
 * Roosevelt's Room Context Menu
 * Right-click context menu for room management
 * 
 * BULLY! Manage your messaging cavalry with precision!
 */

import React from 'react';
import {
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
} from '@mui/material';
import {
  Edit,
  Delete,
  PersonAdd,
  NotificationsOff,
  Notifications,
} from '@mui/icons-material';

const RoomContextMenu = ({ 
  anchorPosition, 
  open, 
  onClose, 
  onRename, 
  onDelete, 
  onAddParticipant,
  onToggleMute,
  isMuted
}) => {
  const handleRename = () => {
    onRename();
    onClose();
  };

  const handleDelete = () => {
    onDelete();
    onClose();
  };

  const handleAddParticipant = () => {
    onAddParticipant();
    onClose();
  };

  const handleToggleMute = () => {
    onToggleMute();
    onClose();
  };

  return (
    <Menu
      open={open}
      onClose={onClose}
      anchorReference="anchorPosition"
      anchorPosition={
        anchorPosition
          ? { top: anchorPosition.mouseY, left: anchorPosition.mouseX }
          : undefined
      }
      PaperProps={{
        sx: {
          minWidth: 200,
        },
      }}
    >
      <MenuItem onClick={handleRename}>
        <ListItemIcon>
          <Edit fontSize="small" />
        </ListItemIcon>
        <ListItemText>Rename Room</ListItemText>
      </MenuItem>

      <MenuItem onClick={handleAddParticipant}>
        <ListItemIcon>
          <PersonAdd fontSize="small" />
        </ListItemIcon>
        <ListItemText>Add Participant</ListItemText>
      </MenuItem>

      <Divider />

      <MenuItem onClick={handleToggleMute}>
        <ListItemIcon>
          {isMuted ? (
            <Notifications fontSize="small" />
          ) : (
            <NotificationsOff fontSize="small" />
          )}
        </ListItemIcon>
        <ListItemText>
          {isMuted ? 'Unmute Notifications' : 'Mute Notifications'}
        </ListItemText>
      </MenuItem>

      <Divider />

      <MenuItem onClick={handleDelete}>
        <ListItemIcon>
          <Delete fontSize="small" color="error" />
        </ListItemIcon>
        <ListItemText sx={{ color: 'error.main' }}>
          Delete Room
        </ListItemText>
      </MenuItem>
    </Menu>
  );
};

export default RoomContextMenu;

