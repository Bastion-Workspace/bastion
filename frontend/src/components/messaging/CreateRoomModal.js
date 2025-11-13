/**
 * Roosevelt's Create Room Modal
 * Modal for creating new messaging rooms with other users
 * 
 * BULLY! Start new messaging cavalry charges!
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Autocomplete,
  Chip,
  CircularProgress,
  Alert,
  Box,
  Typography,
} from '@mui/material';
import apiService from '../../services/apiService';
import { useMessaging } from '../../contexts/MessagingContext';

const CreateRoomModal = ({ open, onClose }) => {
  const { createRoom } = useMessaging();
  const [loading, setLoading] = useState(false);
  const [users, setUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [roomName, setRoomName] = useState('');
  const [error, setError] = useState(null);

  // Load available users when modal opens
  useEffect(() => {
    if (open) {
      loadUsers();
    } else {
      // Reset form when modal closes
      setSelectedUsers([]);
      setRoomName('');
      setError(null);
    }
  }, [open]);

  const loadUsers = async () => {
    setLoadingUsers(true);
    setError(null);
    
    try {
      const response = await apiService.get('/api/messaging/users');
      // ApiServiceBase returns JSON directly, not wrapped in .data
      setUsers(response.users || []);
    } catch (error) {
      console.error('❌ Failed to load users:', error);
      setError('Failed to load users. Please try again.');
    } finally {
      setLoadingUsers(false);
    }
  };

  const handleCreateRoom = async () => {
    if (selectedUsers.length === 0) {
      setError('Please select at least one user to chat with');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const participantIds = selectedUsers.map(user => user.user_id);
      
      // For 1:1 chats, don't set a custom name unless user specified one
      const finalRoomName = (selectedUsers.length === 1 && !roomName) 
        ? null 
        : roomName || null;

      await createRoom(participantIds, finalRoomName);
      
      // Close modal on success
      onClose();
    } catch (error) {
      console.error('❌ Failed to create room:', error);
      setError(error.message || 'Failed to create room. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleCreateRoom();
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Start New Conversation</DialogTitle>
      
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          {/* User selection */}
          <Autocomplete
            multiple
            options={users}
            loading={loadingUsers}
            getOptionLabel={(option) => option.display_name || option.username || 'Unknown'}
            value={selectedUsers}
            onChange={(event, newValue) => setSelectedUsers(newValue)}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Select Users"
                placeholder="Search users..."
                InputProps={{
                  ...params.InputProps,
                  endAdornment: (
                    <>
                      {loadingUsers ? <CircularProgress size={20} /> : null}
                      {params.InputProps.endAdornment}
                    </>
                  ),
                }}
              />
            )}
            renderTags={(value, getTagProps) =>
              value.map((option, index) => (
                <Chip
                  label={option.display_name || option.username}
                  {...getTagProps({ index })}
                  size="small"
                />
              ))
            }
            disabled={loadingUsers || loading}
          />

          {/* Room name (optional for group chats) */}
          {selectedUsers.length > 1 && (
            <TextField
              label="Room Name (optional)"
              value={roomName}
              onChange={(e) => setRoomName(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Give your group chat a name"
              disabled={loading}
              helperText="For group chats, you can set a custom name"
            />
          )}

          {selectedUsers.length === 1 && roomName && (
            <Typography variant="caption" color="text.secondary">
              💡 Custom names for 1:1 chats will be visible to both participants
            </Typography>
          )}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleCreateRoom}
          variant="contained"
          disabled={loading || selectedUsers.length === 0}
          startIcon={loading && <CircularProgress size={16} />}
        >
          {loading ? 'Creating...' : 'Start Chat'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default CreateRoomModal;

