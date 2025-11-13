/**
 * Roosevelt's Add Participant Modal
 * Modal for adding users to existing rooms
 * 
 * BULLY! Expand your messaging cavalry!
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Autocomplete,
  TextField,
  CircularProgress,
  Alert,
  FormControlLabel,
  Checkbox,
  Typography,
  Box,
} from '@mui/material';
import apiService from '../../services/apiService';
import { useMessaging } from '../../contexts/MessagingContext';

const AddParticipantModal = ({ open, onClose, room }) => {
  const { addParticipantToRoom } = useMessaging();
  const [users, setUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [shareHistory, setShareHistory] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load available users when modal opens
  useEffect(() => {
    if (open && room) {
      loadUsers();
    } else {
      setSelectedUser(null);
      setShareHistory(false);
      setError(null);
    }
  }, [open, room]);

  const loadUsers = async () => {
    setLoadingUsers(true);
    setError(null);
    
    try {
      const response = await apiService.get('/api/messaging/users');
      const allUsers = response.users || [];
      
      // Filter out users already in the room
      const participantIds = room.participants?.map(p => p.user_id) || [];
      const availableUsers = allUsers.filter(user => !participantIds.includes(user.user_id));
      
      setUsers(availableUsers);
    } catch (error) {
      console.error('❌ Failed to load users:', error);
      setError('Failed to load users. Please try again.');
    } finally {
      setLoadingUsers(false);
    }
  };

  const handleAddParticipant = async () => {
    if (!selectedUser) {
      setError('Please select a user to add');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await addParticipantToRoom(room.room_id, selectedUser.user_id, shareHistory);
      onClose();
    } catch (error) {
      console.error('❌ Failed to add participant:', error);
      setError(error.message || 'Failed to add participant. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!room) return null;

  return (
    <Dialog 
      open={open} 
      onClose={onClose}
      maxWidth="sm"
      fullWidth
    >
      <DialogTitle>Add Participant to Room</DialogTitle>
      
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <Typography variant="body2" color="text.secondary">
            Room: <strong>{room.display_name || room.room_name || 'Unnamed Room'}</strong>
          </Typography>

          {/* User selection */}
          <Autocomplete
            options={users}
            loading={loadingUsers}
            getOptionLabel={(option) => option.display_name || option.username || 'Unknown'}
            value={selectedUser}
            onChange={(event, newValue) => setSelectedUser(newValue)}
            renderInput={(params) => (
              <TextField
                {...params}
                label="Select User"
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
            disabled={loadingUsers || loading}
          />

          {/* Share history option */}
          <FormControlLabel
            control={
              <Checkbox
                checked={shareHistory}
                onChange={(e) => setShareHistory(e.target.checked)}
                disabled={loading}
              />
            }
            label={
              <Box>
                <Typography variant="body2">
                  Share message history
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {shareHistory 
                    ? 'New participant will see all previous messages in this room'
                    : 'New participant will only see messages from when they joined'}
                </Typography>
              </Box>
            }
          />
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleAddParticipant}
          variant="contained"
          disabled={loading || !selectedUser}
          startIcon={loading && <CircularProgress size={16} />}
        >
          {loading ? 'Adding...' : 'Add Participant'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default AddParticipantModal;

