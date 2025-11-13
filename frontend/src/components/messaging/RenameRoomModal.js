/**
 * Roosevelt's Rename Room Modal
 * Modal for renaming messaging rooms
 * 
 * BULLY! Change those room names like a proper cavalry commander!
 */

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  CircularProgress,
  Alert,
} from '@mui/material';
import { useMessaging } from '../../contexts/MessagingContext';

const RenameRoomModal = ({ open, onClose, room }) => {
  const { updateRoomName } = useMessaging();
  const [roomName, setRoomName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (open && room) {
      setRoomName(room.display_name || room.room_name || '');
      setError(null);
    }
  }, [open, room]);

  const handleRename = async () => {
    if (!roomName.trim()) {
      setError('Room name cannot be empty');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await updateRoomName(room.room_id, roomName);
      onClose();
    } catch (error) {
      console.error('❌ Failed to rename room:', error);
      setError(error.message || 'Failed to rename room. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleRename();
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
      <DialogTitle>Rename Room</DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <TextField
          autoFocus
          fullWidth
          label="Room Name"
          value={roomName}
          onChange={(e) => setRoomName(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading}
          sx={{ mt: 1 }}
          helperText="This name will be visible to all participants"
        />
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          onClick={handleRename}
          variant="contained"
          disabled={loading || !roomName.trim()}
          startIcon={loading && <CircularProgress size={16} />}
        >
          {loading ? 'Renaming...' : 'Rename'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RenameRoomModal;

