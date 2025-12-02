import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Avatar,
  IconButton,
  Typography
} from '@mui/material';
import {
  Close,
  Group,
  CloudUpload
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';

const CreateTeamDialog = ({ open, onClose, onSuccess }) => {
  const { createTeam } = useTeam();
  const [teamName, setTeamName] = useState('');
  const [description, setDescription] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!teamName.trim()) {
      setError('Team name is required');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await createTeam({
        team_name: teamName.trim(),
        description: description.trim() || null,
        avatar_url: avatarUrl.trim() || null
      });
      
      // Reset form
      setTeamName('');
      setDescription('');
      setAvatarUrl('');
      
      onSuccess();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to create team');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setTeamName('');
      setDescription('');
      setAvatarUrl('');
      setError(null);
      onClose();
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Create New Team</Typography>
          <IconButton onClick={handleClose} disabled={isSubmitting}>
            <Close />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <form onSubmit={handleSubmit}>
        <DialogContent>
          {error && (
            <Box sx={{ mb: 2, p: 1, bgcolor: 'error.light', borderRadius: 1 }}>
              <Typography variant="body2" color="error">
                {error}
              </Typography>
            </Box>
          )}
          
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
            {avatarUrl ? (
              <Avatar src={avatarUrl} sx={{ width: 80, height: 80, mb: 2 }} />
            ) : (
              <Avatar sx={{ width: 80, height: 80, mb: 2, bgcolor: 'primary.main' }}>
                <Group sx={{ fontSize: 40 }} />
              </Avatar>
            )}
            <TextField
              label="Avatar URL (optional)"
              value={avatarUrl}
              onChange={(e) => setAvatarUrl(e.target.value)}
              fullWidth
              size="small"
              sx={{ mb: 2 }}
            />
          </Box>
          
          <TextField
            label="Team Name"
            value={teamName}
            onChange={(e) => setTeamName(e.target.value)}
            fullWidth
            required
            margin="normal"
            disabled={isSubmitting}
            autoFocus
          />
          
          <TextField
            label="Description (optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            fullWidth
            multiline
            rows={3}
            margin="normal"
            disabled={isSubmitting}
          />
        </DialogContent>
        
        <DialogActions>
          <Button onClick={handleClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            type="submit"
            variant="contained"
            disabled={isSubmitting || !teamName.trim()}
          >
            {isSubmitting ? 'Creating...' : 'Create Team'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

export default CreateTeamDialog;

