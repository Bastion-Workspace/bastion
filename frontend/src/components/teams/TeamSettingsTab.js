import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  TextField,
  Button,
  Typography,
  Divider,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip
} from '@mui/material';
import {
  Delete,
  PersonAdd
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import { useAuth } from '../../contexts/AuthContext';
import TeamInviteDialog from './TeamInviteDialog';

const TeamSettingsTab = ({ teamId }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const {
    currentTeam,
    updateTeam,
    deleteTeam
  } = useTeam();
  const [teamName, setTeamName] = useState(currentTeam?.team_name || '');
  const [description, setDescription] = useState(currentTeam?.description || '');
  const [avatarUrl, setAvatarUrl] = useState(currentTeam?.avatar_url || '');
  const [isSaving, setIsSaving] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  React.useEffect(() => {
    if (currentTeam) {
      setTeamName(currentTeam.team_name || '');
      setDescription(currentTeam.description || '');
      setAvatarUrl(currentTeam.avatar_url || '');
    }
  }, [currentTeam]);

  const isAdmin = currentTeam?.user_role === 'admin';

  const handleSave = async () => {
    if (!teamName.trim()) {
      setError('Team name is required');
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccess(null);

    try {
      await updateTeam(teamId, {
        team_name: teamName.trim(),
        description: description.trim() || null,
        avatar_url: avatarUrl.trim() || null
      });
      setSuccess('Team updated successfully');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update team');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    try {
      await deleteTeam(teamId);
      setDeleteDialogOpen(false);
      // Navigate to teams page after successful deletion
      navigate('/teams');
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to delete team');
      setDeleteDialogOpen(false);
    }
  };

  if (!isAdmin) {
    return (
      <Alert severity="info">
        Only team admins can access settings.
      </Alert>
    );
  }

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess(null)}>
          {success}
        </Alert>
      )}

      {/* Team Information */}
      <Typography variant="h6" gutterBottom>
        Team Information
      </Typography>
      
      <Box sx={{ mb: 4 }}>
        <TextField
          label="Team Name"
          value={teamName}
          onChange={(e) => setTeamName(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        
        <TextField
          label="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          fullWidth
          multiline
          rows={3}
          margin="normal"
        />
        
        <TextField
          label="Avatar URL"
          value={avatarUrl}
          onChange={(e) => setAvatarUrl(e.target.value)}
          fullWidth
          margin="normal"
        />
        
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={isSaving || !teamName.trim()}
          sx={{ mt: 2 }}
        >
          {isSaving ? 'Saving...' : 'Save Changes'}
        </Button>
      </Box>

      <Divider sx={{ my: 4 }} />

      {/* Invite Members */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">
            Team Members
          </Typography>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="outlined"
              startIcon={<PersonAdd />}
              onClick={() => setAddDialogOpen(true)}
            >
              Add User
            </Button>
            <Button
              variant="outlined"
              startIcon={<PersonAdd />}
              onClick={() => setInviteDialogOpen(true)}
            >
              Invite User
            </Button>
          </Box>
        </Box>
      </Box>

      <Divider sx={{ my: 4 }} />

      {/* Danger Zone */}
      <Box>
        <Typography variant="h6" color="error" gutterBottom>
          Danger Zone
        </Typography>
        <Alert severity="warning" sx={{ mb: 2 }}>
          Deleting a team will permanently remove all team data, including posts, documents, and member associations.
        </Alert>
        <Button
          variant="contained"
          color="error"
          startIcon={<Delete />}
          onClick={() => setDeleteDialogOpen(true)}
        >
          Delete Team
        </Button>
      </Box>

      <TeamInviteDialog
        open={inviteDialogOpen}
        onClose={() => setInviteDialogOpen(false)}
        teamId={teamId}
        mode="invite"
      />

      <TeamInviteDialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        teamId={teamId}
        mode="add"
      />

      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Team</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete this team? This action cannot be undone.
            All team data, posts, documents, and member associations will be permanently deleted.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TeamSettingsTab;

