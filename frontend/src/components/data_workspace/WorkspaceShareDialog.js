import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Switch,
  FormControlLabel,
  Divider,
  Alert
} from '@mui/material';
import {
  Close as CloseIcon,
  Delete as DeleteIcon,
  Share as ShareIcon,
  Person as PersonIcon,
  Group as GroupIcon,
  Public as PublicIcon
} from '@mui/icons-material';
import dataWorkspaceService from '../../services/dataWorkspaceService';
import apiService from '../../services/apiService';

const WorkspaceShareDialog = ({ open, onClose, workspaceId, workspaceName }) => {
  const [shareType, setShareType] = useState('user'); // 'user', 'team', or 'public'
  const [selectedUserId, setSelectedUserId] = useState('');
  const [selectedTeamId, setSelectedTeamId] = useState('');
  const [permissionLevel, setPermissionLevel] = useState('read');
  const [isPublic, setIsPublic] = useState(false);
  const [expiresAt, setExpiresAt] = useState('');
  const [shares, setShares] = useState([]);
  const [users, setUsers] = useState([]);
  const [teams, setTeams] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userSearch, setUserSearch] = useState('');

  useEffect(() => {
    if (open && workspaceId) {
      loadShares();
      loadUsers();
      loadTeams();
    }
  }, [open, workspaceId]);

  const loadShares = async () => {
    try {
      const data = await dataWorkspaceService.listWorkspaceShares(workspaceId);
      setShares(data);
    } catch (error) {
      console.error('Failed to load shares:', error);
      setError('Failed to load shares');
    }
  };

  const loadUsers = async () => {
    try {
      // Note: This would need a user search endpoint
      // For now, we'll use a placeholder
      setUsers([]);
    } catch (error) {
      console.error('Failed to load users:', error);
    }
  };

  const loadTeams = async () => {
    try {
      const data = await apiService.get('/api/teams');
      setTeams(data || []);
    } catch (error) {
      console.error('Failed to load teams:', error);
    }
  };

  const handleShare = async () => {
    try {
      setLoading(true);
      setError(null);

      const shareData = {
        permission_level: permissionLevel,
        is_public: isPublic || shareType === 'public',
        expires_at: expiresAt || null
      };

      if (shareType === 'user' && selectedUserId) {
        shareData.shared_with_user_id = selectedUserId;
      } else if (shareType === 'team' && selectedTeamId) {
        shareData.shared_with_team_id = selectedTeamId;
      } else if (shareType === 'public') {
        shareData.is_public = true;
      } else {
        setError('Please select a user or team, or enable public sharing');
        setLoading(false);
        return;
      }

      await dataWorkspaceService.shareWorkspace(workspaceId, shareData);
      await loadShares();
      
      // Reset form
      setSelectedUserId('');
      setSelectedTeamId('');
      setPermissionLevel('read');
      setIsPublic(false);
      setExpiresAt('');
      setShareType('user');
    } catch (error) {
      console.error('Failed to share workspace:', error);
      setError(error.response?.data?.detail || 'Failed to share workspace');
    } finally {
      setLoading(false);
    }
  };

  const handleRevokeShare = async (shareId) => {
    if (!window.confirm('Are you sure you want to revoke this share?')) {
      return;
    }

    try {
      await dataWorkspaceService.revokeShare(workspaceId, shareId);
      await loadShares();
    } catch (error) {
      console.error('Failed to revoke share:', error);
      setError('Failed to revoke share');
    }
  };

  const getShareTypeLabel = (share) => {
    if (share.is_public) return 'Public';
    if (share.shared_with_team_id) return 'Team';
    if (share.shared_with_user_id) return 'User';
    return 'Unknown';
  };

  const getPermissionLabel = (level) => {
    const labels = {
      read: 'Read',
      write: 'Write',
      admin: 'Admin'
    };
    return labels[level] || level;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShareIcon />
          <Typography variant="h6">Share Workspace: {workspaceName}</Typography>
        </Box>
        <IconButton size="small" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Share Type Selection */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Share With
          </Typography>
          <ToggleButtonGroup
            value={shareType}
            exclusive
            onChange={(e, newValue) => {
              if (newValue !== null) {
                setShareType(newValue);
                setIsPublic(newValue === 'public');
              }
            }}
            fullWidth
            size="small"
          >
            <ToggleButton value="user">
              <PersonIcon sx={{ mr: 1 }} />
              User
            </ToggleButton>
            <ToggleButton value="team">
              <GroupIcon sx={{ mr: 1 }} />
              Team
            </ToggleButton>
            <ToggleButton value="public">
              <PublicIcon sx={{ mr: 1 }} />
              Public
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* User/Team Selection */}
        {shareType === 'user' && (
          <Box sx={{ mb: 2 }}>
            <TextField
              fullWidth
              label="User ID or Email"
              value={selectedUserId}
              onChange={(e) => setSelectedUserId(e.target.value)}
              placeholder="Enter user ID or email"
              helperText="Note: User search functionality needs to be implemented"
            />
          </Box>
        )}

        {shareType === 'team' && (
          <Box sx={{ mb: 2 }}>
            <FormControl fullWidth>
              <InputLabel>Select Team</InputLabel>
              <Select
                value={selectedTeamId}
                onChange={(e) => setSelectedTeamId(e.target.value)}
                label="Select Team"
              >
                {teams.map((team) => (
                  <MenuItem key={team.team_id} value={team.team_id}>
                    {team.team_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        )}

        {shareType === 'public' && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Public shares allow anyone with the link to access this workspace.
          </Alert>
        )}

        {/* Permission Level */}
        <Box sx={{ mb: 2 }}>
          <FormControl fullWidth>
            <InputLabel>Permission Level</InputLabel>
            <Select
              value={permissionLevel}
              onChange={(e) => setPermissionLevel(e.target.value)}
              label="Permission Level"
            >
              <MenuItem value="read">Read - View only</MenuItem>
              <MenuItem value="write">Write - Can edit</MenuItem>
              <MenuItem value="admin">Admin - Full control</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Expiration (Optional) */}
        <Box sx={{ mb: 2 }}>
          <TextField
            fullWidth
            type="datetime-local"
            label="Expires At (Optional)"
            value={expiresAt}
            onChange={(e) => setExpiresAt(e.target.value)}
            InputLabelProps={{ shrink: true }}
          />
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Existing Shares */}
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Existing Shares
          </Typography>
          {shares.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No shares yet
            </Typography>
          ) : (
            <List>
              {shares.map((share) => (
                <ListItem key={share.share_id}>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip
                          label={getShareTypeLabel(share)}
                          size="small"
                          color={share.is_public ? 'primary' : 'default'}
                        />
                        <Chip
                          label={getPermissionLabel(share.permission_level)}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          Expires: {formatDate(share.expires_at)}
                        </Typography>
                        <Typography variant="caption" display="block">
                          Created: {formatDate(share.created_at)}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => handleRevokeShare(share.share_id)}
                      size="small"
                    >
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
        <Button
          onClick={handleShare}
          variant="contained"
          disabled={loading || (shareType === 'user' && !selectedUserId) || (shareType === 'team' && !selectedTeamId)}
        >
          {loading ? 'Sharing...' : 'Share'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default WorkspaceShareDialog;

