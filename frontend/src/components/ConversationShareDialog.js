import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
  Divider,
  Autocomplete,
  Avatar,
  Stack
} from '@mui/material';
import {
  Close,
  Delete,
  Person,
  Edit,
  Comment,
  Visibility,
  Share
} from '@mui/icons-material';
import conversationService from '../services/conversation/ConversationService';
import apiService from '../services/apiService';

const ConversationShareDialog = ({ 
  open, 
  onClose, 
  conversationId,
  conversationTitle 
}) => {
  const [loading, setLoading] = useState(false);
  const [shares, setShares] = useState([]);
  const [participants, setParticipants] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [shareType, setShareType] = useState('read');
  const [expiresAt, setExpiresAt] = useState('');
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [users, setUsers] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');

  // Load shares and participants when dialog opens
  useEffect(() => {
    if (open && conversationId) {
      loadShares();
      loadParticipants();
      loadUsers();
    }
  }, [open, conversationId]);

  const loadShares = async () => {
    try {
      const response = await conversationService.getConversationShares(conversationId);
      if (response && response.shares) {
        setShares(response.shares);
      }
    } catch (err) {
      console.error('Failed to load shares:', err);
    }
  };

  const loadParticipants = async () => {
    try {
      const response = await conversationService.getConversationParticipants(conversationId);
      if (response && response.participants) {
        setParticipants(response.participants);
      }
    } catch (err) {
      console.error('Failed to load participants:', err);
    }
  };

  const loadUsers = async () => {
    try {
      // Try to load users for autocomplete
      // Note: This endpoint may not exist - fallback to manual entry
      try {
        const response = await apiService.get('/api/users');
        if (response && response.users) {
          setUsers(response.users);
        } else if (response && Array.isArray(response)) {
          setUsers(response);
        }
      } catch (endpointError) {
        // Endpoint doesn't exist or failed - allow manual user ID entry
        console.log('Users endpoint not available, allowing manual entry');
        setUsers([]);
      }
    } catch (err) {
      console.error('Failed to load users:', err);
      // Fallback: allow manual user ID entry
      setUsers([]);
    }
  };

  const handleShare = async () => {
    if (!selectedUser) {
      setError('Please select a user to share with');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const userId = typeof selectedUser === 'string' ? selectedUser : selectedUser.user_id;
      const expiresAtDate = expiresAt ? new Date(expiresAt).toISOString() : null;

      await conversationService.shareConversation(
        conversationId,
        userId,
        shareType,
        expiresAtDate
      );

      setSuccess('Conversation shared successfully');
      setSelectedUser(null);
      setShareType('read');
      setExpiresAt('');
      await loadShares();
      await loadParticipants();
      
      // Clear success message after 2 seconds
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to share conversation');
    } finally {
      setLoading(false);
    }
  };

  const handleUnshare = async (shareId) => {
    if (!window.confirm('Are you sure you want to remove this share?')) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await conversationService.unshareConversation(conversationId, shareId);
      await loadShares();
      await loadParticipants();
      setSuccess('Share removed successfully');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to remove share');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdatePermissions = async (shareId, newShareType) => {
    setLoading(true);
    setError(null);

    try {
      await conversationService.updateSharePermissions(conversationId, shareId, newShareType);
      await loadShares();
      await loadParticipants();
      setSuccess('Permissions updated successfully');
      setTimeout(() => setSuccess(null), 2000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to update permissions');
    } finally {
      setLoading(false);
    }
  };

  const getPermissionIcon = (type) => {
    switch (type) {
      case 'edit':
        return <Edit fontSize="small" />;
      case 'comment':
        return <Comment fontSize="small" />;
      case 'read':
        return <Visibility fontSize="small" />;
      default:
        return <Visibility fontSize="small" />;
    }
  };

  const getPermissionLabel = (type) => {
    switch (type) {
      case 'edit':
        return 'Can Edit';
      case 'comment':
        return 'Can Comment';
      case 'read':
        return 'Read Only';
      default:
        return 'Read Only';
    }
  };

  const filteredUsers = users.filter(user => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      user.username?.toLowerCase().includes(query) ||
      user.email?.toLowerCase().includes(query) ||
      user.user_id?.toLowerCase().includes(query)
    );
  });

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <Share />
            <Typography variant="h6">Share Conversation</Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <Close />
          </IconButton>
        </Box>
        {conversationTitle && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {conversationTitle}
          </Typography>
        )}
      </DialogTitle>

      <DialogContent>
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

        {/* Share with new user */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Share with User
          </Typography>
          
          <Stack spacing={2} sx={{ mt: 2 }}>
            <Autocomplete
              freeSolo
              options={filteredUsers}
              getOptionLabel={(option) => {
                if (typeof option === 'string') return option;
                return option.username || option.email || option.user_id || '';
              }}
              value={selectedUser}
              onChange={(event, newValue) => setSelectedUser(newValue)}
              inputValue={searchQuery}
              onInputChange={(event, newInputValue) => setSearchQuery(newInputValue)}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="User ID, username, or email"
                  placeholder="Enter user ID, username, or email"
                  fullWidth
                  helperText={users.length === 0 ? "Enter user ID manually" : "Search or enter user ID"}
                />
              )}
              renderOption={(props, option) => {
                // Handle string options (manual entry)
                if (typeof option === 'string') {
                  return (
                    <Box component="li" {...props}>
                      <Typography variant="body2">{option}</Typography>
                    </Box>
                  );
                }
                // Handle user object options
                return (
                  <Box component="li" {...props}>
                    <Avatar sx={{ mr: 1, width: 32, height: 32 }}>
                      {option.username?.[0]?.toUpperCase() || option.email?.[0]?.toUpperCase() || 'U'}
                    </Avatar>
                    <Box>
                      <Typography variant="body2">
                        {option.username || option.email || option.user_id}
                      </Typography>
                      {option.email && option.username && (
                        <Typography variant="caption" color="text.secondary">
                          {option.email}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                );
              }}
            />

            <FormControl fullWidth>
              <InputLabel>Permission Level</InputLabel>
              <Select
                value={shareType}
                onChange={(e) => setShareType(e.target.value)}
                label="Permission Level"
              >
                <MenuItem value="read">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Visibility fontSize="small" />
                    <Box>
                      <Typography variant="body2">Read Only</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Can view conversation
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
                <MenuItem value="comment">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Comment fontSize="small" />
                    <Box>
                      <Typography variant="body2">Can Comment</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Can view and add messages
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
                <MenuItem value="edit">
                  <Box display="flex" alignItems="center" gap={1}>
                    <Edit fontSize="small" />
                    <Box>
                      <Typography variant="body2">Can Edit</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Can view, add messages, and modify settings
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Expiration Date (Optional)"
              type="datetime-local"
              value={expiresAt}
              onChange={(e) => setExpiresAt(e.target.value)}
              fullWidth
              InputLabelProps={{
                shrink: true,
              }}
            />

            <Button
              variant="contained"
              onClick={handleShare}
              disabled={loading || !selectedUser}
              startIcon={loading ? <CircularProgress size={20} /> : <Share />}
            >
              Share Conversation
            </Button>
          </Stack>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Current participants */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Participants ({participants.length})
          </Typography>
          <List>
            {participants.map((participant) => (
              <ListItem key={participant.user_id}>
                <Avatar sx={{ mr: 2, width: 32, height: 32 }}>
                  {participant.username?.[0]?.toUpperCase() || participant.email?.[0]?.toUpperCase() || 'U'}
                </Avatar>
                <ListItemText
                  primary={participant.username || participant.email || participant.user_id}
                  secondary={participant.email && participant.username ? participant.email : null}
                />
                <Chip
                  icon={getPermissionIcon(participant.share_type)}
                  label={getPermissionLabel(participant.share_type)}
                  size="small"
                  color={participant.is_owner ? 'primary' : 'default'}
                />
              </ListItem>
            ))}
          </List>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Current shares */}
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Active Shares ({shares.length})
          </Typography>
          {shares.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              No active shares
            </Typography>
          ) : (
            <List>
              {shares.map((share) => (
                <ListItem key={share.share_id}>
                  <Avatar sx={{ mr: 2, width: 32, height: 32 }}>
                    {share.username?.[0]?.toUpperCase() || share.email?.[0]?.toUpperCase() || 'U'}
                  </Avatar>
                  <ListItemText
                    primary={share.username || share.email || share.shared_with_user_id}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {share.email && share.username ? share.email : null}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Shared {new Date(share.created_at).toLocaleDateString()}
                          {share.expires_at && ` • Expires ${new Date(share.expires_at).toLocaleDateString()}`}
                        </Typography>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <Box display="flex" alignItems="center" gap={1}>
                      <FormControl size="small" sx={{ minWidth: 120 }}>
                        <Select
                          value={share.share_type}
                          onChange={(e) => handleUpdatePermissions(share.share_id, e.target.value)}
                          disabled={loading}
                        >
                          <MenuItem value="read">Read Only</MenuItem>
                          <MenuItem value="comment">Can Comment</MenuItem>
                          <MenuItem value="edit">Can Edit</MenuItem>
                        </Select>
                      </FormControl>
                      <IconButton
                        edge="end"
                        onClick={() => handleUnshare(share.share_id)}
                        disabled={loading}
                        size="small"
                      >
                        <Delete />
                      </IconButton>
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConversationShareDialog;

