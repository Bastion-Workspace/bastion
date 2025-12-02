import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  CircularProgress,
  Alert,
  Box,
  Typography
} from '@mui/material';
import {
  Person,
  Close
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import apiService from '../../services/apiService';

const TeamInviteDialog = ({ open, onClose, teamId }) => {
  const { inviteMember } = useTeam();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isInviting, setIsInviting] = useState(false);
  const [error, setError] = useState(null);
  const [selectedUser, setSelectedUser] = useState(null);

  useEffect(() => {
    if (open) {
      setSearchQuery('');
      setSearchResults([]);
      setSelectedUser(null);
      setError(null);
    }
  }, [open]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    setError(null);

    try {
      // Get all users and filter on frontend (admin API doesn't have search yet)
      const response = await apiService.get('/api/admin/users?limit=100');
      const allUsers = response.users || [];
      const query = searchQuery.trim().toLowerCase();
      const filtered = allUsers.filter(user => 
        user.username?.toLowerCase().includes(query) ||
        user.email?.toLowerCase().includes(query) ||
        user.display_name?.toLowerCase().includes(query)
      );
      setSearchResults(filtered);
    } catch (error) {
      console.error('Failed to search users:', error);
      setError('Failed to search users');
    } finally {
      setIsSearching(false);
    }
  };

  const handleInvite = async () => {
    if (!selectedUser) return;

    setIsInviting(true);
    setError(null);

    try {
      await inviteMember(teamId, selectedUser.user_id);
      onClose();
    } catch (error) {
      setError(error.response?.data?.detail || 'Failed to invite user');
    } finally {
      setIsInviting(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Invite Team Member</Typography>
          <Button onClick={onClose} size="small">
            <Close />
          </Button>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        <TextField
          label="Search by username or email"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleSearch();
            }
          }}
          fullWidth
          margin="normal"
          disabled={isSearching || isInviting}
        />
        
        <Button
          variant="outlined"
          onClick={handleSearch}
          disabled={isSearching || !searchQuery.trim()}
          sx={{ mt: 1, mb: 2 }}
        >
          {isSearching ? <CircularProgress size={20} /> : 'Search'}
        </Button>
        
        {searchResults.length > 0 && (
          <List>
            {searchResults.map((user) => (
              <ListItem
                key={user.user_id}
                button
                selected={selectedUser?.user_id === user.user_id}
                onClick={() => setSelectedUser(user)}
              >
                <ListItemAvatar>
                  <Avatar>
                    <Person />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={user.display_name || user.username}
                  secondary={user.email}
                />
              </ListItem>
            ))}
          </List>
        )}
        
        {searchQuery && searchResults.length === 0 && !isSearching && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            No users found
          </Typography>
        )}
      </DialogContent>
      
      <DialogActions>
        <Button onClick={onClose} disabled={isInviting}>
          Cancel
        </Button>
        <Button
          variant="contained"
          onClick={handleInvite}
          disabled={!selectedUser || isInviting}
        >
          {isInviting ? 'Inviting...' : 'Send Invitation'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default TeamInviteDialog;

