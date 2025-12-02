import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Avatar,
  Typography,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  MoreVert,
  Person,
  Message
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import { useMessaging } from '../../contexts/MessagingContext';
import { useAuth } from '../../contexts/AuthContext';

const TeamMembersTab = ({ teamId }) => {
  const { user } = useAuth();
  const {
    currentTeam,
    teamMembers,
    loadTeamMembers,
    removeMember,
    updateMemberRole,
    isLoading
  } = useTeam();
  const { createRoom, openRoom, loadRooms } = useMessaging();
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedMember, setSelectedMember] = useState(null);

  useEffect(() => {
    if (teamId) {
      loadTeamMembers(teamId);
    }
  }, [teamId, loadTeamMembers]);

  const handleMenuOpen = (event, member) => {
    setAnchorEl(event.currentTarget);
    setSelectedMember(member);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedMember(null);
  };

  const handleMessageMember = async (member) => {
    try {
      await createRoom([member.user_id], null);
      const rooms = await loadRooms();
      const room = rooms.find(r => 
        r.room_type === 'direct' && 
        r.participants?.some(p => p.user_id === member.user_id)
      );
      if (room) {
        openRoom(room.room_id);
      }
    } catch (error) {
      console.error('Failed to create room:', error);
    }
    handleMenuClose();
  };

  const handleRemoveMember = async () => {
    if (selectedMember) {
      try {
        await removeMember(teamId, selectedMember.user_id);
      } catch (error) {
        console.error('Failed to remove member:', error);
      }
    }
    handleMenuClose();
  };

  const handleUpdateRole = async (newRole) => {
    if (selectedMember) {
      try {
        await updateMemberRole(teamId, selectedMember.user_id, newRole);
      } catch (error) {
        console.error('Failed to update role:', error);
      }
    }
    handleMenuClose();
  };

  const members = teamMembers[teamId] || [];
  const isAdmin = currentTeam?.user_role === 'admin';

  if (isLoading && members.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Grid container spacing={2}>
        {members.map((member) => (
          <Grid item xs={12} sm={6} md={4} key={member.user_id}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {member.avatar_url ? (
                    <Avatar src={member.avatar_url} sx={{ width: 48, height: 48, mr: 2 }} />
                  ) : (
                    <Avatar sx={{ width: 48, height: 48, mr: 2, bgcolor: 'primary.main' }}>
                      <Person />
                    </Avatar>
                  )}
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6">
                      {member.display_name || member.username}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                      <Chip
                        label={member.role}
                        size="small"
                        color={member.role === 'admin' ? 'primary' : 'default'}
                      />
                      {member.is_online && (
                        <Box
                          sx={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            bgcolor: 'success.main'
                          }}
                        />
                      )}
                    </Box>
                  </Box>
                  {isAdmin && member.user_id !== user?.user_id && (
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, member)}
                    >
                      <MoreVert />
                    </IconButton>
                  )}
                </Box>
                
                <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                  <IconButton
                    size="small"
                    onClick={() => handleMessageMember(member)}
                    title="Send message"
                  >
                    <Message />
                  </IconButton>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => handleUpdateRole('admin')}>
          Make Admin
        </MenuItem>
        <MenuItem onClick={() => handleUpdateRole('member')}>
          Make Member
        </MenuItem>
        <MenuItem onClick={() => handleUpdateRole('viewer')}>
          Make Viewer
        </MenuItem>
        <MenuItem onClick={handleRemoveMember} sx={{ color: 'error.main' }}>
          Remove from Team
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default TeamMembersTab;

