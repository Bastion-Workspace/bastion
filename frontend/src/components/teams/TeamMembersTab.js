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
  Alert,
  Button
} from '@mui/material';
import {
  MoreVert,
  Person,
  Message,
  PersonAdd
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import { useMessaging } from '../../contexts/MessagingContext';
import { useAuth } from '../../contexts/AuthContext';
import TeamInviteDialog from './TeamInviteDialog';
import teamService from '../../services/teams/TeamService';

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
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [teamInvitations, setTeamInvitations] = useState([]);

  const loadTeamInvitations = React.useCallback(async () => {
    if (!teamId) return;
    try {
      const invitations = await teamService.getTeamInvitations(teamId);
      setTeamInvitations(invitations || []);
    } catch (error) {
      console.error('Failed to load team invitations:', error);
    }
  }, [teamId]);

  useEffect(() => {
    if (teamId) {
      loadTeamMembers(teamId);
      loadTeamInvitations();
    }
  }, [teamId, loadTeamMembers, loadTeamInvitations]);

  // Listen for team member changes
  useEffect(() => {
    const handleMemberChange = (event) => {
      if (event.detail?.teamId === teamId) {
        loadTeamMembers(teamId);
        loadTeamInvitations();
      }
    };
    
    window.addEventListener('teamMemberChanged', handleMemberChange);
    return () => window.removeEventListener('teamMemberChanged', handleMemberChange);
  }, [teamId, loadTeamMembers, loadTeamInvitations]);

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
  const pendingInvitations = teamInvitations.filter(inv => inv.status === 'pending');
  const isAdmin = currentTeam?.user_role === 'admin';

  if (isLoading && members.length === 0 && pendingInvitations.length === 0) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      {isAdmin && (
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
          <Button
            variant="contained"
            startIcon={<PersonAdd />}
            onClick={() => setAddDialogOpen(true)}
          >
            Add User
          </Button>
        </Box>
      )}
      
      <Grid container spacing={2}>
        {/* Active Members */}
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

        {/* Pending Invitations */}
        {pendingInvitations.map((invitation) => (
          <Grid item xs={12} sm={6} md={4} key={invitation.invitation_id}>
            <Card sx={{ opacity: 0.8 }}>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  {invitation.invited_avatar_url ? (
                    <Avatar src={invitation.invited_avatar_url} sx={{ width: 48, height: 48, mr: 2 }} />
                  ) : (
                    <Avatar sx={{ width: 48, height: 48, mr: 2, bgcolor: 'warning.main' }}>
                      <Person />
                    </Avatar>
                  )}
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6">
                      {invitation.invited_display_name || invitation.invited_username}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                      <Chip
                        label="Member"
                        size="small"
                        color="default"
                      />
                      <Chip
                        label="Invite Sent"
                        size="small"
                        color="warning"
                      />
                    </Box>
                  </Box>
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

      <TeamInviteDialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        teamId={teamId}
        mode="add"
      />
    </Box>
  );
};

export default TeamMembersTab;

