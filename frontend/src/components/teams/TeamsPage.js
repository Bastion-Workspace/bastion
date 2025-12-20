import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Avatar,
  Chip,
  Fab,
  CircularProgress,
  Alert,
  IconButton
} from '@mui/material';
import {
  Add,
  Group,
  People,
  ArrowForward,
  Check,
  Close,
  NotificationsOff,
  Notifications
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import { useAuth } from '../../contexts/AuthContext';
import CreateTeamDialog from './CreateTeamDialog';

const TeamsPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const {
    teams,
    isLoading,
    error,
    loadUserTeams,
    pendingInvitations,
    acceptInvitation,
    rejectInvitation,
    loadPendingInvitations,
    unreadCounts,
    muteTeam
  } = useTeam();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [processingInvitation, setProcessingInvitation] = useState(null);

  useEffect(() => {
    loadUserTeams();
    loadPendingInvitations();
  }, [loadUserTeams, loadPendingInvitations]);

  const handleCreateTeam = () => {
    setCreateDialogOpen(true);
  };

  const handleTeamClick = (teamId) => {
    navigate(`/teams/${teamId}`);
  };

  const handleAcceptInvitation = async (invitationId) => {
    setProcessingInvitation(invitationId);
    try {
      await acceptInvitation(invitationId);
      await loadUserTeams();
      await loadPendingInvitations();
    } catch (error) {
      console.error('Failed to accept invitation:', error);
    } finally {
      setProcessingInvitation(null);
    }
  };

  const handleRejectInvitation = async (invitationId) => {
    setProcessingInvitation(invitationId);
    try {
      await rejectInvitation(invitationId);
      await loadPendingInvitations();
    } catch (error) {
      console.error('Failed to reject invitation:', error);
    } finally {
      setProcessingInvitation(null);
    }
  };

  if (isLoading && teams.length === 0) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Typography variant="h4" component="h1">
          Teams
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleCreateTeam}
        >
          Create Team
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Pending Invitations Section */}
      {pendingInvitations && pendingInvitations.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
            Pending Invitations
          </Typography>
          <Grid container spacing={3}>
            {pendingInvitations.map((invitation) => (
              <Grid item xs={12} sm={6} md={4} key={invitation.invitation_id}>
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    border: '2px solid',
                    borderColor: 'primary.main',
                    bgcolor: (theme) => 
                      theme.palette.mode === 'dark' 
                        ? 'rgba(25, 118, 210, 0.16)' 
                        : 'primary.light',
                    '&:hover': {
                      boxShadow: 4
                    }
                  }}
                >
                  <CardContent sx={{ flexGrow: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Avatar sx={{ width: 48, height: 48, mr: 2, bgcolor: 'primary.main' }}>
                        <Group />
                      </Avatar>
                      <Box>
                        <Typography variant="h6" component="h2">
                          {invitation.team_name}
                        </Typography>
                        <Chip
                          label="Invitation"
                          size="small"
                          color="primary"
                          sx={{ mt: 0.5 }}
                        />
                      </Box>
                    </Box>
                    
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        mb: 2,
                        color: (theme) => 
                          theme.palette.mode === 'dark' 
                            ? 'rgba(255, 255, 255, 0.7)' 
                            : 'text.secondary'
                      }}
                    >
                      Invited by {invitation.inviter_name}
                    </Typography>
                  </CardContent>
                  
                  <CardActions>
                    <Button
                      size="small"
                      variant="contained"
                      color="success"
                      startIcon={<Check />}
                      onClick={() => handleAcceptInvitation(invitation.invitation_id)}
                      disabled={processingInvitation === invitation.invitation_id}
                      sx={{ flex: 1, mr: 1 }}
                    >
                      Accept
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      color="error"
                      startIcon={<Close />}
                      onClick={() => handleRejectInvitation(invitation.invitation_id)}
                      disabled={processingInvitation === invitation.invitation_id}
                    >
                      Decline
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Teams Section */}
      {pendingInvitations && pendingInvitations.length > 0 && (
        <Typography variant="h5" gutterBottom sx={{ mb: 2, mt: 4 }}>
          Your Teams
        </Typography>
      )}

      {teams.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 6 }}>
          <CardContent>
            <Group sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              You're not part of any teams yet
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
              Create a team to start collaborating with others
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<Add />}
              onClick={handleCreateTeam}
            >
              Create Your First Team
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {teams.map((team) => (
            <Grid item xs={12} sm={6} md={4} key={team.team_id}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  cursor: 'pointer',
                  '&:hover': {
                    boxShadow: 4
                  }
                }}
                onClick={() => handleTeamClick(team.team_id)}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {team.avatar_url ? (
                      <Avatar src={team.avatar_url} sx={{ width: 48, height: 48, mr: 2 }} />
                    ) : (
                      <Avatar sx={{ width: 48, height: 48, mr: 2, bgcolor: 'primary.main' }}>
                        <Group />
                      </Avatar>
                    )}
                    <Box>
                      <Typography variant="h6" component="h2">
                        {team.team_name}
                      </Typography>
                      <Chip
                        label={team.user_role}
                        size="small"
                        color={team.user_role === 'admin' ? 'primary' : 'default'}
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Box>
                  
                  {team.description && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {team.description}
                    </Typography>
                  )}
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                      <People sx={{ fontSize: 16, mr: 0.5 }} />
                      <Typography variant="body2">
                        {team.member_count} {team.member_count === 1 ? 'member' : 'members'}
                      </Typography>
                    </Box>
                    {unreadCounts[team.team_id] > 0 && (
                      <Chip
                        label={unreadCounts[team.team_id]}
                        size="small"
                        color="error"
                        sx={{ ml: 1 }}
                      />
                    )}
                  </Box>
                </CardContent>
                
                <CardActions sx={{ justifyContent: 'space-between' }}>
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      muteTeam(team.team_id, !team.muted);
                    }}
                    title={team.muted ? 'Unmute team' : 'Mute team'}
                  >
                    {team.muted ? <NotificationsOff /> : <Notifications />}
                  </IconButton>
                  <Button
                    size="small"
                    endIcon={<ArrowForward />}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTeamClick(team.team_id);
                    }}
                  >
                    Open Team
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      <CreateTeamDialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        onSuccess={() => {
          setCreateDialogOpen(false);
          loadUserTeams();
        }}
      />

      <Fab
        color="primary"
        aria-label="create team"
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24
        }}
        onClick={handleCreateTeam}
      >
        <Add />
      </Fab>
    </Container>
  );
};

export default TeamsPage;

