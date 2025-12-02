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
  Alert
} from '@mui/material';
import {
  Add,
  Group,
  People,
  ArrowForward
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
    pendingInvitations
  } = useTeam();
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  useEffect(() => {
    loadUserTeams();
  }, [loadUserTeams]);

  const handleCreateTeam = () => {
    setCreateDialogOpen(true);
  };

  const handleTeamClick = (teamId) => {
    navigate(`/teams/${teamId}`);
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
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary' }}>
                    <People sx={{ fontSize: 16, mr: 0.5 }} />
                    <Typography variant="body2">
                      {team.member_count} {team.member_count === 1 ? 'member' : 'members'}
                    </Typography>
                  </Box>
                </CardContent>
                
                <CardActions>
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

