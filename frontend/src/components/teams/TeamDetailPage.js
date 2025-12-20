import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Tabs,
  Tab,
  IconButton,
  Avatar,
  Chip,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  ArrowBack,
  Group,
  Settings
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import TeamFeedTab from './TeamFeedTab';
import TeamMembersTab from './TeamMembersTab';
import TeamSettingsTab from './TeamSettingsTab';

const TeamDetailPage = () => {
  const { teamId } = useParams();
  const navigate = useNavigate();
  const {
    currentTeam,
    selectTeam,
    isLoading,
    error
  } = useTeam();
  const [activeTab, setActiveTab] = useState(0);
  const [hasAttemptedLoad, setHasAttemptedLoad] = useState(false);

  useEffect(() => {
    if (teamId) {
      setHasAttemptedLoad(true);
      selectTeam(teamId);
    } else {
      setHasAttemptedLoad(false);
    }
  }, [teamId, selectTeam]);

  // Navigate away if team was deleted (currentTeam becomes null while we have a teamId)
  // Only navigate if we've actually tried to load and it failed (not just initial load)
  useEffect(() => {
    if (teamId && hasAttemptedLoad && !isLoading && !currentTeam && error) {
      // Only navigate if there's an actual error after attempting to load
      const timer = setTimeout(() => {
        if (!currentTeam && !isLoading && error) {
          navigate('/teams');
        }
      }, 500); // Give a bit more time for error to resolve
      return () => clearTimeout(timer);
    }
  }, [teamId, currentTeam, isLoading, error, navigate, hasAttemptedLoad]);

  if (isLoading && !currentTeam) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  if (error && !currentTeam) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  if (!currentTeam) {
    return null;
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 1, mb: 4 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <IconButton onClick={() => navigate('/teams')} sx={{ mr: 2 }}>
          <ArrowBack />
        </IconButton>
        
        {currentTeam.avatar_url ? (
          <Avatar src={currentTeam.avatar_url} sx={{ width: 56, height: 56, mr: 2 }} />
        ) : (
          <Avatar sx={{ width: 56, height: 56, mr: 2, bgcolor: 'primary.main' }}>
            <Group />
          </Avatar>
        )}
        
        <Box sx={{ flexGrow: 1 }}>
          <Typography variant="h4" component="h1">
            {currentTeam.team_name}
          </Typography>
          {currentTeam.description && (
            <Typography variant="body2" color="text.secondary">
              {currentTeam.description}
            </Typography>
          )}
        </Box>
        
        <Chip
          label={currentTeam.user_role}
          color={currentTeam.user_role === 'admin' ? 'primary' : 'default'}
          sx={{ ml: 2 }}
        />
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
          <Tab label="Feed" />
          <Tab label="Members" />
          <Tab label="Settings" icon={<Settings />} iconPosition="end" />
        </Tabs>
      </Box>

      {/* Tab Content */}
      {activeTab === 0 && <TeamFeedTab teamId={teamId} />}
      {activeTab === 1 && <TeamMembersTab teamId={teamId} />}
      {activeTab === 2 && <TeamSettingsTab teamId={teamId} />}
    </Container>
  );
};

export default TeamDetailPage;

