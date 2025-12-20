import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Avatar,
  Chip
} from '@mui/material';
import {
  Group
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import { useAuth } from '../../contexts/AuthContext';

const TeamInvitationMessage = ({ message }) => {
  const { user } = useAuth();
  const { acceptInvitation, rejectInvitation } = useTeam();
  const [isProcessing, setIsProcessing] = useState(false);

  const metadata = message.metadata || {};
  const invitationId = metadata.invitation_id;
  const teamId = metadata.team_id;
  const teamName = metadata.team_name || 'Team';
  const inviterName = metadata.inviter_name || 'Someone';
  const invitationStatus = metadata.invitation_status;

  // Check if current user is the invited user (not the inviter)
  const isInvitedUser = metadata.invited_user_id === user?.user_id;

  const handleAccept = async () => {
    if (!invitationId) return;
    
    setIsProcessing(true);
    try {
      await acceptInvitation(invitationId);
    } catch (error) {
      console.error('Failed to accept invitation:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReject = async () => {
    if (!invitationId) return;
    
    setIsProcessing(true);
    try {
      await rejectInvitation(invitationId);
    } catch (error) {
      console.error('Failed to reject invitation:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  if (invitationStatus === 'accepted') {
    return (
      <Paper 
        sx={{ 
          p: 2, 
          bgcolor: (theme) => 
            theme.palette.mode === 'dark' 
              ? 'rgba(46, 125, 50, 0.2)' 
              : 'success.light', 
          maxWidth: 400 
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <Group />
          <Typography variant="subtitle2">
            Team Invitation Accepted
          </Typography>
        </Box>
        <Typography variant="body2">
          You joined {teamName}
        </Typography>
      </Paper>
    );
  }

  if (invitationStatus === 'rejected') {
    return (
      <Paper 
        sx={{ 
          p: 2, 
          bgcolor: (theme) => 
            theme.palette.mode === 'dark' 
              ? 'rgba(158, 158, 158, 0.15)' 
              : 'grey.200', 
          maxWidth: 400 
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Team invitation declined
        </Typography>
      </Paper>
    );
  }

  if (!isInvitedUser) {
    return (
      <Paper 
        sx={{ 
          p: 2, 
          bgcolor: (theme) => 
            theme.palette.mode === 'dark' 
              ? 'rgba(2, 136, 209, 0.2)' 
              : 'info.light', 
          maxWidth: 400 
        }}
      >
        <Typography variant="body2">
          You invited {inviterName} to join {teamName}
        </Typography>
        <Chip label="Pending" size="small" sx={{ mt: 1 }} />
      </Paper>
    );
  }

  return (
    <Paper 
      sx={{ 
        p: 2, 
        bgcolor: (theme) => 
          theme.palette.mode === 'dark' 
            ? 'rgba(25, 118, 210, 0.16)' 
            : 'primary.light', 
        maxWidth: 400 
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Avatar sx={{ bgcolor: 'primary.main' }}>
          <Group />
        </Avatar>
        <Box>
          <Typography variant="subtitle2">
            Team Invitation
          </Typography>
          <Typography 
            variant="caption" 
            sx={{
              color: (theme) => 
                theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.7)' 
                  : 'text.secondary'
            }}
          >
            from {inviterName}
          </Typography>
        </Box>
      </Box>
      
      <Typography variant="body1" sx={{ mb: 2 }}>
        You've been invited to join <strong>{teamName}</strong>
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button
          variant="contained"
          size="small"
          onClick={handleAccept}
          disabled={isProcessing}
        >
          Accept
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={handleReject}
          disabled={isProcessing}
        >
          Decline
        </Button>
      </Box>
    </Paper>
  );
};

export default TeamInvitationMessage;

