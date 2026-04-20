/**
 * Banner shown in SkillEditor when the current skill has a pending promotion/demotion recommendation.
 */

import React from 'react';
import { Box, Typography, Button, Chip, Alert } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import agentFactoryService from '../../services/agentFactoryService';

export default function SkillRecommendationBanner({ skillId }) {
  const queryClient = useQueryClient();

  const { data: recs = [] } = useQuery(
    ['agentFactorySkillRecommendations', 'pending'],
    () => agentFactoryService.getSkillRecommendations('pending', 100),
    { staleTime: 60_000, retry: false }
  );

  const myRec = recs.find?.((r) => r.skill_id === skillId);

  const applyMutation = useMutation(
    (recId) => agentFactoryService.applyRecommendation(recId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactorySkillRecommendations']);
        queryClient.invalidateQueries(['agentFactorySkill', skillId]);
        queryClient.invalidateQueries('agentFactorySkills');
      },
    }
  );

  const dismissMutation = useMutation(
    (recId) => agentFactoryService.dismissRecommendation(recId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactorySkillRecommendations']);
      },
    }
  );

  if (!myRec) return null;

  const isPromote = myRec.action === 'promote';

  return (
    <Alert
      severity={isPromote ? 'success' : 'warning'}
      icon={isPromote ? <TrendingUp /> : <TrendingDown />}
      sx={{ mb: 2 }}
      action={
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
          <Button
            size="small"
            variant="contained"
            color={isPromote ? 'success' : 'warning'}
            onClick={() => applyMutation.mutate(myRec.id)}
            disabled={applyMutation.isLoading}
          >
            {isPromote ? 'Promote to Core' : 'Demote from Core'}
          </Button>
          <Button
            size="small"
            onClick={() => dismissMutation.mutate(myRec.id)}
            disabled={dismissMutation.isLoading}
          >
            Dismiss
          </Button>
        </Box>
      }
    >
      <Typography variant="body2" sx={{ fontWeight: 500 }}>
        <Chip
          size="small"
          label={isPromote ? 'Promote' : 'Demote'}
          color={isPromote ? 'success' : 'warning'}
          sx={{ mr: 1 }}
        />
        {myRec.reason}
      </Typography>
    </Alert>
  );
}
