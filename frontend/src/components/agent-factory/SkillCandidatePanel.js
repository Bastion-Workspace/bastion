/**
 * Candidate A/B testing panel for skills.
 * Shows candidate version info, weight slider, promote/reject controls.
 */

import React from 'react';
import { Box, Typography, Chip, Slider, Button } from '@mui/material';
import { Science, CheckCircle, Cancel } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import agentFactoryService from '../../services/agentFactoryService';

export default function SkillCandidatePanel({ skillId }) {
  const queryClient = useQueryClient();

  const { data: candidateData } = useQuery(
    ['agentFactorySkillCandidate', skillId],
    () => agentFactoryService.getSkillCandidate(skillId),
    { enabled: !!skillId, staleTime: 30_000, retry: false }
  );

  const promoteMutation = useMutation(
    (candidateId) => agentFactoryService.promoteCandidate(candidateId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactorySkill', skillId]);
        queryClient.invalidateQueries(['agentFactorySkillCandidate', skillId]);
        queryClient.invalidateQueries('agentFactorySkills');
      },
    }
  );
  const rejectMutation = useMutation(
    (candidateId) => agentFactoryService.rejectCandidate(candidateId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactorySkillCandidate', skillId]);
        queryClient.invalidateQueries('agentFactorySkills');
      },
    }
  );
  const weightMutation = useMutation(
    ({ candidateId, weight }) => agentFactoryService.setCandidateWeight(candidateId, weight),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactorySkillCandidate', skillId]);
      },
    }
  );

  if (!candidateData?.has_candidate || !candidateData.candidate) return null;

  const candidate = candidateData.candidate;

  return (
    <Box
      sx={{
        mb: 2,
        p: 1.5,
        borderRadius: 1,
        border: '1px solid',
        borderColor: 'warning.main',
        bgcolor: 'warning.light',
        opacity: 0.95,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Science fontSize="small" color="warning" />
        <Typography variant="subtitle2">
          Candidate v{candidate.version}
        </Typography>
        <Chip
          size="small"
          label={`${candidate.candidate_weight ?? 0}% traffic`}
          color="warning"
          variant="outlined"
        />
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
        <Typography variant="caption" sx={{ minWidth: 55 }}>Weight</Typography>
        <Slider
          size="small"
          min={0}
          max={100}
          step={5}
          value={candidate.candidate_weight ?? 0}
          onChange={(_, v) =>
            weightMutation.mutate({ candidateId: candidate.id, weight: v })
          }
          valueLabelDisplay="auto"
          valueLabelFormat={(v) => `${v}%`}
          sx={{ flex: 1, mx: 1 }}
        />
      </Box>
      <Box sx={{ display: 'flex', gap: 1 }}>
        <Button
          size="small"
          variant="contained"
          color="success"
          startIcon={<CheckCircle />}
          onClick={() => promoteMutation.mutate(candidate.id)}
          disabled={promoteMutation.isLoading}
        >
          Promote
        </Button>
        <Button
          size="small"
          variant="outlined"
          color="error"
          startIcon={<Cancel />}
          onClick={() => rejectMutation.mutate(candidate.id)}
          disabled={rejectMutation.isLoading}
        >
          Reject
        </Button>
      </Box>
      {candidate.procedure && (
        <Box sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary">Candidate procedure preview:</Typography>
          <Box
            component="pre"
            sx={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              p: 1,
              bgcolor: 'background.paper',
              borderRadius: 1,
              fontSize: '0.75rem',
              fontFamily: 'inherit',
              maxHeight: 120,
              overflow: 'auto',
            }}
          >
            {candidate.procedure.slice(0, 500)}
            {candidate.procedure.length > 500 ? '…' : ''}
          </Box>
        </Box>
      )}
    </Box>
  );
}
