/**
 * Compact team execution indicators for the StatusBar.
 * Shows pulsing chips for teams with an active heartbeat (running).
 */

import React from 'react';
import { Box, Chip, Tooltip, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useTeamExecution } from '../../contexts/TeamExecutionContext';

const MAX_NAME_LEN = 12;

function PulsingDot() {
  return (
    <Box
      component="span"
      sx={{
        display: 'inline-block',
        width: 6,
        height: 6,
        borderRadius: '50%',
        bgcolor: 'success.main',
        animation: 'pulse 1.5s ease-in-out infinite',
        '@keyframes pulse': {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.4 },
        },
      }}
    />
  );
}

export default function LineStatusIndicators() {
  const { teamStatusMap } = useTeamExecution();
  const navigate = useNavigate();

  const teamsToShow = Object.entries(teamStatusMap)
    .filter(([, v]) => v.status === 'running' && v.teamName)
    .map(([id, v]) => ({ id, ...v }));

  if (teamsToShow.length === 0) return null;

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
      {teamsToShow.map(({ id, teamName }) => {
        const label = teamName.length > MAX_NAME_LEN ? `${teamName.slice(0, MAX_NAME_LEN)}…` : teamName;
        return (
          <Tooltip key={id} title={`${teamName} — Heartbeat running`} arrow>
            <Chip
              size="small"
              icon={<PulsingDot />}
              label={<Typography variant="caption">{label}</Typography>}
              onClick={() => navigate(`/agent-factory/line/${id}`)}
              sx={{
                cursor: 'pointer',
                maxHeight: 20,
                '& .MuiChip-icon': { overflow: 'visible' },
              }}
            />
          </Tooltip>
        );
      })}
    </Box>
  );
}
