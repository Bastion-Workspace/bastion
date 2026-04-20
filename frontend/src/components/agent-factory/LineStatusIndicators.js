/**
 * Compact line execution indicators for the StatusBar.
 * Shows pulsing chips for agent lines with an active heartbeat (running).
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

  const linesToShow = Object.entries(teamStatusMap)
    .filter(([, v]) => v.status === 'running' && v.teamName)
    .map(([id, v]) => ({ id, ...v }));

  if (linesToShow.length === 0) return null;

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
      {linesToShow.map(({ id, teamName: lineDisplayName }) => {
        const label = lineDisplayName.length > MAX_NAME_LEN ? `${lineDisplayName.slice(0, MAX_NAME_LEN)}…` : lineDisplayName;
        return (
          <Tooltip key={id} title={`${lineDisplayName} — Line heartbeat running`} arrow>
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
