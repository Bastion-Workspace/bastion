/**
 * Reusable banner showing which agents use a shared resource (playbook or data source).
 * Shown at the top of PlaybookEditor and DataSourceEditor when the resource is in use.
 */

import React from 'react';
import { Alert, Typography, Box, Link } from '@mui/material';
import { useNavigate } from 'react-router-dom';

export default function UsageWarningBanner({ resourceLabel, agents = [] }) {
  const navigate = useNavigate();
  if (!agents || agents.length === 0) return null;

  const label = resourceLabel || 'This resource';
  const count = agents.length;
  const text =
    count === 1
      ? `${label} is used by 1 agent. Changes will affect it.`
      : `${label} is used by ${count} agents. Changes will affect all of them.`;

  return (
    <Alert
      severity="warning"
      variant="filled"
      sx={(theme) => {
        const bg =
          theme.palette.mode === 'dark' ? theme.palette.warning.dark : theme.palette.warning.main;
        const fg = theme.palette.getContrastText(bg);
        return {
          mb: 2,
          bgcolor: bg,
          color: fg,
          '& .MuiAlert-icon': {
            color: fg,
            opacity: theme.palette.mode === 'dark' ? 0.95 : 0.9,
          },
          '& .MuiAlert-message': {
            color: fg,
          },
          '& .MuiTypography-root, & .MuiLink-root': {
            color: 'inherit',
          },
        };
      }}
    >
      <Typography variant="body2" component="span" sx={{ color: 'inherit', fontWeight: 500 }}>
        {text}
      </Typography>
      {agents.length > 0 && (
        <Box component="span" sx={{ ml: 0.5 }}>
          {agents.map((a, i) => (
            <span key={a.id}>
              {i > 0 && ', '}
              <Link
                component="button"
                variant="body2"
                onClick={() => navigate(`/agent-factory/agent/${a.id}`)}
                sx={{
                  cursor: 'pointer',
                  fontWeight: 600,
                  color: 'inherit',
                  textDecorationColor: 'currentcolor',
                  '&:hover': { textDecoration: 'underline' },
                }}
              >
                {a.name || a.handle || `@${a.handle}` || 'Unnamed'}
              </Link>
            </span>
          ))}
        </Box>
      )}
    </Alert>
  );
}
