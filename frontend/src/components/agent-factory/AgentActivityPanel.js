/**
 * Sidebar showing per-agent activity within the team (message counts, last activity).
 */

import React from 'react';
import { Box, Typography, List, ListItem, ListItemText } from '@mui/material';
import { Person } from '@mui/icons-material';

export default function AgentActivityPanel({ members = [], activityByAgent = {} }) {
  if (!members.length) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle2" color="text.secondary">
          No members
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 1 }}>
      <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
        Activity by agent
      </Typography>
      <List dense disablePadding>
        {members.map((m) => {
          const name = m.agent_name || m.agent_handle || m.agent_profile_id;
          const handle = m.agent_handle ? `@${m.agent_handle}` : '';
          const stats = activityByAgent[m.agent_profile_id] || {};
          const count = stats.count ?? 0;
          const last = stats.last_activity;
          const accentColor = m.color || undefined;
          return (
            <ListItem key={m.id} disablePadding sx={{ py: 0.5 }}>
              {accentColor ? (
                <Box sx={{ width: 20, height: 20, borderRadius: '50%', bgcolor: accentColor, mr: 1, flexShrink: 0 }} />
              ) : (
                <Person sx={{ fontSize: 18, mr: 1 }} color="action" />
              )}
              <ListItemText
                primary={name}
                secondary={
                  handle
                    ? `${count} message(s)${last ? ` · ${new Date(last).toLocaleDateString()}` : ''}`
                    : `${count} message(s)`
                }
                primaryTypographyProps={{ variant: 'body2' }}
                secondaryTypographyProps={{ variant: 'caption' }}
              />
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
}
