/**
 * Single message in the team timeline with type-specific styling.
 */

import React from 'react';
import { Box, Typography, Chip } from '@mui/material';
import {
  ArrowForward,
  Warning,
  Info,
  Assignment,
  Chat,
  Report,
  Notifications,
} from '@mui/icons-material';

const TYPE_CONFIG = {
  task_assignment: { icon: Assignment, color: 'primary', label: 'Task' },
  status_update: { icon: Info, color: 'info', label: 'Status' },
  request: { icon: Chat, color: 'secondary', label: 'Request' },
  response: { icon: Chat, color: 'success', label: 'Response' },
  delegation: { icon: ArrowForward, color: 'primary', label: 'Delegation' },
  escalation: { icon: Warning, color: 'warning', label: 'Escalation' },
  report: { icon: Report, color: 'default', label: 'Report' },
  system: { icon: Notifications, color: 'default', label: 'System' },
};

export default function TimelineMessage({ message, onExpandThread }) {
  const config = TYPE_CONFIG[message.message_type] || TYPE_CONFIG.report;
  const Icon = config.icon;
  const fromName = message.from_agent_name || message.from_agent_handle || (message.from_agent_id ? 'Agent' : 'System');
  const toName = message.to_agent_id
    ? (message.to_agent_name || message.to_agent_handle || 'Agent')
    : null;
  const time = message.created_at
    ? new Date(message.created_at).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '';
  const borderColor = message.from_agent_color || undefined;

  return (
    <Box
      sx={{
        borderLeft: 3,
        borderColor: borderColor || `${config.color}.main`,
        pl: 1.5,
        py: 1,
        mb: 1,
        borderRadius: 0,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 0.5 }}>
        {borderColor ? (
          <Box sx={{ width: 20, height: 20, borderRadius: '50%', bgcolor: borderColor, flexShrink: 0 }} />
        ) : (
          <Icon sx={{ fontSize: 18 }} color={config.color} />
        )}
        <Chip size="small" label={config.label} variant="outlined" sx={{ fontWeight: 500 }} />
        <Typography variant="caption" color="text.secondary">
          {fromName}
          {toName ? ` → ${toName}` : ''}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {time}
        </Typography>
      </Box>
      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {message.content || '—'}
      </Typography>
      {message.parent_message_id && onExpandThread && (
        <Typography
          variant="caption"
          color="primary"
          sx={{ cursor: 'pointer', mt: 0.5, display: 'block' }}
          onClick={() => onExpandThread(message.parent_message_id)}
        >
          View thread
        </Typography>
      )}
    </Box>
  );
}
