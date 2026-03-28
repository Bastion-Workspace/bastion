/**
 * Filters for team timeline: agent, message type, date range.
 */

import React from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, TextField } from '@mui/material';

const MESSAGE_TYPES = [
  'task_assignment',
  'status_update',
  'request',
  'response',
  'delegation',
  'escalation',
  'report',
  'system',
];

export default function TimelineFilters({
  messageType,
  agentId,
  since,
  agentOptions = [],
  onMessageTypeChange,
  onAgentChange,
  onSinceChange,
}) {
  return (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center' }}>
      <FormControl size="small" sx={{ minWidth: 160 }}>
        <InputLabel>Message type</InputLabel>
        <Select
          value={messageType || ''}
          label="Message type"
          onChange={(e) => onMessageTypeChange(e.target.value || null)}
        >
          <MenuItem value="">All</MenuItem>
          {MESSAGE_TYPES.map((t) => (
            <MenuItem key={t} value={t}>
              {t.replace(/_/g, ' ')}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <FormControl size="small" sx={{ minWidth: 200 }}>
        <InputLabel>Agent</InputLabel>
        <Select
          value={agentId || ''}
          label="Agent"
          onChange={(e) => onAgentChange(e.target.value || null)}
        >
          <MenuItem value="">All agents</MenuItem>
          {agentOptions.map((a) => (
            <MenuItem key={a.id} value={a.id}>
              {a.name || a.handle || a.id}
              {a.handle ? ` @${a.handle}` : ''}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <TextField
        size="small"
        label="Since (ISO date)"
        type="datetime-local"
        value={since || ''}
        onChange={(e) => onSinceChange(e.target.value || null)}
        InputLabelProps={{ shrink: true }}
        sx={{ minWidth: 220 }}
      />
    </Box>
  );
}
