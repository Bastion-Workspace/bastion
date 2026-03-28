/**
 * Task detail: title, description, status, assigned agent, goal, thread messages.
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import TimelineMessage from './TimelineMessage';

const STATUS_OPTIONS = [
  { value: 'backlog', label: 'Backlog' },
  { value: 'assigned', label: 'Assigned' },
  { value: 'in_progress', label: 'In progress' },
  { value: 'review', label: 'Review' },
  { value: 'done', label: 'Done' },
  { value: 'cancelled', label: 'Cancelled' },
];

export default function TaskDetail({ task, onStatusChange, onClose }) {
  if (!task) {
    return (
      <Box sx={{ py: 2 }}>
        <Typography color="text.secondary">Select a task</Typography>
      </Box>
    );
  }

  const thread = task.thread || [];

  return (
    <Box>
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6">{task.title}</Typography>
          {task.description && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1, whiteSpace: 'pre-wrap' }}>
              {task.description}
            </Typography>
          )}
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
            <Chip size="small" label={task.status || 'backlog'} />
            <Chip size="small" label={`Priority ${task.priority ?? 0}`} variant="outlined" />
            {task.assigned_agent_id && (
              <Chip size="small" label={`Assigned: ${task.assigned_agent_name || task.assigned_agent_id}`} variant="outlined" />
            )}
            {task.due_date && <Chip size="small" label={`Due: ${task.due_date}`} variant="outlined" />}
          </Box>
          {onStatusChange && (
            <FormControl size="small" sx={{ minWidth: 160, mt: 2 }}>
              <InputLabel>Status</InputLabel>
              <Select
                value={task.status || 'backlog'}
                onChange={(e) => onStatusChange(task.id, e.target.value)}
                label="Status"
              >
                {STATUS_OPTIONS.map((o) => (
                  <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {onClose && (
            <Button size="small" onClick={onClose} sx={{ mt: 2, ml: 1 }}>
              Close
            </Button>
          )}
        </CardContent>
      </Card>

      {thread.length > 0 && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Discussion
            </Typography>
            {thread.map((msg) => (
              <TimelineMessage key={msg.id} message={msg} />
            ))}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
