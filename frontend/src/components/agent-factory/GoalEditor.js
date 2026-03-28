/**
 * Create or edit a goal: title, description, parent, assigned agent, status, priority, progress, due date.
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

const STATUS_OPTIONS = [
  { value: 'active', label: 'Active' },
  { value: 'completed', label: 'Completed' },
  { value: 'blocked', label: 'Blocked' },
  { value: 'cancelled', label: 'Cancelled' },
];

export default function GoalEditor({
  lineId,
  teamId,
  goalId,
  initialValues,
  parentOptions = [],
  agentOptions = [],
  onSuccess,
  onCancel,
}) {
  const resolvedLineId = lineId ?? teamId;
  const queryClient = useQueryClient();
  const [title, setTitle] = useState(initialValues?.title ?? '');
  const [description, setDescription] = useState(initialValues?.description ?? '');
  const [parentGoalId, setParentGoalId] = useState(initialValues?.parent_goal_id ?? '');
  const [assignedAgentId, setAssignedAgentId] = useState(initialValues?.assigned_agent_id ?? '');
  const [status, setStatus] = useState(initialValues?.status ?? 'active');
  const [priority, setPriority] = useState(initialValues?.priority ?? 0);
  const [progressPct, setProgressPct] = useState(initialValues?.progress_pct ?? 0);
  const [dueDate, setDueDate] = useState(initialValues?.due_date ?? '');

  useEffect(() => {
    if (initialValues) {
      setTitle(initialValues.title ?? '');
      setDescription(initialValues.description ?? '');
      setParentGoalId(initialValues.parent_goal_id ?? '');
      setAssignedAgentId(initialValues.assigned_agent_id ?? '');
      setStatus(initialValues.status ?? 'active');
      setPriority(initialValues.priority ?? 0);
      setProgressPct(initialValues.progress_pct ?? 0);
      setDueDate(initialValues.due_date ?? '');
    }
  }, [initialValues]);

  const createMutation = useMutation(
    (body) => apiService.agentFactory.createGoal(resolvedLineId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeamGoals', resolvedLineId]);
        onSuccess?.();
      },
    }
  );
  const updateMutation = useMutation(
    (body) => apiService.agentFactory.updateGoal(resolvedLineId, goalId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeamGoals', resolvedLineId]);
        onSuccess?.();
      },
    }
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!resolvedLineId) return;
    const body = {
      title: title.trim(),
      description: description.trim() || null,
      parent_goal_id: parentGoalId || null,
      assigned_agent_id: assignedAgentId || null,
      status,
      priority: Number(priority) || 0,
      progress_pct: Math.min(100, Math.max(0, Number(progressPct) || 0)),
      due_date: dueDate.trim() || null,
    };
    if (goalId) {
      updateMutation.mutate(body);
    } else {
      createMutation.mutate(body);
    }
  };

  const loading = createMutation.isLoading || updateMutation.isLoading;
  const error = createMutation.error || updateMutation.error;

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2 }}>
          {goalId ? 'Edit goal' : 'New goal'}
        </Typography>
        <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <TextField
            label="Title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
            fullWidth
            size="small"
          />
          <TextField
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            multiline
            rows={2}
            fullWidth
            size="small"
          />
          {parentOptions.length > 0 && (
            <FormControl size="small" fullWidth>
              <InputLabel>Parent goal</InputLabel>
              <Select
                value={parentGoalId}
                onChange={(e) => setParentGoalId(e.target.value)}
                label="Parent goal"
              >
                <MenuItem value="">None (top-level)</MenuItem>
                {parentOptions.map((opt) => (
                  <MenuItem key={opt.id} value={opt.id}>
                    {opt.title}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          {agentOptions.length > 0 && (
            <FormControl size="small" fullWidth>
              <InputLabel>Assigned agent</InputLabel>
              <Select
                value={assignedAgentId}
                onChange={(e) => setAssignedAgentId(e.target.value)}
                label="Assigned agent"
              >
                <MenuItem value="">Unassigned</MenuItem>
                {agentOptions.map((a) => (
                  <MenuItem key={a.id} value={a.id}>
                    {a.agent_name || a.agent_handle || a.id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}
          <FormControl size="small" fullWidth>
            <InputLabel>Status</InputLabel>
            <Select value={status} onChange={(e) => setStatus(e.target.value)} label="Status">
              {STATUS_OPTIONS.map((o) => (
                <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            label="Priority (higher = more important)"
            type="number"
            value={priority}
            onChange={(e) => setPriority(e.target.value)}
            size="small"
            inputProps={{ min: 0 }}
          />
          <TextField
            label="Progress %"
            type="number"
            value={progressPct}
            onChange={(e) => setProgressPct(e.target.value)}
            size="small"
            inputProps={{ min: 0, max: 100 }}
          />
          <TextField
            label="Due date"
            type="date"
            value={dueDate}
            onChange={(e) => setDueDate(e.target.value)}
            size="small"
            InputLabelProps={{ shrink: true }}
          />
          {error && (
            <Typography color="error" variant="caption">
              {error.message || 'Request failed'}
            </Typography>
          )}
          <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
            {onCancel && <Button onClick={onCancel}>Cancel</Button>}
            <Button type="submit" variant="contained" disabled={loading}>
              {goalId ? 'Update' : 'Create'}
            </Button>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
