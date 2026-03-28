/**
 * Create or edit team: name, description, mission, governance policy, heartbeat config.
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
  { value: 'paused', label: 'Paused' },
  { value: 'archived', label: 'Archived' },
];

export default function TeamEditor({ teamId, initialValues, onSuccess, onCancel }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState(initialValues?.name ?? '');
  const [description, setDescription] = useState(initialValues?.description ?? '');
  const [missionStatement, setMissionStatement] = useState(initialValues?.mission_statement ?? '');
  const [status, setStatus] = useState(initialValues?.status ?? 'active');

  useEffect(() => {
    if (initialValues) {
      setName(initialValues.name ?? '');
      setDescription(initialValues.description ?? '');
      setMissionStatement(initialValues.mission_statement ?? '');
      setStatus(initialValues.status ?? 'active');
    }
  }, [initialValues]);

  const createMutation = useMutation(
    (body) => apiService.agentFactory.createLine(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries('agentFactoryLines');
        onSuccess?.(data);
      },
    }
  );
  const updateMutation = useMutation(
    (body) => apiService.agentFactory.updateLine(teamId, body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries(['agentFactoryTeam', teamId]);
        onSuccess?.(data);
      },
    }
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    const body = {
      name: name.trim(),
      description: description.trim() || null,
      mission_statement: missionStatement.trim() || null,
      status,
    };
    if (teamId) {
      updateMutation.mutate(body);
    } else {
      createMutation.mutate(body);
    }
  };

  const loading = createMutation.isLoading || updateMutation.isLoading;
  const error = createMutation.error || updateMutation.error;

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2 }}>
          {teamId ? 'Edit team' : 'New team'}
        </Typography>
        <Box component="form" onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            margin="normal"
            size="small"
          />
          <TextField
            fullWidth
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            margin="normal"
            size="small"
            multiline
            rows={2}
          />
          <TextField
            fullWidth
            label="Mission statement"
            value={missionStatement}
            onChange={(e) => setMissionStatement(e.target.value)}
            margin="normal"
            size="small"
            multiline
            rows={3}
            placeholder="e.g. Build the #1 AI note-taking app to $1M MRR"
          />
          <FormControl fullWidth margin="normal" size="small">
            <InputLabel>Status</InputLabel>
            <Select
              value={status}
              label="Status"
              onChange={(e) => setStatus(e.target.value)}
            >
              {STATUS_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {error && (
            <Typography color="error" variant="body2" sx={{ mt: 1 }}>
              {error.message || 'Request failed'}
            </Typography>
          )}
          <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
            <Button type="submit" variant="contained" disabled={!name.trim() || loading}>
              {teamId ? 'Save' : 'Create'}
            </Button>
            {onCancel && (
              <Button variant="outlined" onClick={onCancel} disabled={loading}>
                Cancel
              </Button>
            )}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
