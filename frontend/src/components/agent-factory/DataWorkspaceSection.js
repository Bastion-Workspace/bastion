/**
 * Data Workspace section: bind this agent to one or more Data Workspaces.
 * When bound, schema can be auto-injected and context instructions guide the LLM.
 */

import React from 'react';
import { useQuery } from 'react-query';
import {
  Card,
  CardContent,
  Typography,
  TextField,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Switch,
  Chip,
} from '@mui/material';
import StorageIcon from '@mui/icons-material/Storage';
import dataWorkspaceService from '../../services/dataWorkspaceService';

export default function DataWorkspaceSection({ profile, onChange, readOnly = false }) {
  const { data: workspaces = [] } = useQuery(
    'dataWorkspaces',
    () => dataWorkspaceService.listWorkspaces(),
    { staleTime: 60000 }
  );

  if (!profile) return null;

  const config = profile.data_workspace_config || {};
  const workspaceIds = Array.isArray(config.workspace_ids) ? config.workspace_ids : [];
  const autoInject = !!config.auto_inject_schema;
  const contextInstructions = config.context_instructions || '';

  const handleWorkspaceIdsChange = (e) => {
    const value = e.target.value;
    const next = typeof value === 'string' ? (value ? value.split(',') : []) : value;
    onChange({
      ...profile,
      data_workspace_config: { ...config, workspace_ids: next },
    });
  };

  const handleAutoInjectChange = (e) => {
    onChange({
      ...profile,
      data_workspace_config: { ...config, auto_inject_schema: e.target.checked },
    });
  };

  const handleContextInstructionsChange = (e) => {
    onChange({
      ...profile,
      data_workspace_config: { ...config, context_instructions: e.target.value || '' },
    });
  };

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent sx={{ fontSize: '0.875rem' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <StorageIcon fontSize="small" color="action" />
          <Typography variant="h6" sx={{ fontSize: '1rem' }}>
            Data Workspace
          </Typography>
          {workspaceIds.length > 0 && (
            <Chip label={`${workspaceIds.length} workspace(s)`} size="small" variant="outlined" />
          )}
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Bind this agent to Data Workspaces so it can query your tables. Schema and context instructions are injected into the agent when enabled.
        </Typography>
        <FormControl fullWidth size="small" sx={{ mb: 2 }} disabled={readOnly}>
          <InputLabel>Workspaces</InputLabel>
          <Select
            multiple
            value={workspaceIds}
            onChange={handleWorkspaceIdsChange}
            label="Workspaces"
            renderValue={(selected) => (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {selected.map((id) => {
                  const ws = workspaces.find((w) => w.workspace_id === id);
                  return <Chip key={id} size="small" label={ws?.name || id} />;
                })}
              </Box>
            )}
          >
            {workspaces.map((ws) => (
              <MenuItem key={ws.workspace_id} value={ws.workspace_id}>
                {ws.name || ws.workspace_id}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControlLabel
          control={
            <Switch
              checked={autoInject}
              onChange={handleAutoInjectChange}
              disabled={readOnly}
              color="primary"
            />
          }
          label="Auto-inject schema (add workspace schema to agent context)"
          sx={{ mb: 1.5 }}
        />
        <TextField
          fullWidth
          multiline
          minRows={3}
          label="Context instructions (optional)"
          value={contextInstructions}
          onChange={handleContextInstructionsChange}
          disabled={readOnly}
          placeholder="e.g. Negative amounts are debits. When I say 'spending' exclude transfers between my own accounts."
          helperText="Plain-language rules so the agent interprets your data correctly."
        />
      </CardContent>
    </Card>
  );
}
