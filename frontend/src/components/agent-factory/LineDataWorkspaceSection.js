/**
 * Agent line Data Workspace binding (additive with profile; per-workspace R/O or R/W).
 */

import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  Button,
  TextField,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
} from '@mui/material';
import StorageIcon from '@mui/icons-material/Storage';
import DeleteIcon from '@mui/icons-material/Delete';
import dataWorkspaceService from '../../services/dataWorkspaceService';

const defaultConfig = () => ({
  workspaces: [],
  auto_inject_schema: false,
  context_instructions: '',
});

function normalizeIncoming(raw) {
  if (!raw || typeof raw !== 'object') return defaultConfig();
  const ws = Array.isArray(raw.workspaces) ? raw.workspaces : [];
  return {
    workspaces: ws
      .map((w) => ({
        workspace_id: String(w.workspace_id || '').trim(),
        access: w.access === 'read_write' ? 'read_write' : 'read',
      }))
      .filter((w) => w.workspace_id),
    auto_inject_schema: !!raw.auto_inject_schema,
    context_instructions: typeof raw.context_instructions === 'string' ? raw.context_instructions : '',
  };
}

const LineDataWorkspaceSection = forwardRef(function LineDataWorkspaceSection({ team }, ref) {
  const [config, setConfig] = useState(defaultConfig);
  const [addId, setAddId] = useState('');

  useImperativeHandle(ref, () => ({
    getConfig: () => config,
  }), [config]);

  const { data: workspaces = [] } = useQuery(
    ['dataWorkspaces', 'line'],
    () => dataWorkspaceService.listWorkspaces(),
    { staleTime: 60000 }
  );

  useEffect(() => {
    if (team?.data_workspace_config) {
      setConfig(normalizeIncoming(team.data_workspace_config));
    } else {
      setConfig(defaultConfig());
    }
  }, [team]);

  const addWorkspace = () => {
    if (!addId) return;
    setConfig((c) => {
      if (c.workspaces.some((w) => w.workspace_id === addId)) return c;
      return {
        ...c,
        workspaces: [...c.workspaces, { workspace_id: addId, access: 'read' }],
      };
    });
    setAddId('');
  };

  const removeWs = (id) => {
    setConfig((c) => ({
      ...c,
      workspaces: c.workspaces.filter((w) => w.workspace_id !== id),
    }));
  };

  const setAccess = (id, access) => {
    setConfig((c) => ({
      ...c,
      workspaces: c.workspaces.map((w) =>
        w.workspace_id === id ? { ...w, access } : w
      ),
    }));
  };

  const availableToAdd = workspaces.filter(
    (w) => !config.workspaces.some((x) => x.workspace_id === w.workspace_id)
  );

  return (
    <Box>
      <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
        Workspaces for this line (merged with each member profile). Read-only allows SQL SELECT only; read/write allows changes where permitted.
      </Typography>

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 2 }}>
        <FormControl size="small" sx={{ minWidth: 260 }}>
          <InputLabel id="line-dw-add" shrink>
            Workspace
          </InputLabel>
          <Select
            labelId="line-dw-add"
            label="Workspace"
            value={addId}
            onChange={(e) => setAddId(e.target.value)}
            displayEmpty
            renderValue={(selected) => {
              if (!selected) {
                return (
                  <Box component="span" sx={{ color: 'text.secondary' }}>
                    Choose a workspace…
                  </Box>
                );
              }
              const w = workspaces.find((x) => x.workspace_id === selected);
              return w?.name || selected;
            }}
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            {availableToAdd.map((w) => (
              <MenuItem key={w.workspace_id} value={w.workspace_id}>
                {w.name || w.workspace_id}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Button size="small" variant="outlined" onClick={addWorkspace} disabled={!addId}>
          Add
        </Button>
      </Box>

      {config.workspaces.map((w) => {
        const meta = workspaces.find((x) => x.workspace_id === w.workspace_id);
        const label = meta?.name || w.workspace_id;
        return (
          <Box
            key={w.workspace_id}
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              py: 0.5,
              flexWrap: 'wrap',
            }}
          >
            <StorageIcon fontSize="small" color="action" />
            <Typography variant="body2" sx={{ flex: 1, minWidth: 120 }}>
              {label}
            </Typography>
            <ToggleButtonGroup
              size="small"
              value={w.access === 'read_write' ? 'rw' : 'r'}
              exclusive
              onChange={(_, v) => v && setAccess(w.workspace_id, v === 'rw' ? 'read_write' : 'read')}
            >
              <ToggleButton value="r">Read</ToggleButton>
              <ToggleButton value="rw">R/W</ToggleButton>
            </ToggleButtonGroup>
            <IconButton size="small" aria-label="Remove" onClick={() => removeWs(w.workspace_id)}>
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Box>
        );
      })}

      <FormControlLabel
        control={
          <Switch
            checked={config.auto_inject_schema}
            onChange={(e) => setConfig((c) => ({ ...c, auto_inject_schema: e.target.checked }))}
            color="primary"
          />
        }
        label="Auto-inject schema (with profile workspaces when enabled)"
        sx={{ display: 'block', mb: 1 }}
      />
      <TextField
        fullWidth
        multiline
        minRows={2}
        label="Context instructions (optional)"
        value={config.context_instructions}
        onChange={(e) => setConfig((c) => ({ ...c, context_instructions: e.target.value }))}
        placeholder="How to interpret this line's workspace data."
        size="small"
        sx={{ mb: 1 }}
      />
    </Box>
  );
});

export default LineDataWorkspaceSection;
