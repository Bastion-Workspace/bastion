/**
 * Agent Factory grid view when no agent is selected.
 * Shows agent cards with status badges, run counts, and last activity.
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Chip,
  Grid,
  CircularProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import { Download } from '@mui/icons-material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

function StatusChip({ isActive }) {
  const label = isActive ? 'Active' : 'Paused';
  const color = isActive ? 'success' : 'default';
  return <Chip size="small" label={label} color={color} sx={{ fontWeight: 500 }} />;
}

function AgentCard({ profile, executions, onExportYaml }) {
  const navigate = useNavigate();
  const handleClick = () => navigate(`/agent-factory/${profile.id}`);
  const handleExport = (e) => {
    e.stopPropagation();
    onExportYaml?.(profile);
  };
  const list = Array.isArray(executions) ? executions : executions?.executions || [];
  const runCount = typeof executions?.total_count === 'number' ? executions.total_count : list.length;
  const lastRun = list[0];
  const lastActivity = lastRun?.started_at
    ? new Date(lastRun.started_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : '—';

  return (
    <Card variant="outlined" sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardActionArea onClick={handleClick} sx={{ flex: 1, display: 'flex', alignItems: 'stretch' }}>
        <CardContent sx={{ flex: 1, minWidth: 0 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography variant="subtitle1" fontWeight={600} noWrap sx={{ flex: 1, minWidth: 0 }}>
              {profile.name || profile.handle || 'Unnamed'}
            </Typography>
            {onExportYaml && (
              <Tooltip title="Export YAML">
                <IconButton size="small" onClick={handleExport} aria-label="Export YAML">
                  <Download fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            <StatusChip isActive={!!profile.is_active} />
          </Box>
          <Typography variant="body2" color="text.secondary" noWrap sx={{ mb: 1 }}>
            {profile.handle ? `@${profile.handle}` : 'Schedule only'}
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            <Typography variant="caption" color="text.secondary">
              Runs: {runCount}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Last: {lastActivity}
            </Typography>
          </Box>
        </CardContent>
      </CardActionArea>
    </Card>
  );
}

export default function AgentGrid() {
  const { data: profiles = [], isLoading } = useQuery(
    'agentFactoryProfiles',
    () => apiService.agentFactory.listProfiles(),
    { retry: false }
  );

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (profiles.length === 0) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="text.secondary">
          No agents yet. Create one to get started.
        </Typography>
      </Box>
    );
  }

  const handleExportYaml = async (p) => {
    if (!p?.id) return;
    try {
      const res = await apiService.agentFactory.exportAgentBundle(p.id);
      const text = await res.text();
      const disp = res.headers.get('Content-Disposition');
      const match = disp && disp.match(/filename="?([^"]+)"?/);
      const filename = match ? match[1] : `${p.handle || p.name || 'agent'}.yaml`;
      const blob = new Blob([text], { type: 'application/x-yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Export failed:', e);
    }
  };

  return (
    <Box sx={{ p: 2, overflow: 'auto' }}>
      <Typography variant="h6" sx={{ mb: 2 }}>
        Your agents ({profiles.length})
      </Typography>
      <Grid container spacing={2}>
        {profiles.map((profile) => (
          <Grid item xs={12} sm={6} md={4} key={profile.id}>
            <AgentCardWithExecutions profile={profile} onExportYaml={handleExportYaml} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
}

function AgentCardWithExecutions({ profile, onExportYaml }) {
  const { data: executions } = useQuery(
    ['agentFactoryExecutions', profile.id],
    () => apiService.agentFactory.listProfileExecutions(profile.id, 5),
    { retry: false, staleTime: 60000 }
  );
  return <AgentCard profile={profile} executions={executions} onExportYaml={onExportYaml} />;
}
