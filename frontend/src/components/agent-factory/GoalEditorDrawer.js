/**
 * Goals: tree and create/edit forms in a resizable right drawer.
 */

import React, { useState, useEffect } from 'react';
import { Box, Button, CircularProgress, Typography, IconButton } from '@mui/material';
import { Close, Add } from '@mui/icons-material';
import { useQuery, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import GoalTreeView from './GoalTreeView';
import GoalEditor from './GoalEditor';
import ResizableDrawer from './ResizableDrawer';

function flattenGoals(nodes, out = []) {
  if (!nodes) return out;
  for (const n of nodes) {
    out.push({ id: n.id, title: n.title });
    flattenGoals(n.children, out);
  }
  return out;
}

export default function GoalEditorDrawer({ open, onClose, lineId }) {
  const queryClient = useQueryClient();
  const [editingGoal, setEditingGoal] = useState(null);
  const [createOpen, setCreateOpen] = useState(false);

  useEffect(() => {
    if (!open || !lineId) return;
    const token = typeof window !== 'undefined' && window.localStorage.getItem('token');
    if (!token) return;
    const base = window.location.origin.replace(/^http/, 'ws');
    const url = `${base}/api/ws/team-timeline/${lineId}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data?.type === 'goal_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
        }
      } catch (_) {}
    };
    return () => ws.close();
  }, [open, lineId, queryClient]);

  const { data: team, isLoading: teamLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: open && !!lineId }
  );
  const { data: goalsTree = [] } = useQuery(
    ['agentFactoryTeamGoals', lineId],
    () => apiService.agentFactory.getLineGoals(lineId),
    { enabled: open && !!lineId }
  );
  const parentOptions = flattenGoals(goalsTree);
  const members = team?.members ?? [];
  const agentOptions = members.map((m) => ({ id: m.agent_profile_id, agent_name: m.agent_name, agent_handle: m.agent_handle }));

  const handleEditGoal = (goal) => setEditingGoal(goal);
  const handleDeleteGoal = (goal) => {
    if (!window.confirm('Delete this goal?')) return;
    apiService.agentFactory.deleteGoal(lineId, goal.id).then(() => {
      queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
      setEditingGoal(null);
    });
  };

  if (!open) return null;

  return (
    <ResizableDrawer
      open={open}
      onClose={onClose}
      storageKey="agent-factory-line-goals-drawer-width"
      defaultWidth={440}
      minWidth={320}
      maxWidth={900}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0, gap: 1 }}>
        <Typography variant="h6" sx={{ flex: 1, minWidth: 0 }} noWrap>
          Goals{team?.name ? ` — ${team.name}` : ''}
        </Typography>
        <Button startIcon={<Add />} variant="contained" size="small" onClick={() => setCreateOpen(true)} disabled={!team}>
          New goal
        </Button>
        <IconButton onClick={onClose} aria-label="Close" size="small">
          <Close />
        </IconButton>
      </Box>
      <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', p: 2 }}>
        {(teamLoading || !lineId) && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress />
          </Box>
        )}
        {!teamLoading && lineId && !team && (
          <Typography color="text.secondary">Line not found.</Typography>
        )}
        {!teamLoading && team && (
          <>
            {createOpen && (
              <Box sx={{ mb: 2 }}>
                <GoalEditor
                  lineId={lineId}
                  parentOptions={parentOptions}
                  agentOptions={agentOptions}
                  onSuccess={() => {
                    queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
                    setCreateOpen(false);
                  }}
                  onCancel={() => setCreateOpen(false)}
                />
              </Box>
            )}
            {editingGoal && (
              <Box sx={{ mb: 2 }}>
                <GoalEditor
                  lineId={lineId}
                  goalId={editingGoal.id}
                  initialValues={editingGoal}
                  parentOptions={parentOptions.filter((p) => p.id !== editingGoal.id)}
                  agentOptions={agentOptions}
                  onSuccess={() => {
                    queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
                    setEditingGoal(null);
                  }}
                  onCancel={() => setEditingGoal(null)}
                />
              </Box>
            )}
            <GoalTreeView tree={goalsTree} onEditGoal={handleEditGoal} onDeleteGoal={handleDeleteGoal} />
          </>
        )}
      </Box>
    </ResizableDrawer>
  );
}
