/**
 * Line tasks tab: TaskBoard with create/detail handled by parent drawer.
 */

import React, { useEffect, useMemo } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Typography,
} from '@mui/material';
import { Add } from '@mui/icons-material';
import { useQuery, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import TaskBoard from './TaskBoard';

export default function LineTasksPanel({ lineId, onRequestCreate, onSelectTask }) {
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!lineId) return;
    const token = typeof window !== 'undefined' && window.localStorage.getItem('token');
    if (!token) return;
    const base = window.location.origin.replace(/^http/, 'ws');
    const url = `${base}/api/ws/team-timeline/${lineId}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data?.type === 'team_timeline_update') {
          queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
          queryClient.invalidateQueries(['agentFactoryTeamTimelineRecent', lineId]);
        }
        if (data?.type === 'task_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
        }
        if (data?.type === 'goal_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
        }
      } catch (_) {}
    };
    return () => ws.close();
  }, [lineId, queryClient]);
  const { data: team, isLoading: teamLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: !!lineId }
  );
  const { data: tasks = [] } = useQuery(
    ['agentFactoryTeamTasks', lineId],
    () => apiService.agentFactory.listLineTasks(lineId),
    { enabled: !!lineId }
  );
  const { data: goals = [] } = useQuery(
    ['agentFactoryTeamGoals', lineId],
    () => apiService.agentFactory.getLineGoals(lineId),
    { enabled: !!lineId }
  );
  const flattenGoals = (list) => {
    const out = [];
    for (const g of list || []) {
      out.push(g);
      if (g.children?.length) out.push(...flattenGoals(g.children));
    }
    return out;
  };
  const goalsFlat = flattenGoals(goals);

  const agentNameById = useMemo(() => {
    const m = {};
    for (const mem of team?.members || []) {
      const id = mem.agent_profile_id;
      if (!id) continue;
      m[id] = mem.agent_name || mem.agent_handle || id;
    }
    return m;
  }, [team]);

  const goalTitleById = useMemo(() => {
    const m = {};
    for (const g of goalsFlat) {
      if (g?.id) m[g.id] = g.title || g.id;
    }
    return m;
  }, [goalsFlat]);

  const handleTransition = (taskId, newStatus) => {
    apiService.agentFactory.transitionTask(lineId, taskId, newStatus).then(() => {
      queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
      queryClient.invalidateQueries(['agentFactoryTask', lineId, taskId]);
    });
  };

  const handleDeleteTask = (task) => {
    if (!window.confirm('Delete this task?')) return;
    apiService.agentFactory.deleteTask(lineId, task.id).then(() => {
      queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
      onSelectTask?.(null);
    });
  };

  if (teamLoading || !lineId) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }
  if (!team) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="text.secondary">Team not found.</Typography>
      </Box>
    );
  }

  const agentColorMap = Object.fromEntries(
    (team.members || []).filter((m) => m.color).map((m) => [m.agent_profile_id, m.color])
  );

  return (
    <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexShrink: 0 }}>
        <Typography variant="h6" sx={{ flex: 1 }}>{team.name} – Tasks</Typography>
        <Button variant="contained" startIcon={<Add />} onClick={() => onRequestCreate?.()}>
          Create task
        </Button>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, minWidth: 300 }}>
        <TaskBoard
          tasks={tasks}
          onEditTask={(t) => onSelectTask?.(t.id)}
          onDeleteTask={handleDeleteTask}
          onTransitionTask={handleTransition}
          agentColorMap={agentColorMap}
          agentNameById={agentNameById}
          goalTitleById={goalTitleById}
        />
      </Box>
    </Box>
  );
}
