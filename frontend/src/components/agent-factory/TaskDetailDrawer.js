/**
 * Create task or view task detail in a resizable right drawer.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  DialogActions,
  IconButton,
  CircularProgress,
} from '@mui/material';
import { Close } from '@mui/icons-material';
import { useQuery, useQueryClient, useMutation } from 'react-query';
import apiService from '../../services/apiService';
import TaskDetail from './TaskDetail';
import ResizableDrawer from './ResizableDrawer';

function flattenGoals(list) {
  const out = [];
  for (const g of list || []) {
    out.push(g);
    if (g.children?.length) out.push(...flattenGoals(g.children));
  }
  return out;
}

export default function TaskDetailDrawer({ open, onClose, lineId, taskId }) {
  const queryClient = useQueryClient();
  const isCreate = !taskId;
  const [createTitle, setCreateTitle] = useState('');
  const [createDescription, setCreateDescription] = useState('');
  const [createPriority, setCreatePriority] = useState(0);
  const [createAssignedId, setCreateAssignedId] = useState('');
  const [createGoalId, setCreateGoalId] = useState('');
  const [createDueDate, setCreateDueDate] = useState('');

  const { data: team } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: open && !!lineId }
  );
  const { data: goals = [] } = useQuery(
    ['agentFactoryTeamGoals', lineId],
    () => apiService.agentFactory.getLineGoals(lineId),
    { enabled: open && !!lineId }
  );
  const { data: task, isLoading: taskLoading } = useQuery(
    ['agentFactoryTask', lineId, taskId],
    () => apiService.agentFactory.getTask(lineId, taskId),
    { enabled: open && !!lineId && !!taskId }
  );

  const goalsFlat = useMemo(() => flattenGoals(goals), [goals]);

  const createMutation = useMutation(
    (body) => apiService.agentFactory.createTask(lineId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
        setCreateTitle('');
        setCreateDescription('');
        setCreatePriority(0);
        setCreateAssignedId('');
        setCreateGoalId('');
        setCreateDueDate('');
        onClose?.();
      },
    }
  );

  const handleTransition = (tid, newStatus) => {
    apiService.agentFactory.transitionTask(lineId, tid, newStatus).then(() => {
      queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
      queryClient.invalidateQueries(['agentFactoryTask', lineId, tid]);
    });
  };

  const handleCreateSubmit = () => {
    if (!createTitle.trim()) return;
    createMutation.mutate({
      title: createTitle.trim(),
      description: createDescription.trim() || undefined,
      priority: createPriority,
      assigned_agent_id: createAssignedId || undefined,
      goal_id: createGoalId || undefined,
      due_date: createDueDate || undefined,
    });
  };

  if (!open) return null;

  return (
    <ResizableDrawer
      open={open}
      onClose={onClose}
      storageKey="agent-factory-line-task-drawer-width"
      defaultWidth={400}
      minWidth={320}
      maxWidth={720}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
        <Typography variant="h6">{isCreate ? 'Create task' : 'Task'}</Typography>
        <IconButton onClick={onClose} aria-label="Close" size="small">
          <Close />
        </IconButton>
      </Box>
      <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', p: 2 }}>
        {isCreate && team && (
          <>
            <TextField
              fullWidth
              label="Title"
              value={createTitle}
              onChange={(e) => setCreateTitle(e.target.value)}
              required
              size="small"
              margin="dense"
              sx={{ mt: 1 }}
            />
            <TextField
              fullWidth
              label="Description"
              value={createDescription}
              onChange={(e) => setCreateDescription(e.target.value)}
              size="small"
              margin="dense"
              multiline
              rows={3}
            />
            <FormControl fullWidth size="small" margin="dense">
              <InputLabel>Priority</InputLabel>
              <Select value={createPriority} label="Priority" onChange={(e) => setCreatePriority(Number(e.target.value))}>
                {[0, 1, 2, 3, 4].map((p) => (
                  <MenuItem key={p} value={p}>
                    P{p}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small" margin="dense">
              <InputLabel>Assigned agent</InputLabel>
              <Select value={createAssignedId} label="Assigned agent" onChange={(e) => setCreateAssignedId(e.target.value)}>
                <MenuItem value="">None</MenuItem>
                {(team.members || []).map((m) => (
                  <MenuItem key={m.agent_profile_id} value={m.agent_profile_id}>
                    {m.agent_name || m.agent_handle || m.agent_profile_id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small" margin="dense">
              <InputLabel>Goal</InputLabel>
              <Select value={createGoalId} label="Goal" onChange={(e) => setCreateGoalId(e.target.value)}>
                <MenuItem value="">None</MenuItem>
                {goalsFlat.map((g) => (
                  <MenuItem key={g.id} value={g.id}>
                    {g.title}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              fullWidth
              label="Due date"
              type="date"
              value={createDueDate}
              onChange={(e) => setCreateDueDate(e.target.value)}
              size="small"
              margin="dense"
              InputLabelProps={{ shrink: true }}
            />
            <DialogActions sx={{ px: 0, mt: 2, justifyContent: 'flex-start' }}>
              <Button onClick={onClose} disabled={createMutation.isLoading}>
                Cancel
              </Button>
              <Button variant="contained" onClick={handleCreateSubmit} disabled={!createTitle.trim() || createMutation.isLoading}>
                {createMutation.isLoading ? 'Creating…' : 'Create'}
              </Button>
            </DialogActions>
          </>
        )}
        {!isCreate && taskLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}
        {!isCreate && !taskLoading && task && (
          <TaskDetail task={task} onStatusChange={handleTransition} onClose={onClose} />
        )}
        {!isCreate && !taskLoading && !task && (
          <Typography color="text.secondary">Task not found.</Typography>
        )}
      </Box>
    </ResizableDrawer>
  );
}
