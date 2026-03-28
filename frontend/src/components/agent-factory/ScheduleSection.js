/**
 * Schedule section for Agent Editor: list schedules, add, pause, resume, delete.
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  CircularProgress,
} from '@mui/material';
import ScheduleIcon from '@mui/icons-material/Schedule';
import AddIcon from '@mui/icons-material/Add';
import PauseIcon from '@mui/icons-material/Pause';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

const USER_TIMEZONE_QUERY_KEY = 'userTimezone';

const PRESETS = [
  { label: 'Every hour', schedule_type: 'cron', cron_expression: '0 * * * *' },
  { label: 'Every 30 min', schedule_type: 'cron', cron_expression: '*/30 * * * *' },
  { label: 'Daily at 9am', schedule_type: 'cron', cron_expression: '0 9 * * *' },
  { label: 'Weekdays at 8am', schedule_type: 'cron', cron_expression: '0 8 * * 1-5' },
  { label: 'Custom cron', schedule_type: 'cron', cron_expression: '' },
  { label: 'Every 5 min (interval)', schedule_type: 'interval', interval_seconds: 300 },
  { label: 'Every 15 min (interval)', schedule_type: 'interval', interval_seconds: 900 },
];

export default function ScheduleSection({ profileId }) {
  const queryClient = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [preset, setPreset] = useState('');
  const [scheduleType, setScheduleType] = useState('cron');
  const [cronExpression, setCronExpression] = useState('');
  const [intervalSeconds, setIntervalSeconds] = useState(300);
  const [timezone, setTimezone] = useState('UTC');

  const { data: userTzData } = useQuery(
    USER_TIMEZONE_QUERY_KEY,
    () => apiService.getUserTimezone(),
    { staleTime: 5 * 60 * 1000 }
  );

  const { data: schedules = [], isLoading } = useQuery(
    ['agentFactorySchedules', profileId],
    () => apiService.agentFactory.listSchedules(profileId),
    { enabled: !!profileId }
  );

  const createMutation = useMutation(
    (body) => apiService.agentFactory.createSchedule(profileId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactorySchedules', profileId]);
        setAddOpen(false);
        setPreset('');
        setCronExpression('');
        setIntervalSeconds(300);
      },
    }
  );

  const pauseMutation = useMutation(
    (scheduleId) => apiService.agentFactory.pauseSchedule(scheduleId),
    {
      onSuccess: () => queryClient.invalidateQueries(['agentFactorySchedules', profileId]),
    }
  );

  const resumeMutation = useMutation(
    (scheduleId) => apiService.agentFactory.resumeSchedule(scheduleId),
    {
      onSuccess: () => queryClient.invalidateQueries(['agentFactorySchedules', profileId]),
    }
  );

  const deleteMutation = useMutation(
    (scheduleId) => apiService.agentFactory.deleteSchedule(scheduleId),
    {
      onSuccess: () => queryClient.invalidateQueries(['agentFactorySchedules', profileId]),
    }
  );

  const handlePresetChange = (e) => {
    const v = e.target.value;
    setPreset(v);
    const p = PRESETS.find((x) => x.label === v);
    if (p) {
      setScheduleType(p.schedule_type);
      setCronExpression(p.cron_expression || '');
      setIntervalSeconds(p.interval_seconds ?? 300);
    }
  };

  const handleAddSubmit = () => {
    const body = {
      schedule_type: scheduleType,
      timezone,
    };
    if (scheduleType === 'cron') {
      if (!cronExpression.trim()) return;
      body.cron_expression = cronExpression.trim();
    } else {
      body.interval_seconds = intervalSeconds;
    }
    createMutation.mutate(body);
  };

  if (!profileId) return null;

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <ScheduleIcon /> Schedule
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Run this agent on a schedule (cron or interval). Failed runs are counted; after 5 consecutive failures the schedule is paused.
        </Typography>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <>
            <List dense>
              {schedules.length === 0 ? (
                <ListItem>
                  <ListItemText primary="No schedules" secondary="Add one to run this agent automatically." />
                </ListItem>
              ) : (
                schedules.map((s) => (
                  <ListItem
                    key={s.id}
                    secondaryAction={
                      <Box component="span" sx={{ display: 'flex', gap: 0.5 }}>
                        {s.is_active ? (
                          <IconButton
                            size="small"
                            title="Pause"
                            onClick={() => pauseMutation.mutate(s.id)}
                            disabled={pauseMutation.isLoading}
                          >
                            <PauseIcon />
                          </IconButton>
                        ) : (
                          <IconButton
                            size="small"
                            title="Resume"
                            onClick={() => resumeMutation.mutate(s.id)}
                            disabled={resumeMutation.isLoading}
                          >
                            <PlayArrowIcon />
                          </IconButton>
                        )}
                        <IconButton
                          size="small"
                          title="Delete"
                          onClick={() => deleteMutation.mutate(s.id)}
                          disabled={deleteMutation.isLoading}
                        >
                          <DeleteOutlineIcon />
                        </IconButton>
                      </Box>
                    }
                  >
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                          {s.schedule_type === 'cron' ? s.cron_expression : `Every ${s.interval_seconds}s`}
                          {!s.is_active && <Chip label="Paused" size="small" color="default" />}
                          {s.consecutive_failures > 0 && (
                            <Chip label={`${s.consecutive_failures} failures`} size="small" color="warning" />
                          )}
                        </Box>
                      }
                      secondary={
                        <>
                          Next: {s.next_run_at ? new Date(s.next_run_at).toLocaleString() : '—'}
                          {s.last_status && ` · Last: ${s.last_status}`}
                        </>
                      }
                    />
                  </ListItem>
                ))
              )}
            </List>
            <Button
              startIcon={<AddIcon />}
              variant="outlined"
              size="small"
              onClick={() => {
                setAddOpen(true);
                setTimezone(userTzData?.timezone ?? 'UTC');
              }}
            >
              Add schedule
            </Button>
          </>
        )}
      </CardContent>

      <Dialog open={addOpen} onClose={() => setAddOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add schedule</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 1, mb: 2 }}>
            <InputLabel>Preset</InputLabel>
            <Select value={preset} label="Preset" onChange={handlePresetChange}>
              <MenuItem value="">—</MenuItem>
              {PRESETS.map((p) => (
                <MenuItem key={p.label} value={p.label}>
                  {p.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {scheduleType === 'cron' && (
            <TextField
              fullWidth
              label="Cron expression"
              placeholder="0 8 * * 1-5"
              value={cronExpression}
              onChange={(e) => setCronExpression(e.target.value)}
              sx={{ mb: 2 }}
            />
          )}
          {scheduleType === 'interval' && (
            <TextField
              fullWidth
              type="number"
              label="Interval (seconds)"
              value={intervalSeconds}
              onChange={(e) => setIntervalSeconds(parseInt(e.target.value, 10) || 300)}
              sx={{ mb: 2 }}
            />
          )}
          <TextField
            fullWidth
            label="Timezone"
            value={timezone}
            onChange={(e) => setTimezone(e.target.value)}
            sx={{ mb: 2 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleAddSubmit}
            disabled={
              createMutation.isLoading ||
              (scheduleType === 'cron' && !cronExpression.trim()) ||
              (scheduleType === 'interval' && (!intervalSeconds || intervalSeconds < 60))
            }
          >
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Card>
  );
}
