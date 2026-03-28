/**
 * Line editor: internal tabs (Dashboard, Timeline, Tasks, Analytics) and settings/goals/task drawers.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Box, Tabs, Tab, IconButton, CircularProgress } from '@mui/material';
import { Settings as SettingsIcon } from '@mui/icons-material';
import LineDashboardPanel from './LineDashboardPanel';
import LineTimelinePanel from './LineTimelinePanel';
import LineTasksPanel from './LineTasksPanel';
import LineAnalyticsPanel from './LineAnalyticsPanel';
import LineSettingsDrawer from './LineSettingsDrawer';
import GoalEditorDrawer from './GoalEditorDrawer';
import TaskDetailDrawer from './TaskDetailDrawer';

const TAB_KEYS = ['dashboard', 'timeline', 'tasks', 'analytics'];
const AF_LINE_TAB_KEY = 'af-line-internal-tab';

function readStoredTab() {
  try {
    const v = localStorage.getItem(AF_LINE_TAB_KEY);
    const idx = parseInt(v, 10);
    if (Number.isFinite(idx) && idx >= 0 && idx < TAB_KEYS.length) return idx;
  } catch (_) {}
  return 0;
}

export default function LineEditor({ lineId, onCloseEntityTab }) {
  const [tab, setTab] = useState(readStoredTab);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [goalsOpen, setGoalsOpen] = useState(false);
  const [taskDrawerOpen, setTaskDrawerOpen] = useState(false);
  const [taskDrawerTaskId, setTaskDrawerTaskId] = useState(null);

  useEffect(() => {
    try {
      localStorage.setItem(AF_LINE_TAB_KEY, String(tab));
    } catch (_) {}
  }, [tab]);

  const handleTabChange = useCallback((_, v) => setTab(v), []);
  const openTaskCreate = useCallback(() => {
    setTaskDrawerTaskId(null);
    setTaskDrawerOpen(true);
  }, []);
  const openTaskDetail = useCallback((id) => {
    setTaskDrawerTaskId(id);
    setTaskDrawerOpen(true);
  }, []);
  const closeTaskDrawer = useCallback(() => {
    setTaskDrawerOpen(false);
    setTaskDrawerTaskId(null);
  }, []);

  const handleLineDeleted = useCallback(() => {
    onCloseEntityTab?.('line', lineId);
  }, [onCloseEntityTab, lineId]);

  if (!lineId) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      <Box
        sx={{
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          px: 1,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
        }}
      >
        <Tabs value={tab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto" sx={{ flex: 1, minWidth: 0 }}>
          <Tab label="Dashboard" />
          <Tab label="Timeline" />
          <Tab label="Tasks" />
          <Tab label="Analytics" />
        </Tabs>
        <IconButton aria-label="Line settings" onClick={() => setSettingsOpen(true)} size="small" sx={{ mr: 0.5 }}>
          <SettingsIcon />
        </IconButton>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {tab === 0 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <LineDashboardPanel
              lineId={lineId}
              onSelectTab={setTab}
              onOpenSettings={() => setSettingsOpen(true)}
              onOpenGoalsDrawer={() => setGoalsOpen(true)}
            />
          </Box>
        )}
        {tab === 1 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <LineTimelinePanel lineId={lineId} />
          </Box>
        )}
        {tab === 2 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <LineTasksPanel lineId={lineId} onRequestCreate={openTaskCreate} onSelectTask={openTaskDetail} />
          </Box>
        )}
        {tab === 3 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <LineAnalyticsPanel lineId={lineId} />
          </Box>
        )}
      </Box>

      <LineSettingsDrawer
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        lineId={lineId}
        onDeleted={handleLineDeleted}
      />
      <GoalEditorDrawer open={goalsOpen} onClose={() => setGoalsOpen(false)} lineId={lineId} />
      <TaskDetailDrawer
        open={taskDrawerOpen}
        onClose={closeTaskDrawer}
        lineId={lineId}
        taskId={taskDrawerTaskId}
      />
    </Box>
  );
}
