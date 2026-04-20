/**
 * Line editor: tabs (Dashboard, Timeline, Tasks, Analytics, Goals, Settings) and task detail drawer.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Box, Tabs, Tab, CircularProgress } from '@mui/material';
import LineDashboardPanel from './LineDashboardPanel';
import LineTimelinePanel from './LineTimelinePanel';
import LineTasksPanel from './LineTasksPanel';
import LineAnalyticsPanel from './LineAnalyticsPanel';
import LineSettingsPanel from './LineSettingsPanel';
import GoalEditorPanel from './GoalEditorPanel';
import TaskDetailDrawer from './TaskDetailDrawer';

const LINE_EDITOR_TAB_KEYS = ['dashboard', 'timeline', 'tasks', 'analytics', 'goals', 'settings'];

const AF_LINE_TAB_KEY = 'af-line-internal-tab';

function readStoredTab() {
  try {
    const v = localStorage.getItem(AF_LINE_TAB_KEY);
    const idx = parseInt(v, 10);
    if (Number.isFinite(idx) && idx >= 0 && idx < LINE_EDITOR_TAB_KEYS.length) return idx;
  } catch (_) {}
  return 0;
}

export default function LineEditor({ lineId, onCloseEntityTab }) {
  const [tab, setTab] = useState(readStoredTab);
  const [taskDrawerOpen, setTaskDrawerOpen] = useState(false);
  const [taskDrawerTaskId, setTaskDrawerTaskId] = useState(null);

  const goToTab = useCallback((key) => {
    const idx = LINE_EDITOR_TAB_KEYS.indexOf(key);
    if (idx >= 0) setTab(idx);
  }, []);

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
          bgcolor: (t) => t.palette.surface?.main ?? t.palette.background.default,
        }}
      >
        <Tabs
          value={tab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            flex: 1,
            minWidth: 0,
            '& .MuiTab-root': {
              bgcolor: (t) => t.palette.surface?.main ?? t.palette.background.default,
              '&.Mui-selected': { bgcolor: 'background.default' },
            },
          }}
        >
          <Tab label="Dashboard" />
          <Tab label="Timeline" />
          <Tab label="Tasks" />
          <Tab label="Analytics" />
          <Tab label="Goals" />
          <Tab label="Settings" />
        </Tabs>
      </Box>

      <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {tab === 0 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
            <LineDashboardPanel lineId={lineId} onGoToTab={goToTab} />
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
        {tab === 4 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <GoalEditorPanel lineId={lineId} />
          </Box>
        )}
        {tab === 5 && (
          <Box sx={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            <LineSettingsPanel lineId={lineId} onDeleted={handleLineDeleted} />
          </Box>
        )}
      </Box>

      <TaskDetailDrawer
        open={taskDrawerOpen}
        onClose={closeTaskDrawer}
        lineId={lineId}
        taskId={taskDrawerTaskId}
      />
    </Box>
  );
}
