/**
 * Line timeline: chronological feed of inter-agent messages with filters and live updates.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  CircularProgress,
  Button,
  Paper,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Menu,
  MenuItem,
  Snackbar,
  Alert,
} from '@mui/material';
import { DeleteSweep, FileDownload } from '@mui/icons-material';
import { useQuery, useQueryClient } from 'react-query';
import { useAuth } from '../../contexts/AuthContext';
import apiService from '../../services/apiService';
import exportService from '../../services/exportService';
import TimelineMessage from './TimelineMessage';
import TimelineFilters from './TimelineFilters';
import AgentActivityPanel from './AgentActivityPanel';
import {
  fetchAllTimelineMessages,
  buildTimelineMarkdown,
  timelineExportBasename,
} from './timelineExportUtils';

function useTeamTimelineWebSocket(lineId, token, onMessage, queryClient) {
  const wsRef = useRef(null);
  useEffect(() => {
    if (!lineId || !token) return;
    const base = window.location.origin.replace(/^http/, 'ws');
    const url = `${base}/api/ws/team-timeline/${lineId}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'team_timeline_update') {
          if (data.message) onMessage(data.message);
          queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
          queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
        }
        if (data.type === 'task_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
        }
        if (data.type === 'goal_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
        }
      } catch (_) {}
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [lineId, token, onMessage, queryClient]);
}

export default function LineTimelinePanel({ lineId }) {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const [messageType, setMessageType] = useState(null);
  const [agentFilter, setAgentFilter] = useState(null);
  const [since, setSince] = useState(null);
  const [offset, setOffset] = useState(0);
  const [expandedThreadId, setExpandedThreadId] = useState(null);
  const [userMessage, setUserMessage] = useState('');
  const [postingMessage, setPostingMessage] = useState(false);
  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);
  const [clearLoading, setClearLoading] = useState(false);
  const [exportMenuAnchor, setExportMenuAnchor] = useState(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const limit = 50;

  const { data: team, isLoading: teamLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: !!lineId }
  );

  const { data: timeline, isLoading: timelineLoading, refetch } = useQuery(
    ['agentFactoryTeamTimeline', lineId, messageType, agentFilter, since, offset],
    () =>
      apiService.agentFactory.getLineTimeline(lineId, {
        limit,
        offset,
        message_type: messageType,
        agent: agentFilter,
        since,
      }),
    { enabled: !!lineId }
  );

  const token = typeof window !== 'undefined' && window.localStorage.getItem('token');
  const [liveMessages, setLiveMessages] = useState([]);
  useTeamTimelineWebSocket(lineId, token, (msg) => setLiveMessages((prev) => [msg, ...prev].slice(0, 20)), queryClient);

  const items = timeline?.items ?? [];
  const total = timeline?.total ?? 0;
  const members = team?.members ?? [];
  const agentOptions = members.map((m) => ({
    id: m.agent_profile_id,
    name: m.agent_name,
    handle: m.agent_handle,
  }));

  const activityByAgent = {};
  [...items, ...liveMessages].forEach((msg) => {
    const id = msg.from_agent_id || msg.to_agent_id;
    if (id) {
      if (!activityByAgent[id]) activityByAgent[id] = { count: 0 };
      activityByAgent[id].count += 1;
      if (msg.created_at && (!activityByAgent[id].last_activity || activityByAgent[id].last_activity < msg.created_at)) {
        activityByAgent[id].last_activity = msg.created_at;
      }
    }
  });

  const combinedItems = [...liveMessages, ...items.filter((i) => !liveMessages.find((l) => l.id === i.id))];

  const { data: threadMessages = [] } = useQuery(
    ['agentFactoryTeamThread', lineId, expandedThreadId],
    () => apiService.agentFactory.getLineMessageThread(lineId, expandedThreadId),
    { enabled: !!lineId && !!expandedThreadId }
  );

  const handlePostMessage = (e) => {
    e.preventDefault();
    if (!userMessage.trim() || postingMessage) return;
    setPostingMessage(true);
    apiService.agentFactory
      .postLineMessage(lineId, { content: userMessage.trim() })
      .then(() => {
        setUserMessage('');
        queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
      })
      .finally(() => setPostingMessage(false));
  };

  const handleClearTimeline = () => {
    setClearLoading(true);
    apiService.agentFactory.clearLineTimeline(lineId).then(() => {
      setClearConfirmOpen(false);
      setLiveMessages([]);
      queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimelineRecent', lineId]);
    }).finally(() => setClearLoading(false));
  };

  const exportFilterDescription = () => {
    const parts = [];
    if (messageType) parts.push(`message type = ${messageType}`);
    if (agentFilter) {
      const a = agentOptions.find((x) => x.id === agentFilter);
      parts.push(`agent = ${a ? a.name || a.handle || agentFilter : agentFilter}`);
    }
    if (since) parts.push(`since ${since}`);
    return parts.length
      ? `This export uses the current filters: ${parts.join(', ')}.`
      : 'This export includes all messages (no filters).';
  };

  const handleExport = async (mode) => {
    setExportMenuAnchor(null);
    if (!lineId || !team || exportBusy) return;
    setExportBusy(true);
    try {
      const msgs = await fetchAllTimelineMessages(
        (id, params) => apiService.agentFactory.getLineTimeline(id, params),
        lineId,
        { messageType, agentFilter, since }
      );
      if (msgs.length === 0) {
        setSnackbar({ open: true, message: 'No messages match the current filters.', severity: 'warning' });
        return;
      }
      const md = buildTimelineMarkdown(team.name, msgs, exportFilterDescription());
      const base = timelineExportBasename(team.name);
      if (mode === 'pdf') {
        await exportService.exportMarkdownAsPDF(md, { filename: `${base}.pdf` });
        setSnackbar({ open: true, message: 'PDF downloaded.', severity: 'success' });
      } else {
        if (!user?.user_id) {
          setSnackbar({ open: true, message: 'Sign in to save to your library.', severity: 'warning' });
          return;
        }
        const res = await apiService.createDocumentFromContent({
          content: md,
          title: `Timeline — ${team.name}`,
          filename: `${base}.md`,
          userId: user.user_id,
          folderId: null,
          docType: 'md',
        });
        setSnackbar({ open: true, message: 'Timeline saved as a Markdown document in your library.', severity: 'success' });
        queryClient.invalidateQueries(['folders', 'tree', user.user_id, user?.role]);
        if (res?.document_id && window.tabbedContentManagerRef?.openDocument) {
          window.tabbedContentManagerRef.openDocument(res.document_id, res.filename || `${base}.md`);
        }
      }
    } catch (e) {
      setSnackbar({
        open: true,
        message: e?.message || 'Export failed',
        severity: 'error',
      });
    } finally {
      setExportBusy(false);
    }
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
      <Box sx={{ p: 4 }}>
        <Typography color="text.secondary">Line not found.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, height: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexShrink: 0 }}>
        <Typography variant="h6">Timeline: {team.name}</Typography>
        <Box sx={{ flex: 1 }} />
        <Button
          size="small"
          variant="outlined"
          startIcon={<FileDownload />}
          onClick={(e) => setExportMenuAnchor(e.currentTarget)}
          disabled={exportBusy}
        >
          {exportBusy ? 'Export…' : 'Export'}
        </Button>
        <Menu
          anchorEl={exportMenuAnchor}
          open={Boolean(exportMenuAnchor)}
          onClose={() => setExportMenuAnchor(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <MenuItem
            onClick={() => handleExport('pdf')}
            disabled={exportBusy}
          >
            Download PDF
          </MenuItem>
          <MenuItem
            onClick={() => handleExport('library')}
            disabled={exportBusy}
          >
            Save to library (Markdown)
          </MenuItem>
        </Menu>
        <Button
          size="small"
          variant="outlined"
          color="warning"
          startIcon={<DeleteSweep />}
          onClick={() => setClearConfirmOpen(true)}
          disabled={total === 0 && liveMessages.length === 0}
        >
          Clear timeline
        </Button>
      </Box>
      <Dialog open={clearConfirmOpen} onClose={() => !clearLoading && setClearConfirmOpen(false)}>
        <DialogTitle>Clear timeline</DialogTitle>
        <DialogContent>
          <Typography>Remove all messages from this line&apos;s timeline? This cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearConfirmOpen(false)} disabled={clearLoading}>Cancel</Button>
          <Button variant="contained" color="warning" onClick={handleClearTimeline} disabled={clearLoading}>
            {clearLoading ? 'Clearing…' : 'Clear timeline'}
          </Button>
        </DialogActions>
      </Dialog>
      <Paper variant="outlined" sx={{ p: 1.5, mb: 2 }}>
        <form onSubmit={handlePostMessage}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Post a message to the line timeline…"
              value={userMessage}
              onChange={(e) => setUserMessage(e.target.value)}
              multiline
              maxRows={3}
            />
            <Button type="submit" variant="contained" disabled={!userMessage.trim() || postingMessage}>
              {postingMessage ? 'Sending…' : 'Post'}
            </Button>
          </Box>
        </form>
      </Paper>
      <TimelineFilters
        messageType={messageType}
        agentId={agentFilter}
        since={since}
        agentOptions={agentOptions}
        onMessageTypeChange={setMessageType}
        onAgentChange={setAgentFilter}
        onSinceChange={setSince}
      />
      <Box sx={{ display: 'flex', gap: 2, flex: 1, minHeight: 0 }}>
        <Paper variant="outlined" sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {timelineLoading && offset === 0 ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          ) : combinedItems.length === 0 ? (
            <Typography color="text.secondary">No messages yet.</Typography>
          ) : (
            <>
              {combinedItems.map((msg) => (
                <TimelineMessage
                  key={msg.id}
                  message={msg}
                  onExpandThread={(rootId) => setExpandedThreadId(rootId)}
                />
              ))}
              {total > offset + limit && (
                <Button size="small" onClick={() => setOffset((o) => o + limit)}>
                  Load more
                </Button>
              )}
            </>
          )}
        </Paper>
        <Paper variant="outlined" sx={{ width: 240, overflow: 'auto' }}>
          <AgentActivityPanel members={members} activityByAgent={activityByAgent} />
        </Paper>
      </Box>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar((s) => ({ ...s, open: false }))} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>

      <Dialog open={!!expandedThreadId} onClose={() => setExpandedThreadId(null)} maxWidth="sm" fullWidth>
        <DialogTitle>Thread</DialogTitle>
        <DialogContent>
          {threadMessages.length === 0 && expandedThreadId && (
            <Box sx={{ py: 2, textAlign: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          )}
          {threadMessages.map((msg) => (
            <TimelineMessage key={msg.id} message={msg} />
          ))}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
