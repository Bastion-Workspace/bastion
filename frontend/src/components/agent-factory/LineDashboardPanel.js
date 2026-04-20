/**
 * Line dashboard tab: org chart, goals summary, tasks, timeline, workspace (embedded in LineEditor).
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Button,
  Link,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Snackbar,
  Alert,
  LinearProgress,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Switch,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
} from '@mui/material';
import { PlayArrow, Person, CheckCircle, Cancel, Stop, Forum, PlayCircleFilled, ExpandMore, Refresh } from '@mui/icons-material';
import { useQuery, useQueryClient } from 'react-query';
import { useNotifications } from '../../contexts/NotificationContext';
import { useTeamExecution } from '../../contexts/TeamExecutionContext';
import apiService from '../../services/apiService';
import OrgChartView from './OrgChartView';
import GoalTreeView from './GoalTreeView';
import TaskBoard from './TaskBoard';
import TimelineMessage from './TimelineMessage';

export default function LineDashboardPanel({ lineId, onGoToTab }) {
  const queryClient = useQueryClient();
  const { notifications } = useNotifications();
  const { teamStatusMap, setTeamExecutionStatus } = useTeamExecution();
  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const teamNotificationShownRef = React.useRef(new Set());
  const [invokeOpen, setInvokeOpen] = useState(false);
  const [invokeAgentId, setInvokeAgentId] = useState('');
  const [invokeQuery, setInvokeQuery] = useState('');
  const [invokeLoading, setInvokeLoading] = useState(false);
  const [respondApprovalLoading, setRespondApprovalLoading] = useState(false);
  const [stopAutonomousConfirmOpen, setStopAutonomousConfirmOpen] = useState(false);
  const [stopAutonomousLoading, setStopAutonomousLoading] = useState(false);
  const [resumeLoading, setResumeLoading] = useState(false);
  const [discussionOpen, setDiscussionOpen] = useState(false);
  const [discussionParticipants, setDiscussionParticipants] = useState([]);
  const [discussionTopic, setDiscussionTopic] = useState('');
  const [discussionModeratorId, setDiscussionModeratorId] = useState('');
  const [discussionMaxTurns, setDiscussionMaxTurns] = useState(10);
  const [discussionLoading, setDiscussionLoading] = useState(false);
  const wsRef = useRef(null);

  const executionStatus = lineId && teamStatusMap[lineId]
    ? { status: teamStatusMap[lineId].status, agent_id: teamStatusMap[lineId].agentId }
    : null;

  useEffect(() => {
    if (!lineId) return;
    const token = typeof window !== 'undefined' && window.localStorage.getItem('token');
    if (!token) return;
    const base = window.location.origin.replace(/^http/, 'ws');
    const url = `${base}/api/ws/team-timeline/${lineId}?token=${encodeURIComponent(token)}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (!data) return;
        if (data.type === 'team_timeline_update') {
          queryClient.invalidateQueries(['agentFactoryTeamTimelineRecent', lineId]);
          queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
          queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
          queryClient.invalidateQueries(['agentFactoryLineBriefSnapshots', lineId]);
        }
        if (data.type === 'task_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
        }
        if (data.type === 'goal_updated') {
          queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
        }
        if (data.type === 'execution_status') {
          setTeamExecutionStatus(lineId, {
            status: data.status,
            agent_id: data.agent_id ?? null,
            timestamp: data.timestamp || new Date().toISOString(),
          });
          if (data.status === 'idle') {
            queryClient.invalidateQueries(['agentFactoryLineBriefSnapshots', lineId]);
          }
        }
      } catch (_) {}
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [lineId, queryClient, setTeamExecutionStatus]);

  const { data: team, isLoading: teamLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: !!lineId }
  );

  const { data: orgChart } = useQuery(
    ['agentFactoryOrgChart', lineId],
    () => apiService.agentFactory.getLineOrgChart(lineId),
    { enabled: !!lineId }
  );

  const { data: goals = [] } = useQuery(
    ['agentFactoryTeamGoals', lineId],
    () => apiService.agentFactory.getLineGoals(lineId),
    { enabled: !!lineId }
  );

  const { data: tasks = [] } = useQuery(
    ['agentFactoryTeamTasks', lineId],
    () => apiService.agentFactory.listLineTasks(lineId),
    { enabled: !!lineId }
  );

  const { data: budget } = useQuery(
    ['agentFactoryTeamBudget', lineId],
    () => apiService.agentFactory.getLineBudgetSummary(lineId),
    { enabled: !!lineId }
  );

  const { data: timeline } = useQuery(
    ['agentFactoryTeamTimelineRecent', lineId],
    () => apiService.agentFactory.getLineTimeline(lineId, { limit: 10 }),
    { enabled: !!lineId }
  );

  const { data: teamApprovals = [], refetch: refetchTeamApprovals } = useQuery(
    ['agentFactoryTeamApprovals', lineId],
    () => apiService.agentFactory.getLineApprovals(lineId),
    { enabled: !!lineId }
  );

  const { data: agentHealth = [] } = useQuery(
    ['agentFactoryTeamAgentHealth', lineId],
    () => apiService.agentFactory.getLineAgentHealth(lineId, 7),
    { enabled: !!lineId }
  );

  const { data: workspaceData, refetch: refetchWorkspace } = useQuery(
    ['agentFactoryTeamWorkspace', lineId],
    () => apiService.agentFactory.getLineWorkspace(lineId),
    { enabled: !!lineId, refetchInterval: 30000 }
  );

  const { data: briefSnapshots = [], refetch: refetchBriefSnapshots } = useQuery(
    ['agentFactoryLineBriefSnapshots', lineId],
    () => apiService.agentFactory.getLineBriefSnapshots(lineId, 30),
    { enabled: !!lineId }
  );

  const [snapshotDialogOpen, setSnapshotDialogOpen] = useState(false);
  const [snapshotDialogLoading, setSnapshotDialogLoading] = useState(false);
  const [snapshotDialogTitle, setSnapshotDialogTitle] = useState('');
  const [snapshotDialogBody, setSnapshotDialogBody] = useState('');

  const governanceMode = team?.governance_mode || 'hierarchical';
  const governancePolicy = team?.governance_policy || {};
  const orgLayoutMode = ['committee', 'round_robin', 'consensus'].includes(governanceMode)
    ? 'circle'
    : 'tree';
  const orgHighlightIds = React.useMemo(() => {
    const ids = [];
    if (governanceMode === 'committee' && governancePolicy.chair_agent_id) {
      ids.push(String(governancePolicy.chair_agent_id));
    }
    if (
      governanceMode === 'round_robin'
      && Array.isArray(governancePolicy.rotation_order)
      && governancePolicy.rotation_order.length
      && team?.members?.length
    ) {
      const idx = Number(governancePolicy.current_leader_idx) || 0;
      const mid = governancePolicy.rotation_order[idx % governancePolicy.rotation_order.length];
      const mem = team.members.find((m) => String(m.id) === String(mid));
      if (mem?.agent_profile_id) ids.push(String(mem.agent_profile_id));
    }
    return ids;
  }, [governanceMode, governancePolicy, team]);

  const workspaceEntries = workspaceData?.entries ?? [];
  const [workspaceEntryValues, setWorkspaceEntryValues] = useState({});
  const workspaceEntryRequestedRef = useRef(new Set());

  const fetchWorkspaceEntryValue = React.useCallback(
    (key) => {
      if (!lineId || workspaceEntryRequestedRef.current.has(key)) return;
      workspaceEntryRequestedRef.current.add(key);
      setWorkspaceEntryValues((prev) => ({ ...prev, [key]: null }));
      apiService.agentFactory
        .getLineWorkspaceEntry(lineId, key)
        .then((res) => {
          const value = res?.value ?? '';
          setWorkspaceEntryValues((prev) => ({ ...prev, [key]: value }));
        })
        .catch(() => {
          setWorkspaceEntryValues((prev) => ({ ...prev, [key]: '(failed to load)' }));
        });
    },
    [lineId]
  );

  const teamEventSubtypes = ['heartbeat_failed', 'heartbeat_completed', 'team_budget_exceeded', 'team_emergency_stop', 'team_escalations', 'team_budget_warning'];
  React.useEffect(() => {
    const latest = notifications[0];
    if (!latest || !lineId || !teamEventSubtypes.includes(latest.subtype)) return;
    if (latest.team_id !== lineId) return;
    if (teamNotificationShownRef.current.has(latest.id)) return;
    teamNotificationShownRef.current.add(latest.id);
    const message = latest.message || latest.error_details || latest.subtype?.replace(/_/g, ' ') || 'Line event';
    const severity = latest.subtype === 'heartbeat_failed' || latest.subtype === 'team_budget_exceeded' ? 'error' : 'warning';
    setSnackbar({ open: true, message, severity });
  }, [notifications, lineId]);

  const flattenGoalsForTasks = (list) => {
    const out = [];
    for (const g of list || []) {
      out.push(g);
      if (g.children?.length) out.push(...flattenGoalsForTasks(g.children));
    }
    return out;
  };
  const goalsFlatForTasks = React.useMemo(() => flattenGoalsForTasks(goals), [goals]);

  const taskBoardAgentNameById = React.useMemo(() => {
    const m = {};
    for (const mem of team?.members || []) {
      const id = mem.agent_profile_id;
      if (!id) continue;
      m[id] = mem.agent_name || mem.agent_handle || id;
    }
    return m;
  }, [team]);

  const taskBoardGoalTitleById = React.useMemo(() => {
    const m = {};
    for (const g of goalsFlatForTasks) {
      if (g?.id) m[g.id] = g.title || g.id;
    }
    return m;
  }, [goalsFlatForTasks]);

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
        <Typography color="text.secondary">Line not found.</Typography>
      </Box>
    );
  }

  const recentMessages = timeline?.items ?? [];
  const members = team.members ?? [];

  const autonomyEnabled = Boolean(team?.heartbeat_config?.enabled);
  const handleActivateAutonomous = () => {
    setAutonomyLoading(true);
    const hb = team?.heartbeat_config || {};
    apiService.agentFactory
      .updateLine(lineId, { heartbeat_config: { ...hb, enabled: true } })
      .then(() => queryClient.invalidateQueries(['agentFactoryTeam', lineId]))
      .finally(() => setAutonomyLoading(false));
  };

  const handleInvokeSubmit = () => {
    if (!invokeAgentId || !invokeQuery.trim()) return;
    setInvokeLoading(true);
    apiService.agentFactory
      .invokeLineAgent(lineId, { agent_profile_id: invokeAgentId, query: invokeQuery.trim() })
      .then(() => {
        setInvokeOpen(false);
        setInvokeAgentId('');
        setInvokeQuery('');
      })
      .finally(() => setInvokeLoading(false));
  };

  const handleRespondApproval = (approvalId, approved) => {
    setRespondApprovalLoading(true);
    apiService.agentFactory.respondApproval(approvalId, approved).finally(() => {
      setRespondApprovalLoading(false);
      refetchTeamApprovals();
      queryClient.invalidateQueries('agentPendingApprovals');
    });
  };

  const handleStopAutonomous = () => {
    setStopAutonomousLoading(true);
    apiService.agentFactory.stopAutonomous(lineId).then(() => {
      setStopAutonomousConfirmOpen(false);
      queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
    }).finally(() => setStopAutonomousLoading(false));
  };

  const handleResumeTeam = () => {
    setResumeLoading(true);
    apiService.agentFactory.updateLine(lineId, { status: 'active' }).then(() => {
      queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
      setSnackbar({ open: true, message: 'Line resumed', severity: 'success' });
    }).catch((e) => {
      setSnackbar({ open: true, message: e?.message || 'Failed to resume line', severity: 'error' });
    }).finally(() => setResumeLoading(false));
  };

  const handleDiscussionToggle = (agentProfileId) => {
    setDiscussionParticipants((prev) =>
      prev.includes(agentProfileId)
        ? prev.filter((id) => id !== agentProfileId)
        : [...prev, agentProfileId]
    );
  };

  const handleDiscussionSubmit = () => {
    if (discussionParticipants.length < 2 || !discussionTopic.trim()) return;
    setDiscussionLoading(true);
    apiService.agentFactory
      .startLineDiscussion(lineId, {
        participant_ids: discussionParticipants,
        seed_message: discussionTopic.trim(),
        moderator_id: discussionModeratorId || undefined,
        max_turns: Math.min(30, Math.max(2, discussionMaxTurns)),
      })
      .then(() => {
        setDiscussionOpen(false);
        setDiscussionParticipants([]);
        setDiscussionTopic('');
        setDiscussionModeratorId('');
        setDiscussionMaxTurns(10);
        queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
        setSnackbar({ open: true, message: 'Discussion started', severity: 'success' });
      })
      .catch((e) => {
        setSnackbar({ open: true, message: e?.message || 'Failed to start discussion', severity: 'error' });
      })
      .finally(() => setDiscussionLoading(false));
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
      <Box
        sx={{
          position: 'sticky',
          top: 0,
          zIndex: 5,
          backgroundColor: 'background.default',
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexWrap: 'wrap', flexShrink: 0, p: 2, pb: 0 }}>
        <Typography variant="h6">{team.name}</Typography>
        {team.status && <Chip size="small" label={team.status} />}
        {budget && (() => {
          const spend = Number(budget.total_current_period_spend_usd) || 0;
          const teamLimit = team.budget_config?.monthly_limit_usd != null ? Number(team.budget_config.monthly_limit_usd) : null;
          const limit = teamLimit ?? (budget.total_monthly_limit_usd != null ? Number(budget.total_monthly_limit_usd) : null);
          const enforce = team.budget_config?.enforce_hard_limit !== false;
          const pct = (team.budget_config?.warning_threshold_pct ?? 80) / 100;
          const isOver = limit != null && enforce && spend >= limit;
          const isWarning = limit != null && !isOver && spend >= limit * pct;
          const chipColor = isOver ? 'error' : isWarning ? 'warning' : 'default';
          const label = limit != null ? `$${spend.toFixed(2)} / $${limit.toFixed(2)}` : `$${spend.toFixed(2)}`;
          return <Chip size="small" label={label} color={chipColor} variant="outlined" sx={{ fontWeight: 500 }} />;
        })()}
        <Box sx={{ flex: 1 }} />
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          {executionStatus?.status === 'running' && (
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: 'success.main',
                animation: 'pulse 1.2s ease-in-out infinite',
                '@keyframes pulse': { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.4 } },
              }}
              title="Heartbeat or agent running"
            />
          )}
          {team.status === 'active' && !autonomyEnabled && (
            <Button
              size="small"
              variant="outlined"
              color="primary"
              startIcon={<PlayArrow />}
              onClick={handleActivateAutonomous}
              disabled={autonomyLoading}
            >
              {autonomyLoading ? 'Updating…' : 'Activate autonomous'}
            </Button>
          )}
          {team.status === 'active' && (
            <Button
              size="small"
              variant="contained"
              color="error"
              startIcon={<Stop />}
              onClick={() => setStopAutonomousConfirmOpen(true)}
              disabled={stopAutonomousLoading}
            >
              Stop autonomous
            </Button>
          )}
        </Box>
        <Button size="small" variant="outlined" startIcon={<Person />} onClick={() => setInvokeOpen(true)}>
          Invoke agent
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<Forum />}
          onClick={() => setDiscussionOpen(true)}
          disabled={team.status !== 'active' || members.length < 2}
        >
          Start discussion
        </Button>
        {team.status === 'paused' && (
          <Button
            size="small"
            variant="contained"
            color="primary"
            startIcon={<PlayCircleFilled />}
            onClick={handleResumeTeam}
            disabled={resumeLoading}
          >
            {resumeLoading ? 'Resuming…' : 'Resume line'}
          </Button>
        )}
      </Box>

      {team.status === 'paused' && (
        <Alert severity="warning" sx={{ mx: 2, mb: 1 }}>
          This line is paused: no autonomous heartbeats, worker dispatches, or background message replies. Use Invoke agent for manual runs. Use Resume line above to turn scheduling back on or run heartbeats.
        </Alert>
      )}
      </Box>

      <Box sx={{ p: 2, pt: 0 }}>
      <Dialog open={stopAutonomousConfirmOpen} onClose={() => !stopAutonomousLoading && setStopAutonomousConfirmOpen(false)}>
        <DialogTitle>Stop autonomous</DialogTitle>
        <DialogContent>
          <Typography>
            Pause this line, turn off scheduled heartbeats, cancel any in-flight CEO run, and stop background worker
            dispatches (including from outstanding tasks). You can still use Invoke agent manually after this.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setStopAutonomousConfirmOpen(false)} disabled={stopAutonomousLoading}>Cancel</Button>
          <Button variant="contained" color="error" onClick={handleStopAutonomous} disabled={stopAutonomousLoading}>
            {stopAutonomousLoading ? 'Stopping…' : 'Stop autonomous'}
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar((s) => ({ ...s, open: false }))}>
          {snackbar.message}
        </Alert>
      </Snackbar>

      <Dialog open={invokeOpen} onClose={() => !invokeLoading && setInvokeOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Invoke agent</DialogTitle>
        <DialogContent>
          <FormControl fullWidth size="small" sx={{ mt: 1 }}>
            <InputLabel>Agent</InputLabel>
            <Select
              value={invokeAgentId}
              label="Agent"
              onChange={(e) => setInvokeAgentId(e.target.value)}
            >
              {members.map((m) => (
                <MenuItem key={m.agent_profile_id} value={m.agent_profile_id}>
                  {m.agent_name || m.agent_handle || m.agent_profile_id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Query / prompt"
            value={invokeQuery}
            onChange={(e) => setInvokeQuery(e.target.value)}
            multiline
            rows={3}
            size="small"
            margin="normal"
            placeholder="e.g. Review pending tasks and assign any blockers."
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setInvokeOpen(false)} disabled={invokeLoading}>Cancel</Button>
          <Button variant="contained" onClick={handleInvokeSubmit} disabled={!invokeAgentId || !invokeQuery.trim() || invokeLoading}>
            {invokeLoading ? 'Sending…' : 'Invoke'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={discussionOpen} onClose={() => !discussionLoading && setDiscussionOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start discussion</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Select at least 2 participants. Each must have an @handle set in their profile.
          </Typography>
          <FormGroup sx={{ mb: 2 }}>
            {members.map((m) => (
              <FormControlLabel
                key={m.agent_profile_id}
                control={
                  <Checkbox
                    checked={discussionParticipants.includes(m.agent_profile_id)}
                    onChange={() => handleDiscussionToggle(m.agent_profile_id)}
                  />
                }
                label={
                  <span>
                    {m.agent_name || m.agent_handle || m.agent_profile_id}
                    {m.agent_handle ? ` @${m.agent_handle}` : ' (no handle)'}
                  </span>
                }
              />
            ))}
          </FormGroup>
          <TextField
            fullWidth
            label="Topic / seed message"
            value={discussionTopic}
            onChange={(e) => setDiscussionTopic(e.target.value)}
            multiline
            rows={3}
            size="small"
            margin="normal"
            required
            placeholder="e.g. Discuss the pros and cons of communism vs capitalism."
          />
          <FormControl fullWidth size="small" sx={{ mt: 1 }}>
            <InputLabel>Moderator (optional)</InputLabel>
            <Select
              value={discussionModeratorId}
              label="Moderator (optional)"
              onChange={(e) => setDiscussionModeratorId(e.target.value)}
            >
              <MenuItem value="">None</MenuItem>
              {members.map((m) => (
                <MenuItem key={m.agent_profile_id} value={m.agent_profile_id}>
                  {m.agent_name || m.agent_handle || m.agent_profile_id}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Max turns"
            type="number"
            value={discussionMaxTurns}
            onChange={(e) => setDiscussionMaxTurns(parseInt(e.target.value, 10) || 10)}
            size="small"
            margin="normal"
            inputProps={{ min: 2, max: 30 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDiscussionOpen(false)} disabled={discussionLoading}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleDiscussionSubmit}
            disabled={discussionParticipants.length < 2 || !discussionTopic.trim() || discussionLoading}
          >
            {discussionLoading ? 'Starting…' : 'Start discussion'}
          </Button>
        </DialogActions>
      </Dialog>

      {team.description && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {team.description}
        </Typography>
      )}

      {teamApprovals.length > 0 && (
        <Card variant="outlined" sx={{ mb: 2, borderColor: 'warning.main' }}>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Pending approvals
            </Typography>
            {teamApprovals.map((a) => (
              <Box key={a.id} sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, py: 1, borderBottom: 1, borderColor: 'divider' }}>
                <Box>
                  <Typography variant="body2" fontWeight={500}>{a.agent_name}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {a.governance_type === 'hire_agent' ? 'Hire request' : a.governance_type === 'strategy_change' ? 'Strategy change' : a.step_name}: {(a.prompt || '').slice(0, 120)}{(a.prompt?.length || 0) > 120 ? '…' : ''}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  <Button size="small" variant="contained" color="primary" startIcon={<CheckCircle />} onClick={() => handleRespondApproval(a.id, true)} disabled={respondApprovalLoading}>Approve</Button>
                  <Button size="small" variant="outlined" color="secondary" startIcon={<Cancel />} onClick={() => handleRespondApproval(a.id, false)} disabled={respondApprovalLoading}>Reject</Button>
                </Box>
              </Box>
            ))}
          </CardContent>
        </Card>
      )}

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) minmax(0, 1.15fr)' },
          gap: 2,
        }}
      >
        <Card variant="outlined" sx={{ gridColumn: { xs: '1', md: '1 / -1' } }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 1 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Org chart
              </Typography>
              <Chip size="small" label={`Governance: ${governanceMode}`} variant="outlined" />
            </Box>
            <OrgChartView
              orgChart={orgChart ?? []}
              layoutMode={orgLayoutMode}
              highlightAgentProfileIds={orgHighlightIds}
              activeAgentProfileId={
                executionStatus?.status === 'running' && executionStatus?.agent_id
                  ? executionStatus.agent_id
                  : undefined
              }
            />
          </CardContent>
        </Card>

        <Card variant="outlined" sx={{ height: '100%' }}>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
              Agent health (7d)
            </Typography>
            {agentHealth.length === 0 && (
              <Typography variant="body2" color="text.secondary">No execution data</Typography>
            )}
            {agentHealth.map((h) => (
              <Box
                key={h.agent_profile_id}
                sx={{
                  display: 'grid',
                  gridTemplateColumns: 'auto 1fr auto auto',
                  alignItems: 'center',
                  gap: 1,
                  py: 0.75,
                  borderBottom: 1,
                  borderColor: 'divider',
                  '&:last-child': { borderBottom: 0 },
                }}
              >
                <Box
                  sx={{
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    bgcolor: h.agent_color || 'grey.500',
                  }}
                />
                <Typography variant="body2" fontWeight={500} sx={{ minWidth: 0 }}>
                  {h.agent_name}
                </Typography>
                <Box sx={{ minWidth: 64 }}>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, h.success_rate ?? 0)}
                    color={h.success_rate >= 80 ? 'success' : h.success_rate >= 50 ? 'warning' : 'error'}
                    sx={{ height: 6, borderRadius: 1 }}
                  />
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                  {h.last_run_at
                    ? (() => {
                        const d = new Date(h.last_run_at);
                        const n = new Date();
                        const s = Math.floor((n - d) / 1000);
                        if (s < 60) return 'just now';
                        if (s < 3600) return `${Math.floor(s / 60)}m ago`;
                        if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
                        return `${Math.floor(s / 86400)}d ago`;
                      })()
                    : '—'}
                  {' · $'}
                  {(h.total_cost_usd ?? 0).toFixed(2)}
                </Typography>
              </Box>
            ))}
          </CardContent>
        </Card>

        <Card variant="outlined" sx={{ height: '100%' }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" color="text.secondary">
                Goals (summary)
              </Typography>
              <Link component="button" variant="body2" onClick={() => onGoToTab?.('goals')}>
                View all
              </Link>
            </Box>
            <GoalTreeView tree={goals} />
          </CardContent>
        </Card>
      </Box>

      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Tasks
            </Typography>
            <Link component="button" variant="body2" onClick={() => onGoToTab?.('tasks')}>
              View board
            </Link>
          </Box>
          <TaskBoard
            tasks={tasks}
            onEditTask={null}
            onDeleteTask={null}
            onTransitionTask={(taskId, newStatus) => {
              apiService.agentFactory.transitionTask(lineId, taskId, newStatus).then(() => {
                queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
              });
            }}
            agentColorMap={Object.fromEntries((members || []).filter((m) => m.color).map((m) => [m.agent_profile_id, m.color]))}
            agentNameById={taskBoardAgentNameById}
            goalTitleById={taskBoardGoalTitleById}
            compact
            maxTasksPerColumn={3}
          />
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1, mt: 1.5, pt: 1, borderTop: 1, borderColor: 'divider' }}>
            <Typography variant="caption" color="text.secondary">
              One line per task; columns scroll after three rows. Hover a task for full details.
            </Typography>
            <Link component="button" variant="body2" onClick={() => onGoToTab?.('tasks')} sx={{ fontWeight: 600, whiteSpace: 'nowrap' }}>
              Open full task board
            </Link>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Recent timeline
            </Typography>
            <Link component="button" variant="body2" onClick={() => onGoToTab?.('timeline')}>
              View all
            </Link>
          </Box>
          {recentMessages.length === 0 && (
            <Typography variant="body2" color="text.secondary">No recent messages</Typography>
          )}
          {recentMessages.slice(0, 5).map((msg) => (
            <TimelineMessage key={msg.id} message={msg} />
          ))}
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Brief history (snapshots)
            </Typography>
            <IconButton size="small" onClick={() => refetchBriefSnapshots()} aria-label="Refresh brief snapshots">
              <Refresh fontSize="small" />
            </IconButton>
          </Box>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
            Stored after each successful heartbeat for diff context on the next run. Newest first.
          </Typography>
          {briefSnapshots.length === 0 && (
            <Typography variant="body2" color="text.secondary">No snapshots yet.</Typography>
          )}
          {briefSnapshots.map((row) => (
            <Box
              key={row.id}
              sx={{
                py: 0.75,
                borderBottom: 1,
                borderColor: 'divider',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                gap: 1,
              }}
            >
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="caption" color="text.secondary" display="block">
                  {row.created_at ? new Date(row.created_at).toLocaleString() : '—'} · {row.source || 'heartbeat'}
                </Typography>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {row.preview || ''}
                </Typography>
              </Box>
              <Button
                size="small"
                variant="text"
                onClick={() => {
                  setSnapshotDialogOpen(true);
                  setSnapshotDialogTitle(row.created_at ? new Date(row.created_at).toLocaleString() : 'Snapshot');
                  setSnapshotDialogLoading(true);
                  setSnapshotDialogBody('');
                  apiService.agentFactory
                    .getLineBriefSnapshotDetail(lineId, row.id)
                    .then((res) => {
                      setSnapshotDialogBody(res?.content || '');
                    })
                    .catch(() => {
                      setSnapshotDialogBody('Failed to load snapshot.');
                    })
                    .finally(() => setSnapshotDialogLoading(false));
                }}
              >
                Open
              </Button>
            </Box>
          ))}
        </CardContent>
      </Card>

      <Dialog open={snapshotDialogOpen} onClose={() => setSnapshotDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>{snapshotDialogTitle}</DialogTitle>
        <DialogContent dividers>
          {snapshotDialogLoading ? (
            <CircularProgress size={28} />
          ) : (
            <Typography component="pre" variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', m: 0 }}>
              {snapshotDialogBody}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSnapshotDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      <Card variant="outlined" sx={{ mt: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Shared Workspace
            </Typography>
            <IconButton size="small" onClick={() => refetchWorkspace()} aria-label="Refresh workspace">
              <Refresh fontSize="small" />
            </IconButton>
          </Box>
          {workspaceEntries.length === 0 && (
            <Typography variant="body2" color="text.secondary">No workspace entries yet.</Typography>
          )}
          {workspaceEntries.map((entry) => {
            const updatedAt = entry.updated_at;
            const relativeTime =
              updatedAt
                ? (() => {
                    const s = Math.floor((Date.now() - new Date(updatedAt).getTime()) / 1000);
                    if (s < 60) return 'just now';
                    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
                    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
                    return `${Math.floor(s / 86400)}d ago`;
                  })()
                : '—';
            return (
              <Accordion
                key={entry.key}
                disableGutters
                elevation={0}
                sx={{ '&:before': { display: 'none' }, borderBottom: 1, borderColor: 'divider' }}
                onChange={(_, expanded) => {
                  if (expanded) fetchWorkspaceEntryValue(entry.key);
                }}
              >
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="body2" fontWeight={500}>
                    {entry.key}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
                    {entry.updated_by_agent_name ?? 'unknown'} · {relativeTime}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Box
                    component="pre"
                    sx={{
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontFamily: 'monospace',
                      fontSize: 12,
                      m: 0,
                      maxHeight: 320,
                      overflow: 'auto',
                    }}
                  >
                    {workspaceEntryValues[entry.key] === undefined
                      ? 'Loading…'
                      : workspaceEntryValues[entry.key] === null
                        ? 'Loading…'
                        : workspaceEntryValues[entry.key] ?? '(empty)'}
                  </Box>
                </AccordionDetails>
              </Accordion>
            );
          })}
        </CardContent>
      </Card>
    </Box>
    </Box>
  );
}
