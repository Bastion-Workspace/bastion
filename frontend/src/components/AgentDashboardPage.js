/**
 * Agent Operations Dashboard: fleet overview, live activity feed, cost & health summary.
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Button,
  CircularProgress,
  Alert,
  useTheme,
} from '@mui/material';
import { Build, Schedule, Error as ErrorIcon, Warning, TrendingUp, CheckCircle, Cancel } from '@mui/icons-material';
import { useQuery, useQueryClient, useMutation } from 'react-query';
import apiService from '../services/apiService';

function formatUsd(val) {
  if (val == null || val === '') return '—';
  const n = Number(val);
  if (Number.isNaN(n)) return '—';
  if (n === 0) return '$0';
  if (n < 0.01) return '<$0.01';
  return `$${n.toFixed(2)}`;
}

function formatDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    const now = new Date();
    const diff = now - d;
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return d.toLocaleDateString();
  } catch (_) {
    return iso;
  }
}

function formatDuration(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

const DASHBOARD_WS_SUBTYPES = [
  'execution_started',
  'execution_completed',
  'execution_failed',
  'budget_warning',
  'budget_exceeded',
  'approval_required',
];

export default function AgentDashboardPage() {
  const navigate = useNavigate();
  const theme = useTheme();
  const queryClient = useQueryClient();
  const [activityLimit] = useState(50);

  useEffect(() => {
    const token = apiService.getToken?.() || localStorage.getItem('auth_token') || localStorage.getItem('token');
    if (!token) return;
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/conversations?token=${encodeURIComponent(token)}`;
    let ws = null;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'agent_notification' && data.subtype && DASHBOARD_WS_SUBTYPES.includes(data.subtype)) {
            queryClient.invalidateQueries('agentDashboardFleet');
            queryClient.invalidateQueries('agentDashboardCost');
            queryClient.invalidateQueries(['agentDashboardActivity', activityLimit]);
            if (data.subtype === 'approval_required') queryClient.invalidateQueries('agentPendingApprovals');
          }
        } catch (_) {}
      };
    } catch (_) {}
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    };
  }, [queryClient, activityLimit]);

  const { data: fleet = [], isLoading: fleetLoading } = useQuery(
    'agentDashboardFleet',
    () => apiService.agentFactory.dashboardFleetStatus(),
    { refetchInterval: 30000, retry: false }
  );
  const { data: costSummary, isLoading: costLoading } = useQuery(
    'agentDashboardCost',
    () => apiService.agentFactory.dashboardCostSummary('month'),
    { refetchInterval: 60000, retry: false }
  );
  const { data: activity = [], isLoading: activityLoading } = useQuery(
    ['agentDashboardActivity', activityLimit],
    () => apiService.agentFactory.dashboardActivityFeed(activityLimit),
    { refetchInterval: 15000, retry: false }
  );
  const { data: pendingApprovals = [], isLoading: approvalsLoading, refetch: refetchApprovals } = useQuery(
    'agentPendingApprovals',
    () => apiService.agentFactory.listPendingApprovals(),
    { refetchInterval: 20000, retry: false }
  );

  const overLimitCount = fleet.filter((a) => a.budget?.over_limit).length;
  const pausedCount = fleet.filter((a) => !a.is_active).length;
  const errorStreakCount = fleet.filter((a) => (a.consecutive_failures || 0) >= (a.max_consecutive_failures || 5)).length;

  const respondApprovalMutation = useMutation(
    ({ approvalId, approved }) => apiService.agentFactory.respondApproval(approvalId, approved),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentPendingApprovals');
        queryClient.invalidateQueries('agentDashboardFleet');
        queryClient.invalidateQueries('agentDashboardCost');
        queryClient.invalidateQueries(['agentDashboardActivity', activityLimit]);
      },
    }
  );

  const handleRespondApproval = (approvalId, approved) => {
    respondApprovalMutation.mutate({ approvalId, approved });
  };

  return (
    <Box sx={{ p: 2, maxWidth: 1400, mx: 'auto' }}>
      <Typography variant="h5" sx={{ mb: 2 }}>
        Agent operations
        {pendingApprovals.length > 0 && (
          <Chip
            size="small"
            label={`${pendingApprovals.length} approval${pendingApprovals.length !== 1 ? 's' : ''} pending`}
            color="warning"
            sx={{ ml: 1, verticalAlign: 'middle' }}
          />
        )}
      </Typography>

      {pendingApprovals.length > 0 && (
        <Card variant="outlined" sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
              Pending approvals
            </Typography>
            {approvalsLoading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            ) : (
              <List dense>
                {pendingApprovals.map((a) => (
                  <ListItem key={a.id} sx={{ alignItems: 'flex-start', flexWrap: 'wrap' }}>
                    <ListItemText
                      primary={a.agent_name}
                      secondary={
                        <>
                          <Typography variant="body2" component="span" display="block">
                            {a.governance_type === 'hire_agent' ? 'Hire request' : a.governance_type === 'strategy_change' ? 'Strategy change' : 'Step'}: {a.step_name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" component="span">
                            {typeof a.prompt === 'string' ? a.prompt.slice(0, 200) : ''}
                            {(a.prompt?.length || 0) > 200 ? '…' : ''}
                          </Typography>
                        </>
                      }
                    />
                    <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                      <Button
                        size="small"
                        variant="contained"
                        color="primary"
                        startIcon={<CheckCircle />}
                        onClick={() => handleRespondApproval(a.id, true)}
                        disabled={respondApprovalMutation.isLoading}
                      >
                        Approve
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        color="secondary"
                        startIcon={<Cancel />}
                        onClick={() => handleRespondApproval(a.id, false)}
                        disabled={respondApprovalMutation.isLoading}
                      >
                        Reject
                      </Button>
                    </Box>
                  </ListItem>
                ))}
              </List>
            )}
          </CardContent>
        </Card>
      )}

      <Grid container spacing={2}>
        {/* Cost & health summary */}
        <Grid item xs={12} md={4}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
                Cost & health
              </Typography>
              {costLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : (
                <>
                  <Typography variant="h4" color="primary.main" sx={{ mb: 1 }}>
                    {formatUsd(costSummary?.total_spend_usd)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    This period (since {costSummary?.period_start || '—'})
                  </Typography>
                  {(overLimitCount > 0 || pausedCount > 0 || errorStreakCount > 0) && (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
                      {overLimitCount > 0 && (
                        <Chip size="small" icon={<ErrorIcon />} label={`${overLimitCount} over budget`} color="error" />
                      )}
                      {pausedCount > 0 && (
                        <Chip size="small" icon={<Warning />} label={`${pausedCount} paused`} color="warning" />
                      )}
                      {errorStreakCount > 0 && (
                        <Chip size="small" icon={<ErrorIcon />} label={`${errorStreakCount} error streak`} color="error" />
                      )}
                    </Box>
                  )}
                  {costSummary?.by_agent?.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary">
                        Spend by agent
                      </Typography>
                      {costSummary.by_agent.slice(0, 8).map((a) => (
                        <Box key={a.agent_profile_id} sx={{ mt: 0.5 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body2" noWrap sx={{ maxWidth: 120 }}>
                              {a.name || a.handle || a.agent_profile_id}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {formatUsd(a.spend_usd)}
                              {a.limit_usd != null ? ` / ${formatUsd(a.limit_usd)}` : ''}
                            </Typography>
                          </Box>
                          {a.limit_usd != null && a.limit_usd > 0 && (
                            <LinearProgress
                              variant="determinate"
                              value={Math.min(100, (a.spend_usd / a.limit_usd) * 100)}
                              color={a.spend_usd >= a.limit_usd ? 'error' : 'primary'}
                              sx={{ height: 4, borderRadius: 1 }}
                            />
                          )}
                        </Box>
                      ))}
                    </Box>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Fleet grid */}
        <Grid item xs={12} md={8}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
                Agent fleet
              </Typography>
              {fleetLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : fleet.length === 0 ? (
                <Typography color="text.secondary">No agents yet. Create one in Agent Factory.</Typography>
              ) : (
                <Grid container spacing={1}>
                  {fleet.map((a) => (
                    <Grid item xs={12} sm={6} key={a.id}>
                      <Card
                        variant="outlined"
                        sx={{
                          cursor: 'pointer',
                          '&:hover': { bgcolor: 'action.hover' },
                          borderColor: a.budget?.over_limit ? 'error.main' : 'divider',
                        }}
                        onClick={() => navigate(`/agent-factory/agent/${a.id}`)}
                      >
                        <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                            <Build sx={{ fontSize: 18, color: 'text.secondary' }} />
                            <Typography variant="body2" fontWeight={600} noWrap sx={{ flex: 1, minWidth: 0 }}>
                              {a.name || a.handle || 'Unnamed'}
                            </Typography>
                            <Chip
                              size="small"
                              label={a.is_active ? 'Active' : 'Paused'}
                              color={a.is_active ? 'success' : 'default'}
                              sx={{ height: 20 }}
                            />
                          </Box>
                          <Typography variant="caption" color="text.secondary" display="block">
                            Last: {formatDate(a.last_execution_at)} · {a.last_execution_status || '—'}
                            {a.last_cost_usd != null && a.last_cost_usd > 0 && ` · ${formatUsd(a.last_cost_usd)}`}
                          </Typography>
                          {a.next_run_at && a.schedule_active && (
                            <Typography variant="caption" color="text.secondary" display="flex" alignItems="center" sx={{ mt: 0.25 }}>
                              <Schedule sx={{ fontSize: 12 }} /> Next: {formatDate(a.next_run_at)}
                            </Typography>
                          )}
                          {a.budget?.over_limit && (
                            <Alert severity="error" sx={{ mt: 0.5, py: 0.25 }}>Over budget</Alert>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Activity feed */}
        <Grid item xs={12}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 2 }}>
                Recent activity
              </Typography>
              {activityLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                  <CircularProgress size={24} />
                </Box>
              ) : activity.length === 0 ? (
                <Typography color="text.secondary">No recent runs.</Typography>
              ) : (
                <List dense disablePadding sx={{ maxHeight: 320, overflow: 'auto' }}>
                  {activity.map((ev) => (
                    <ListItem key={ev.id} disablePadding sx={{ borderBottom: 1, borderColor: 'divider', py: 0.5 }}>
                      <ListItemText
                        primary={
                          <>
                            <Typography component="span" variant="body2" fontWeight={500}>
                              {ev.agent_name || ev.agent_handle || 'Agent'}
                            </Typography>
                            <Chip
                              size="small"
                              label={ev.status}
                              color={ev.status === 'completed' ? 'success' : ev.status === 'failed' ? 'error' : 'default'}
                              sx={{ ml: 1, height: 18 }}
                            />
                            {ev.trigger_type && ev.trigger_type !== 'manual' && (
                              <Chip size="small" label={ev.trigger_type} variant="outlined" sx={{ ml: 0.5, height: 18 }} />
                            )}
                          </>
                        }
                        secondary={
                          <>
                            {ev.query ? `${ev.query.slice(0, 80)}${ev.query.length > 80 ? '…' : ''}` : '—'}
                            {' · '}
                            {formatDate(ev.started_at)}
                            {ev.duration_ms != null && ` · ${formatDuration(ev.duration_ms)}`}
                            {ev.cost_usd != null && ev.cost_usd > 0 && ` · ${formatUsd(ev.cost_usd)}`}
                            {ev.error_details && (
                              <Typography component="span" variant="caption" color="error" display="block">
                                {ev.error_details.slice(0, 100)}…
                              </Typography>
                            )}
                          </>
                        }
                        primaryTypographyProps={{ variant: 'body2' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
