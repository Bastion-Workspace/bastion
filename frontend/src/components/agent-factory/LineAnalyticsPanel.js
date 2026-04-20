/**
 * Line analytics: task throughput, cost trend, goal progress, agent breakdown, message volume.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

export default function LineAnalyticsPanel({ lineId }) {
  const [days, setDays] = useState(30);

  const { data: team, isLoading: teamLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: !!lineId }
  );

  const { data: analytics, isLoading: analyticsLoading } = useQuery(
    ['agentFactoryTeamAnalytics', lineId, days],
    () => apiService.agentFactory.getLineAnalytics(lineId, days),
    { enabled: !!lineId }
  );

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

  const taskThroughput = analytics?.task_throughput ?? [];
  const costOverTime = analytics?.cost_over_time ?? [];
  const goalProgress = analytics?.goal_progress ?? [];
  const agentActivity = analytics?.agent_activity ?? [];
  const messageVolume = analytics?.message_volume ?? [];
  const monthlyLimit = team?.budget_config?.monthly_limit_usd != null ? Number(team.budget_config.monthly_limit_usd) : null;
  const dailyBudgetRef = monthlyLimit != null && days > 0 ? monthlyLimit / (days < 31 ? 30 : days) : null;

  const formatDate = (d) => {
    if (!d) return '';
    const dt = new Date(d);
    return dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2, flexWrap: 'wrap' }}>
        <Typography variant="h6">{team.name} – Analytics</Typography>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Period</InputLabel>
          <Select value={days} label="Period" onChange={(e) => setDays(Number(e.target.value))}>
            <MenuItem value={7}>7 days</MenuItem>
            <MenuItem value={14}>14 days</MenuItem>
            <MenuItem value={30}>30 days</MenuItem>
            <MenuItem value={90}>90 days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {analyticsLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!analyticsLoading && (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Task throughput
              </Typography>
              {taskThroughput.length === 0 ? (
                <Typography variant="body2" color="text.secondary">No task data in this period</Typography>
              ) : (
                <Box sx={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={taskThroughput} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={formatDate} />
                      <YAxis allowDecimals={false} />
                      <Tooltip labelFormatter={formatDate} />
                      <Legend />
                      <Area type="monotone" dataKey="created" stroke="#1976d2" fill="#1976d2" fillOpacity={0.4} name="Created" />
                      <Area type="monotone" dataKey="completed" stroke="#2e7d32" fill="#2e7d32" fillOpacity={0.4} name="Completed" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              )}
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Cost over time
              </Typography>
              {costOverTime.length === 0 ? (
                <Typography variant="body2" color="text.secondary">No cost data in this period</Typography>
              ) : (
                <Box sx={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={costOverTime} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={formatDate} />
                      <YAxis tickFormatter={(v) => `$${v}`} />
                      <Tooltip formatter={(v) => [`$${Number(v).toFixed(2)}`, 'Cost']} labelFormatter={formatDate} />
                      {dailyBudgetRef != null && (
                        <ReferenceLine y={dailyBudgetRef} stroke="orange" strokeDasharray="3 3" name="Daily budget (avg)" />
                      )}
                      <Line type="monotone" dataKey="cost_usd" stroke="#1976d2" name="Cost ($)" dot />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              )}
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Goal progress
              </Typography>
              {goalProgress.length === 0 ? (
                <Typography variant="body2" color="text.secondary">No goals</Typography>
              ) : (
                <Box sx={{ height: Math.max(200, goalProgress.length * 36) }}>
                  <ResponsiveContainer width="100%" height={Math.max(180, goalProgress.length * 32)}>
                    <BarChart data={goalProgress} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[0, 100]} unit="%" />
                      <YAxis type="category" dataKey="title" width={76} tick={{ fontSize: 12 }} />
                      <Tooltip formatter={(v) => [`${v}%`, 'Progress']} />
                      <Bar dataKey="progress_pct" fill="#1976d2" name="Progress %" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              )}
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Agent activity
              </Typography>
              {agentActivity.length === 0 ? (
                <Typography variant="body2" color="text.secondary">No execution data</Typography>
              ) : (
                <Box sx={{ height: Math.max(200, agentActivity.length * 48) }}>
                  <ResponsiveContainer width="100%" height={Math.max(180, agentActivity.length * 40)}>
                    <BarChart data={agentActivity} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="agent_name" tick={{ fontSize: 12 }} />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="runs" fill="#1976d2" name="Runs" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="successes" fill="#2e7d32" name="Successes" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="failures" fill="#d32f2f" name="Failures" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              )}
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
                Message volume
              </Typography>
              {messageVolume.length === 0 ? (
                <Typography variant="body2" color="text.secondary">No messages in this period</Typography>
              ) : (
                <Box sx={{ height: 260 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={messageVolume} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tickFormatter={formatDate} />
                      <YAxis allowDecimals={false} />
                      <Tooltip labelFormatter={formatDate} />
                      <Area type="monotone" dataKey="count" stroke="#7b1fa2" fill="#7b1fa2" name="Messages" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  );
}
