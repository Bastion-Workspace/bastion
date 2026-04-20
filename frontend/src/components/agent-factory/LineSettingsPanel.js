/**
 * Line settings: general, heartbeat config, governance policy, danger zone.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  FormControlLabel,
  Switch,
  CircularProgress,
  Divider,
  Chip,
  Checkbox,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  FormLabel,
  InputLabel,
  Select,
  MenuItem,
  Radio,
  RadioGroup,
  Snackbar,
  Alert,
} from '@mui/material';
import { DeleteForever, Lock, DeleteSweep, RestartAlt } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import { invalidateAgentHandlesQuery } from '../../services/agentFactoryService';
import TeamMembershipEditor from './TeamMembershipEditor';
import LineReferenceSection from './LineReferenceSection';
import LineDataWorkspaceSection from './LineDataWorkspaceSection';
import { buildLineUpdatePayload, buildHeartbeatConfig } from './lineSettingsSavePayload';

const formatUtcPreview = (iso) => {
  try {
    return new Date(iso).toLocaleString(undefined, { timeZone: 'UTC', timeZoneName: 'short' });
  } catch {
    return iso;
  }
};

const HEARTBEAT_CRON_PRESETS = [
  { label: 'Hourly', cron: '0 * * * *' },
  { label: 'Every 6h', cron: '0 */6 * * *' },
  { label: 'Daily 7:00', cron: '0 7 * * *' },
  { label: 'Weekdays 8:00', cron: '0 8 * * 1-5' },
  { label: 'Mon 9:00', cron: '0 9 * * 1' },
];

const HEARTBEAT_INTERVAL_PRESETS = [
  { label: '1 hour', seconds: 3600 },
  { label: '6 hours', seconds: 21600 },
  { label: '24 hours', seconds: 86400 },
];

const HEARTBEAT_TIMEZONES = [
  'UTC',
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'Europe/London',
  'Europe/Paris',
  'Asia/Tokyo',
  'Australia/Sydney',
];

export default function LineSettingsPanel({ lineId, onDeleted }) {
  const queryClient = useQueryClient();
  const referenceSectionRef = useRef(null);
  const dataWorkspaceSectionRef = useRef(null);
  const [saveSnackbar, setSaveSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const { data: team, isLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: !!lineId }
  );
  const { data: orgChart = [] } = useQuery(
    ['agentFactoryOrgChart', lineId],
    () => apiService.agentFactory.getLineOrgChart(lineId),
    { enabled: !!lineId }
  );

  const [name, setName] = useState('');
  const [handle, setHandle] = useState('');
  const [description, setDescription] = useState('');
  const [missionStatement, setMissionStatement] = useState('');
  const [status, setStatus] = useState('active');
  const [heartbeatEnabled, setHeartbeatEnabled] = useState(false);
  const [scheduleType, setScheduleType] = useState('none');
  const [heartbeatTimezone, setHeartbeatTimezone] = useState('UTC');
  const [intervalSeconds, setIntervalSeconds] = useState('');
  const [cronExpression, setCronExpression] = useState('');
  const [schedulePreviewOccurrences, setSchedulePreviewOccurrences] = useState([]);
  const [schedulePreviewError, setSchedulePreviewError] = useState('');
  const [schedulePreviewLoading, setSchedulePreviewLoading] = useState(false);
  const [requireOpenGoals, setRequireOpenGoals] = useState(true);
  const [maxAutonomousRuns, setMaxAutonomousRuns] = useState('');
  const [continuityWorkspaceKeys, setContinuityWorkspaceKeys] = useState('');
  const [continuityMaxChars, setContinuityMaxChars] = useState('');
  const [deliveryOutputSections, setDeliveryOutputSections] = useState('');
  const [deliveryDisclaimer, setDeliveryDisclaimer] = useState('');
  const [deliveryPublishKey, setDeliveryPublishKey] = useState('');
  const [deliveryCanonicalKey, setDeliveryCanonicalKey] = useState('');
  const [deliveryExtraInstructions, setDeliveryExtraInstructions] = useState('');
  const [deliveryNotifySuccess, setDeliveryNotifySuccess] = useState(false);
  const [deliveryNotifyFailure, setDeliveryNotifyFailure] = useState(true);
  const [deliveryPublishOverwrite, setDeliveryPublishOverwrite] = useState(false);
  const [governanceDescription, setGovernanceDescription] = useState('');
  const [governanceMode, setGovernanceMode] = useState('hierarchical');
  const [chairAgentId, setChairAgentId] = useState('');
  const [quorumCount, setQuorumCount] = useState('');
  const [quorumPct, setQuorumPct] = useState('60');
  const [tiebreakerAgentId, setTiebreakerAgentId] = useState('');
  const [monthlyLimit, setMonthlyLimit] = useState('');
  const [enforceHardLimit, setEnforceHardLimit] = useState(true);
  const [warningThresholdPct, setWarningThresholdPct] = useState('80');
  const [teamSkillIds, setTeamSkillIds] = useState([]);

  const { data: skillsList = [] } = useQuery(
    ['agentFactorySkills'],
    () => apiService.agentFactory.listSkills({ include_builtin: true }),
    { enabled: !!lineId, staleTime: 60000 }
  );

  useEffect(() => {
    if (team) {
      setName(team.name ?? '');
      setHandle(team.handle ?? '');
      setDescription(team.description ?? '');
      setMissionStatement(team.mission_statement ?? '');
      setStatus(team.status ?? 'active');
      const hb = team.heartbeat_config || {};
      setHeartbeatEnabled(Boolean(hb.enabled));
      const st = hb.schedule_type;
      if (st === 'interval' || st === 'cron' || st === 'none') {
        setScheduleType(st);
      } else if ((hb.cron_expression || '').trim()) {
        setScheduleType('cron');
      } else if (hb.interval_seconds != null && Number(hb.interval_seconds) > 0) {
        setScheduleType('interval');
      } else {
        setScheduleType('none');
      }
      setHeartbeatTimezone(typeof hb.timezone === 'string' && hb.timezone.trim() ? hb.timezone.trim() : 'UTC');
      setIntervalSeconds(hb.interval_seconds != null ? String(hb.interval_seconds) : '');
      setCronExpression(hb.cron_expression ?? '');
      const rog = hb.require_open_goals;
      const briefMode = rog === false || String(rog).toLowerCase() === 'false' || String(rog) === '0' || String(rog).toLowerCase() === 'no';
      setRequireOpenGoals(!briefMode);
      setMaxAutonomousRuns(hb.max_autonomous_runs != null && hb.max_autonomous_runs !== '' ? String(hb.max_autonomous_runs) : '');
      const cwk = hb.continuity_workspace_keys;
      if (Array.isArray(cwk)) {
        setContinuityWorkspaceKeys(cwk.filter(Boolean).join('\n'));
      } else if (typeof cwk === 'string') {
        setContinuityWorkspaceKeys(cwk);
      } else {
        setContinuityWorkspaceKeys('');
      }
      setContinuityMaxChars(
        hb.continuity_max_chars_per_key != null && hb.continuity_max_chars_per_key !== ''
          ? String(hb.continuity_max_chars_per_key)
          : ''
      );
      const d = hb.delivery && typeof hb.delivery === 'object' ? hb.delivery : {};
      const secs = d.output_sections;
      setDeliveryOutputSections(Array.isArray(secs) ? secs.filter(Boolean).join('\n') : '');
      setDeliveryDisclaimer(typeof d.disclaimer_block === 'string' ? d.disclaimer_block : '');
      setDeliveryPublishKey(typeof d.publish_workspace_key === 'string' ? d.publish_workspace_key : '');
      setDeliveryCanonicalKey(typeof d.canonical_snapshot_key === 'string' ? d.canonical_snapshot_key : '');
      setDeliveryExtraInstructions(typeof d.extra_instructions === 'string' ? d.extra_instructions : '');
      setDeliveryNotifySuccess(Boolean(d.notify_on_success));
      setDeliveryNotifyFailure(d.notify_on_failure !== false);
      setDeliveryPublishOverwrite(Boolean(d.publish_workspace_overwrite));
      const gov = team.governance_policy || {};
      setGovernanceDescription(gov.description ?? gov.policy_description ?? '');
      setGovernanceMode(team.governance_mode || 'hierarchical');
      setChairAgentId(gov.chair_agent_id ? String(gov.chair_agent_id) : '');
      setQuorumCount(gov.quorum_count != null ? String(gov.quorum_count) : '');
      setQuorumPct(gov.quorum_pct != null ? String(gov.quorum_pct) : '60');
      setTiebreakerAgentId(gov.tiebreaker_agent_id ? String(gov.tiebreaker_agent_id) : '');
      const bc = team.budget_config || {};
      setMonthlyLimit(bc.monthly_limit_usd != null ? String(bc.monthly_limit_usd) : '');
      setEnforceHardLimit(bc.enforce_hard_limit !== false);
      setWarningThresholdPct(bc.warning_threshold_pct != null ? String(bc.warning_threshold_pct) : '80');
      setTeamSkillIds(Array.isArray(team.team_skill_ids) ? [...team.team_skill_ids] : []);
    }
  }, [team]);

  const updateMutation = useMutation(
    (body) => apiService.agentFactory.updateLine(lineId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries('agentFactoryLines');
        invalidateAgentHandlesQuery(queryClient);
      },
    }
  );

  const deleteMutation = useMutation(
    () => apiService.agentFactory.deleteLine(lineId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries('agentFactoryLines');
        invalidateAgentHandlesQuery(queryClient);
        onDeleted?.();
      },
    }
  );

  const handleSaveAll = useCallback(async () => {
    if (!name.trim() || !team) return;
    const referenceConfig = referenceSectionRef.current?.getConfig?.();
    const dataWorkspaceConfig = dataWorkspaceSectionRef.current?.getConfig?.();
    const payload = buildLineUpdatePayload({
      team,
      name,
      handle,
      description,
      missionStatement,
      status,
      monthlyLimit,
      enforceHardLimit,
      warningThresholdPct,
      heartbeatEnabled,
      scheduleType,
      heartbeatTimezone,
      intervalSeconds,
      cronExpression,
      requireOpenGoals,
      maxAutonomousRuns,
      continuityWorkspaceKeys,
      continuityMaxChars,
      deliveryNotifySuccess,
      deliveryNotifyFailure,
      deliveryPublishOverwrite,
      deliveryOutputSections,
      deliveryDisclaimer,
      deliveryPublishKey,
      deliveryCanonicalKey,
      deliveryExtraInstructions,
      governanceMode,
      governanceDescription,
      chairAgentId,
      quorumCount,
      quorumPct,
      tiebreakerAgentId,
      teamSkillIds,
      referenceConfig,
      dataWorkspaceConfig,
    });
    try {
      await updateMutation.mutateAsync(payload);
      setSaveSnackbar({ open: true, message: 'Settings saved', severity: 'success' });
    } catch (err) {
      const detail = err?.response?.data?.detail;
      const msg = Array.isArray(detail)
        ? detail.join('; ')
        : detail || err?.message || 'Save failed';
      setSaveSnackbar({ open: true, message: typeof msg === 'string' ? msg : 'Save failed', severity: 'error' });
    }
  }, [
    team,
    name,
    handle,
    description,
    missionStatement,
    status,
    monthlyLimit,
    enforceHardLimit,
    warningThresholdPct,
    heartbeatEnabled,
    scheduleType,
    heartbeatTimezone,
    intervalSeconds,
    cronExpression,
    requireOpenGoals,
    maxAutonomousRuns,
    continuityWorkspaceKeys,
    continuityMaxChars,
    deliveryNotifySuccess,
    deliveryNotifyFailure,
    deliveryPublishOverwrite,
    deliveryOutputSections,
    deliveryDisclaimer,
    deliveryPublishKey,
    deliveryCanonicalKey,
    deliveryExtraInstructions,
    governanceMode,
    governanceDescription,
    chairAgentId,
    quorumCount,
    quorumPct,
    tiebreakerAgentId,
    teamSkillIds,
    updateMutation,
  ]);

  const handlePreviewSchedule = useCallback(async () => {
    if (!team || !lineId) return;
    setSchedulePreviewLoading(true);
    setSchedulePreviewError('');
    setSchedulePreviewOccurrences([]);
    const hb = buildHeartbeatConfig({
      team,
      heartbeatEnabled,
      scheduleType,
      heartbeatTimezone,
      intervalSeconds,
      cronExpression,
      requireOpenGoals,
      maxAutonomousRuns,
      continuityWorkspaceKeys,
      continuityMaxChars,
      deliveryNotifySuccess,
      deliveryNotifyFailure,
      deliveryPublishOverwrite,
      deliveryOutputSections,
      deliveryDisclaimer,
      deliveryPublishKey,
      deliveryCanonicalKey,
      deliveryExtraInstructions,
    });
    try {
      const res = await apiService.agentFactory.previewHeartbeatSchedule(lineId, {
        heartbeat_config: hb,
        count: 5,
      });
      if (res.errors && res.errors.length) {
        setSchedulePreviewError(res.errors.join('; '));
      } else {
        setSchedulePreviewOccurrences(res.next_occurrences || []);
      }
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Preview failed';
      setSchedulePreviewError(typeof msg === 'string' ? msg : 'Preview failed');
    } finally {
      setSchedulePreviewLoading(false);
    }
  }, [
    team,
    lineId,
    heartbeatEnabled,
    scheduleType,
    heartbeatTimezone,
    intervalSeconds,
    cronExpression,
    requireOpenGoals,
    maxAutonomousRuns,
    continuityWorkspaceKeys,
    continuityMaxChars,
    deliveryNotifySuccess,
    deliveryNotifyFailure,
    deliveryPublishOverwrite,
    deliveryOutputSections,
    deliveryDisclaimer,
    deliveryPublishKey,
    deliveryCanonicalKey,
    deliveryExtraInstructions,
  ]);

  // setPackMode removed — tool packs retired in Skills-First Architecture

  const toggleTeamSkill = (skillId) => {
    setTeamSkillIds((prev) =>
      prev.includes(skillId) ? prev.filter((id) => id !== skillId) : [...prev, skillId]
    );
  };

  const skillsByCategory = React.useMemo(() => {
    const map = {};
    (skillsList || []).forEach((s) => {
      const cat = s.category || 'General';
      if (!map[cat]) map[cat] = [];
      map[cat].push(s);
    });
    return map;
  }, [skillsList]);
  const skillCategories = Object.keys(skillsByCategory).sort();

  const handlePause = () => {
    updateMutation.mutate({ status: 'paused' });
  };

  const handleResume = () => {
    updateMutation.mutate({ status: 'active' });
  };

  const [clearTimelineConfirmOpen, setClearTimelineConfirmOpen] = useState(false);
  const [clearTimelineLoading, setClearTimelineLoading] = useState(false);
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
  const [resetLoading, setResetLoading] = useState(false);

  const handleClearTimeline = () => {
    setClearTimelineLoading(true);
    apiService.agentFactory.clearLineTimeline(lineId).then(() => {
      setClearTimelineConfirmOpen(false);
      queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimelineRecent', lineId]);
    }).finally(() => setClearTimelineLoading(false));
  };

  const handleResetLine = () => {
    setResetLoading(true);
    apiService.agentFactory.resetLine(lineId).then(() => {
      setResetConfirmOpen(false);
      queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimelineRecent', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamWorkspace', lineId]);
      invalidateAgentHandlesQuery(queryClient);
    }).finally(() => setResetLoading(false));
  };

  const handleDelete = () => {
    if (window.confirm('Delete this line and all its members, messages, goals, and tasks? This cannot be undone.')) {
      deleteMutation.mutate();
    }
  };

  const saving = updateMutation.isLoading;
  const lastBeat = team?.last_beat_at ? new Date(team.last_beat_at).toLocaleString() : '—';
  const nextBeat = team?.next_beat_at ? new Date(team.next_beat_at).toLocaleString() : '—';

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 2, p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
        <Typography variant="h6" sx={{ flex: 1, minWidth: 0 }}>
          Line settings{team?.name ? `: ${team.name}` : ''}
        </Typography>
        <Button
          variant="contained"
          size="small"
          onClick={handleSaveAll}
          disabled={!team || !name.trim() || saving}
        >
          {saving ? 'Saving…' : 'Save'}
        </Button>
      </Box>
      <Box sx={{ flex: 1, minHeight: 0, overflow: 'auto', p: 2 }}>
      {(isLoading || !lineId) && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}
      {!isLoading && lineId && !team && (
        <Box sx={{ p: 2 }}>
          <Typography color="text.secondary">Line not found.</Typography>
        </Box>
      )}
      {!isLoading && team && (
        <>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            General
          </Typography>
          <Box>
            <TextField
              fullWidth
              label="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              size="small"
              margin="dense"
            />
            <TextField
              fullWidth
              label="Handle"
              value={handle}
              onChange={(e) => setHandle(e.target.value)}
              size="small"
              margin="dense"
              placeholder="e.g. political-tracker"
              helperText="Chat: @handle for a line summary (e.g. @political-tracker what's new?)"
            />
            <TextField
              fullWidth
              label="Description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              size="small"
              margin="dense"
              multiline
              rows={2}
            />
            <TextField
              fullWidth
              label="Mission statement"
              value={missionStatement}
              onChange={(e) => setMissionStatement(e.target.value)}
              size="small"
              margin="dense"
              multiline
              rows={2}
            />
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Members
          </Typography>
          <TeamMembershipEditor
            lineId={lineId}
            members={team.members ?? []}
            orgChart={orgChart}
          />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Line skills
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
            Skills applied to all members when they run.
            Line collaboration tools are always available in line context.
          </Typography>
          <Box>
            <Box sx={{ maxHeight: 180, overflowY: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
              {skillCategories.map((cat) => (
                <Box key={cat} sx={{ mb: 1 }}>
                  <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    {cat}
                  </Typography>
                  {(skillsByCategory[cat] || []).map((skill) => {
                    const checked = teamSkillIds.includes(skill.id);
                    return (
                      <FormControlLabel
                        key={skill.id}
                        control={
                          <Checkbox
                            size="small"
                            checked={checked}
                            onChange={() => toggleTeamSkill(skill.id)}
                            sx={{ p: 0.25, mr: 0.5 }}
                          />
                        }
                        label={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                            {skill.is_builtin && <Lock sx={{ fontSize: 14 }} color="action" titleAccess="Built-in" />}
                            <span>{skill.name || skill.slug}</span>
                            {Array.isArray(skill.required_tools) && skill.required_tools.length > 0 && (
                              <Chip
                                size="small"
                                label={`+${skill.required_tools.length} tools`}
                                sx={{ height: 18, fontSize: '0.7rem', cursor: 'help' }}
                                variant="outlined"
                                title={skill.required_tools.join(', ')}
                              />
                            )}
                          </Box>
                        }
                        sx={{ display: 'flex', alignItems: 'center', mx: 0, my: 0.25, '& .MuiFormControlLabel-label': { fontSize: '0.8125rem' } }}
                      />
                    );
                  })}
                </Box>
              ))}
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Reference files &amp; folders
          </Typography>
          <LineReferenceSection ref={referenceSectionRef} team={team} />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Data Workspaces
          </Typography>
          <LineDataWorkspaceSection ref={dataWorkspaceSectionRef} team={team} />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Budget
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
            Monthly spend cap for this line. When enforced, runs stop once over the limit.
          </Typography>
          <Box>
            <TextField
              fullWidth
              label="Monthly limit (USD)"
              type="number"
              value={monthlyLimit}
              onChange={(e) => setMonthlyLimit(e.target.value)}
              size="small"
              margin="dense"
              placeholder="e.g. 50 (leave empty for no limit)"
              inputProps={{ min: 0, step: 0.01 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={enforceHardLimit}
                  onChange={(e) => setEnforceHardLimit(e.target.checked)}
                  color="primary"
                />
              }
              label="Enforce hard limit (block runs when over)"
            />
            <TextField
              fullWidth
              label="Warning threshold (%)"
              type="number"
              value={warningThresholdPct}
              onChange={(e) => setWarningThresholdPct(e.target.value)}
              size="small"
              margin="dense"
              inputProps={{ min: 1, max: 99 }}
            />
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Heartbeat
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={heartbeatEnabled}
                onChange={(e) => setHeartbeatEnabled(e.target.checked)}
                color="primary"
              />
            }
            label="Enable autonomous heartbeat (CEO runs periodically)"
          />
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
            Disabling heartbeat pauses the line and stops background workers.
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
            Last beat: {lastBeat} · Next beat: {nextBeat}
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
            Due times are checked about every 60 seconds. Minimum interval is 60 seconds.
          </Typography>
          <Box sx={{ mt: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={requireOpenGoals}
                  onChange={(e) => setRequireOpenGoals(e.target.checked)}
                  color="primary"
                />
              }
              label="Only schedule when there are open goals (under 100% progress)"
            />
            <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
              Turn off for &quot;briefing mode&quot;: the line can beat on schedule even with no active goals.
            </Typography>
            <TextField
              fullWidth
              label="Stop autonomous heartbeat after N successful runs (optional)"
              type="number"
              value={maxAutonomousRuns}
              onChange={(e) => setMaxAutonomousRuns(e.target.value)}
              size="small"
              margin="dense"
              placeholder="Leave empty for no limit"
              inputProps={{ min: 1 }}
              helperText="Counts scheduled heartbeats only (not manual Run heartbeat). Heartbeat turns off when the limit is reached."
            />
            <FormLabel id="line-schedule-type-label" sx={{ mt: 2, mb: 0.5, display: 'block' }}>
              Periodic schedule
            </FormLabel>
            <RadioGroup
              aria-labelledby="line-schedule-type-label"
              value={scheduleType}
              onChange={(e) => setScheduleType(e.target.value)}
            >
              <FormControlLabel value="none" control={<Radio size="small" />} label="None (no automatic cadence)" />
              <FormControlLabel value="interval" control={<Radio size="small" />} label="Every N seconds" />
              <FormControlLabel value="cron" control={<Radio size="small" />} label="Cron (wall-clock in time zone)" />
            </RadioGroup>
            <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
              Pick one mode. Cron uses the time zone below; interval is a fixed period in UTC.
            </Typography>
            {scheduleType === 'interval' && (
              <Box sx={{ mb: 1 }}>
                <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 0.5 }}>
                  Interval presets
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                  {HEARTBEAT_INTERVAL_PRESETS.map((p) => (
                    <Chip
                      key={p.label}
                      label={p.label}
                      size="small"
                      onClick={() => {
                        setScheduleType('interval');
                        setIntervalSeconds(String(p.seconds));
                      }}
                      variant="outlined"
                    />
                  ))}
                </Box>
                <TextField
                  fullWidth
                  label="Interval (seconds)"
                  type="number"
                  value={intervalSeconds}
                  onChange={(e) => setIntervalSeconds(e.target.value)}
                  size="small"
                  margin="dense"
                  placeholder="Minimum 60"
                  inputProps={{ min: 60 }}
                />
              </Box>
            )}
            {scheduleType === 'cron' && (
              <Box sx={{ mb: 1 }}>
                <FormControl fullWidth size="small" margin="dense" sx={{ mb: 1 }}>
                  <InputLabel id="hb-tz-label">Time zone (cron)</InputLabel>
                  <Select
                    labelId="hb-tz-label"
                    label="Time zone (cron)"
                    value={heartbeatTimezone || 'UTC'}
                    onChange={(e) => setHeartbeatTimezone(e.target.value)}
                  >
                    {!HEARTBEAT_TIMEZONES.includes(heartbeatTimezone || '') && (heartbeatTimezone || '').trim() ? (
                      <MenuItem value={heartbeatTimezone}>{heartbeatTimezone}</MenuItem>
                    ) : null}
                    {HEARTBEAT_TIMEZONES.map((tz) => (
                      <MenuItem key={tz} value={tz}>
                        {tz}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 0.5 }}>
                  Cron presets
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                  {HEARTBEAT_CRON_PRESETS.map((p) => (
                    <Chip
                      key={p.label}
                      label={p.label}
                      size="small"
                      onClick={() => {
                        setScheduleType('cron');
                        setCronExpression(p.cron);
                      }}
                      variant="outlined"
                    />
                  ))}
                </Box>
                <TextField
                  fullWidth
                  label="Cron expression"
                  value={cronExpression}
                  onChange={(e) => setCronExpression(e.target.value)}
                  size="small"
                  margin="dense"
                  placeholder="e.g. 0 7 * * 1-5"
                />
              </Box>
            )}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mt: 1 }}>
              <Button size="small" variant="outlined" onClick={handlePreviewSchedule} disabled={schedulePreviewLoading}>
                {schedulePreviewLoading ? 'Preview…' : 'Preview next runs'}
              </Button>
            </Box>
            {schedulePreviewError && (
              <Alert severity="warning" sx={{ mt: 1 }} onClose={() => setSchedulePreviewError('')}>
                {schedulePreviewError}
              </Alert>
            )}
            {schedulePreviewOccurrences.length > 0 && (
              <Typography variant="caption" component="div" color="text.secondary" sx={{ mt: 1 }}>
                Next runs (UTC):{' '}
                {schedulePreviewOccurrences.map((iso) => formatUtcPreview(iso)).join(' · ')}
              </Typography>
            )}
            <TextField
              fullWidth
              label="Continuity: workspace keys (one per line)"
              value={continuityWorkspaceKeys}
              onChange={(e) => setContinuityWorkspaceKeys(e.target.value)}
              size="small"
              margin="dense"
              multiline
              minRows={2}
              placeholder={'e.g. current_brief\nuser_preferences'}
              helperText="CEO heartbeat context will include full values for these keys (see continuity max chars)."
            />
            <TextField
              fullWidth
              label="Continuity: max characters per key"
              type="number"
              value={continuityMaxChars}
              onChange={(e) => setContinuityMaxChars(e.target.value)}
              size="small"
              margin="dense"
              placeholder="8000"
              inputProps={{ min: 500, max: 50000 }}
            />
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Delivery (brief output)
            </Typography>
            <TextField
              fullWidth
              label="Delivery section headings (one per line; model uses as ## headings)"
              value={deliveryOutputSections}
              onChange={(e) => setDeliveryOutputSections(e.target.value)}
              size="small"
              margin="dense"
              multiline
              minRows={2}
            />
            <TextField
              fullWidth
              label="Disclaimer (appended to summaries)"
              value={deliveryDisclaimer}
              onChange={(e) => setDeliveryDisclaimer(e.target.value)}
              size="small"
              margin="dense"
              multiline
              minRows={2}
            />
            <TextField
              fullWidth
              label="Publish full response to workspace key (on success)"
              value={deliveryPublishKey}
              onChange={(e) => setDeliveryPublishKey(e.target.value)}
              size="small"
              margin="dense"
              placeholder="e.g. current_brief"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={deliveryPublishOverwrite}
                  onChange={(e) => setDeliveryPublishOverwrite(e.target.checked)}
                  size="small"
                />
              }
              label="Overwrite workspace key even if it already has content"
            />
            <TextField
              fullWidth
              label="Canonical snapshot workspace key (optional)"
              value={deliveryCanonicalKey}
              onChange={(e) => setDeliveryCanonicalKey(e.target.value)}
              size="small"
              margin="dense"
              helperText="If set and non-empty, brief snapshots prefer this key over timeline text."
            />
            <TextField
              fullWidth
              label="Extra delivery instructions"
              value={deliveryExtraInstructions}
              onChange={(e) => setDeliveryExtraInstructions(e.target.value)}
              size="small"
              margin="dense"
              multiline
              minRows={2}
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={deliveryNotifySuccess}
                  onChange={(e) => setDeliveryNotifySuccess(e.target.checked)}
                  size="small"
                />
              }
              label="Notify on successful heartbeat"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={deliveryNotifyFailure}
                  onChange={(e) => setDeliveryNotifyFailure(e.target.checked)}
                  size="small"
                />
              }
              label="Notify on failed heartbeat"
            />
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Governance
          </Typography>
          <Box>
            <FormControl fullWidth size="small" margin="dense">
              <InputLabel id="gov-mode-label">Governance mode</InputLabel>
              <Select
                labelId="gov-mode-label"
                label="Governance mode"
                value={governanceMode}
                onChange={(e) => setGovernanceMode(e.target.value)}
              >
                <MenuItem value="hierarchical">Hierarchical (CEO root + reports)</MenuItem>
                <MenuItem value="committee">Committee (parallel members, optional chair)</MenuItem>
                <MenuItem value="round_robin">Round-robin leader per heartbeat</MenuItem>
                <MenuItem value="consensus">Consensus (quorum on proposed actions)</MenuItem>
              </Select>
            </FormControl>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5, mb: 1 }}>
              Round-robin uses member join order; the active leader advances after each heartbeat.
            </Typography>
            {governanceMode === 'committee' && (
              <>
                <FormControl fullWidth size="small" margin="dense">
                  <InputLabel id="chair-label">Chair agent (optional)</InputLabel>
                  <Select
                    labelId="chair-label"
                    label="Chair agent (optional)"
                    value={chairAgentId}
                    onChange={(e) => setChairAgentId(e.target.value)}
                  >
                    <MenuItem value="">
                      <em>None — merge member outputs only</em>
                    </MenuItem>
                    {(team?.members || []).map((m) => (
                      <MenuItem key={m.agent_profile_id} value={String(m.agent_profile_id)}>
                        {m.agent_name || m.agent_handle || m.agent_profile_id}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  label="Quorum member count (optional)"
                  value={quorumCount}
                  onChange={(e) => setQuorumCount(e.target.value)}
                  size="small"
                  margin="dense"
                  type="number"
                  placeholder="e.g. 3"
                />
              </>
            )}
            {governanceMode === 'consensus' && (
              <>
                <TextField
                  fullWidth
                  label="Quorum percent (1–100)"
                  value={quorumPct}
                  onChange={(e) => setQuorumPct(e.target.value)}
                  size="small"
                  margin="dense"
                  type="number"
                  inputProps={{ min: 1, max: 100 }}
                />
                <FormControl fullWidth size="small" margin="dense">
                  <InputLabel id="tiebreaker-label">Tiebreaker agent (optional)</InputLabel>
                  <Select
                    labelId="tiebreaker-label"
                    label="Tiebreaker agent (optional)"
                    value={tiebreakerAgentId}
                    onChange={(e) => setTiebreakerAgentId(e.target.value)}
                  >
                    <MenuItem value="">
                      <em>None</em>
                    </MenuItem>
                    {(team?.members || []).map((m) => (
                      <MenuItem key={m.agent_profile_id} value={String(m.agent_profile_id)}>
                        {m.agent_name || m.agent_handle || m.agent_profile_id}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </>
            )}
            <TextField
              fullWidth
              label="Policy description"
              value={governanceDescription}
              onChange={(e) => setGovernanceDescription(e.target.value)}
              size="small"
              margin="dense"
              multiline
              rows={3}
              placeholder="e.g. All hires and strategy changes require approval."
            />
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ borderColor: 'error.main' }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} color="error" sx={{ mb: 1 }}>
            Danger zone
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Clear timeline removes messages only. Reset line also clears tasks; goals, members, and settings stay.
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {status === 'active' ? (
              <Button variant="outlined" color="warning" size="small" onClick={handlePause} disabled={saving}>
                Pause line
              </Button>
            ) : (
              <Button variant="outlined" color="primary" size="small" onClick={handleResume} disabled={saving}>
                Resume line
              </Button>
            )}
            <Button
              variant="outlined"
              color="warning"
              size="small"
              startIcon={<DeleteSweep />}
              onClick={() => setClearTimelineConfirmOpen(true)}
              disabled={clearTimelineLoading || saving}
            >
              Clear timeline
            </Button>
            <Button
              variant="outlined"
              color="warning"
              size="small"
              startIcon={<RestartAlt />}
              onClick={() => setResetConfirmOpen(true)}
              disabled={resetLoading || saving}
            >
              Reset line
            </Button>
            <Button
              variant="outlined"
              color="error"
              size="small"
              startIcon={<DeleteForever />}
              onClick={handleDelete}
              disabled={deleteMutation.isLoading}
            >
              Delete line
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Dialog open={clearTimelineConfirmOpen} onClose={() => !clearTimelineLoading && setClearTimelineConfirmOpen(false)}>
        <DialogTitle>Clear timeline</DialogTitle>
        <DialogContent>
          <Typography>Remove all messages from this line&apos;s timeline? This cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearTimelineConfirmOpen(false)} disabled={clearTimelineLoading}>Cancel</Button>
          <Button variant="contained" color="warning" onClick={handleClearTimeline} disabled={clearTimelineLoading}>
            {clearTimelineLoading ? 'Clearing…' : 'Clear timeline'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog open={resetConfirmOpen} onClose={() => !resetLoading && setResetConfirmOpen(false)}>
        <DialogTitle>Reset line</DialogTitle>
        <DialogContent>
          <Typography>
            Clear all timeline messages and all tasks for this line? Goals, members, and settings are kept. This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetConfirmOpen(false)} disabled={resetLoading}>Cancel</Button>
          <Button variant="contained" color="warning" onClick={handleResetLine} disabled={resetLoading}>
            {resetLoading ? 'Resetting…' : 'Reset line'}
          </Button>
        </DialogActions>
      </Dialog>
        </>
      )}
      </Box>
      <Snackbar
        open={saveSnackbar.open}
        autoHideDuration={6000}
        onClose={() => setSaveSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          severity={saveSnackbar.severity}
          onClose={() => setSaveSnackbar((s) => ({ ...s, open: false }))}
          sx={{ width: '100%' }}
        >
          {saveSnackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
