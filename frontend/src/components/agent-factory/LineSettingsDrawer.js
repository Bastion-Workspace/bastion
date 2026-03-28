/**
 * Team settings: general, heartbeat config, governance policy, danger zone.
 */

import React, { useState, useEffect } from 'react';
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
  ToggleButtonGroup,
  ToggleButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
} from '@mui/material';
import { Close, Settings, DeleteForever, Lock, DeleteSweep, RestartAlt } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import TeamMembershipEditor from './TeamMembershipEditor';
import LineReferenceSection from './LineReferenceSection';
import LineDataWorkspaceSection from './LineDataWorkspaceSection';
import ResizableDrawer from './ResizableDrawer';

export default function LineSettingsDrawer({ open, onClose, lineId, onDeleted }) {
  const queryClient = useQueryClient();

  const { data: team, isLoading } = useQuery(
    ['agentFactoryTeam', lineId],
    () => apiService.agentFactory.getLine(lineId),
    { enabled: open && !!lineId }
  );
  const { data: orgChart = [] } = useQuery(
    ['agentFactoryOrgChart', lineId],
    () => apiService.agentFactory.getLineOrgChart(lineId),
    { enabled: open && !!lineId }
  );

  const [name, setName] = useState('');
  const [handle, setHandle] = useState('');
  const [description, setDescription] = useState('');
  const [missionStatement, setMissionStatement] = useState('');
  const [status, setStatus] = useState('active');
  const [heartbeatEnabled, setHeartbeatEnabled] = useState(false);
  const [intervalSeconds, setIntervalSeconds] = useState('');
  const [cronExpression, setCronExpression] = useState('');
  const [governanceDescription, setGovernanceDescription] = useState('');
  const [monthlyLimit, setMonthlyLimit] = useState('');
  const [enforceHardLimit, setEnforceHardLimit] = useState(true);
  const [warningThresholdPct, setWarningThresholdPct] = useState('80');
  const [teamToolPacks, setTeamToolPacks] = useState([]);
  const [teamSkillIds, setTeamSkillIds] = useState([]);

  const { data: toolPacksList = [] } = useQuery(
    ['agentFactoryToolPacks'],
    () => apiService.agentFactory.getToolPacks(),
    { enabled: !!lineId }
  );
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
      setIntervalSeconds(hb.interval_seconds != null ? String(hb.interval_seconds) : '');
      setCronExpression(hb.cron_expression ?? '');
      const gov = team.governance_policy || {};
      setGovernanceDescription(gov.description ?? gov.policy_description ?? '');
      const bc = team.budget_config || {};
      setMonthlyLimit(bc.monthly_limit_usd != null ? String(bc.monthly_limit_usd) : '');
      setEnforceHardLimit(bc.enforce_hard_limit !== false);
      setWarningThresholdPct(bc.warning_threshold_pct != null ? String(bc.warning_threshold_pct) : '80');
      const raw = Array.isArray(team.team_tool_packs) ? team.team_tool_packs : [];
      setTeamToolPacks(raw.map((e) =>
        typeof e === 'object' && e && e.pack
          ? { pack: e.pack, mode: e.mode === 'read' ? 'read' : 'full' }
          : { pack: String(e).trim(), mode: 'full' }
      ).filter((e) => e.pack));
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
      },
    }
  );

  const deleteMutation = useMutation(
    () => apiService.agentFactory.deleteLine(lineId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries('agentFactoryLines');
        onClose?.();
        onDeleted?.();
      },
    }
  );

  const handleSaveGeneral = (e) => {
    e.preventDefault();
    updateMutation.mutate({
      name: name.trim(),
      handle: handle.trim() || null,
      description: description.trim() || null,
      mission_statement: missionStatement.trim() || null,
      status,
    });
  };

  const handleSaveBudget = (e) => {
    e.preventDefault();
    const budget_config = {
      monthly_limit_usd: monthlyLimit ? parseFloat(monthlyLimit) : undefined,
      enforce_hard_limit: enforceHardLimit,
      warning_threshold_pct: warningThresholdPct ? parseInt(warningThresholdPct, 10) : 80,
    };
    updateMutation.mutate({ budget_config });
  };

  const handleSaveHeartbeat = (e) => {
    e.preventDefault();
    const heartbeat_config = {
      enabled: heartbeatEnabled,
      interval_seconds: intervalSeconds ? parseInt(intervalSeconds, 10) : undefined,
      cron_expression: cronExpression.trim() || undefined,
    };
    updateMutation.mutate({ heartbeat_config });
  };

  const handleSaveGovernance = (e) => {
    e.preventDefault();
    updateMutation.mutate({
      governance_policy: { description: governanceDescription.trim() || undefined },
    });
  };

  const handleSaveToolsAndSkills = (e) => {
    e.preventDefault();
    updateMutation.mutate({
      team_tool_packs: teamToolPacks,
      team_skill_ids: teamSkillIds,
    });
  };

  const handleSaveReferences = (referenceConfig) => {
    updateMutation.mutate({ reference_config: referenceConfig });
  };

  const handleSaveDataWorkspaces = (dataWorkspaceConfig) => {
    updateMutation.mutate({ data_workspace_config: dataWorkspaceConfig });
  };

  const getPackEntry = (packName) => teamToolPacks.find((e) => e.pack === packName);

  const toggleToolPack = (packName, hasWriteTools) => {
    setTeamToolPacks((prev) => {
      const idx = prev.findIndex((e) => e.pack === packName);
      if (idx >= 0) return prev.filter((_, i) => i !== idx);
      return [...prev, { pack: packName, mode: hasWriteTools ? 'full' : 'full' }];
    });
  };

  const setPackMode = (packName, mode) => {
    setTeamToolPacks((prev) => {
      const next = prev.map((e) => (e.pack === packName ? { ...e, mode } : e));
      if (next.some((e) => e.pack === packName)) return next;
      return [...prev, { pack: packName, mode }];
    });
  };

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

  const handleResetTeam = () => {
    setResetLoading(true);
    apiService.agentFactory.resetLine(lineId).then(() => {
      setResetConfirmOpen(false);
      queryClient.invalidateQueries(['agentFactoryTeam', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimeline', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTimelineRecent', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamTasks', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamGoals', lineId]);
      queryClient.invalidateQueries(['agentFactoryTeamWorkspace', lineId]);
    }).finally(() => setResetLoading(false));
  };

  const handleDelete = () => {
    if (window.confirm('Delete this team and all its members, messages, goals, and tasks? This cannot be undone.')) {
      deleteMutation.mutate();
    }
  };

  if (!open) return null;

  const saving = updateMutation.isLoading;
  const lastBeat = team?.last_beat_at ? new Date(team.last_beat_at).toLocaleString() : '—';
  const nextBeat = team?.next_beat_at ? new Date(team.next_beat_at).toLocaleString() : '—';

  return (
    <ResizableDrawer
      open={open}
      onClose={onClose}
      storageKey="agent-factory-line-settings-drawer-width"
      defaultWidth={480}
      minWidth={360}
      maxWidth={900}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
        <Typography variant="h6">Line settings{team?.name ? `: ${team.name}` : ''}</Typography>
        <IconButton onClick={onClose} aria-label="Close">
          <Close />
        </IconButton>
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
          <Box component="form" onSubmit={handleSaveGeneral}>
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
              helperText="Use in chat: @handle to get a team summary (e.g. @political-tracker what's new?)"
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
            <Box sx={{ mt: 1 }}>
              <Button type="submit" variant="contained" size="small" disabled={!name.trim() || saving}>
                Save general
              </Button>
            </Box>
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
            Tools &amp; Skills
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
            Tool packs and skills applied to all team members when they run. team_tools are always available in team context.
          </Typography>
          <Box component="form" onSubmit={handleSaveToolsAndSkills}>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1, mb: 0.5 }}>
              Team tool packs
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 2 }}>
              {(toolPacksList || []).map((pack) => {
                const entry = getPackEntry(pack.name);
                const selected = !!entry;
                const hasWriteTools = pack.has_write_tools === true;
                return (
                  <Box key={pack.name} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
                    <Chip
                      label={pack.name}
                      size="small"
                      onClick={() => toggleToolPack(pack.name, hasWriteTools)}
                      color={selected ? 'primary' : 'default'}
                      variant={selected ? 'filled' : 'outlined'}
                      sx={{ cursor: 'pointer' }}
                    />
                    {selected && hasWriteTools && (
                      <ToggleButtonGroup
                        value={entry.mode || 'full'}
                        exclusive
                        size="small"
                        onChange={(_, v) => v != null && setPackMode(pack.name, v)}
                        sx={{ ml: 0.25, '& .MuiToggleButton-root': { py: 0, px: 0.75, fontSize: '0.7rem' } }}
                      >
                        <ToggleButton value="read" aria-label="Read only">Read</ToggleButton>
                        <ToggleButton value="full" aria-label="Full access">Full</ToggleButton>
                      </ToggleButtonGroup>
                    )}
                  </Box>
                );
              })}
            </Box>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mt: 1, mb: 0.5 }}>
              Team skills
            </Typography>
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
            <Box sx={{ mt: 1 }}>
              <Button type="submit" variant="contained" size="small" disabled={saving}>
                Save tools &amp; skills
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Reference files &amp; folders
          </Typography>
          <LineReferenceSection
            team={team}
            onSave={handleSaveReferences}
            saving={saving}
          />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Data Workspaces
          </Typography>
          <LineDataWorkspaceSection
            team={team}
            onSave={handleSaveDataWorkspaces}
            saving={saving}
          />
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Budget
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
            Team-level monthly cap. When enforced, heartbeat and agent invocations are blocked when over limit.
          </Typography>
          <Box component="form" onSubmit={handleSaveBudget}>
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
            <Box sx={{ mt: 1 }}>
              <Button type="submit" variant="contained" size="small" disabled={saving}>
                Save budget
              </Button>
            </Box>
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
            Turning this off pauses the line and stops background workers (same as dashboard Stop autonomous). Turning
            it on sets the line back to active. You can also use Stop autonomous / Activate autonomous on the team
            dashboard.
          </Typography>
          <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
            Last beat: {lastBeat} · Next beat: {nextBeat}
          </Typography>
          <Box component="form" onSubmit={handleSaveHeartbeat} sx={{ mt: 2 }}>
            <TextField
              fullWidth
              label="Interval (seconds)"
              type="number"
              value={intervalSeconds}
              onChange={(e) => setIntervalSeconds(e.target.value)}
              size="small"
              margin="dense"
              placeholder="e.g. 3600"
              inputProps={{ min: 60 }}
            />
            <TextField
              fullWidth
              label="Cron expression (optional, overrides interval)"
              value={cronExpression}
              onChange={(e) => setCronExpression(e.target.value)}
              size="small"
              margin="dense"
              placeholder="e.g. 0 */6 * * * for every 6 hours"
            />
            <Box sx={{ mt: 1 }}>
              <Button type="submit" variant="contained" size="small" disabled={saving}>
                Save heartbeat
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1 }}>
            Governance
          </Typography>
          <Box component="form" onSubmit={handleSaveGovernance}>
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
            <Box sx={{ mt: 1 }}>
              <Button type="submit" variant="contained" size="small" disabled={saving}>
                Save governance
              </Button>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Card variant="outlined" sx={{ borderColor: 'error.main' }}>
        <CardContent>
          <Typography variant="subtitle1" fontWeight={600} color="error" sx={{ mb: 1 }}>
            Danger zone
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Clear timeline: remove all messages. Reset team: clear timeline and all tasks; goals and settings are kept.
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {status === 'active' ? (
              <Button variant="outlined" color="warning" size="small" onClick={handlePause} disabled={saving}>
                Pause team
              </Button>
            ) : (
              <Button variant="outlined" color="primary" size="small" onClick={handleResume} disabled={saving}>
                Resume team
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
              Reset team
            </Button>
            <Button
              variant="outlined"
              color="error"
              size="small"
              startIcon={<DeleteForever />}
              onClick={handleDelete}
              disabled={deleteMutation.isLoading}
            >
              Delete team
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Dialog open={clearTimelineConfirmOpen} onClose={() => !clearTimelineLoading && setClearTimelineConfirmOpen(false)}>
        <DialogTitle>Clear timeline</DialogTitle>
        <DialogContent>
          <Typography>Remove all messages from this team&apos;s timeline? This cannot be undone.</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearTimelineConfirmOpen(false)} disabled={clearTimelineLoading}>Cancel</Button>
          <Button variant="contained" color="warning" onClick={handleClearTimeline} disabled={clearTimelineLoading}>
            {clearTimelineLoading ? 'Clearing…' : 'Clear timeline'}
          </Button>
        </DialogActions>
      </Dialog>
      <Dialog open={resetConfirmOpen} onClose={() => !resetLoading && setResetConfirmOpen(false)}>
        <DialogTitle>Reset team</DialogTitle>
        <DialogContent>
          <Typography>
            Clear all timeline messages and all tasks for this team? Goals, members, and settings are kept. This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setResetConfirmOpen(false)} disabled={resetLoading}>Cancel</Button>
          <Button variant="contained" color="warning" onClick={handleResetTeam} disabled={resetLoading}>
            {resetLoading ? 'Resetting…' : 'Reset team'}
          </Button>
        </DialogActions>
      </Dialog>
        </>
      )}
      </Box>
    </ResizableDrawer>
  );
}
