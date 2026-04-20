/**
 * Quick Team wizard: pick agents, name team, auto-create with members (CEO + workers).
 */

import React, { useState, useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  FormControlLabel,
  Checkbox,
  Stepper,
  Step,
  StepLabel,
} from '@mui/material';
import { GroupAdd } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import { invalidateAgentHandlesQuery } from '../../services/agentFactoryService';

const PATTERNS = [
  { id: 'general', label: 'General', mission: 'Collaborate and coordinate as a team.' },
  { id: 'discussion', label: 'Discussion Panel', mission: 'Hold structured multi-agent discussions and debates.' },
  { id: 'task_force', label: 'Task Force', mission: 'Execute tasks with clear roles and reporting.' },
  { id: 'research', label: 'Research Team', mission: 'Research topics and synthesize findings.' },
];

const ROLE_OPTIONS = [
  { value: 'ceo', label: 'CEO' },
  { value: 'manager', label: 'Manager' },
  { value: 'worker', label: 'Worker' },
  { value: 'specialist', label: 'Specialist' },
];

const GOVERNANCE_OPTIONS = [
  { value: 'hierarchical', label: 'Hierarchical (CEO + reports)' },
  { value: 'committee', label: 'Committee (flat; parallel heartbeats)' },
  { value: 'round_robin', label: 'Round-robin leader' },
  { value: 'consensus', label: 'Consensus (quorum actions)' },
];

export default function QuickTeamWizard({ open, onClose, onSuccess }) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState(0);
  const [selectedIds, setSelectedIds] = useState([]);
  const [patternId, setPatternId] = useState('general');
  const [teamName, setTeamName] = useState('');
  const [missionStatement, setMissionStatement] = useState('');
  const [roleOverrides, setRoleOverrides] = useState({});
  const [governanceMode, setGovernanceMode] = useState('hierarchical');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');

  const { data: profilesRaw } = useQuery(
    ['agentFactoryProfiles'],
    () => apiService.agentFactory.listProfiles(),
    { enabled: open }
  );

  const profiles = useMemo(() => {
    const raw = profilesRaw ?? [];
    return Array.isArray(raw) ? raw : (raw?.data ?? []);
  }, [profilesRaw]);

  const selectedProfiles = useMemo(() => {
    const idSet = new Set(selectedIds.map((id) => String(id)));
    return profiles.filter((p) => p && idSet.has(String(p.id)));
  }, [profiles, selectedIds]);

  const ceoId = useMemo(() => {
    const ordered = selectedProfiles.slice();
    const ceoCandidate = ordered.find((p) => roleOverrides[p.id] === 'ceo');
    if (ceoCandidate) return ceoCandidate.id;
    return ordered[0]?.id ?? null;
  }, [selectedProfiles, roleOverrides]);

  const toggleAgent = (id) => {
    const sid = id != null ? String(id) : '';
    if (!sid) return;
    setSelectedIds((prev) => {
      const prevSet = new Set(prev.map((x) => String(x)));
      if (prevSet.has(sid)) {
        return prev.filter((x) => String(x) !== sid);
      }
      return [...prev, sid];
    });
  };

  const handlePatternChange = (id) => {
    setPatternId(id);
    const p = PATTERNS.find((x) => x.id === id);
    if (p && !missionStatement) setMissionStatement(p.mission);
  };

  const canNextStep0 = selectedIds.length >= 2;
  const canSubmit =
    step === 1
    && teamName.trim()
    && selectedProfiles.length >= 2
    && (governanceMode === 'hierarchical' ? !!ceoId : true);

  const handleNext = () => {
    setError('');
    if (step === 0) {
      const pattern = PATTERNS.find((p) => p.id === patternId);
      if (pattern && !missionStatement) setMissionStatement(pattern.mission);
      setTeamName(
        selectedProfiles
          .map((p) => p.name || p.handle || p.id)
          .slice(0, 3)
          .join(', ')
      );
      setStep(1);
    }
  };

  const handleBack = () => {
    setError('');
    setStep(0);
  };

  const handleCreate = async () => {
    if (!canSubmit) return;
    setError('');
    setCreating(true);
    try {
      const gp0 =
        governanceMode === 'consensus'
          ? { quorum_pct: 60 }
          : {};
      const team = await apiService.agentFactory.createLine({
        name: teamName.trim(),
        description: '',
        mission_statement: missionStatement.trim() || undefined,
        status: 'active',
        governance_mode: governanceMode,
        governance_policy: gp0,
      });
      const teamId = team?.id;
      if (!teamId) throw new Error('Team creation did not return an id');

      if (governanceMode === 'hierarchical') {
        const ordered = [...selectedProfiles].sort((a, b) =>
          a.id === ceoId ? -1 : b.id === ceoId ? 1 : 0
        );
        let ceoMembershipId = null;
        for (let i = 0; i < ordered.length; i++) {
          const p = ordered[i];
          const isCeo = p.id === ceoId;
          const role = roleOverrides[p.id] || (isCeo ? 'ceo' : 'worker');
          const reportsTo = isCeo ? null : ceoMembershipId;
          const res = await apiService.agentFactory.addLineMember(teamId, {
            agent_profile_id: p.id,
            role,
            reports_to: reportsTo,
          });
          if (isCeo && res?.id) ceoMembershipId = res.id;
        }
      } else {
        for (const p of selectedProfiles) {
          await apiService.agentFactory.addLineMember(teamId, {
            agent_profile_id: p.id,
            role: roleOverrides[p.id] || 'worker',
            reports_to: null,
          });
        }
        const lineFull = await apiService.agentFactory.getLine(teamId);
        const mids = (lineFull?.members || []).map((m) => m.id).filter(Boolean);
        const gp = { ...(lineFull?.governance_policy || {}) };
        if (governanceMode === 'round_robin' && mids.length) {
          gp.rotation_order = mids;
          gp.current_leader_idx = 0;
        }
        if (governanceMode === 'consensus') {
          gp.quorum_pct = gp.quorum_pct ?? 60;
        }
        await apiService.agentFactory.updateLine(teamId, {
          governance_mode: governanceMode,
          governance_policy: gp,
        });
      }

      queryClient.invalidateQueries('agentFactoryTeams');
      queryClient.invalidateQueries('agentFactoryLines');
      invalidateAgentHandlesQuery(queryClient);
      if (onSuccess) onSuccess(team);
      onClose();
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setCreating(false);
    }
  };

  const handleClose = () => {
    if (!creating) {
      setStep(0);
      setSelectedIds([]);
      setPatternId('general');
      setTeamName('');
      setMissionStatement('');
      setRoleOverrides({});
      setGovernanceMode('hierarchical');
      setError('');
      onClose();
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Quick team</DialogTitle>
      <DialogContent>
        <Stepper activeStep={step} sx={{ pt: 0, pb: 2 }}>
          <Step completed={step > 0}>
            <StepLabel>Pick agents</StepLabel>
          </Step>
          <Step>
            <StepLabel>Name and confirm</StepLabel>
          </Step>
        </Stepper>

        {step === 0 && (
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Select at least 2 agents. The first will be CEO by default; you can change roles in the next step.
            </Typography>
            <FormControl fullWidth size="small" sx={{ mb: 2 }}>
              <InputLabel>Pattern</InputLabel>
              <Select
                value={patternId}
                label="Pattern"
                onChange={(e) => handlePatternChange(e.target.value)}
              >
                {PATTERNS.map((p) => (
                  <MenuItem key={p.id} value={p.id}>
                    {p.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Box
              sx={{
                maxHeight: 280,
                overflowY: 'auto',
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
              }}
            >
              <List dense disablePadding>
                {profiles.map((p) => {
                  const pid = p?.id != null ? String(p.id) : '';
                  if (!pid) return null;
                  const checked = selectedIds.some((x) => String(x) === pid);
                  return (
                    <ListItemButton
                      key={pid}
                      onClick={() => toggleAgent(pid)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={checked}
                            onChange={() => toggleAgent(pid)}
                            onClick={(e) => e.stopPropagation()}
                          />
                        }
                        label=""
                        onClick={(e) => e.stopPropagation()}
                      />
                      <ListItemText
                        primary={p.name || p.handle || p.id}
                        secondary={p.handle ? `@${p.handle}` : null}
                      />
                    </ListItemButton>
                  );
                })}
              </List>
            </Box>
            {selectedIds.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                {selectedIds.length} selected
              </Typography>
            )}
          </Box>
        )}

        {step === 1 && (
          <Box>
            <TextField
              fullWidth
              label="Team name"
              value={teamName}
              onChange={(e) => setTeamName(e.target.value)}
              size="small"
              margin="normal"
              required
            />
            <TextField
              fullWidth
              label="Mission statement"
              value={missionStatement}
              onChange={(e) => setMissionStatement(e.target.value)}
              size="small"
              margin="normal"
              multiline
              rows={2}
            />
            <FormControl fullWidth size="small" margin="normal">
              <InputLabel id="qt-gov-label">Governance</InputLabel>
              <Select
                labelId="qt-gov-label"
                label="Governance"
                value={governanceMode}
                onChange={(e) => setGovernanceMode(e.target.value)}
              >
                {GOVERNANCE_OPTIONS.map((g) => (
                  <MenuItem key={g.value} value={g.value}>
                    {g.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
              {governanceMode === 'hierarchical' ? 'Members and roles' : 'Members (flat; adjust hierarchy in line settings)'}
            </Typography>
            <List dense disablePadding>
              {selectedProfiles.map((p) => (
                <ListItem key={p.id} sx={{ py: 0 }}>
                  <ListItemText
                    primary={p.name || p.handle || p.id}
                    secondary={p.handle ? `@${p.handle}` : null}
                    primaryTypographyProps={{ variant: 'body2' }}
                  />
                  {governanceMode === 'hierarchical' ? (
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                      <Select
                        value={roleOverrides[p.id] ?? (p.id === ceoId ? 'ceo' : 'worker')}
                        onChange={(e) =>
                          setRoleOverrides((prev) => ({ ...prev, [p.id]: e.target.value }))
                        }
                        variant="outlined"
                        sx={{ height: 32 }}
                      >
                        {ROLE_OPTIONS.map((opt) => (
                          <MenuItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  ) : (
                    <Typography variant="caption" color="text.secondary">
                      worker
                    </Typography>
                  )}
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {error && (
          <Typography color="error" variant="body2" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose} disabled={creating}>
          Cancel
        </Button>
        {step === 1 && (
          <Button onClick={handleBack} disabled={creating}>
            Back
          </Button>
        )}
        {step === 0 ? (
          <Button variant="contained" onClick={handleNext} disabled={!canNextStep0}>
            Next
          </Button>
        ) : (
          <Button
            variant="contained"
            onClick={handleCreate}
            disabled={!canSubmit || creating}
          >
            {creating ? 'Creating…' : 'Create team'}
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
