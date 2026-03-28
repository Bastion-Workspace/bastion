/**
 * Agent Factory main area: section-card editor (Identity, Data Connections, Schedule, Watches).
 * Shown when an agent profile is selected.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Switch,
  FormControlLabel,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { PlayArrow, Delete, Download, Lock, Restore, DeleteSweep, Star, StarBorder } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import IdentitySection from './IdentitySection';
import DataSourcesSection from './DataSourcesSection';
import DataWorkspaceSection from './DataWorkspaceSection';
import ScheduleSection from './ScheduleSection';
import MonitorsSection from './MonitorsSection';
import BudgetSection from './BudgetSection';
import ExecutionHistoryCard from './ExecutionHistoryCard';

export default function AgentEditor({ profileId, onCloseEntityTab }) {
  const queryClient = useQueryClient();
  const [localProfile, setLocalProfile] = useState(null);
  const [localPlaybook, setLocalPlaybook] = useState(null);
  const [testQuery, setTestQuery] = useState('');
  const [testResult, setTestResult] = useState('');
  const [testLoading, setTestLoading] = useState(false);
  const [deleteAgentConfirmOpen, setDeleteAgentConfirmOpen] = useState(false);
  const [saveError, setSaveError] = useState(null);

  const { data: profile, isLoading: profileLoading, error: profileError } = useQuery(
    ['agentFactoryProfile', profileId],
    () => apiService.agentFactory.getProfile(profileId),
    { enabled: !!profileId, retry: false }
  );

  const { data: playbooks = [] } = useQuery(
    'agentFactoryPlaybooks',
    () => apiService.agentFactory.listPlaybooks(),
    { retry: false }
  );

  const { data: defaultChatPref } = useQuery(
    'defaultChatAgentProfile',
    () => apiService.settings.getDefaultChatAgentProfile(),
    { retry: false }
  );

  const setDefaultChatMutation = useMutation(
    (nextId) => apiService.settings.setDefaultChatAgentProfile(nextId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('defaultChatAgentProfile');
      },
    }
  );

  const { data: profileMemory = [], isLoading: memoryLoading } = useQuery(
    ['agentFactoryProfileMemory', profileId],
    () => apiService.agentFactory.getProfileMemory(profileId),
    { enabled: !!profileId, retry: false }
  );

  const clearMemoryMutation = useMutation(
    () => apiService.agentFactory.clearProfileMemory(profileId),
    {
      onSuccess: () => queryClient.invalidateQueries(['agentFactoryProfileMemory', profileId]),
    }
  );

  const currentProfile = localProfile ?? profile;

  const updateProfileMutation = useMutation(
    ({ id, body }) => apiService.agentFactory.updateProfile(id, body),
    {
      onSuccess: (data, { id }) => {
        setSaveError(null);
        if (data?.id) {
          queryClient.setQueryData(['agentFactoryProfile', id], data);
        } else {
          queryClient.invalidateQueries(['agentFactoryProfile', id]);
        }
        queryClient.invalidateQueries('agentFactoryProfiles');
        setLocalProfile(null);
      },
      onError: (err) => {
        const d = err?.response?.data?.detail;
        let message = err?.message || 'Failed to save profile';
        if (typeof d === 'string') {
          message = d;
        } else if (Array.isArray(d)) {
          message = d.map((x) => (typeof x === 'string' ? x : x?.msg || JSON.stringify(x))).join('; ');
        } else if (d && typeof d === 'object') {
          message = d.msg || JSON.stringify(d);
        }
        setSaveError(message);
      },
    }
  );

  const pauseProfileMutation = useMutation(
    (id) => apiService.agentFactory.pauseProfile(id),
    {
      onSuccess: (_, id) => {
        queryClient.invalidateQueries(['agentFactoryProfile', id]);
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('defaultChatAgentProfile');
        setLocalProfile((prev) => (prev ? { ...prev, is_active: false } : null));
      },
    }
  );
  const resumeProfileMutation = useMutation(
    (id) => apiService.agentFactory.resumeProfile(id),
    {
      onSuccess: (_, id) => {
        queryClient.invalidateQueries(['agentFactoryProfile', id]);
        queryClient.invalidateQueries('agentFactoryProfiles');
        queryClient.invalidateQueries('defaultChatAgentProfile');
        setLocalProfile((prev) => (prev ? { ...prev, is_active: true } : null));
      },
    }
  );

  const deleteProfileMutation = useMutation(
    (id) => apiService.agentFactory.deleteProfile(id),
    {
      onSuccess: (_, id) => {
        queryClient.invalidateQueries('agentFactoryProfiles');
        setDeleteAgentConfirmOpen(false);
        onCloseEntityTab?.('agent', id);
      },
    }
  );

  const resetDefaultsMutation = useMutation(
    (id) => apiService.agentFactory.resetProfileDefaults(id),
    {
      onSuccess: (data, id) => {
        queryClient.setQueryData(['agentFactoryProfile', id], data);
        queryClient.invalidateQueries('agentFactoryProfiles');
        setLocalProfile(null);
      },
    }
  );

  const saveTimeoutRef = useRef(null);
  const pendingProfileRef = useRef(null);

  // Reset local overlay state when switching profiles to prevent stale data leaking
  useEffect(() => {
    setLocalProfile(null);
    setSaveError(null);
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = null;
    }
    pendingProfileRef.current = null;
  }, [profileId]);

  const handleProfileChange = useCallback(
    (next) => {
      if (profile?.is_builtin) return;
      setLocalProfile(next);
      if (!profileId || !next) return;
      pendingProfileRef.current = next;
      if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
      saveTimeoutRef.current = setTimeout(() => {
        // Read latest pending profile from ref (not stale closure)
        const pending = pendingProfileRef.current;
        if (!pending) { saveTimeoutRef.current = null; return; }
        const body = {
          name: pending.name,
          handle: pending.handle,
          description: pending.description,
          model_preference: pending.model_preference,
          model_source: pending.model_source ?? undefined,
          model_provider_type: pending.model_provider_type ?? undefined,
          system_prompt_additions: pending.system_prompt_additions,
          default_playbook_id: pending.default_playbook_id ?? null,
          team_config: pending.team_config,
          watch_config: pending.watch_config,
          prompt_history_enabled: pending.prompt_history_enabled ?? pending.chat_history_enabled,
          chat_history_lookback: pending.chat_history_lookback,
          summary_threshold_tokens: pending.summary_threshold_tokens,
          summary_keep_messages: pending.summary_keep_messages,
          chat_visible: pending.chat_visible ?? true,
          persona_mode: pending.persona_mode || 'none',
          persona_id: pending.persona_mode === 'specific' ? (pending.persona_id || null) : null,
          include_user_context: pending.include_user_context,
          include_datetime_context: pending.include_datetime_context,
          include_user_facts: pending.include_user_facts,
          include_facts_categories: pending.include_facts_categories ?? [],
          include_agent_memory: pending.include_agent_memory,
          auto_routable: pending.auto_routable,
          data_workspace_config: pending.data_workspace_config ?? {},
        };
        updateProfileMutation.mutate({ id: profileId, body });
        pendingProfileRef.current = null;
        saveTimeoutRef.current = null;
      }, 600);
    },
    [profileId, profile?.is_builtin, updateProfileMutation]
  );

  useEffect(() => () => {
    if (saveTimeoutRef.current) clearTimeout(saveTimeoutRef.current);
  }, []);

  const runTest = useCallback(async () => {
    if (!profileId) return;
    setTestLoading(true);
    setTestResult('');
    const token = apiService.getToken?.();
    // Use a transient ID for Agent Factory runs so no "log" conversation is created in the sidebar.
    // Backend skips persisting when agent_profile_id is set; output is via channels / Recent Runs only.
    const conversationId = `agent-run-${crypto.randomUUID?.() ?? Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    try {
      const response = await fetch('/api/async/orchestrator/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          query: testQuery,
          conversation_id: conversationId,
          session_id: 'agent-factory-test',
          agent_profile_id: profileId,
        }),
      });
      if (!response.ok) {
        setTestResult(`HTTP ${response.status}: ${await response.text()}`);
        setTestLoading(false);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let full = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'content' && data.content) full += data.content;
              if (data.type === 'complete' && data.content) full += data.content;
            } catch (_) {}
          }
        }
      }
      setTestResult(full || '(No content in stream)');
      queryClient.invalidateQueries(['agentFactoryExecutions', profileId]);
    } catch (e) {
      setTestResult('Error: ' + (e.message || String(e)));
    }
    setTestLoading(false);
  }, [profileId, testQuery, queryClient]);

  if (!profileId) {
    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Select an agent from the list or create one to edit.
        </Typography>
      </Box>
    );
  }

  if (profileLoading || profileError) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        {profileError && (
          <Alert severity="error">
            Failed to load profile. It may have been deleted.
          </Alert>
        )}
        {profileLoading && <CircularProgress />}
      </Box>
    );
  }

  if (!currentProfile) return null;

  const isChatDefault =
    (!!defaultChatPref?.agent_profile_id && defaultChatPref.agent_profile_id === profileId) ||
    (!!currentProfile?.is_builtin && !defaultChatPref?.agent_profile_id);

  const handleActiveToggle = (e) => {
    const checked = e.target.checked;
    if (checked) resumeProfileMutation.mutate(profileId);
    else pauseProfileMutation.mutate(profileId);
  };

  const handleLockToggle = (e) => {
    const locked = e.target.checked;
    updateProfileMutation.mutate({ id: profileId, body: { is_locked: locked } });
    setLocalProfile((prev) => (prev ? { ...prev, is_locked: locked } : null));
  };

  const handleExportYaml = async () => {
    if (!profileId || !currentProfile) return;
    try {
      const res = await apiService.agentFactory.exportAgentBundle(profileId);
      const text = await res.text();
      const disp = res.headers.get('Content-Disposition');
      const match = disp && disp.match(/filename="?([^"]+)"?/);
      const filename = match ? match[1] : `${currentProfile.handle || 'agent'}.yaml`;
      const blob = new Blob([text], { type: 'application/x-yaml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Export failed:', e);
    }
  };

  return (
    <Box sx={{ p: 2, overflow: 'auto', maxWidth: 720, flex: 1, minHeight: 0, fontSize: '0.875rem' }}>
      {saveError && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setSaveError(null)}>
          {saveError}
        </Alert>
      )}
      {currentProfile?.is_builtin && (
        <Alert severity="info" sx={{ mb: 2 }}>
          This is the factory built-in agent: read-only here. Create a custom agent to customize behavior, then use
          {' '}<strong>Set as default for chat</strong> so new messages use it without @mention.
        </Alert>
      )}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 2,
          flexWrap: 'wrap',
          gap: 1,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, flex: '1 1 auto' }}>
          <Typography variant="h6" component="h2" noWrap sx={{ minWidth: 0 }}>
            {currentProfile.name || currentProfile.handle || 'Unnamed'}
          </Typography>
          {isChatDefault && (
            <Chip size="small" color="primary" icon={<Star fontSize="small" />} label="Default" />
          )}
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <FormControlLabel
            control={
              <Switch
                checked={!!currentProfile.is_locked || !!currentProfile.is_builtin}
                onChange={handleLockToggle}
                disabled={updateProfileMutation.isLoading || !!currentProfile?.is_builtin}
                color="primary"
              />
            }
            label={currentProfile.is_builtin ? 'Built-in' : (currentProfile.is_locked ? 'Locked' : 'Unlocked')}
            labelPlacement="start"
          />
          <FormControlLabel
            control={
              <Switch
                checked={!!currentProfile.is_active}
                onChange={handleActiveToggle}
                disabled={pauseProfileMutation.isLoading || resumeProfileMutation.isLoading}
                color="primary"
              />
            }
            label={currentProfile.is_active ? 'Active' : 'Paused'}
            labelPlacement="start"
          />
          {!currentProfile?.is_builtin && currentProfile?.is_active && (
            <>
              {defaultChatPref?.agent_profile_id === profileId ? (
                <Button
                  size="small"
                  variant="outlined"
                  color="warning"
                  startIcon={<StarBorder />}
                  onClick={() => setDefaultChatMutation.mutate(null)}
                  disabled={setDefaultChatMutation.isLoading}
                >
                  Remove as default for chat
                </Button>
              ) : (
                <Button
                  size="small"
                  variant="contained"
                  color="primary"
                  startIcon={<Star />}
                  onClick={() => setDefaultChatMutation.mutate(profileId)}
                  disabled={setDefaultChatMutation.isLoading}
                >
                  Set as default for chat
                </Button>
              )}
            </>
          )}
          <Button
            size="small"
            variant="outlined"
            startIcon={<Download />}
            onClick={handleExportYaml}
          >
            Export YAML
          </Button>
          {currentProfile?.is_builtin && (
            <Button
              size="small"
              variant="outlined"
              startIcon={<Restore />}
              onClick={() => resetDefaultsMutation.mutate(profileId)}
              disabled={resetDefaultsMutation.isLoading}
            >
              {resetDefaultsMutation.isLoading ? 'Resetting…' : 'Reset to defaults'}
            </Button>
          )}
          <Button
            size="small"
            color="error"
            variant="outlined"
            startIcon={<Delete />}
            onClick={() => setDeleteAgentConfirmOpen(true)}
            disabled={!!currentProfile.is_locked || !!currentProfile.is_builtin}
          >
            Delete
          </Button>
        </Box>
      </Box>
      <Dialog open={deleteAgentConfirmOpen} onClose={() => !deleteProfileMutation.isLoading && setDeleteAgentConfirmOpen(false)}>
        <DialogTitle>Delete agent</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete <strong>{currentProfile.name || currentProfile.handle || 'this agent'}</strong>?
            This will also remove its data connections, skills, playbook links, and execution history.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteAgentConfirmOpen(false)} disabled={deleteProfileMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => deleteProfileMutation.mutate(profileId)}
            disabled={deleteProfileMutation.isLoading}
          >
            {deleteProfileMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
      <IdentitySection
        profile={currentProfile}
        playbooks={playbooks}
        onChange={handleProfileChange}
        readOnly={!!currentProfile?.is_builtin || !!currentProfile?.is_locked}
      />

      <DataSourcesSection profileId={profileId} />

      <DataWorkspaceSection
        profile={currentProfile}
        onChange={handleProfileChange}
        readOnly={!!currentProfile?.is_builtin || !!currentProfile?.is_locked}
      />

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="subtitle1" gutterBottom>
            External tools (email, calendar, MCP)
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Connect accounts in Settings → External connections. In the Workflow Composer, open each LLM Agent or
            Deep Agent step to choose external tool packs and which connections apply to that step.
          </Typography>
        </CardContent>
      </Card>

      <ScheduleSection profileId={profileId} />

      <MonitorsSection
        profile={currentProfile}
        onChange={handleProfileChange}
        readOnly={!!currentProfile?.is_builtin || !!currentProfile?.is_locked}
      />

      <BudgetSection profileId={profileId} budget={currentProfile?.budget} />

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 1 }}>
            <Typography variant="h6">Agent memory</Typography>
            <Button
              size="small"
              variant="outlined"
              color="secondary"
              startIcon={clearMemoryMutation.isLoading ? <CircularProgress size={16} /> : <DeleteSweep />}
              disabled={!profileMemory?.length || clearMemoryMutation.isLoading}
              onClick={() => clearMemoryMutation.mutate()}
            >
              Clear memory
            </Button>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Key/value store for this agent (when &quot;Include agent memory&quot; is on, this is injected into each run).
          </Typography>
          {memoryLoading ? (
            <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress size={24} />
            </Box>
          ) : !profileMemory?.length ? (
            <Typography variant="body2" color="text.secondary">
              No memory entries yet.
            </Typography>
          ) : (
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>Key</TableCell>
                  <TableCell>Value</TableCell>
                  <TableCell>Updated</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {profileMemory.map((row) => (
                  <TableRow key={row.key}>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{row.key}</TableCell>
                    <TableCell sx={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {typeof row.value === 'object' ? JSON.stringify(row.value) : String(row.value)}
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
                      {row.updated_at ? new Date(row.updated_at).toLocaleString() : '—'}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <ExecutionHistoryCard profileId={profileId} />

      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Test
          </Typography>
          <TextField
            fullWidth
            multiline
            minRows={2}
            label="Query (optional)"
            value={testQuery}
            onChange={(e) => setTestQuery(e.target.value)}
            placeholder="Leave blank for tool-only runs; add text if steps use {query}"
            sx={{ mb: 2 }}
          />
          <Button
            variant="contained"
            startIcon={testLoading ? <CircularProgress size={20} /> : <PlayArrow />}
            disabled={testLoading}
            onClick={runTest}
            sx={{ mb: 2 }}
          >
            Run
          </Button>
          {testResult !== '' && (
            <Paper variant="outlined" sx={{ p: 2, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.875rem' }}>
              {testResult}
            </Paper>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}
