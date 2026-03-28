/**
 * Inline panel for adding a playbook step (Tool / LLM Task / LLM Agent / Approval / etc).
 * Tools are grouped by category from the I/O registry; invoke_agent is a regular tool.
 * LLM Agent steps can add "Invoke agents" (agent:profile_id[:playbook_id]) as tools.
 */

import React, { useState } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Autocomplete,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { Add } from '@mui/icons-material';
import CollapsibleToolPicker from './CollapsibleToolPicker';
import apiService from '../../services/apiService';

const STEP_TYPE_OPTIONS = [
  { value: 'tool', label: 'Tool' },
  { value: 'llm_task', label: 'LLM Task' },
  { value: 'llm_agent', label: 'LLM Agent' },
  { value: 'deep_agent', label: 'Deep Agent' },
  { value: 'approval', label: 'Approval' },
  { value: 'browser_authenticate', label: 'Browser Auth' },
  { value: 'loop', label: 'Loop' },
  { value: 'parallel', label: 'Parallel' },
  { value: 'branch', label: 'Branch' },
];

export default function AddStepPanel({
  stepType,
  setStepType,
  action,
  setAction,
  outputKey,
  setOutputKey,
  actions = [],
  onAdd,
  onCancel,
  addDisabled,
  availableTools = [],
  setAvailableTools,
}) {
  const [addAgentDialogOpen, setAddAgentDialogOpen] = useState(false);
  const [selectedProfileIdForAgent, setSelectedProfileIdForAgent] = useState('');
  const [selectedPlaybookIdForAgent, setSelectedPlaybookIdForAgent] = useState('');

  const { data: profiles = [] } = useQuery(
    'agentFactoryProfiles',
    () => apiService.agentFactory.listProfiles(),
    { enabled: stepType === 'llm_agent', retry: false }
  );
  const { data: playbooks = [] } = useQuery(
    'agentFactoryPlaybooks',
    () => apiService.agentFactory.listPlaybooks(),
    { enabled: stepType === 'llm_agent', retry: false }
  );

  const toolOptions = React.useMemo(() => {
    const map = {};
    actions.forEach((a) => {
      const name = typeof a === 'string' ? a : a?.name;
      if (!name) return;
      const cat = (typeof a === 'object' && a?.category) ? a.category : 'General';
      if (!map[cat]) map[cat] = [];
      map[cat].push({ name, description: typeof a === 'object' ? (a.short_description || a.description || a.name) : a, category: cat });
    });
    return Object.keys(map).sort().flatMap((cat) => (map[cat] || []));
  }, [actions]);

  return (
    <Paper variant="outlined" sx={{ p: 2, mt: 2, bgcolor: 'action.hover' }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Add step
      </Typography>
      <FormControl fullWidth size="small" sx={{ mb: 2 }}>
        <InputLabel>Step type</InputLabel>
        <Select
          value={stepType}
          label="Step type"
          onChange={(e) => {
            const v = e.target.value;
            setStepType(v);
            if (v !== 'tool') setAction('');
            if (v !== 'llm_agent' && setAvailableTools) setAvailableTools([]);
          }}
          MenuProps={{ disableScrollLock: true }}
        >
          {STEP_TYPE_OPTIONS.map((opt) => (
            <MenuItem key={opt.value} value={opt.value}>
              {opt.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {stepType === 'tool' && (
          <Autocomplete
            size="small"
            options={toolOptions}
            groupBy={(opt) => opt.category}
            getOptionLabel={(opt) => (typeof opt === 'string' ? opt : opt.description || opt.name)}
            value={toolOptions.find((o) => o.name === action) || null}
            onChange={(_, v) => setAction(v?.name ?? '')}
            filterOptions={(opts, { inputValue }) => {
              const q = (inputValue || '').toLowerCase().trim();
              if (!q) return opts;
              return opts.filter(
                (o) =>
                  (o.name || '').toLowerCase().includes(q) ||
                  (o.description || '').toLowerCase().includes(q)
              );
            }}
            renderInput={(params) => (
              <TextField {...params} label="Tool" placeholder="Type to search..." />
            )}
            ListboxProps={{ sx: { maxHeight: 320 } }}
          />
        )}
        {stepType === 'llm_task' && (
          <Typography variant="body2" color="text.secondary">
            Add an LLM task step. Set the prompt and output schema in the step configuration after adding.
          </Typography>
        )}
        {stepType === 'llm_agent' && (
          <Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Select tools the LLM can choose from at runtime. Configure prompt and max iterations in step config.
            </Typography>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Invoke agents (as tools)</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
              {(Array.isArray(availableTools) ? availableTools : [])
                .filter((t) => typeof t === 'string' && t.startsWith('agent:'))
                .map((agentKey) => {
                  const parts = agentKey.split(':');
                  const profileId = parts[1] || '';
                  const playbookId = parts[2] || '';
                  const profile = profiles.find((p) => p.id === profileId);
                  const playbook = playbookId ? playbooks.find((p) => p.id === playbookId) : null;
                  const label = playbookId
                    ? `${profile?.name || profileId} (${playbook?.name || playbookId})`
                    : `${profile?.name || profileId} (default playbook)`;
                  return (
                    <Chip
                      key={agentKey}
                      label={label}
                      size="small"
                      onDelete={() => {
                        const current = Array.isArray(availableTools) ? availableTools : [];
                        setAvailableTools && setAvailableTools(current.filter((t) => t !== agentKey));
                      }}
                      sx={{ mb: 0.5 }}
                    />
                  );
                })}
            </Box>
            <Button
              size="small"
              startIcon={<Add />}
              onClick={() => {
                setSelectedProfileIdForAgent('');
                setSelectedPlaybookIdForAgent('');
                setAddAgentDialogOpen(true);
              }}
              sx={{ mb: 1 }}
            >
              Add agent
            </Button>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Tools</Typography>
            <CollapsibleToolPicker
              actions={(actions || []).filter((a) => {
                const name = typeof a === 'string' ? a : a?.name;
                return name && !String(name).startsWith('agent:');
              })}
              selectedTools={(Array.isArray(availableTools) ? availableTools : []).filter(
                (t) => typeof t !== 'string' || !t.startsWith('agent:')
              )}
              onToggleTool={(next) => {
                const agentTools = (Array.isArray(availableTools) ? availableTools : []).filter(
                  (t) => typeof t === 'string' && t.startsWith('agent:')
                );
                setAvailableTools && setAvailableTools([...next, ...agentTools]);
              }}
            />
            <Dialog open={addAgentDialogOpen} onClose={() => setAddAgentDialogOpen(false)} maxWidth="sm" fullWidth>
              <DialogTitle>Add agent as tool</DialogTitle>
              <DialogContent>
                <FormControl fullWidth sx={{ mt: 1, mb: 2 }}>
                  <InputLabel>Agent</InputLabel>
                  <Select
                    value={selectedProfileIdForAgent}
                    label="Agent"
                    onChange={(e) => {
                      setSelectedProfileIdForAgent(e.target.value || '');
                      setSelectedPlaybookIdForAgent('');
                    }}
                  >
                    <MenuItem value="">— Select —</MenuItem>
                    {(profiles || []).map((p) => (
                      <MenuItem key={p.id} value={p.id}>{p.name || p.handle || p.id}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth sx={{ mb: 1 }}>
                  <InputLabel>Playbook</InputLabel>
                  <Select
                    value={selectedPlaybookIdForAgent}
                    label="Playbook"
                    onChange={(e) => setSelectedPlaybookIdForAgent(e.target.value || '')}
                    disabled={!selectedProfileIdForAgent}
                  >
                    <MenuItem value="">Default (agent&apos;s default playbook)</MenuItem>
                    {(playbooks || []).map((p) => (
                      <MenuItem key={p.id} value={p.id}>{p.name || p.id}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setAddAgentDialogOpen(false)}>Cancel</Button>
                <Button
                  variant="contained"
                  onClick={() => {
                    if (!selectedProfileIdForAgent) return;
                    const agentKey = selectedPlaybookIdForAgent
                      ? `agent:${selectedProfileIdForAgent}:${selectedPlaybookIdForAgent}`
                      : `agent:${selectedProfileIdForAgent}`;
                    const current = Array.isArray(availableTools) ? availableTools : [];
                    if (current.includes(agentKey)) {
                      setAddAgentDialogOpen(false);
                      return;
                    }
                    setAvailableTools && setAvailableTools([...current, agentKey]);
                    setAddAgentDialogOpen(false);
                  }}
                  disabled={!selectedProfileIdForAgent}
                >
                  Add
                </Button>
              </DialogActions>
            </Dialog>
          </Box>
        )}
        {stepType === 'approval' && (
          <Typography variant="body2" color="text.secondary">
            Add an approval gate. Configure prompt and timeout in the step configuration after adding.
          </Typography>
        )}
        {stepType === 'browser_authenticate' && (
          <Typography variant="body2" color="text.secondary">
            Open a browser session and verify or capture login for a site. Set site_domain, login_url, verify_url and optional verify_selector in step config.
          </Typography>
        )}
        {stepType === 'parallel' && (
          <Typography variant="body2" color="text.secondary">
            Creates a parallel group. Add child steps after creating in the step configuration drawer.
          </Typography>
        )}
        {stepType === 'branch' && (
          <Typography variant="body2" color="text.secondary">
            If/else branch. Set the condition and THEN/ELSE steps in the step configuration after adding.
          </Typography>
        )}
        <TextField
          size="small"
          fullWidth
          label="Output key"
          placeholder="e.g. step_1"
          value={outputKey}
          onChange={(e) => setOutputKey(e.target.value)}
        />
        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
          <Button size="small" onClick={onCancel}>Cancel</Button>
          <Button size="small" variant="contained" onClick={onAdd} disabled={addDisabled}>
            Add
          </Button>
        </Box>
      </Box>
    </Paper>
  );
}
