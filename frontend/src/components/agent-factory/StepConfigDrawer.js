/**
 * Configuration drawer for a single playbook step.
 * Shows step type, action, output key, and inputs with type-aware wiring dropdowns.
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import {
  Box,
  Typography,
  TextField,
  Button,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  ListSubheader,
  Divider,
  FormControlLabel,
  Checkbox,
  Alert,
  CircularProgress,
  Chip,
  Collapse,
  Autocomplete,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Switch,
  Tooltip,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material';
import { Close, Delete, Add, Extension, ExpandMore, ExpandLess, Lock, Lens, RadioButtonUnchecked, Schedule } from '@mui/icons-material';
import CollapsibleToolPicker from './CollapsibleToolPicker';
import ExternalPackConfigurator from './ExternalPackConfigurator';
import DeepAgentPhaseEditor from './DeepAgentPhaseEditor';
import IsolatedPromptTemplateField from './IsolatedPromptTemplateField';
import IsolatedDebouncedPlainPromptField from './IsolatedDebouncedPlainPromptField';
import { getCompatibleUpstreamOptions, getGroupedWireOptions, indexActionsByName, extractPromptPlaceholders } from '../../utils/agentFactoryTypeWiring';
import apiService from '../../services/apiService';
import { getSelectableChatModels } from '../../utils/chatSelectableModels';
import agentFactoryService from '../../services/agentFactoryService';
import ResizableDrawer from './ResizableDrawer';

const LITERAL = '__literal__';
const SCHEMA_TYPES = ['string', 'number', 'boolean', 'array', 'object'];

const NOTIFICATION_CHANNELS = [
  { value: 'in_app', label: 'In-app conversation' },
  { value: 'telegram', label: 'Telegram' },
  { value: 'discord', label: 'Discord' },
  { value: 'slack', label: 'Slack' },
  { value: 'email', label: 'Email' },
];

function contentRefOptionsFromSteps(steps, actions) {
  const options = [];
  if (!Array.isArray(steps) || !Array.isArray(actions)) return options;
  const byName = {};
  actions.forEach((a) => { const n = typeof a === 'string' ? a : a?.name; if (n) byName[n] = a; });
  steps.forEach((s) => {
    const key = s.output_key || s.name || s.action || '';
    if (!key) return;
    const action = byName[s.action];
    const fields = action?.output_fields || [];
    fields.forEach((f) => {
      options.push({ value: `${key}.${f.name}`, label: `${key}.${f.name}` });
    });
    options.push({ value: `${key}.formatted`, label: `${key}.formatted` });
  });
  return options;
}

function schemaPropertiesToFields(properties) {
  if (!properties || typeof properties !== 'object') return [];
  return Object.entries(properties).map(([name, def]) => ({
    name,
    type: (def && def.type) || 'string',
  }));
}

function fieldsToSchemaProperties(fields) {
  const properties = {};
  (fields || []).forEach(({ name, type }) => {
    if (name && type) properties[name] = { type };
  });
  return properties;
}

function LLMTaskModelOverride({ step, setStep }) {
  const { data: enabledData } = useQuery('enabledModels', () => apiService.getEnabledModels(), { staleTime: 300000 });
  const { data: availableData } = useQuery('availableModels', () => apiService.getAvailableModels(), { staleTime: 300000 });
  const { data: userProviders = [] } = useQuery('userLlmProviders', () => apiService.getUserLlmProviders(), { staleTime: 300000 });
  const enabledModels = enabledData?.enabled_models || [];
  const chatModels = getSelectableChatModels(enabledData);
  const current = step.model_override || '';
  
  // When user has configured their own LLM providers, filter to only user-sourced models
  const hasUserProviders = userProviders && userProviders.length > 0;
  let filteredChatModels = chatModels;
  if (hasUserProviders && availableData?.models) {
    const userModelIds = new Set(
      availableData.models
        .filter((m) => m.source === 'user')
        .map((m) => m.id)
    );
    filteredChatModels = chatModels.filter((id) => userModelIds.has(id));
    // If current override is not in filtered list but is a valid model, include it
    if (current && !filteredChatModels.includes(current) && enabledModels.includes(current)) {
      const modelInfo = availableData.models.find((m) => m.id === current);
      if (modelInfo?.source === 'user') {
        filteredChatModels = [current, ...filteredChatModels];
      }
    }
  }
  
  const options = current && !filteredChatModels.includes(current) && enabledModels.includes(current) 
    ? [current, ...filteredChatModels] 
    : filteredChatModels;
  const getModelLabel = (id) => availableData?.models?.find((m) => m.id === id)?.name || id;
  const getModelSourceTag = (id) => {
    const m = availableData?.models?.find((x) => x.id === id);
    if (!m?.source) return null;
    const sourceLabel = m.source === 'admin' ? 'Admin' : 'My providers';
    const provider = (m.provider_type || '').replace(/-/g, ' ');
    return provider ? `${sourceLabel} · ${provider}` : sourceLabel;
  };
  return (
    <FormControl fullWidth sx={{ mb: 2 }}>
      <InputLabel>Model override</InputLabel>
      <Select
        value={current}
        label="Model override"
        onChange={(e) => setStep((s) => ({ ...s, model_override: e.target.value || undefined }))}
      >
        <MenuItem value="">— Use agent default</MenuItem>
        {options.map((id) => (
          <MenuItem key={id} value={id}>
            {getModelLabel(id)}
            {getModelSourceTag(id) && (
              <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                · {getModelSourceTag(id)}
              </Typography>
            )}
          </MenuItem>
        ))}
      </Select>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
        Optional. Override the agent default model for this step.
      </Typography>
    </FormControl>
  );
}

function StepSkillsPicker({ step, setStep, readOnly }) {
  const { data: skillsList = [] } = useQuery(
    ['agentFactorySkills'],
    () => agentFactoryService.listSkills({ include_builtin: true }),
    { staleTime: 60000 }
  );
  const selectedIds = useMemo(
    () => new Set(Array.isArray(step?.skill_ids) ? step.skill_ids : Array.isArray(step?.skills) ? step.skills : []),
    [step?.skill_ids, step?.skills]
  );
  const byCategory = useMemo(() => {
    const map = {};
    (skillsList || []).forEach((s) => {
      const cat = s.category || 'General';
      if (!map[cat]) map[cat] = [];
      map[cat].push(s);
    });
    return map;
  }, [skillsList]);
  const categories = useMemo(() => Object.keys(byCategory).sort(), [byCategory]);

  const toggleSkill = (skillId) => {
    if (readOnly) return;
    const next = new Set(selectedIds);
    if (next.has(skillId)) next.delete(skillId);
    else next.add(skillId);
    setStep((s) => ({ ...s, skill_ids: Array.from(next) }));
  };

  if (categories.length === 0) return null;

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>Skills</Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Step-level skills add procedure and auto-bind required tools for this step.
      </Typography>
      <Box sx={{ maxHeight: 200, overflowY: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
        {categories.map((cat) => (
          <Box key={cat} sx={{ mb: 1 }}>
            <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              {cat}
            </Typography>
            {(byCategory[cat] || []).map((skill) => {
              const checked = selectedIds.has(skill.id);
              return (
                <FormControlLabel
                  key={skill.id}
                  control={
                    <Checkbox
                      size="small"
                      checked={checked}
                      disabled={readOnly}
                      onChange={() => toggleSkill(skill.id)}
                      sx={{ p: 0.25, mr: 0.5 }}
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
                      {skill.is_builtin && <Lock sx={{ fontSize: 14 }} color="action" titleAccess="Built-in" />}
                      <span>{skill.name || skill.slug}</span>
                      {Array.isArray(skill.required_tools) && skill.required_tools.length > 0 && (
                        <Tooltip title={skill.required_tools.join(', ')}>
                          <Chip
                            size="small"
                            label={`+${skill.required_tools.length} tools`}
                            sx={{ height: 18, fontSize: '0.7rem', cursor: 'help' }}
                            variant="outlined"
                          />
                        </Tooltip>
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
  );
}

function OutputSchemaEditor({ step, setStep }) {
  const [fields, setFields] = useState(() => schemaPropertiesToFields(step.output_schema?.properties));
  useEffect(() => {
    setFields(schemaPropertiesToFields(step.output_schema?.properties));
  }, [step.output_schema]);
  const persistSchema = (newFields) => {
    setStep((s) => ({
      ...s,
      output_schema: { type: 'object', properties: fieldsToSchemaProperties(newFields) },
    }));
  };
  const addField = () => {
    const next = [...fields, { name: '', type: 'string' }];
    setFields(next);
    persistSchema(next);
  };
  const removeField = (idx) => {
    const next = fields.filter((_, i) => i !== idx);
    setFields(next);
    persistSchema(next);
  };
  const changeField = (idx, key, value) => {
    const next = [...fields];
    if (key === 'name') next[idx] = { ...next[idx], name: value };
    else next[idx] = { ...next[idx], type: value };
    setFields(next);
    persistSchema(next);
  };
  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>Output schema</Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Expected JSON shape from the LLM (field name and type). Used for structured parsing.
      </Typography>
      {fields.map((f, idx) => (
        <Box key={idx} sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
          <TextField
            size="small"
            placeholder="Field name"
            value={f.name}
            onChange={(e) => changeField(idx, 'name', e.target.value)}
            sx={{ flex: 1, minWidth: 0 }}
          />
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <Select
              value={f.type}
              onChange={(e) => changeField(idx, 'type', e.target.value)}
              displayEmpty
            >
              {SCHEMA_TYPES.map((t) => (
                <MenuItem key={t} value={t}>{t}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <IconButton size="small" onClick={() => removeField(idx)} aria-label="Remove field">
            <Delete />
          </IconButton>
        </Box>
      ))}
      <Button size="small" startIcon={<Add />} onClick={addField}>Add field</Button>
    </Box>
  );
}

export default function StepConfigDrawer({
  open,
  onClose,
  step: initialStep,
  stepIndex,
  steps = [],
  stepPath = null,
  actions = [],
  playbookInputs = [],
  onSave,
  profileId,
  readOnly = false,
}) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState(initialStep || {});
  const [inputs, setInputs] = useState({});
  const [newInputKey, setNewInputKey] = useState('');
  const [newInputValue, setNewInputValue] = useState('');
  const [pluginCreds, setPluginCreds] = useState({});
  const [showPluginForm, setShowPluginForm] = useState(false);
  const [conditionSectionOpen, setConditionSectionOpen] = useState(false);
  const [insertRefValue, setInsertRefValue] = useState('');
  const [addAgentDialogOpen, setAddAgentDialogOpen] = useState(false);
  const [selectedProfileIdForAgent, setSelectedProfileIdForAgent] = useState('');
  const [selectedPlaybookIdForAgent, setSelectedPlaybookIdForAgent] = useState('');
  const promptDraftRef = useRef('');
  const approvalPromptDraftRef = useRef('');
  const [promptExternalRev, setPromptExternalRev] = useState(0);

  const drawerStepResetKey = useMemo(
    () => `${stepIndex ?? 'x'}|${open}|${JSON.stringify(stepPath ?? null)}`,
    [stepIndex, open, stepPath]
  );

  useEffect(() => {
    setPromptExternalRev(0);
  }, [drawerStepResetKey]);

  const promptFieldResetKey = useMemo(
    () => `${drawerStepResetKey}|${promptExternalRev}`,
    [drawerStepResetKey, promptExternalRev]
  );

  const actionsByName = useMemo(() => indexActionsByName(actions), [actions]);
  const upstreamSteps = useMemo(
    () => (stepIndex != null && stepIndex > 0 ? steps.slice(0, stepIndex) : []),
    [steps, stepIndex]
  );
  const actionNamesSet = useMemo(() => {
    const set = new Set();
    (actions || []).forEach((a) => {
      const name = typeof a === 'string' ? a : a?.name;
      if (name) set.add(name);
    });
    return set;
  }, [actions]);
  const retainedToolNames = useMemo(() => {
    if (step?.step_type !== 'llm_agent' && step?.type !== 'llm_agent' && step?.action !== 'llm_agent') return [];
    const current = Array.isArray(step?.available_tools) ? step.available_tools : [];
    return current.filter(
      (t) => typeof t === 'string' && !t.startsWith('agent:') && !actionNamesSet.has(t)
    );
  }, [step?.step_type, step?.action, step?.available_tools, actionNamesSet]);

  const { data: toolPacksList = [] } = useQuery(
    ['agentFactoryToolPacks'],
    () => apiService.agentFactory.getToolPacks(),
    { enabled: open }
  );

  const builtinToolPacks = useMemo(
    () => (toolPacksList || []).filter((p) => (p.type || 'builtin') !== 'external'),
    [toolPacksList]
  );
  const externalToolPacks = useMemo(
    () => (toolPacksList || []).filter((p) => p.type === 'external'),
    [toolPacksList]
  );

  const commitExternalToolPacks = useCallback(
    (nextExt) => {
      setStep((s) => {
        const extNames = new Set((externalToolPacks || []).map((p) => p.name));
        const raw = Array.isArray(s.tool_packs) ? s.tool_packs : [];
        const normalized = raw.map((e) =>
          typeof e === 'object' && e?.pack ? { ...e } : { pack: String(e).trim(), mode: 'full' }
        );
        const kept = normalized.filter((e) => !extNames.has(e.pack));
        return { ...s, tool_packs: [...kept, ...nextExt] };
      });
    },
    [externalToolPacks, setStep]
  );

  const toolsCoveredBySelectedPacks = useMemo(() => {
    const isLlmOrDeep = step?.step_type === 'llm_agent' || step?.type === 'llm_agent' || step?.action === 'llm_agent' ||
      step?.step_type === 'deep_agent' || step?.type === 'deep_agent' || step?.action === 'deep_agent';
    if (!isLlmOrDeep) return [];
    const raw = Array.isArray(step?.tool_packs) ? step.tool_packs : [];
    const packNames = raw.map((e) => (typeof e === 'object' && e?.pack ? e.pack : String(e).trim())).filter(Boolean);
    if (packNames.length === 0 || !Array.isArray(toolPacksList) || toolPacksList.length === 0) return [];
    const packToolSet = new Set();
    packNames.forEach((name) => {
      const pack = toolPacksList.find((p) => p.name === name);
      (pack?.tools || []).forEach((t) => {
        packToolSet.add(t);
        if (t.endsWith('_tool')) packToolSet.add(t.slice(0, -5));
      });
    });
    const current = Array.isArray(step?.available_tools) ? step.available_tools : [];
    return current.filter(
      (t) => typeof t === 'string' && !t.startsWith('agent:') && (packToolSet.has(t) || packToolSet.has(t + '_tool'))
    );
  }, [step?.step_type, step?.action, step?.tool_packs, step?.available_tools, toolPacksList]);
  const currentAction = step?.action ? actionsByName[step.action] : null;
  const inputFields = currentAction?.input_fields || [];
  const paramsSchema = currentAction?.params_schema;
  const isSendChannelMessage = step?.action === 'send_channel_message';
  const isInvokeAgent = step?.action === 'invoke_agent';

  const { data: agentHandles = [] } = useQuery(
    ['agentHandles'],
    () => agentFactoryService.fetchAgentHandles(),
    { enabled: open && !!isInvokeAgent, staleTime: 60000 }
  );

  const isLlmAgentStep = step?.step_type === 'llm_agent' || step?.type === 'llm_agent' || step?.action === 'llm_agent';
  const { data: profiles = [] } = useQuery(
    'agentFactoryProfiles',
    () => apiService.agentFactory.listProfiles(),
    { enabled: open && !!isLlmAgentStep, retry: false }
  );
  const { data: playbooks = [] } = useQuery(
    'agentFactoryPlaybooks',
    () => apiService.agentFactory.listPlaybooks(),
    { enabled: open && !!isLlmAgentStep, retry: false }
  );

  const selectedChannel = isSendChannelMessage
    ? (step?.params?.channel ?? paramsSchema?.properties?.channel?.default ?? 'telegram')
    : '';

  const { data: connectionsData } = useQuery(
    ['connections', 'chat_bot'],
    () => apiService.get('/api/connections?connection_type=chat_bot'),
    { enabled: open && !!isSendChannelMessage, staleTime: 60000 }
  );
  const chatBotConnections = connectionsData?.connections ?? [];

  const selectedConnectionId = step?.params?.connection_id ? String(step.params.connection_id) : '';
  const { data: knownChatsData } = useQuery(
    ['known-chats', selectedConnectionId],
    () => apiService.get(`/api/connections/${selectedConnectionId}/known-chats`),
    { enabled: open && !!isSendChannelMessage && !!selectedConnectionId, staleTime: 30000 }
  );
  const knownChats = knownChatsData?.known_chats ?? [];

  const pluginCategory = currentAction?.category;
  const pluginName = typeof pluginCategory === 'string' && pluginCategory.startsWith('plugin:')
    ? pluginCategory.replace(/^plugin:/, '')
    : null;

  const { data: plugins = [], isLoading: pluginsLoading } = useQuery(
    'agentFactoryPlugins',
    () => apiService.agentFactory.getPlugins(),
    { enabled: open && !!profileId, retry: false }
  );
  const { data: pluginConfigs = [] } = useQuery(
    ['agentFactoryPluginConfigs', profileId],
    () => apiService.agentFactory.listPluginConfigs(profileId),
    { enabled: open && !!profileId, retry: false }
  );
  const upsertPluginConfigsMutation = useMutation(
    ({ profileId: id, body }) => apiService.agentFactory.upsertPluginConfigs(id, body),
    {
      onSuccess: (_, { profileId: id }) => {
        queryClient.invalidateQueries(['agentFactoryPluginConfigs', id]);
        setShowPluginForm(false);
        setPluginCreds({});
      },
    }
  );

  const pluginMeta = pluginName ? plugins.find((p) => p.name === pluginName) : null;
  const pluginConfig = pluginName ? pluginConfigs.find((c) => c.plugin_name === pluginName) : null;
  const pluginConnected = !!(pluginConfig?.has_credentials);
  const pluginNeedsConfig = pluginMeta?.connection_requirements?.length > 0 && (!pluginConnected || showPluginForm);

  useEffect(() => {
    if (initialStep) {
      setStep(initialStep);
      setInputs(initialStep.inputs || {});
      const p = initialStep.prompt_template || initialStep.prompt || '';
      promptDraftRef.current = p;
      approvalPromptDraftRef.current = initialStep.prompt || '';
    }
    if (!open) {
      setShowPluginForm(false);
      setPluginCreds({});
    }
  }, [initialStep, open]);

  const commitPromptDraftToStep = useCallback((val) => {
    setStep((s) => ({ ...s, prompt_template: val, prompt: val }));
  }, []);

  const commitApprovalPromptToStep = useCallback((val) => {
    setStep((s) => ({ ...s, prompt: val }));
  }, []);

  const handleSave = () => {
    const stepTypeInner = step.step_type || step.type || 'tool';
    let merged = { ...step, inputs: { ...inputs } };
    if (stepTypeInner === 'llm_task' || stepTypeInner === 'llm_agent') {
      const p = promptDraftRef.current;
      merged = { ...merged, prompt_template: p, prompt: p };
    } else if (stepTypeInner === 'approval') {
      merged = { ...merged, prompt: approvalPromptDraftRef.current };
    }
    onSave(stepIndex, merged, stepPath);
    onClose();
  };

  const removeInput = (key) => {
    const next = { ...inputs };
    delete next[key];
    setInputs(next);
  };

  const handleInputChange = (key, value) => {
    const next = { ...inputs };
    if (value === '' || value == null) delete next[key];
    else next[key] = value;
    setInputs(next);
    if (stepType === 'llm_task' && key === 'context' && value && typeof value === 'string' && value.startsWith('{')) {
      const template = step?.prompt_template || step?.prompt || '';
      if (!template.includes('{context}')) {
        const nextTemplate =
          (step?.prompt_template || step?.prompt || '').trim() || 'Summarize or analyze the following:\n\n{context}';
        setStep((s) => ({ ...s, prompt_template: nextTemplate, prompt: nextTemplate }));
        promptDraftRef.current = nextTemplate;
        setPromptExternalRev((r) => r + 1);
      }
    }
  };

  const handleParamChange = (key, value) => {
    setStep((s) => ({
      ...s,
      params: { ...(s.params || {}), [key]: value === '' || value == null ? undefined : value },
    }));
  };

  const actionName = step.action || step.step_type || step.type || '';
  const stepType = step.step_type || step.type || 'tool';

  const promptPlaceholders = useMemo(
    () => ((stepType === 'llm_task' || stepType === 'llm_agent') ? extractPromptPlaceholders(step?.prompt_template || step?.prompt || '') : []),
    [stepType, step?.prompt_template, step?.prompt]
  );

  const llmDefaultInputKeys = useMemo(() => {
    if (stepType !== 'llm_task' && stepType !== 'llm_agent') return [];
    return ['context'];
  }, [stepType]);

  const inputKeys = useMemo(() => {
    const fromSchema = (currentAction?.input_fields || []).map((f) => f.name);
    const fromPrompt = (stepType === 'llm_task' || stepType === 'llm_agent') ? promptPlaceholders : [];
    const fromLlmDefault = (stepType === 'llm_task' || stepType === 'llm_agent') ? llmDefaultInputKeys : [];
    const combined = [];
    const seen = new Set();
    for (const k of [...fromSchema, ...fromLlmDefault, ...fromPrompt]) {
      if (!seen.has(k)) {
        seen.add(k);
        combined.push(k);
      }
    }
    const fromInputs = Object.keys(inputs).filter((k) => !seen.has(k));
    return [...combined, ...fromInputs];
  }, [currentAction?.input_fields, stepType, llmDefaultInputKeys, promptPlaceholders, inputs]);

  const actionOptions = actions.map((a) => (typeof a === 'string' ? a : a.name)).filter(Boolean);
  const toolOptions = useMemo(
    () =>
      actions
        .map((a) =>
          typeof a === 'string'
            ? { name: a, description: a, category: 'General' }
            : { name: a.name, description: a.short_description || a.description || a.name, category: a.category || 'General' }
        )
        .filter((o) => o.name),
    [actions]
  );
  const getInputFieldMeta = (key) => inputFields.find((f) => f.name === key);

  const getTargetTypeForInput = (inputName) => {
    const field = inputFields.find((f) => f.name === inputName);
    return field?.type || 'text';
  };

  const isPromptPlaceholder = (key) => (stepType === 'llm_task' || stepType === 'llm_agent') && promptPlaceholders.includes(key);

  // Prompt placeholder wiring: store in inputs (not template).
  // The pipeline executor resolves {context} via step_inputs automatically.

  const getWireOptionsForInput = (inputName) => {
    const targetType = getTargetTypeForInput(inputName);
    return getCompatibleUpstreamOptions(targetType, upstreamSteps, actionsByName);
  };

  const getGroupedOptionsForInput = (inputName) => {
    const targetType = getTargetTypeForInput(inputName);
    return getGroupedWireOptions(targetType, upstreamSteps, actionsByName, playbookInputs);
  };

  const contentRefOpts = useMemo(
    () => contentRefOptionsFromSteps(upstreamSteps, actions),
    [upstreamSteps, actions]
  );

  return (
    <ResizableDrawer
      open={open}
      onClose={onClose}
      storageKey="agent-factory-step-drawer-width"
      defaultWidth={400}
      minWidth={320}
      maxWidth={900}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 2, borderBottom: 1, borderColor: 'divider', flexShrink: 0 }}>
        <Typography variant="h6">Step configuration</Typography>
        <IconButton onClick={onClose} aria-label="Close">
          <Close />
        </IconButton>
      </Box>
      <Box sx={{ p: 2, overflow: 'auto', flex: 1, minHeight: 0 }}>
        {readOnly && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Playbook is locked. Unlock the playbook to edit steps.
          </Alert>
        )}
        <Box sx={readOnly ? { pointerEvents: 'none', opacity: 0.9 } : {}}>
        <FormControl fullWidth sx={{ mb: 2 }} disabled={readOnly}>
          <InputLabel>Step type</InputLabel>
          <Select
            value={stepType}
            label="Step type"
            onChange={(e) => setStep((s) => ({ ...s, step_type: e.target.value }))}
          >
            <MenuItem value="tool">Tool</MenuItem>
            <MenuItem value="llm_task">LLM task</MenuItem>
            <MenuItem value="llm_agent">LLM Agent</MenuItem>
            <MenuItem value="deep_agent">Deep Agent</MenuItem>
            <MenuItem value="approval">Approval</MenuItem>
            <MenuItem value="browser_authenticate">Browser Auth</MenuItem>
            <MenuItem value="loop">Loop</MenuItem>
            <MenuItem value="parallel">Parallel</MenuItem>
            <MenuItem value="branch">Branch</MenuItem>
          </Select>
        </FormControl>

        {stepType === 'tool' && (
          <Autocomplete
            size="small"
            options={toolOptions}
            groupBy={(opt) => opt.category}
            getOptionLabel={(opt) => (typeof opt === 'string' ? opt : opt.description || opt.name)}
            value={toolOptions.find((o) => o.name === actionName) || null}
            onChange={(_, v) => {
              setStep((s) => ({ ...s, action: v?.name ?? '' }));
              setShowPluginForm(false);
              setPluginCreds({});
            }}
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
            sx={{ mb: 2 }}
          />
        )}

        {stepType === 'tool' && isInvokeAgent && (
          <Box sx={{ mb: 2, p: 1.5, bgcolor: 'action.hover', borderRadius: 1 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Invoke Agent</Typography>
            <FormControl fullWidth size="small" sx={{ mb: 1 }}>
              <InputLabel>Agent to invoke</InputLabel>
              <Select
                value={step?.params?.agent_handle ?? ''}
                label="Agent to invoke"
                onChange={(e) => handleParamChange('agent_handle', e.target.value || undefined)}
                MenuProps={{ disableScrollLock: true }}
              >
                <MenuItem value="">—</MenuItem>
                {(agentHandles || []).map((h) => (
                  <MenuItem key={h.handle} value={h.handle}>
                    @{h.handle} {h.name ? `(${h.name})` : ''}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Typography variant="caption" color="text.secondary">
              Content wired to <strong>input_content</strong> is passed to the invoked agent as <code>{'{trigger_input}'}</code>.
            </Typography>
          </Box>
        )}

        {stepType === 'tool' && pluginName && (pluginsLoading || (pluginMeta?.connection_requirements?.length > 0) || pluginConnected) && (
          <Box sx={{ mb: 2, p: 1.5, bgcolor: 'action.hover', borderRadius: 1 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Integration: {pluginName}
            </Typography>
            {pluginsLoading ? (
              <Box sx={{ py: 1, display: 'flex', justifyContent: 'center' }}>
                <CircularProgress size={20} />
              </Box>
            ) : pluginNeedsConfig ? (
              <>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  {pluginConnected ? 'Update credentials for this integration.' : 'Connect this integration to use it in the playbook.'}
                </Typography>
                {(pluginMeta?.connection_requirements || []).map((req) => (
                  <TextField
                    key={req.key}
                    fullWidth
                    size="small"
                    type="password"
                    autoComplete="off"
                    label={req.label || req.key}
                    value={pluginCreds[req.key] ?? ''}
                    onChange={(e) =>
                      setPluginCreds((prev) => ({ ...prev, [req.key]: e.target.value }))
                    }
                    sx={{ mt: 1 }}
                  />
                ))}
                {upsertPluginConfigsMutation.isError && (
                  <Alert severity="error" sx={{ mt: 1 }}>
                    {upsertPluginConfigsMutation.error?.message || 'Failed to save'}
                  </Alert>
                )}
                <Box sx={{ display: 'flex', gap: 1, mt: 1.5 }}>
                  {pluginConnected && (
                    <Button size="small" onClick={() => { setShowPluginForm(false); setPluginCreds({}); }}>
                      Cancel
                    </Button>
                  )}
                  <Button
                    size="small"
                    variant="contained"
                    startIcon={<Extension />}
                    onClick={() => {
                      if (!profileId || !pluginMeta) return;
                      upsertPluginConfigsMutation.mutate({
                        profileId,
                        body: {
                          configs: [
                            {
                              plugin_name: pluginName,
                              credentials_encrypted: pluginCreds,
                              is_enabled: true,
                            },
                          ],
                        },
                      });
                    }}
                    disabled={
                      upsertPluginConfigsMutation.isLoading ||
                      (pluginMeta?.connection_requirements?.length > 0 && Object.keys(pluginCreds).length === 0)
                    }
                  >
                    {pluginConnected ? 'Update credentials' : 'Connect'}
                  </Button>
                </Box>
              </>
            ) : (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Chip size="small" label={`${pluginName} connected`} color="success" variant="outlined" />
                <Button size="small" onClick={() => setShowPluginForm(true)}>
                  Reconfigure
                </Button>
              </Box>
            )}
          </Box>
        )}

        <TextField
          fullWidth
          label="Output key"
          value={step.output_key || ''}
          onChange={(e) => setStep((s) => ({ ...s, output_key: e.target.value }))}
          placeholder="e.g. step_1"
          sx={{ mb: 2 }}
        />

        <TextField
          fullWidth
          type="number"
          label="Heading level (optional)"
          value={step.heading_level ?? ''}
          onChange={(e) => {
            const v = e.target.value;
            setStep((s) => ({
              ...s,
              heading_level: v === '' ? undefined : Math.max(1, Math.min(6, parseInt(v, 10) || 2)),
            }));
          }}
          placeholder="Default: 2"
          helperText="Section granularity for editor/ref location variables (# and ## by default). 1–6. First step that sets it applies to the whole run."
          inputProps={{ min: 1, max: 6, step: 1 }}
          sx={{ mb: 2 }}
        />

        <Box sx={{ mb: 2 }}>
          <Button
            fullWidth
            size="small"
            onClick={() => setConditionSectionOpen((o) => !o)}
            endIcon={conditionSectionOpen ? <ExpandLess /> : <ExpandMore />}
            sx={{ justifyContent: 'space-between', textTransform: 'none' }}
          >
            Condition (optional)
          </Button>
          <Collapse in={conditionSectionOpen}>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
              Step runs only when this expression is true. Use {'{step_name.field}'} for upstream values. Combine with AND or OR for multiple conditions. Operators: &gt;, &lt;, ==, !=, &gt;=, &lt;=; &quot;is defined&quot;, &quot;is not defined&quot;; &quot;matches&quot; (regex, e.g. {'{query}'} matches &quot;edit|fix&quot;); AND, OR.
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
              <TextField
                size="small"
                fullWidth
                label="Condition expression"
                placeholder="{search.count} > 0"
                value={step.condition || ''}
                onChange={(e) => setStep((s) => ({ ...s, condition: e.target.value || undefined }))}
                sx={{ minWidth: 200 }}
              />
              <FormControl size="small" sx={{ minWidth: 160 }}>
                <InputLabel>Insert reference</InputLabel>
                <Select
                  value={insertRefValue}
                  label="Insert reference"
                  onChange={(e) => {
                    const v = e.target.value;
                    setInsertRefValue('');
                    if (!v) return;
                    setStep((s) => ({ ...s, condition: ((s.condition || '') + (s.condition ? ' ' : '') + v).trim() }));
                  }}
                  MenuProps={{ disableScrollLock: true }}
                >
                  <MenuItem value="">—</MenuItem>
                  <ListSubheader>Upstream step outputs</ListSubheader>
                  {(() => {
                    const grouped = getGroupedWireOptions('text', upstreamSteps, actionsByName, playbookInputs || []);
                    return grouped.upstream.flatMap(({ options }) =>
                      options.map((opt) => <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>)
                    );
                  })()}
                  <ListSubheader>Playbook inputs</ListSubheader>
                  {(() => {
                    const grouped = getGroupedWireOptions('text', upstreamSteps, actionsByName, playbookInputs || []);
                    return (grouped.playbookInputs || []).map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                    ));
                  })()}
                  <ListSubheader>Runtime</ListSubheader>
                  {(() => {
                    const grouped = getGroupedWireOptions('text', upstreamSteps, actionsByName, playbookInputs || []);
                    return (grouped.runtime || []).map((opt) => {
                      const tip = opt.alwaysAvailable ? 'Always available' : opt.requiresOpenFile ? 'Requires open document' : opt.scheduleOnly ? 'Scheduled or webhook trigger only' : null;
                      const Icon = opt.alwaysAvailable ? Lens : opt.requiresOpenFile ? RadioButtonUnchecked : opt.scheduleOnly ? Schedule : null;
                      return (
                        <MenuItem key={opt.value} value={opt.value}>
                          {Icon && (
                            <Tooltip title={tip}>
                              <Icon sx={{ fontSize: 12, mr: 0.5, verticalAlign: 'middle' }} color="action" />
                            </Tooltip>
                          )}
                          {opt.label}
                        </MenuItem>
                      );
                    });
                  })()}
                </Select>
              </FormControl>
            </Box>
          </Collapse>
        </Box>

        {(stepType === 'llm_task' || stepType === 'llm_agent' || stepType === 'deep_agent' || stepType === 'tool') && (
          <Box sx={{ mb: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel id="user-facts-policy-label">User facts policy</InputLabel>
              <Select
                labelId="user-facts-policy-label"
                label="User facts policy"
                value={step.user_facts_policy || 'inherit'}
                disabled={readOnly}
                onChange={(e) => {
                  const v = e.target.value;
                  setStep((s) => {
                    const next = { ...s };
                    if (v === 'inherit') {
                      delete next.user_facts_policy;
                    } else {
                      next.user_facts_policy = v;
                    }
                    return next;
                  });
                }}
              >
                <MenuItem value="inherit">Inherit (use profile defaults)</MenuItem>
                <MenuItem value="no_write">No write — facts in context; cannot save</MenuItem>
                <MenuItem value="isolated">Isolated — omit facts from prompt; no fact tools</MenuItem>
              </Select>
            </FormControl>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
              Applies when the agent profile has user facts enabled. Step settings can only restrict, not expand.
            </Typography>
          </Box>
        )}

        {stepType === 'llm_task' && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Prompt template</Typography>
            <IsolatedPromptTemplateField
              resetKey={promptFieldResetKey}
              seedPrompt={step.prompt_template || step.prompt || ''}
              onCommit={commitPromptDraftToStep}
              promptDraftRef={promptDraftRef}
              label="Prompt template"
              minLines={3}
              upstreamSteps={upstreamSteps}
              playbookInputs={playbookInputs}
              actionsByName={actionsByName}
              placeholder="Use {step_name.field} for upstream values. Type { for variables."
            />
            <LLMTaskModelOverride step={step} setStep={setStep} />
            <OutputSchemaEditor step={step} setStep={setStep} />
            <StepSkillsPicker step={step} setStep={setStep} readOnly={readOnly} />
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={step.auto_discover_skills !== false}
                    disabled={readOnly}
                    onChange={(e) => setStep((s) => ({ ...s, auto_discover_skills: e.target.checked }))}
                  />
                }
                label="Auto-discover skills"
              />
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Automatically finds the most relevant skills based on the step&apos;s prompt.
              </Typography>
              {step.auto_discover_skills !== false && (
                <TextField
                  fullWidth
                  type="number"
                  label="Max auto-discovered skills"
                  value={step.max_auto_skills ?? 3}
                  onChange={(e) => {
                    const v = e.target.value;
                    setStep((s) => ({
                      ...s,
                      max_auto_skills: v === '' ? 3 : Math.max(1, Math.min(10, Number(v) || 3)),
                    }));
                  }}
                  inputProps={{ min: 1, max: 10, step: 1 }}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </Box>
          </>
        )}

        {stepType === 'llm_agent' && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Prompt template</Typography>
            <IsolatedPromptTemplateField
              resetKey={promptFieldResetKey}
              seedPrompt={step.prompt_template || step.prompt || ''}
              onCommit={commitPromptDraftToStep}
              promptDraftRef={promptDraftRef}
              label="Prompt template"
              minLines={3}
              upstreamSteps={upstreamSteps}
              playbookInputs={playbookInputs}
              actionsByName={actionsByName}
              placeholder="Use {step_name.field} for upstream values. Type { for variables."
            />
            <Typography variant="subtitle2" sx={{ mt: 1.5, mb: 0.5 }}>Tool packs</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1.5 }}>
              {(builtinToolPacks || []).map((pack) => {
                const raw = Array.isArray(step.tool_packs) ? step.tool_packs : [];
                const entries = raw.map((e) =>
                  typeof e === 'object' && e?.pack
                    ? { pack: e.pack, mode: e.mode === 'read' ? 'read' : 'full' }
                    : { pack: String(e).trim(), mode: 'full' }
                ).filter((e) => e.pack);
                const entry = entries.find((e) => e.pack === pack.name);
                const selected = !!entry;
                const hasWriteTools = pack.has_write_tools === true;
                return (
                  <Box key={pack.name} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
                    <Chip
                      label={pack.name}
                      size="small"
                      onClick={() => {
                        if (selected) {
                          setStep((s) => ({
                            ...s,
                            tool_packs: (s.tool_packs || []).filter((e) => (typeof e === 'object' && e?.pack ? e.pack : String(e)) !== pack.name),
                          }));
                        } else {
                          setStep((s) => {
                            const list = (Array.isArray(s.tool_packs) ? s.tool_packs : []).map((e) =>
                              typeof e === 'object' && e?.pack ? e : { pack: String(e), mode: 'full' }
                            );
                            return { ...s, tool_packs: [...list, { pack: pack.name, mode: 'full' }] };
                          });
                        }
                      }}
                      color={selected ? 'primary' : 'default'}
                      variant={selected ? 'filled' : 'outlined'}
                      sx={{ cursor: 'pointer' }}
                    />
                    {selected && hasWriteTools && (
                      <ToggleButtonGroup
                        value={entry?.mode || 'full'}
                        exclusive
                        size="small"
                        onChange={(_, v) => {
                          if (v == null) return;
                          setStep((s) => {
                            const list = (Array.isArray(s.tool_packs) ? s.tool_packs : []).map((e) =>
                              typeof e === 'object' && e?.pack ? e : { pack: String(e), mode: 'full' }
                            );
                            const idx = list.findIndex((e) => e.pack === pack.name);
                            const next = list.slice();
                            if (idx >= 0) next[idx] = { ...next[idx], mode: v };
                            else next.push({ pack: pack.name, mode: v });
                            return { ...s, tool_packs: next };
                          });
                        }}
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
            <ExternalPackConfigurator
              externalPacks={externalToolPacks}
              stepToolPacks={step.tool_packs}
              onCommit={commitExternalToolPacks}
              readOnly={readOnly}
            />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Invoke agents (as tools)</Typography>
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                {(Array.isArray(step.available_tools) ? step.available_tools : [])
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
                          const current = Array.isArray(step.available_tools) ? step.available_tools : [];
                          setStep((s) => ({
                            ...s,
                            available_tools: current.filter((t) => t !== agentKey),
                          }));
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
              >
                Add agent
              </Button>
            </Box>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Available tools</Typography>
            {retainedToolNames.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Also selected (not in current list): {retainedToolNames.join(', ')}
              </Typography>
            )}
            {toolsCoveredBySelectedPacks.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Already in selected packs: {toolsCoveredBySelectedPacks.join(', ')}
              </Typography>
            )}
            <Box sx={{ mb: 2 }}>
              <CollapsibleToolPicker
                actions={(actions || []).filter((a) => {
                  const name = typeof a === 'string' ? a : a?.name;
                  return name && !String(name).startsWith('agent:');
                })}
                selectedTools={(Array.isArray(step.available_tools) ? step.available_tools : []).filter(
                  (t) => typeof t !== 'string' || !t.startsWith('agent:')
                )}
                onToggleTool={(next) => {
                  const agentTools = (Array.isArray(step.available_tools) ? step.available_tools : []).filter(
                    (t) => typeof t === 'string' && t.startsWith('agent:')
                  );
                  const merged = [...next, ...agentTools];
                  const seen = new Set();
                  const deduped = merged.filter((t) => {
                    if (seen.has(t)) return false;
                    seen.add(t);
                    return true;
                  });
                  setStep((s) => ({ ...s, available_tools: deduped }));
                }}
              />
            </Box>
            <StepSkillsPicker step={step} setStep={setStep} readOnly={readOnly} />
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={step.auto_discover_skills !== false}
                    disabled={readOnly}
                    onChange={(e) => setStep((s) => ({ ...s, auto_discover_skills: e.target.checked }))}
                  />
                }
                label="Auto-discover skills"
              />
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                Automatically finds the most relevant skills based on the step&apos;s prompt.
              </Typography>
              {step.auto_discover_skills !== false && (
                <TextField
                  fullWidth
                  type="number"
                  label="Max auto-discovered skills"
                  value={step.max_auto_skills ?? 3}
                  onChange={(e) => {
                    const v = e.target.value;
                    setStep((s) => ({
                      ...s,
                      max_auto_skills: v === '' ? 3 : Math.max(1, Math.min(10, Number(v) || 3)),
                    }));
                  }}
                  inputProps={{ min: 1, max: 10, step: 1 }}
                  size="small"
                  sx={{ mt: 1 }}
                />
              )}
            </Box>
            <TextField
              fullWidth
              type="number"
              label="Max iterations (ReAct loop)"
              value={step.max_iterations ?? 3}
              onChange={(e) => {
                const v = e.target.value;
                setStep((s) => ({ ...s, max_iterations: v === '' ? 3 : Math.max(1, Math.min(25, Number(v) || 3)) }));
              }}
              inputProps={{ min: 1, max: 25, step: 1 }}
              sx={{ mb: 2 }}
            />
            <LLMTaskModelOverride step={step} setStep={setStep} />
            <OutputSchemaEditor step={step} setStep={setStep} />
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
                    const current = Array.isArray(step.available_tools) ? step.available_tools : [];
                    if (current.includes(agentKey)) {
                      setAddAgentDialogOpen(false);
                      return;
                    }
                    setStep((s) => ({ ...s, available_tools: [...current, agentKey] }));
                    setAddAgentDialogOpen(false);
                  }}
                  disabled={!selectedProfileIdForAgent}
                >
                  Add
                </Button>
              </DialogActions>
            </Dialog>
          </>
        )}

        {stepType === 'deep_agent' && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Tool packs</Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1.5 }}>
              {(builtinToolPacks || []).map((pack) => {
                const raw = Array.isArray(step.tool_packs) ? step.tool_packs : [];
                const entries = raw.map((e) =>
                  typeof e === 'object' && e?.pack
                    ? { pack: e.pack, mode: e.mode === 'read' ? 'read' : 'full' }
                    : { pack: String(e).trim(), mode: 'full' }
                ).filter((e) => e.pack);
                const entry = entries.find((e) => e.pack === pack.name);
                const selected = !!entry;
                const hasWriteTools = pack.has_write_tools === true;
                return (
                  <Box key={pack.name} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
                    <Chip
                      label={pack.name}
                      size="small"
                      onClick={() => {
                        if (selected) {
                          setStep((s) => ({
                            ...s,
                            tool_packs: (s.tool_packs || []).filter((e) => (typeof e === 'object' && e?.pack ? e.pack : String(e)) !== pack.name),
                          }));
                        } else {
                          setStep((s) => {
                            const list = (Array.isArray(s.tool_packs) ? s.tool_packs : []).map((e) =>
                              typeof e === 'object' && e?.pack ? e : { pack: String(e), mode: 'full' }
                            );
                            return { ...s, tool_packs: [...list, { pack: pack.name, mode: 'full' }] };
                          });
                        }
                      }}
                      color={selected ? 'primary' : 'default'}
                      variant={selected ? 'filled' : 'outlined'}
                      sx={{ cursor: 'pointer' }}
                    />
                    {selected && hasWriteTools && (
                      <ToggleButtonGroup
                        value={entry?.mode || 'full'}
                        exclusive
                        size="small"
                        onChange={(_, v) => {
                          if (v == null) return;
                          setStep((s) => {
                            const list = (Array.isArray(s.tool_packs) ? s.tool_packs : []).map((e) =>
                              typeof e === 'object' && e?.pack ? e : { pack: String(e), mode: 'full' }
                            );
                            const idx = list.findIndex((e) => e.pack === pack.name);
                            const next = list.slice();
                            if (idx >= 0) next[idx] = { ...next[idx], mode: v };
                            else next.push({ pack: pack.name, mode: v });
                            return { ...s, tool_packs: next };
                          });
                        }}
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
            <ExternalPackConfigurator
              externalPacks={externalToolPacks}
              stepToolPacks={step.tool_packs}
              onCommit={commitExternalToolPacks}
              readOnly={readOnly}
            />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Step-level tools (shared by all phases)</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Tools selected here are available to Act and Search phases. Phases can optionally restrict to a subset.
            </Typography>
            {toolsCoveredBySelectedPacks.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Already in selected packs: {toolsCoveredBySelectedPacks.join(', ')}
              </Typography>
            )}
            <Box sx={{ mb: 2 }}>
              <CollapsibleToolPicker
                actions={(actions || []).filter((a) => {
                  const name = typeof a === 'string' ? a : a?.name;
                  return name && !String(name).startsWith('agent:');
                })}
                selectedTools={Array.isArray(step.available_tools) ? step.available_tools : []}
                onToggleTool={(next) => setStep((s) => ({ ...s, available_tools: next }))}
              />
            </Box>
            <StepSkillsPicker step={step} setStep={setStep} readOnly={readOnly} />
            <Box sx={{ mb: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={step.auto_discover_skills !== false}
                    disabled={readOnly}
                    onChange={(e) => setStep((s) => ({ ...s, auto_discover_skills: e.target.checked }))}
                  />
                }
                label="Auto-discover skills"
              />
            </Box>
            <DeepAgentPhaseEditor
              step={step}
              setStep={setStep}
              actions={actions}
              stepPaletteTools={Array.isArray(step.available_tools) ? step.available_tools : []}
              readOnly={readOnly}
            />
            <TextField
              fullWidth
              label="Model override (optional)"
              value={step.model_override ?? ''}
              onChange={(e) => setStep((s) => ({ ...s, model_override: e.target.value || undefined }))}
              placeholder="e.g. anthropic/claude-3.5-sonnet"
              size="small"
              sx={{ mt: 2 }}
              disabled={readOnly}
            />
          </Box>
        )}

        {stepType === 'approval' && (
          <>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Preview from</InputLabel>
              <Select
                value={step.preview_from || ''}
                label="Preview from"
                onChange={(e) => setStep((s) => ({ ...s, preview_from: e.target.value || undefined }))}
              >
                <MenuItem value="">— None</MenuItem>
                {upstreamSteps.map((s, i) => {
                  const key = s.output_key || s.action || `step_${i + 1}`;
                  return <MenuItem key={key} value={key}>{key}</MenuItem>;
                })}
              </Select>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
                Which step output to show in the approval UI.
              </Typography>
            </FormControl>
            <IsolatedDebouncedPlainPromptField
              resetKey={drawerStepResetKey}
              seedValue={step.prompt || ''}
              onCommit={commitApprovalPromptToStep}
              draftRef={approvalPromptDraftRef}
              label="Prompt text"
              placeholder="Approve to continue?"
              minRows={2}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              type="number"
              label="Timeout (minutes)"
              value={step.timeout_minutes ?? ''}
              onChange={(e) => {
                const v = e.target.value;
                setStep((s) => ({ ...s, timeout_minutes: v === '' ? undefined : Math.max(0, Number(v)) }));
              }}
              placeholder="Optional"
              inputProps={{ min: 0, step: 1 }}
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>On reject</InputLabel>
              <Select
                value={step.on_reject || 'stop'}
                label="On reject"
                onChange={(e) => setStep((s) => ({ ...s, on_reject: e.target.value }))}
              >
                <MenuItem value="stop">Stop (default)</MenuItem>
                <MenuItem value="skip">Skip and continue</MenuItem>
                <MenuItem value="retry">Retry</MenuItem>
              </Select>
            </FormControl>
          </>
        )}

        {stepType === 'browser_authenticate' && (
          <>
            <TextField
              fullWidth
              size="small"
              label="Site domain"
              value={step.site_domain || ''}
              onChange={(e) => setStep((s) => ({ ...s, site_domain: e.target.value }))}
              placeholder="example.com"
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              size="small"
              label="Login URL"
              value={step.login_url || ''}
              onChange={(e) => setStep((s) => ({ ...s, login_url: e.target.value }))}
              placeholder="https://example.com/login"
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              size="small"
              label="Verify URL"
              value={step.verify_url || ''}
              onChange={(e) => setStep((s) => ({ ...s, verify_url: e.target.value }))}
              placeholder="https://example.com/dashboard"
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              size="small"
              label="Verify selector (optional)"
              value={step.verify_selector || ''}
              onChange={(e) => setStep((s) => ({ ...s, verify_selector: e.target.value }))}
              placeholder="#user-menu or .dashboard"
              sx={{ mb: 2 }}
            />
          </>
        )}

        {stepType === 'loop' && (
          <>
            <TextField
              fullWidth
              type="number"
              label="Max iterations"
              value={step.max_iterations ?? 3}
              onChange={(e) => {
                const v = e.target.value;
                setStep((s) => ({ ...s, max_iterations: v === '' ? 3 : Math.max(1, parseInt(v, 10) || 3) }));
              }}
              inputProps={{ min: 1, step: 1 }}
              sx={{ mb: 2 }}
            />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Steps in loop</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Child steps run repeatedly. Add tool, LLM task, output, or approval steps.
            </Typography>
            <List dense disablePadding sx={{ mb: 1 }}>
              {(step.steps || []).map((child, cIdx) => (
                <ListItem
                  key={cIdx}
                  secondaryAction={
                    <IconButton size="small" onClick={() => setStep((s) => ({ ...s, steps: (s.steps || []).filter((_, i) => i !== cIdx) }))} aria-label="Remove step">
                      <Delete />
                    </IconButton>
                  }
                  sx={{ py: 0.5, borderLeft: 2, borderColor: 'divider', pl: 1.5, mb: 0.5 }}
                >
                  <ListItemText
                    primary={child.name || child.output_key || child.action || `Step ${cIdx + 1}`}
                    secondary={child.step_type || 'tool'}
                  />
                </ListItem>
              ))}
            </List>
            <Button
              size="small"
              variant="outlined"
              startIcon={<Add />}
              onClick={() => setStep((s) => ({ ...s, steps: [...(s.steps || []), { step_type: 'tool', name: '', action: '', output_key: '' }] }))}
            >
              Add step to loop
            </Button>
          </>
        )}

        {stepType === 'parallel' && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Parallel sub-steps</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              All sub-steps run simultaneously. Each produces its own output. Configure a step by clicking it in the workflow.
            </Typography>
            <List dense disablePadding sx={{ mb: 1 }}>
              {(step.parallel_steps || []).map((child, cIdx) => (
                <ListItem
                  key={cIdx}
                  secondaryAction={
                    <IconButton size="small" onClick={() => setStep((s) => ({ ...s, parallel_steps: (s.parallel_steps || []).filter((_, i) => i !== cIdx) }))} aria-label="Remove step">
                      <Delete />
                    </IconButton>
                  }
                  sx={{ py: 0.5, borderLeft: 2, borderColor: 'divider', pl: 1.5, mb: 0.5 }}
                >
                  <ListItemText
                    primary={child.name || child.output_key || child.action || `Step ${String.fromCharCode(65 + cIdx)}`}
                    secondary={child.step_type || 'tool'}
                  />
                </ListItem>
              ))}
            </List>
            <Button
              size="small"
              variant="outlined"
              startIcon={<Add />}
              onClick={() => setStep((s) => ({ ...s, parallel_steps: [...(s.parallel_steps || []), { step_type: 'tool', name: '', action: '', output_key: '', inputs: {} }] }))}
            >
              Add to group
            </Button>
          </>
        )}

        {stepType === 'branch' && (
          <>
            <TextField
              fullWidth
              label="Condition (required)"
              placeholder="{classify.score} > 7"
              value={step.branch_condition || ''}
              onChange={(e) => setStep((s) => ({ ...s, branch_condition: e.target.value || undefined }))}
              helperText={'Expression that decides THEN vs ELSE. Use {step_name.field} for upstream values. Operators: >, <, ==, !=, is defined, "matches" (regex); AND, OR.'}
              sx={{ mb: 2 }}
            />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>THEN steps</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Run when condition is true. Configure by clicking a step in the workflow.
            </Typography>
            <List dense disablePadding sx={{ mb: 1 }}>
              {(step.then_steps || []).map((child, cIdx) => (
                <ListItem
                  key={cIdx}
                  secondaryAction={
                    <IconButton size="small" onClick={() => setStep((s) => ({ ...s, then_steps: (s.then_steps || []).filter((_, i) => i !== cIdx) }))} aria-label="Remove step">
                      <Delete />
                    </IconButton>
                  }
                  sx={{ py: 0.5, borderLeft: 2, borderColor: 'success.main', pl: 1.5, mb: 0.5 }}
                >
                  <ListItemText
                    primary={child.name || child.output_key || child.action || `Step ${cIdx + 1}`}
                    secondary={child.step_type || 'tool'}
                  />
                </ListItem>
              ))}
            </List>
            <Button
              size="small"
              variant="outlined"
              startIcon={<Add />}
              onClick={() => setStep((s) => ({ ...s, then_steps: [...(s.then_steps || []), { step_type: 'tool', name: '', action: '', output_key: '', inputs: {} }] }))}
              sx={{ mb: 2 }}
            >
              Add to THEN
            </Button>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>ELSE steps</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Run when condition is false. Can be empty.
            </Typography>
            <List dense disablePadding sx={{ mb: 1 }}>
              {(step.else_steps || []).map((child, cIdx) => (
                <ListItem
                  key={cIdx}
                  secondaryAction={
                    <IconButton size="small" onClick={() => setStep((s) => ({ ...s, else_steps: (s.else_steps || []).filter((_, i) => i !== cIdx) }))} aria-label="Remove step">
                      <Delete />
                    </IconButton>
                  }
                  sx={{ py: 0.5, borderLeft: 2, borderColor: 'divider', pl: 1.5, mb: 0.5 }}
                >
                  <ListItemText
                    primary={child.name || child.output_key || child.action || `Step ${cIdx + 1}`}
                    secondary={child.step_type || 'tool'}
                  />
                </ListItem>
              ))}
            </List>
            <Button
              size="small"
              variant="outlined"
              startIcon={<Add />}
              onClick={() => setStep((s) => ({ ...s, else_steps: [...(s.else_steps || []), { step_type: 'tool', name: '', action: '', output_key: '', inputs: {} }] }))}
            >
              Add to ELSE
            </Button>
          </>
        )}

        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle2" sx={{ mb: 1 }}>Inputs</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {(stepType === 'llm_task' || stepType === 'llm_agent')
            ? 'Wire "context" (and any {ref} in your prompt) to upstream step outputs below. Use {context} in the prompt to insert the wired content.'
            : stepType === 'approval'
              ? 'Wire from upstream steps (e.g. {step_1.formatted}) to use in the approval prompt, or add inputs manually.'
              : inputFields.length > 0
                ? 'Configure each input: wire from upstream steps (e.g. {step_1.formatted}) or enter a literal value.'
                : 'Wire from upstream steps (e.g. {search.formatted}) or enter a literal value.'}
        </Typography>
        {inputKeys.length > 0 && (
          <Box sx={{ mb: 2 }}>
            {inputKeys.map((key) => {
              const grouped = getGroupedOptionsForInput(key);
              const fieldMeta = getInputFieldMeta(key);
              const isPlaceholder = isPromptPlaceholder(key);
              const value = isPlaceholder ? (inputs[key] ?? `{${key}}`) : (inputs[key] ?? '');
              const isRef = typeof value === 'string' && value.startsWith('{') && value.endsWith('}');
              const selectedRef = isRef ? value : LITERAL;
              const targetType = getTargetTypeForInput(key);
              const isFromSchema = !!fieldMeta;
              const isRequired = fieldMeta?.required === true;
              return (
                <Box
                  key={key}
                  sx={{
                    mb: 2,
                    p: 1.5,
                    bgcolor: 'action.hover',
                    borderRadius: 1,
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" fontWeight={600}>
                      {key}
                      {isRequired && (
                        <Typography component="span" variant="caption" color="error.main" sx={{ ml: 0.5 }}>
                          (required)
                        </Typography>
                      )}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">({targetType})</Typography>
                    {!isFromSchema && !isPlaceholder && (
                      <IconButton size="small" onClick={() => removeInput(key)} aria-label="Remove input">
                        <Delete />
                      </IconButton>
                    )}
                    {isFromSchema && !isPlaceholder && (
                      <IconButton size="small" onClick={() => removeInput(key)} aria-label="Clear input" sx={{ visibility: value ? 'visible' : 'hidden' }}>
                        <Delete />
                      </IconButton>
                    )}
                    {isPlaceholder && (
                      <IconButton size="small" onClick={() => removeInput(key)} aria-label="Clear wiring" sx={{ visibility: inputs[key] ? 'visible' : 'hidden' }}>
                        <Delete />
                      </IconButton>
                    )}
                  </Box>
                  {fieldMeta?.description && (
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                      {fieldMeta.description}
                    </Typography>
                  )}
                  <FormControl size="small" fullWidth sx={{ mb: 1 }}>
                    <InputLabel>Wire from</InputLabel>
                    <Select
                      value={selectedRef}
                      label="Wire from"
                      onChange={(e) => {
                        const v = e.target.value;
                        if (v === LITERAL) handleInputChange(key, '');
                        else handleInputChange(key, v);
                      }}
                      MenuProps={{ disableScrollLock: true }}
                    >
                      <MenuItem value={LITERAL}>Literal value</MenuItem>
                      <ListSubheader sx={{ fontWeight: 600 }}>Upstream step outputs</ListSubheader>
                      {grouped.upstream.map(({ stepKey, options }) =>
                        options.map((opt) => (
                          <MenuItem key={`${stepKey}-${opt.value}`} value={opt.value} sx={{ fontWeight: opt.compatible ? 600 : 400 }}>
                            {opt.label}
                            {opt.type && (
                              <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                                {opt.type}{opt.compatible ? ' (compatible)' : ''}
                              </Typography>
                            )}
                          </MenuItem>
                        ))
                      )}
                      {grouped.playbookInputs.length > 0 && (
                        <>
                          <ListSubheader sx={{ fontWeight: 600 }}>Playbook inputs</ListSubheader>
                          {grouped.playbookInputs.map((opt) => (
                            <MenuItem key={opt.value} value={opt.value} sx={{ fontWeight: opt.compatible ? 600 : 400 }}>
                              {opt.label}
                              {opt.type && (
                                <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                                  {opt.type}
                                </Typography>
                              )}
                            </MenuItem>
                          ))}
                        </>
                      )}
                      <ListSubheader sx={{ fontWeight: 600 }}>Runtime variables</ListSubheader>
                      {grouped.runtime.map((opt) => {
                        const tip = opt.alwaysAvailable ? 'Always available' : opt.requiresOpenFile ? 'Requires open document' : opt.scheduleOnly ? 'Scheduled or webhook trigger only' : null;
                        const Icon = opt.alwaysAvailable ? Lens : opt.requiresOpenFile ? RadioButtonUnchecked : opt.scheduleOnly ? Schedule : null;
                        return (
                          <MenuItem key={opt.value} value={opt.value} sx={{ fontWeight: opt.compatible ? 600 : 400 }}>
                            {Icon && (
                              <Tooltip title={tip}>
                                <Icon sx={{ fontSize: 12, mr: 0.5, verticalAlign: 'middle' }} color="action" />
                              </Tooltip>
                            )}
                            {opt.label}
                            <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                              {opt.type}
                            </Typography>
                          </MenuItem>
                        );
                      })}
                      {selectedRef && selectedRef !== LITERAL && !grouped.upstream.some(({ options }) => options.some((o) => o.value === selectedRef)) && !grouped.playbookInputs.some((o) => o.value === selectedRef) && !grouped.runtime.some((o) => o.value === selectedRef) && (
                        <MenuItem value={selectedRef}>Current: {selectedRef}</MenuItem>
                      )}
                    </Select>
                  </FormControl>
                  <TextField
                    size="small"
                    fullWidth
                    value={selectedRef === LITERAL ? value : value}
                    onChange={(e) => selectedRef === LITERAL && handleInputChange(key, e.target.value)}
                    placeholder={
                      isPlaceholder && selectedRef === LITERAL
                        ? 'Enter literal (used in prompt)'
                        : selectedRef === LITERAL
                          ? (fieldMeta?.default !== undefined && fieldMeta?.default !== null ? `Optional. Default: ${fieldMeta.default}` : 'Enter value or leave empty for default')
                          : 'Wired from upstream'
                    }
                    disabled={selectedRef !== LITERAL}
                  />
                </Box>
              );
            })}
          </Box>
        )}
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
          <TextField
            size="small"
            label="Input name"
            placeholder="e.g. query"
            value={newInputKey}
            onChange={(e) => setNewInputKey(e.target.value)}
            sx={{ minWidth: 120 }}
          />
          <TextField
            size="small"
            label="Value or ref"
            placeholder="{step.field}"
            value={newInputValue}
            onChange={(e) => setNewInputValue(e.target.value)}
            sx={{ flex: 1, minWidth: 140 }}
          />
          <Button
            size="small"
            variant="outlined"
            disabled={!newInputKey.trim()}
            onClick={() => {
              const k = newInputKey.trim();
              if (k) {
                handleInputChange(k, newInputValue);
                setNewInputKey('');
                setNewInputValue('');
              }
            }}
          >
            Add
          </Button>
        </Box>

        {stepType === 'tool' && paramsSchema?.properties && Object.keys(paramsSchema.properties).length > 0 && (
          <>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Params</Typography>
            {isSendChannelMessage && selectedChannel === 'in_app' && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Creates an in-app conversation message for the current user (shows in the chat sidebar and notification bell).
              </Typography>
            )}
            {isSendChannelMessage && selectedChannel !== 'in_app' && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Sends using your configured bot. <strong>Recipient</strong>: leave empty to use the last chat that messaged the bot. Or set Telegram numeric chat_id / <strong>@channelname</strong> (public channels only), or Discord channel ID. For DMs to a person, they must message the bot first (then use their id or leave empty).
              </Typography>
            )}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, mb: 2 }}>
              {Object.entries(paramsSchema.properties).map(([propName, propSchema]) => {
                  const schemaType = propSchema?.type || 'string';
                  const description = propSchema?.description || '';
                  const defaultValue = propSchema?.default;
                  const currentValue = step.params?.[propName];
                  const value = currentValue !== undefined && currentValue !== '' ? currentValue : (defaultValue ?? '');

                  if (isSendChannelMessage && propName === 'channel') {
                    return (
                      <FormControl key={propName} size="small" fullWidth>
                        <InputLabel>Channel</InputLabel>
                        <Select
                          value={value || 'telegram'}
                          label="Channel"
                          onChange={(e) => handleParamChange(propName, e.target.value || 'telegram')}
                          MenuProps={{ disableScrollLock: true }}
                        >
                          <MenuItem value="in_app">In-app conversation</MenuItem>
                          <MenuItem value="telegram">Telegram</MenuItem>
                          <MenuItem value="discord">Discord</MenuItem>
                          <MenuItem value="email">Email</MenuItem>
                        </Select>
                        {description && (
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                            {description}
                          </Typography>
                        )}
                      </FormControl>
                    );
                  }

                  if (isSendChannelMessage && selectedChannel === 'in_app' && (propName === 'connection_id' || propName === 'recipient_chat_id')) {
                    return null;
                  }
                  if (isSendChannelMessage && selectedChannel === 'email' && (propName === 'connection_id' || propName === 'recipient_chat_id')) {
                    return null;
                  }
                  if (isSendChannelMessage && selectedChannel !== 'email' && (propName === 'to_email' || propName === 'from_source' || propName === 'subject')) {
                    return null;
                  }

                  if (isSendChannelMessage && propName === 'connection_id') {
                  return (
                    <FormControl key={propName} size="small" fullWidth>
                      <InputLabel>Connection</InputLabel>
                      <Select
                        value={value || ''}
                        label="Connection"
                        onChange={(e) => handleParamChange(propName, e.target.value || '')}
                        MenuProps={{ disableScrollLock: true }}
                      >
                        <MenuItem value="">Default (first for channel)</MenuItem>
                        {chatBotConnections.map((c) => (
                          <MenuItem key={c.id} value={String(c.id)}>
                            {c.provider === 'telegram' ? 'Telegram' : c.provider === 'discord' ? 'Discord' : c.provider} {c.display_name || c.account_identifier || ''} (ID: {c.id})
                          </MenuItem>
                        ))}
                      </Select>
                      {description && (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                          {description}
                        </Typography>
                      )}
                    </FormControl>
                  );
                }

                if (isSendChannelMessage && propName === 'recipient_chat_id' && selectedConnectionId) {
                  const recipientValue = value || '';
                  const selectValue = recipientValue === '' ? '__default__' : recipientValue;
                  const inKnown = knownChats.some((c) => String(c.chat_id) === String(recipientValue));
                  const labelFor = (chat) => {
                    const parts = [chat.chat_title || chat.chat_username || chat.chat_id];
                    if (chat.chat_id && (chat.chat_title || chat.chat_username)) parts.push(`(${chat.chat_id})`);
                    return parts.join(' ');
                  };
                  return (
                    <Box key={propName}>
                      <FormControl size="small" fullWidth sx={{ mb: 1 }}>
                        <InputLabel>Recipient</InputLabel>
                        <Select
                          value={selectValue}
                          label="Recipient"
                          onChange={(e) => handleParamChange(propName, e.target.value === '__default__' ? '' : e.target.value)}
                          MenuProps={{ disableScrollLock: true }}
                        >
                          <MenuItem value="__default__">Last chat (default)</MenuItem>
                          {knownChats.map((chat) => (
                            <MenuItem key={chat.chat_id} value={chat.chat_id}>
                              {labelFor(chat)}
                            </MenuItem>
                          ))}
                          {recipientValue && !inKnown && (
                            <MenuItem value={recipientValue}>Custom: {recipientValue}</MenuItem>
                          )}
                        </Select>
                      </FormControl>
                      <TextField
                        size="small"
                        fullWidth
                        placeholder="Or paste chat ID from /chatid"
                        value={recipientValue}
                        onChange={(e) => handleParamChange(propName, e.target.value.trim() || '')}
                        helperText="Send /chatid in Telegram to get the ID, then paste it here or pick from the list above once the chat has messaged the bot."
                      />
                    </Box>
                  );
                }

                if (isSendChannelMessage && propName === 'to_email' && selectedChannel === 'email') {
                  return (
                    <TextField
                      key={propName}
                      size="small"
                      fullWidth
                      label="To (email)"
                      value={value}
                      onChange={(e) => handleParamChange(propName, e.target.value)}
                      placeholder="Leave empty for your email"
                      helperText={description}
                    />
                  );
                }
                if (isSendChannelMessage && propName === 'from_source' && selectedChannel === 'email') {
                  return (
                    <FormControl key={propName} size="small" fullWidth>
                      <InputLabel>From</InputLabel>
                      <Select
                        value={value || 'system'}
                        label="From"
                        onChange={(e) => handleParamChange(propName, e.target.value || 'system')}
                        MenuProps={{ disableScrollLock: true }}
                      >
                        <MenuItem value="system">System (SMTP)</MenuItem>
                        <MenuItem value="user">My email account</MenuItem>
                      </Select>
                      {description && (
                        <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                          {description}
                        </Typography>
                      )}
                    </FormControl>
                  );
                }
                if (isSendChannelMessage && propName === 'subject' && selectedChannel === 'email') {
                  return (
                    <TextField
                      key={propName}
                      size="small"
                      fullWidth
                      label="Subject"
                      value={value}
                      onChange={(e) => handleParamChange(propName, e.target.value)}
                      placeholder="Leave empty to auto-generate from message"
                      helperText={description}
                    />
                  );
                }

                if (schemaType === 'boolean') {
                  return (
                    <FormControlLabel
                      key={propName}
                      control={
                        <Checkbox
                          checked={!!value}
                          onChange={(e) => handleParamChange(propName, e.target.checked)}
                        />
                      }
                      label={propName.replace(/_/g, ' ')}
                    />
                  );
                }
                if (schemaType === 'number' || schemaType === 'integer') {
                  return (
                    <TextField
                      key={propName}
                      size="small"
                      fullWidth
                      type="number"
                      label={propName.replace(/_/g, ' ')}
                      value={value}
                      onChange={(e) => handleParamChange(propName, e.target.value === '' ? '' : Number(e.target.value))}
                      placeholder={description}
                      helperText={description ? undefined : (defaultValue !== undefined ? `Default: ${defaultValue}` : '')}
                    />
                  );
                }
                return (
                  <TextField
                    key={propName}
                    size="small"
                    fullWidth
                    label={propName.replace(/_/g, ' ')}
                    value={value}
                    onChange={(e) => handleParamChange(propName, e.target.value)}
                    placeholder={description || (defaultValue !== undefined ? `Default: ${defaultValue}` : '')}
                    helperText={description}
                  />
                );
              })}
            </Box>
          </>
        )}

        </Box>
        <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end', mt: 2 }}>
          <Button onClick={onClose}>{readOnly ? 'Close' : 'Cancel'}</Button>
          {!readOnly && (
            <Button variant="contained" onClick={handleSave}>
              Save
            </Button>
          )}
        </Box>
      </Box>
    </ResizableDrawer>
  );
}
