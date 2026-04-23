/**
 * Agent Factory Identity section card: name, handle, icon, execution mode, model, system prompt, default playbook.
 * Model preference is a dropdown of admin-enabled models (or, in future, user-enabled if they have API keys).
 */

import React from 'react';
import { useQuery } from 'react-query';
import {
  Card,
  CardContent,
  Typography,
  TextField,
  FormControl,
  FormControlLabel,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Switch,
  Autocomplete,
  Chip,
} from '@mui/material';
import apiService from '../../services/apiService';
import { useAuth } from '../../contexts/AuthContext';
import { getSelectableChatModels } from '../../utils/chatSelectableModels';

export default function IdentitySection({
  profile,
  playbooks = [],
  onChange,
  readOnly = false,
}) {
  const { user, loading: authLoading } = useAuth();
  const { data: enabledData } = useQuery(
    ['enabledModels', user?.user_id],
    () => apiService.getEnabledModels(),
    { staleTime: 300000, enabled: !!(user?.user_id && !authLoading) }
  );
  const { data: availableData } = useQuery(
    ['availableModels', user?.user_id],
    () => apiService.getAvailableModels(),
    { staleTime: 300000, enabled: !!(user?.user_id && !authLoading) }
  );
  const { data: personasData } = useQuery(
    'personas',
    () => apiService.settings.getPersonas(),
    { staleTime: 60000 }
  );
  const personasList = personasData?.personas || [];

  const defaultPlaybookId = profile?.default_playbook_id || '';

  if (!profile) return null;

  const handleField = (field) => (e) => {
    const value = e.target.value;
    onChange({ ...profile, [field]: value === '' ? null : value });
  };

  const enabledModels = enabledData?.enabled_models || [];
  const chatModels = getSelectableChatModels(enabledData);
  const currentPref = profile.model_preference || '';
  const modelOptions = currentPref && !chatModels.includes(currentPref)
    ? [currentPref, ...chatModels]
    : chatModels;
  const getModelLabel = (id) => availableData?.models?.find((m) => m.id === id)?.name || id;
  const handleModelPreferenceChange = (e) => {
    const modelId = e.target.value || null;
    const modelInfo = modelId ? availableData?.models?.find((m) => m.id === modelId) : null;
    onChange({
      ...profile,
      model_preference: modelId,
      model_source: modelInfo?.source ?? null,
      model_provider_type: modelInfo?.provider_type ?? null,
    });
  };
  const meta = profile.model_source_meta;
  const showRetargeted = meta?.retargeted;
  const showUnavailable = meta && meta.available === false;

  return (
    <Card variant="outlined" sx={{ mb: 1.5 }}>
      <CardContent sx={{ fontSize: '0.875rem' }}>
        <Typography variant="h6" sx={{ mb: 1, fontSize: '1rem' }}>
          Identity
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <TextField
            label="Name"
            value={profile.name || ''}
            onChange={handleField('name')}
            disabled={readOnly}
            fullWidth
            required
          />
          <TextField
            label="Handle (optional)"
            value={profile.handle || ''}
            onChange={handleField('handle')}
            disabled={readOnly}
            fullWidth
            placeholder="Leave blank for schedule/Run-only (not @mentionable in chat)"
            helperText="If set, users can invoke this agent via @handle in chat."
          />
          <FormControlLabel
            control={
              <Switch
                checked={profile.chat_visible !== false}
                onChange={(e) => onChange({ ...profile, chat_visible: e.target.checked })}
                disabled={readOnly || !profile.handle}
                color="primary"
              />
            }
            label="Show in chat @ menu"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: -0.5, display: 'block' }}>
            When off, this agent is still addressable by other agents in lines.
          </Typography>
          <FormControl fullWidth disabled={readOnly}>
            <InputLabel>Model preference</InputLabel>
            <Select
              value={profile.model_preference || ''}
              onChange={handleModelPreferenceChange}
              label="Model preference"
            >
              <MenuItem value="">— Default</MenuItem>
              {modelOptions.map((modelId) => (
                <MenuItem key={modelId} value={modelId}>
                  {getModelLabel(modelId)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {showRetargeted && (
            <Chip size="small" color="info" label="Model retargeted to current source" sx={{ alignSelf: 'flex-start' }} />
          )}
          {showUnavailable && (
            <Chip size="small" color="warning" label="Model unavailable in current source; using fallback" sx={{ alignSelf: 'flex-start' }} />
          )}
          <TextField
            label="System prompt additions"
            value={profile.system_prompt_additions || ''}
            onChange={handleField('system_prompt_additions')}
            disabled={readOnly}
            fullWidth
            multiline
            minRows={2}
            placeholder="Optional instructions for this agent"
          />
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <FormControl fullWidth disabled={readOnly} sx={{ minWidth: 200 }}>
              <InputLabel>Playbook</InputLabel>
              <Select
                value={defaultPlaybookId}
                label="Playbook"
                onChange={(e) => onChange({ ...profile, default_playbook_id: e.target.value || undefined })}
              >
                <MenuItem value="">—</MenuItem>
                {playbooks.map((pb) => (
                  <MenuItem key={pb.id} value={pb.id}>
                    {pb.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          <FormControlLabel
            control={
              <Switch
                checked={!!(profile.prompt_history_enabled ?? profile.chat_history_enabled)}
                onChange={(e) => onChange({ ...profile, prompt_history_enabled: e.target.checked })}
                disabled={readOnly}
                color="primary"
              />
            }
            label="Include history in prompt (recent messages in LLM steps)"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
          {(profile.prompt_history_enabled ?? profile.chat_history_enabled) && (
            <>
              <TextField
                fullWidth
                type="number"
                label="Chat history lookback (exchanges)"
                value={profile.chat_history_lookback ?? 10}
                onChange={(e) => {
                  const v = e.target.value;
                  onChange({
                    ...profile,
                    chat_history_lookback: v === '' ? 10 : Math.max(1, Math.min(50, Number(v) || 10)),
                  });
                }}
                inputProps={{ min: 1, max: 50, step: 1 }}
                helperText="Number of recent user+assistant exchanges to include (1–50)"
                disabled={readOnly}
              />
              <TextField
                fullWidth
                type="number"
                label="Summarize when over (~tokens)"
                value={profile.summary_threshold_tokens ?? 5000}
                onChange={(e) => {
                  const v = e.target.value;
                  onChange({
                    ...profile,
                    summary_threshold_tokens:
                      v === '' ? 5000 : Math.max(500, Math.min(100000, Number(v) || 5000)),
                  });
                }}
                inputProps={{ min: 500, max: 100000, step: 100 }}
                helperText="Rough estimate (about 4 characters per token). Older messages compress into one summary when exceeded."
                disabled={readOnly}
              />
              <TextField
                fullWidth
                type="number"
                label="Keep recent messages verbatim"
                value={profile.summary_keep_messages ?? 10}
                onChange={(e) => {
                  const v = e.target.value;
                  onChange({
                    ...profile,
                    summary_keep_messages:
                      v === '' ? 10 : Math.max(1, Math.min(50, Number(v) || 10)),
                  });
                }}
                inputProps={{ min: 1, max: 50, step: 1 }}
                helperText="When summarizing, how many latest messages stay full text (1–50). Use lookback ≥ this + 1 to include the summary in the prompt."
                disabled={readOnly}
              />
            </>
          )}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, alignSelf: 'flex-start' }}>
            <Typography variant="body2" color="text.secondary">
              Persona
            </Typography>
            <FormControl disabled={readOnly} sx={{ minWidth: 260 }}>
              <InputLabel>Persona mode</InputLabel>
              <Select
                value={profile.persona_mode || 'none'}
                label="Persona mode"
                onChange={(e) => {
                  const mode = e.target.value;
                  onChange({
                    ...profile,
                    persona_mode: mode,
                    persona_id: mode === 'specific' ? (profile.persona_id || null) : null,
                  });
                }}
              >
                <MenuItem value="none">No persona</MenuItem>
                <MenuItem value="default">Use default persona</MenuItem>
                <MenuItem value="specific">Select specific persona</MenuItem>
              </Select>
            </FormControl>
            {profile.persona_mode === 'specific' && (
              <>
                <FormControl disabled={readOnly} sx={{ minWidth: 260, mt: 1 }}>
                  <InputLabel>Persona</InputLabel>
                  <Select
                    value={profile.persona_id || ''}
                    label="Persona"
                    onChange={(e) => onChange({ ...profile, persona_id: e.target.value || null })}
                  >
                    <MenuItem value="">—</MenuItem>
                    {personasList.map((p) => (
                      <MenuItem key={p.id} value={p.id}>
                        {p.name}{p.is_builtin ? ' (built-in)' : ''}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                {profile.persona_id && (() => {
                  const p = personasList.find((x) => x.id === profile.persona_id);
                  return p?.style_instruction ? (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block', maxWidth: 400 }}>
                      {p.style_instruction.slice(0, 120)}{p.style_instruction.length > 120 ? '…' : ''}
                    </Typography>
                  ) : null;
                })()}
              </>
            )}
          </Box>
          <FormControlLabel
            control={
              <Switch
                checked={!!profile.include_user_context}
                onChange={(e) => onChange({ ...profile, include_user_context: e.target.checked })}
                disabled={readOnly}
                color="primary"
              />
            }
            label="Include user context (add name, email, timezone, ZIP, AI context to system prompt for every LLM step)"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={!!profile.include_datetime_context}
                onChange={(e) => onChange({ ...profile, include_datetime_context: e.target.checked })}
                disabled={readOnly}
                color="primary"
              />
            }
            label="Include date/time context (current date and time in user's timezone for every LLM step)"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={!!profile.include_user_facts}
                onChange={(e) => onChange({ ...profile, include_user_facts: e.target.checked })}
                disabled={readOnly}
                color="primary"
              />
            }
            label="Include user facts (remembered facts from Settings → Profile for every LLM step)"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
          {profile.include_user_facts && (
            <>
              <FormControlLabel
                control={
                  <Switch
                    checked={profile.use_themed_memory !== false}
                    onChange={(e) => onChange({ ...profile, use_themed_memory: e.target.checked })}
                    disabled={readOnly}
                    color="primary"
                  />
                }
                label="Themed memory (cluster similar facts; retrieve by query relevance with adaptive context size)"
                labelPlacement="start"
                sx={{ alignSelf: 'flex-start' }}
              />
              <Autocomplete
                multiple
                options={['general', 'work', 'preferences', 'personal']}
                value={Array.isArray(profile.include_facts_categories) ? profile.include_facts_categories : []}
                onChange={(_, value) => onChange({ ...profile, include_facts_categories: value || [] })}
                disabled={readOnly}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Fact categories to include"
                    placeholder="All if empty"
                    size="small"
                  />
                )}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip size="small" label={option} {...getTagProps({ index })} key={option} />
                  ))
                }
                sx={{ maxWidth: 400 }}
              />
            </>
          )}
          <FormControlLabel
            control={
              <Switch
                checked={!!profile.include_agent_memory}
                onChange={(e) => onChange({ ...profile, include_agent_memory: e.target.checked })}
                disabled={readOnly}
                color="primary"
              />
            }
            label="Include agent memory (persistent key/value memory injected into each run)"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
          <FormControlLabel
            control={
              <Switch
                checked={!!profile.auto_routable}
                onChange={(e) => onChange({ ...profile, auto_routable: e.target.checked })}
                disabled={readOnly}
                color="primary"
              />
            }
            label="Auto-routable (allow this agent to be selected automatically for matching queries without @mention)"
            labelPlacement="start"
            sx={{ alignSelf: 'flex-start' }}
          />
        </Box>
      </CardContent>
    </Card>
  );
}
