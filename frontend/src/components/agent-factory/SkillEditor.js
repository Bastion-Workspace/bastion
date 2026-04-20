/**
 * Standalone skill editor: view built-in skills (read-only + Duplicate), edit user skills.
 */

import React, { useState, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CircularProgress,
  Alert,
  TextField,
  Button,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Autocomplete,
  FormControlLabel,
  Switch,
} from '@mui/material';
import { Lock, ContentCopy, Delete, Save, Science } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import agentFactoryService from '../../services/agentFactoryService';
import SkillCandidatePanel from './SkillCandidatePanel';
import SkillRecommendationBanner from './SkillRecommendationBanner';

const CONNECTION_TYPE_PRESETS = ['email', 'calendar', 'code_platform', 'contacts', 'messaging'];

function ChipListField({ value = [], onChange, label, placeholder, disabled }) {
  const [input, setInput] = useState('');
  const items = Array.isArray(value) ? [...value] : [];

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      const v = (e.key === ',' ? input.replace(/,/g, '') : input).trim();
      if (v && !items.includes(v)) {
        onChange([...items, v]);
        setInput('');
      }
    }
  };

  const remove = (idx) => {
    const next = items.filter((_, i) => i !== idx);
    onChange(next);
  };

  return (
    <Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        {label}
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, alignItems: 'center', minHeight: 40 }}>
        {items.map((item, idx) => (
          <Chip
            key={`${item}-${idx}`}
            label={item}
            size="small"
            onDelete={disabled ? undefined : () => remove(idx)}
          />
        ))}
        {!disabled && (
          <TextField
            size="small"
            placeholder={placeholder}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            sx={{ minWidth: 120, '& .MuiInputBase-root': { py: 0.25 } }}
            inputProps={{ 'aria-label': label }}
          />
        )}
      </Box>
    </Box>
  );
}

export default function SkillEditor({ skillId: skillIdProp, onCloseEntityTab }) {
  const { skillId: skillIdParam } = useParams();
  const skillId = skillIdProp ?? skillIdParam;
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [localDraft, setLocalDraft] = useState(null);

  const { data: skill, isLoading, error } = useQuery(
    ['agentFactorySkill', skillId],
    () => apiService.agentFactory.getSkill(skillId),
    { enabled: !!skillId, retry: false }
  );

  const { data: actionsRaw = [] } = useQuery(
    ['agentFactoryActions'],
    () => agentFactoryService.getActions(),
    { staleTime: 300_000 }
  );
  const { data: allSkillsRaw = [] } = useQuery(
    ['agentFactorySkills'],
    () => apiService.agentFactory.listSkills({ include_builtin: true }),
    { staleTime: 120_000 }
  );

  const isBuiltin = skill?.is_builtin === true;
  const isShared = skill?.ownership === 'shared';
  const isUserSkill = !isBuiltin && !isShared && skill?.user_id;
  const effective = localDraft ?? skill;

  const dependsOnOptions = useMemo(() => {
    if (!Array.isArray(allSkillsRaw)) return [];
    return allSkillsRaw
      .filter((s) => s.slug && s.slug !== (effective?.slug ?? ''))
      .map((s) => s.slug)
      .sort();
  }, [allSkillsRaw, effective?.slug]);
  const allToolOptions = useMemo(() => {
    if (!Array.isArray(actionsRaw)) return [];
    return actionsRaw
      .map((a) => ({ name: a.name || '', description: a.description || '' }))
      .filter((o) => o.name)
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [actionsRaw]);

  const updateSkillMutation = useMutation(
    ({ id, body }) => apiService.agentFactory.updateSkill(id, body),
    {
      onSuccess: (data, { id }) => {
        if (data?.id) queryClient.setQueryData(['agentFactorySkill', id], data);
        queryClient.invalidateQueries(['agentFactorySkill', skillId]);
        queryClient.invalidateQueries('agentFactorySkills');
        setLocalDraft(null);
      },
    }
  );

  const deleteSkillMutation = useMutation(
    (id) => apiService.agentFactory.deleteSkill(id),
    {
      onSuccess: (_, id) => {
        queryClient.invalidateQueries('agentFactorySkills');
        setDeleteConfirm(false);
        onCloseEntityTab?.('skill', id);
      },
    }
  );

  const createSkillMutation = useMutation(
    (body) => apiService.agentFactory.createSkill(body),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('agentFactorySkills');
        if (data?.id) navigate(`/agent-factory/skill/${data.id}`);
      },
    }
  );

  const { data: skillMetrics } = useQuery(
    ['agentFactorySkillMetrics', skillId],
    () => apiService.agentFactory.getSkillMetrics(skillId),
    { enabled: !!skillId, staleTime: 60_000, retry: false }
  );

  const { data: sharedWithMe = [] } = useQuery(
    'agentFactorySharedWithMe',
    () => agentFactoryService.listSharedWithMe(),
    { retry: false }
  );
  const shareIdForSkill = (sharedWithMe || []).find(
    (s) => s.artifact_type === 'skill' && s.artifact_id === skillId
  )?.id;
  const copySharedToMineMutation = useMutation(
    (shareId) => agentFactoryService.copySharedToMine(shareId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('agentFactorySkills');
        queryClient.invalidateQueries('agentFactorySharedWithMe');
      },
    }
  );

  const handleSave = useCallback(() => {
    if (!skillId || !effective || isBuiltin || isShared) return;
    updateSkillMutation.mutate({
      id: skillId,
      body: {
        name: effective.name,
        description: effective.description ?? undefined,
        category: effective.category ?? undefined,
        procedure: effective.procedure ?? '',
        required_tools: effective.required_tools ?? [],
        optional_tools: effective.optional_tools ?? [],
        tags: effective.tags ?? [],
        required_connection_types: effective.required_connection_types ?? [],
        is_core: effective.is_core ?? false,
        depends_on: effective.depends_on ?? [],
      },
    });
  }, [skillId, effective, isBuiltin, isShared, updateSkillMutation]);

  const handleSaveAsCandidate = useCallback(() => {
    if (!skillId || !effective || isBuiltin || isShared) return;
    updateSkillMutation.mutate({
      id: skillId,
      body: {
        name: effective.name,
        description: effective.description ?? undefined,
        category: effective.category ?? undefined,
        procedure: effective.procedure ?? '',
        required_tools: effective.required_tools ?? [],
        optional_tools: effective.optional_tools ?? [],
        tags: effective.tags ?? [],
        required_connection_types: effective.required_connection_types ?? [],
        is_core: effective.is_core ?? false,
        depends_on: effective.depends_on ?? [],
        as_candidate: true,
      },
    });
  }, [skillId, effective, isBuiltin, isShared, updateSkillMutation]);

  const handleDuplicate = useCallback(() => {
    if (!skill) return;
    const baseSlug = (skill.slug || '').replace(/-custom$/, '');
    const slug = `${baseSlug}-custom`;
    createSkillMutation.mutate({
      name: (skill.name || skill.slug) + ' (copy)',
      slug,
      procedure: skill.procedure || '',
      required_tools: skill.required_tools ?? [],
      optional_tools: skill.optional_tools ?? [],
      description: skill.description ?? undefined,
      category: skill.category ?? undefined,
      tags: skill.tags ?? [],
      required_connection_types: skill.required_connection_types ?? [],
    });
  }, [skill, createSkillMutation]);

  const handleFieldChange = useCallback((field, value) => {
    if (isBuiltin || isShared) return;
    setLocalDraft((prev) => ({ ...(prev ?? skill), [field]: value }));
  }, [skill, isBuiltin, isShared]);

  if (!skillId) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="text.secondary">Select a skill from the sidebar or create one.</Typography>
      </Box>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !skill) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error">Skill not found.</Alert>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        flex: 1,
        minHeight: 0,
        overflow: 'auto',
        p: 2,
        maxWidth: 720,
        width: '100%',
        boxSizing: 'border-box',
      }}
    >
      {isShared && (
        <Alert
          severity="info"
          sx={{ mb: 2 }}
          action={
            shareIdForSkill && (
              <Button
                color="inherit"
                size="small"
                startIcon={<ContentCopy />}
                onClick={() => copySharedToMineMutation.mutate(shareIdForSkill)}
                disabled={copySharedToMineMutation.isLoading}
              >
                {copySharedToMineMutation.isLoading ? 'Copying…' : 'Make my own copy'}
              </Button>
            )
          }
        >
          This skill is shared by {skill?.owner_display_name || skill?.owner_username || 'another user'}. You can use it as-is or make your own copy to customize.
        </Alert>
      )}
      <Card variant="outlined" sx={{ mb: 2 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            {isBuiltin && (
              <Tooltip title="Built-in skill (read-only)">
                <Lock color="action" />
              </Tooltip>
            )}
            <Typography variant="h6" sx={{ flex: 1 }}>
              {effective?.name || effective?.slug || 'Skill'}
            </Typography>
            {isBuiltin && (
              <Button
                variant="contained"
                startIcon={<ContentCopy />}
                onClick={handleDuplicate}
                disabled={createSkillMutation.isLoading}
              >
                Duplicate to My Skills
              </Button>
            )}
            {isUserSkill && (
              <>
                <Button
                  variant="outlined"
                  startIcon={<Save />}
                  onClick={handleSave}
                  disabled={updateSkillMutation.isLoading || !localDraft}
                >
                  Save
                </Button>
                <Tooltip title="Save changes as a candidate version for A/B testing">
                  <Button
                    variant="outlined"
                    color="secondary"
                    startIcon={<Science />}
                    onClick={handleSaveAsCandidate}
                    disabled={updateSkillMutation.isLoading || !localDraft}
                    size="small"
                  >
                    Save as Candidate
                  </Button>
                </Tooltip>
                <Tooltip title="Delete skill">
                  <IconButton color="error" onClick={() => setDeleteConfirm(true)} aria-label="Delete skill">
                    <Delete />
                  </IconButton>
                </Tooltip>
              </>
            )}
          </Box>

          <SkillRecommendationBanner skillId={skillId} />

          {skillMetrics && skillMetrics.total_uses > 0 && (
            <Box
              sx={{
                display: 'flex',
                gap: 2,
                mb: 2,
                p: 1.5,
                borderRadius: 1,
                bgcolor: 'action.hover',
                flexWrap: 'wrap',
              }}
            >
              <Box sx={{ textAlign: 'center', minWidth: 60 }}>
                <Typography variant="h6" sx={{ lineHeight: 1.2 }}>{skillMetrics.total_uses}</Typography>
                <Typography variant="caption" color="text.secondary">Uses</Typography>
              </Box>
              <Box sx={{ textAlign: 'center', minWidth: 60 }}>
                <Typography variant="h6" sx={{ lineHeight: 1.2 }}>
                  {skillMetrics.success_rate != null ? `${Math.round(skillMetrics.success_rate * 100)}%` : '—'}
                </Typography>
                <Typography variant="caption" color="text.secondary">Success</Typography>
              </Box>
              <Box sx={{ textAlign: 'center', minWidth: 60 }}>
                <Typography variant="h6" sx={{ lineHeight: 1.2 }}>{skillMetrics.unique_agents || 0}</Typography>
                <Typography variant="caption" color="text.secondary">Agents</Typography>
              </Box>
              <Box sx={{ textAlign: 'center', minWidth: 60 }}>
                <Typography variant="h6" sx={{ lineHeight: 1.2 }}>{skillMetrics.uses_last_7d || 0}</Typography>
                <Typography variant="caption" color="text.secondary">Last 7d</Typography>
              </Box>
              {skillMetrics.last_used_at && (
                <Box sx={{ textAlign: 'center', minWidth: 80 }}>
                  <Typography variant="body2" sx={{ lineHeight: 1.4 }}>
                    {new Date(skillMetrics.last_used_at).toLocaleDateString()}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">Last Used</Typography>
                </Box>
              )}
            </Box>
          )}

          <SkillCandidatePanel skillId={skillId} />

          {(effective?.description != null && effective.description !== '') || !isBuiltin ? (
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                Description
              </Typography>
              {isBuiltin ? (
                <Typography variant="body2">{effective.description}</Typography>
              ) : (
                <TextField
                  fullWidth
                  multiline
                  minRows={1}
                  maxRows={3}
                  size="small"
                  value={effective?.description ?? ''}
                  onChange={(e) => handleFieldChange('description', e.target.value)}
                  placeholder="Optional description"
                />
              )}
            </Box>
          ) : null}

          <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
            <Box sx={{ minWidth: 160 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Slug
              </Typography>
              <TextField
                fullWidth
                size="small"
                value={effective?.slug ?? ''}
                disabled
                helperText="Slug is fixed after creation"
              />
            </Box>
            <Box sx={{ minWidth: 160 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Category
              </Typography>
              {isBuiltin ? (
                <Typography variant="body2">{effective?.category || '—'}</Typography>
              ) : (
                <TextField
                  fullWidth
                  size="small"
                  value={effective?.category ?? ''}
                  onChange={(e) => handleFieldChange('category', e.target.value)}
                  placeholder="e.g. search, org"
                />
              )}
            </Box>
          </Box>

          {!isBuiltin && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Name
              </Typography>
              <TextField
                fullWidth
                size="small"
                value={effective?.name ?? ''}
                onChange={(e) => handleFieldChange('name', e.target.value)}
                placeholder="Display name"
              />
            </Box>
          )}

          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Procedure
            </Typography>
            {isBuiltin ? (
              <Box
                component="pre"
                sx={{
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  p: 1.5,
                  bgcolor: 'action.hover',
                  borderRadius: 1,
                  fontSize: '0.875rem',
                  fontFamily: 'inherit',
                }}
              >
                {effective?.procedure || '—'}
              </Box>
            ) : (
              <TextField
                fullWidth
                multiline
                minRows={4}
                size="small"
                value={effective?.procedure ?? ''}
                onChange={(e) => handleFieldChange('procedure', e.target.value)}
                placeholder="Markdown or plain text procedure for the LLM"
              />
            )}
          </Box>

          <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
            <Box sx={{ flex: '1 1 200px' }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Required tools
              </Typography>
              {isBuiltin ? (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(effective?.required_tools || []).map((t) => (
                    <Chip key={t} size="small" label={t} />
                  ))}
                  {(!effective?.required_tools || effective.required_tools.length === 0) && (
                    <Typography variant="body2" color="text.secondary">None</Typography>
                  )}
                </Box>
              ) : (
                <Autocomplete
                  multiple
                  freeSolo
                  size="small"
                  options={allToolOptions}
                  getOptionLabel={(opt) => (typeof opt === 'string' ? opt : opt.name)}
                  renderOption={(props, opt) => (
                    <li {...props} key={opt.name}>
                      <Box>
                        <Typography variant="body2">{opt.name}</Typography>
                        {opt.description && (
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                            {opt.description.slice(0, 100)}
                          </Typography>
                        )}
                      </Box>
                    </li>
                  )}
                  value={effective?.required_tools ?? []}
                  onChange={(_, v) => handleFieldChange('required_tools', v.map((x) => (typeof x === 'string' ? x : x.name)))}
                  renderTags={(vals, getTagProps) =>
                    vals.map((v, i) => <Chip {...getTagProps({ index: i })} key={v} size="small" label={v} />)
                  }
                  renderInput={(params) => (
                    <TextField {...params} placeholder="Search tools…" />
                  )}
                />
              )}
            </Box>
            <Box sx={{ flex: '1 1 200px' }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Optional tools
              </Typography>
              {isBuiltin ? (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {(effective?.optional_tools || []).map((t) => (
                    <Chip key={t} size="small" label={t} />
                  ))}
                  {(!effective?.optional_tools || effective.optional_tools.length === 0) && (
                    <Typography variant="body2" color="text.secondary">None</Typography>
                  )}
                </Box>
              ) : (
                <Autocomplete
                  multiple
                  freeSolo
                  size="small"
                  options={allToolOptions}
                  getOptionLabel={(opt) => (typeof opt === 'string' ? opt : opt.name)}
                  renderOption={(props, opt) => (
                    <li {...props} key={opt.name}>
                      <Box>
                        <Typography variant="body2">{opt.name}</Typography>
                        {opt.description && (
                          <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                            {opt.description.slice(0, 100)}
                          </Typography>
                        )}
                      </Box>
                    </li>
                  )}
                  value={effective?.optional_tools ?? []}
                  onChange={(_, v) => handleFieldChange('optional_tools', v.map((x) => (typeof x === 'string' ? x : x.name)))}
                  renderTags={(vals, getTagProps) =>
                    vals.map((v, i) => <Chip {...getTagProps({ index: i })} key={v} size="small" label={v} />)
                  }
                  renderInput={(params) => (
                    <TextField {...params} placeholder="Search tools…" />
                  )}
                />
              )}
            </Box>
          </Box>

          <Box sx={{ mb: 2, mt: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Required connection types
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              If set, this skill is only auto-suggested when the user has an active connection of each type (e.g. email, calendar).
            </Typography>
            {isBuiltin ? (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {Array.isArray(effective?.required_connection_types) && effective.required_connection_types.length > 0 ? (
                  effective.required_connection_types.map((t) => (
                    <Chip key={t} size="small" label={t} variant="outlined" />
                  ))
                ) : (
                  <Typography variant="body2" color="text.secondary">None</Typography>
                )}
              </Box>
            ) : (
              <>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                  {CONNECTION_TYPE_PRESETS.map((t) => {
                    const cur = Array.isArray(effective?.required_connection_types) ? effective.required_connection_types : [];
                    const has = cur.includes(t);
                    return (
                      <Button
                        key={t}
                        size="small"
                        variant={has ? 'contained' : 'outlined'}
                        onClick={() => {
                          if (has) {
                            handleFieldChange('required_connection_types', cur.filter((x) => x !== t));
                          } else {
                            handleFieldChange('required_connection_types', [...cur, t]);
                          }
                        }}
                      >
                        {t}
                      </Button>
                    );
                  })}
                </Box>
                <ChipListField
                  label="Custom types (Enter to add)"
                  value={effective?.required_connection_types ?? []}
                  onChange={(v) => handleFieldChange('required_connection_types', v)}
                  placeholder="e.g. code_platform"
                  disabled={false}
                />
              </>
            )}
          </Box>

          <Box sx={{ mb: 2, mt: 2 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Depends on (skill slugs)
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Skills this skill depends on. Their tools and procedures are automatically included at runtime.
            </Typography>
            {isBuiltin ? (
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {(effective?.depends_on || []).length > 0
                  ? effective.depends_on.map((d) => <Chip key={d} size="small" label={d} variant="outlined" />)
                  : <Typography variant="body2" color="text.secondary">None</Typography>}
              </Box>
            ) : (
              <Autocomplete
                multiple
                freeSolo
                size="small"
                options={dependsOnOptions}
                value={effective?.depends_on ?? []}
                onChange={(_, v) => handleFieldChange('depends_on', v)}
                renderTags={(vals, getTagProps) =>
                  vals.map((v, i) => <Chip {...getTagProps({ index: i })} key={v} size="small" label={v} variant="outlined" />)
                }
                renderInput={(params) => (
                  <TextField {...params} placeholder="Search skills by slug…" />
                )}
              />
            )}
          </Box>

          <Box sx={{ mt: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={!!effective?.is_core}
                  onChange={(e) => handleFieldChange('is_core', e.target.checked)}
                  disabled={isBuiltin}
                />
              }
              label="Core skill — always visible in agent catalog"
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 5.5 }}>
              Core skills appear in the condensed catalog injected into every agent's system prompt.
              Non-core skills are discoverable via vector search at runtime.
            </Typography>
          </Box>

          {!isBuiltin && (
            <Box sx={{ mt: 2 }}>
              <ChipListField
                label="Tags"
                value={effective?.tags}
                onChange={(v) => handleFieldChange('tags', v)}
                placeholder="Add tag, Enter"
                disabled={false}
              />
            </Box>
          )}

          {updateSkillMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {updateSkillMutation.error?.message || 'Failed to save'}
            </Alert>
          )}
          {createSkillMutation.isError && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {createSkillMutation.error?.message || 'Failed to duplicate'}
            </Alert>
          )}
        </CardContent>
      </Card>

      <Dialog open={deleteConfirm} onClose={() => !deleteSkillMutation.isLoading && setDeleteConfirm(false)}>
        <DialogTitle>Delete skill</DialogTitle>
        <DialogContent>
          <Typography>
            Permanently delete skill <strong>{skill?.name || skill?.slug}</strong>? This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirm(false)} disabled={deleteSkillMutation.isLoading}>
            Cancel
          </Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => skillId && deleteSkillMutation.mutate(skillId)}
            disabled={deleteSkillMutation.isLoading}
          >
            {deleteSkillMutation.isLoading ? 'Deleting…' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
