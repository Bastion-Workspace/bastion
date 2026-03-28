/**
 * Standalone skill editor: view built-in skills (read-only + Duplicate), edit user skills.
 */

import React, { useState, useCallback } from 'react';
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
} from '@mui/material';
import { Lock, ContentCopy, Delete, Save } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

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

  const isBuiltin = skill?.is_builtin === true;
  const isUserSkill = !isBuiltin && skill?.user_id;
  const effective = localDraft ?? skill;

  const handleSave = useCallback(() => {
    if (!skillId || !effective || isBuiltin) return;
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
      },
    });
  }, [skillId, effective, isBuiltin, updateSkillMutation]);

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
    });
  }, [skill, createSkillMutation]);

  const handleFieldChange = useCallback((field, value) => {
    setLocalDraft((prev) => ({ ...(prev ?? skill), [field]: value }));
  }, [skill]);

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
    <Box sx={{ p: 2, maxWidth: 720 }}>
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
                <Tooltip title="Delete skill">
                  <IconButton color="error" onClick={() => setDeleteConfirm(true)} aria-label="Delete skill">
                    <Delete />
                  </IconButton>
                </Tooltip>
              </>
            )}
          </Box>

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
              <ChipListField
                label="Required tools"
                value={effective?.required_tools}
                onChange={(v) => handleFieldChange('required_tools', v)}
                placeholder="Add tool name, Enter"
                disabled={isBuiltin}
              />
            </Box>
            <Box sx={{ flex: '1 1 200px' }}>
              <ChipListField
                label="Optional tools"
                value={effective?.optional_tools}
                onChange={(v) => handleFieldChange('optional_tools', v)}
                placeholder="Add tool name, Enter"
                disabled={isBuiltin}
              />
            </Box>
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
