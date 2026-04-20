import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  IconButton,
  Paper,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
  Collapse,
} from '@mui/material';
import { Add, Delete, ExpandMore, ExpandLess } from '@mui/icons-material';

/** Subagents for LLM agent or deep agent steps. `variant` controls parallel/sequential help text (task source differs). */
export default function SubagentsBlock({
  step,
  setStep,
  profiles,
  playbooks,
  readOnly,
  onOpenAddDialog,
  variant = 'llm_agent',
  resetKey,
}) {
  const subagentCount = (Array.isArray(step.subagents) ? step.subagents : []).length;
  const [expanded, setExpanded] = useState(() => subagentCount > 0);
  const prevCountRef = useRef(subagentCount);

  useEffect(() => {
    if (resetKey === undefined || resetKey === null) return;
    const n = (Array.isArray(step.subagents) ? step.subagents : []).length;
    setExpanded(n > 0);
    prevCountRef.current = n;
  }, [resetKey]);

  useEffect(() => {
    const n = (Array.isArray(step.subagents) ? step.subagents : []).length;
    if (n > prevCountRef.current && prevCountRef.current === 0) {
      setExpanded(true);
    }
    prevCountRef.current = n;
  }, [step.subagents]);

  const delegationMode = step.delegation_mode || 'supervised';
  const modeExplanation =
    delegationMode === 'supervised'
      ? 'The supervisor model chooses when to call each delegation tool during the step. **Role**, **Accepts**, and **Returns** are embedded in that tool’s description so the model can pick the right specialist. On each call, the model supplies a **task** string (and optional context) as the tool arguments.'
      : variant === 'deep_agent'
        ? 'Each subagent runs **once** before the deep-agent phases. Their initial task combines the **user query** with prompts and criteria from the **first few phases** on this step (not the LLM Agent prompt template). Results go to a **shared scratchpad** for later phases to use.'
        : 'Each subagent runs **once** before the supervisor’s ReAct loop, using the **resolved prompt template** on this step (above) as their initial task. Results land in a **shared scratchpad**; the supervisor then integrates them. Write that prompt as a shared objective; use **Role** / **Accepts** / **Returns** so each specialist knows their part.';

  const title =
    subagentCount > 0 ? `Subagents (${subagentCount})` : 'Subagents';

  return (
    <Box sx={{ mt: 2, mb: 0 }}>
      <Button
        fullWidth
        size="small"
        onClick={() => setExpanded((e) => !e)}
        endIcon={expanded ? <ExpandLess /> : <ExpandMore />}
        aria-expanded={expanded}
        sx={{ justifyContent: 'space-between', textTransform: 'none', mb: expanded ? 1 : 0 }}
      >
        {title}
      </Button>
      <Collapse in={expanded}>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Add <strong>agent profiles</strong> this step may delegate to. Each profile gets its own <strong>delegation tool</strong> whose description tells the parent model when to use that specialist. Subagents share a <strong>scratchpad</strong> for outputs. Each run uses that profile&apos;s <strong>default playbook</strong> (same as anywhere else you use the profile).
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, mb: 1.5 }}>
        {(Array.isArray(step.subagents) ? step.subagents : []).map((sa, idx) => {
          const pid = sa.agent_profile_id || '';
          const profile = profiles.find((p) => p.id === pid);
          const pbid = sa.playbook_id || '';
          const playbook = pbid ? playbooks.find((p) => p.id === pbid) : null;
          return (
            <Paper key={`${pid}-${idx}`} variant="outlined" sx={{ p: 1.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1, gap: 1 }}>
                <Box>
                  <Typography variant="body2" fontWeight={600}>
                    {profile?.name || pid || 'Profile'}
                  </Typography>
                  {!pbid && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      Uses profile default playbook
                    </Typography>
                  )}
                  {pbid && (
                    <Typography variant="caption" color="warning.main" display="block">
                      Stored playbook override (legacy): {playbook?.name || pbid}. Remove and re-add to use only the profile default.
                    </Typography>
                  )}
                </Box>
                {!readOnly && (
                  <IconButton
                    size="small"
                    aria-label="Remove subagent"
                    onClick={() => {
                      setStep((s) => {
                        const list = [...(Array.isArray(s.subagents) ? s.subagents : [])];
                        list.splice(idx, 1);
                        return { ...s, subagents: list };
                      });
                    }}
                  >
                    <Delete fontSize="small" />
                  </IconButton>
                )}
              </Box>
              <TextField
                size="small"
                fullWidth
                label="Role"
                helperText="Included in this subagent’s tool description (name + role line). Use a short label, e.g. “Research specialist”."
                value={sa.role ?? ''}
                disabled={readOnly}
                onChange={(e) => {
                  setStep((s) => {
                    const list = [...(Array.isArray(s.subagents) ? s.subagents : [])];
                    list[idx] = { ...list[idx], role: e.target.value };
                    return { ...s, subagents: list };
                  });
                }}
                sx={{ mb: 1 }}
              />
              <TextField
                size="small"
                fullWidth
                label="Accepts"
                helperText='Shown in the tool description as “Best for: …” so the parent model knows what to send this subagent.'
                value={sa.accepts ?? ''}
                disabled={readOnly}
                onChange={(e) => {
                  setStep((s) => {
                    const list = [...(Array.isArray(s.subagents) ? s.subagents : [])];
                    list[idx] = { ...list[idx], accepts: e.target.value };
                    return { ...s, subagents: list };
                  });
                }}
                sx={{ mb: 1 }}
              />
              <TextField
                size="small"
                fullWidth
                label="Returns"
                helperText='Shown as “Typically returns: …” in the tool description (expected output shape or topic).'
                value={sa.returns ?? ''}
                disabled={readOnly}
                onChange={(e) => {
                  setStep((s) => {
                    const list = [...(Array.isArray(s.subagents) ? s.subagents : [])];
                    list[idx] = { ...list[idx], returns: e.target.value };
                    return { ...s, subagents: list };
                  });
                }}
              />
            </Paper>
          );
        })}
      </Box>
      <Box sx={{ mb: 1.5 }}>
        <Button
          size="small"
          startIcon={<Add />}
          disabled={readOnly}
          onClick={onOpenAddDialog}
        >
          Add subagent
        </Button>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
        Delegation mode
      </Typography>
      <ToggleButtonGroup
        exclusive
        size="small"
        value={delegationMode}
        disabled={readOnly}
        onChange={(_, v) => {
          if (v == null) return;
          setStep((s) => ({ ...s, delegation_mode: v }));
        }}
        sx={{ mb: 1, flexWrap: 'wrap' }}
      >
        <Tooltip title="Parent model chooses when to call each delegation tool; each call includes a task string and optional JSON context.">
          <ToggleButton value="supervised">Supervised</ToggleButton>
        </Tooltip>
        <Tooltip
          title={
            variant === 'deep_agent'
              ? 'All subagents run at once before phases, using query + phase context as their task; then the graph continues.'
              : 'All subagents run at once before the ReAct loop, each using this step’s resolved prompt as their task.'
          }
        >
          <ToggleButton value="parallel">Parallel</ToggleButton>
        </Tooltip>
        <Tooltip
          title={
            variant === 'deep_agent'
              ? 'Subagents run in list order before phases, same task source as parallel.'
              : 'Subagents run in list order before the ReAct loop, each using this step’s resolved prompt as their task.'
          }
        >
          <ToggleButton value="sequential">Sequential</ToggleButton>
        </Tooltip>
      </ToggleButtonGroup>
      <Typography
        variant="caption"
        color="text.secondary"
        component="div"
        sx={{ display: 'block', mb: 2, lineHeight: 1.45 }}
      >
        {modeExplanation.split('**').map((chunk, i) => (i % 2 === 1 ? <strong key={i}>{chunk}</strong> : <React.Fragment key={i}>{chunk}</React.Fragment>))}
      </Typography>
      </Collapse>
    </Box>
  );
}
