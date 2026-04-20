import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  FormControlLabel,
  Checkbox,
  Collapse,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
} from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

const MAX_SAMPLES = 5;

/** Best-of-N sampling for llm_agent / deep_agent steps (orchestrator: playbook_graph_builder). */
export default function BestOfNBlock({
  step,
  setStep,
  readOnly,
  variant = 'llm_agent',
  resetKey,
}) {
  const raw = step.samples;
  let n = 1;
  try {
    n = raw == null || raw === '' ? 1 : Math.max(1, Math.min(MAX_SAMPLES, parseInt(String(raw), 10) || 1));
  } catch {
    n = 1;
  }
  const enabled = n > 1;
  const [expanded, setExpanded] = useState(() => enabled);
  const prevEnabledRef = useRef(enabled);

  useEffect(() => {
    if (resetKey === undefined || resetKey === null) return;
    const on = (() => {
      const r = step.samples;
      try {
        const v = r == null || r === '' ? 1 : Math.max(1, Math.min(MAX_SAMPLES, parseInt(String(r), 10) || 1));
        return v > 1;
      } catch {
        return false;
      }
    })();
    setExpanded(on);
    prevEnabledRef.current = on;
  }, [resetKey, step.samples]);

  useEffect(() => {
    if (enabled && !prevEnabledRef.current) {
      setExpanded(true);
    }
    prevEnabledRef.current = enabled;
  }, [enabled]);

  const hasEvaluatePhase = useMemo(
    () =>
      variant === 'deep_agent' &&
      Array.isArray(step.phases) &&
      step.phases.some((p) => String(p?.type || '').toLowerCase() === 'evaluate'),
    [variant, step.phases],
  );

  useEffect(() => {
    if (
      variant === 'deep_agent' &&
      (step.selection_strategy || '').toLowerCase() === 'highest_score' &&
      !hasEvaluatePhase
    ) {
      setStep((s) => ({ ...s, selection_strategy: 'llm_judge' }));
    }
  }, [variant, step.selection_strategy, hasEvaluatePhase, setStep]);

  const strategy = (step.selection_strategy || 'llm_judge').toLowerCase();
  const criteria = step.selection_criteria ?? '';

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
        Best-of-N sampling{enabled ? ` (${n} runs)` : ''}
      </Button>
      <Collapse in={expanded}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Runs the step <strong>N times</strong> independently (higher temperature for diversity), then{' '}
          <strong>picks the best</strong> result. Cost and latency scale with N. Use for high-stakes outputs.
        </Typography>
        <FormControlLabel
          control={
            <Checkbox
              checked={enabled}
              disabled={readOnly}
              onChange={(e) => {
                if (e.target.checked) {
                  setStep((s) => ({
                    ...s,
                    samples: 3,
                    selection_strategy: s.selection_strategy || 'llm_judge',
                  }));
                } else {
                  setStep((s) => {
                    const { samples, selection_strategy, selection_criteria, ...rest } = s;
                    return rest;
                  });
                }
              }}
            />
          }
          label="Run multiple samples and select the best"
        />
        {enabled && (
          <>
            <TextField
              fullWidth
              size="small"
              type="number"
              label="Samples (runs)"
              value={n}
              disabled={readOnly}
              onChange={(e) => {
                const v = e.target.value;
                const parsed = v === '' ? 2 : Math.max(2, Math.min(MAX_SAMPLES, parseInt(v, 10) || 2));
                setStep((s) => ({ ...s, samples: parsed }));
              }}
              inputProps={{ min: 2, max: MAX_SAMPLES, step: 1 }}
              sx={{ mb: 1.5 }}
              helperText={`Between 2 and ${MAX_SAMPLES}.`}
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Selection strategy
            </Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={strategy}
              disabled={readOnly}
              onChange={(_, v) => {
                if (v == null) return;
                setStep((s) => ({ ...s, selection_strategy: v }));
              }}
              sx={{ mb: 1, flexWrap: 'wrap' }}
            >
              <Tooltip title="An LLM compares all outputs and picks the best (extra LLM call).">
                <ToggleButton value="llm_judge">LLM judge</ToggleButton>
              </Tooltip>
              {hasEvaluatePhase && (
                <Tooltip title="Uses the score from the last evaluate phase in each run (no judge call). Falls back to LLM judge if scores are missing.">
                  <ToggleButton value="highest_score">Highest score</ToggleButton>
                </Tooltip>
              )}
            </ToggleButtonGroup>
            {strategy !== 'highest_score' && (
              <TextField
                fullWidth
                size="small"
                multiline
                minRows={2}
                label="Selection criteria (for LLM judge)"
                value={criteria}
                disabled={readOnly}
                onChange={(e) => setStep((s) => ({ ...s, selection_criteria: e.target.value }))}
                placeholder="e.g. Pick the most thorough, well-structured answer."
                sx={{ mb: 1 }}
              />
            )}
          </>
        )}
      </Collapse>
    </Box>
  );
}
