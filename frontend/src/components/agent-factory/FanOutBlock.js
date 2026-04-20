import React, { useState, useEffect, useRef } from 'react';
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

const MAX_ITEMS_CAP = 10;

/** Dynamic fan-out: one run per list item (orchestrator: playbook_graph_builder). */
export default function FanOutBlock({ step, setStep, readOnly, resetKey }) {
  const fo = step.fan_out;
  const enabled = fo != null && typeof fo === 'object';

  const [expanded, setExpanded] = useState(() => enabled);
  const prevEnabledRef = useRef(enabled);

  useEffect(() => {
    if (resetKey === undefined || resetKey === null) return;
    const on = step.fan_out != null && typeof step.fan_out === 'object';
    setExpanded(on);
    prevEnabledRef.current = on;
  }, [resetKey, step.fan_out]);

  useEffect(() => {
    if (enabled && !prevEnabledRef.current) {
      setExpanded(true);
    }
    prevEnabledRef.current = enabled;
  }, [enabled]);

  const source = (enabled && fo.source) || '';
  const itemVariable = (enabled && fo.item_variable) || 'current_item';
  const maxItems = enabled ? Math.max(1, Math.min(MAX_ITEMS_CAP, parseInt(String(fo.max_items ?? 10), 10) || 10)) : 10;
  const merge = (enabled && (fo.merge || 'list')) || 'list';

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
        Dynamic fan-out{enabled ? ' (on)' : ''}
      </Button>
      <Collapse in={expanded}>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
          Reads a <strong>list</strong> from an earlier step (dot-path in playbook state) and runs this step{' '}
          <strong>once per item in parallel</strong>. Inject <code>{`{${itemVariable || 'current_item'}}`}</code> in your
          prompt template. Results merge into one output under this step&apos;s <strong>output_key</strong>.
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
                    fan_out: {
                      source: '',
                      item_variable: 'current_item',
                      max_items: 10,
                      merge: 'list',
                    },
                  }));
                } else {
                  setStep((s) => {
                    const { fan_out, ...rest } = s;
                    return rest;
                  });
                }
              }}
            />
          }
          label="Run once per item in a list"
        />
        {enabled && (
          <>
            <TextField
              fullWidth
              size="small"
              label="Source (dot-path to list)"
              value={source}
              disabled={readOnly}
              onChange={(e) => {
                const v = e.target.value;
                setStep((s) => ({
                  ...s,
                  fan_out: { ...(s.fan_out || {}), source: v },
                }));
              }}
              placeholder="e.g. plan.items or classify.topics"
              helperText="Path under prior step outputs, e.g. my_step.items if that step stored a list at items."
              sx={{ mb: 1.5 }}
            />
            <TextField
              fullWidth
              size="small"
              label="Item variable name"
              value={itemVariable}
              disabled={readOnly}
              onChange={(e) => {
                const v = (e.target.value || 'current_item').replace(/[^A-Za-z0-9_]/g, '');
                setStep((s) => ({
                  ...s,
                  fan_out: { ...(s.fan_out || {}), item_variable: v || 'current_item' },
                }));
              }}
              helperText={`Use {${itemVariable || 'current_item'}} in the prompt template.`}
              sx={{ mb: 1.5 }}
            />
            <TextField
              fullWidth
              size="small"
              type="number"
              label="Max parallel items"
              value={maxItems}
              disabled={readOnly}
              onChange={(e) => {
                const v = e.target.value;
                const parsed = v === '' ? 10 : Math.max(1, Math.min(MAX_ITEMS_CAP, parseInt(v, 10) || 10));
                setStep((s) => ({
                  ...s,
                  fan_out: { ...(s.fan_out || {}), max_items: parsed },
                }));
              }}
              inputProps={{ min: 1, max: MAX_ITEMS_CAP, step: 1 }}
              sx={{ mb: 1.5 }}
            />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Merge mode
            </Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              value={merge}
              disabled={readOnly}
              onChange={(_, v) => {
                if (v == null) return;
                setStep((s) => ({
                  ...s,
                  fan_out: { ...(s.fan_out || {}), merge: v },
                }));
              }}
              sx={{ mb: 1, flexWrap: 'wrap' }}
            >
              <Tooltip title="Output includes items (full result dicts per item) plus a combined formatted string.">
                <ToggleButton value="list">List + formatted</ToggleButton>
              </Tooltip>
              <Tooltip title="Output formatted text with a section per item (still includes items array).">
                <ToggleButton value="concat">Concatenate sections</ToggleButton>
              </Tooltip>
            </ToggleButtonGroup>
          </>
        )}
      </Collapse>
    </Box>
  );
}
