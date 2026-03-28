/**
 * Phase list editor for deep_agent steps. Ordered list of phases with add/remove/reorder,
 * per-phase config: name, type (reason, act, search, evaluate, synthesize, refine), type-specific fields.
 */

import React, { useState, useMemo } from 'react';
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  FormControlLabel,
  Checkbox,
} from '@mui/material';
import { Add, Delete, ExpandMore, DragIndicator } from '@mui/icons-material';
import CollapsibleToolPicker from './CollapsibleToolPicker';

const PHASE_TYPES = [
  { value: 'reason', label: 'Reason' },
  { value: 'act', label: 'Act' },
  { value: 'search', label: 'Search' },
  { value: 'evaluate', label: 'Evaluate' },
  { value: 'synthesize', label: 'Synthesize' },
  { value: 'refine', label: 'Refine' },
];

const STARTER_TEMPLATES = [
  { value: '', label: 'Start from template (optional)' },
  {
    value: 'research',
    label: 'Research',
    available_tools: ['search_documents', 'crawl4ai_web_search'],
    phases: [
      { name: 'plan', type: 'reason', prompt: 'Analyze the query and plan a search strategy.\n\nQuery: {query}' },
      { name: 'gather', type: 'search', prompt: 'Gather relevant information.', strategy: 'parallel' },
      { name: 'synthesize', type: 'synthesize', prompt: 'Synthesize the gathered information into a clear, comprehensive answer.\n\nFindings:\n{gather.output}' },
      { name: 'critique', type: 'evaluate', criteria: 'Is the synthesis comprehensive and well-supported by the findings?', pass_threshold: 0.7, on_pass: 'end', on_fail: 'revise', max_retries: 2 },
      { name: 'revise', type: 'refine', target: 'synthesize', prompt: 'Improve the synthesis based on this feedback:\n{critique.feedback}\n\nCurrent synthesis:\n{synthesize.output}', next: 'critique' },
    ],
  },
  {
    value: 'iterative_refinement',
    label: 'Iterative Refinement',
    available_tools: [],
    phases: [
      { name: 'outline', type: 'reason', prompt: 'Create an outline or plan for the task.\n\nTask: {query}' },
      { name: 'draft', type: 'synthesize', prompt: 'Produce an initial draft based on the outline.\n\nOutline: {outline.output}' },
      { name: 'evaluate', type: 'evaluate', criteria: 'Does the draft meet the requirements and quality bar?', pass_threshold: 0.75, on_pass: 'end', on_fail: 'revise', max_retries: 3 },
      { name: 'revise', type: 'refine', target: 'draft', prompt: 'Revise the draft based on feedback:\n{evaluate.feedback}\n\nCurrent draft:\n{draft.output}', next: 'evaluate' },
    ],
  },
  {
    value: 'analyze',
    label: 'Analyze',
    available_tools: ['search_documents'],
    phases: [
      { name: 'analyze', type: 'reason', prompt: 'Analyze the following and identify key points, implications, and recommendations.\n\nContext: {query}' },
      { name: 'synthesize', type: 'synthesize', prompt: 'Turn the analysis into a structured summary with clear conclusions.\n\nAnalysis: {analyze.output}' },
    ],
  },
];

export default function DeepAgentPhaseEditor({
  step,
  setStep,
  actions = [],
  stepPaletteTools = [],
  readOnly = false,
}) {
  const phases = Array.isArray(step?.phases) ? step.phases : [];
  const [expandedPhase, setExpandedPhase] = useState(null);

  const actionsFilteredByPalette = useMemo(() => {
    if (!stepPaletteTools || stepPaletteTools.length === 0) return actions;
    const set = new Set(stepPaletteTools);
    return actions.filter((a) => {
      const name = typeof a === 'string' ? a : a?.name;
      return name && set.has(name);
    });
  }, [actions, stepPaletteTools]);

  const handleTemplateChange = (templateValue) => {
    if (!templateValue) return;
    const t = STARTER_TEMPLATES.find((x) => x.value === templateValue);
    if (t?.phases) {
      setPhases(t.phases.map((p) => ({ ...p })));
      if (Array.isArray(t.available_tools) && t.available_tools.length > 0) {
        setStep((s) => ({ ...s, available_tools: t.available_tools }));
      }
      setExpandedPhase(0);
    }
  };

  const setPhases = (next) => {
    setStep((s) => ({ ...s, phases: next }));
  };

  const addPhase = () => {
    const name = `phase_${phases.length + 1}`;
    setPhases([...phases, { name, type: 'reason', prompt: '' }]);
    setExpandedPhase(phases.length);
  };

  const removePhase = (idx) => {
    const next = phases.filter((_, i) => i !== idx);
    setPhases(next);
    if (expandedPhase === idx) setExpandedPhase(null);
    else if (expandedPhase > idx) setExpandedPhase(expandedPhase - 1);
  };

  const updatePhase = (idx, patch) => {
    const next = [...phases];
    next[idx] = { ...(next[idx] || {}), ...patch };
    setPhases(next);
  };

  const phaseNames = phases.map((p) => (p?.name || '').trim()).filter(Boolean);

  const handleFlowNodeClick = (idx) => {
    setExpandedPhase(expandedPhase === idx ? null : idx);
  };

  return (
    <Box>
      {!readOnly && (
        <FormControl size="small" fullWidth sx={{ mb: 2 }}>
          <InputLabel>Start from template</InputLabel>
          <Select
            value=""
            label="Start from template"
            onChange={(e) => handleTemplateChange(e.target.value)}
            disabled={readOnly}
          >
            {STARTER_TEMPLATES.map((opt) => (
              <MenuItem key={opt.value || 'none'} value={opt.value || ''}>
                {opt.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      {phases.length > 0 && (
        <Box
          sx={{
            mb: 2,
            p: 1.5,
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            bgcolor: 'action.hover',
            overflowX: 'auto',
          }}
        >
          <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ display: 'block', mb: 1 }}>
            Phase flow
          </Typography>
          <Box
            component="svg"
            viewBox={`0 0 ${Math.max(320, phaseNames.length * 28)} ${phases.length * 52 + 20}`}
            width="100%"
            height={phases.length * 52 + 20}
            sx={{ minHeight: 120 }}
          >
            {phases.map((phase, idx) => {
              const pname = (phase?.name || '').trim() || `Phase ${idx + 1}`;
              const ptype = phase?.type || 'reason';
              const isEvaluate = ptype === 'evaluate';
              const isRefine = ptype === 'refine';
              const y = 26 + idx * 52;
              const w = 160;
              const x = 12;
              const nodeId = `phase-node-${idx}`;
              return (
                <g key={nodeId}>
                  {idx > 0 && (
                    <line x1={x + w / 2} y1={y - 26} x2={x + w / 2} y2={y - 14} stroke="var(--mui-palette-divider)" strokeWidth={1} />
                  )}
                  <rect
                    x={x}
                    y={y - 14}
                    width={w}
                    height={28}
                    rx={4}
                    fill={expandedPhase === idx ? 'var(--mui-palette-primary-main)' : 'var(--mui-palette-background-paper)'}
                    stroke="var(--mui-palette-primary-main)"
                    strokeWidth={expandedPhase === idx ? 2 : 1}
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleFlowNodeClick(idx)}
                  />
                  <text
                    x={x + w / 2}
                    y={y + 2}
                    textAnchor="middle"
                    fontSize={11}
                    fill={expandedPhase === idx ? 'var(--mui-palette-primary-contrastText)' : 'var(--mui-palette-text-primary)'}
                    style={{ pointerEvents: 'none', cursor: 'pointer' }}
                    onClick={() => handleFlowNodeClick(idx)}
                  >
                    {pname} ({ptype})
                  </text>
                  {isEvaluate && idx < phases.length - 1 && (
                    <text x={x + w + 6} y={y + 2} fontSize={9} fill="var(--mui-palette-text-secondary)">
                      pass→{phase?.on_pass || 'end'} / fail→{phase?.on_fail || 'end'}
                    </text>
                  )}
                  {isRefine && phase?.target && (
                    <text x={x + w + 6} y={y + 2} fontSize={9} fill="var(--mui-palette-text-secondary)">
                      → {phase.target}
                    </text>
                  )}
                  {idx < phases.length - 1 && !isEvaluate && (
                    <line x1={x + w / 2} y1={y + 14} x2={x + w / 2} y2={y + 38} stroke="var(--mui-palette-divider)" strokeWidth={1} />
                  )}
                </g>
              );
            })}
          </Box>
        </Box>
      )}
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Phases (ordered)
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Add phases to define the reasoning workflow. Reason / Synthesize / Refine use prompts; Act and Search use tools; Evaluate uses criteria and routing.
      </Typography>
      {phases.map((phase, idx) => (
        <Accordion
          key={idx}
          expanded={expandedPhase === idx}
          onChange={() => setExpandedPhase(expandedPhase === idx ? null : idx)}
          disableGutters
          sx={{ mb: 0.5, '&:before': { display: 'none' } }}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="body2">
              {idx + 1}. {phase?.name || 'Unnamed'} ({PHASE_TYPES.find((t) => t.value === (phase?.type || 'reason'))?.label || phase?.type || 'reason'})
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ flexDirection: 'column', gap: 1.5 }}>
            <TextField
              size="small"
              fullWidth
              label="Phase name"
              value={phase?.name || ''}
              onChange={(e) => updatePhase(idx, { name: e.target.value })}
              disabled={readOnly}
              placeholder="e.g. plan, draft, critique"
            />
            <FormControl size="small" fullWidth>
              <InputLabel>Phase type</InputLabel>
              <Select
                value={phase?.type || 'reason'}
                label="Phase type"
                onChange={(e) => updatePhase(idx, { type: e.target.value })}
                disabled={readOnly}
              >
                {PHASE_TYPES.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {(phase?.type || 'reason') !== 'evaluate' && (
              <TextField
                size="small"
                fullWidth
                multiline
                minRows={2}
                label="Prompt"
                value={phase?.prompt ?? ''}
                onChange={(e) => updatePhase(idx, { prompt: e.target.value })}
                disabled={readOnly}
                placeholder="Use {query}, {phase_name.output}, {editor}..."
              />
            )}

            {(phase?.type || 'reason') === 'evaluate' && (
              <>
                <TextField
                  size="small"
                  fullWidth
                  multiline
                  minRows={2}
                  label="Criteria"
                  value={phase?.criteria ?? ''}
                  onChange={(e) => updatePhase(idx, { criteria: e.target.value })}
                  disabled={readOnly}
                  placeholder="What to evaluate (e.g. quality, completeness)"
                />
                <Typography variant="caption" color="text.secondary">
                  Pass threshold: {(phase?.pass_threshold ?? 0.7) * 100}%
                </Typography>
                <Slider
                  value={phase?.pass_threshold ?? 0.7}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(_, v) => updatePhase(idx, { pass_threshold: v })}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(v) => `${Math.round(v * 100)}%`}
                  disabled={readOnly}
                />
                <FormControl size="small" fullWidth>
                  <InputLabel>On pass</InputLabel>
                  <Select
                    value={phase?.on_pass ?? 'end'}
                    label="On pass"
                    onChange={(e) => updatePhase(idx, { on_pass: e.target.value })}
                    disabled={readOnly}
                  >
                    <MenuItem value="end">End</MenuItem>
                    {phaseNames.filter((n) => n !== (phase?.name || '').trim()).map((n) => (
                      <MenuItem key={n} value={n}>{n}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl size="small" fullWidth>
                  <InputLabel>On fail</InputLabel>
                  <Select
                    value={phase?.on_fail ?? ''}
                    label="On fail"
                    onChange={(e) => updatePhase(idx, { on_fail: e.target.value })}
                    disabled={readOnly}
                  >
                    <MenuItem value="">End</MenuItem>
                    {phaseNames.map((n) => (
                      <MenuItem key={n} value={n}>{n}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <TextField
                  size="small"
                  type="number"
                  label="Max retries"
                  value={phase?.max_retries ?? 2}
                  onChange={(e) => updatePhase(idx, { max_retries: Math.max(0, parseInt(e.target.value, 10) || 0) })}
                  inputProps={{ min: 0, max: 10 }}
                  disabled={readOnly}
                />
              </>
            )}

            {(phase?.type || 'reason') === 'refine' && (
              <>
                <FormControl size="small" fullWidth>
                  <InputLabel>Target phase</InputLabel>
                  <Select
                    value={phase?.target ?? ''}
                    label="Target phase"
                    onChange={(e) => updatePhase(idx, { target: e.target.value })}
                    disabled={readOnly}
                  >
                    {phaseNames.filter((n) => n !== (phase?.name || '').trim()).map((n) => (
                      <MenuItem key={n} value={n}>{n}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl size="small" fullWidth>
                  <InputLabel>Next phase</InputLabel>
                  <Select
                    value={phase?.next ?? ''}
                    label="Next phase"
                    onChange={(e) => updatePhase(idx, { next: e.target.value })}
                    disabled={readOnly}
                  >
                    {phaseNames.map((n) => (
                      <MenuItem key={n} value={n}>{n}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </>
            )}

            {(phase?.type || 'reason') === 'act' && (
              <Box>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={!(Array.isArray(phase?.available_tools) && phase.available_tools.length > 0)}
                      disabled={readOnly}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updatePhase(idx, { available_tools: undefined });
                        } else {
                          updatePhase(idx, { available_tools: stepPaletteTools?.length ? [...stepPaletteTools] : [] });
                        }
                      }}
                    />
                  }
                  label="Inherit all step-level tools"
                  sx={{ display: 'block', mb: 0.5 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  {!(Array.isArray(phase?.available_tools) && phase.available_tools.length > 0)
                    ? 'This phase uses all tools from the step palette.'
                    : 'Restrict to a subset of step-level tools:'}
                </Typography>
                {(!(Array.isArray(phase?.available_tools) && phase.available_tools.length > 0) && stepPaletteTools?.length > 0) ? (
                  <Typography variant="body2" color="text.secondary" sx={{ py: 0.5 }}>
                    Using: {stepPaletteTools.join(', ')}
                  </Typography>
                ) : !(Array.isArray(phase?.available_tools) && phase.available_tools.length > 0) && (!stepPaletteTools || stepPaletteTools.length === 0) ? (
                  <Typography variant="body2" color="text.secondary" sx={{ py: 0.5 }}>
                    No step-level tools selected. Add tools in the step-level palette above.
                  </Typography>
                ) : (
                  <CollapsibleToolPicker
                    actions={actionsFilteredByPalette.filter((a) => {
                      const name = typeof a === 'string' ? a : a?.name;
                      return name && !String(name).startsWith('agent:');
                    })}
                    selectedTools={phase?.available_tools || []}
                    onToggleTool={(next) => updatePhase(idx, { available_tools: next })}
                  />
                )}
              </Box>
            )}

            {(phase?.type || 'reason') === 'search' && (
              <>
                <FormControl size="small" fullWidth>
                  <InputLabel>Strategy</InputLabel>
                  <Select
                    value={phase?.strategy ?? 'parallel'}
                    label="Strategy"
                    onChange={(e) => updatePhase(idx, { strategy: e.target.value })}
                    disabled={readOnly}
                  >
                    <MenuItem value="parallel">Parallel</MenuItem>
                    <MenuItem value="sequential">Sequential</MenuItem>
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={!(Array.isArray(phase?.search_tools) && phase.search_tools.length > 0)}
                      disabled={readOnly}
                      onChange={(e) => {
                        if (e.target.checked) {
                          updatePhase(idx, { search_tools: undefined, available_tools: undefined });
                        } else {
                          const tools = stepPaletteTools?.length ? [...stepPaletteTools] : [];
                          updatePhase(idx, { search_tools: tools, available_tools: tools });
                        }
                      }}
                    />
                  }
                  label="Inherit all step-level tools"
                  sx={{ display: 'block', mb: 0.5 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  {!(Array.isArray(phase?.search_tools) && phase.search_tools.length > 0)
                    ? 'This phase uses all tools from the step palette.'
                    : 'Restrict to a subset of step-level tools:'}
                </Typography>
                {(!(Array.isArray(phase?.search_tools) && phase.search_tools.length > 0) && stepPaletteTools?.length > 0) ? (
                  <Typography variant="body2" color="text.secondary" sx={{ py: 0.5 }}>
                    Using: {stepPaletteTools.join(', ')}
                  </Typography>
                ) : !(Array.isArray(phase?.search_tools) && phase.search_tools.length > 0) && (!stepPaletteTools || stepPaletteTools.length === 0) ? (
                  <Typography variant="body2" color="text.secondary" sx={{ py: 0.5 }}>
                    No step-level tools selected. Add tools in the step-level palette above.
                  </Typography>
                ) : (
                  <>
                    <CollapsibleToolPicker
                      actions={actionsFilteredByPalette.filter((a) => {
                        const name = typeof a === 'string' ? a : a?.name;
                        return name && !String(name).startsWith('agent:');
                      })}
                      selectedTools={phase?.search_tools || phase?.available_tools || []}
                      onToggleTool={(next) => updatePhase(idx, { search_tools: next, available_tools: next })}
                    />
                  </>
                )}
              </>
            )}

            {!readOnly && (
              <Button
                size="small"
                color="error"
                startIcon={<Delete />}
                onClick={() => removePhase(idx)}
                sx={{ alignSelf: 'flex-start' }}
              >
                Remove phase
              </Button>
            )}
          </AccordionDetails>
        </Accordion>
      ))}
      {!readOnly && (
        <Button size="small" startIcon={<Add />} onClick={addPhase} sx={{ mt: 1 }}>
          Add phase
        </Button>
      )}
    </Box>
  );
}
