/**
 * Phase list editor for deep_agent steps. Ordered list of phases with add/remove/reorder,
 * per-phase config: name, type (reason, act, search, evaluate, synthesize, refine, rerank), type-specific fields.
 */

import React, { useState, useMemo, useCallback } from 'react';
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
import { alpha, useTheme } from '@mui/material/styles';
import { Add, Delete, ExpandMore, DragIndicator } from '@mui/icons-material';
import CollapsibleToolPicker from './CollapsibleToolPicker';
import IsolatedPromptTemplateField from './IsolatedPromptTemplateField';
import { STEP_DRAWER_SELECT_MENU_PROPS } from './stepDrawerSelectMenuProps';

/** phase_results keys exposed for {phaseName.field} template autocomplete (see deep_agent_executor). */
const SISTER_PHASE_RESULT_FIELDS = ['output', 'feedback', 'score', 'pass'];

const PROMPT_TEMPLATE_PLACEHOLDER = 'Use {step_name.field} for upstream values. Type { for variables.';

const PHASE_TYPES = [
  { value: 'reason', label: 'Reason' },
  { value: 'act', label: 'Act' },
  { value: 'search', label: 'Search' },
  { value: 'evaluate', label: 'Evaluate' },
  { value: 'synthesize', label: 'Synthesize' },
  { value: 'refine', label: 'Refine' },
  { value: 'rerank', label: 'Rerank' },
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
  upstreamSteps = [],
  playbookInputs = [],
  actionsByName = {},
  drawerStepResetKey = '',
}) {
  const theme = useTheme();
  const phases = Array.isArray(step?.phases) ? step.phases : [];
  const [expandedPhase, setExpandedPhase] = useState(null);

  /** SVG fill/stroke do not reliably resolve MUI CSS variables; use palette + alpha for light/dark. */
  const phaseFlowPalette = useMemo(() => {
    const { primary, text, divider, mode } = theme.palette;
    const idleFillAlpha = mode === 'dark' ? 0.24 : 0.14;
    return {
      nodeIdleFill: alpha(primary.main, idleFillAlpha),
      nodeIdleStroke: primary.main,
      nodeSelectedFill: primary.main,
      nodeSelectedStroke: primary.main,
      textOnIdle: text.primary,
      textOnSelected: primary.contrastText,
      textAnnotation: text.secondary,
      connector: divider,
    };
  }, [theme]);

  const phasesFingerprint = useMemo(
    () => phases.map((p) => (p?.name ?? '').trim()).join('\0'),
    [phases]
  );

  const extraRefCompletionsByPhaseIndex = useMemo(() => {
    const n = phases.length;
    const rows = Array.from({ length: n }, () => []);
    for (let idx = 0; idx < n; idx++) {
      for (let i = 0; i < n; i++) {
        if (i === idx) continue;
        const name = (phases[i]?.name || '').trim();
        if (name) rows[idx].push({ prefix: name, fields: SISTER_PHASE_RESULT_FIELDS });
      }
    }
    return rows;
  }, [phases]);

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
    if (!t?.phases) return;
    const phasesCopy = t.phases.map((p) => ({ ...p }));
    setStep((s) => {
      const next = { ...s, phases: phasesCopy };
      if (Array.isArray(t.available_tools) && t.available_tools.length > 0) {
        next.available_tools = t.available_tools;
      }
      return next;
    });
    setExpandedPhase(0);
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

  const updatePhase = useCallback((idx, patch) => {
    setStep((s) => {
      const list = Array.isArray(s?.phases) ? s.phases : [];
      const next = [...list];
      next[idx] = { ...(next[idx] || {}), ...patch };
      return { ...s, phases: next };
    });
  }, [setStep]);

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
            MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
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
                    <line
                      x1={x + w / 2}
                      y1={y - 26}
                      x2={x + w / 2}
                      y2={y - 14}
                      stroke={phaseFlowPalette.connector}
                      strokeWidth={1}
                    />
                  )}
                  <rect
                    x={x}
                    y={y - 14}
                    width={w}
                    height={28}
                    rx={4}
                    fill={expandedPhase === idx ? phaseFlowPalette.nodeSelectedFill : phaseFlowPalette.nodeIdleFill}
                    stroke={expandedPhase === idx ? phaseFlowPalette.nodeSelectedStroke : phaseFlowPalette.nodeIdleStroke}
                    strokeWidth={expandedPhase === idx ? 2 : 1}
                    style={{ cursor: 'pointer' }}
                    onClick={() => handleFlowNodeClick(idx)}
                  />
                  <text
                    x={x + w / 2}
                    y={y + 2}
                    textAnchor="middle"
                    fontSize={11}
                    fill={expandedPhase === idx ? phaseFlowPalette.textOnSelected : phaseFlowPalette.textOnIdle}
                    style={{ pointerEvents: 'none', cursor: 'pointer' }}
                    onClick={() => handleFlowNodeClick(idx)}
                  >
                    {pname} ({ptype})
                  </text>
                  {isEvaluate && idx < phases.length - 1 && (
                    <text x={x + w + 6} y={y + 2} fontSize={9} fill={phaseFlowPalette.textAnnotation}>
                      pass→{phase?.on_pass || 'end'} / fail→{phase?.on_fail || 'end'}
                    </text>
                  )}
                  {isRefine && phase?.target && (
                    <text x={x + w + 6} y={y + 2} fontSize={9} fill={phaseFlowPalette.textAnnotation}>
                      → {phase.target}
                    </text>
                  )}
                  {idx < phases.length - 1 && !isEvaluate && (
                    <line
                      x1={x + w / 2}
                      y1={y + 14}
                      x2={x + w / 2}
                      y2={y + 38}
                      stroke={phaseFlowPalette.connector}
                      strokeWidth={1}
                    />
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
        Add phases to define the reasoning workflow. Reason / Synthesize / Refine use prompts; Act and Search use
        tools; Evaluate uses criteria and routing; Rerank reorders raw_results from a search phase (needs
        rerank_documents in the step tool palette).
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
                MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
              >
                {PHASE_TYPES.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {(phase?.type || 'reason') !== 'evaluate' && (phase?.type || 'reason') !== 'rerank' && (
              <>
                <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                  Prompt
                </Typography>
                <IsolatedPromptTemplateField
                  resetKey={`${drawerStepResetKey}|ph-prompt|${idx}|${phasesFingerprint}`}
                  seedPrompt={phase?.prompt ?? ''}
                  onCommit={(val) => updatePhase(idx, { prompt: val })}
                  readOnly={readOnly}
                  label="Prompt"
                  minLines={2}
                  upstreamSteps={upstreamSteps}
                  playbookInputs={playbookInputs}
                  actionsByName={actionsByName}
                  extraRefCompletions={extraRefCompletionsByPhaseIndex[idx]}
                  placeholder={PROMPT_TEMPLATE_PLACEHOLDER}
                />
              </>
            )}

            {(phase?.type || 'reason') === 'evaluate' && (
              <>
                <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                  Criteria
                </Typography>
                <IsolatedPromptTemplateField
                  resetKey={`${drawerStepResetKey}|ph-criteria|${idx}|${phasesFingerprint}`}
                  seedPrompt={phase?.criteria ?? ''}
                  onCommit={(val) => updatePhase(idx, { criteria: val })}
                  readOnly={readOnly}
                  label="Criteria"
                  minLines={2}
                  upstreamSteps={upstreamSteps}
                  playbookInputs={playbookInputs}
                  actionsByName={actionsByName}
                  extraRefCompletions={extraRefCompletionsByPhaseIndex[idx]}
                  placeholder="What to evaluate (e.g. quality, completeness). Use {query}, {phase_name.output}, or {{#var}}…{{/var}}."
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
                    MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
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
                    MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
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
                    MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
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
                    MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
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
                    ? 'This phase inherits the full resolved tool palette for the step (same tools as leaving the checklist unchecked).'
                    : 'Only the selected tools are passed to this act phase’s mini ReAct loop (must be a subset of the step palette).'}
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
                    MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
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

            {(phase?.type || 'reason') === 'rerank' && (
              <>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  Reranks raw_results from a search phase using the rerank_documents tool. Ensure that tool is in the
                  step-level palette. Leave source empty to use the latest search phase with raw_results.
                </Typography>
                <FormControl size="small" fullWidth>
                  <InputLabel>Source search phase</InputLabel>
                  <Select
                    value={phase?.source_phase ?? ''}
                    label="Source search phase"
                    onChange={(e) => updatePhase(idx, { source_phase: e.target.value || undefined })}
                    disabled={readOnly}
                    displayEmpty
                    MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
                  >
                    <MenuItem value="">
                      <em>Auto (latest with raw_results)</em>
                    </MenuItem>
                    {phaseNames.map((n) => (
                      <MenuItem key={`src-${n}`} value={n}>
                        {n}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <TextField
                  size="small"
                  type="number"
                  label="Top N after rerank"
                  value={phase?.top_n ?? 10}
                  onChange={(e) => updatePhase(idx, { top_n: Math.max(1, parseInt(e.target.value, 10) || 10) })}
                  inputProps={{ min: 1, max: 100 }}
                  disabled={readOnly}
                  fullWidth
                />
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

      <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
          Step output
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
          What becomes this step&apos;s main result for chat and downstream <code>{'{step.formatted}'}</code>. If both
          template and phase are set, the template wins.
        </Typography>
        <FormControl size="small" fullWidth sx={{ mb: 1.5 }}>
          <InputLabel>Return output from phase</InputLabel>
          <Select
            value={step?.output_phase ?? ''}
            label="Return output from phase"
            onChange={(e) => {
              const v = e.target.value;
              setStep((s) => {
                const next = { ...s };
                if (v) {
                  next.output_phase = v;
                } else {
                  delete next.output_phase;
                }
                return next;
              });
            }}
            disabled={readOnly}
            displayEmpty
            MenuProps={STEP_DRAWER_SELECT_MENU_PROPS}
          >
            <MenuItem value="">
              <em>Auto (last non-evaluate phase)</em>
            </MenuItem>
            {phaseNames.map((n) => (
              <MenuItem key={n} value={n}>
                {n}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
          Output template (optional)
        </Typography>
        <IsolatedPromptTemplateField
          resetKey={`${drawerStepResetKey}|deep-output-template|${phasesFingerprint}`}
          seedPrompt={step?.output_template ?? ''}
          onCommit={(val) =>
            setStep((s) => {
              const next = { ...s };
              if (val && String(val).trim()) {
                next.output_template = val;
              } else {
                delete next.output_template;
              }
              return next;
            })
          }
          readOnly={readOnly}
          label="Output template"
          minLines={2}
          upstreamSteps={upstreamSteps}
          playbookInputs={playbookInputs}
          actionsByName={actionsByName}
          extraRefCompletions={phaseNames.map((name) => ({
            prefix: name,
            fields: SISTER_PHASE_RESULT_FIELDS,
          }))}
          placeholder="Leave empty to use the phase above. Example: {draft.output}\n\n---\n\n{qc.feedback}"
        />
      </Box>
    </Box>
  );
}
