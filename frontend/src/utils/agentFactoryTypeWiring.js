/**
 * Type compatibility for Agent Factory input wiring.
 * Mirrors backend is_type_compatible rules so the UI only shows compatible upstream outputs.
 */
import { PROMPT_VARIABLES, isRuntimeVar } from './promptVariableManifest';

/**
 * Map JSON Schema type to tool type system (text, number, boolean, list[...], record).
 * @param {string} schemaType - e.g. "string", "integer", "array"
 * @returns {string}
 */
export function mapSchemaType(schemaType) {
  if (!schemaType) return 'text';
  const t = String(schemaType).toLowerCase();
  if (t === 'string') return 'text';
  if (t === 'integer' || t === 'number') return 'number';
  if (t === 'boolean') return 'boolean';
  if (t === 'array') return 'list[any]';
  if (t === 'object') return 'record';
  return 'text';
}

/**
 * Get output fields for a step. Uses action registry when available; otherwise fallback for llm_task, approval, or tool.
 * @param {object} step - step object with step_type, output_key, action, output_schema
 * @param {Record<string, object>} actionsByName - map action name -> action with output_fields
 * @returns {Array<{ name: string, type: string }>}
 */
export function getOutputFieldsForStep(step, actionsByName) {
  const action = actionsByName[step?.action];
  let outputFields = action?.output_fields || [];
  if (outputFields.length) return outputFields;

  const stepType = step?.step_type || 'tool';
  if (stepType === 'llm_task') {
    outputFields = [
      { name: 'formatted', type: 'text' },
      { name: 'raw', type: 'text' },
    ];
    const schemaProps = step?.output_schema?.properties;
    if (schemaProps && typeof schemaProps === 'object') {
      for (const [fname, fdef] of Object.entries(schemaProps)) {
        if (fname !== 'formatted' && fname !== 'raw') {
          const ftype = (fdef && fdef.type) ? mapSchemaType(fdef.type) : 'text';
          outputFields.push({ name: fname, type: ftype });
        }
      }
    }
    return outputFields;
  }
  if (stepType === 'approval') {
    return [{ name: 'approved', type: 'boolean' }];
  }
  if (stepType === 'deep_agent') {
    return [
      { name: 'formatted', type: 'text' },
      { name: 'phase_trace', type: 'record' },
      { name: 'raw', type: 'text' },
    ];
  }
  return [{ name: 'formatted', type: 'text' }];
}

/**
 * Check if sourceType can be wired to targetType (with coercion).
 * @param {string} sourceType - e.g. "text", "number", "list[record]"
 * @param {string} targetType - e.g. "text", "number"
 * @returns {boolean}
 */
export function isTypeCompatible(sourceType, targetType) {
  if (!sourceType || !targetType) return true;
  if (sourceType === targetType) return true;
  if (targetType === 'any' || sourceType === 'any') return true;
  if (targetType === 'text') return true;
  if (sourceType === 'text') {
    if (targetType.startsWith('list[')) return true;
    return ['number', 'boolean', 'date', 'record', 'any'].includes(targetType);
  }
  if (sourceType === 'number' && targetType === 'boolean') return true;
  if (sourceType === 'file_ref') return ['text', 'record', 'any'].includes(targetType);
  if (sourceType.startsWith('list[') && targetType === 'text') return true;
  if (sourceType.startsWith('list[') && targetType.startsWith('list[')) {
    const innerSource = sourceType.slice(5, -1);
    const innerTarget = targetType.slice(5, -1);
    return isTypeCompatible(innerSource, innerTarget);
  }
  if (sourceType === 'record' && targetType === 'text') return true;
  return false;
}

/**
 * Expand a step into the list of steps whose outputs are available for wiring.
 * Parallel: children in parallel_steps. Branch: children in then_steps and else_steps (both shown; only one path runs at runtime).
 * @param {object} step - playbook step
 * @returns {Array<{ step: object, labelSuffix?: string }>}
 */
function getStepsWithOutputs(step) {
  if (!step) return [];
  if (step.step_type === 'parallel' && Array.isArray(step.parallel_steps)) {
    return step.parallel_steps.map((s) => ({ step: s }));
  }
  if (step.step_type === 'branch') {
    const then = (step.then_steps || []).map((s) => ({ step: s, labelSuffix: ' (THEN)' }));
    const el = (step.else_steps || []).map((s) => ({ step: s, labelSuffix: ' (ELSE)' }));
    return [...then, ...el];
  }
  return [{ step }];
}

/**
 * Build compatible upstream wiring options for a given input.
 * @param {string} targetInputType - type of the input we're wiring to
 * @param {Array<{ output_key: string, action: string }>} upstreamSteps - steps before current
 * @param {Record<string, { name: string, input_fields?: Array<{name, type}>, output_fields?: Array<{name, type}> }>} actionsByName - map action name -> action with input_fields, output_fields
 * @returns {Array<{ value: string, label: string, type?: string }>} options for dropdown, value is e.g. "{step_1.formatted}"
 */
export function getCompatibleUpstreamOptions(targetInputType, upstreamSteps, actionsByName) {
  const options = [];
  for (const s of upstreamSteps) {
    for (const { step, labelSuffix } of getStepsWithOutputs(s)) {
      const key = step.output_key || step.name || step.action || '';
      if (!key) continue;
      const outputFields = getOutputFieldsForStep(step, actionsByName);
      for (const f of outputFields) {
        const type = f.type || 'text';
        if (isTypeCompatible(type, targetInputType)) {
          options.push({
            value: `{${key}.${f.name}}`,
            label: `${key}.${f.name}${labelSuffix || ''}`,
            type,
          });
        }
      }
    }
  }
  return options;
}

/**
 * Build grouped wiring options: Upstream Step Outputs, Playbook Inputs, Runtime Variables.
 * Each option has compatible flag for highlighting.
 */
export function getGroupedWireOptions(targetInputType, upstreamSteps, actionsByName, playbookInputs = []) {
  const upstream = [];
  for (const s of upstreamSteps) {
    for (const { step, labelSuffix } of getStepsWithOutputs(s)) {
      const key = step.output_key || step.name || step.action || '';
      if (!key) continue;
      const outputFields = getOutputFieldsForStep(step, actionsByName);
      const options = outputFields.map((f) => {
        const type = f.type || 'text';
        return {
          value: `{${key}.${f.name}}`,
          label: `${key}.${f.name}${labelSuffix || ''}`,
          type,
          compatible: isTypeCompatible(type, targetInputType),
        };
      });
      if (options.length) upstream.push({ stepKey: key, options });
    }
  }
  const playbook = (playbookInputs || []).map((p) => {
    const type = (p.type || 'string').toLowerCase();
    const name = typeof p === 'object' && p.name ? p.name : String(p);
    return {
      value: `{${name}}`,
      label: name,
      type,
      compatible: isTypeCompatible(type, targetInputType),
    };
  });
  const runtime = PROMPT_VARIABLES.map((v) => ({
    value: `{${v.key}}`,
    label: v.key,
    type: v.group === 'datetime' ? 'date' : 'text',
    compatible: isTypeCompatible(v.group === 'datetime' ? 'date' : 'text', targetInputType),
    alwaysAvailable: v.alwaysAvailable,
    requiresOpenFile: v.requiresOpenFile,
    scheduleOnly: v.scheduleOnly,
  }));
  return { upstream, playbookInputs: playbook, runtime };
}

/**
 * Index actions by name for lookup.
 * @param {Array} actions - list from API (objects with name, input_fields, output_fields)
 * @returns {Record<string, object>}
 */
export function indexActionsByName(actions) {
  const byName = {};
  if (!Array.isArray(actions)) return byName;
  for (const a of actions) {
    const name = typeof a === 'string' ? a : a?.name;
    if (name) byName[name] = a;
  }
  return byName;
}

/**
 * Extract {ref} placeholders from a prompt template (e.g. for LLM task steps).
 * @param {string} template - prompt template string
 * @returns {string[]} unique placeholder refs, e.g. ["weather.formatted", "step_1.formatted"]
 */
export function extractPromptPlaceholders(template) {
  if (!template || typeof template !== 'string') return [];
  const matches = [...template.matchAll(/\{([^}]+)\}/g)];
  const refs = matches.map((m) => m[1].trim()).filter(Boolean);
  return [...new Set(refs)];
}

const VALID_STEP_TYPES = new Set([
  'tool',
  'llm_task',
  'llm_agent',
  'approval',
  'loop',
  'parallel',
  'branch',
  'deep_agent',
  'browser_authenticate',
]);

/** Same-phase refs allowed in deep_agent prompts (mirrors backend DEEP_AGENT_SISTER_PHASE_OUTPUT_FIELDS). */
const DEEP_SISTER_PHASE_FIELDS = new Set(['output', 'feedback', 'score', 'pass']);

function stepHasNonemptyCondition(step) {
  const c = step?.condition;
  if (c == null) return false;
  return String(c).trim().length > 0;
}

function stepExclusiveSet(step) {
  return !!step?.exclusive;
}

function stepTypeForExclusiveWarn(step) {
  return String(step?.step_type || step?.type || '').trim().toLowerCase();
}

/**
 * Warn when 2+ consecutive steps have a condition but not `exclusive`, followed by an unconditional step.
 * @param {Array<object>} steps - top-level playbook steps
 * @returns {Array<{ stepIndex: number, stepName: string, inputKey: string, message: string }>}
 */
export function validateExclusiveConditions(steps) {
  const errors = [];
  if (!Array.isArray(steps) || steps.length < 3) return errors;
  const n = steps.length;
  let i = 0;
  while (i < n) {
    const step = steps[i];
    if (!step || typeof step !== 'object' || !stepHasNonemptyCondition(step) || stepExclusiveSet(step)) {
      i += 1;
      continue;
    }
    let j = i;
    const runIndices = [];
    while (j < n) {
      const s = steps[j];
      if (!s || typeof s !== 'object' || !stepHasNonemptyCondition(s) || stepExclusiveSet(s)) break;
      runIndices.push(j);
      j += 1;
    }
    if (runIndices.length >= 2 && j < n) {
      const nxt = steps[j];
      if (nxt && typeof nxt === 'object' && !stepHasNonemptyCondition(nxt)) {
        const allBranch = runIndices.every((k) => stepTypeForExclusiveWarn(steps[k]) === 'branch');
        if (!allBranch) {
          const names = runIndices.map((k) => {
            const st = steps[k];
            const nm = String(st?.name || st?.output_key || `step_${k}`).trim();
            return nm ? `"${nm}"` : `step_${k}`;
          });
          const qnames = names.join(', ');
          const lo = runIndices[0];
          const hi = runIndices[runIndices.length - 1];
          const catchName = String(nxt?.name || nxt?.output_key || `step_${j}`).trim() || `step_${j}`;
          const stepName = String(steps[lo]?.name || steps[lo]?.output_key || `Step ${lo + 1}`);
          errors.push({
            stepIndex: lo,
            stepName,
            inputKey: 'exclusive',
            message: `Steps ${qnames} (steps ${lo}-${hi}) have conditions but are not marked exclusive; step ${j} ("${catchName}") has no condition and will always run — even after a conditional step matches. Enable "Exclusive (stop after match)" on the conditional steps, or add a condition to "${catchName}".`,
          });
        }
      }
    }
    i = runIndices[0] + 1;
  }
  return errors;
}

/**
 * @param {string} template
 * @param {string} fieldLabel
 * @param {Array<{ key: string, step: object }>} expandedUpstream
 * @param {Record<string, object>} actionsByName
 * @param {object} currentStep
 * @param {Set<string> | null} sisterPhaseNames
 * @param {boolean} checkToolInputTypes
 * @param {string | null} toolInputKey
 * @returns {Array<{ inputKey: string, message: string }>}
 */
function collectTemplateRefIssues(
  template,
  fieldLabel,
  expandedUpstream,
  actionsByName,
  currentStep,
  sisterPhaseNames,
  checkToolInputTypes,
  toolInputKey,
) {
  const issues = [];
  if (typeof template !== 'string') return issues;
  const re = /\{([^}]+)\}/g;
  let m;
  while ((m = re.exec(template)) !== null) {
    const ref = m[1].trim();
    const dot = ref.indexOf('.');
    if (ref.startsWith('{')) {
      const inner = ref.replace(/^\{+/, '').trim();
      if (inner.startsWith('#') || inner.startsWith('/')) continue;
      issues.push({
        inputKey: fieldLabel,
        message:
          'Double braces detected — use single braces for references (exception: {{#var}}...{{/var}} blocks).',
      });
      continue;
    }
    if (dot === -1) {
      if (isRuntimeVar(ref)) continue;
      issues.push({
        inputKey: fieldLabel,
        message: `Invalid reference "{${ref}}": use {output_key.field_name} or a runtime variable.`,
      });
      continue;
    }
    const refStepKey = ref.slice(0, dot).trim();
    const refField = ref.slice(dot + 1).trim();
    const upstreamEntry = expandedUpstream.find((e) => e.key === refStepKey);
    const upstream = upstreamEntry?.step;
    if (upstream) {
      const outputFields = getOutputFieldsForStep(upstream, actionsByName);
      const outField = outputFields.find((f) => f.name === refField);
      if (!outField) {
        issues.push({
          inputKey: fieldLabel,
          message: `Upstream step "${refStepKey}" has no output "${refField}".`,
        });
        continue;
      }
      if (checkToolInputTypes && toolInputKey && currentStep?.step_type === 'tool') {
        const currentAction = actionsByName[currentStep?.action];
        const inputFields = currentAction?.input_fields || [];
        const inField = inputFields.find((f) => f.name === toolInputKey);
        const targetType = inField?.type || 'text';
        if (!isTypeCompatible(outField.type || 'text', targetType)) {
          issues.push({
            inputKey: fieldLabel,
            message: `Type mismatch: ${refStepKey}.${refField} is not compatible with ${toolInputKey} (${targetType})`,
          });
        }
      }
      continue;
    }
    if (sisterPhaseNames && sisterPhaseNames.has(refStepKey)) {
      if (!DEEP_SISTER_PHASE_FIELDS.has(refField)) {
        issues.push({
          inputKey: fieldLabel,
          message: `Unknown field "${refField}" for phase "${refStepKey}" in this deep_agent (use ${[...DEEP_SISTER_PHASE_FIELDS].sort().join(', ')}).`,
        });
      }
      continue;
    }
    if (sisterPhaseNames) {
      issues.push({
        inputKey: fieldLabel,
        message: `Unknown upstream step or phase "${refStepKey}" in ${fieldLabel}.`,
      });
    } else {
      issues.push({
        inputKey: fieldLabel,
        message: `Unknown upstream step "${refStepKey}" in ${fieldLabel}.`,
      });
    }
  }
  return issues;
}

/**
 * Validate playbook step wirings: references must point to upstream steps and compatible output fields.
 * Also warns when a tool step has a required input with no value wired.
 * @param {Array<{ name?: string, action?: string, output_key?: string, inputs?: Record<string, string> }>} steps
 * @param {Record<string, object>} actionsByName - map action name -> action with input_fields, output_fields
 * @returns {Array<{ stepIndex: number, stepName: string, inputKey: string, message: string }>} validation errors
 */
export function validatePlaybookWiring(steps, actionsByName) {
  const errors = [];
  if (!Array.isArray(steps) || !actionsByName) return errors;
  const stepTypesHuman = [...VALID_STEP_TYPES].sort().join(', ');
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const stepName = step?.name || step?.output_key || step?.action || `Step ${i + 1}`;
    const inputs = step?.inputs || {};
    const stepType = step?.step_type || 'tool';

    if (!VALID_STEP_TYPES.has(stepType)) {
      errors.push({
        stepIndex: i,
        stepName,
        inputKey: 'step_type',
        message: `Invalid step_type "${stepType}". Must be one of: ${stepTypesHuman}.`,
      });
    }

    // Tool steps: warn when a required input has no value
    if (stepType === 'tool' && step?.action) {
      const action = actionsByName[step.action];
      const inputFields = action?.input_fields || [];
      for (const field of inputFields) {
        if (field.required !== true) continue;
        const value = inputs[field.name];
        const isEmpty = value === undefined || value === null || String(value).trim() === '';
        if (isEmpty) {
          errors.push({
            stepIndex: i,
            stepName,
            inputKey: field.name,
            message: `Required input "${field.name}" has no value. Wire it to an upstream step (e.g. {step_1.formatted}) or enter a literal.`,
          });
        }
      }
    }

    const upstreamSteps = steps.slice(0, i);
    const expandedUpstream = [];
    for (const s of upstreamSteps) {
      for (const { step: innerStep } of getStepsWithOutputs(s)) {
        const key = innerStep.output_key || innerStep.name || innerStep.action;
        if (key) expandedUpstream.push({ key, step: innerStep });
      }
    }

    let sisterPhaseNames = null;
    if (stepType === 'deep_agent' && Array.isArray(step.phases)) {
      sisterPhaseNames = new Set(step.phases.map((p) => (p?.name || '').trim()).filter(Boolean));
    }

    const pushTemplateIssues = (tpl, fld, sisters, checkTypes, toolIk) => {
      const batch = collectTemplateRefIssues(
        tpl,
        fld,
        expandedUpstream,
        actionsByName,
        step,
        sisters,
        checkTypes,
        toolIk,
      );
      for (const iss of batch) {
        errors.push({
          stepIndex: i,
          stepName,
          inputKey: iss.inputKey,
          message: iss.message,
        });
      }
    };

    for (const [inputKey, value] of Object.entries(inputs)) {
      if (typeof value !== 'string') continue;
      pushTemplateIssues(
        value,
        inputKey,
        stepType === 'deep_agent' ? sisterPhaseNames : null,
        true,
        inputKey,
      );
    }

    if (stepType === 'llm_task' || stepType === 'llm_agent') {
      const prompt = step.prompt || step.prompt_template || '';
      pushTemplateIssues(prompt, 'prompt', null, false, null);
    }

    if (stepType === 'deep_agent') {
      (step.phases || []).forEach((phase, pj) => {
        if (!phase || typeof phase !== 'object') return;
        const pname = (phase.name || '').trim() || `phase_${pj}`;
        ['prompt', 'criteria'].forEach((key) => {
          const val = phase[key];
          if (typeof val === 'string') {
            pushTemplateIssues(val, `phases[${pj}].${key} (${pname})`, sisterPhaseNames, false, null);
          }
        });
      });
      ['output_template', 'prompt', 'prompt_template'].forEach((tk) => {
        const tv = step[tk];
        if (typeof tv === 'string' && tv.trim()) {
          pushTemplateIssues(tv, tk, sisterPhaseNames, false, null);
        }
      });
    }
  }
  return errors;
}
