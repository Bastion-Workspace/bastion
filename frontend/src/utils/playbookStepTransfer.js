/**
 * Copy/paste helpers for Agent Factory playbook steps (deep clone, key remapping, reference rewrite).
 */

import { isRuntimeVar } from './promptVariableManifest';

export const PLAYBOOK_STEP_CLIPBOARD_SCHEMA = 1;
export const SESSION_STORAGE_CLIPBOARD_KEY = 'bastion_playbook_steps_clipboard_v1';
export const CLIPBOARD_MIME = 'application/x-bastion-playbook-steps';

const NESTED_STEP_KEYS = ['steps', 'parallel_steps', 'then_steps', 'else_steps'];

/**
 * Wire identity used in {key.field} — matches agentFactoryTypeWiring / executor.
 * @param {object} step
 * @returns {string}
 */
export function wireKeyForStep(step) {
  if (!step || typeof step !== 'object') return '';
  const ok = (step.output_key || '').trim();
  if (ok) return ok;
  const nm = (step.name || '').trim();
  if (nm) return nm;
  const ac = (step.action || '').trim();
  return ac || '';
}

/**
 * @param {Set<string>} occupied
 * @param {string} base
 * @returns {string}
 */
export function nextAvailableKey(base, occupied) {
  if (!occupied.has(base)) return base;
  let n = 2;
  let candidate = `${base}_${n}`;
  while (occupied.has(candidate)) {
    n += 1;
    candidate = `${base}_${n}`;
  }
  return candidate;
}

/**
 * DFS visit for nested steps (not phases).
 * @param {object} step
 * @param {(s: object) => void} visitor
 */
export function walkNestedSteps(step, visitor) {
  if (!step || typeof step !== 'object') return;
  visitor(step);
  for (const k of NESTED_STEP_KEYS) {
    if (Array.isArray(step[k])) {
      for (const child of step[k]) walkNestedSteps(child, visitor);
    }
  }
}

/**
 * Collect every wire key in a step tree (top-level steps array or single subtree).
 * @param {object[]|object} stepsOrStep
 * @returns {Set<string>}
 */
export function collectWireKeysDeep(stepsOrStep) {
  const out = new Set();
  const visit = (step) => {
    walkNestedSteps(step, (s) => {
      const w = wireKeyForStep(s);
      if (w) out.add(w);
    });
  };
  if (Array.isArray(stepsOrStep)) {
    for (const s of stepsOrStep) visit(s);
  } else if (stepsOrStep) {
    visit(stepsOrStep);
  }
  return out;
}

/**
 * List wire keys in DFS order (for duplicate detection).
 * @param {object[]} steps
 * @returns {string[]}
 */
export function listWireKeysDfs(steps) {
  const list = [];
  const visit = (step) => {
    walkNestedSteps(step, (s) => {
      const w = wireKeyForStep(s);
      if (w) list.push(w);
    });
  };
  if (Array.isArray(steps)) {
    for (const s of steps) visit(s);
  }
  return list;
}

/**
 * Expand parallel/branch children like agentFactoryTypeWiring getStepsWithOutputs.
 * @param {object} step
 * @returns {object[]}
 */
export function getStepsWithOutputsFlat(step) {
  if (!step) return [];
  const st = step.step_type || step.type;
  if (st === 'parallel' && Array.isArray(step.parallel_steps)) {
    return step.parallel_steps;
  }
  if (st === 'branch') {
    return [...(step.then_steps || []), ...(step.else_steps || [])];
  }
  return [step];
}

/**
 * Upstream wire keys available when inserting at insertIndex (top-level only).
 * @param {object[]} topLevelSteps
 * @param {number} insertIndex
 * @returns {Set<string>}
 */
export function collectUpstreamWireKeysAtIndex(topLevelSteps, insertIndex) {
  const keys = new Set();
  const n = Array.isArray(topLevelSteps) ? topLevelSteps.length : 0;
  const until = Math.max(0, Math.min(insertIndex, n));
  for (let i = 0; i < until; i++) {
    const top = topLevelSteps[i];
    for (const inner of getStepsWithOutputsFlat(top)) {
      const w = wireKeyForStep(inner);
      if (w) keys.add(w);
    }
  }
  return keys;
}

/**
 * @param {object[]} steps
 * @returns {boolean}
 */
export function hasDuplicateWireKeysInSteps(steps) {
  const list = listWireKeysDfs(steps);
  return new Set(list).size !== list.length;
}

/**
 * Sorted indices form one contiguous block (required for multi-step copy).
 * @param {number[]} sortedIdx
 * @returns {boolean}
 */
export function isContiguousSortedIndices(sortedIdx) {
  if (!sortedIdx.length) return false;
  if (sortedIdx.length === 1) return true;
  for (let i = 1; i < sortedIdx.length; i++) {
    if (sortedIdx[i] !== sortedIdx[i - 1] + 1) return false;
  }
  return true;
}

/**
 * Deep clone playbook steps (JSON).
 * @param {object[]} steps
 * @returns {object[]}
 */
export function deepCloneSteps(steps) {
  if (!Array.isArray(steps)) return [];
  try {
    return JSON.parse(JSON.stringify(steps));
  } catch {
    return [];
  }
}

/**
 * Build remap for clipboard keys that conflict with target tree keys.
 * @param {Set<string>|string[]} clipboardKeys
 * @param {Set<string>} targetKeys
 * @returns {Record<string, string>} old -> new (only for keys that change)
 */
export function buildRemapForTargetConflicts(clipboardKeys, targetKeys) {
  const occupied = new Set(targetKeys);
  const remap = {};
  const keys = Array.isArray(clipboardKeys) ? clipboardKeys : [...clipboardKeys];
  for (const k of keys.sort()) {
    if (!k) continue;
    if (occupied.has(k)) {
      const nk = nextAvailableKey(k, occupied);
      occupied.add(nk);
      remap[k] = nk;
    } else {
      occupied.add(k);
    }
  }
  return remap;
}

/**
 * Apply output_key / name updates from remap (old wire key -> new wire key).
 * @param {object[]} steps
 * @param {Record<string, string>} remap
 */
export function applyRemapToStepsTree(steps, remap) {
  if (!Array.isArray(steps) || !remap || Object.keys(remap).length === 0) return;
  for (const step of steps) {
    walkNestedSteps(step, (s) => {
      const ok = (s.output_key || '').trim();
      const nm = (s.name || '').trim();
      if (ok && remap[ok]) {
        const nk = remap[ok];
        s.output_key = nk;
        if (nm === ok) s.name = nk;
        return;
      }
      if (!ok && nm && remap[nm]) {
        s.name = remap[nm];
        if (!s.output_key) s.output_key = remap[nm];
      }
    });
  }
}

/**
 * Rewrite {stepKey.field} when stepKey was remapped and does not refer to an upstream step
 * at the paste position (upstreamWireKeys). If stepKey is upstream, keep the reference so it
 * still points at the existing step (e.g. pasted llm step renamed away from colliding s1 while
 * {s1.formatted} meant the search step above).
 * @param {string} str
 * @param {Record<string, string>} remap
 * @param {Set<string>|null} upstreamWireKeys
 * @returns {string}
 */
export function rewriteRefsInString(str, remap, upstreamWireKeys = null) {
  if (typeof str !== 'string' || !str.includes('{')) return str;
  return str.replace(/\{([^}]+)\}/g, (full, inner) => {
    const trimmed = inner.trim();
    if (trimmed.startsWith('literal:')) return full;
    const dot = trimmed.indexOf('.');
    if (dot === -1) {
      return full;
    }
    const stepKey = trimmed.slice(0, dot).trim();
    const rest = trimmed.slice(dot);
    if (isRuntimeVar(stepKey)) return full;
    if (remap[stepKey] && !(upstreamWireKeys && upstreamWireKeys.has(stepKey))) {
      return `{${remap[stepKey]}${rest}}`;
    }
    return full;
  });
}

/**
 * Recursively rewrite all string values in step objects (inputs, prompts, conditions, phases).
 * @param {unknown} node
 * @param {Record<string, string>} remap
 * @param {Set<string>|null} upstreamWireKeys
 */
export function rewriteReferencesInSubtree(node, remap, upstreamWireKeys = null) {
  if (!remap || Object.keys(remap).length === 0) return;
  if (typeof node === 'string') {
    return rewriteRefsInString(node, remap, upstreamWireKeys);
  }
  if (Array.isArray(node)) {
    for (let i = 0; i < node.length; i++) {
      const v = node[i];
      if (typeof v === 'string') node[i] = rewriteRefsInString(v, remap, upstreamWireKeys);
      else rewriteReferencesInSubtree(v, remap, upstreamWireKeys);
    }
    return;
  }
  if (node && typeof node === 'object') {
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (typeof v === 'string') node[k] = rewriteRefsInString(v, remap, upstreamWireKeys);
      else rewriteReferencesInSubtree(v, remap, upstreamWireKeys);
    }
  }
}

/**
 * Scan strings for unresolved {key.xxx} references.
 * @param {unknown} node
 * @param {Set<string>} allowedKeys
 * @param {string[]} path
 * @returns {Array<{ path: string, ref: string, stepKey: string }>}
 */
export function findUnresolvedReferences(node, allowedKeys, path = []) {
  const issues = [];
  const visitStr = (s, p) => {
    if (typeof s !== 'string' || !s.includes('{')) return;
    const matches = [...s.matchAll(/\{([^}]+)\}/g)];
    for (const m of matches) {
      const trimmed = m[1].trim();
      if (trimmed.startsWith('literal:')) continue;
      const dot = trimmed.indexOf('.');
      if (dot === -1) {
        if (!isRuntimeVar(trimmed) && trimmed.length > 0) {
          issues.push({ path: p.join('.'), ref: trimmed, stepKey: trimmed, message: `Not a known runtime variable: {${trimmed}}` });
        }
        continue;
      }
      const stepKey = trimmed.slice(0, dot).trim();
      if (isRuntimeVar(stepKey)) continue;
      if (!allowedKeys.has(stepKey)) {
        issues.push({ path: p.join('.'), ref: trimmed, stepKey, message: `Unknown step "${stepKey}" in {${trimmed}}` });
      }
    }
  };
  const walk = (n, p) => {
    if (typeof n === 'string') {
      visitStr(n, p);
      return;
    }
    if (Array.isArray(n)) {
      n.forEach((item, i) => walk(item, [...p, String(i)]));
      return;
    }
    if (n && typeof n === 'object') {
      for (const k of Object.keys(n)) {
        walk(n[k], [...p, k]);
      }
    }
  };
  walk(node, path);
  return issues;
}

/**
 * Prepare paste: clone, remap vs target, rewrite strings, return validation issues.
 * @param {{
 *  clipboardSteps: object[],
 *  targetTopLevelSteps: object[],
 *  insertIndex: number,
 * }} opts
 * @returns {{
 *  mergedSteps: object[],
 *  remap: Record<string, string>,
 *  referenceIssues: Array<{ path: string, ref: string, stepKey: string, message: string }>,
 * }}
 */
export function preparePasteSteps({ clipboardSteps, targetTopLevelSteps, insertIndex }) {
  const cloned = deepCloneSteps(clipboardSteps);
  const targetKeys = collectWireKeysDeep(targetTopLevelSteps || []);
  const clipboardKeys = collectWireKeysDeep(cloned);
  const remap = buildRemapForTargetConflicts([...clipboardKeys], targetKeys);

  applyRemapToStepsTree(cloned, remap);

  const insertAt = Math.max(0, Math.min(insertIndex | 0, (targetTopLevelSteps || []).length));
  const upstream = collectUpstreamWireKeysAtIndex(targetTopLevelSteps || [], insertAt);
  rewriteReferencesInSubtree(cloned, remap, upstream);
  const pastedKeys = collectWireKeysDeep(cloned);
  const allowed = new Set([...upstream, ...pastedKeys]);

  const referenceIssues = findUnresolvedReferences(cloned, allowed);

  const top = Array.isArray(targetTopLevelSteps) ? [...targetTopLevelSteps] : [];
  const mergedSteps = [...top.slice(0, insertAt), ...cloned, ...top.slice(insertAt)];

  return {
    mergedSteps,
    remap,
    referenceIssues,
  };
}

/**
 * @param {object} payload
 * @returns {boolean}
 */
export function isValidClipboardPayload(payload) {
  if (!payload || typeof payload !== 'object') return false;
  if (payload.schemaVersion !== PLAYBOOK_STEP_CLIPBOARD_SCHEMA) return false;
  if (!Array.isArray(payload.steps)) return false;
  return payload.steps.length > 0;
}

/**
 * @param {object[]} steps
 * @returns {object}
 */
export function buildClipboardPayload(steps) {
  return {
    schemaVersion: PLAYBOOK_STEP_CLIPBOARD_SCHEMA,
    steps,
    copiedAt: Date.now(),
  };
}
