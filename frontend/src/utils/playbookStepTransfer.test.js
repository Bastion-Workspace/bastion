import {
  buildRemapForTargetConflicts,
  rewriteRefsInString,
  preparePasteSteps,
  hasDuplicateWireKeysInSteps,
  isContiguousSortedIndices,
  collectUpstreamWireKeysAtIndex,
  collectWireKeysDeep,
} from './playbookStepTransfer';

describe('playbookStepTransfer', () => {
  test('isContiguousSortedIndices', () => {
    expect(isContiguousSortedIndices([])).toBe(false);
    expect(isContiguousSortedIndices([2])).toBe(true);
    expect(isContiguousSortedIndices([1, 2, 3])).toBe(true);
    expect(isContiguousSortedIndices([1, 3])).toBe(false);
  });

  test('hasDuplicateWireKeysInSteps detects duplicate output_keys', () => {
    const steps = [
      { step_type: 'tool', action: 'x', output_key: 'a', inputs: {} },
      { step_type: 'tool', action: 'y', output_key: 'a', inputs: {} },
    ];
    expect(hasDuplicateWireKeysInSteps(steps)).toBe(true);
    expect(hasDuplicateWireKeysInSteps([{ step_type: 'tool', output_key: 'a', inputs: {} }, { step_type: 'tool', output_key: 'b', inputs: {} }])).toBe(false);
  });

  test('buildRemapForTargetConflicts renames conflicting keys', () => {
    const target = new Set(['keep', 'dup']);
    const remap = buildRemapForTargetConflicts(['dup', 'new_only'], target);
    expect(remap.dup).toBe('dup_2');
    expect(remap.new_only).toBeUndefined();
  });

  test('rewriteRefsInString remaps step.field but not runtime vars', () => {
    const remap = { old_step: 'old_step_2' };
    expect(rewriteRefsInString('{old_step.formatted}', remap, new Set())).toBe('{old_step_2.formatted}');
    expect(rewriteRefsInString('{query}', remap, new Set())).toBe('{query}');
    expect(rewriteRefsInString('{today}', remap, new Set())).toBe('{today}');
  });

  test('rewriteRefsInString skips remap when key is upstream', () => {
    const remap = { s1: 's1_2' };
    const upstream = new Set(['s1']);
    expect(rewriteRefsInString('{s1.formatted}', remap, upstream)).toBe('{s1.formatted}');
  });

  test('preparePasteSteps merges and renames keys; keeps {s1} prompt when s1 is upstream', () => {
    const targetTop = [
      { step_type: 'tool', action: 'search_documents', output_key: 's1', inputs: { query: 'x' } },
    ];
    const clipboard = [
      {
        step_type: 'llm_task',
        action: 'llm_task',
        output_key: 's1',
        prompt_template: 'Use {s1.formatted}',
        inputs: {},
      },
    ];
    const { mergedSteps, remap, referenceIssues } = preparePasteSteps({
      clipboardSteps: clipboard,
      targetTopLevelSteps: targetTop,
      insertIndex: 1,
    });
    expect(remap.s1).toBe('s1_2');
    expect(mergedSteps).toHaveLength(2);
    const pasted = mergedSteps[1];
    expect(pasted.output_key).toBe('s1_2');
    expect(pasted.prompt_template).toContain('{s1.formatted}');
    expect(referenceIssues.length).toBe(0);
  });

  test('collectUpstreamWireKeysAtIndex', () => {
    const steps = [
      { step_type: 'tool', output_key: 'a', inputs: {} },
      {
        step_type: 'parallel',
        output_key: 'p',
        parallel_steps: [
          { step_type: 'tool', output_key: 'b', inputs: {} },
        ],
      },
    ];
    const k0 = collectUpstreamWireKeysAtIndex(steps, 0);
    expect(k0.has('a')).toBe(false);
    const k1 = collectUpstreamWireKeysAtIndex(steps, 1);
    expect(k1.has('a')).toBe(true);
    const k2 = collectUpstreamWireKeysAtIndex(steps, 2);
    expect(k2.has('b')).toBe(true);
  });

  test('collectWireKeysDeep includes nested parallel children', () => {
    const steps = [
      {
        step_type: 'parallel',
        output_key: 'par',
        parallel_steps: [{ step_type: 'tool', output_key: 'child', inputs: {} }],
      },
    ];
    const keys = collectWireKeysDeep(steps);
    expect(keys.has('par')).toBe(true);
    expect(keys.has('child')).toBe(true);
  });
});
