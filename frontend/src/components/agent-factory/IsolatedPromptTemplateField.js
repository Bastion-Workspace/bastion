/**
 * Keeps prompt text in local state so typing does not re-render the parent StepConfigDrawer
 * on every keystroke. Commits to parent on a debounce for wiring / placeholder derivation.
 */
import React, { useState, useEffect, useRef, useCallback, memo } from 'react';
import PromptTemplateEditor from './PromptTemplateEditor';

function IsolatedPromptTemplateField({
  resetKey,
  seedPrompt,
  onCommit,
  debounceMs = 200,
  promptDraftRef,
  ...editorProps
}) {
  const [local, setLocal] = useState(() => seedPrompt);
  const timerRef = useRef(null);

  // Only reset when switching steps or external template patch — not when parent debounces
  useEffect(() => {
    setLocal(seedPrompt);
    if (promptDraftRef) {
      promptDraftRef.current = seedPrompt;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seedPrompt is snapshot when resetKey changes
  }, [resetKey, promptDraftRef]);

  useEffect(
    () => () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    },
    []
  );

  const handleChange = useCallback(
    (val) => {
      setLocal(val);
      if (promptDraftRef) {
        promptDraftRef.current = val;
      }
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        onCommit(val);
      }, debounceMs);
    },
    [onCommit, debounceMs, promptDraftRef]
  );

  return <PromptTemplateEditor {...editorProps} value={local} onChange={handleChange} />;
}

export default memo(IsolatedPromptTemplateField);
