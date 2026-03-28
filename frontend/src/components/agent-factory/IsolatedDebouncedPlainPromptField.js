/**
 * Multiline prompt field with local state + debounced parent commit (approval steps, etc.).
 */
import React, { useState, useEffect, useRef, useCallback, memo } from 'react';
import { TextField } from '@mui/material';

function IsolatedDebouncedPlainPromptField({
  resetKey,
  seedValue,
  onCommit,
  debounceMs = 200,
  draftRef,
  label,
  placeholder,
  minRows = 2,
  sx,
}) {
  const [local, setLocal] = useState(() => seedValue);
  const timerRef = useRef(null);

  useEffect(() => {
    setLocal(seedValue);
    if (draftRef) {
      draftRef.current = seedValue;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seedValue is snapshot when resetKey changes
  }, [resetKey, draftRef]);

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
    (e) => {
      const val = e.target.value;
      setLocal(val);
      if (draftRef) {
        draftRef.current = val;
      }
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        onCommit(val);
      }, debounceMs);
    },
    [onCommit, debounceMs, draftRef]
  );

  return (
    <TextField
      fullWidth
      multiline
      minRows={minRows}
      label={label}
      value={local}
      onChange={handleChange}
      placeholder={placeholder}
      sx={sx}
    />
  );
}

export default memo(IsolatedDebouncedPlainPromptField);
