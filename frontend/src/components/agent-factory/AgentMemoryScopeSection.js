/**
 * Step-level agent_memory_policy: inherit | off.
 * When the profile has agent memory enabled, inherit (default) injects memory context and tools;
 * off disables both for this step.
 */

import React, { useCallback } from 'react';
import { Box, Typography, ToggleButtonGroup, ToggleButton, Alert } from '@mui/material';

export default function AgentMemoryScopeSection({ step, setStep, readOnly }) {
  const policy = (step?.agent_memory_policy || 'inherit').toLowerCase();

  const setPolicy = useCallback(
    (_e, next) => {
      if (readOnly || next == null) return;
      setStep((s) => ({ ...s, agent_memory_policy: next }));
    },
    [readOnly, setStep]
  );

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 600 }}>
        Agent memory
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, lineHeight: 1.45 }}>
        Profile memory is on for this agent. <strong>Inherit</strong> loads memory into the prompt and exposes
        memory tools on this step. <strong>Off</strong> disables memory for this step only.
      </Typography>
      <ToggleButtonGroup
        value={policy === 'off' ? 'off' : 'inherit'}
        exclusive
        onChange={setPolicy}
        size="small"
        disabled={readOnly}
        sx={{ flexWrap: 'wrap' }}
      >
        <ToggleButton value="inherit">Inherit</ToggleButton>
        <ToggleButton value="off">Off</ToggleButton>
      </ToggleButtonGroup>
      {policy === 'off' && (
        <Alert severity="info" sx={{ mt: 1 }} variant="outlined">
          This step will not include stored agent memory in the system prompt or memory read/write tools.
        </Alert>
      )}
    </Box>
  );
}
