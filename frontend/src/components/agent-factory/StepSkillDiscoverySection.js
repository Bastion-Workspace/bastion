import React from 'react';
import {
  Box,
  Typography,
  TextField,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material';
import {
  effectiveSkillDiscoveryMode,
  setSkillDiscoveryMode,
  maxDiscoveredSkillsValue,
  setMaxDiscoveredSkills,
} from './stepDiscoveryUtils';

const MODE_DESCRIPTIONS = {
  off: 'Only skills you select.',
  auto: 'Discovers skills matching the prompt before the run starts.',
  catalog: 'Injects skill catalog into the prompt; the agent loads skills by name on demand.',
  full: 'Catalog + Auto + mid-run search via search_and_acquire_skills.',
};

export default function StepSkillDiscoverySection({
  step,
  setStep,
  readOnly,
  allowFullMode,
  embedded,
}) {
  const rawMode = effectiveSkillDiscoveryMode(step);
  const mode = allowFullMode ? rawMode : (rawMode === 'full' || rawMode === 'catalog' ? 'auto' : rawMode);

  const onModeChange = (_, v) => {
    if (v == null || readOnly) return;
    if (!allowFullMode && (v === 'full' || v === 'catalog')) return;
    setSkillDiscoveryMode(setStep, v);
  };

  const showMaxSkills = mode === 'auto' || mode === 'full';

  return (
    <Box sx={{ mb: embedded ? 0 : 2, mt: embedded ? 1.5 : 0 }}>
      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Skill discovery</Typography>
      <ToggleButtonGroup
        exclusive
        value={mode}
        disabled={readOnly}
        onChange={onModeChange}
        size="small"
        sx={{ mb: 1, flexWrap: 'wrap' }}
      >
        <ToggleButton value="off">Off</ToggleButton>
        <ToggleButton value="auto">Auto</ToggleButton>
        {allowFullMode && <ToggleButton value="catalog">Catalog</ToggleButton>}
        {allowFullMode && <ToggleButton value="full">Full</ToggleButton>}
      </ToggleButtonGroup>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        {MODE_DESCRIPTIONS[mode] || MODE_DESCRIPTIONS.auto}
      </Typography>
      {showMaxSkills && (
        <TextField
          fullWidth
          type="number"
          label="Max discovered skills"
          value={maxDiscoveredSkillsValue(step)}
          onChange={(e) => {
            const v = e.target.value;
            const n = v === '' ? 3 : Math.max(1, Math.min(10, Number(v) || 3));
            setMaxDiscoveredSkills(setStep, n);
          }}
          inputProps={{ min: 1, max: 10, step: 1 }}
          size="small"
          disabled={readOnly}
        />
      )}
    </Box>
  );
}
