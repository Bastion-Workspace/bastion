import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import { ExpandMore } from '@mui/icons-material';
import StepSkillsPicker from './StepSkillsPicker';
import StepSkillDiscoverySection from './StepSkillDiscoverySection';
import AgentMemoryScopeSection from './AgentMemoryScopeSection';

/**
 * Capabilities for playbook steps: skills-only layout for all step types.
 * Skills are the single unit of capability assignment.
 */
export default function StepCapabilitiesPanel({
  variant,
  step,
  setStep,
  readOnly,
  capabilityChips,
  connectionsForSkills,
  profileAllowedConnections = [],
  actions,
  advancedFooter,
  advancedToolsCaption,
  profileIncludeAgentMemory = false,
}) {
  const skillSelectedCount = Array.isArray(step?.skill_ids ?? step?.skills)
    ? (step.skill_ids ?? step.skills).length
    : 0;

  const isTaskStep = variant === 'llm_task';
  const allowFull = !isTaskStep;

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 600 }}>Capabilities</Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1, lineHeight: 1.45 }}>
        <strong>Selected skills</strong> provide tools and procedures for this step.
        {allowFull
          ? ' Catalog and Full discovery modes allow mid-run skill acquisition.'
          : ' Auto discovery can add skills matched from the prompt at run time.'}
      </Typography>
      <Accordion
        defaultExpanded
        disableGutters
        sx={{ mb: 1, '&:before': { display: 'none' } }}
      >
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, pr: 1 }}>
            <Typography variant="subtitle2" fontWeight={600}>Skills & discovery</Typography>
            {skillSelectedCount > 0 && (
              <Chip size="small" label={skillSelectedCount} color="primary" sx={{ height: 22 }} />
            )}
          </Box>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0 }}>
          <StepSkillsPicker
            step={step}
            setStep={setStep}
            readOnly={readOnly}
            connections={connectionsForSkills}
            embedded
          />
          <StepSkillDiscoverySection
            step={step}
            setStep={setStep}
            readOnly={readOnly}
            capabilityChips={capabilityChips}
            allowFullMode={allowFull}
            embedded
          />
        </AccordionDetails>
      </Accordion>

      {profileIncludeAgentMemory && (
        <AgentMemoryScopeSection step={step} setStep={setStep} readOnly={readOnly} />
      )}

      {advancedFooter}
    </Box>
  );
}
