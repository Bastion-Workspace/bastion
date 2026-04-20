import React, { useMemo } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Chip,
  Tooltip,
} from '@mui/material';
import { Lock } from '@mui/icons-material';
import agentFactoryService from '../../services/agentFactoryService';

function skillConnectionRequirementsMet(skill, activeTypesSet) {
  const req = skill.required_connection_types;
  if (!Array.isArray(req) || req.length === 0) return true;
  return req.every((t) => activeTypesSet.has(String(t || '').trim()));
}

export default function StepSkillsPicker({ step, setStep, readOnly, connections = [], embedded = false }) {
  const { data: skillsList = [] } = useQuery(
    ['agentFactorySkills'],
    () => agentFactoryService.listSkills({ include_builtin: true }),
    { staleTime: 60000 }
  );

  const activeTypesSet = useMemo(() => {
    const s = new Set();
    (connections || []).forEach((c) => {
      const t = (c.connection_type || '').trim();
      if (t) s.add(t);
    });
    return s;
  }, [connections]);

  const selectedIds = useMemo(
    () => new Set(Array.isArray(step?.skill_ids) ? step.skill_ids : Array.isArray(step?.skills) ? step.skills : []),
    [step?.skill_ids, step?.skills]
  );

  const byCategory = useMemo(() => {
    const map = {};
    (skillsList || []).forEach((sk) => {
      const cat = sk.category || 'General';
      if (!map[cat]) map[cat] = [];
      map[cat].push(sk);
    });
    return map;
  }, [skillsList]);

  const categories = useMemo(() => Object.keys(byCategory).sort(), [byCategory]);

  const toggleSkill = (skillId) => {
    if (readOnly) return;
    const next = new Set(selectedIds);
    if (next.has(skillId)) next.delete(skillId);
    else next.add(skillId);
    setStep((s) => ({ ...s, skill_ids: Array.from(next) }));
  };

  if (categories.length === 0) return null;

  return (
    <Box sx={{ mb: embedded ? 0 : 2 }}>
      {!embedded && (
        <Typography variant="subtitle2" sx={{ mb: 1 }}>Skills</Typography>
      )}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Step-level skills add procedure and auto-bind required tools for this step.
      </Typography>
      <Box sx={{ maxHeight: 200, overflowY: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
        {categories.map((cat) => (
          <Box key={cat} sx={{ mb: 1 }}>
            <Typography variant="caption" fontWeight={600} color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              {cat}
            </Typography>
            {(byCategory[cat] || []).map((skill) => {
              const checked = selectedIds.has(skill.id);
              const met = skillConnectionRequirementsMet(skill, activeTypesSet);
              const canSelect = met || checked;
              const reqTypes = Array.isArray(skill.required_connection_types) ? skill.required_connection_types : [];
              const missingLabel = reqTypes.length
                ? `Requires connection type(s): ${reqTypes.join(', ')}. Configure in Settings → External Connections.`
                : '';

              return (
                <Tooltip
                  key={skill.id}
                  title={!met && !checked ? missingLabel : ''}
                  disableHoverListener={met || checked}
                >
                  <span>
                    <FormControlLabel
                      control={(
                        <Checkbox
                          size="small"
                          checked={checked}
                          disabled={readOnly || !canSelect}
                          onChange={() => toggleSkill(skill.id)}
                          sx={{ p: 0.25, mr: 0.5 }}
                        />
                      )}
                      label={(
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 0.5,
                            flexWrap: 'wrap',
                            opacity: !met && !checked ? 0.5 : 1,
                          }}
                        >
                          {skill.is_builtin && <Lock sx={{ fontSize: 14 }} color="action" titleAccess="Built-in" />}
                          <span>{skill.name || skill.slug}</span>
                          {reqTypes.map((rt) => (
                            <Chip key={rt} size="small" label={rt} variant="outlined" sx={{ height: 18, fontSize: '0.65rem' }} />
                          ))}
                          {Array.isArray(skill.required_tools) && skill.required_tools.length > 0 && (
                            <Tooltip title={skill.required_tools.join(', ')}>
                              <Chip
                                size="small"
                                label={`+${skill.required_tools.length} tools`}
                                sx={{ height: 18, fontSize: '0.7rem', cursor: 'help' }}
                                variant="outlined"
                              />
                            </Tooltip>
                          )}
                        </Box>
                      )}
                      sx={{ display: 'flex', alignItems: 'center', mx: 0, my: 0.25, '& .MuiFormControlLabel-label': { fontSize: '0.8125rem' } }}
                    />
                  </span>
                </Tooltip>
              );
            })}
          </Box>
        ))}
      </Box>
    </Box>
  );
}
