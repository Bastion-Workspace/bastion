/**
 * Skill discovery mode: off | auto | catalog | full
 * Maps to backend skill_discovery_mode + legacy flags.
 */

const VALID_MODES = new Set(['off', 'auto', 'catalog', 'full']);

export function effectiveSkillDiscoveryMode(step) {
  const dm = step?.discovery_mode;
  if (typeof dm === 'string') {
    const v = dm.trim().toLowerCase();
    if (v === 'on_demand') return 'full';
    if (VALID_MODES.has(v)) return v;
  }
  const m = step?.skill_discovery_mode;
  if (VALID_MODES.has(m)) return m;
  if (step?.inject_skill_manifest) return 'catalog';
  const auto = step?.auto_discover_skills;
  const dyn = step?.dynamic_tool_discovery;
  if (auto === false && !dyn) return 'off';
  if (dyn) return 'full';
  if (auto === false) return 'off';
  return 'auto';
}

/**
 * Persist mode and keep legacy boolean fields in sync for older clients / stored playbooks.
 */
export function setSkillDiscoveryMode(setStep, mode) {
  setStep((s) => ({
    ...s,
    discovery_mode: mode === 'full' ? 'on_demand' : mode,
    skill_discovery_mode: mode,
    auto_discover_skills: mode === 'auto' || mode === 'full',
    dynamic_tool_discovery: mode === 'full',
    inject_skill_manifest: mode === 'catalog' || mode === 'full',
  }));
}

export function maxDiscoveredSkillsValue(step) {
  if (step?.max_discovered_skills != null) return Number(step.max_discovered_skills) || 3;
  if (step?.max_auto_skills != null) return Number(step.max_auto_skills) || 3;
  if (step?.max_skill_acquisitions != null) return Number(step.max_skill_acquisitions) || 3;
  return 3;
}

export function setMaxDiscoveredSkills(setStep, n) {
  const v = Math.max(1, Math.min(10, n));
  setStep((s) => ({
    ...s,
    max_discovered_skills: v,
    max_auto_skills: v,
    max_skill_acquisitions: v,
  }));
}
