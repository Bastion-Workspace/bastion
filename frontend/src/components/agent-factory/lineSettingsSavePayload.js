/**
 * Builds the full PUT body for agent line settings (single save).
 */

/** @param {'none'|'interval'|'cron'} scheduleType */
export function buildHeartbeatConfig({
  team,
  heartbeatEnabled,
  scheduleType,
  heartbeatTimezone,
  intervalSeconds,
  cronExpression,
  requireOpenGoals,
  maxAutonomousRuns,
  continuityWorkspaceKeys,
  continuityMaxChars,
  deliveryNotifySuccess,
  deliveryNotifyFailure,
  deliveryPublishOverwrite,
  deliveryOutputSections,
  deliveryDisclaimer,
  deliveryPublishKey,
  deliveryCanonicalKey,
  deliveryExtraInstructions,
}) {
  const prevHb =
    team?.heartbeat_config && typeof team.heartbeat_config === 'object' ? { ...team.heartbeat_config } : {};

  const keysLines = (continuityWorkspaceKeys || '')
    .split(/[\n,]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const sectionLines = (deliveryOutputSections || '')
    .split(/[\n]+/)
    .map((s) => s.trim())
    .filter(Boolean);
  const delivery = {
    notify_on_success: Boolean(deliveryNotifySuccess),
    notify_on_failure: Boolean(deliveryNotifyFailure),
    publish_workspace_overwrite: Boolean(deliveryPublishOverwrite),
  };
  if (sectionLines.length) delivery.output_sections = sectionLines;
  if ((deliveryDisclaimer || '').trim()) delivery.disclaimer_block = deliveryDisclaimer.trim();
  if ((deliveryPublishKey || '').trim()) delivery.publish_workspace_key = deliveryPublishKey.trim();
  if ((deliveryCanonicalKey || '').trim()) delivery.canonical_snapshot_key = deliveryCanonicalKey.trim();
  if ((deliveryExtraInstructions || '').trim()) delivery.extra_instructions = deliveryExtraInstructions.trim();

  const heartbeat_config = {
    ...prevHb,
    enabled: heartbeatEnabled,
    schedule_type: scheduleType,
    require_open_goals: requireOpenGoals,
    continuity_workspace_keys: keysLines,
    delivery,
  };
  if ((maxAutonomousRuns || '').trim()) {
    heartbeat_config.max_autonomous_runs = parseInt(maxAutonomousRuns, 10);
  } else {
    delete heartbeat_config.max_autonomous_runs;
  }
  if ((continuityMaxChars || '').trim()) {
    heartbeat_config.continuity_max_chars_per_key = parseInt(continuityMaxChars, 10);
  } else {
    delete heartbeat_config.continuity_max_chars_per_key;
  }

  if (scheduleType === 'cron') {
    heartbeat_config.timezone = (heartbeatTimezone || 'UTC').trim() || 'UTC';
    heartbeat_config.cron_expression = (cronExpression || '').trim();
    delete heartbeat_config.interval_seconds;
  } else if (scheduleType === 'interval') {
    const n = intervalSeconds ? parseInt(intervalSeconds, 10) : NaN;
    if (!Number.isNaN(n) && n >= 60) {
      heartbeat_config.interval_seconds = n;
    } else {
      delete heartbeat_config.interval_seconds;
    }
    delete heartbeat_config.cron_expression;
    delete heartbeat_config.timezone;
  } else {
    delete heartbeat_config.interval_seconds;
    delete heartbeat_config.cron_expression;
    delete heartbeat_config.timezone;
  }

  return heartbeat_config;
}

export function buildLineUpdatePayload({
  team,
  name,
  handle,
  description,
  missionStatement,
  status,
  monthlyLimit,
  enforceHardLimit,
  warningThresholdPct,
  heartbeatEnabled,
  scheduleType,
  heartbeatTimezone,
  intervalSeconds,
  cronExpression,
  requireOpenGoals,
  maxAutonomousRuns,
  continuityWorkspaceKeys,
  continuityMaxChars,
  deliveryNotifySuccess,
  deliveryNotifyFailure,
  deliveryPublishOverwrite,
  deliveryOutputSections,
  deliveryDisclaimer,
  deliveryPublishKey,
  deliveryCanonicalKey,
  deliveryExtraInstructions,
  governanceMode,
  governanceDescription,
  chairAgentId,
  quorumCount,
  quorumPct,
  tiebreakerAgentId,
  teamSkillIds,
  referenceConfig,
  dataWorkspaceConfig,
}) {
  const budget_config = {
    monthly_limit_usd: monthlyLimit ? parseFloat(monthlyLimit) : undefined,
    enforce_hard_limit: enforceHardLimit,
    warning_threshold_pct: warningThresholdPct ? parseInt(warningThresholdPct, 10) : 80,
  };

  const heartbeat_config = buildHeartbeatConfig({
    team,
    heartbeatEnabled,
    scheduleType,
    heartbeatTimezone,
    intervalSeconds,
    cronExpression,
    requireOpenGoals,
    maxAutonomousRuns,
    continuityWorkspaceKeys,
    continuityMaxChars,
    deliveryNotifySuccess,
    deliveryNotifyFailure,
    deliveryPublishOverwrite,
    deliveryOutputSections,
    deliveryDisclaimer,
    deliveryPublishKey,
    deliveryCanonicalKey,
    deliveryExtraInstructions,
  });

  const prev =
    team?.governance_policy && typeof team.governance_policy === 'object'
      ? { ...team.governance_policy }
      : {};
  const gp = {
    ...prev,
    description: (governanceDescription || '').trim() || undefined,
  };
  delete gp.chair_agent_id;
  delete gp.quorum_count;
  delete gp.quorum_pct;
  delete gp.tiebreaker_agent_id;
  if (governanceMode === 'committee') {
    if (chairAgentId) gp.chair_agent_id = chairAgentId;
    if (quorumCount) gp.quorum_count = parseInt(quorumCount, 10);
  }
  if (governanceMode === 'consensus') {
    if (quorumPct) gp.quorum_pct = parseInt(quorumPct, 10);
    if (tiebreakerAgentId) gp.tiebreaker_agent_id = tiebreakerAgentId;
  }

  return {
    name: (name || '').trim(),
    handle: (handle || '').trim() || null,
    description: (description || '').trim() || null,
    mission_statement: (missionStatement || '').trim() || null,
    status,
    budget_config,
    heartbeat_config,
    governance_mode: governanceMode,
    governance_policy: gp,
    team_skill_ids: teamSkillIds,
    reference_config: referenceConfig,
    data_workspace_config: dataWorkspaceConfig,
  };
}
