-- Add data_workspace_config to agent_profiles for workspace-bound agents (schema auto-inject, context instructions).

ALTER TABLE agent_profiles
  ADD COLUMN IF NOT EXISTS data_workspace_config JSONB DEFAULT '{}';

COMMENT ON COLUMN agent_profiles.data_workspace_config IS 'Optional: workspace_ids, auto_inject_schema, context_instructions for Data Workspace-bound agents';
