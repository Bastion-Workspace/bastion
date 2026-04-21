-- Agent lines: bind Data Workspaces with per-workspace read/read_write access (additive with profile config).
ALTER TABLE agent_lines ADD COLUMN IF NOT EXISTS data_workspace_config JSONB DEFAULT '{}';
