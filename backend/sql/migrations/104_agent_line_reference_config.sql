-- Agent line reference files/folders (My Documents, Global, Teams) for pipeline context injection
ALTER TABLE agent_lines ADD COLUMN IF NOT EXISTS reference_config JSONB DEFAULT '{}';
