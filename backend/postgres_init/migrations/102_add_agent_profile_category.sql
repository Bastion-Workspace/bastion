-- Add category column to agent_profiles for sidebar grouping (Agent Factory).
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS category VARCHAR(100);
COMMENT ON COLUMN agent_profiles.category IS 'Sidebar category/folder for Agent Factory list; optional.';
