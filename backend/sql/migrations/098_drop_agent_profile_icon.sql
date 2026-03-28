-- Remove icon column from agent_profiles (no longer used).
ALTER TABLE agent_profiles DROP COLUMN IF EXISTS icon;
