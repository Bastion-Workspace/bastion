-- Add handle column to agent_teams for @team-name chat mentions.
-- Handle is optional; when set, must be unique across teams.

ALTER TABLE agent_teams
  ADD COLUMN IF NOT EXISTS handle VARCHAR(100);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_teams_handle
  ON agent_teams(handle)
  WHERE handle IS NOT NULL AND handle != '';
