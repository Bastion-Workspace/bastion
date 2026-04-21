-- Team-level tool packs and skills; member-level additional tools.
-- team_tool_packs / team_skill_ids: JSON arrays of pack names / skill IDs applied to all team agents.
-- additional_tools: per-member JSON array of individual tool names.

ALTER TABLE agent_teams
  ADD COLUMN IF NOT EXISTS team_tool_packs JSONB NOT NULL DEFAULT '[]',
  ADD COLUMN IF NOT EXISTS team_skill_ids JSONB NOT NULL DEFAULT '[]';

ALTER TABLE agent_team_memberships
  ADD COLUMN IF NOT EXISTS additional_tools JSONB NOT NULL DEFAULT '[]';
