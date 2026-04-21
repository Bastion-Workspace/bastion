-- Repair additional_tools: ensure JSONB is array type (fix double-encoded string values).
UPDATE agent_team_memberships
SET additional_tools = '[]'::jsonb
WHERE jsonb_typeof(additional_tools) != 'array';
