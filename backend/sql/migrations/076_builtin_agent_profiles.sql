-- Built-in default agent profile and playbook (permanently locked, visible in AF UI).
-- Adds is_builtin to agent_profiles and custom_playbooks; seeds shared default playbook.

ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS is_builtin BOOLEAN DEFAULT false;
ALTER TABLE custom_playbooks ADD COLUMN IF NOT EXISTS is_builtin BOOLEAN DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_agent_profiles_builtin ON agent_profiles(is_builtin) WHERE is_builtin = true;
CREATE INDEX IF NOT EXISTS idx_custom_playbooks_builtin ON custom_playbooks(is_builtin) WHERE is_builtin = true;

-- Seed shared default playbook (fixed UUID for idempotency). Referenced by per-user builtin profiles.
INSERT INTO custom_playbooks (
    id,
    user_id,
    name,
    description,
    version,
    definition,
    triggers,
    is_template,
    is_locked,
    is_builtin,
    category,
    tags,
    required_connectors
) VALUES (
    '00000000-0001-4000-8000-000000000001'::uuid,
    NULL,
    'Default Agent Playbook',
    'Built-in single-step ReAct agent with skill search and research escalation',
    '1.0',
    '{"steps": [{"type": "llm_agent", "name": "main", "prompt_template": "{query}", "available_tools": ["escalate_to_research"], "auto_discover_skills": true, "max_auto_skills": 5, "max_iterations": 15}]}'::jsonb,
    '[]'::jsonb,
    true,
    true,
    true,
    'default',
    '{}',
    '{}'
) ON CONFLICT (id) DO NOTHING;
