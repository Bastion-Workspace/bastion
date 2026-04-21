-- ========================================
-- Agent Factory Skills System
-- agent_skills table + skill_ids on agent_profiles
-- Greenfield: DDL is also in backend/postgres_init/01_init.sql; keep this file for existing DBs.
-- ========================================

-- agent_skills: procedural knowledge for LLM steps (built-in and user-authored)
CREATE TABLE IF NOT EXISTS agent_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    procedure TEXT NOT NULL,
    required_tools TEXT[] DEFAULT '{}',
    optional_tools TEXT[] DEFAULT '{}',
    inputs_schema JSONB DEFAULT '{}',
    outputs_schema JSONB DEFAULT '{}',
    examples JSONB DEFAULT '[]',
    tags TEXT[] DEFAULT '{}',
    is_builtin BOOLEAN DEFAULT false,
    is_locked BOOLEAN DEFAULT false,
    version INTEGER DEFAULT 1,
    parent_skill_id UUID REFERENCES agent_skills(id) ON DELETE SET NULL,
    improvement_rationale TEXT,
    evidence_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_skills_user ON agent_skills(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_skills_slug ON agent_skills(slug);
CREATE INDEX IF NOT EXISTS idx_agent_skills_builtin ON agent_skills(is_builtin) WHERE is_builtin = true;
CREATE INDEX IF NOT EXISTS idx_agent_skills_category ON agent_skills(category);

-- Profile-level skill bindings (array of skill UUIDs)
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS skill_ids JSONB DEFAULT '[]'::jsonb;

COMMENT ON TABLE agent_skills IS 'Agent Factory skills: procedural knowledge injected into LLM steps';
COMMENT ON COLUMN agent_profiles.skill_ids IS 'Profile-level skill UUIDs applied to all playbook steps';
