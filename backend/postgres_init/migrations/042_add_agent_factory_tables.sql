-- ========================================
-- AGENT FACTORY (Phase 0)
-- Agent profiles, data source bindings, skills, playbooks, execution log, discoveries
-- ========================================
-- Idempotent: uses CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS.
-- Same DDL is in backend/postgres_init/01_init.sql for fresh installs.
-- Run on existing DB from host: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/postgres_init/migrations/042_add_agent_factory_tables.sql
-- ========================================

-- ============================================================
-- DATA SOURCE CONNECTORS (Template Definitions)
-- Must exist before agent_data_sources
-- ============================================================

CREATE TABLE IF NOT EXISTS data_source_connectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    connector_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) DEFAULT '1.0',
    definition JSONB NOT NULL,
    is_template BOOLEAN DEFAULT false,
    requires_auth BOOLEAN DEFAULT false,
    auth_fields JSONB DEFAULT '[]',
    icon VARCHAR(50),
    category VARCHAR(100),
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_connectors_user ON data_source_connectors(user_id);
CREATE INDEX IF NOT EXISTS idx_connectors_type ON data_source_connectors(connector_type);
CREATE INDEX IF NOT EXISTS idx_connectors_template ON data_source_connectors(is_template);
CREATE INDEX IF NOT EXISTS idx_connectors_category ON data_source_connectors(category);

-- ============================================================
-- AGENT PROFILES
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    handle VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    model_preference VARCHAR(255),
    max_research_rounds INTEGER DEFAULT 3,
    system_prompt_additions TEXT,
    knowledge_config JSONB DEFAULT '{}',
    output_config JSONB DEFAULT '{}',
    default_execution_mode VARCHAR(50) DEFAULT 'hybrid',
    default_run_context VARCHAR(50) DEFAULT 'interactive',
    default_approval_policy VARCHAR(50) DEFAULT 'require',
    journal_config JSONB DEFAULT '{"auto_journal": true, "detail_level": "summary", "retention_days": 90}',
    team_config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, handle)
);

CREATE INDEX IF NOT EXISTS idx_agent_profiles_user ON agent_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_profiles_active ON agent_profiles(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_agent_profiles_handle ON agent_profiles(user_id, handle);

-- ============================================================
-- AGENT DATA SOURCE BINDINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_data_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    connector_id UUID NOT NULL REFERENCES data_source_connectors(id),
    credentials_encrypted JSONB,
    config_overrides JSONB DEFAULT '{}',
    permissions JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_sources_profile ON agent_data_sources(agent_profile_id);

-- ============================================================
-- CUSTOM PLAYBOOKS
-- ============================================================

CREATE TABLE IF NOT EXISTS custom_playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) DEFAULT '1.0',
    definition JSONB NOT NULL,
    triggers JSONB DEFAULT '[]',
    is_template BOOLEAN DEFAULT false,
    category VARCHAR(100),
    tags TEXT[] DEFAULT '{}',
    required_connectors TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_playbooks_user ON custom_playbooks(user_id);
CREATE INDEX IF NOT EXISTS idx_playbooks_triggers ON custom_playbooks USING GIN (triggers);

-- ============================================================
-- AGENT SKILL BINDINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    skill_type VARCHAR(50) NOT NULL,
    skill_reference VARCHAR(255) NOT NULL,
    priority INTEGER DEFAULT 0,
    parameters JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_skills_profile ON agent_skills(agent_profile_id);

-- ============================================================
-- EXECUTION TRACKING
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_execution_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id),
    query TEXT NOT NULL,
    strategy VARCHAR(50),
    playbook_id UUID REFERENCES custom_playbooks(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    status VARCHAR(50) DEFAULT 'running',
    connectors_called JSONB DEFAULT '[]',
    entities_discovered INTEGER DEFAULT 0,
    relationships_discovered INTEGER DEFAULT 0,
    output_destinations JSONB DEFAULT '[]',
    error_details TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_execution_log_profile ON agent_execution_log(agent_profile_id);
CREATE INDEX IF NOT EXISTS idx_execution_log_user ON agent_execution_log(user_id);
CREATE INDEX IF NOT EXISTS idx_execution_log_time ON agent_execution_log(started_at DESC);

-- ============================================================
-- DISCOVERY LOG (Entities/Relationships Found Per Execution)
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_discoveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES agent_execution_log(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id),
    discovery_type VARCHAR(50) NOT NULL,
    entity_name VARCHAR(500),
    entity_type VARCHAR(50),
    entity_neo4j_id VARCHAR(255),
    relationship_type VARCHAR(100),
    related_entity_name VARCHAR(500),
    source_connector VARCHAR(255),
    source_endpoint VARCHAR(255),
    confidence REAL,
    details JSONB DEFAULT '{}',
    discovered_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_discoveries_execution ON agent_discoveries(execution_id);
CREATE INDEX IF NOT EXISTS idx_discoveries_user ON agent_discoveries(user_id);
CREATE INDEX IF NOT EXISTS idx_discoveries_entity ON agent_discoveries(entity_name);
CREATE INDEX IF NOT EXISTS idx_discoveries_type ON agent_discoveries(discovery_type);
CREATE INDEX IF NOT EXISTS idx_discoveries_time ON agent_discoveries(discovered_at DESC);
