-- Skill execution metrics: events table + materialized usage stats view.

CREATE TABLE IF NOT EXISTS skill_execution_events (
    id BIGSERIAL PRIMARY KEY,
    skill_id UUID NOT NULL REFERENCES agent_skills(id) ON DELETE CASCADE,
    skill_slug VARCHAR(100) NOT NULL,
    agent_profile_id UUID,
    step_name VARCHAR(255),
    user_id VARCHAR(255) NOT NULL,
    discovery_method VARCHAR(20) NOT NULL,
    tool_calls_made INTEGER DEFAULT 0,
    success BOOLEAN,
    execution_ms INTEGER,
    skill_version INTEGER DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skill_exec_skill ON skill_execution_events(skill_id);
CREATE INDEX IF NOT EXISTS idx_skill_exec_created ON skill_execution_events(created_at);
CREATE INDEX IF NOT EXISTS idx_skill_exec_user ON skill_execution_events(user_id);
CREATE INDEX IF NOT EXISTS idx_skill_exec_slug ON skill_execution_events(skill_slug);

CREATE MATERIALIZED VIEW IF NOT EXISTS skill_usage_stats AS
SELECT
    skill_id,
    skill_slug,
    COUNT(*) AS total_uses,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT agent_profile_id) AS unique_agents,
    AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) AS success_rate,
    AVG(execution_ms) AS avg_execution_ms,
    MAX(created_at) AS last_used_at,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') AS uses_last_7d,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') AS uses_last_30d
FROM skill_execution_events
GROUP BY skill_id, skill_slug;

CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_usage_stats_skill ON skill_usage_stats(skill_id);
