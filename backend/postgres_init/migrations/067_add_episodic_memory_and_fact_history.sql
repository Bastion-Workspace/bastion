-- Next-gen memory: episodic memory (user_episodes) and fact change history (user_fact_history).

-- Episodic memory: conversation-derived events for "remember what we worked on?"
CREATE TABLE IF NOT EXISTS user_episodes (
    id              SERIAL PRIMARY KEY,
    user_id         VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255),
    summary         TEXT NOT NULL,
    episode_type    VARCHAR(50) DEFAULT 'general',
    agent_used      VARCHAR(255),
    tools_used      JSONB DEFAULT '[]'::jsonb,
    key_topics      JSONB DEFAULT '[]'::jsonb,
    outcome         VARCHAR(50) DEFAULT 'completed',
    embedding       FLOAT[],
    created_at      TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_episodes_user_id ON user_episodes(user_id);
CREATE INDEX IF NOT EXISTS idx_user_episodes_created_at ON user_episodes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_episodes_conversation_id ON user_episodes(conversation_id);

-- Fact change history and pending review for contradiction detection
CREATE TABLE IF NOT EXISTS user_fact_history (
    id             SERIAL PRIMARY KEY,
    user_id        VARCHAR(255) NOT NULL,
    fact_key       VARCHAR(255) NOT NULL,
    old_value      TEXT NOT NULL,
    new_value      TEXT NOT NULL,
    old_source     VARCHAR(50),
    new_source     VARCHAR(50),
    old_confidence FLOAT,
    new_confidence FLOAT,
    resolution     VARCHAR(50) DEFAULT 'auto_replaced',
    resolved_at    TIMESTAMPTZ,
    created_at     TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_fact_history_user_id ON user_fact_history(user_id);
CREATE INDEX IF NOT EXISTS idx_user_fact_history_fact_key ON user_fact_history(user_id, fact_key);
CREATE INDEX IF NOT EXISTS idx_user_fact_history_resolution ON user_fact_history(resolution);
CREATE INDEX IF NOT EXISTS idx_user_fact_history_created_at ON user_fact_history(created_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON user_episodes TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON user_fact_history TO bastion_user;
GRANT USAGE, SELECT ON SEQUENCE user_episodes_id_seq TO bastion_user;
GRANT USAGE, SELECT ON SEQUENCE user_fact_history_id_seq TO bastion_user;
