-- Add persistent user fact store for per-user AI context facts.
CREATE TABLE IF NOT EXISTS user_facts (
    id         SERIAL PRIMARY KEY,
    user_id    VARCHAR(255) NOT NULL,
    fact_key   VARCHAR(255) NOT NULL,
    value      TEXT NOT NULL,
    category   VARCHAR(100) DEFAULT 'general',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, fact_key)
);
CREATE INDEX IF NOT EXISTS idx_user_facts_user_id ON user_facts(user_id);
