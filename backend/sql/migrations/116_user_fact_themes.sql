-- Fact theme clustering: group user_facts by embedding similarity for themed retrieval.

CREATE TABLE IF NOT EXISTS user_fact_themes (
    id          SERIAL PRIMARY KEY,
    user_id     VARCHAR(255) NOT NULL,
    label       VARCHAR(255) NOT NULL,
    centroid    FLOAT[],
    fact_count  INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_fact_themes_user ON user_fact_themes(user_id);

ALTER TABLE user_facts
    ADD COLUMN IF NOT EXISTS theme_id INTEGER REFERENCES user_fact_themes(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_user_facts_theme_id ON user_facts(theme_id);

ALTER TABLE agent_profiles
    ADD COLUMN IF NOT EXISTS use_themed_memory BOOLEAN DEFAULT true;
