-- Enhance user_facts with provenance, TTL, and embedding for next-gen relevance filtering.
ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'user_manual';
ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS confidence FLOAT DEFAULT 1.0;
ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE user_facts ADD COLUMN IF NOT EXISTS embedding FLOAT[];

-- Agent Factory: filter which fact categories to inject per profile (empty = all).
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS include_facts_categories JSONB DEFAULT '[]'::jsonb;
