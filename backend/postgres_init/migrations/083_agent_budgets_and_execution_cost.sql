-- Agent budgets and execution cost tracking (Phase 1 autonomous operations)

-- Add token and cost columns to agent_execution_log
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_execution_log' AND column_name = 'tokens_input'
    ) THEN
        ALTER TABLE agent_execution_log ADD COLUMN tokens_input INTEGER DEFAULT 0;
    END IF;
END $$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_execution_log' AND column_name = 'tokens_output'
    ) THEN
        ALTER TABLE agent_execution_log ADD COLUMN tokens_output INTEGER DEFAULT 0;
    END IF;
END $$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_execution_log' AND column_name = 'cost_usd'
    ) THEN
        ALTER TABLE agent_execution_log ADD COLUMN cost_usd NUMERIC(12, 6) DEFAULT 0;
    END IF;
END $$;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_execution_log' AND column_name = 'model_used'
    ) THEN
        ALTER TABLE agent_execution_log ADD COLUMN model_used VARCHAR(255);
    END IF;
END $$;

-- Per-agent budget (monthly limit, current period spend, warning threshold)
CREATE TABLE IF NOT EXISTS agent_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    monthly_limit_usd NUMERIC(12, 2),
    current_period_start DATE NOT NULL,
    current_period_spend_usd NUMERIC(12, 6) DEFAULT 0,
    warning_threshold_pct INTEGER DEFAULT 80,
    enforce_hard_limit BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_profile_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_budgets_profile ON agent_budgets(agent_profile_id);
CREATE INDEX IF NOT EXISTS idx_agent_budgets_user ON agent_budgets(user_id);
