-- ========================================
-- AGENT SCHEDULES (Phase 5 Scheduler)
-- ========================================
-- Idempotent: uses CREATE TABLE IF NOT EXISTS and ADD COLUMN IF NOT EXISTS.
-- Run on existing DB: docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/043_add_agent_schedules.sql
-- ========================================

-- ============================================================
-- AGENT SCHEDULES
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    schedule_type VARCHAR(50) NOT NULL CHECK (schedule_type IN ('cron', 'interval')),
    cron_expression VARCHAR(100),
    interval_seconds INTEGER,
    timezone VARCHAR(100) DEFAULT 'UTC',
    is_active BOOLEAN DEFAULT true,
    next_run_at TIMESTAMPTZ,
    last_run_at TIMESTAMPTZ,
    last_status VARCHAR(50),
    run_count INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    max_consecutive_failures INTEGER DEFAULT 5,
    timeout_seconds INTEGER DEFAULT 300,
    input_context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schedules_due ON agent_schedules(next_run_at) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_schedules_user ON agent_schedules(user_id);
CREATE INDEX IF NOT EXISTS idx_schedules_profile ON agent_schedules(agent_profile_id);

-- ============================================================
-- EXTEND AGENT_EXECUTION_LOG
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_execution_log' AND column_name = 'trigger_type'
    ) THEN
        ALTER TABLE agent_execution_log ADD COLUMN trigger_type VARCHAR(50) DEFAULT 'manual';
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_execution_log' AND column_name = 'schedule_id'
    ) THEN
        ALTER TABLE agent_execution_log ADD COLUMN schedule_id UUID REFERENCES agent_schedules(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_execution_trigger ON agent_execution_log(trigger_type);
CREATE INDEX IF NOT EXISTS idx_execution_running ON agent_execution_log(agent_profile_id, status) WHERE status = 'running';
