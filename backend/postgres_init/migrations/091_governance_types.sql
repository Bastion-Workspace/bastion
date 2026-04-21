-- Extend approval queue with governance_type for structural changes (Phase 6)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_approval_queue' AND column_name = 'governance_type'
    ) THEN
        ALTER TABLE agent_approval_queue ADD COLUMN governance_type VARCHAR(50) DEFAULT 'playbook_step';
    END IF;
END $$;
