-- Team heartbeat scheduling: next_beat_at, last_beat_at on agent_teams (Phase 5)

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_teams' AND column_name = 'next_beat_at'
    ) THEN
        ALTER TABLE agent_teams ADD COLUMN next_beat_at TIMESTAMPTZ;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'agent_teams' AND column_name = 'last_beat_at'
    ) THEN
        ALTER TABLE agent_teams ADD COLUMN last_beat_at TIMESTAMPTZ;
    END IF;
END $$;
