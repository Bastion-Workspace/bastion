-- Phase 3: team budget config, active task tracking, member color

-- agent_teams: budget_config (monthly_limit_usd, enforce_hard_limit, warning_threshold_pct), active_celery_task_id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'agent_teams' AND column_name = 'budget_config'
    ) THEN
        ALTER TABLE agent_teams ADD COLUMN budget_config JSONB DEFAULT '{}';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'agent_teams' AND column_name = 'active_celery_task_id'
    ) THEN
        ALTER TABLE agent_teams ADD COLUMN active_celery_task_id VARCHAR(255);
    END IF;
END $$;

-- agent_team_memberships: color (hex for UI differentiation)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'agent_team_memberships' AND column_name = 'color'
    ) THEN
        ALTER TABLE agent_team_memberships ADD COLUMN color VARCHAR(7);
    END IF;
END $$;

-- Backfill color for existing members (cycle palette by team and join order)
DO $$
DECLARE
    pal VARCHAR(7)[] := ARRAY['#1976d2','#00897b','#43a047','#7b1fa2','#e65100','#3949ab','#d81b60','#00838f','#f57c00','#c62828'];
    r RECORD;
    idx INT := 0;
    tid UUID := NULL;
BEGIN
    FOR r IN (SELECT id, team_id FROM agent_team_memberships WHERE color IS NULL ORDER BY team_id, joined_at)
    LOOP
        IF r.team_id IS DISTINCT FROM tid THEN
            tid := r.team_id;
            idx := 0;
        END IF;
        UPDATE agent_team_memberships SET color = pal[1 + (idx % array_length(pal, 1))] WHERE id = r.id;
        idx := idx + 1;
    END LOOP;
END $$;
