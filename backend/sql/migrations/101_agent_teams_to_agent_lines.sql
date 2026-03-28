-- Rename Agent Teams to Agent Lines: tables, columns, indexes, constraints.
-- Run after 086–100. Safe for DBs that have agent_teams (and optionally agent_team_watches, team_workspace).

-- 1. Rename main tables (order: referenced first)
ALTER TABLE agent_teams RENAME TO agent_lines;
ALTER TABLE agent_team_memberships RENAME TO agent_line_memberships;
ALTER TABLE agent_team_goals RENAME TO agent_line_goals;

-- 2. Rename column in agent_line_memberships and update unique constraint (constraint follows column rename)
ALTER TABLE agent_line_memberships RENAME COLUMN team_id TO line_id;

-- 3. Rename column in agent_line_goals
ALTER TABLE agent_line_goals RENAME COLUMN team_id TO line_id;

-- 4. agent_tasks: rename column (FK to agent_teams becomes agent_lines automatically after step 1)
ALTER TABLE agent_tasks RENAME COLUMN team_id TO line_id;

-- 5. agent_messages: rename column
ALTER TABLE agent_messages RENAME COLUMN team_id TO line_id;

-- 6. agent_team_watches (if exists, from migration 047)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_team_watches') THEN
        ALTER TABLE agent_team_watches RENAME TO agent_line_watches;
        ALTER TABLE agent_line_watches RENAME COLUMN team_id TO line_id;
        ALTER TABLE agent_line_watches DROP CONSTRAINT IF EXISTS agent_team_watches_team_id_fkey;
        ALTER TABLE agent_line_watches DROP CONSTRAINT IF EXISTS agent_line_watches_line_id_fkey;
        ALTER TABLE agent_line_watches ADD CONSTRAINT agent_line_watches_line_id_fkey
            FOREIGN KEY (line_id) REFERENCES agent_lines(id) ON DELETE CASCADE;
    END IF;
END $$;

-- 7. team_workspace (if exists, from migration 094) -> agent_line_workspace
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'team_workspace') THEN
        ALTER TABLE team_workspace RENAME TO agent_line_workspace;
        ALTER TABLE agent_line_workspace RENAME COLUMN team_id TO line_id;
    END IF;
END $$;

-- 8. Rename indexes on agent_lines (formerly agent_teams)
ALTER INDEX IF EXISTS idx_agent_teams_user RENAME TO idx_agent_lines_user;
ALTER INDEX IF EXISTS idx_agent_teams_status RENAME TO idx_agent_lines_status;
ALTER INDEX IF EXISTS idx_agent_teams_handle RENAME TO idx_agent_lines_handle;

-- 9. Rename indexes on agent_line_memberships
ALTER INDEX IF EXISTS idx_agent_team_memberships_team RENAME TO idx_agent_line_memberships_line;
ALTER INDEX IF EXISTS idx_agent_team_memberships_profile RENAME TO idx_agent_line_memberships_profile;
ALTER INDEX IF EXISTS idx_agent_team_memberships_reports_to RENAME TO idx_agent_line_memberships_reports_to;

-- 10. Rename indexes on agent_line_goals
ALTER INDEX IF EXISTS idx_agent_team_goals_team RENAME TO idx_agent_line_goals_line;
ALTER INDEX IF EXISTS idx_agent_team_goals_parent RENAME TO idx_agent_line_goals_parent;
ALTER INDEX IF EXISTS idx_agent_team_goals_assigned RENAME TO idx_agent_line_goals_assigned;

-- 11. agent_tasks index (column already renamed to line_id)
ALTER INDEX IF EXISTS idx_agent_tasks_team_status RENAME TO idx_agent_tasks_line_status;

-- 12. agent_messages index
ALTER INDEX IF EXISTS idx_agent_messages_team_created RENAME TO idx_agent_messages_line_created;

-- 13. agent_line_watches index (if exists)
ALTER INDEX IF EXISTS idx_agent_team_watches_team RENAME TO idx_agent_line_watches_line;

-- 14. agent_line_workspace index (if exists)
ALTER INDEX IF EXISTS idx_team_workspace_team_id RENAME TO idx_agent_line_workspace_line_id;

-- 15. Re-add unique constraint on agent_line_memberships if needed (UNIQUE(line_id, agent_profile_id))
-- PostgreSQL keeps the constraint when renaming column; verify name. If constraint was named by column, it may need renaming.
DO $$
DECLARE
    cname text;
BEGIN
    SELECT conname INTO cname FROM pg_constraint
    WHERE conrelid = 'agent_line_memberships'::regclass AND contype = 'u';
    IF cname IS NOT NULL AND cname LIKE '%team%' THEN
        EXECUTE format('ALTER TABLE agent_line_memberships RENAME CONSTRAINT %I TO agent_line_memberships_line_id_agent_profile_id_key', cname);
    END IF;
END $$;
