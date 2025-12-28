-- ========================================
-- FIX TEAMS SELECT POLICY FOR TEAM CREATION
-- Allow users to see teams they created even if not yet members
-- ========================================
-- This migration fixes the RLS policy issue that prevents team creation.
--
-- Problem: When a user creates a team and tries to add themselves as a member,
-- the team_members_insert_policy checks if the team exists and was created by the user.
-- But the teams_select_policy only allows users to see teams they're members of.
-- Since the user isn't a member yet (chicken-and-egg problem!), the EXISTS check fails.
--
-- Solution: Update teams_select_policy to also allow users to see teams they created
-- (even if they're not yet members). This matches the pattern used in migration 021
-- for chat_rooms_select_policy.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/026_fix_teams_select_policy_for_creation.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/026_fix_teams_select_policy_for_creation.sql
-- ========================================

-- Fix teams SELECT policy to allow creators to see their teams
DROP POLICY IF EXISTS teams_select_policy ON teams;

CREATE POLICY teams_select_policy ON teams
    FOR SELECT USING (
        -- Allow if user is a member
        EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = teams.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)
        )
        -- OR if user is the creator (needed for adding first member during team creation)
        OR created_by = current_setting('app.current_user_id', true)::varchar
    );

COMMENT ON POLICY teams_select_policy ON teams IS 'Allows users to see teams they are members of, or teams they created';

