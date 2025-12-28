-- ========================================
-- FIX TEAM MEMBERS SELECT POLICY
-- Allow team members to see all members of their teams
-- ========================================
-- This migration fixes the RLS policy issue that prevents team members
-- from seeing other members in their teams.
--
-- Problem: The team_members_select_policy only allowed users to see their
-- own team memberships. This meant when fetching team members, users could
-- only see themselves, not other team members.
--
-- Solution: Update team_members_select_policy to allow users to see all
-- members of teams they belong to.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/027_fix_team_members_select_policy.sql
-- Or from within container:
-- psql -U postgres -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/027_fix_team_members_select_policy.sql
-- ========================================

-- Fix team_members SELECT policy to allow team members to see all members of their teams
DROP POLICY IF EXISTS team_members_select_policy ON team_members;

CREATE POLICY team_members_select_policy ON team_members
    FOR SELECT
    USING (
        -- User can see members of teams they belong to
        team_id IN (
            SELECT tm.team_id FROM team_members tm
            WHERE tm.user_id = current_setting('app.current_user_id', true)
        )
    );

COMMENT ON POLICY team_members_select_policy ON team_members IS 'Allows users to see all members of teams they belong to';

