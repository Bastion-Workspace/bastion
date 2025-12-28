-- ========================================
-- FIX TEAM MEMBERS SELECT POLICY RECURSION
-- Break infinite recursion in team_members SELECT policy
-- ========================================
-- This migration fixes the infinite recursion error in team_members_select_policy.
--
-- Problem: The team_members_select_policy was querying team_members from within
-- the team_members policy itself, causing infinite recursion:
--   team_members policy -> queries team_members -> triggers policy -> queries team_members -> ...
--
-- Solution: Create a SECURITY DEFINER function to check team membership.
-- Security definer functions bypass RLS for the table they query,
-- allowing us to break the recursion.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/028_fix_team_members_select_recursion.sql
-- Or from within container:
-- psql -U postgres -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/028_fix_team_members_select_recursion.sql
-- ========================================

-- 1. Create the helper function to check team membership (bypasses RLS)
CREATE OR REPLACE FUNCTION check_team_membership(check_team_id UUID, check_user_id VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    -- This function runs with the privileges of the creator (postgres)
    -- and thus ignores RLS policies on the tables it queries.
    RETURN EXISTS (
        SELECT 1 FROM team_members
        WHERE team_id = check_team_id 
        AND user_id = check_user_id
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 2. Update team_members SELECT policy to use the function (breaks recursion)
DROP POLICY IF EXISTS team_members_select_policy ON team_members;

CREATE POLICY team_members_select_policy ON team_members
    FOR SELECT
    USING (
        -- User can see members of teams they belong to (using function to break recursion)
        check_team_membership(team_id, current_setting('app.current_user_id', true)::varchar)
    );

COMMENT ON FUNCTION check_team_membership IS 'Checks if user is a member of a team, bypassing RLS to break recursion';
COMMENT ON POLICY team_members_select_policy ON team_members IS 'Allows users to see all members of teams they belong to (uses function to prevent recursion)';

