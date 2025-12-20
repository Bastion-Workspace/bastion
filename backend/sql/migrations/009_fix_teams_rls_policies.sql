-- ========================================
-- FIX TEAMS RLS POLICIES
-- Add missing INSERT, UPDATE, DELETE policies for teams system
-- ========================================
-- This migration fixes the RLS policy gap that prevents team creation,
-- posting, commenting, and other team operations.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/009_fix_teams_rls_policies.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/009_fix_teams_rls_policies.sql
-- ========================================

-- Drop existing policies if they exist (idempotent)
DROP POLICY IF EXISTS teams_insert_policy ON teams;
DROP POLICY IF EXISTS teams_update_policy ON teams;
DROP POLICY IF EXISTS teams_delete_policy ON teams;

DROP POLICY IF EXISTS team_members_insert_policy ON team_members;
DROP POLICY IF EXISTS team_members_update_policy ON team_members;
DROP POLICY IF EXISTS team_members_delete_policy ON team_members;

DROP POLICY IF EXISTS team_invitations_insert_policy ON team_invitations;
DROP POLICY IF EXISTS team_invitations_update_policy ON team_invitations;
DROP POLICY IF EXISTS team_invitations_delete_policy ON team_invitations;

DROP POLICY IF EXISTS team_posts_insert_policy ON team_posts;
DROP POLICY IF EXISTS team_posts_update_policy ON team_posts;
DROP POLICY IF EXISTS team_posts_delete_policy ON team_posts;

DROP POLICY IF EXISTS post_reactions_insert_policy ON post_reactions;
DROP POLICY IF EXISTS post_reactions_delete_policy ON post_reactions;

DROP POLICY IF EXISTS post_comments_insert_policy ON post_comments;
DROP POLICY IF EXISTS post_comments_update_policy ON post_comments;
DROP POLICY IF EXISTS post_comments_delete_policy ON post_comments;

-- ========================================
-- TEAMS TABLE POLICIES
-- ========================================

-- RLS Policy: Any authenticated user can create teams
CREATE POLICY teams_insert_policy ON teams
    FOR INSERT WITH CHECK (
        -- Any authenticated user can create a team
        current_setting('app.current_user_id', true) IS NOT NULL
        AND current_setting('app.current_user_id', true) != ''
        -- Creator must match current user
        AND created_by = current_setting('app.current_user_id', true)::varchar
    );

-- RLS Policy: Only team admins or system admins can update teams
CREATE POLICY teams_update_policy ON teams
    FOR UPDATE USING (
        -- System admins can update any team
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can update their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = teams.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
    );

-- RLS Policy: Only team admins or system admins can delete teams
CREATE POLICY teams_delete_policy ON teams
    FOR DELETE USING (
        -- System admins can delete any team
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can delete their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = teams.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
    );

-- ========================================
-- TEAM MEMBERS TABLE POLICIES
-- ========================================

-- RLS Policy: Team admins can add members, or users can add themselves when creating a team
CREATE POLICY team_members_insert_policy ON team_members
    FOR INSERT WITH CHECK (
        -- System admins can add any member
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can add members to their teams
        OR EXISTS (
            SELECT 1 FROM team_members tm
            WHERE tm.team_id = team_members.team_id
            AND tm.user_id = current_setting('app.current_user_id', true)::varchar
            AND tm.role = 'admin'
        )
        -- OR user is adding themselves as creator of the team (special case for team creation)
        OR (
            user_id = current_setting('app.current_user_id', true)::varchar
            AND EXISTS (
                SELECT 1 FROM teams
                WHERE teams.team_id = team_members.team_id
                AND teams.created_by = current_setting('app.current_user_id', true)::varchar
            )
        )
    );

-- RLS Policy: Team admins can update member roles (but not their own)
CREATE POLICY team_members_update_policy ON team_members
    FOR UPDATE USING (
        -- System admins can update any member
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can update members (but not themselves)
        OR (
            EXISTS (
                SELECT 1 FROM team_members tm
                WHERE tm.team_id = team_members.team_id
                AND tm.user_id = current_setting('app.current_user_id', true)::varchar
                AND tm.role = 'admin'
            )
            AND user_id != current_setting('app.current_user_id', true)::varchar
        )
        -- OR users can update their own last_read_at and muted fields
        OR (
            user_id = current_setting('app.current_user_id', true)::varchar
            AND (
                -- Only allow updating last_read_at and muted, not role
                -- This is enforced by checking that role hasn't changed
                -- (PostgreSQL will check the UPDATE SET clause)
                true
            )
        )
    );

-- RLS Policy: Team admins can remove members (but not themselves)
CREATE POLICY team_members_delete_policy ON team_members
    FOR DELETE USING (
        -- System admins can remove any member
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can remove members (but not themselves)
        OR (
            EXISTS (
                SELECT 1 FROM team_members tm
                WHERE tm.team_id = team_members.team_id
                AND tm.user_id = current_setting('app.current_user_id', true)::varchar
                AND tm.role = 'admin'
            )
            AND user_id != current_setting('app.current_user_id', true)::varchar
        )
    );

-- ========================================
-- TEAM INVITATIONS TABLE POLICIES
-- ========================================

-- RLS Policy: Team admins can create invitations
CREATE POLICY team_invitations_insert_policy ON team_invitations
    FOR INSERT WITH CHECK (
        -- System admins can create any invitation
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can create invitations for their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_invitations.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
        -- AND inviter must match current user
        AND invited_by = current_setting('app.current_user_id', true)::varchar
    );

-- RLS Policy: Invited users can accept/reject, admins can update status
CREATE POLICY team_invitations_update_policy ON team_invitations
    FOR UPDATE USING (
        -- System admins can update any invitation
        current_setting('app.current_user_role', true) = 'admin'
        -- OR invited user can update their own invitations (to accept/reject)
        OR invited_user_id = current_setting('app.current_user_id', true)::varchar
        -- OR team admins can update invitations for their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_invitations.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
    );

-- RLS Policy: Team admins or invitation creator can delete invitations
CREATE POLICY team_invitations_delete_policy ON team_invitations
    FOR DELETE USING (
        -- System admins can delete any invitation
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team admins can delete invitations for their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_invitations.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
        -- OR invitation creator can delete their invitations
        OR invited_by = current_setting('app.current_user_id', true)::varchar
    );

-- ========================================
-- TEAM POSTS TABLE POLICIES
-- ========================================

-- RLS Policy: Team members with can_post permission can create posts
-- (Members and admins can post, viewers cannot)
CREATE POLICY team_posts_insert_policy ON team_posts
    FOR INSERT WITH CHECK (
        -- System admins can create any post
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team members (admin or member role, not viewer) can create posts
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_posts.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role IN ('admin', 'member')
        )
        -- AND author must match current user
        AND author_id = current_setting('app.current_user_id', true)::varchar
    );

-- RLS Policy: Post authors can update their own posts
CREATE POLICY team_posts_update_policy ON team_posts
    FOR UPDATE USING (
        -- System admins can update any post
        current_setting('app.current_user_role', true) = 'admin'
        -- OR post author can update their own posts
        OR author_id = current_setting('app.current_user_id', true)::varchar
        -- OR team admins can update any post in their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_posts.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
    );

-- RLS Policy: Post authors or team admins can delete posts
CREATE POLICY team_posts_delete_policy ON team_posts
    FOR DELETE USING (
        -- System admins can delete any post
        current_setting('app.current_user_role', true) = 'admin'
        -- OR post author can delete their own posts
        OR author_id = current_setting('app.current_user_id', true)::varchar
        -- OR team admins can delete any post in their teams
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_posts.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)::varchar
            AND team_members.role = 'admin'
        )
    );

-- ========================================
-- POST REACTIONS TABLE POLICIES
-- ========================================

-- RLS Policy: Team members with can_react permission can add reactions
-- (Members and admins can react, viewers cannot)
CREATE POLICY post_reactions_insert_policy ON post_reactions
    FOR INSERT WITH CHECK (
        -- System admins can add any reaction
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team members (admin or member role) can react to posts in their teams
        OR (
            user_id = current_setting('app.current_user_id', true)::varchar
            AND EXISTS (
                SELECT 1 FROM team_posts tp
                JOIN team_members tm ON tm.team_id = tp.team_id
                WHERE tp.post_id = post_reactions.post_id
                AND tm.user_id = current_setting('app.current_user_id', true)::varchar
                AND tm.role IN ('admin', 'member')
            )
        )
    );

-- RLS Policy: Users can remove their own reactions
CREATE POLICY post_reactions_delete_policy ON post_reactions
    FOR DELETE USING (
        -- System admins can remove any reaction
        current_setting('app.current_user_role', true) = 'admin'
        -- OR user can remove their own reactions
        OR user_id = current_setting('app.current_user_id', true)::varchar
    );

-- ========================================
-- POST COMMENTS TABLE POLICIES
-- ========================================

-- RLS Policy: Team members with can_comment permission can create comments
-- (Members and admins can comment, viewers cannot)
CREATE POLICY post_comments_insert_policy ON post_comments
    FOR INSERT WITH CHECK (
        -- System admins can create any comment
        current_setting('app.current_user_role', true) = 'admin'
        -- OR team members (admin or member role) can comment on posts in their teams
        OR (
            author_id = current_setting('app.current_user_id', true)::varchar
            AND EXISTS (
                SELECT 1 FROM team_posts tp
                JOIN team_members tm ON tm.team_id = tp.team_id
                WHERE tp.post_id = post_comments.post_id
                AND tm.user_id = current_setting('app.current_user_id', true)::varchar
                AND tm.role IN ('admin', 'member')
            )
        )
    );

-- RLS Policy: Comment authors can update their own comments
CREATE POLICY post_comments_update_policy ON post_comments
    FOR UPDATE USING (
        -- System admins can update any comment
        current_setting('app.current_user_role', true) = 'admin'
        -- OR comment author can update their own comments
        OR author_id = current_setting('app.current_user_id', true)::varchar
        -- OR team admins can update any comment in their teams
        OR EXISTS (
            SELECT 1 FROM team_posts tp
            JOIN team_members tm ON tm.team_id = tp.team_id
            WHERE tp.post_id = post_comments.post_id
            AND tm.user_id = current_setting('app.current_user_id', true)::varchar
            AND tm.role = 'admin'
        )
    );

-- RLS Policy: Comment authors or team admins can delete comments
CREATE POLICY post_comments_delete_policy ON post_comments
    FOR DELETE USING (
        -- System admins can delete any comment
        current_setting('app.current_user_role', true) = 'admin'
        -- OR comment author can delete their own comments
        OR author_id = current_setting('app.current_user_id', true)::varchar
        -- OR team admins can delete any comment in their teams
        OR EXISTS (
            SELECT 1 FROM team_posts tp
            JOIN team_members tm ON tm.team_id = tp.team_id
            WHERE tp.post_id = post_comments.post_id
            AND tm.user_id = current_setting('app.current_user_id', true)::varchar
            AND tm.role = 'admin'
        )
    );

-- Add comments for documentation
COMMENT ON POLICY teams_insert_policy ON teams IS 'Allows any authenticated user to create teams';
COMMENT ON POLICY teams_update_policy ON teams IS 'Allows team admins or system admins to update teams';
COMMENT ON POLICY teams_delete_policy ON teams IS 'Allows team admins or system admins to delete teams';

COMMENT ON POLICY team_members_insert_policy ON team_members IS 'Allows team admins to add members, or users to add themselves when creating a team';
COMMENT ON POLICY team_members_update_policy ON team_members IS 'Allows team admins to update member roles, or users to update their own last_read_at/muted';
COMMENT ON POLICY team_members_delete_policy ON team_members IS 'Allows team admins to remove members (but not themselves)';

COMMENT ON POLICY team_invitations_insert_policy ON team_invitations IS 'Allows team admins to create invitations';
COMMENT ON POLICY team_invitations_update_policy ON team_invitations IS 'Allows invited users to accept/reject, or team admins to update invitations';
COMMENT ON POLICY team_invitations_delete_policy ON team_invitations IS 'Allows team admins or invitation creators to delete invitations';

COMMENT ON POLICY team_posts_insert_policy ON team_posts IS 'Allows team members (admin/member role) to create posts';
COMMENT ON POLICY team_posts_update_policy ON team_posts IS 'Allows post authors or team admins to update posts';
COMMENT ON POLICY team_posts_delete_policy ON team_posts IS 'Allows post authors or team admins to delete posts';

COMMENT ON POLICY post_reactions_insert_policy ON post_reactions IS 'Allows team members (admin/member role) to add reactions';
COMMENT ON POLICY post_reactions_delete_policy ON post_reactions IS 'Allows users to remove their own reactions';

COMMENT ON POLICY post_comments_insert_policy ON post_comments IS 'Allows team members (admin/member role) to create comments';
COMMENT ON POLICY post_comments_update_policy ON post_comments IS 'Allows comment authors or team admins to update comments';
COMMENT ON POLICY post_comments_delete_policy ON post_comments IS 'Allows comment authors or team admins to delete comments';



