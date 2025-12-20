-- ========================================
-- FIX MESSAGING RLS POLICIES
-- Fix circular dependency in room_participants_select_policy
-- ========================================
-- This migration fixes the room_participants_select_policy which had
-- a circular dependency that prevented recipients from seeing messages.
-- The policy was querying room_participants to check room_participants,
-- which created an infinite loop.
--
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/017_fix_messaging_rls_policies.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/017_fix_messaging_rls_policies.sql
-- ========================================

-- Drop the incorrect circular policy
DROP POLICY IF EXISTS room_participants_select_policy ON room_participants;

-- Create the correct policy: users can see their own participation records
CREATE POLICY room_participants_select_policy ON room_participants
    FOR SELECT USING (
        -- User can see their own participation records
        user_id = current_setting('app.current_user_id', false)::varchar
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- Also fix the insert policy to allow adding participants to existing rooms
-- (room creators and existing participants can add others)
DROP POLICY IF EXISTS room_participants_insert_policy ON room_participants;

CREATE POLICY room_participants_insert_policy ON room_participants
    FOR INSERT WITH CHECK (
        -- Allow if current user is creator of the room
        room_id IN (
            SELECT room_id FROM chat_rooms 
            WHERE created_by = current_setting('app.current_user_id', false)::varchar
        )
        -- OR if current user is already a participant (can add others)
        OR room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR if current user is admin
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- Fix delete policy to use false instead of true for missing_setting_is_null
DROP POLICY IF EXISTS room_participants_delete_policy ON room_participants;

CREATE POLICY room_participants_delete_policy ON room_participants
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', false)::varchar
        OR room_id IN (
            SELECT room_id FROM chat_rooms 
            WHERE created_by = current_setting('app.current_user_id', false)::varchar
        )
        OR current_setting('app.current_user_role', false) = 'admin'
    );

