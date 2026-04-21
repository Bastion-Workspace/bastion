-- Migration 021: Fix room_participants INSERT RLS policy to allow creators to add themselves
-- 
-- Problem: When a user creates a room and tries to add themselves as a participant,
-- the RLS policy blocks the INSERT. This happens because:
-- 1. The room_participants_insert_policy checks if the user is the creator by querying chat_rooms
-- 2. But the chat_rooms_select_policy only allows users to see rooms they're participants in
-- 3. The user isn't a participant yet (chicken-and-egg problem!)
--
-- Solution: 
-- 1. Update chat_rooms_select_policy to allow users to see rooms they created (even if not yet participants)
-- 2. Keep the room_participants_insert_policy as-is (it should work once chat_rooms is accessible)

-- Fix chat_rooms SELECT policy to allow creators to see their rooms
DROP POLICY IF EXISTS chat_rooms_select_policy ON chat_rooms;

CREATE POLICY chat_rooms_select_policy ON chat_rooms
    FOR SELECT USING (
        -- Allow if user is a participant
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR if user is the creator (needed for adding first participant)
        OR created_by = current_setting('app.current_user_id', false)::varchar
        -- OR if user is admin
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- Ensure room_participants insert policy is correct (should already be, but ensure it's right)
DROP POLICY IF EXISTS room_participants_insert_policy ON room_participants;

CREATE POLICY room_participants_insert_policy ON room_participants
    FOR INSERT WITH CHECK (
        -- Allow if current user is creator of the room (can add any participant, including themselves)
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

