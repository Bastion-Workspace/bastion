-- Migration 024: Break circular RLS dependency with security definer function
-- 
-- Problem: chat_rooms SELECT policy depends on room_participants,
-- and room_participants SELECT policy depends on chat_rooms.
-- This circular dependency causes RLS to return empty results on initial load.
--
-- Solution: Create a SECURITY DEFINER function to check room membership.
-- Security definer functions bypass RLS for the table they query,
-- allowing us to break the recursion.

-- 1. Create the helper function
CREATE OR REPLACE FUNCTION check_room_membership(check_room_id UUID, check_user_id VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
    -- This function runs with the privileges of the creator (postgres)
    -- and thus ignores RLS policies on the tables it queries.
    RETURN EXISTS (
        SELECT 1 FROM room_participants
        WHERE room_id = check_room_id 
        AND user_id = check_user_id
    ) OR EXISTS (
        SELECT 1 FROM chat_rooms
        WHERE room_id = check_room_id
        AND created_by = check_user_id
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- 2. Update chat_rooms SELECT policy
DROP POLICY IF EXISTS chat_rooms_select_policy ON chat_rooms;
CREATE POLICY chat_rooms_select_policy ON chat_rooms
    FOR SELECT USING (
        check_room_membership(room_id, current_setting('app.current_user_id', false)::varchar)
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- 3. Update room_participants SELECT policy
DROP POLICY IF EXISTS room_participants_select_policy ON room_participants;
CREATE POLICY room_participants_select_policy ON room_participants
    FOR SELECT USING (
        check_room_membership(room_id, current_setting('app.current_user_id', false)::varchar)
        OR current_setting('app.current_user_role', false) = 'admin'
    );


