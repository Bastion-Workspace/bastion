-- Migration 020: Fix conversations UPDATE RLS policy to allow title updates
-- 
-- Problem: Conversation titles are being generated but not showing up in the UI.
-- The UPDATE policy for conversations is missing a WITH CHECK clause, which
-- can cause RLS to block updates even when the USING clause passes.
--
-- Solution: Add explicit WITH CHECK clause to the UPDATE policy to ensure
-- title updates (and other updates) work correctly with RLS.

-- Drop and recreate the conversations update policy with both USING and WITH CHECK
DROP POLICY IF EXISTS conversations_update_policy ON conversations;

CREATE POLICY conversations_update_policy ON conversations
    FOR UPDATE 
    USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );


