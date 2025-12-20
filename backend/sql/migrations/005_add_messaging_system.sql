-- ========================================
-- ROOSEVELT'S MESSAGING SYSTEM MIGRATION
-- User-to-User Communication Cavalry!
-- ========================================
-- This migration can be run on existing databases without data loss
-- Idempotent design - safe to run multiple times
--
-- Usage:
-- docker exec -i <postgres-container> psql -U plato_user -d plato_knowledge_base < backend/sql/migrations/005_add_messaging_system.sql
-- Or from within container:
-- psql -U plato_user -d plato_knowledge_base -f /docker-entrypoint-initdb.d/migrations/005_add_messaging_system.sql
-- ========================================

-- Create enums for messaging system (idempotent)
DO $$ BEGIN
    CREATE TYPE room_type_enum AS ENUM ('direct', 'group');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'Type room_type_enum already exists, skipping';
END $$;

DO $$ BEGIN
    CREATE TYPE message_type_enum AS ENUM ('text', 'ai_share', 'system');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'Type message_type_enum already exists, skipping';
END $$;

DO $$ BEGIN
    CREATE TYPE user_status_enum AS ENUM ('online', 'offline', 'away');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'Type user_status_enum already exists, skipping';
END $$;

-- Chat rooms table
CREATE TABLE IF NOT EXISTS chat_rooms (
    room_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_name VARCHAR(255),
    room_type room_type_enum NOT NULL DEFAULT 'direct',
    created_by VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Room participants table
CREATE TABLE IF NOT EXISTS room_participants (
    room_id UUID NOT NULL REFERENCES chat_rooms(room_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_read_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    notification_settings JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (room_id, user_id)
);

-- Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_id UUID NOT NULL REFERENCES chat_rooms(room_id) ON DELETE CASCADE,
    sender_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    message_content TEXT NOT NULL,
    message_type message_type_enum NOT NULL DEFAULT 'text',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE,
    encryption_version INTEGER DEFAULT 0
);

-- Message reactions table
CREATE TABLE IF NOT EXISTS message_reactions (
    reaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    emoji VARCHAR(10) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(message_id, user_id, emoji)
);

-- Room encryption keys table (for future E2EE with agent as participant)
CREATE TABLE IF NOT EXISTS room_encryption_keys (
    room_id UUID PRIMARY KEY REFERENCES chat_rooms(room_id) ON DELETE CASCADE,
    encrypted_key TEXT,
    key_version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User presence table
CREATE TABLE IF NOT EXISTS user_presence (
    user_id VARCHAR(255) PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    status user_status_enum NOT NULL DEFAULT 'offline',
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status_message TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for messaging performance (idempotent with IF NOT EXISTS)
CREATE INDEX IF NOT EXISTS idx_chat_rooms_created_by ON chat_rooms(created_by);
CREATE INDEX IF NOT EXISTS idx_chat_rooms_last_message_at ON chat_rooms(last_message_at DESC);
CREATE INDEX IF NOT EXISTS idx_room_participants_user_id ON room_participants(user_id);
CREATE INDEX IF NOT EXISTS idx_room_participants_room_id ON room_participants(room_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_room_id ON chat_messages(room_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_sender_id ON chat_messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_room_created ON chat_messages(room_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_message_reactions_message_id ON message_reactions(message_id);
CREATE INDEX IF NOT EXISTS idx_message_reactions_user_id ON message_reactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_presence_status ON user_presence(status);
CREATE INDEX IF NOT EXISTS idx_user_presence_last_seen ON user_presence(last_seen_at DESC);

-- Grant permissions on messaging tables
GRANT ALL PRIVILEGES ON chat_rooms TO plato_user;
GRANT ALL PRIVILEGES ON room_participants TO plato_user;
GRANT ALL PRIVILEGES ON chat_messages TO plato_user;
GRANT ALL PRIVILEGES ON message_reactions TO plato_user;
GRANT ALL PRIVILEGES ON room_encryption_keys TO plato_user;
GRANT ALL PRIVILEGES ON user_presence TO plato_user;

-- Enable RLS on messaging tables (idempotent)
ALTER TABLE chat_rooms ENABLE ROW LEVEL SECURITY;
ALTER TABLE room_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_reactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE room_encryption_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_presence ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotent re-runs)
DROP POLICY IF EXISTS chat_rooms_select_policy ON chat_rooms;
DROP POLICY IF EXISTS chat_rooms_insert_policy ON chat_rooms;
DROP POLICY IF EXISTS chat_rooms_update_policy ON chat_rooms;
DROP POLICY IF EXISTS chat_rooms_delete_policy ON chat_rooms;

DROP POLICY IF EXISTS room_participants_select_policy ON room_participants;
DROP POLICY IF EXISTS room_participants_insert_policy ON room_participants;
DROP POLICY IF EXISTS room_participants_delete_policy ON room_participants;

DROP POLICY IF EXISTS chat_messages_select_policy ON chat_messages;
DROP POLICY IF EXISTS chat_messages_insert_policy ON chat_messages;
DROP POLICY IF EXISTS chat_messages_update_policy ON chat_messages;
DROP POLICY IF EXISTS chat_messages_delete_policy ON chat_messages;

DROP POLICY IF EXISTS message_reactions_select_policy ON message_reactions;
DROP POLICY IF EXISTS message_reactions_insert_policy ON message_reactions;
DROP POLICY IF EXISTS message_reactions_delete_policy ON message_reactions;

DROP POLICY IF EXISTS room_encryption_keys_select_policy ON room_encryption_keys;
DROP POLICY IF EXISTS room_encryption_keys_insert_policy ON room_encryption_keys;

DROP POLICY IF EXISTS user_presence_select_policy ON user_presence;
DROP POLICY IF EXISTS user_presence_insert_policy ON user_presence;
DROP POLICY IF EXISTS user_presence_update_policy ON user_presence;

-- RLS policies for chat_rooms: users can only see rooms they're participants in
CREATE POLICY chat_rooms_select_policy ON chat_rooms
    FOR SELECT USING (
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY chat_rooms_insert_policy ON chat_rooms
    FOR INSERT WITH CHECK (
        created_by = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY chat_rooms_update_policy ON chat_rooms
    FOR UPDATE USING (
        -- Allow if user is a participant in the room
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR if user is admin
        OR current_setting('app.current_user_role', false) = 'admin'
    );

CREATE POLICY chat_rooms_delete_policy ON chat_rooms
    FOR DELETE USING (
        -- Allow room creator to delete
        created_by = current_setting('app.current_user_id', false)::varchar
        -- OR any participant can delete
        OR room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', false)::varchar
        )
        -- OR admin can delete
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- RLS policies for room_participants
CREATE POLICY room_participants_select_policy ON room_participants
    FOR SELECT USING (
        -- User can see their own participation records
        user_id = current_setting('app.current_user_id', false)::varchar
        OR current_setting('app.current_user_role', false) = 'admin'
    );

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

CREATE POLICY room_participants_delete_policy ON room_participants
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', false)::varchar
        OR room_id IN (
            SELECT room_id FROM chat_rooms 
            WHERE created_by = current_setting('app.current_user_id', false)::varchar
        )
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- RLS policies for chat_messages
CREATE POLICY chat_messages_select_policy ON chat_messages
    FOR SELECT USING (
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY chat_messages_insert_policy ON chat_messages
    FOR INSERT WITH CHECK (
        (sender_id = current_setting('app.current_user_id', true)::varchar
        AND room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY chat_messages_update_policy ON chat_messages
    FOR UPDATE USING (
        sender_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY chat_messages_delete_policy ON chat_messages
    FOR DELETE USING (
        sender_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- RLS policies for message_reactions
CREATE POLICY message_reactions_select_policy ON message_reactions
    FOR SELECT USING (
        message_id IN (
            SELECT message_id FROM chat_messages 
            WHERE room_id IN (
                SELECT room_id FROM room_participants 
                WHERE user_id = current_setting('app.current_user_id', true)::varchar
            )
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY message_reactions_insert_policy ON message_reactions
    FOR INSERT WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY message_reactions_delete_policy ON message_reactions
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- RLS policies for room_encryption_keys
CREATE POLICY room_encryption_keys_select_policy ON room_encryption_keys
    FOR SELECT USING (
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY room_encryption_keys_insert_policy ON room_encryption_keys
    FOR INSERT WITH CHECK (true);

-- RLS policies for user_presence (all users can see all presence)
CREATE POLICY user_presence_select_policy ON user_presence
    FOR SELECT USING (true);

CREATE POLICY user_presence_insert_policy ON user_presence
    FOR INSERT WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY user_presence_update_policy ON user_presence
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Add updated_at triggers for messaging tables (drop first for idempotency)
DROP TRIGGER IF EXISTS update_chat_rooms_updated_at ON chat_rooms;
CREATE TRIGGER update_chat_rooms_updated_at
    BEFORE UPDATE ON chat_rooms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_chat_messages_updated_at ON chat_messages;
CREATE TRIGGER update_chat_messages_updated_at
    BEFORE UPDATE ON chat_messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_presence_updated_at ON user_presence;
CREATE TRIGGER update_user_presence_updated_at
    BEFORE UPDATE ON user_presence
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE chat_rooms IS 'Chat rooms for user-to-user messaging';
COMMENT ON TABLE room_participants IS 'Participants in chat rooms with read tracking';
COMMENT ON TABLE chat_messages IS 'Messages sent in chat rooms with optional encryption';
COMMENT ON TABLE message_reactions IS 'Emoji reactions to messages';
COMMENT ON TABLE room_encryption_keys IS 'Encryption keys for rooms (future E2EE support)';
COMMENT ON TABLE user_presence IS 'User online/offline/away status tracking';

-- Report success
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'BULLY! Messaging system migration complete!';
    RAISE NOTICE 'Tables created: chat_rooms, room_participants, chat_messages, message_reactions, room_encryption_keys, user_presence';
    RAISE NOTICE 'Indexes created: 12 performance indexes';
    RAISE NOTICE 'RLS policies enabled for all tables';
    RAISE NOTICE '========================================';
END $$;

