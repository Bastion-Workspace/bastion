-- Migration 130: Messaging improvements
-- Phase 2a: Bot users for agent profiles
-- Phase 4a: Reply-to / quoting
-- Phase 4c: Message editing

-- Bot user flag on users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_bot BOOLEAN NOT NULL DEFAULT FALSE;

-- Link agent profiles to their bot user identity
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS bot_user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_agent_profiles_bot_user_id ON agent_profiles(bot_user_id);

-- Reply-to support for threaded conversations
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS reply_to_message_id UUID REFERENCES chat_messages(message_id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_chat_messages_reply_to ON chat_messages(reply_to_message_id) WHERE reply_to_message_id IS NOT NULL;

-- Message editing support
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS is_edited BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS edited_at TIMESTAMPTZ;
