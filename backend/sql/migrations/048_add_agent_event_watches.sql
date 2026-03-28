-- ========================================
-- Agent event watches: email, folder, conversation monitoring
-- ========================================
-- Run: docker exec -i bastion-postgres psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/048_add_agent_event_watches.sql
-- ========================================

-- Add watch_config to agent_profiles
ALTER TABLE agent_profiles ADD COLUMN IF NOT EXISTS watch_config JSONB DEFAULT '{}';

-- Email watches: which agents watch which email connections
CREATE TABLE IF NOT EXISTS agent_email_watches (
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    connection_id BIGINT NOT NULL REFERENCES external_connections(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT true,
    subject_pattern VARCHAR(500),
    sender_pattern VARCHAR(500),
    folder VARCHAR(255) DEFAULT 'Inbox',
    last_checked_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_profile_id, connection_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_email_watches_active
    ON agent_email_watches(user_id) WHERE is_active = true;

-- Folder watches: which agents watch which document folders
CREATE TABLE IF NOT EXISTS agent_folder_watches (
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    folder_id VARCHAR(255) NOT NULL REFERENCES document_folders(folder_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT true,
    file_type_filter VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (agent_profile_id, folder_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_folder_watches_folder
    ON agent_folder_watches(folder_id) WHERE is_active = true;

-- Conversation watches: AI conversations or chat rooms
CREATE TABLE IF NOT EXISTS agent_conversation_watches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    watch_type VARCHAR(20) NOT NULL,
    room_id UUID REFERENCES chat_rooms(room_id) ON DELETE CASCADE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(agent_profile_id, watch_type, room_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_convo_watches_user
    ON agent_conversation_watches(user_id, watch_type) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_agent_convo_watches_room
    ON agent_conversation_watches(room_id) WHERE is_active = true AND room_id IS NOT NULL;
-- One ai_conversations watch per profile (room_id is NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_convo_watches_ai_unique
    ON agent_conversation_watches(agent_profile_id) WHERE watch_type = 'ai_conversations' AND room_id IS NULL;
