-- ========================================
-- TEAMS SYSTEM MIGRATION
-- Team Collaboration Platform
-- ========================================
-- This migration can be run on existing databases without data loss
-- Idempotent design - safe to run multiple times
--
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/006_add_teams_system.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/006_add_teams_system.sql
-- ========================================

-- Create enums for teams system (idempotent)
DO $$ BEGIN
    CREATE TYPE team_role_enum AS ENUM ('admin', 'member', 'viewer');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'Type team_role_enum already exists, skipping';
END $$;

DO $$ BEGIN
    CREATE TYPE invitation_status_enum AS ENUM ('pending', 'accepted', 'rejected', 'expired');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'Type invitation_status_enum already exists, skipping';
END $$;

DO $$ BEGIN
    CREATE TYPE post_type_enum AS ENUM ('text', 'image', 'file');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'Type post_type_enum already exists, skipping';
END $$;

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    team_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_name VARCHAR(255) NOT NULL,
    description TEXT,
    created_by VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    avatar_url VARCHAR(500),
    settings JSONB DEFAULT '{}'::jsonb
);

-- Team members table
CREATE TABLE IF NOT EXISTS team_members (
    team_id UUID NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    role team_role_enum NOT NULL DEFAULT 'member',
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    invited_by VARCHAR(255) REFERENCES users(user_id) ON DELETE SET NULL,
    PRIMARY KEY (team_id, user_id)
);

-- Team invitations table
CREATE TABLE IF NOT EXISTS team_invitations (
    invitation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    invited_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    invited_by VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    status invitation_status_enum NOT NULL DEFAULT 'pending',
    message_id UUID REFERENCES chat_messages(message_id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
    responded_at TIMESTAMP WITH TIME ZONE
);

-- Team posts table
CREATE TABLE IF NOT EXISTS team_posts (
    post_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES teams(team_id) ON DELETE CASCADE,
    author_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    post_type post_type_enum NOT NULL DEFAULT 'text',
    attachments JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Post reactions table
CREATE TABLE IF NOT EXISTS post_reactions (
    post_id UUID NOT NULL REFERENCES team_posts(post_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    reaction_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (post_id, user_id, reaction_type)
);

-- Post comments table
CREATE TABLE IF NOT EXISTS post_comments (
    comment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID NOT NULL REFERENCES team_posts(post_id) ON DELETE CASCADE,
    author_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Update existing tables to support teams

-- Add team_id to document_metadata
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_metadata' AND column_name = 'team_id'
    ) THEN
        ALTER TABLE document_metadata ADD COLUMN team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL;
    END IF;
END $$;

-- Add team_id to document_folders
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_folders' AND column_name = 'team_id'
    ) THEN
        ALTER TABLE document_folders ADD COLUMN team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL;
    END IF;
END $$;

-- Add team_id to chat_rooms
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'chat_rooms' AND column_name = 'team_id'
    ) THEN
        ALTER TABLE chat_rooms ADD COLUMN team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL;
    END IF;
END $$;

-- Update message_type_enum to include 'team_invitation'
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_enum 
        WHERE enumlabel = 'team_invitation' 
        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'message_type_enum')
    ) THEN
        ALTER TYPE message_type_enum ADD VALUE 'team_invitation';
    END IF;
END $$;

-- Create indexes for teams performance
CREATE INDEX IF NOT EXISTS idx_teams_created_by ON teams(created_by);
CREATE INDEX IF NOT EXISTS idx_teams_created_at ON teams(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_team_members_team_id ON team_members(team_id);
CREATE INDEX IF NOT EXISTS idx_team_members_user_id ON team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_team_members_role ON team_members(role);
CREATE INDEX IF NOT EXISTS idx_team_invitations_team_id ON team_invitations(team_id);
CREATE INDEX IF NOT EXISTS idx_team_invitations_invited_user_id ON team_invitations(invited_user_id);
CREATE INDEX IF NOT EXISTS idx_team_invitations_status ON team_invitations(status);
CREATE INDEX IF NOT EXISTS idx_team_invitations_message_id ON team_invitations(message_id);
CREATE INDEX IF NOT EXISTS idx_team_invitations_expires_at ON team_invitations(expires_at);
CREATE INDEX IF NOT EXISTS idx_team_posts_team_id ON team_posts(team_id);
CREATE INDEX IF NOT EXISTS idx_team_posts_author_id ON team_posts(author_id);
CREATE INDEX IF NOT EXISTS idx_team_posts_created_at ON team_posts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_team_posts_team_created ON team_posts(team_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_team_posts_deleted_at ON team_posts(deleted_at) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_post_reactions_post_id ON post_reactions(post_id);
CREATE INDEX IF NOT EXISTS idx_post_reactions_user_id ON post_reactions(user_id);
CREATE INDEX IF NOT EXISTS idx_post_comments_post_id ON post_comments(post_id);
CREATE INDEX IF NOT EXISTS idx_post_comments_author_id ON post_comments(author_id);
CREATE INDEX IF NOT EXISTS idx_post_comments_created_at ON post_comments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_post_comments_deleted_at ON post_comments(deleted_at) WHERE deleted_at IS NULL;

-- Indexes for updated tables
CREATE INDEX IF NOT EXISTS idx_document_metadata_team_id ON document_metadata(team_id) WHERE team_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_document_folders_team_id ON document_folders(team_id) WHERE team_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chat_rooms_team_id ON chat_rooms(team_id) WHERE team_id IS NOT NULL;

-- Row-Level Security policies for teams
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_invitations ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE post_reactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE post_comments ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can see teams they are members of
CREATE POLICY teams_select_policy ON teams
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = teams.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)
        )
    );

-- RLS Policy: Users can see their own team memberships
CREATE POLICY team_members_select_policy ON team_members
    FOR SELECT
    USING (user_id = current_setting('app.current_user_id', true));

-- RLS Policy: Users can see invitations sent to them
CREATE POLICY team_invitations_select_policy ON team_invitations
    FOR SELECT
    USING (
        invited_user_id = current_setting('app.current_user_id', true)
        OR EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_invitations.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)
            AND team_members.role = 'admin'
        )
    );

-- RLS Policy: Users can see posts from teams they are members of
CREATE POLICY team_posts_select_policy ON team_posts
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = team_posts.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)
        )
    );

-- RLS Policy: Users can see reactions on posts they can see
CREATE POLICY post_reactions_select_policy ON post_reactions
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM team_posts tp
            JOIN team_members tm ON tm.team_id = tp.team_id
            WHERE tp.post_id = post_reactions.post_id
            AND tm.user_id = current_setting('app.current_user_id', true)
        )
    );

-- RLS Policy: Users can see comments on posts they can see
CREATE POLICY post_comments_select_policy ON post_comments
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM team_posts tp
            JOIN team_members tm ON tm.team_id = tp.team_id
            WHERE tp.post_id = post_comments.post_id
            AND tm.user_id = current_setting('app.current_user_id', true)
        )
    );

-- Grant permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON teams TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON team_members TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON team_invitations TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON team_posts TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON post_reactions TO bastion_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON post_comments TO bastion_user;

-- Add comments for documentation
COMMENT ON TABLE teams IS 'Teams for collaboration and document sharing';
COMMENT ON TABLE team_members IS 'Team membership with roles (admin, member, viewer)';
COMMENT ON TABLE team_invitations IS 'Team invitations linked to chat messages';
COMMENT ON TABLE team_posts IS 'Social posts in team feeds';
COMMENT ON TABLE post_reactions IS 'Emoji reactions on team posts';
COMMENT ON TABLE post_comments IS 'Comments on team posts';

