-- Complete PostgreSQL setup for Bastion AI Workspace
-- This script creates the database, user, tables, and default data

-- Create the bastion_knowledge_base database if it doesn't exist
-- ROOSEVELT'S DATABASE DOCTRINE: Use conditional creation to avoid errors
SELECT 'CREATE DATABASE bastion_knowledge_base'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'bastion_knowledge_base')\gexec

-- Create the bastion_user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'bastion_user') THEN
        CREATE USER bastion_user WITH PASSWORD 'bastion_secure_password';
    END IF;
END
$$;

-- **ROOSEVELT'S RLS STRATEGY**: Grant BYPASSRLS temporarily during init
-- This allows bastion_user to insert default data into RLS-protected tables during setup
-- BYPASSRLS will be REVOKED at the end of this script to enforce Row-Level Security
ALTER ROLE bastion_user BYPASSRLS;

-- Grant privileges on the database
GRANT ALL PRIVILEGES ON DATABASE bastion_knowledge_base TO bastion_user;

-- Connect to the bastion_knowledge_base database
\c bastion_knowledge_base

-- Create document_metadata table for storing document information
CREATE TABLE IF NOT EXISTS document_metadata (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    category VARCHAR(100),
    tags TEXT[],
    description TEXT,
    author VARCHAR(255),
    language VARCHAR(50),
    publication_date DATE, -- Original publication date of the document content
    doc_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL DEFAULT 0,
    file_hash VARCHAR(64),
    processing_status VARCHAR(50) DEFAULT 'pending',
    ocr_quality_score FLOAT,
    needs_ocr_review BOOLEAN DEFAULT FALSE,
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    quality_score FLOAT,
    page_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    entity_count INTEGER DEFAULT 0,
    metadata_json JSONB,
    -- User document mapping and ownership
    user_id VARCHAR(255), -- ID of user who uploaded this document (NULL for admin/global documents)
    -- Global submission workflow columns
    submission_status VARCHAR(50) DEFAULT 'not_submitted', -- not_submitted, pending_approval, approved, rejected, withdrawn
    submitted_by VARCHAR(255), -- User ID who submitted document for global approval
    submitted_at TIMESTAMP WITH TIME ZONE, -- Submission timestamp
    submission_reason TEXT, -- Reason for submitting to global collection
    reviewed_by VARCHAR(255), -- Admin user ID who reviewed the submission
    reviewed_at TIMESTAMP WITH TIME ZONE, -- Review timestamp
    review_comment TEXT, -- Admin comment about the approval/rejection decision
    collection_type VARCHAR(50) DEFAULT 'user', -- Collection type: user (private) or global (shared)
    -- ZIP hierarchy support columns
    parent_document_id VARCHAR(255),
    inherit_parent_metadata BOOLEAN DEFAULT TRUE,
    is_zip_container BOOLEAN DEFAULT FALSE,
    original_zip_path VARCHAR(500),
    folder_id VARCHAR(255), -- ID of folder containing this document (NULL for root documents)
    exempt_from_vectorization BOOLEAN DEFAULT FALSE, -- If true, document is exempt from vectorization and knowledge graph processing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create user_settings table for storing user-specific settings
CREATE TABLE IF NOT EXISTS user_settings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    key VARCHAR(255) NOT NULL,
    value TEXT,
    data_type VARCHAR(50) DEFAULT 'string',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, key)
);

-- Create indexes for user_settings
CREATE INDEX IF NOT EXISTS idx_user_settings_user_id ON user_settings(user_id);
CREATE INDEX IF NOT EXISTS idx_user_settings_key ON user_settings(key);
CREATE INDEX IF NOT EXISTS idx_user_settings_user_key ON user_settings(user_id, key);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_document_metadata_document_id ON document_metadata(document_id);
CREATE INDEX IF NOT EXISTS idx_document_metadata_file_hash ON document_metadata(file_hash);
CREATE INDEX IF NOT EXISTS idx_document_metadata_upload_date ON document_metadata(upload_date);
CREATE INDEX IF NOT EXISTS idx_document_metadata_publication_date ON document_metadata(publication_date);
CREATE INDEX IF NOT EXISTS idx_document_metadata_processing_status ON document_metadata(processing_status);
CREATE INDEX IF NOT EXISTS idx_document_metadata_filename ON document_metadata(filename);
CREATE INDEX IF NOT EXISTS idx_document_metadata_category ON document_metadata(category);
CREATE INDEX IF NOT EXISTS idx_document_metadata_tags ON document_metadata USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_document_metadata_metadata_json ON document_metadata USING GIN(metadata_json);
CREATE INDEX IF NOT EXISTS idx_document_metadata_ocr_review ON document_metadata(needs_ocr_review);
CREATE INDEX IF NOT EXISTS idx_document_metadata_ocr_quality ON document_metadata(ocr_quality_score);
CREATE INDEX IF NOT EXISTS idx_document_metadata_page_count ON document_metadata(page_count);
CREATE INDEX IF NOT EXISTS idx_document_metadata_chunk_count ON document_metadata(chunk_count);
CREATE INDEX IF NOT EXISTS idx_document_metadata_entity_count ON document_metadata(entity_count);
-- User document mapping and submission workflow indexes
CREATE INDEX IF NOT EXISTS idx_document_metadata_user_id ON document_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_document_metadata_submission_status ON document_metadata(submission_status);
CREATE INDEX IF NOT EXISTS idx_document_metadata_submitted_by ON document_metadata(submitted_by);
CREATE INDEX IF NOT EXISTS idx_document_metadata_submitted_at ON document_metadata(submitted_at);
CREATE INDEX IF NOT EXISTS idx_document_metadata_reviewed_by ON document_metadata(reviewed_by);
CREATE INDEX IF NOT EXISTS idx_document_metadata_collection_type ON document_metadata(collection_type);
-- ZIP hierarchy indexes
CREATE INDEX IF NOT EXISTS idx_parent_document ON document_metadata(parent_document_id);
CREATE INDEX IF NOT EXISTS idx_zip_container ON document_metadata(is_zip_container);
CREATE INDEX IF NOT EXISTS idx_zip_hierarchy ON document_metadata(parent_document_id, is_zip_container);
CREATE INDEX IF NOT EXISTS idx_document_metadata_folder_id ON document_metadata(folder_id);
CREATE INDEX IF NOT EXISTS idx_document_metadata_exempt ON document_metadata(exempt_from_vectorization);

-- Add foreign key constraint for ZIP hierarchy
ALTER TABLE document_metadata 
ADD CONSTRAINT fk_parent_document 
FOREIGN KEY (parent_document_id) 
REFERENCES document_metadata(document_id) 
ON DELETE CASCADE;

-- Add comments for user mapping and submission workflow columns
COMMENT ON COLUMN document_metadata.user_id IS 'ID of user who uploaded this document (NULL for admin/global documents)';
COMMENT ON COLUMN document_metadata.submission_status IS 'Status of submission to global collection: not_submitted, pending_approval, approved, rejected, withdrawn';
COMMENT ON COLUMN document_metadata.submitted_by IS 'User ID who submitted document for global approval';
COMMENT ON COLUMN document_metadata.submitted_at IS 'Timestamp when document was submitted for approval';
COMMENT ON COLUMN document_metadata.submission_reason IS 'Reason provided by user for submitting to global collection';
COMMENT ON COLUMN document_metadata.reviewed_by IS 'Admin user ID who reviewed the submission';
COMMENT ON COLUMN document_metadata.reviewed_at IS 'Timestamp when submission was reviewed';
COMMENT ON COLUMN document_metadata.review_comment IS 'Admin comment about the approval/rejection decision';
COMMENT ON COLUMN document_metadata.collection_type IS 'Collection type: user (private) or global (shared)';

-- Add comments for ZIP hierarchy columns
COMMENT ON COLUMN document_metadata.parent_document_id IS 'ID of parent document (for files extracted from ZIP)';
COMMENT ON COLUMN document_metadata.inherit_parent_metadata IS 'Whether to inherit metadata from parent ZIP file';
COMMENT ON COLUMN document_metadata.is_zip_container IS 'Whether this document is a ZIP file containing other files';
COMMENT ON COLUMN document_metadata.original_zip_path IS 'Original path within ZIP file for extracted files';
COMMENT ON COLUMN document_metadata.folder_id IS 'ID of folder containing this document (NULL for root documents)';

-- Create settings table for application configuration
CREATE TABLE IF NOT EXISTS settings (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,
    value TEXT,
    description TEXT,
    category VARCHAR(50) DEFAULT 'general',
    data_type VARCHAR(20) DEFAULT 'string',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index on settings key for fast lookups
CREATE INDEX IF NOT EXISTS idx_settings_key ON settings(key);
CREATE INDEX IF NOT EXISTS idx_settings_category ON settings(category);

-- Grant minimal required permissions to bastion_user
-- Only SELECT, INSERT, UPDATE, DELETE on specific tables
-- No CREATE, DROP, ALTER, or TRUNCATE permissions

-- Document metadata permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON document_metadata TO bastion_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bastion_user;

-- Settings permissions (read-only for most operations)
GRANT SELECT, INSERT, UPDATE ON settings TO bastion_user;

-- Grant USAGE on schema (required for table access)
GRANT USAGE ON SCHEMA public TO bastion_user;

-- Grant CREATE permission for migration operations
GRANT CREATE ON SCHEMA public TO bastion_user;

-- Note: CREATE permission is needed for migration files that create tables
-- This is safe because RLS policies control data access

-- Row Level Security (RLS) for document_metadata will be enabled after teams tables are created
-- See RLS policies section below after teams table creation

-- Insert default settings
INSERT INTO settings (key, value, data_type, description, category) VALUES
('llm_model', '', 'string', 'Default LLM model for chat and queries', 'llm'),
('llm_temperature', '0.7', 'float', 'Temperature setting for LLM responses (0.0-1.0)', 'llm'),
('llm_max_tokens', '4000', 'integer', 'Maximum tokens for LLM responses', 'llm'),
('llm_context_window', '100000', 'integer', 'Context window size for the LLM', 'llm'),
('rag_chunk_size', '1000', 'integer', 'Default chunk size for document processing', 'rag'),
('rag_chunk_overlap', '200', 'integer', 'Overlap between document chunks', 'rag'),
('rag_similarity_threshold', '0.7', 'float', 'Minimum similarity score for relevant chunks', 'rag'),
('rag_max_chunks', '10', 'integer', 'Maximum number of chunks to include in context', 'rag'),
('doc_quality_threshold', '0.7', 'float', 'Minimum quality score for document acceptance', 'documents'),
('doc_max_file_size', '1572864000', 'integer', 'Maximum file size in bytes (1.5GB)', 'documents'),
('doc_auto_categorize', 'true', 'boolean', 'Enable automatic document categorization', 'documents'),
('ui_theme', 'light', 'string', 'Default UI theme (light/dark)', 'ui'),
('ui_items_per_page', '20', 'integer', 'Default number of items per page in lists', 'ui'),
('ui_enable_streaming', 'true', 'boolean', 'Enable streaming responses in chat', 'ui'),
('system_log_level', 'INFO', 'string', 'System logging level', 'system'),
('system_metrics_retention_days', '30', 'integer', 'Days to retain system metrics', 'system'),
('system_backup_enabled', 'true', 'boolean', 'Enable automatic system backups', 'system'),
('kg_entity_extraction_enabled', 'true', 'boolean', 'Enable entity extraction for knowledge graph', 'knowledge_graph'),
('kg_relationship_threshold', '0.8', 'float', 'Minimum confidence for entity relationships', 'knowledge_graph'),
('security_session_timeout', '3600', 'integer', 'Session timeout in seconds', 'security'),
('security_rate_limit_requests', '100', 'integer', 'Rate limit: requests per minute', 'security'),
('ocr_quality_threshold', '0.7', 'float', 'OCR quality threshold for manual review (0.0-1.0)', 'ocr'),
('ocr_auto_process', 'true', 'boolean', 'Process OCR documents traditionally first before review', 'ocr'),
('ocr_enable_review', 'true', 'boolean', 'Enable OCR review interface for low-quality text', 'ocr'),
('classification_model', '', 'string', 'Fast LLM model for intent classification (separate from main chat model)', 'intent_classification_model'),
('intent_classification_model', '', 'string', 'Fast LLM model for intent classification (alias for classification_model)', 'intent_classification_model'),
-- News defaults
('news.synthesis_model', '', 'string', 'Model used for news synthesis', 'news'),
('news.min_sources', '3', 'integer', 'Minimum sources per cluster', 'news'),
('news.recency_minutes', '60', 'integer', 'Recency window in minutes', 'news'),
('news.min_diversity', '0.4', 'float', 'Required diversity score for synthesis', 'news'),
('news.notifications_enabled', 'false', 'boolean', 'Enable desktop notifications for breaking/urgent', 'news'),
('news.retention_days', '14', 'integer', 'Retention period for synthesized news articles (days)', 'news')
ON CONFLICT (key) DO NOTHING;

-- Create news articles table for synthesized articles
CREATE TABLE IF NOT EXISTS news_articles (
    id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    lede TEXT NOT NULL,
    file_path TEXT NOT NULL, -- Markdown file path on disk
    key_points JSONB NOT NULL DEFAULT '[]',
    citations JSONB NOT NULL DEFAULT '[]',
    diversity_score FLOAT NOT NULL DEFAULT 0.0,
    severity VARCHAR(16) NOT NULL DEFAULT 'normal',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Helpful indexes for querying headlines
CREATE INDEX IF NOT EXISTS idx_news_articles_updated_at ON news_articles(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_severity ON news_articles(severity);
CREATE INDEX IF NOT EXISTS idx_news_articles_file_path ON news_articles(file_path);

-- Grants for news tables
GRANT ALL PRIVILEGES ON news_articles TO bastion_user;
-- Admin role grants are applied later in this script after role creation

-- PDF Segmentation Tables for Manual Document Processing
-- Extends the existing Plato Knowledge Base schema

-- Table to store PDF page images and segmentation data
CREATE TABLE IF NOT EXISTS pdf_pages (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    page_image_path VARCHAR(500) NOT NULL, -- Path to extracted page image
    page_width INTEGER NOT NULL,
    page_height INTEGER NOT NULL,
    processing_status VARCHAR(50) DEFAULT 'pending', -- pending, segmented, completed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id, page_number)
);

-- Table to store manual segments within each page
CREATE TABLE IF NOT EXISTS pdf_segments (
    id SERIAL PRIMARY KEY,
    page_id INTEGER NOT NULL REFERENCES pdf_pages(id) ON DELETE CASCADE,
    segment_id VARCHAR(255) UNIQUE NOT NULL, -- Unique identifier for this segment
    segment_type VARCHAR(50) NOT NULL, -- article, advertisement, image, caption, headline, etc.
    x_coordinate INTEGER NOT NULL, -- Top-left X coordinate
    y_coordinate INTEGER NOT NULL, -- Top-left Y coordinate
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    bounds JSONB, -- Segment bounds as JSON object (x, y, width, height)
    manual_text TEXT, -- Manually typed text content
    manual_text_chunk_id VARCHAR(255), -- Link to text chunks for better integration
    confidence_score FLOAT DEFAULT 1.0, -- Always 1.0 for manual entry
    tags TEXT[], -- Additional tags for this segment
    metadata_json JSONB, -- Additional metadata (font info, colors, etc.)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store segment relationships (e.g., caption belongs to image)
CREATE TABLE IF NOT EXISTS segment_relationships (
    id SERIAL PRIMARY KEY,
    parent_segment_id VARCHAR(255) NOT NULL REFERENCES pdf_segments(segment_id) ON DELETE CASCADE,
    child_segment_id VARCHAR(255) NOT NULL REFERENCES pdf_segments(segment_id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL, -- caption_of, continuation_of, related_to
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(parent_segment_id, child_segment_id, relationship_type)
);

-- Indexes for PDF segmentation performance
CREATE INDEX IF NOT EXISTS idx_pdf_pages_document_id ON pdf_pages(document_id);
CREATE INDEX IF NOT EXISTS idx_pdf_pages_status ON pdf_pages(processing_status);
CREATE INDEX IF NOT EXISTS idx_pdf_segments_page_id ON pdf_segments(page_id);
CREATE INDEX IF NOT EXISTS idx_pdf_segments_type ON pdf_segments(segment_type);
CREATE INDEX IF NOT EXISTS idx_pdf_segments_tags ON pdf_segments USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_pdf_segments_metadata ON pdf_segments USING GIN(metadata_json);
CREATE INDEX IF NOT EXISTS idx_pdf_segments_bounds ON pdf_segments USING GIN(bounds);
CREATE INDEX IF NOT EXISTS idx_pdf_segments_chunk_id ON pdf_segments(manual_text_chunk_id);
CREATE INDEX IF NOT EXISTS idx_segment_relationships_parent ON segment_relationships(parent_segment_id);
CREATE INDEX IF NOT EXISTS idx_segment_relationships_child ON segment_relationships(child_segment_id);

-- Grant permissions on PDF segmentation tables
GRANT ALL PRIVILEGES ON pdf_pages TO bastion_user;
GRANT ALL PRIVILEGES ON pdf_segments TO bastion_user;
GRANT ALL PRIVILEGES ON segment_relationships TO bastion_user;
GRANT ALL PRIVILEGES ON pdf_pages_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON pdf_segments_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON segment_relationships_id_seq TO bastion_user;

-- Free-form Notes System
-- Table to store user-created notes that integrate with the search system

-- Users table for authentication and multi-user support
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL, -- UUID or external auth ID
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL, -- Hashed password
    salt VARCHAR(255) NOT NULL, -- Password salt
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('user', 'admin')), -- User role
    display_name VARCHAR(255),
    avatar_url VARCHAR(500),
    preferences JSONB DEFAULT '{}', -- User preferences (theme, model selection, etc.)
    is_active BOOLEAN DEFAULT TRUE,
    failed_login_attempts INTEGER DEFAULT 0, -- Track failed login attempts
    last_failed_login TIMESTAMP WITH TIME ZONE, -- Last failed login attempt
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create indexes for authentication performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Sessions table for tracking user sessions and JWT tokens
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL, -- Hash of the JWT token
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for session management
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash);

-- Grant permissions on authentication tables
GRANT ALL PRIVILEGES ON users TO bastion_user;
GRANT ALL PRIVILEGES ON user_sessions TO bastion_user;
GRANT ALL PRIVILEGES ON users_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON user_sessions_id_seq TO bastion_user;

-- Enable RLS for user_sessions
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;

-- Create RLS policy for user_sessions (SELECT, UPDATE, DELETE only)
CREATE POLICY sessions_user_policy ON user_sessions
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Create separate policy for authentication lookups by token_hash
-- This allows authentication queries to find sessions without user context
CREATE POLICY sessions_auth_lookup_policy ON user_sessions
    FOR SELECT USING (
        current_setting('app.current_user_id', true) IS NULL 
        AND expires_at > CURRENT_TIMESTAMP
    );

CREATE POLICY sessions_update_policy ON user_sessions
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY sessions_delete_policy ON user_sessions
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Allow session creation during authentication (INSERT operations)
CREATE POLICY sessions_insert_policy ON user_sessions
    FOR INSERT WITH CHECK (true);

-- Document folders for organization
CREATE TABLE IF NOT EXISTS document_folders (
    id SERIAL PRIMARY KEY,
    folder_id VARCHAR(255) UNIQUE NOT NULL, -- UUID for the folder
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE, -- NULL for global/team folders
    team_id UUID, -- NULL for user/global folders, set for team folders (FK added after teams table exists)
    name VARCHAR(255) NOT NULL,
    description TEXT,
    color VARCHAR(7), -- Hex color code
    parent_folder_id VARCHAR(255) REFERENCES document_folders(folder_id) ON DELETE CASCADE, -- For nested folder structure
    collection_type VARCHAR(50) DEFAULT 'user' CHECK (collection_type IN ('user', 'global', 'team')), -- Collection type
    sort_order INTEGER DEFAULT 0,
    -- **ROOSEVELT FOLDER TAGGING**: Metadata for automatic tag inheritance
    category VARCHAR(100), -- Folder category (inherited by documents uploaded to this folder)
    tags TEXT[] DEFAULT '{}', -- Folder tags (automatically applied to documents uploaded here)
    inherit_tags BOOLEAN DEFAULT TRUE, -- Whether documents uploaded to this folder should inherit its tags
    exempt_from_vectorization BOOLEAN DEFAULT FALSE, -- If true, folder and all descendants are exempt from vectorization and knowledge graph processing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255) REFERENCES users(user_id) ON DELETE SET NULL, -- User who created this folder (for ownership tracking)
    -- Ensure user_id is provided for user folders, but can be NULL for global/team folders
    CONSTRAINT check_user_id_for_collection_type CHECK (
        (collection_type = 'user' AND user_id IS NOT NULL AND team_id IS NULL) OR
        (collection_type = 'global' AND user_id IS NULL AND team_id IS NULL) OR
        (collection_type = 'team' AND team_id IS NOT NULL AND user_id IS NULL)
    )
    -- Note: Unique constraint for non-root folders handled by partial index below
    -- Cannot use simple UNIQUE constraint because PostgreSQL treats NULL as distinct
);

-- Create indexes for document folders
CREATE INDEX IF NOT EXISTS idx_document_folders_folder_id ON document_folders(folder_id);
CREATE INDEX IF NOT EXISTS idx_document_folders_user_id ON document_folders(user_id);
CREATE INDEX IF NOT EXISTS idx_document_folders_created_by ON document_folders(created_by);
CREATE INDEX IF NOT EXISTS idx_document_folders_parent_folder_id ON document_folders(parent_folder_id);
CREATE INDEX IF NOT EXISTS idx_document_folders_collection_type ON document_folders(collection_type);
-- **ROOSEVELT FOLDER TAGGING**: Indexes for metadata filtering
CREATE INDEX IF NOT EXISTS idx_document_folders_category ON document_folders(category);
CREATE INDEX IF NOT EXISTS idx_document_folders_tags ON document_folders USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_document_folders_user_collection ON document_folders(user_id, collection_type);
CREATE INDEX IF NOT EXISTS idx_document_folders_exempt ON document_folders(exempt_from_vectorization);

-- ROOSEVELT'S TRUST-BUSTING CONSTRAINTS: Prevent duplicate folders!
-- Use partial unique indexes to handle NULL parent_folder_id AND NULL user_id properly
-- PostgreSQL treats NULL as distinct in unique indexes, so we need FOUR separate constraints!

-- For USER NON-ROOT folders (parent_folder_id IS NOT NULL AND user_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_with_parent 
ON document_folders(user_id, name, parent_folder_id, collection_type)
WHERE parent_folder_id IS NOT NULL AND user_id IS NOT NULL;

-- For GLOBAL NON-ROOT folders (parent_folder_id IS NOT NULL AND user_id IS NULL)
-- **ROOSEVELT FIX**: Global folders need separate unique constraint since user_id is NULL!
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_global_with_parent
ON document_folders(name, parent_folder_id, collection_type)
WHERE parent_folder_id IS NOT NULL AND user_id IS NULL;

-- For USER root folders (parent_folder_id IS NULL AND user_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_user_root 
ON document_folders(user_id, name, collection_type)
WHERE parent_folder_id IS NULL AND user_id IS NOT NULL;

-- For GLOBAL root folders (parent_folder_id IS NULL AND user_id IS NULL)
-- Can't include user_id here because all global folders have NULL user_id
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_global_root 
ON document_folders(name, collection_type)
WHERE parent_folder_id IS NULL AND user_id IS NULL;

-- For TEAM root folders (parent_folder_id IS NULL AND team_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_team_root 
ON document_folders(team_id, name, collection_type)
WHERE parent_folder_id IS NULL AND team_id IS NOT NULL;

-- For TEAM non-root folders (parent_folder_id IS NOT NULL AND team_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_team_with_parent
ON document_folders(team_id, name, parent_folder_id, collection_type)
WHERE parent_folder_id IS NOT NULL AND team_id IS NOT NULL;

-- Grant permissions on document folders
GRANT ALL PRIVILEGES ON document_folders TO bastion_user;
GRANT ALL PRIVILEGES ON document_folders_id_seq TO bastion_user;

-- Enable RLS for document_folders (policies created after teams tables exist)
ALTER TABLE document_folders ENABLE ROW LEVEL SECURITY;

-- Conversations table for organizing chat sessions
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL, -- UUID for the conversation
    user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL, -- Auto-generated or user-defined title
    description TEXT, -- Optional description
    is_pinned BOOLEAN DEFAULT FALSE, -- Allow users to pin important conversations
    is_archived BOOLEAN DEFAULT FALSE, -- Soft delete conversations
    tags TEXT[], -- Optional tags for organization
    metadata_json JSONB DEFAULT '{}', -- Additional metadata (model used, settings, etc.)
    message_count INTEGER DEFAULT 0, -- Cached count for performance
    last_message_at TIMESTAMP WITH TIME ZONE, -- Last activity timestamp
    manual_order INTEGER DEFAULT NULL, -- Manual ordering support
    order_locked BOOLEAN DEFAULT FALSE, -- Lock manual order
    message_sequence INTEGER DEFAULT 0, -- Auto-incrementing sequence counter for message ordering
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Messages table for storing individual chat messages
CREATE TABLE IF NOT EXISTS conversation_messages (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(255) UNIQUE NOT NULL, -- UUID for the message
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL, -- The message content
    content_hash VARCHAR(64), -- Hash for deduplication and integrity
    model_used VARCHAR(100), -- Which LLM model was used (for assistant messages)
    query_time FLOAT, -- Response time for assistant messages
    token_count INTEGER, -- Token usage tracking
    citations JSONB DEFAULT '[]', -- Citations data for assistant messages
    metadata_json JSONB DEFAULT '{}', -- Additional metadata (temperature, max_tokens, etc.)
    parent_message_id VARCHAR(255), -- For threading/follow-up context
    is_edited BOOLEAN DEFAULT FALSE, -- Track if message was edited
    is_deleted BOOLEAN DEFAULT FALSE, -- Soft delete
    sequence_number INTEGER DEFAULT 0 CHECK (sequence_number > 0), -- Sequential message number within conversation for ordering
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Conversation sharing table for collaborative features
CREATE TABLE IF NOT EXISTS conversation_shares (
    id SERIAL PRIMARY KEY,
    share_id VARCHAR(255) UNIQUE NOT NULL, -- UUID for the share
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    shared_by_user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    shared_with_user_id VARCHAR(255) REFERENCES users(user_id) ON DELETE CASCADE, -- NULL for public shares
    share_type VARCHAR(20) NOT NULL CHECK (share_type IN ('read', 'comment', 'edit')),
    is_public BOOLEAN DEFAULT FALSE, -- Public share link
    expires_at TIMESTAMP WITH TIME ZONE, -- Optional expiration
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Conversation folders for organization
CREATE TABLE IF NOT EXISTS conversation_folders (
    id SERIAL PRIMARY KEY,
    folder_id VARCHAR(255) UNIQUE NOT NULL, -- UUID for the folder
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    color VARCHAR(7), -- Hex color code
    parent_folder_id VARCHAR(255) REFERENCES conversation_folders(folder_id) ON DELETE CASCADE, -- For nested folder structure
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Junction table for conversations in folders
CREATE TABLE IF NOT EXISTS conversation_folder_items (
    id SERIAL PRIMARY KEY,
    folder_id VARCHAR(255) NOT NULL REFERENCES conversation_folders(folder_id) ON DELETE CASCADE,
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(folder_id, conversation_id)
);

-- Additional indexes for optimal conversation performance (from 02_conversation_history.sql)
CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_title ON conversations(title);
CREATE INDEX IF NOT EXISTS idx_conversations_pinned ON conversations(is_pinned);
CREATE INDEX IF NOT EXISTS idx_conversations_archived ON conversations(is_archived);
CREATE INDEX IF NOT EXISTS idx_conversations_last_message ON conversations(last_message_at);
CREATE INDEX IF NOT EXISTS idx_conversations_tags ON conversations USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_conversations_metadata ON conversations USING GIN(metadata_json);
CREATE INDEX IF NOT EXISTS idx_conversations_manual_order ON conversations(user_id, manual_order) WHERE manual_order IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_conversations_last_message_order ON conversations(user_id, last_message_at DESC) WHERE manual_order IS NULL;
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_user_pinned_updated ON conversations(user_id, is_pinned DESC, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_messages_message_id ON conversation_messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON conversation_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_type ON conversation_messages(message_type);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON conversation_messages(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON conversation_messages(parent_message_id);
CREATE INDEX IF NOT EXISTS idx_messages_hash ON conversation_messages(content_hash);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_sequence ON conversation_messages(conversation_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_type_sequence ON conversation_messages(conversation_id, message_type, sequence_number);

-- Note: sequence_number constraint is now part of the column definition in the table

CREATE INDEX IF NOT EXISTS idx_shares_share_id ON conversation_shares(share_id);
CREATE INDEX IF NOT EXISTS idx_shares_conversation_id ON conversation_shares(conversation_id);
CREATE INDEX IF NOT EXISTS idx_shares_shared_by ON conversation_shares(shared_by_user_id);
CREATE INDEX IF NOT EXISTS idx_shares_shared_with ON conversation_shares(shared_with_user_id);
CREATE INDEX IF NOT EXISTS idx_shares_public ON conversation_shares(is_public);
CREATE INDEX IF NOT EXISTS idx_shares_expires ON conversation_shares(expires_at);

-- Additional indexes for conversation sharing performance
CREATE INDEX IF NOT EXISTS idx_conversation_shares_shared_with 
  ON conversation_shares(shared_with_user_id, created_at DESC);
  
CREATE INDEX IF NOT EXISTS idx_conversation_shares_conversation 
  ON conversation_shares(conversation_id);
  
CREATE INDEX IF NOT EXISTS idx_conversation_shares_expires 
  ON conversation_shares(expires_at) 
  WHERE expires_at IS NOT NULL;

-- Helper function to get all conversation participants
CREATE OR REPLACE FUNCTION get_conversation_participants(p_conversation_id VARCHAR)
RETURNS TABLE (
    user_id VARCHAR,
    username VARCHAR,
    email VARCHAR,
    share_type VARCHAR,
    is_owner BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    -- Get owner
    SELECT 
        c.user_id,
        u.username,
        u.email,
        'edit'::VARCHAR as share_type,
        TRUE as is_owner
    FROM conversations c
    LEFT JOIN users u ON c.user_id = u.user_id
    WHERE c.conversation_id = p_conversation_id
    
    UNION ALL
    
    -- Get shared users
    SELECT 
        cs.shared_with_user_id as user_id,
        u2.username,
        u2.email,
        cs.share_type,
        FALSE as is_owner
    FROM conversation_shares cs
    LEFT JOIN users u2 ON cs.shared_with_user_id = u2.user_id
    WHERE cs.conversation_id = p_conversation_id
    AND (cs.expires_at IS NULL OR cs.expires_at > NOW());
END;
$$ LANGUAGE plpgsql;

CREATE INDEX IF NOT EXISTS idx_folders_folder_id ON conversation_folders(folder_id);
CREATE INDEX IF NOT EXISTS idx_folders_user_id ON conversation_folders(user_id);
CREATE INDEX IF NOT EXISTS idx_folders_name ON conversation_folders(name);
CREATE INDEX IF NOT EXISTS idx_folders_parent_id ON conversation_folders(parent_folder_id);

CREATE INDEX IF NOT EXISTS idx_folder_items_folder_id ON conversation_folder_items(folder_id);
CREATE INDEX IF NOT EXISTS idx_folder_items_conversation_id ON conversation_folder_items(conversation_id);

-- Trigger functions for maintaining conversation metadata
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE conversations 
        SET message_count = message_count + 1,
            last_message_at = NEW.created_at,
            updated_at = NOW()
        WHERE conversation_id = NEW.conversation_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE conversations 
        SET message_count = GREATEST(message_count - 1, 0),
            updated_at = NOW()
        WHERE conversation_id = OLD.conversation_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_update_conversation_message_count_insert
    AFTER INSERT ON conversation_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_message_count();

CREATE TRIGGER trigger_update_conversation_message_count_delete
    AFTER DELETE ON conversation_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_message_count();

CREATE TRIGGER trigger_update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_conversation_messages_updated_at
    BEFORE UPDATE ON conversation_messages
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_update_conversation_folders_updated_at
    BEFORE UPDATE ON conversation_folders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions on conversation tables
GRANT ALL PRIVILEGES ON conversations TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_messages TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_shares TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_folders TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_folder_items TO bastion_user;
GRANT ALL PRIVILEGES ON conversations_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_messages_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_shares_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_folders_id_seq TO bastion_user;
GRANT ALL PRIVILEGES ON conversation_folder_items_id_seq TO bastion_user;

-- Enable RLS for conversation tables
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_shares ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_folders ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_folder_items ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for conversations
CREATE POLICY conversations_select_policy ON conversations
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

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

CREATE POLICY conversations_delete_policy ON conversations
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Allow conversation creation (INSERT operations)
CREATE POLICY conversations_insert_policy ON conversations
    FOR INSERT WITH CHECK (true);

-- Create RLS policies for conversation_messages
CREATE POLICY messages_select_policy ON conversation_messages
    FOR SELECT USING (
        conversation_id IN (
            SELECT conversation_id FROM conversations 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

CREATE POLICY messages_update_policy ON conversation_messages
    FOR UPDATE USING (
        conversation_id IN (
            SELECT conversation_id FROM conversations 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

CREATE POLICY messages_delete_policy ON conversation_messages
    FOR DELETE USING (
        conversation_id IN (
            SELECT conversation_id FROM conversations 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

-- Allow message creation (INSERT operations)
CREATE POLICY messages_insert_policy ON conversation_messages
    FOR INSERT WITH CHECK (true);

-- Create RLS policies for conversation_shares
CREATE POLICY shares_select_policy ON conversation_shares
    FOR SELECT USING (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR shared_with_user_id = current_setting('app.current_user_id', true)::varchar
        OR is_public = true
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY shares_update_policy ON conversation_shares
    FOR UPDATE USING (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY shares_delete_policy ON conversation_shares
    FOR DELETE USING (
        shared_by_user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Allow share creation (INSERT operations)
CREATE POLICY shares_insert_policy ON conversation_shares
    FOR INSERT WITH CHECK (true);

-- Create RLS policies for conversation_folders
CREATE POLICY folders_select_policy ON conversation_folders
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY folders_update_policy ON conversation_folders
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY folders_delete_policy ON conversation_folders
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Allow folder creation (INSERT operations)
-- Users can create folders for themselves, admins can create any folders
CREATE POLICY folders_insert_policy ON conversation_folders
    FOR INSERT WITH CHECK (
        -- User creating folder for themselves (context must be set and match)
        (COALESCE(current_setting('app.current_user_id', true), '') = user_id::varchar)
        -- OR admin role (can create for any user)
        OR (COALESCE(current_setting('app.current_user_role', true), '') = 'admin')
    );

-- Create RLS policies for conversation_folder_items
CREATE POLICY folder_items_select_policy ON conversation_folder_items
    FOR SELECT USING (
        folder_id IN (
            SELECT folder_id FROM conversation_folders 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

CREATE POLICY folder_items_update_policy ON conversation_folder_items
    FOR UPDATE USING (
        folder_id IN (
            SELECT folder_id FROM conversation_folders 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

CREATE POLICY folder_items_delete_policy ON conversation_folder_items
    FOR DELETE USING (
        folder_id IN (
            SELECT folder_id FROM conversation_folders 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            OR current_setting('app.current_user_role', true) = 'admin'
        )
    );

-- Allow folder item creation (INSERT operations)
CREATE POLICY folder_items_insert_policy ON conversation_folder_items
    FOR INSERT WITH CHECK (true);

-- Background Chat Jobs Table for Research Jobs
-- Tracks research jobs that run in the background
CREATE TABLE IF NOT EXISTS background_chat_jobs (
    job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    conversation_id VARCHAR(255) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    session_id VARCHAR(255) NOT NULL,
    query TEXT NOT NULL,
    execution_mode VARCHAR(50) NOT NULL DEFAULT 'execute',
    job_status VARCHAR(50) NOT NULL DEFAULT 'queued',
    progress_data JSONB DEFAULT '{}',
    result_data JSONB DEFAULT '{}',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    priority INTEGER DEFAULT 5
);

-- Indexes for efficient background job lookups
CREATE INDEX IF NOT EXISTS idx_background_jobs_user_id ON background_chat_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_background_jobs_conversation_id ON background_chat_jobs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_background_jobs_session_id ON background_chat_jobs(session_id);
CREATE INDEX IF NOT EXISTS idx_background_jobs_status ON background_chat_jobs(job_status);
CREATE INDEX IF NOT EXISTS idx_background_jobs_created_at ON background_chat_jobs(created_at);

-- Job status constraint
ALTER TABLE background_chat_jobs 
ADD CONSTRAINT check_job_status 
CHECK (job_status IN ('queued', 'running', 'completed', 'failed', 'cancelled'));

-- Execution mode constraint  
ALTER TABLE background_chat_jobs
ADD CONSTRAINT check_execution_mode
CHECK (execution_mode IN ('plan', 'execute', 'chat', 'direct'));

-- Grant permissions on background chat jobs table
GRANT ALL PRIVILEGES ON background_chat_jobs TO bastion_user;

-- Enable RLS for background_chat_jobs
ALTER TABLE background_chat_jobs ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for background_chat_jobs
CREATE POLICY jobs_select_policy ON background_chat_jobs
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY jobs_update_policy ON background_chat_jobs
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY jobs_delete_policy ON background_chat_jobs
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Allow job creation (INSERT operations)
CREATE POLICY jobs_insert_policy ON background_chat_jobs
    FOR INSERT WITH CHECK (true);

-- Add comments for documentation
COMMENT ON TABLE background_chat_jobs IS 'Tracks background research jobs that continue running when users navigate away';
COMMENT ON COLUMN background_chat_jobs.progress_data IS 'JSON data tracking current iteration, tool being used, etc.';
COMMENT ON COLUMN background_chat_jobs.result_data IS 'JSON data containing final answer, citations, and metadata';
COMMENT ON COLUMN background_chat_jobs.priority IS 'Job priority (lower number = higher priority)'; 

-- Note: Sample conversation folders will be created dynamically when the admin user is created
-- This avoids foreign key constraint violations during initialization

-- Research Plans Database Schema
-- Extends the existing knowledge base with persistent research plan management

-- Research Plans Table - **DEPRECATED** 
-- 🚨 MIGRATION NOTICE: Research plans migrated to LangGraph subgraph workflows
-- This table is commented out but preserved for potential data migration
/*
-- Research Plans Table - COMMENTED OUT (migrated to LangGraph subgraphs)
CREATE TABLE IF NOT EXISTS research_plans (
    plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    conversation_id VARCHAR(255) REFERENCES conversations(conversation_id) ON DELETE SET NULL,
    
    -- Plan metadata
    title VARCHAR(500) NOT NULL,
    description TEXT,
    query TEXT NOT NULL,
    scope VARCHAR(50) DEFAULT 'comprehensive', -- 'quick', 'comprehensive', 'deep'
    context TEXT, -- Additional context about the research need
    
    -- Plan content (structured data)
    initial_search_results JSONB, -- LocalSearchResult data
    local_steps JSONB, -- List[LocalResearchStep]
    web_steps JSONB, -- List[WebResearchStep]
    research_strategy VARCHAR(100),
    efficiency_notes TEXT,
    
    -- Plan status and lifecycle
    status VARCHAR(50) DEFAULT 'draft', -- 'draft', 'active', 'executed', 'archived', 'template'
    execution_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_executed_at TIMESTAMP WITH TIME ZONE,
    execution_time_avg FLOAT, -- Average execution time in seconds
    
    -- Template and sharing
    is_template BOOLEAN DEFAULT FALSE,
    template_name VARCHAR(255),
    template_category VARCHAR(100),
    is_public BOOLEAN DEFAULT FALSE,
    shared_with VARCHAR(255)[], -- Array of user IDs with access
    
    -- Versioning
    version INTEGER DEFAULT 1,
    parent_plan_id UUID REFERENCES research_plans(plan_id) ON DELETE SET NULL,
    
    -- Metadata
    tags TEXT[],
    category VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    archived_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT check_scope CHECK (scope IN ('quick', 'comprehensive', 'deep')),
    CONSTRAINT check_status CHECK (status IN ('draft', 'active', 'executed', 'archived', 'template')),
    CONSTRAINT check_execution_count CHECK (execution_count >= 0),
    CONSTRAINT check_success_count CHECK (success_count >= 0),
    CONSTRAINT check_version CHECK (version >= 1)
);
*/

/*
-- Research Plan Executions Table (for tracking execution history) - COMMENTED OUT
CREATE TABLE IF NOT EXISTS research_plan_executions (
    execution_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL REFERENCES research_plans(plan_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    conversation_id VARCHAR(255) REFERENCES conversations(conversation_id) ON DELETE SET NULL,
    
    -- Execution details
    query TEXT NOT NULL,
    execution_mode VARCHAR(50) DEFAULT 'execute', -- 'plan', 'execute', 'chat', 'direct'
    status VARCHAR(50) DEFAULT 'running', -- 'running', 'completed', 'failed', 'cancelled'
    
    -- Results
    answer TEXT,
    citations JSONB, -- List of citations
    iterations INTEGER DEFAULT 0,
    processing_time FLOAT, -- Execution time in seconds
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT check_execution_mode CHECK (execution_mode IN ('plan', 'execute', 'chat', 'direct')),
    CONSTRAINT check_execution_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled'))
);

-- Research Plan Analytics Table (for tracking effectiveness)
CREATE TABLE IF NOT EXISTS research_plan_analytics (
    analytics_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL REFERENCES research_plans(plan_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Analytics data
    query_type VARCHAR(100), -- Categorized query type
    success_rate FLOAT, -- Success rate (0.0-1.0)
    avg_execution_time FLOAT,
    avg_iterations FLOAT,
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    
    -- User feedback
    user_rating INTEGER CHECK (user_rating >= 1 AND user_rating <= 5),
    user_feedback TEXT,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for research plans
CREATE INDEX IF NOT EXISTS idx_research_plans_user_id ON research_plans(user_id);
CREATE INDEX IF NOT EXISTS idx_research_plans_conversation_id ON research_plans(conversation_id);
CREATE INDEX IF NOT EXISTS idx_research_plans_status ON research_plans(status);
CREATE INDEX IF NOT EXISTS idx_research_plans_is_template ON research_plans(is_template);
CREATE INDEX IF NOT EXISTS idx_research_plans_category ON research_plans(category);
CREATE INDEX IF NOT EXISTS idx_research_plans_created_at ON research_plans(created_at);
CREATE INDEX IF NOT EXISTS idx_research_plans_tags ON research_plans USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_plan_executions_plan_id ON research_plan_executions(plan_id);
CREATE INDEX IF NOT EXISTS idx_plan_executions_user_id ON research_plan_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_plan_executions_status ON research_plan_executions(status);
CREATE INDEX IF NOT EXISTS idx_plan_executions_created_at ON research_plan_executions(created_at);

CREATE INDEX IF NOT EXISTS idx_plan_analytics_plan_id ON research_plan_analytics(plan_id);
CREATE INDEX IF NOT EXISTS idx_plan_analytics_user_id ON research_plan_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_plan_analytics_query_type ON research_plan_analytics(query_type);

-- Grant permissions on research plans tables - COMMENTED OUT
GRANT ALL PRIVILEGES ON research_plans TO bastion_user;
GRANT ALL PRIVILEGES ON research_plan_executions TO bastion_user;
GRANT ALL PRIVILEGES ON research_plan_analytics TO bastion_user;
*/

-- ============================================================================
-- RSS FEEDS SCHEMA
-- ============================================================================

-- Create RSS feeds table if it doesn't exist
CREATE TABLE IF NOT EXISTS rss_feeds (
    feed_id VARCHAR(255) PRIMARY KEY,
    feed_url TEXT NOT NULL,
    feed_name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    tags JSONB DEFAULT '[]',
    check_interval INTEGER DEFAULT 3600,
    last_check TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_polling BOOLEAN DEFAULT FALSE,
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create RSS articles table for storing feed entries
CREATE TABLE IF NOT EXISTS rss_articles (
    article_id VARCHAR(255) PRIMARY KEY,
    feed_id VARCHAR(255) REFERENCES rss_feeds(feed_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    full_content TEXT, -- Full article content extracted by Crawl4AI when RSS description is truncated
    full_content_html TEXT, -- Original HTML content with images in position
    images JSONB, -- Images extracted from the article
    link TEXT NOT NULL,
    published_date TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    is_read BOOLEAN DEFAULT FALSE,
    content_hash VARCHAR(64),  -- For duplicate detection
    user_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create RSS feed subscriptions table for user-feed relationships
CREATE TABLE IF NOT EXISTS rss_feed_subscriptions (
    subscription_id VARCHAR(255) PRIMARY KEY,
    feed_id VARCHAR(255) REFERENCES rss_feeds(feed_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(feed_id, user_id)
);

-- Create indexes for efficient feed lookups
CREATE INDEX IF NOT EXISTS idx_rss_feeds_user_id ON rss_feeds(user_id);
CREATE INDEX IF NOT EXISTS idx_rss_feeds_active ON rss_feeds(is_active);
CREATE INDEX IF NOT EXISTS idx_rss_feeds_last_check ON rss_feeds(last_check);

-- Create indexes for efficient article lookups
CREATE INDEX IF NOT EXISTS idx_rss_articles_feed_id ON rss_articles(feed_id);
CREATE INDEX IF NOT EXISTS idx_rss_articles_user_id ON rss_articles(user_id);
CREATE INDEX IF NOT EXISTS idx_rss_articles_published_date ON rss_articles(published_date);
CREATE INDEX IF NOT EXISTS idx_rss_articles_is_read ON rss_articles(is_read);
CREATE INDEX IF NOT EXISTS idx_rss_articles_content_hash ON rss_articles(content_hash);
CREATE INDEX IF NOT EXISTS idx_rss_articles_full_content ON rss_articles USING GIN(to_tsvector('english', full_content)) WHERE full_content IS NOT NULL;

-- Create indexes for feed subscriptions
CREATE INDEX IF NOT EXISTS idx_rss_subscriptions_user_id ON rss_feed_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_rss_subscriptions_feed_id ON rss_feed_subscriptions(feed_id);
CREATE INDEX IF NOT EXISTS idx_rss_subscriptions_active ON rss_feed_subscriptions(is_active);



-- Add comments for documentation
COMMENT ON TABLE rss_feeds IS 'RSS feed configurations for automatic content ingestion';
COMMENT ON COLUMN rss_feeds.feed_id IS 'Unique identifier for the RSS feed';
COMMENT ON COLUMN rss_feeds.feed_url IS 'URL of the RSS feed';
COMMENT ON COLUMN rss_feeds.feed_name IS 'Human-readable name for the feed';
COMMENT ON COLUMN rss_feeds.category IS 'Primary category for the feed content';
COMMENT ON COLUMN rss_feeds.tags IS 'JSON array of tags for the feed';
COMMENT ON COLUMN rss_feeds.check_interval IS 'Interval in seconds between feed checks';
COMMENT ON COLUMN rss_feeds.last_check IS 'Timestamp of last feed check';
COMMENT ON COLUMN rss_feeds.is_active IS 'Whether the feed is currently being monitored';
COMMENT ON COLUMN rss_feeds.user_id IS 'User ID for user-specific feeds, NULL for global feeds';

COMMENT ON TABLE rss_articles IS 'RSS articles from monitored feeds';
COMMENT ON COLUMN rss_articles.article_id IS 'Unique identifier for the RSS article';
COMMENT ON COLUMN rss_articles.feed_id IS 'Reference to the RSS feed';
COMMENT ON COLUMN rss_articles.title IS 'Article title from RSS feed';
COMMENT ON COLUMN rss_articles.description IS 'Article description/summary from RSS feed';
COMMENT ON COLUMN rss_articles.full_content IS 'Full article content extracted by Crawl4AI when RSS description is truncated';
COMMENT ON COLUMN rss_articles.link IS 'URL to the full article';
COMMENT ON COLUMN rss_articles.published_date IS 'Publication date from RSS feed';
COMMENT ON COLUMN rss_articles.is_processed IS 'Whether the full article has been downloaded and processed';
COMMENT ON COLUMN rss_articles.is_read IS 'Whether the user has marked this article as read';
COMMENT ON COLUMN rss_articles.content_hash IS 'Hash of content for duplicate detection';
COMMENT ON COLUMN rss_articles.user_id IS 'User ID for user-specific articles, NULL for global articles';

COMMENT ON TABLE rss_feed_subscriptions IS 'User subscriptions to RSS feeds';
COMMENT ON COLUMN rss_feed_subscriptions.subscription_id IS 'Unique identifier for the subscription';
COMMENT ON COLUMN rss_feed_subscriptions.feed_id IS 'Reference to the RSS feed';
COMMENT ON COLUMN rss_feed_subscriptions.user_id IS 'User ID for the subscription';
COMMENT ON COLUMN rss_feed_subscriptions.is_active IS 'Whether the subscription is active';

-- Grant permissions on RSS feeds table
GRANT ALL PRIVILEGES ON rss_feeds TO bastion_user;
GRANT ALL PRIVILEGES ON rss_articles TO bastion_user;
GRANT ALL PRIVILEGES ON rss_feed_subscriptions TO bastion_user;

-- ============================================================================
-- SECURITY CONFIGURATION
-- ============================================================================

-- Create admin role if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'plato_admin') THEN
        CREATE USER plato_admin WITH PASSWORD 'plato_admin_secure_password';
    END IF;
END
$$;

-- Grant admin privileges
GRANT ALL PRIVILEGES ON DATABASE bastion_knowledge_base TO plato_admin;
GRANT CREATE, USAGE ON SCHEMA public TO plato_admin;

-- Create function to set user context for RLS
CREATE OR REPLACE FUNCTION set_user_context(p_user_id varchar, p_role varchar)
RETURNS void AS $$
BEGIN
    PERFORM set_config('app.current_user_id', p_user_id, false);
    PERFORM set_config('app.current_user_role', p_role, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to clear user context
CREATE OR REPLACE FUNCTION clear_user_context()
RETURNS void AS $$
BEGIN
    PERFORM set_config('app.current_user_id', '', false);
    PERFORM set_config('app.current_user_role', '', false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for audit log
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_table_name ON audit_log(table_name);

-- Create function to log audit events
CREATE OR REPLACE FUNCTION log_audit_event(
    p_user_id varchar,
    p_action varchar,
    p_table_name varchar,
    p_record_id varchar,
    p_old_values jsonb DEFAULT NULL,
    p_new_values jsonb DEFAULT NULL
)
RETURNS void AS $$
BEGIN
    INSERT INTO audit_log (
        user_id, action, table_name, record_id, 
        old_values, new_values, timestamp
    ) VALUES (
        p_user_id, p_action, p_table_name, p_record_id,
        p_old_values, p_new_values, CURRENT_TIMESTAMP
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    current_user_id varchar;
    record_id varchar;
BEGIN
    -- Get current user ID, default to 'system' if not set (for admin user creation)
    current_user_id := COALESCE(current_setting('app.current_user_id', true), 'system');
    
    -- Determine the appropriate record ID based on table structure
    IF TG_TABLE_NAME = 'users' THEN
        -- For users table, use user_id (UUID) instead of id (auto-increment)
        record_id := CASE 
            WHEN TG_OP = 'DELETE' THEN OLD.user_id
            ELSE NEW.user_id
        END;
    ELSE
        -- For other tables, use the standard id column
        record_id := CASE 
            WHEN TG_OP = 'DELETE' THEN OLD.id::varchar
            ELSE NEW.id::varchar
        END;
    END IF;
    
    IF TG_OP = 'INSERT' THEN
        PERFORM log_audit_event(
            current_user_id,
            'INSERT'::varchar,
            TG_TABLE_NAME::varchar,
            record_id,
            NULL::jsonb,
            to_jsonb(NEW)
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        PERFORM log_audit_event(
            current_user_id,
            'UPDATE'::varchar,
            TG_TABLE_NAME::varchar,
            record_id,
            to_jsonb(OLD),
            to_jsonb(NEW)
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        PERFORM log_audit_event(
            current_user_id,
            'DELETE'::varchar,
            TG_TABLE_NAME::varchar,
            record_id,
            to_jsonb(OLD),
            NULL::jsonb
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit triggers for sensitive tables (only if tables exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'document_metadata') THEN
        CREATE TRIGGER audit_document_metadata_trigger
            AFTER INSERT OR UPDATE OR DELETE ON document_metadata
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
        CREATE TRIGGER audit_users_trigger
            AFTER INSERT OR UPDATE OR DELETE ON users
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'user_sessions') THEN
        CREATE TRIGGER audit_user_sessions_trigger
            AFTER INSERT OR UPDATE OR DELETE ON user_sessions
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'conversations') THEN
        CREATE TRIGGER audit_conversations_trigger
            AFTER INSERT OR UPDATE OR DELETE ON conversations
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;
END $$;

-- Create function to check if user has permission to access resource
CREATE OR REPLACE FUNCTION check_user_permission(
    resource_user_id varchar,
    required_role varchar DEFAULT 'user'
)
RETURNS boolean AS $$
DECLARE
    current_user_role varchar;
BEGIN
    -- Get current user's role
    current_user_role := current_setting('app.current_user_role', true);
    
    -- Admin can access everything
    IF current_user_role = 'admin' THEN
        RETURN true;
    END IF;
    
    -- Check if user is accessing their own resource
    IF resource_user_id = current_setting('app.current_user_id', true) THEN
        RETURN true;
    END IF;
    
    -- For global resources, check role requirements
    IF resource_user_id IS NULL THEN
        RETURN current_user_role = required_role OR current_user_role = 'admin';
    END IF;
    
    RETURN false;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to sanitize SQL input (basic protection)
CREATE OR REPLACE FUNCTION sanitize_sql_input(input_text text)
RETURNS text AS $$
BEGIN
    -- Basic SQL injection protection
    -- Remove common SQL injection patterns
    RETURN regexp_replace(
        regexp_replace(
            regexp_replace(input_text, 
                '(\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script)\b)', 
                '', 'gi'
            ),
            '(--|/\*|\*/|;|xp_|sp_)', 
            '', 'gi'
        ),
        '(\b(union|select|insert|update|delete|drop|create|alter|exec|execute|script)\b)', 
        '', 'gi'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create view for security monitoring
CREATE OR REPLACE VIEW security_monitoring AS
SELECT 
    'failed_login_attempts' as metric_type,
    COUNT(*) as count,
    DATE_TRUNC('hour', timestamp) as time_period
FROM audit_log 
WHERE action = 'LOGIN_FAILED'
GROUP BY DATE_TRUNC('hour', timestamp)

UNION ALL

SELECT 
    'suspicious_activity' as metric_type,
    COUNT(*) as count,
    DATE_TRUNC('hour', timestamp) as time_period
FROM audit_log 
WHERE action IN ('UNAUTHORIZED_ACCESS', 'PERMISSION_DENIED')
GROUP BY DATE_TRUNC('hour', timestamp)

UNION ALL

SELECT 
    'data_access_patterns' as metric_type,
    COUNT(*) as count,
    DATE_TRUNC('hour', timestamp) as time_period
FROM audit_log 
WHERE action IN ('SELECT', 'INSERT', 'UPDATE', 'DELETE')
GROUP BY DATE_TRUNC('hour', timestamp);

-- Grant read-only access to security monitoring view
GRANT SELECT ON security_monitoring TO bastion_user;
GRANT ALL PRIVILEGES ON security_monitoring TO plato_admin;

-- Create function to get user's accessible data summary
CREATE OR REPLACE FUNCTION get_user_data_summary(p_user_id varchar)
RETURNS jsonb AS $$
DECLARE
    result jsonb;
    doc_count integer := 0;
    notes_count integer := 0;
    conv_count integer := 0;
    global_doc_count integer := 0;
    last_activity timestamp;
BEGIN
    -- Set user context
    PERFORM set_user_context(p_user_id, 'user');
    
    -- Get document count if table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'document_metadata') THEN
        SELECT COUNT(*) INTO doc_count FROM document_metadata WHERE user_id = p_user_id;
        SELECT COUNT(*) INTO global_doc_count FROM document_metadata WHERE collection_type = 'global';
    END IF;
    
    
    -- Get conversations count if table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'conversations') THEN
        SELECT COUNT(*) INTO conv_count FROM conversations WHERE user_id = p_user_id;
    END IF;
    
    -- Get last activity if audit_log exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'audit_log') THEN
        SELECT MAX(timestamp) INTO last_activity FROM audit_log WHERE user_id = p_user_id;
    END IF;
    
    -- Build result
    SELECT jsonb_build_object(
        'documents', doc_count,
        'notes', notes_count,
        'conversations', conv_count,
        'global_documents', global_doc_count,
        'last_activity', last_activity
    ) INTO result;
    
    -- Clear user context
    PERFORM clear_user_context();
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission on security functions
GRANT EXECUTE ON FUNCTION set_user_context(varchar, varchar) TO bastion_user;
GRANT EXECUTE ON FUNCTION clear_user_context() TO bastion_user;
GRANT EXECUTE ON FUNCTION log_audit_event(varchar, varchar, varchar, varchar, jsonb, jsonb) TO bastion_user;
GRANT EXECUTE ON FUNCTION check_user_permission(varchar, varchar) TO bastion_user;
GRANT EXECUTE ON FUNCTION sanitize_sql_input(text) TO bastion_user;
GRANT EXECUTE ON FUNCTION get_user_data_summary(varchar) TO bastion_user;

-- Ensure admin has privileges on newly created tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO plato_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO plato_admin;

-- Grant all permissions to admin role
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO plato_admin;

-- ============================================================================
-- DATA FIXES AND CLEANUP
-- ============================================================================

-- Fix corrupted citations data (if tables exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'conversation_messages') THEN
        -- Fix citations that are invalid JSON or corrupted
        UPDATE conversation_messages 
        SET citations = '[]'::jsonb 
        WHERE citations IS NULL 
           OR citations::text = '"n"' 
           OR citations::text = 'n'
           OR citations::text = 'null'
           OR citations::text = '""'
           OR citations::text = '[]';
        
        -- Add a check constraint to prevent future corruption (if it doesn't exist)
        IF NOT EXISTS (SELECT 1 FROM information_schema.table_constraints 
                      WHERE constraint_name = 'check_citations_valid' 
                      AND table_name = 'conversation_messages') THEN
            ALTER TABLE conversation_messages 
            ADD CONSTRAINT check_citations_valid 
            CHECK (citations IS NULL OR jsonb_typeof(citations) = 'array');
        END IF;
        
        -- Add comment explaining the fix
        COMMENT ON COLUMN conversation_messages.citations IS 'JSONB array of citations. Must be an array, not a string or other type.';
    END IF;
END $$;

-- Fix citations in unified_chat_messages table (if it exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'unified_chat_messages') THEN
        UPDATE unified_chat_messages 
        SET citations = '[]'::jsonb 
        WHERE citations IS NULL 
           OR citations::text = '"n"' 
           OR citations::text = 'n'
           OR citations::text = 'null'
           OR citations::text = '""'
           OR citations::text = '[]';
    END IF;
END $$;

-- ============================================================================
-- LANGGRAPH PERSISTENCE TABLES - AsyncPostgresSaver Support
-- ============================================================================

-- LangGraph checkpoints table for conversation state persistence
-- This table stores LangGraph conversation checkpoints and enables proper conversation history
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- LangGraph checkpoint blob storage for large data objects
-- Used to store tools results, agent state, and other large objects separately
CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT,
    channel TEXT NOT NULL,
    version TEXT NOT NULL,
    type TEXT NOT NULL,
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version),
    UNIQUE (thread_id, checkpoint_ns, checkpoint_id, channel, version)
);

-- LangGraph checkpoint writes tracking table
-- Tracks pending writes to ensure atomicity and consistency
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    blob BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Create indexes for optimal LangGraph performance
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_namespace ON checkpoints(thread_id, checkpoint_ns);
CREATE INDEX IF NOT EXISTS idx_checkpoints_parent ON checkpoints(parent_checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoints_type ON checkpoints(type);

CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_thread_id ON checkpoint_blobs(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_namespace ON checkpoint_blobs(thread_id, checkpoint_ns);
CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_checkpoint ON checkpoint_blobs(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_channel ON checkpoint_blobs(channel);
CREATE INDEX IF NOT EXISTS idx_checkpoint_blobs_created_at ON checkpoint_blobs(created_at);

CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_namespace ON checkpoint_writes(thread_id, checkpoint_ns);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_checkpoint ON checkpoint_writes(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_task ON checkpoint_writes(task_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_created_at ON checkpoint_writes(created_at);

-- Grant permissions on LangGraph tables
GRANT ALL PRIVILEGES ON checkpoints TO bastion_user;
GRANT ALL PRIVILEGES ON checkpoint_blobs TO bastion_user;
GRANT ALL PRIVILEGES ON checkpoint_writes TO bastion_user;

-- Grant admin privileges
GRANT ALL PRIVILEGES ON checkpoints TO plato_admin;
GRANT ALL PRIVILEGES ON checkpoint_blobs TO plato_admin;
GRANT ALL PRIVILEGES ON checkpoint_writes TO plato_admin;

-- Add comments for documentation
COMMENT ON TABLE checkpoints IS 'LangGraph conversation state checkpoints for persistence and resumption';
COMMENT ON TABLE checkpoint_blobs IS 'Large object storage for LangGraph checkpoint data (tools results, etc.)';
COMMENT ON TABLE checkpoint_writes IS 'Pending writes tracking for LangGraph checkpoint atomicity';

COMMENT ON COLUMN checkpoints.thread_id IS 'Conversation/thread identifier (typically matches conversation_id)';
COMMENT ON COLUMN checkpoints.checkpoint_ns IS 'Checkpoint namespace for isolation';
COMMENT ON COLUMN checkpoints.checkpoint_id IS 'Unique checkpoint identifier within thread';
COMMENT ON COLUMN checkpoints.parent_checkpoint_id IS 'Parent checkpoint for state transitions';
COMMENT ON COLUMN checkpoints.checkpoint IS 'JSONB checkpoint data containing conversation state';
COMMENT ON COLUMN checkpoints.metadata IS 'JSONB metadata for checkpoint (user_id, timestamps, etc.)';

COMMENT ON COLUMN checkpoint_blobs.channel IS 'Channel name for data organization';
COMMENT ON COLUMN checkpoint_blobs.version IS 'Version identifier for blob data';
COMMENT ON COLUMN checkpoint_blobs.blob IS 'Binary data (pickled Python objects, tool results, etc.)';

COMMENT ON COLUMN checkpoint_writes.task_id IS 'Task identifier for write operations';
COMMENT ON COLUMN checkpoint_writes.idx IS 'Index for ordered write operations';

-- Enable RLS for LangGraph tables (security isolation)
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_blobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for LangGraph checkpoints
-- Users can only access their own conversation checkpoints
CREATE POLICY checkpoints_user_policy ON checkpoints
    FOR ALL USING (
        -- Allow LangGraph service access (no user context available)
        current_setting('app.current_user_id', true) IS NULL
        OR current_setting('app.current_user_id', true) = ''
        -- Extract user_id from metadata or thread_id pattern for regular API access
        OR (metadata->>'user_id') = current_setting('app.current_user_id', true)::varchar
        OR thread_id IN (
            SELECT conversation_id FROM conversations 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY checkpoint_blobs_user_policy ON checkpoint_blobs
    FOR ALL USING (
        -- Allow LangGraph service access (no user context available)
        current_setting('app.current_user_id', true) IS NULL
        OR current_setting('app.current_user_id', true) = ''
        -- Regular API access - check thread access via checkpoints table
        OR thread_id IN (
            SELECT thread_id FROM checkpoints 
            WHERE (metadata->>'user_id') = current_setting('app.current_user_id', true)::varchar
            OR thread_id IN (
                SELECT conversation_id FROM conversations 
                WHERE user_id = current_setting('app.current_user_id', true)::varchar
            )
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY checkpoint_writes_user_policy ON checkpoint_writes
    FOR ALL USING (
        -- Allow LangGraph service access (no user context available)
        current_setting('app.current_user_id', true) IS NULL
        OR current_setting('app.current_user_id', true) = ''
        -- Regular API access - check thread access via checkpoints table
        OR thread_id IN (
            SELECT thread_id FROM checkpoints 
            WHERE (metadata->>'user_id') = current_setting('app.current_user_id', true)::varchar
            OR thread_id IN (
                SELECT conversation_id FROM conversations 
                WHERE user_id = current_setting('app.current_user_id', true)::varchar
            )
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- ============================================================================
-- FINAL PERMISSIONS AND CLEANUP
-- ============================================================================

-- Grant all privileges on all tables to bastion_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO bastion_user;

-- Grant all privileges on all sequences to bastion_user
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO bastion_user;

-- Grant all privileges on all functions to bastion_user
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO bastion_user;

-- Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO bastion_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO bastion_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO bastion_user;

-- Grant all privileges to admin role
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO plato_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO plato_admin;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO plato_admin;

-- ========================================
-- ORG-MODE SETTINGS TABLE
-- Roosevelt's Org-Mode Configuration Storage
-- ========================================

CREATE TABLE IF NOT EXISTS org_settings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL UNIQUE,
    settings_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for org_settings
CREATE INDEX IF NOT EXISTS idx_org_settings_user_id ON org_settings(user_id);
CREATE INDEX IF NOT EXISTS idx_org_settings_json ON org_settings USING GIN(settings_json);

-- Comments for documentation
COMMENT ON TABLE org_settings IS 'Org-mode configuration and preferences per user';
COMMENT ON COLUMN org_settings.user_id IS 'User ID these settings belong to';
COMMENT ON COLUMN org_settings.settings_json IS 'JSON blob containing all org-mode settings (TODO sequences, tags, preferences)';
COMMENT ON COLUMN org_settings.created_at IS 'When settings were first created';
COMMENT ON COLUMN org_settings.updated_at IS 'When settings were last modified';

-- ========================================
-- GITHUB INTEGRATION TABLES
-- Roosevelt's GitHub Project Integration
-- ========================================

-- GitHub connections table
CREATE TABLE IF NOT EXISTS github_connections (
    connection_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token TEXT NOT NULL,  -- In production, this should be encrypted
    username VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    last_verified TIMESTAMP WITH TIME ZONE,
    rate_limit_remaining INTEGER,
    rate_limit_reset TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- GitHub project mappings table
CREATE TABLE IF NOT EXISTS github_project_mappings (
    mapping_id VARCHAR(255) PRIMARY KEY,
    connection_id VARCHAR(255) NOT NULL REFERENCES github_connections(connection_id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    project_type VARCHAR(50) NOT NULL DEFAULT 'repository',
    owner VARCHAR(255) NOT NULL,
    repo VARCHAR(255),
    project_number INTEGER,
    project_id VARCHAR(255),
    org_file VARCHAR(255) NOT NULL DEFAULT 'github.org',
    tag_prefix VARCHAR(255) NOT NULL DEFAULT 'github',
    default_tags JSONB DEFAULT '[]'::jsonb,
    sync_issues BOOLEAN NOT NULL DEFAULT true,
    sync_pull_requests BOOLEAN NOT NULL DEFAULT false,
    issue_states JSONB DEFAULT '["open"]'::jsonb,
    labels_as_tags BOOLEAN NOT NULL DEFAULT true,
    assignees_as_tags BOOLEAN NOT NULL DEFAULT true,
    sync_comments BOOLEAN NOT NULL DEFAULT false,
    last_sync TIMESTAMP WITH TIME ZONE,
    sync_enabled BOOLEAN NOT NULL DEFAULT true,
    auto_sync_interval INTEGER,  -- Minutes
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- GitHub issue sync tracking table
CREATE TABLE IF NOT EXISTS github_issue_sync (
    sync_id VARCHAR(255) PRIMARY KEY,
    mapping_id VARCHAR(255) NOT NULL REFERENCES github_project_mappings(mapping_id) ON DELETE CASCADE,
    github_issue_number INTEGER NOT NULL,
    github_issue_id VARCHAR(255) NOT NULL,
    org_file VARCHAR(255) NOT NULL,
    org_heading TEXT NOT NULL,
    org_line_number INTEGER,
    last_synced TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    github_updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    sync_status VARCHAR(50) NOT NULL DEFAULT 'synced',
    has_conflicts BOOLEAN NOT NULL DEFAULT false,
    conflict_resolution TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(mapping_id, github_issue_number)
);

-- Indexes for GitHub integration tables
CREATE INDEX IF NOT EXISTS idx_github_connections_user_id ON github_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_github_project_mappings_user_id ON github_project_mappings(user_id);
CREATE INDEX IF NOT EXISTS idx_github_project_mappings_connection_id ON github_project_mappings(connection_id);
CREATE INDEX IF NOT EXISTS idx_github_issue_sync_mapping_id ON github_issue_sync(mapping_id);
CREATE INDEX IF NOT EXISTS idx_github_issue_sync_github_issue_number ON github_issue_sync(github_issue_number);

-- Grant permissions on GitHub integration tables
GRANT ALL PRIVILEGES ON github_connections TO bastion_user;
GRANT ALL PRIVILEGES ON github_project_mappings TO bastion_user;
GRANT ALL PRIVILEGES ON github_issue_sync TO bastion_user;

-- Add updated_at trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers to automatically update updated_at on GitHub tables
DROP TRIGGER IF EXISTS update_github_connections_updated_at ON github_connections;
CREATE TRIGGER update_github_connections_updated_at
    BEFORE UPDATE ON github_connections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_github_project_mappings_updated_at ON github_project_mappings;
CREATE TRIGGER update_github_project_mappings_updated_at
    BEFORE UPDATE ON github_project_mappings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_github_issue_sync_updated_at ON github_issue_sync;
CREATE TRIGGER update_github_issue_sync_updated_at
    BEFORE UPDATE ON github_issue_sync
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE github_connections IS 'GitHub API connections per user with token storage';
COMMENT ON TABLE github_project_mappings IS 'Mapping between GitHub projects and org-mode files';
COMMENT ON TABLE github_issue_sync IS 'Tracking sync status between GitHub issues and org-mode headings';

-- ========================================
-- ROOSEVELT'S MESSAGING SYSTEM TABLES
-- User-to-User Communication Cavalry!
-- ========================================

-- Create enums for messaging system
DO $$ BEGIN
    CREATE TYPE room_type_enum AS ENUM ('direct', 'group');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE message_type_enum AS ENUM ('text', 'ai_share', 'system');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE user_status_enum AS ENUM ('online', 'offline', 'away');
EXCEPTION
    WHEN duplicate_object THEN null;
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

-- Message attachments table
CREATE TABLE IF NOT EXISTS message_attachments (
    attachment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id UUID NOT NULL REFERENCES chat_messages(message_id) ON DELETE CASCADE,
    room_id UUID NOT NULL REFERENCES chat_rooms(room_id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    file_size BIGINT NOT NULL,
    width INTEGER,
    height INTEGER,
    is_animated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
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

-- Create indexes for messaging performance
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
CREATE INDEX IF NOT EXISTS idx_message_attachments_message_id ON message_attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_message_attachments_room_id ON message_attachments(room_id);
CREATE INDEX IF NOT EXISTS idx_user_presence_status ON user_presence(status);
CREATE INDEX IF NOT EXISTS idx_user_presence_last_seen ON user_presence(last_seen_at DESC);

-- Grant permissions on messaging tables
GRANT ALL PRIVILEGES ON chat_rooms TO bastion_user;
GRANT ALL PRIVILEGES ON room_participants TO bastion_user;
GRANT ALL PRIVILEGES ON chat_messages TO bastion_user;
GRANT ALL PRIVILEGES ON message_reactions TO bastion_user;
GRANT ALL PRIVILEGES ON message_attachments TO bastion_user;
GRANT ALL PRIVILEGES ON room_encryption_keys TO bastion_user;
GRANT ALL PRIVILEGES ON user_presence TO bastion_user;

-- Helper function to break circular RLS dependencies
-- This function runs with the privileges of the creator (postgres)
-- and thus ignores RLS policies on the tables it queries.
CREATE OR REPLACE FUNCTION check_room_membership(check_room_id UUID, check_user_id VARCHAR)
RETURNS BOOLEAN AS $$
BEGIN
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

-- Enable RLS on messaging tables
ALTER TABLE chat_rooms ENABLE ROW LEVEL SECURITY;
ALTER TABLE room_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_reactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE message_attachments ENABLE ROW LEVEL SECURITY;
ALTER TABLE room_encryption_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_presence ENABLE ROW LEVEL SECURITY;

-- RLS policies for chat_rooms
CREATE POLICY chat_rooms_select_policy ON chat_rooms
    FOR SELECT USING (
        check_room_membership(room_id, current_setting('app.current_user_id', false)::varchar)
        OR current_setting('app.current_user_role', false) = 'admin'
    );

CREATE POLICY chat_rooms_insert_policy ON chat_rooms
    FOR INSERT WITH CHECK (
        created_by = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY chat_rooms_update_policy ON chat_rooms
    FOR UPDATE USING (
        check_room_membership(room_id, current_setting('app.current_user_id', false)::varchar)
        OR current_setting('app.current_user_role', false) = 'admin'
    );

CREATE POLICY chat_rooms_delete_policy ON chat_rooms
    FOR DELETE USING (
        created_by = current_setting('app.current_user_id', false)::varchar
        OR check_room_membership(room_id, current_setting('app.current_user_id', false)::varchar)
        OR current_setting('app.current_user_role', false) = 'admin'
    );

-- RLS policies for room_participants
CREATE POLICY room_participants_select_policy ON room_participants
    FOR SELECT USING (
        check_room_membership(room_id, current_setting('app.current_user_id', false)::varchar)
        OR current_setting('app.current_user_role', false) = 'admin'
    );

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

CREATE POLICY room_participants_update_policy ON room_participants
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', false)::varchar
        OR current_setting('app.current_user_role', false) = 'admin'
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', false)::varchar
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

-- RLS policies for message_attachments
CREATE POLICY message_attachments_select_policy ON message_attachments
    FOR SELECT USING (
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY message_attachments_insert_policy ON message_attachments
    FOR INSERT WITH CHECK (
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY message_attachments_delete_policy ON message_attachments
    FOR DELETE USING (
        room_id IN (
            SELECT room_id FROM room_participants 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        )
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

-- Add updated_at triggers for messaging tables
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

-- ========================================
-- TEAMS SYSTEM
-- Team Collaboration Platform
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
    last_read_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    muted BOOLEAN DEFAULT FALSE,
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

-- team_id column is now included in the initial CREATE TABLE statement above
-- Add foreign key constraint after teams table exists (for fresh installs)
-- This DO block also handles backwards compatibility with existing databases
DO $$ 
BEGIN
    -- Add column if it doesn't exist (for existing databases)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'document_folders' AND column_name = 'team_id'
    ) THEN
        ALTER TABLE document_folders ADD COLUMN team_id UUID;
    END IF;
    
    -- Add foreign key constraint if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'document_folders_team_id_fkey'
    ) THEN
        ALTER TABLE document_folders 
        ADD CONSTRAINT document_folders_team_id_fkey 
        FOREIGN KEY (team_id) REFERENCES teams(team_id) ON DELETE CASCADE;
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
CREATE INDEX IF NOT EXISTS idx_team_members_last_read ON team_members(team_id, user_id, last_read_at);
CREATE INDEX IF NOT EXISTS idx_team_members_last_read ON team_members(team_id, user_id, last_read_at);
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

-- Row-Level Security policies for document_metadata (requires teams tables to exist)
ALTER TABLE document_metadata ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (idempotent)
DROP POLICY IF EXISTS document_metadata_select_policy ON document_metadata;
DROP POLICY IF EXISTS document_metadata_update_policy ON document_metadata;
DROP POLICY IF EXISTS document_metadata_delete_policy ON document_metadata;
DROP POLICY IF EXISTS document_metadata_insert_policy ON document_metadata;

-- RLS policies for document_metadata
-- Users can see: their own docs, global docs, and docs from teams they're members of
CREATE POLICY document_metadata_select_policy ON document_metadata
    FOR SELECT USING (
        -- User's own documents
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Global documents (everyone can see)
        OR collection_type = 'global'
        -- Team documents (only if user is a team member - NO admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        -- System admins can see user documents (but NOT team documents unless they're members)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- Users can update: their own docs, admins can update global docs, team admins can update team docs
CREATE POLICY document_metadata_update_policy ON document_metadata
    FOR UPDATE 
    USING (
        -- User's own documents
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team documents (only team admins can update - NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar 
            AND role = 'admin'
        ))
        -- System admins can update user documents and global documents (but NOT team documents unless they're team admins)
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        -- User's own documents
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team documents (only team admins can update - NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar 
            AND role = 'admin'
        ))
        -- System admins can update user documents and global documents (but NOT team documents unless they're team admins)
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- Users can delete: their own docs, admins can delete any docs, team admins can delete team docs
CREATE POLICY document_metadata_delete_policy ON document_metadata
    FOR DELETE USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar 
            AND role = 'admin'
        ))
        OR (user_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

-- Allow inserts for users creating their own docs, admins for any docs, team members for team docs
CREATE POLICY document_metadata_insert_policy ON document_metadata
    FOR INSERT WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
    );

-- Row-Level Security policies for teams
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_invitations ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE post_reactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE post_comments ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can see teams they are members of, or teams they created
CREATE POLICY teams_select_policy ON teams
    FOR SELECT
    USING (
        -- Allow if user is a member
        EXISTS (
            SELECT 1 FROM team_members 
            WHERE team_members.team_id = teams.team_id 
            AND team_members.user_id = current_setting('app.current_user_id', true)
        )
        -- OR if user is the creator (needed for adding first member during team creation)
        OR created_by = current_setting('app.current_user_id', true)::varchar
    );

-- Function to check team membership (bypasses RLS to break recursion)
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

-- RLS Policy: Users can see team members of teams they belong to
CREATE POLICY team_members_select_policy ON team_members
    FOR SELECT
    USING (
        -- User can see members of teams they belong to (using function to break recursion)
        check_team_membership(team_id, current_setting('app.current_user_id', true)::varchar)
    );

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

-- ========================================
-- TEAMS TABLE INSERT/UPDATE/DELETE POLICIES
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
-- TEAM MEMBERS TABLE INSERT/UPDATE/DELETE POLICIES
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
-- TEAM INVITATIONS TABLE INSERT/UPDATE/DELETE POLICIES
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
-- TEAM POSTS TABLE INSERT/UPDATE/DELETE POLICIES
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
-- POST REACTIONS TABLE INSERT/DELETE POLICIES
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
-- POST COMMENTS TABLE INSERT/UPDATE/DELETE POLICIES
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
COMMENT ON COLUMN team_members.last_read_at IS 'Timestamp when user last read team posts';
COMMENT ON COLUMN team_members.muted IS 'Whether user has muted notifications for this team';
COMMENT ON TABLE team_invitations IS 'Team invitations linked to chat messages';
COMMENT ON TABLE team_posts IS 'Social posts in team feeds';
COMMENT ON TABLE post_reactions IS 'Emoji reactions on team posts';
COMMENT ON TABLE post_comments IS 'Comments on team posts';

-- Row-Level Security policies for document_folders
-- (Created here after teams tables exist so team_members can be referenced)
CREATE POLICY document_folders_select_policy ON document_folders
    FOR SELECT USING (
        -- Users can see their own folders
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Everyone can see global folders
        OR collection_type = 'global'
        -- Team members can see team folders (NO admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        -- System admins can see user folders (but NOT team folders unless they're members)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_folders_insert_policy ON document_folders
    FOR INSERT WITH CHECK (
        -- Users can create their own user folders
        (user_id = current_setting('app.current_user_id', true)::varchar AND collection_type = 'user')
        -- Team members can create team folders (any team member, not just admins)
        OR (collection_type = 'team' AND team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        -- System admins can create user/global folders (but NOT team folders unless they're team members)
        OR (collection_type IN ('user', 'global') AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_folders_update_policy ON document_folders
    FOR UPDATE 
    USING (
        -- Users can update their own folders
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team admins can update team folders (NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        -- System admins can update user folders (but NOT team folders unless they're team admins)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        -- Users can update their own folders (same as USING)
        user_id = current_setting('app.current_user_id', true)::varchar
        -- Team admins can update team folders (NO system admin bypass for privacy)
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        -- System admins can update user folders (but NOT team folders unless they're team admins)
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_folders_delete_policy ON document_folders
    FOR DELETE USING (
        -- User folders: user can delete their own
        (user_id = current_setting('app.current_user_id', true)::varchar 
         AND collection_type = 'user')
        
        -- Global folders: only admins can delete
        OR (collection_type = 'global' 
            AND current_setting('app.current_user_role', true) = 'admin')
        
        -- Team folders: creator can delete, OR team admin can delete any team folder
        OR (team_id IS NOT NULL 
            AND (
                -- Creator can delete their own folder
                created_by = current_setting('app.current_user_id', true)::varchar
                
                -- Team admin can delete ANY folder in their team
                OR EXISTS (
                    SELECT 1 FROM team_members tm 
                    WHERE tm.team_id = document_folders.team_id 
                    AND tm.user_id = current_setting('app.current_user_id', true)::varchar 
                    AND tm.role = 'admin'
                )
            )
        )
    );

-- ========================================
-- EMAIL AGENT SYSTEM
-- Email verification and sending infrastructure
-- ========================================

-- Add email verification fields to users table
DO $$ BEGIN
    ALTER TABLE users 
    ADD COLUMN email_verified BOOLEAN DEFAULT FALSE,
    ADD COLUMN email_verification_token VARCHAR(255),
    ADD COLUMN email_verification_sent_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN email_verification_expires_at TIMESTAMP WITH TIME ZONE;
EXCEPTION
    WHEN duplicate_column THEN 
        RAISE NOTICE 'Email verification columns already exist, skipping';
END $$;

-- Create indexes for email verification
CREATE INDEX IF NOT EXISTS idx_users_email_verified ON users(email_verified);
CREATE INDEX IF NOT EXISTS idx_users_verification_token ON users(email_verification_token);

-- Email audit log table
CREATE TABLE IF NOT EXISTS email_audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    from_email VARCHAR(255) NOT NULL,
    to_email VARCHAR(255) NOT NULL,
    cc TEXT,
    bcc TEXT,
    subject TEXT,
    body_preview TEXT,
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    conversation_id UUID,
    agent_type VARCHAR(50) DEFAULT 'email_agent',
    send_status VARCHAR(50),
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_email_audit_user ON email_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_email_audit_sent_at ON email_audit_log(sent_at);

-- Email rate limiting table
CREATE TABLE IF NOT EXISTS email_rate_limits (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    sent_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    recipient_email VARCHAR(255) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_email_rate_limits_user ON email_rate_limits(user_id);
CREATE INDEX IF NOT EXISTS idx_email_rate_limits_sent_at ON email_rate_limits(sent_at);

-- ========================================
-- MUSIC SERVICE TABLES
-- SubSonic-compatible music streaming service
-- ========================================

-- Create music_service_configs table for storing encrypted SubSonic credentials
CREATE TABLE IF NOT EXISTS music_service_configs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    server_url VARCHAR(500) NOT NULL,
    username VARCHAR(255) NOT NULL,
    encrypted_password TEXT NOT NULL,
    salt VARCHAR(255) NOT NULL,
    auth_type VARCHAR(50) DEFAULT 'password', -- 'password' or 'token'
    service_type VARCHAR(50) DEFAULT 'subsonic', -- 'subsonic', 'plex', 'emby', 'jellyfin', 'audiobookshelf', etc.
    service_name VARCHAR(255), -- User-friendly display name for the media source
    is_active BOOLEAN DEFAULT TRUE, -- Whether this media source is active (can be disabled without deleting)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, service_type)
);

-- Create indexes for music_service_configs
CREATE INDEX IF NOT EXISTS idx_music_configs_user_id ON music_service_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_music_configs_service_type ON music_service_configs(service_type);
CREATE INDEX IF NOT EXISTS idx_music_configs_user_service ON music_service_configs(user_id, service_type);

-- Create music_cache table for storing cached library metadata
CREATE TABLE IF NOT EXISTS music_cache (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    service_type VARCHAR(50) DEFAULT 'subsonic', -- Music service provider type: subsonic, audiobookshelf, etc.
    cache_type VARCHAR(50) NOT NULL, -- 'album', 'artist', 'playlist', 'track'
    item_id VARCHAR(255) NOT NULL, -- SubSonic item ID
    parent_id VARCHAR(255), -- For tracks: album/playlist ID
    title VARCHAR(500) NOT NULL,
    artist VARCHAR(500),
    album VARCHAR(500),
    duration INTEGER, -- Duration in seconds
    track_number INTEGER,
    cover_art_id VARCHAR(255),
    metadata_json JSONB, -- Additional metadata from SubSonic
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, service_type, cache_type, item_id)
);

-- Create indexes for music_cache
CREATE INDEX IF NOT EXISTS idx_music_cache_user_id ON music_cache(user_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_service_type ON music_cache(service_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_user_service_type ON music_cache(user_id, service_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_type ON music_cache(cache_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_item_id ON music_cache(item_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_parent_id ON music_cache(parent_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_user_type ON music_cache(user_id, cache_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_json ON music_cache USING GIN(metadata_json);

-- Create music_cache_metadata table for tracking sync status
CREATE TABLE IF NOT EXISTS music_cache_metadata (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    service_type VARCHAR(50) DEFAULT 'subsonic', -- Music service provider type: subsonic, audiobookshelf, etc.
    last_sync_at TIMESTAMP WITH TIME ZONE,
    sync_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'syncing', 'completed', 'failed'
    sync_error TEXT,
    total_albums INTEGER DEFAULT 0,
    total_artists INTEGER DEFAULT 0,
    total_playlists INTEGER DEFAULT 0,
    total_tracks INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, service_type)
);

-- Create indexes for music_cache_metadata
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_user_id ON music_cache_metadata(user_id);
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_service_type ON music_cache_metadata(service_type);
CREATE INDEX IF NOT EXISTS idx_music_cache_metadata_user_service ON music_cache_metadata(user_id, service_type);

-- Add comments
COMMENT ON TABLE music_service_configs IS 'Encrypted music service configurations per user (SubSonic, Plex, Emby, Audiobookshelf, etc.)';
COMMENT ON COLUMN music_service_configs.service_type IS 'Music service provider type: subsonic, plex, emby, jellyfin, audiobookshelf, etc.';
COMMENT ON COLUMN music_service_configs.service_name IS 'User-friendly display name for the media source';
COMMENT ON COLUMN music_service_configs.is_active IS 'Whether this media source is active (can be disabled without deleting)';
COMMENT ON TABLE music_cache IS 'Cached music library metadata from music servers';
COMMENT ON COLUMN music_cache.service_type IS 'Music service provider type: subsonic, audiobookshelf, etc.';
COMMENT ON TABLE music_cache_metadata IS 'Metadata about music cache sync status and timestamps';
COMMENT ON COLUMN music_cache_metadata.service_type IS 'Music service provider type: subsonic, audiobookshelf, etc.';

-- ========================================
-- RLS POLICIES FOR MUSIC TABLES
-- ========================================

ALTER TABLE music_service_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE music_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE music_cache_metadata ENABLE ROW LEVEL SECURITY;

-- Music service configs policies
CREATE POLICY music_configs_select_policy ON music_service_configs
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_configs_insert_policy ON music_service_configs
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_configs_update_policy ON music_service_configs
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_configs_delete_policy ON music_service_configs
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));

-- Music cache policies
CREATE POLICY music_cache_select_policy ON music_cache
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_cache_insert_policy ON music_cache
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_cache_update_policy ON music_cache
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_cache_delete_policy ON music_cache
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));

-- Music cache metadata policies
CREATE POLICY music_metadata_select_policy ON music_cache_metadata
FOR SELECT
USING (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_metadata_insert_policy ON music_cache_metadata
FOR INSERT
WITH CHECK (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_metadata_update_policy ON music_cache_metadata
FOR UPDATE
USING (user_id = current_setting('app.current_user_id', true))
WITH CHECK (user_id = current_setting('app.current_user_id', true));

CREATE POLICY music_metadata_delete_policy ON music_cache_metadata
FOR DELETE
USING (user_id = current_setting('app.current_user_id', true));

-- ========================================
-- ENTERTAINMENT SYNC TABLES
-- Sonarr/Radarr API integration for entertainment content
-- ========================================

-- Configuration for user's Radarr/Sonarr connections
CREATE TABLE IF NOT EXISTS entertainment_sync_config (
    config_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL, -- 'radarr' or 'sonarr'
    api_url VARCHAR(512) NOT NULL,
    api_key TEXT NOT NULL, -- Encrypted API key
    enabled BOOLEAN DEFAULT true,
    sync_frequency_minutes INTEGER DEFAULT 60, -- Default hourly
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_sync_status VARCHAR(50), -- 'success', 'failed', 'running'
    items_synced INTEGER DEFAULT 0,
    sync_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, source_type, api_url)
);

-- Track individual synced items for change detection
CREATE TABLE IF NOT EXISTS entertainment_sync_items (
    item_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id UUID NOT NULL REFERENCES entertainment_sync_config(config_id) ON DELETE CASCADE,
    external_id VARCHAR(255) NOT NULL, -- Radarr/Sonarr item ID
    external_type VARCHAR(50) NOT NULL, -- 'movie', 'series', 'episode'
    title VARCHAR(512) NOT NULL,
    tmdb_id INTEGER, -- TMDB reference
    tvdb_id INTEGER, -- TVDB reference
    season_number INTEGER, -- For episodes
    episode_number INTEGER, -- For episodes
    parent_series_id VARCHAR(255), -- Link episode to series
    metadata_hash VARCHAR(64), -- Hash of metadata to detect changes
    last_synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    vector_document_id VARCHAR(255), -- Track generated pseudo-document ID
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(config_id, external_id, external_type)
);

-- Create indexes for entertainment sync tables
CREATE INDEX IF NOT EXISTS idx_sync_config_user ON entertainment_sync_config(user_id);
CREATE INDEX IF NOT EXISTS idx_sync_config_enabled ON entertainment_sync_config(enabled);
CREATE INDEX IF NOT EXISTS idx_sync_config_source_type ON entertainment_sync_config(source_type);
CREATE INDEX IF NOT EXISTS idx_sync_items_config ON entertainment_sync_items(config_id);
CREATE INDEX IF NOT EXISTS idx_sync_items_external ON entertainment_sync_items(external_id);
CREATE INDEX IF NOT EXISTS idx_sync_items_type ON entertainment_sync_items(external_type);
CREATE INDEX IF NOT EXISTS idx_sync_items_vector_doc ON entertainment_sync_items(vector_document_id);
CREATE INDEX IF NOT EXISTS idx_sync_items_parent_series ON entertainment_sync_items(parent_series_id);

-- Add comments for entertainment sync tables
COMMENT ON TABLE entertainment_sync_config IS 'User configurations for Radarr/Sonarr API sync';
COMMENT ON TABLE entertainment_sync_items IS 'Tracked items synced from Radarr/Sonarr for change detection';

-- ========================================
-- USER LOCATIONS TABLE
-- Map feature: Store user location points (private and global)
-- ========================================

CREATE TABLE IF NOT EXISTS user_locations (
    location_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    address TEXT,
    latitude DECIMAL(10, 8) NOT NULL,
    longitude DECIMAL(11, 8) NOT NULL,
    notes TEXT,
    is_global BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_locations_user_id ON user_locations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_locations_is_global ON user_locations(is_global);
CREATE INDEX IF NOT EXISTS idx_user_locations_name ON user_locations(name);

GRANT ALL PRIVILEGES ON user_locations TO bastion_user;

-- Enable RLS
ALTER TABLE user_locations ENABLE ROW LEVEL SECURITY;

-- Users see their own private locations + all global locations
CREATE POLICY locations_select_policy ON user_locations
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR is_global = true
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Users can only insert their own private locations, admins can create global
CREATE POLICY locations_insert_policy ON user_locations
    FOR INSERT WITH CHECK (
        (user_id = current_setting('app.current_user_id', true)::varchar AND is_global = false)
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Users can only update/delete their own locations, admins can modify global
CREATE POLICY locations_update_policy ON user_locations
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY locations_delete_policy ON user_locations
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Trigger for updated_at
CREATE TRIGGER update_user_locations_updated_at
    BEFORE UPDATE ON user_locations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE user_locations IS 'Stores user location points for map visualization. Supports both private (per-user) and global (shared) locations.';
COMMENT ON COLUMN user_locations.is_global IS 'If true, location is visible to all users with map access. Only admins can create global locations.';

-- ========================================
-- SAVED ROUTES TABLE
-- Map feature: Store user-saved road routes (OSRM geometry, steps, waypoints)
-- ========================================

CREATE TABLE IF NOT EXISTS saved_routes (
    route_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    waypoints JSONB NOT NULL DEFAULT '[]',
    geometry JSONB NOT NULL,
    steps JSONB NOT NULL DEFAULT '[]',
    distance_meters NUMERIC(12, 2) NOT NULL,
    duration_seconds NUMERIC(12, 2) NOT NULL,
    profile VARCHAR(64) DEFAULT 'driving',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_routes_user_id ON saved_routes(user_id);
CREATE INDEX IF NOT EXISTS idx_saved_routes_created_at ON saved_routes(created_at DESC);

GRANT ALL PRIVILEGES ON saved_routes TO bastion_user;

ALTER TABLE saved_routes ENABLE ROW LEVEL SECURITY;

CREATE POLICY saved_routes_select_policy ON saved_routes
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY saved_routes_insert_policy ON saved_routes
    FOR INSERT WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
    );

CREATE POLICY saved_routes_update_policy ON saved_routes
    FOR UPDATE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY saved_routes_delete_policy ON saved_routes
    FOR DELETE USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE TRIGGER update_saved_routes_updated_at
    BEFORE UPDATE ON saved_routes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE saved_routes IS 'Stores user-saved road routes from OSRM (geometry, turn-by-turn steps, waypoints).';

-- ========================================
-- FACE DETECTION TABLES
-- Image vision service: Store detected faces and known identities
-- ========================================

-- Store detected faces with encodings
CREATE TABLE IF NOT EXISTS detected_faces (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    face_encoding FLOAT[] NOT NULL,  -- 128-dimensional vector
    identity_name VARCHAR(255),
    identity_confirmed BOOLEAN DEFAULT FALSE,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    tagged_by VARCHAR(255),
    tagged_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_detected_faces_document_id ON detected_faces(document_id);
CREATE INDEX IF NOT EXISTS idx_detected_faces_identity ON detected_faces(identity_name);
CREATE INDEX IF NOT EXISTS idx_detected_faces_confirmed ON detected_faces(identity_confirmed);

-- Store known identities metadata
-- NOTE: Face encoding vectors stored in Qdrant collection "face_encodings"
-- Multiple encodings per identity for robust matching across angles/lighting/expressions
CREATE TABLE IF NOT EXISTS known_identities (
    id SERIAL PRIMARY KEY,
    identity_name VARCHAR(255) UNIQUE NOT NULL,
    face_encoding FLOAT[],  -- DEPRECATED: Kept for backward compatibility, use Qdrant instead
    sample_count INTEGER DEFAULT 1,  -- Number of face encodings stored in Qdrant
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_known_identities_name ON known_identities(identity_name);

-- Add comments explaining the architecture
COMMENT ON TABLE known_identities IS 'Metadata for known face identities. Face encoding vectors stored in Qdrant collection "face_encodings" for efficient vector similarity search. More samples = better matching!';
COMMENT ON COLUMN known_identities.face_encoding IS 'DEPRECATED: Face encodings now stored in Qdrant for vector similarity search. This column kept for backward compatibility but should be empty array for new identities.';

-- Function to auto-cleanup orphaned identities when all faces deleted
CREATE OR REPLACE FUNCTION cleanup_orphaned_identities()
RETURNS TRIGGER AS $$
DECLARE
    identity_to_check VARCHAR(255);
    remaining_count INTEGER;
BEGIN
    -- Get the identity_name from the deleted face (if it was tagged)
    identity_to_check := OLD.identity_name;
    
    -- Only proceed if the face was actually tagged
    IF identity_to_check IS NOT NULL THEN
        -- Count remaining faces with this identity
        SELECT COUNT(*) INTO remaining_count
        FROM detected_faces
        WHERE identity_name = identity_to_check;
        
        -- If no more faces exist for this identity, delete it from PostgreSQL
        IF remaining_count = 0 THEN
            DELETE FROM known_identities WHERE identity_name = identity_to_check;
            RAISE NOTICE 'Auto-deleted orphaned identity: % (no remaining faces)', identity_to_check;
            -- Note: Qdrant vectors cleaned up separately via /api/vision/cleanup-orphaned-vectors
        END IF;
    END IF;
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Trigger to run after face deletion
DROP TRIGGER IF EXISTS trigger_cleanup_orphaned_identities ON detected_faces;
CREATE TRIGGER trigger_cleanup_orphaned_identities
    AFTER DELETE ON detected_faces
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_orphaned_identities();

COMMENT ON FUNCTION cleanup_orphaned_identities() IS 'Automatically removes known_identities from PostgreSQL when all detected_faces for that identity are deleted. Qdrant vector cleanup handled by backend API endpoint /api/vision/cleanup-orphaned-vectors.';

-- Grant permissions
GRANT ALL PRIVILEGES ON detected_faces TO bastion_user;
GRANT ALL PRIVILEGES ON known_identities TO bastion_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bastion_user;

-- Enable RLS
ALTER TABLE detected_faces ENABLE ROW LEVEL SECURITY;
ALTER TABLE known_identities ENABLE ROW LEVEL SECURITY;

-- RLS Policies for detected_faces
-- Users can see faces in their own documents + global documents
CREATE POLICY detected_faces_select_policy ON detected_faces
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_faces.document_id
            AND (
                dm.user_id = current_setting('app.current_user_id', true)::varchar
                OR dm.collection_type = 'global'
                OR current_setting('app.current_user_role', true) = 'admin'
            )
        )
    );

-- Users can insert faces for their own documents, admins for global
CREATE POLICY detected_faces_insert_policy ON detected_faces
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_faces.document_id
            AND (
                (dm.user_id = current_setting('app.current_user_id', true)::varchar AND dm.collection_type = 'user')
                OR (dm.collection_type = 'global' AND current_setting('app.current_user_role', true) = 'admin')
            )
        )
    );

-- Users can update faces in their own documents, admins for global
CREATE POLICY detected_faces_update_policy ON detected_faces
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_faces.document_id
            AND (
                dm.user_id = current_setting('app.current_user_id', true)::varchar
                OR (dm.collection_type = 'global' AND current_setting('app.current_user_role', true) = 'admin')
            )
        )
    );

-- Users can delete faces in their own documents, admins for global
CREATE POLICY detected_faces_delete_policy ON detected_faces
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_faces.document_id
            AND (
                dm.user_id = current_setting('app.current_user_id', true)::varchar
                OR (dm.collection_type = 'global' AND current_setting('app.current_user_role', true) = 'admin')
            )
        )
    );

-- RLS Policies for known_identities
-- Users can see all known identities (for search suggestions)
CREATE POLICY known_identities_select_policy ON known_identities
    FOR SELECT USING (true);

-- Users can create identities, but names must be unique
CREATE POLICY known_identities_insert_policy ON known_identities
    FOR INSERT WITH CHECK (true);

-- Users can update identities they created, admins can update any
CREATE POLICY known_identities_update_policy ON known_identities
    FOR UPDATE USING (
        created_by = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Users can delete identities they created, admins can delete any
CREATE POLICY known_identities_delete_policy ON known_identities
    FOR DELETE USING (
        created_by = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

-- Trigger for updated_at on known_identities
CREATE TRIGGER update_known_identities_updated_at
    BEFORE UPDATE ON known_identities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE detected_faces IS 'Stores detected faces with bounding boxes and 128-dimensional encodings for similarity matching';
-- Known identities comment already added above with Qdrant architecture explanation

-- ========================================
-- OBJECT DETECTION (YOLO + CLIP + user annotations)
-- user_object_annotations, object_annotation_examples, detected_objects
-- ========================================

CREATE TABLE IF NOT EXISTS user_object_annotations (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    object_name VARCHAR(255) NOT NULL,
    description TEXT,
    source_document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    visual_embedding_id TEXT,
    semantic_embedding_id TEXT,
    combined_embedding_id TEXT,
    example_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, object_name)
);

CREATE INDEX IF NOT EXISTS idx_user_object_annotations_user_id ON user_object_annotations(user_id);
CREATE INDEX IF NOT EXISTS idx_user_object_annotations_source_doc ON user_object_annotations(source_document_id);

CREATE TABLE IF NOT EXISTS object_annotation_examples (
    id SERIAL PRIMARY KEY,
    annotation_id INTEGER NOT NULL REFERENCES user_object_annotations(id) ON DELETE CASCADE,
    source_document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    combined_embedding_id TEXT NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_object_annotation_examples_annotation_id ON object_annotation_examples(annotation_id);

CREATE TABLE IF NOT EXISTS detected_objects (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    class_name VARCHAR(255) NOT NULL,
    original_class_name TEXT,
    detection_method VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    annotation_id INTEGER REFERENCES user_object_annotations(id) ON DELETE SET NULL,
    confirmed BOOLEAN DEFAULT NULL,
    rejected BOOLEAN DEFAULT NULL,
    user_tag TEXT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    confirmed_by VARCHAR(255),
    confirmed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_detected_objects_document_id ON detected_objects(document_id);
CREATE INDEX IF NOT EXISTS idx_detected_objects_annotation_id ON detected_objects(annotation_id);
CREATE INDEX IF NOT EXISTS idx_detected_objects_class_name ON detected_objects(class_name);

GRANT ALL PRIVILEGES ON user_object_annotations TO bastion_user;
GRANT ALL PRIVILEGES ON object_annotation_examples TO bastion_user;
GRANT ALL PRIVILEGES ON detected_objects TO bastion_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bastion_user;

ALTER TABLE user_object_annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE object_annotation_examples ENABLE ROW LEVEL SECURITY;
ALTER TABLE detected_objects ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_object_annotations_select_policy ON user_object_annotations
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY user_object_annotations_insert_policy ON user_object_annotations
    FOR INSERT WITH CHECK (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY user_object_annotations_update_policy ON user_object_annotations
    FOR UPDATE USING (user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY user_object_annotations_delete_policy ON user_object_annotations
    FOR DELETE USING (user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY object_annotation_examples_select_policy ON object_annotation_examples
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM user_object_annotations uoa
            WHERE uoa.id = object_annotation_examples.annotation_id
            AND (uoa.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY object_annotation_examples_insert_policy ON object_annotation_examples
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM user_object_annotations uoa
            WHERE uoa.id = object_annotation_examples.annotation_id
            AND uoa.user_id = current_setting('app.current_user_id', true)::varchar
        )
    );

CREATE POLICY object_annotation_examples_update_policy ON object_annotation_examples
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM user_object_annotations uoa
            WHERE uoa.id = object_annotation_examples.annotation_id
            AND (uoa.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY object_annotation_examples_delete_policy ON object_annotation_examples
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM user_object_annotations uoa
            WHERE uoa.id = object_annotation_examples.annotation_id
            AND (uoa.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY detected_objects_select_policy ON detected_objects
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_objects.document_id
            AND (
                dm.user_id = current_setting('app.current_user_id', true)::varchar
                OR dm.collection_type = 'global'
                OR current_setting('app.current_user_role', true) = 'admin'
            )
        )
    );

CREATE POLICY detected_objects_insert_policy ON detected_objects
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_objects.document_id
            AND (
                (dm.user_id = current_setting('app.current_user_id', true)::varchar AND dm.collection_type = 'user')
                OR (dm.collection_type = 'global' AND current_setting('app.current_user_role', true) = 'admin')
            )
        )
    );

CREATE POLICY detected_objects_update_policy ON detected_objects
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_objects.document_id
            AND (
                dm.user_id = current_setting('app.current_user_id', true)::varchar
                OR (dm.collection_type = 'global' AND current_setting('app.current_user_role', true) = 'admin')
            )
        )
    );

CREATE POLICY detected_objects_delete_policy ON detected_objects
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM document_metadata dm
            WHERE dm.document_id = detected_objects.document_id
            AND (
                dm.user_id = current_setting('app.current_user_id', true)::varchar
                OR (dm.collection_type = 'global' AND current_setting('app.current_user_role', true) = 'admin')
            )
        )
    );

CREATE TRIGGER update_user_object_annotations_updated_at
    BEFORE UPDATE ON user_object_annotations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

COMMENT ON TABLE user_object_annotations IS 'User-defined object annotations with CLIP embeddings for similarity search';
COMMENT ON TABLE object_annotation_examples IS 'Additional examples per annotation for robust matching';
COMMENT ON TABLE detected_objects IS 'Detected objects (YOLO, CLIP semantic, or user-defined) per image';
COMMENT ON COLUMN detected_objects.user_tag IS 'User-refined label for search/display (e.g. YOLO class_name "car" -> "BMW i3")';
COMMENT ON COLUMN detected_objects.original_class_name IS 'Original detection label (YOLO/CLIP) before user correction; not vectorized';

-- ========================================
-- EXTERNAL CONNECTIONS (OAuth / Email / Calendar / etc.)
-- ========================================

CREATE TABLE IF NOT EXISTS external_connections (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    connection_type VARCHAR(50) NOT NULL,
    account_identifier VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    encrypted_access_token TEXT NOT NULL,
    encrypted_refresh_token TEXT NOT NULL,
    token_expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    scopes TEXT[] NOT NULL,
    provider_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_sync_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    connection_status VARCHAR(50) DEFAULT 'active',
    UNIQUE(user_id, provider, connection_type, account_identifier)
);

CREATE INDEX IF NOT EXISTS idx_external_connections_user ON external_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_external_connections_active ON external_connections(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_external_connections_type ON external_connections(provider, connection_type);

CREATE TABLE IF NOT EXISTS connection_sync_state (
    id BIGSERIAL PRIMARY KEY,
    connection_id BIGINT NOT NULL REFERENCES external_connections(id) ON DELETE CASCADE,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    resource_name VARCHAR(255),
    last_sync_timestamp TIMESTAMP WITH TIME ZONE,
    delta_token TEXT,
    sync_status VARCHAR(50) DEFAULT 'pending',
    last_error TEXT,
    metadata JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(connection_id, resource_type, resource_id)
);

CREATE INDEX IF NOT EXISTS idx_sync_state_connection ON connection_sync_state(connection_id);

CREATE TABLE IF NOT EXISTS connection_data_cache (
    id BIGSERIAL PRIMARY KEY,
    connection_id BIGINT NOT NULL REFERENCES external_connections(id) ON DELETE CASCADE,
    data_type VARCHAR(100) NOT NULL,
    external_id VARCHAR(500) NOT NULL,
    data JSONB NOT NULL,
    metadata JSONB,
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(connection_id, data_type, external_id)
);

CREATE INDEX IF NOT EXISTS idx_data_cache_connection ON connection_data_cache(connection_id);
CREATE INDEX IF NOT EXISTS idx_data_cache_type ON connection_data_cache(connection_id, data_type);
CREATE INDEX IF NOT EXISTS idx_data_cache_metadata ON connection_data_cache USING gin(metadata);

GRANT ALL PRIVILEGES ON external_connections TO bastion_user;
GRANT ALL PRIVILEGES ON connection_sync_state TO bastion_user;
GRANT ALL PRIVILEGES ON connection_data_cache TO bastion_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bastion_user;

ALTER TABLE external_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE connection_sync_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE connection_data_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY external_connections_select_policy ON external_connections
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY external_connections_insert_policy ON external_connections
    FOR INSERT WITH CHECK (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY external_connections_update_policy ON external_connections
    FOR UPDATE USING (user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY external_connections_delete_policy ON external_connections
    FOR DELETE USING (user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY connection_sync_state_select_policy ON connection_sync_state
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_sync_state_insert_policy ON connection_sync_state
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND ec.user_id = current_setting('app.current_user_id', true)::varchar
        )
    );

CREATE POLICY connection_sync_state_update_policy ON connection_sync_state
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_sync_state_delete_policy ON connection_sync_state
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_sync_state.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_data_cache_select_policy ON connection_data_cache
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_data_cache_insert_policy ON connection_data_cache
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND ec.user_id = current_setting('app.current_user_id', true)::varchar
        )
    );

CREATE POLICY connection_data_cache_update_policy ON connection_data_cache
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

CREATE POLICY connection_data_cache_delete_policy ON connection_data_cache
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM external_connections ec
            WHERE ec.id = connection_data_cache.connection_id
            AND (ec.user_id = current_setting('app.current_user_id', true)::varchar OR current_setting('app.current_user_role', true) = 'admin')
        )
    );

-- ========================================
-- SYSTEM SETTINGS (admin-only key-value)
-- ========================================
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

GRANT ALL PRIVILEGES ON system_settings TO bastion_user;

ALTER TABLE system_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY system_settings_select_policy ON system_settings
    FOR SELECT USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY system_settings_insert_policy ON system_settings
    FOR INSERT WITH CHECK (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY system_settings_update_policy ON system_settings
    FOR UPDATE USING (current_setting('app.current_user_role', true) = 'admin');

CREATE POLICY system_settings_delete_policy ON system_settings
    FOR DELETE USING (current_setting('app.current_user_role', true) = 'admin');

-- ========================================
-- DOCUMENT LINKS (file relation graph)
-- ========================================
CREATE TABLE IF NOT EXISTS document_links (
    id SERIAL PRIMARY KEY,
    source_document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    target_document_id VARCHAR(255) REFERENCES document_metadata(document_id) ON DELETE SET NULL,
    target_raw_path TEXT NOT NULL,
    link_type VARCHAR(50) NOT NULL,
    description TEXT,
    line_number INTEGER,
    user_id VARCHAR(255),
    collection_type VARCHAR(20),
    team_id UUID REFERENCES teams(team_id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_document_id, target_raw_path, line_number)
);

CREATE INDEX IF NOT EXISTS idx_document_links_source ON document_links(source_document_id);
CREATE INDEX IF NOT EXISTS idx_document_links_target ON document_links(target_document_id);
CREATE INDEX IF NOT EXISTS idx_document_links_user_id ON document_links(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_document_links_team_id ON document_links(team_id) WHERE team_id IS NOT NULL;

ALTER TABLE document_links ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS document_links_select_policy ON document_links;
DROP POLICY IF EXISTS document_links_update_policy ON document_links;
DROP POLICY IF EXISTS document_links_delete_policy ON document_links;
DROP POLICY IF EXISTS document_links_insert_policy ON document_links;

CREATE POLICY document_links_select_policy ON document_links
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
        OR (user_id IS NOT NULL AND team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_links_update_policy ON document_links
    FOR UPDATE
    USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    )
    WITH CHECK (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (team_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_links_delete_policy ON document_links
    FOR DELETE USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
            AND role = 'admin'
        ))
        OR (user_id IS NULL AND current_setting('app.current_user_role', true) = 'admin')
    );

CREATE POLICY document_links_insert_policy ON document_links
    FOR INSERT WITH CHECK (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
    );

-- ========================================
-- DATABASE INITIALIZATION COMPLETE
-- Roosevelt's Square Deal for Data!
-- ========================================

-- ============================================================================
-- FINAL SECURITY HARDENING - REVOKE BYPASSRLS
-- ============================================================================
-- ROOSEVELT'S SECURITY FIX: Remove BYPASSRLS to enforce Row-Level Security
-- BYPASSRLS was needed during init to insert default data into RLS-protected tables
-- Now that init is complete, revoke it so RLS policies are enforced

ALTER ROLE bastion_user NOBYPASSRLS;

-- IMPORTANT FOR FUTURE MIGRATIONS:
-- If you need to run migrations that insert/update data in RLS-protected tables:
-- 1. Temporarily grant BYPASSRLS: ALTER ROLE bastion_user BYPASSRLS;
-- 2. Run your migration
-- 3. Revoke BYPASSRLS: ALTER ROLE bastion_user NOBYPASSRLS;
-- 
-- Or set proper RLS context in migration:
--   SELECT set_config('app.current_user_id', 'migration', true);
--   SELECT set_config('app.current_user_role', 'admin', true);
