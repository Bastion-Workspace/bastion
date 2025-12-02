-- ========================================
-- EMAIL AGENT SYSTEM MIGRATION
-- Email verification and sending infrastructure
-- ========================================
-- This migration can be run on existing databases without data loss
-- Idempotent design - safe to run multiple times
--
-- Usage:
-- docker exec -i <postgres-container> psql -U bastion_user -d bastion_knowledge_base < backend/sql/migrations/007_add_email_agent_tables.sql
-- Or from within container:
-- psql -U bastion_user -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/007_add_email_agent_tables.sql
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

