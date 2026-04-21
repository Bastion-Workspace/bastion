-- Conversation message attachments (chat sidebar). Distinct from message_attachments (room messaging).
CREATE TABLE IF NOT EXISTS conversation_message_attachments (
    attachment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id VARCHAR(255) NOT NULL REFERENCES conversation_messages(message_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TIMESTAMP NOT NULL DEFAULT NOW(),
    is_image BOOLEAN DEFAULT false,
    image_width INTEGER,
    image_height INTEGER,
    vision_description TEXT,
    detected_identities TEXT[],
    face_detection_metadata JSONB,
    metadata_json JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT positive_file_size CHECK (file_size > 0)
);

CREATE INDEX IF NOT EXISTS idx_conversation_message_attachments_message_id ON conversation_message_attachments(message_id);
CREATE INDEX IF NOT EXISTS idx_conversation_message_attachments_uploaded_at ON conversation_message_attachments(uploaded_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversation_message_attachments_content_type ON conversation_message_attachments(content_type);
CREATE INDEX IF NOT EXISTS idx_conversation_message_attachments_detected_identities ON conversation_message_attachments USING GIN(detected_identities);
