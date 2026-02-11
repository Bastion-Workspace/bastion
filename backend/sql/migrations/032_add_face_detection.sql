-- ========================================
-- ADD FACE DETECTION TABLES
-- Create tables for storing detected faces and known identities
-- ========================================
-- This migration creates tables for face detection and tagging:
-- - detected_faces: Stores detected faces with bounding boxes and encodings
-- - known_identities: Stores known person identities metadata (encodings in Qdrant)
--
-- Usage:
-- docker exec -i <postgres-container> psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/032_add_face_detection.sql
-- Or from within container:
-- psql -U postgres -d bastion_knowledge_base -f /docker-entrypoint-initdb.d/migrations/032_add_face_detection.sql
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
-- NOTE: Face encodings stored in Qdrant collection "face_encodings"
-- Multiple encodings per identity for robust matching
CREATE TABLE IF NOT EXISTS known_identities (
    id SERIAL PRIMARY KEY,
    identity_name VARCHAR(255) UNIQUE NOT NULL,
    face_encoding FLOAT[],  -- DEPRECATED: Use Qdrant instead
    sample_count INTEGER DEFAULT 1,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_known_identities_name ON known_identities(identity_name);

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
COMMENT ON TABLE known_identities IS 'Stores known person identities with averaged face encodings for quick lookup and matching';
