-- ========================================
-- ADD OBJECT DETECTION TABLES
-- User-defined object annotations and detected objects (YOLO + CLIP + user annotations)
-- ========================================
--
-- Run this migration on an existing database (e.g. after deploying object detection):
--
--   docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/sql/migrations/036_add_object_detection.sql
--
-- Or from host with psql:
--   psql -U postgres -d bastion_knowledge_base -f backend/sql/migrations/036_add_object_detection.sql
--

-- User-defined object annotations (like known_identities for faces)
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

-- Additional examples of same object
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

-- Detected objects in images (like detected_faces)
CREATE TABLE IF NOT EXISTS detected_objects (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
    class_name VARCHAR(255) NOT NULL,
    detection_method VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    annotation_id INTEGER REFERENCES user_object_annotations(id) ON DELETE SET NULL,
    confirmed BOOLEAN DEFAULT NULL,
    rejected BOOLEAN DEFAULT NULL,
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

-- RLS: user_object_annotations - users see/edit their own
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

-- RLS: object_annotation_examples - same as annotation owner
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

-- RLS: detected_objects - same as detected_faces (document access)
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
