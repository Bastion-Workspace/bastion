-- Migration 033: Move face encodings to Qdrant vector database
-- Makes face_encoding column nullable in known_identities since we store in Qdrant now

-- Make face_encoding nullable (we store in Qdrant for vector similarity search)
ALTER TABLE known_identities 
ALTER COLUMN face_encoding DROP NOT NULL;

-- Add index on identity_name for faster lookups
CREATE INDEX IF NOT EXISTS idx_known_identities_name ON known_identities(identity_name);

-- Add comment explaining architecture
COMMENT ON COLUMN known_identities.face_encoding IS 'DEPRECATED: Face encodings now stored in Qdrant for vector similarity search. This column kept for backward compatibility but should be NULL for new identities.';
COMMENT ON TABLE known_identities IS 'Metadata for known face identities. Face encoding vectors stored in Qdrant collection "face_encodings" for efficient similarity search.';
