-- Migration 141: Per-document password-based file encryption metadata

ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS is_encrypted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS encryption_version INTEGER;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS encryption_salt BYTEA;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS password_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_document_metadata_encrypted
  ON document_metadata(is_encrypted) WHERE is_encrypted = TRUE;

COMMENT ON COLUMN document_metadata.is_encrypted IS 'When true, file on disk is AES-256-GCM ciphertext; content APIs require an active decrypt session';
COMMENT ON COLUMN document_metadata.encryption_version IS 'Key derivation / format version for future rotation';
COMMENT ON COLUMN document_metadata.encryption_salt IS 'Random salt for Argon2id key derivation (per document)';
COMMENT ON COLUMN document_metadata.password_hash IS 'Argon2id hash of document password for verification only';
