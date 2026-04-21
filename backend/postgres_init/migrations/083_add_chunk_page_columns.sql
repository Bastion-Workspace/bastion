-- Page-aware chunks and metadata FTS: page_start/page_end on document_chunks,
-- meta_tsv tsvector on document_metadata for title+tags full-text search.
-- Idempotent (IF NOT EXISTS). meta_tsv maintained by trigger (array_to_string
-- is STABLE so cannot be used in a GENERATED column).

-- document_chunks: page range columns for PDF (and other page-aware) chunks
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_start INT;
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_end INT;
CREATE INDEX IF NOT EXISTS idx_document_chunks_page
    ON document_chunks(document_id, page_start);

-- document_metadata: tsvector for title + tags (FTS). Plain column + trigger
-- because GENERATED requires immutable expression and array_to_string is STABLE.
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS meta_tsv TSVECTOR;

CREATE OR REPLACE FUNCTION document_metadata_meta_tsv_trigger_fn()
RETURNS TRIGGER AS $$
BEGIN
    NEW.meta_tsv := to_tsvector('english',
        COALESCE(NEW.title, '') || ' ' ||
        COALESCE(array_to_string(NEW.tags, ' '), '')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS document_metadata_meta_tsv_trigger ON document_metadata;
CREATE TRIGGER document_metadata_meta_tsv_trigger
    BEFORE INSERT OR UPDATE OF title, tags ON document_metadata
    FOR EACH ROW EXECUTE FUNCTION document_metadata_meta_tsv_trigger_fn();

-- Backfill existing rows
UPDATE document_metadata
SET meta_tsv = to_tsvector('english',
    COALESCE(title, '') || ' ' || COALESCE(array_to_string(tags, ' '), '')
)
WHERE meta_tsv IS NULL;

CREATE INDEX IF NOT EXISTS idx_document_metadata_meta_tsv
    ON document_metadata USING GIN(meta_tsv);
