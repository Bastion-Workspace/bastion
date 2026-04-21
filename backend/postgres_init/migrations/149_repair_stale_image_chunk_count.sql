-- Repair stale chunk_count on image documents that have no document_chunks rows.
-- Main image pipeline does not embed pixels; chunk_count > 0 without document_chunks
-- usually means a bad row and causes admin audit-and-reembed to queue reprocess forever.
-- Safe to re-run (idempotent). Apply manually if needed:
--   docker exec -i bastion-postgres psql -U postgres -d bastion_knowledge_base < backend/postgres_init/migrations/149_repair_stale_image_chunk_count.sql

UPDATE document_metadata dm
SET chunk_count = 0,
    updated_at = CURRENT_TIMESTAMP
WHERE LOWER(COALESCE(doc_type::text, '')) = 'image'
  AND COALESCE(dm.chunk_count, 0) > 0
  AND NOT EXISTS (
    SELECT 1 FROM document_chunks dc WHERE dc.document_id = dm.document_id
  );
