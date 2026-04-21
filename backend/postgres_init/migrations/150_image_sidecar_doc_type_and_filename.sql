-- Unify image metadata sidecar rows: doc_type image_sidecar, filename = <stem>.metadata.json
-- Skip rows that would collide with an existing document in the same folder/user/collection.

UPDATE document_metadata dm
SET doc_type = 'image_sidecar',
    filename = regexp_replace(dm.filename, '\.[^.]+$', '') || '.metadata.json',
    updated_at = NOW()
WHERE dm.metadata_json->>'has_searchable_metadata' = 'true'
  AND LOWER(COALESCE(dm.doc_type::text, '')) <> 'image_sidecar'
  AND NOT EXISTS (
    SELECT 1
    FROM document_metadata o
    WHERE o.document_id <> dm.document_id
      AND o.folder_id IS NOT DISTINCT FROM dm.folder_id
      AND o.user_id IS NOT DISTINCT FROM dm.user_id
      AND o.collection_type = dm.collection_type
      AND o.filename = (regexp_replace(dm.filename, '\.[^.]+$', '') || '.metadata.json')
  );

-- Force re-embed through unified pipeline on next backfill or reprocess
UPDATE document_metadata
SET chunk_indexed_at = NULL,
    chunk_indexed_file_hash = NULL,
    chunk_index_schema_version = 0,
    updated_at = NOW()
WHERE LOWER(COALESCE(doc_type::text, '')) = 'image_sidecar';
