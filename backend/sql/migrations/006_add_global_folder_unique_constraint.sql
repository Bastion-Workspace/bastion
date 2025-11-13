-- Migration 006: Add unique constraint for global non-root folders
-- **ROOSEVELT'S DUPLICATE FOLDER FIX!** 🏇
--
-- Problem: Global folders were being created twice (once by API, once by file watcher)
-- because the unique constraint didn't properly handle global non-root folders
-- where user_id IS NULL but parent_folder_id IS NOT NULL
--
-- Solution: Add separate unique index for global non-root folders

-- Drop the old non-root folder index that doesn't handle global folders properly
DROP INDEX IF EXISTS idx_document_folders_unique_with_parent;

-- Recreate it for USER non-root folders only
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_with_parent 
ON document_folders(user_id, name, parent_folder_id, collection_type)
WHERE parent_folder_id IS NOT NULL AND user_id IS NOT NULL;

-- Add new unique index for GLOBAL non-root folders
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_folders_unique_global_with_parent
ON document_folders(name, parent_folder_id, collection_type)
WHERE parent_folder_id IS NOT NULL AND user_id IS NULL;

-- By George! Now both the API and file watcher will recognize existing folders!
-- The UPSERT will work properly and return the existing folder_id instead of creating duplicates!

