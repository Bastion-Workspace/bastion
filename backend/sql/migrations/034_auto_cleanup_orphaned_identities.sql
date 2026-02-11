-- Migration 034: Auto-cleanup orphaned face identities
-- When all faces for an identity are deleted, remove the identity

-- Function to check and clean up identities with no remaining faces
CREATE OR REPLACE FUNCTION cleanup_orphaned_identities()
RETURNS TRIGGER AS $$
DECLARE
    identity_to_check VARCHAR(255);
    remaining_count INTEGER;
BEGIN
    -- Get the identity_name from the deleted face (if it was tagged)
    identity_to_check := OLD.identity_name;
    
    -- Only proceed if the face was actually tagged
    IF identity_to_check IS NOT NULL THEN
        -- Count remaining faces with this identity
        SELECT COUNT(*) INTO remaining_count
        FROM detected_faces
        WHERE identity_name = identity_to_check;
        
        -- If no more faces exist for this identity, delete it from PostgreSQL
        IF remaining_count = 0 THEN
            DELETE FROM known_identities WHERE identity_name = identity_to_check;
            RAISE NOTICE 'Auto-deleted orphaned identity: % (no remaining faces)', identity_to_check;
            -- Note: Qdrant vectors cleaned up separately via /api/vision/cleanup-orphaned-vectors
        END IF;
    END IF;
    
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to run after face deletion
DROP TRIGGER IF EXISTS trigger_cleanup_orphaned_identities ON detected_faces;
CREATE TRIGGER trigger_cleanup_orphaned_identities
    AFTER DELETE ON detected_faces
    FOR EACH ROW
    EXECUTE FUNCTION cleanup_orphaned_identities();

COMMENT ON FUNCTION cleanup_orphaned_identities() IS 'Automatically removes known_identities from PostgreSQL when all detected_faces for that identity are deleted. Qdrant vector cleanup handled by backend API endpoint /api/vision/cleanup-orphaned-vectors.';
