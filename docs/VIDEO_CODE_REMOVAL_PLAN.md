# Video Code Removal Plan

**Date:** November 10, 2025  
**Status:** Ready for Execution  
**Prerequisite for:** Vector Service Migration

## Rationale

Video embedding capabilities are being removed to:
1. Simplify the Vector Service migration
2. Reduce codebase complexity
3. Can be re-added post-refactor if needed

## Files to Remove

### Backend Services (7 files)

```
backend/services/
├── video_service.py                    # Main video processing service
├── video_segment_service.py            # Video segment management
├── video_extraction_service.py         # Video extraction logic
└── video_search_service.py             # Video semantic search
```

### Backend API (1 file)

```
backend/api/
└── video_api.py                        # Video API endpoints
```

### Backend Models (1 file)

```
backend/models/
└── video_models.py                     # Video data models
```

### Backend Repositories (1 file)

```
backend/repositories/
└── video_repository.py                 # Video database access
```

**Total: 7 files to delete**

## Files to Modify

### 1. `backend/main.py`

**Remove:**
- Video API router registration
- Video-related imports

**Lines to check:**
- Search for `video_api` imports
- Search for `video` router registration
- Remove from `app.include_router()` calls

### 2. `backend/sql/01_init.sql`

**Remove:**
- Video-related database tables:
  - `videos` table
  - `video_segments` table
  - `video_transcripts` table
  - Video-related indexes
  - Video-related foreign keys

### 3. `backend/Dockerfile`

**Check for:**
- Video-specific dependencies (e.g., ffmpeg, video codecs)
- Remove if present

### 4. `backend/services/langgraph_tools/unified_search_tools.py`

**Remove:**
- Video search integration
- `_search_video()` method (if exists)
- Video-related imports

### 5. `backend/services/langgraph_agents/rss_background_agent.py`

**Remove:**
- Video processing logic (if exists)
- Video-related imports

### 6. `backend/utils/embedding_manager.py`

**Remove:**
- Video collection management
- Video-specific embedding methods
- References to `user_{user_id}_videos` collections
- References to `global_videos` collection

### 7. `backend/utils/parallel_embedding_manager.py`

**Remove:**
- Video embedding support (if exists)

## Database Migration

### SQL to Run (if database has video tables)

```sql
-- Drop video-related tables (if they exist)
DROP TABLE IF EXISTS video_segments CASCADE;
DROP TABLE IF EXISTS video_transcripts CASCADE;
DROP TABLE IF EXISTS videos CASCADE;

-- Drop video-related indexes
DROP INDEX IF EXISTS idx_videos_user_id;
DROP INDEX IF EXISTS idx_video_segments_video_id;
```

**Note:** Only run if these tables were created. Check database first.

## Qdrant Collection Cleanup

### Collections to Delete

```python
# Check for and delete video collections
collections_to_check = [
    "global_videos",
    # User-specific: "user_{user_id}_videos" for all users
]
```

**Action:** Create a cleanup script or manually delete from Qdrant UI.

## Execution Steps

### Step 1: Backup (Safety First)

```bash
# Commit current state
git add -A
git commit -m "Pre-video-removal backup"

# Create backup branch
git checkout -b backup/pre-video-removal
git checkout main
```

### Step 2: Delete Files

```bash
# Delete video-related files
rm backend/services/video_service.py
rm backend/services/video_segment_service.py
rm backend/services/video_extraction_service.py
rm backend/services/video_search_service.py
rm backend/api/video_api.py
rm backend/models/video_models.py
rm backend/repositories/video_repository.py
```

### Step 3: Modify main.py

Remove video router registration:

```python
# Remove these lines (if present):
from api import video_api
app.include_router(video_api.router, tags=["videos"])
```

### Step 4: Modify unified_search_tools.py

Remove video search integration (if present).

### Step 5: Modify embedding_manager.py

Remove video collection references:
- Search for `videos` in collection names
- Remove `_ensure_video_collection_exists()` (if exists)
- Remove video-specific methods

### Step 6: Update SQL Schema

Edit `backend/sql/01_init.sql`:
- Remove video table definitions
- Remove video-related indexes

### Step 7: Clean Imports

```bash
# Search for orphaned video imports
grep -r "from.*video" backend/
grep -r "import.*video" backend/
```

Remove any remaining video imports.

### Step 8: Test

```bash
# Rebuild and test
docker compose build backend
docker compose up backend

# Check for import errors
docker compose logs backend | grep -i "importerror\|modulenotfounderror"
```

### Step 9: Verify

**Check that backend starts successfully:**
- No import errors
- All non-video endpoints work
- No references to video code

## Rollback Plan

If something breaks:

```bash
# Restore from backup
git checkout backup/pre-video-removal -- backend/

# Rebuild
docker compose build backend
docker compose up backend
```

## Expected Impact

### ✅ No Impact On

- Document processing
- Document search
- Chat service
- LLM orchestrator
- User authentication
- All other services

### ⚠️ Breaks

- Video upload endpoints (if anyone uses them)
- Video search endpoints (if anyone uses them)
- Video-related UI features (if they exist in frontend)

### 📊 Benefits

- Reduced backend code (~2000+ lines)
- Simpler embedding manager
- Cleaner service container
- Faster Docker builds (if video dependencies removed)
- Reduced memory footprint

## Verification Checklist

After removal, verify:

- [ ] Backend starts successfully
- [ ] No import errors in logs
- [ ] Document upload works
- [ ] Document search works
- [ ] Chat works
- [ ] LLM orchestrator works
- [ ] No references to "video" in remaining code (except comments)
- [ ] `docker compose ps` shows all services healthy

## Frontend Consideration

**Check if frontend has video UI:**

```bash
# Search frontend for video references
grep -r "video" frontend/src/
```

If video UI exists:
- Remove video-related components
- Remove video upload UI
- Remove video search UI
- Update navigation/routing

## Timeline

**Estimated time:** 1-2 hours

1. Backup: 5 minutes
2. Delete files: 2 minutes
3. Modify files: 30 minutes
4. Test: 15 minutes
5. Verify: 10 minutes
6. Cleanup: 10 minutes

## Ready to Execute

This plan is **ready for execution**. All files are identified, steps are clear, and rollback is simple.

**Recommend:** Execute this cleanup before starting Vector Service Phase 1 implementation.

---

**Prepared by:** Claude (AI Assistant)  
**Status:** ✅ Ready for Execution  
**Estimated Impact:** Low (isolated code removal)

