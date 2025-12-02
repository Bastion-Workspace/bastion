# Document Library Security Fixes Applied

## Critical Security Vulnerabilities Fixed

### 1. Authorization on Document Endpoints ✅ COMPLETED

**Added `check_document_access()` helper function** that enforces:
- **Global documents**: Anyone can read, only admins can write/delete
- **Team documents**: Team members can read, team admins can write/delete
- **User documents**: Only owner can access, admins have full access

**Protected endpoints:**
- `GET /api/documents/{doc_id}/pdf` - ✅ Added auth + access check
- `GET /api/documents/{doc_id}/content` - ✅ Added auth + access check  
- `DELETE /api/documents/{doc_id}` - ✅ Added auth + access check
- `PUT /api/documents/{doc_id}/content` - ✅ Added auth + access check

### 2. Path Traversal Protection ✅ COMPLETED

**Added security checks to prevent path traversal attacks:**
- Sanitize filenames with `os.path.basename()` to remove path components
- Validate resolved paths stay within `UPLOAD_DIR`
- Reject invalid filenames (`.`, `..`, empty)
- Log all path traversal attempts

### 3. Row-Level Security (RLS) ⚠️ MANUAL STEP REQUIRED

**Created `backend/sql/02_enable_document_rls.sql`** with:
- RLS enabled on `document_metadata` table
- SELECT policy: Users see their own docs, global docs, and team docs
- UPDATE policy: Users update their own docs, team admins update team docs
- DELETE policy: Users delete their own docs, team admins delete team docs  
- INSERT policy: Allows inserts (ownership enforced at app layer)

**To apply RLS policies, run:**
```bash
# Start containers if not running
docker compose up -d

# Apply RLS (choose the correct postgres container name from docker ps)
docker exec -i bastion-postgres psql -U postgres -d codex < backend/sql/02_enable_document_rls.sql
```

### 4. Real Ownership Checking ✅ COMPLETED

**Implemented `_document_belongs_to_user()` in UserDocumentService:**
- Checks if user owns the document
- Checks if document is global (accessible to all)
- Checks if user is a member of the document's team
- Returns False for all other cases

## Access Control Matrix

| Collection Type | Owner | Team Member | Team Admin | Other Users | Admin |
|----------------|-------|-------------|------------|-------------|-------|
| **User Documents** | Full | ❌ | ❌ | ❌ | Full |
| **Team Documents** | N/A | Read | Full | ❌ | Full |
| **Global Documents** | N/A | Read | Read | Read | Full |

## Security Testing Checklist

After applying RLS policies, test:

1. **User isolation:**
   - User A uploads document
   - User B tries to access User A's document → Should get 403
   
2. **Team access:**
   - User A (team member) accesses team document → Should succeed
   - User B (not member) tries to access team document → Should get 403
   
3. **Global access:**
   - Any authenticated user can read global documents → Should succeed
   - Non-admin tries to delete global document → Should get 403
   
4. **Path traversal:**
   - Try accessing: `/api/documents/{doc_id}/pdf` with malicious filename in DB
   - Should get 403 Access Denied

5. **Admin privileges:**
   - Admin can access/modify any document → Should succeed

## Files Modified

- `backend/main.py` - Added auth checks and `check_document_access()` function
- `backend/services/user_document_service.py` - Fixed `_document_belongs_to_user()`
- `backend/sql/02_enable_document_rls.sql` - RLS policies (needs manual application)

## Impact

**Before fixes:**
- ❌ ANY user could read/delete ANY document
- ❌ NO path traversal protection  
- ❌ RLS completely disabled
- ❌ Stub functions returned `True` always

**After fixes:**
- ✅ Proper authorization on all endpoints
- ✅ Path traversal protection
- ✅ RLS policies created (needs DB application)
- ✅ Real ownership checking implemented

