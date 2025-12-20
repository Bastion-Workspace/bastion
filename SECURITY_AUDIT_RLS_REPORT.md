# Database Row-Level Security (RLS) Audit Report

**Date:** December 18, 2025  
**Auditor:** AI Security Review  
**Scope:** PostgreSQL Row-Level Security policies and user data isolation

---

## Executive Summary

This audit reviewed the Row-Level Security (RLS) implementation across the Bastion platform to ensure users cannot access data belonging to other users. The audit covered:

1. Backend database (`bastion_knowledge_base`)
2. Data-service database (`data_workspace`)
3. Application-level access controls
4. Team-based sharing mechanisms

**Overall Risk Level:** 🔴 **HIGH** - Critical security gaps identified

---

## Critical Security Issues Found

### 🔴 CRITICAL #1: Conversations and Messages Have RLS Disabled

**Location:** `/opt/bastion/backend/sql/01_init.sql` lines 734-736

**Issue:**
```sql
-- ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
-- ROOSEVELT'S TEMPORARY DIAGNOSTIC: Disable RLS for conversation_messages to test the issue
-- ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
```

**Impact:** 
- **ANY USER CAN ACCESS ANY OTHER USER'S CONVERSATIONS**
- **ANY USER CAN READ ANY OTHER USER'S MESSAGES**
- RLS policies are defined but NOT ENFORCED because RLS is disabled on these tables

**Risk:** CRITICAL - Complete exposure of private conversation data

**Recommendation:** 
```sql
-- IMMEDIATELY ENABLE RLS:
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
```

**Note:** The policies exist (lines 742-793) but are ineffective without enabling RLS on the tables.

---

### 🔴 CRITICAL #2: Document Folders Have RLS Disabled

**Location:** `/opt/bastion/backend/sql/01_init.sql` line 479

**Issue:**
```sql
-- ALTER TABLE document_folders ENABLE ROW LEVEL SECURITY;
```

**Impact:**
- Users can potentially access folder structures of other users
- Folder-based document organization security is bypassed
- Application-level checks are the only protection

**Risk:** HIGH - Folder metadata exposure, potential path traversal

**Recommendation:**
```sql
-- ENABLE RLS:
ALTER TABLE document_folders ENABLE ROW LEVEL SECURITY;

-- ADD POLICIES:
CREATE POLICY document_folders_select_policy ON document_folders
    FOR SELECT USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR team_id IN (SELECT team_id FROM team_members WHERE user_id = current_setting('app.current_user_id', true)::varchar)
        OR current_setting('app.current_user_role', true) = 'admin'
    );

CREATE POLICY document_folders_modify_policy ON document_folders
    FOR ALL USING (
        user_id = current_setting('app.current_user_id', true)::varchar
        OR current_setting('app.current_user_role', true) = 'admin'
    );
```

---

### 🟡 MEDIUM #3: Database User Has BYPASSRLS Privilege

**Location:** `/opt/bastion/backend/sql/01_init.sql` line 20

**Issue:**
```sql
ALTER ROLE bastion_user BYPASSRLS;
```

**Impact:**
- The application database user bypasses ALL RLS policies
- RLS policies are effectively disabled for all application queries
- This defeats the entire purpose of RLS

**Risk:** CRITICAL - RLS is completely ineffective

**Current State:** This is likely why RLS was disabled on conversations - it wasn't working because the user bypasses it!

**Recommendation:**
```sql
-- REMOVE BYPASSRLS:
ALTER ROLE bastion_user NOBYPASSRLS;

-- For migrations/setup, use a separate admin role:
-- CREATE ROLE bastion_admin WITH BYPASSRLS;
-- Use bastion_admin only for schema migrations
-- Use bastion_user (without BYPASSRLS) for application queries
```

**CRITICAL NOTE:** This is the root cause. With BYPASSRLS, all RLS policies are ignored. The application must use a role WITHOUT this privilege for RLS to work.

---

### 🟡 MEDIUM #4: Missing RLS on Multiple Tables

**Tables Without RLS Protection:**

1. **user_settings** - Contains user preferences and settings
2. **settings** - Global settings (acceptable)
3. **news_articles** - Synthesized news (may be global)
4. **pdf_pages** / **pdf_segments** - Document content fragments
5. **rss_feeds** / **rss_articles** / **rss_feed_subscriptions** - RSS data
6. **research_plans** / **research_plan_executions** / **research_plan_analytics** - Research data
7. **github_connections** / **github_project_mappings** / **github_issue_sync** - GitHub integrations
8. **org_settings** - Organization settings
9. **email_audit_log** / **email_rate_limits** - Email tracking
10. **music_service_configs** / **music_cache** / **music_cache_metadata** - Music service data
11. **entertainment_sync_config** / **entertainment_sync_items** - Entertainment sync

**Impact:** Depends on table usage and data sensitivity

**Recommendation:** Review each table and add RLS where user-specific data exists:

```sql
-- Example for user_settings:
ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_settings_policy ON user_settings
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::varchar);

-- Example for research_plans:
ALTER TABLE research_plans ENABLE ROW LEVEL SECURITY;
CREATE POLICY research_plans_policy ON research_plans
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::varchar);
```

---

### 🟡 MEDIUM #5: Data-Service Has No RLS Implementation

**Location:** `/opt/bastion/data-service/sql/01_init.sql`

**Issue:**
- Data workspace database has NO RLS policies defined
- Relies entirely on application-level permission checks
- Permission checking exists in code but is NOT enforced at database level

**Tables Affected:**
- `data_workspaces` - User workspaces
- `custom_databases` - User databases
- `custom_tables` - User tables
- `custom_data_rows` - User data
- `data_workspace_shares` - Sharing configuration

**Current Protection:**
- Application-level checks in `workspace_permissions.py`
- **NOT USED** - The permission checking code exists but is never imported or called

**Risk:** HIGH - Users could potentially query other users' workspaces if application checks fail

**Recommendation:**
```sql
-- Add RLS to data-service database:
ALTER TABLE data_workspaces ENABLE ROW LEVEL SECURITY;
CREATE POLICY data_workspaces_owner_policy ON data_workspaces
    FOR ALL USING (user_id = current_setting('app.current_user_id', true)::varchar);

CREATE POLICY data_workspaces_shared_policy ON data_workspaces
    FOR SELECT USING (
        workspace_id IN (
            SELECT workspace_id FROM data_workspace_shares 
            WHERE shared_with_user_id = current_setting('app.current_user_id', true)::varchar
            AND (expires_at IS NULL OR expires_at > NOW())
        )
    );

-- Similar policies for other tables...
```

---

### 🟢 LOW #6: API Endpoint Missing RLS Context

**Location:** `/opt/bastion/backend/api/conversation_api.py` lines 64-101

**Issue:**
```python
conn = await asyncpg.connect(connection_string)
rows = await conn.fetch("""
    SELECT * FROM conversations conv
    WHERE conv.user_id = $1
    ...
""", current_user.user_id, limit, skip)
```

**Impact:**
- Direct database connection without setting RLS context
- Relies on WHERE clause instead of RLS policy
- If RLS were enabled, this would fail without setting context

**Risk:** LOW (currently) - WHERE clause provides protection, but not defense-in-depth

**Recommendation:**
```python
conn = await asyncpg.connect(connection_string)
# Set RLS context:
await conn.execute("SELECT set_config('app.current_user_id', $1, true)", current_user.user_id)
await conn.execute("SELECT set_config('app.current_user_role', $1, true)", current_user.role)

# Then query (RLS will enforce automatically):
rows = await conn.fetch("""
    SELECT * FROM conversations conv
    ORDER BY conv.updated_at DESC
    LIMIT $2 OFFSET $3
""", limit, skip)  # No need for WHERE user_id = $1
```

---

## Security Controls That ARE Working

### ✅ Document Metadata RLS (Properly Configured)

**Location:** Lines 2391-2441

- RLS is ENABLED on `document_metadata`
- Policies properly check:
  - User ownership
  - Global collection access
  - Team membership via subquery
  - Admin role

**Example:**
```sql
ALTER TABLE document_metadata ENABLE ROW LEVEL SECURITY;

CREATE POLICY document_metadata_select_policy ON document_metadata
    FOR SELECT USING (
        current_setting('app.current_user_role', true) = 'admin'
        OR user_id = current_setting('app.current_user_id', true)::varchar
        OR collection_type = 'global'
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members 
            WHERE user_id = current_setting('app.current_user_id', true)::varchar
        ))
    );
```

**Status:** ✅ GOOD (but ineffective due to BYPASSRLS on bastion_user)

---

### ✅ Team Tables RLS (Properly Configured)

**Location:** Lines 2444-2506

- RLS enabled on all team tables
- Proper membership checks via subqueries
- Role-based access for invitations

**Status:** ✅ GOOD (but ineffective due to BYPASSRLS)

---

### ✅ Messaging System RLS (Properly Configured)

**Location:** Lines 1997-2173

- RLS enabled on chat rooms, messages, reactions, attachments
- Proper participant checks
- Encryption key isolation

**Status:** ✅ GOOD (but ineffective due to BYPASSRLS)

---

### ✅ LangGraph Checkpoints RLS (Properly Configured)

**Location:** Lines 1662-1714

- RLS enabled with flexible policies
- Allows LangGraph service access (no user context)
- Enforces user isolation when context is set

**Status:** ✅ GOOD (but ineffective due to BYPASSRLS)

---

### ✅ Application-Level Access Controls

**Document Access:**
- `check_document_access()` function in `main.py` (lines 3011-3066)
- Validates ownership, team membership, collection type
- Path traversal protection (lines 3117-3127, 3212-3222)

**Folder Access:**
- `FolderService.get_folder()` checks ownership and collection type
- Proper access control for user/global/team folders

**Team Access:**
- `TeamService.check_team_access()` validates membership
- Role-based permissions enforced

**Status:** ✅ GOOD - Application layer has proper checks

---

## SECURITY DEFINER Functions Review

**Functions with SECURITY DEFINER privilege:**

1. `set_user_context(p_user_id, p_role)` - Line 1220
2. `clear_user_context()` - Line 1229
3. `log_audit_event(...)` - Line 1270
4. `audit_trigger_function()` - Line 1330
5. `check_user_permission(...)` - Line 1398
6. `sanitize_sql_input(...)` - Line 1419
7. `get_user_data_summary(...)` - Line 1500

**Risk Assessment:** ✅ LOW
- These functions are administrative/utility functions
- They don't expose user data inappropriately
- `get_user_data_summary()` properly sets user context before querying

---

## Recommendations Priority

### 🔴 IMMEDIATE (Critical - Fix Today)

1. **Remove BYPASSRLS from bastion_user**
   ```sql
   ALTER ROLE bastion_user NOBYPASSRLS;
   ```
   
2. **Enable RLS on conversations and conversation_messages**
   ```sql
   ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
   ALTER TABLE conversation_messages ENABLE ROW LEVEL SECURITY;
   ```

3. **Test all application functionality** after removing BYPASSRLS
   - Ensure all queries properly set RLS context
   - Fix any queries that break

### 🟡 HIGH PRIORITY (Fix This Week)

4. **Enable RLS on document_folders** with proper policies

5. **Add RLS to user_settings, research_plans, and other user-specific tables**

6. **Implement RLS in data-service database**

7. **Add RLS context setting to direct database connections** in API endpoints

### 🟢 MEDIUM PRIORITY (Fix This Month)

8. **Review and add RLS to remaining tables** (RSS, GitHub, music, etc.)

9. **Create separate admin role** for migrations with BYPASSRLS

10. **Add automated tests** for RLS policy enforcement

11. **Audit all SECURITY DEFINER functions** for potential privilege escalation

---

## Testing Recommendations

### Manual Testing

After fixing BYPASSRLS, test:

```sql
-- As user1:
SET app.current_user_id = 'user1-uuid';
SET app.current_user_role = 'user';

-- Should see only user1's conversations:
SELECT * FROM conversations;

-- Should NOT see user2's conversations:
SELECT * FROM conversations WHERE user_id = 'user2-uuid';  -- Should return empty

-- As user2:
SET app.current_user_id = 'user2-uuid';
SET app.current_user_role = 'user';

-- Should see only user2's conversations:
SELECT * FROM conversations;
```

### Automated Testing

Create integration tests:

```python
async def test_rls_isolation():
    # Create test users
    user1 = create_test_user()
    user2 = create_test_user()
    
    # User1 creates conversation
    conv1 = await create_conversation(user1)
    
    # User2 should NOT see user1's conversation
    user2_convs = await list_conversations(user2)
    assert conv1.id not in [c.id for c in user2_convs]
    
    # User2 should NOT be able to access user1's conversation
    with pytest.raises(PermissionError):
        await get_conversation(conv1.id, user2)
```

---

## Conclusion

The current RLS implementation has **critical security vulnerabilities**:

1. **BYPASSRLS privilege** makes all RLS policies ineffective
2. **Conversations and messages** have RLS disabled
3. **Document folders** have RLS disabled
4. **Data-service** has no RLS implementation
5. **Many user-specific tables** lack RLS protection

**Immediate Action Required:**
- Remove BYPASSRLS from bastion_user
- Enable RLS on conversations, conversation_messages, and document_folders
- Test thoroughly before deploying to production

**Current State:** The application relies entirely on application-level access controls. While these are implemented, they provide only a single layer of defense. Database-level RLS should be the second line of defense (defense-in-depth).

**Risk if Not Fixed:** Users can potentially access other users' private data through:
- SQL injection vulnerabilities
- Application bugs that skip access checks
- Direct database access if credentials are compromised
- Misconfigured API endpoints

---

**Report Generated:** December 18, 2025  
**Next Review:** After implementing critical fixes





