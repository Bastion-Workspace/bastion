# SQL Schema Directory

**BULLY!** Welcome to Roosevelt's Database Schema Headquarters! 

## 📁 Current Structure

```
backend/sql/
└── 01_init.sql          ← THE ONLY FILE YOU NEED
```

## 🎯 Single Unified Init Script

**Everything is consolidated into `01_init.sql`:**

- ✅ All table definitions
- ✅ All indexes and constraints  
- ✅ All permissions and grants
- ✅ All foreign key relationships
- ✅ Folder duplication prevention (UNIQUE constraint)
- ✅ GitHub integration tables
- ✅ Complete schema from scratch

## 🚀 How It Works

### Automatic Initialization

When PostgreSQL container starts for the **first time**, it automatically runs:
1. Creates `plato_knowledge_base` database
2. Creates `plato_user` role
3. Executes `01_init.sql` (via `/docker-entrypoint-initdb.d/`)
4. **Done!** Full schema ready

### Fresh Database Setup

```bash
# 1. Stop containers
docker compose down

# 2. Delete database volume (fresh start)
docker volume rm plato_postgres_data

# 3. Start with fresh init
docker compose up --build
```

That's it! No migrations, no multiple SQL files, no complex setup.

## 📊 What's In 01_init.sql

### Tables Created (in order):

1. **Document Management**
   - `document_metadata` - Main document table
   - `document_chunks` - Processed text chunks
   - `entities` - Named entities extraction

2. **PDF Processing**
   - `pdf_pages` - Page metadata
   - `pdf_segments` - OCR segments
   - `segment_relationships` - Layout analysis

3. **Authentication**
   - `users` - User accounts
   - `user_sessions` - Session tokens

4. **Document Folders** ⭐
   - `document_folders` - Hierarchical organization
   - **Includes UNIQUE constraint** - prevents duplicates
   - **Includes UPSERT index** - fast conflict resolution

5. **Conversations**
   - `conversations` - Chat sessions
   - `conversation_messages` - Message history
   - `conversation_shares` - Sharing system
   - `conversation_folders` - Organization

6. **Background Jobs**
   - `background_chat_jobs` - Async chat processing

7. **Templates & Pipelines**
   - `report_templates` - Custom report templates
   - Various pipeline tables

8. **Research Plans**
   - Research planning and execution tracking

9. **RSS Feeds**
    - `rss_feeds` - Feed configurations
    - `rss_articles` - Fetched articles
    - `rss_feed_subscriptions` - User subscriptions

10. **LangGraph State**
    - `checkpoints` - Conversation state persistence
    - `checkpoint_blobs` - Large object storage
    - `checkpoint_writes` - Pending writes

11. **Org-Mode**
    - `org_settings` - Per-user org preferences

12. **GitHub Integration** ⭐
    - `github_connections` - API connections
    - `github_project_mappings` - Repository mappings
    - `github_issue_sync` - Sync tracking

13. **Messaging System**
    - `rooms` - Chat rooms/channels
    - `room_participants` - Room membership
    - `room_messages` - Message storage

14. **Teams & Collaboration**
    - `teams` - Team entities
    - `team_members` - Team membership
    - `team_invitations` - Pending invites
    - `team_folders` - Shared folder structure

15. **Music Library** 🎵
    - `music_sync_configs` - Service configurations (Plex/Jellyfin)
    - `music_sync_items` - Artist/Album/Track metadata
    - `music_liked_albums` - User favorites
    - `music_cache_metadata` - Sync state tracking

16. **Entertainment Tracking** 🎬
    - `entertainment_sync_configs` - Movie/TV service configs
    - `entertainment_sync_items` - Content metadata

17. **Email Agent** 📧
    - `email_agent_connections` - Email account connections
    - `email_agent_folders` - Folder mappings
    - `email_agent_messages` - Cached email data

### Security Features:
- Row Level Security (RLS) on most user tables
- Foreign key constraints with CASCADE deletes
- Check constraints for data validation
- Proper index coverage for performance

## 🔧 For Developers

### Adding New Tables

1. Edit `01_init.sql`
2. Add table in appropriate section
3. Include indexes immediately
4. Add proper constraints
5. Test with fresh DB init:
   ```bash
   docker compose down
   docker volume rm plato_postgres_data
   docker compose up --build
   ```

### Schema Changes

**For fresh deployments:**
- Modify `01_init.sql` directly
- The "source of truth" for clean init

**For existing DBs with data:**
- Keep a migration script separately
- Or use the cleanup utilities in `backend/utils/`

### Testing Changes

```bash
# Quick schema test
docker compose down
docker volume rm plato_postgres_data
docker compose up --build -d

# Check logs
docker compose logs postgres | grep "SQL"
docker compose logs backend | grep "Database"

# Connect to DB
docker compose exec postgres psql -U plato_user -d plato_knowledge_base

# List tables
\dt

# Check constraints
SELECT conname, contype 
FROM pg_constraint 
WHERE conrelid = 'document_folders'::regclass;
```

## 📝 Recent Changes

### November 13, 2025 - FreeForm Notes Complete Removal

**What Changed:**
- ❌ Removed `free_form_notes` table definition from 01_init.sql
- ❌ Removed all indexes, grants, and RLS policies
- ❌ Removed FreeFormNotesService backend
- ❌ Removed notes API endpoints (292 lines)
- ❌ Removed frontend NotesService
- ❌ Removed 6 model classes from api_models.py

**Why:**
- Feature was fully implemented but never had a UI
- No frontend components using the service
- Backend service and API were orphaned code
- ~960 lines of dead code eliminated

**Migration:** 
- Fresh installations: Table no longer created
- Existing deployments: Drop the volume and rebuild for clean slate
  ```bash
  docker compose down
  docker volume rm bastion_postgres_data
  docker compose up --build
  ```

### October 24, 2025 - Consolidation Campaign

**What Changed:**
- ✅ Consolidated 3 SQL files → 1 unified init
- ✅ Added folder duplication prevention (UNIQUE constraint)
- ✅ Integrated GitHub tables into main schema
- ✅ Added UPSERT optimization indexes
- ❌ Deleted `03_fix_folder_duplication.sql` (integrated)
- ❌ Deleted `github_integration_tables.sql` (integrated)

**Why:**
- Simpler deployment (one file!)
- Fresh DB includes all fixes from day one
- No migration complexity for new instances
- Easier to understand and maintain

## 🎯 Key Features

### Folder Management (lines 469-499)
```sql
CREATE TABLE document_folders (
    ...
    -- Roosevelt's Trust-Busting Constraint
    CONSTRAINT unique_folder_per_parent_user 
    UNIQUE (user_id, name, parent_folder_id, collection_type)
);
```
**Prevents race conditions during concurrent folder creation!**

### GitHub Integration (lines 2033-2138)
```sql
-- Complete GitHub project integration
-- Connections, mappings, sync tracking
-- Auto-update triggers included
```

### LangGraph Persistence (lines 1487-1632)
```sql
-- Native LangGraph PostgreSQL checkpointing
-- Supports conversation state persistence
-- Human-in-the-loop patterns enabled
```

## 🏇 Roosevelt's Database Doctrine

**"Speak softly and carry a big schema!"**

- **Single file** = Single source of truth
- **Well organized** = Easy to maintain  
- **Properly indexed** = Fast queries
- **Constraints built-in** = Data integrity
- **Clean slate friendly** = Fast deployments

---

**BULLY!** That's a Square Deal for database management! 

**By George!**, just delete the volume and run `docker compose up --build` - you'll have a perfectly initialized database ready for action! 🎯













