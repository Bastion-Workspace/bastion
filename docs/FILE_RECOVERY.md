# File Recovery System - Roosevelt's Recovery Cavalry 🏇

**BULLY!** Recover lost files after database resets!

## Purpose

When you reset the database (e.g., recreating database volumes or running SQL scripts), files on disk may become "orphaned" - they exist in the filesystem but not in the database. The File Recovery System brings them back without re-vectorizing.

## How It Works

### 1. Filesystem Scan
- Scans user's directory for all supported file types (`.md`, `.txt`, `.org`, `.pdf`, `.epub`)
- Compares against database records
- Identifies files not in `document_metadata`

### 2. Vector Check
- **Checks Qdrant** for existing vectors before recovery
- If vectors exist: marks document as `COMPLETED` (no processing needed)
- If no vectors: queues for vectorization

### 3. Smart Recovery
- Re-creates database records
- Maps files to correct folders based on path
- Creates "Recovered Files" folder for orphaned files
- **Skips re-vectorization for files already in Qdrant**

## Usage

### API Endpoint

```bash
POST /api/user/documents/rescan?dry_run=false
```

**Parameters:**
- `dry_run` (optional, default: false): If true, only reports what would be recovered without making changes

**Returns:**
```json
{
  "success": true,
  "recovered": [
    {
      "document_id": "doc_abc123",
      "filename": "myfile.md",
      "folder_id": "folder_xyz",
      "had_vectors": true,
      "needs_processing": false
    }
  ],
  "recovered_count": 5,
  "skipped": [],
  "skipped_count": 0,
  "errors": [],
  "error_count": 0,
  "total_scanned": 5
}
```

### Frontend Button

**Coming Soon:** A "Rescan Files" button in the file library UI that calls this endpoint.

## Use Cases

### Scenario 1: Database Volume Reset
```bash
# You recreated the database
docker compose down -v
docker compose up --build

# Files are on disk but not in database
# Use rescan to recover them
```

### Scenario 2: Missing Table Recovery
```bash
# You added a new table (like org_settings) manually
# But document_metadata still exists with some records
# Rescan finds files missing from database
```

### Scenario 3: Migration or Sync Issues
- Files added via WebDAV while system was down
- Filesystem restored from backup
- Database restored from older backup

## Key Features

### ✅ Intelligent Vector Checking
- **Queries Qdrant** to see if document already has vectors
- **Avoids re-vectorization** if vectors exist
- **Saves time and resources** on recovery

### ✅ Folder Mapping
- Maps filesystem paths to folder IDs
- `OrgMode/inbox.org` → OrgMode folder
- Unknown paths → "Recovered Files" folder

### ✅ Dry Run Mode
- Test recovery without making changes
- See what would be recovered
- Review before committing

### ✅ .org File Handling
- Recognizes `.org` files don't need vectorization
- Sets appropriate `doc_type`
- Works with org-mode search system

## Recovery Statistics

The system reports:
- **Recovered**: Files successfully re-added to database
- **Skipped**: Files that already exist in database
- **Errors**: Files that failed to recover
- **Vectorization Status**: Whether re-vectorization is needed

## Technical Details

### File Discovery
```python
# Scans user directory recursively
for file_path in user_dir.rglob('*'):
    if file_path.suffix in ['.md', '.txt', '.org', '.pdf', '.epub']:
        # Check if in database
        # Recover if missing
```

### Vector Check
```python
# Queries Qdrant for existing chunks
results = await qdrant_service.search(
    query_text="test",
    limit=1,
    filters={"document_id": document_id}
)
has_vectors = len(results) > 0
```

### Document ID Generation
```python
# Deterministic based on file path
path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
document_id = f"doc_{path_hash[:24]}"
```

## Safety

- **Non-destructive**: Only adds missing records, never deletes
- **Dry run mode**: Test before committing
- **Error handling**: Continues on errors, reports all issues
- **Ownership verification**: Only recovers files for authenticated user

**Remember: "A well-recovered codebase is like a cavalry charge after a strategic retreat - we regroup and charge forward again!"** 🏇



