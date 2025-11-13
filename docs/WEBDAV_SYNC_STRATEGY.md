# WebDAV File Synchronization Strategy

**ROOSEVELT'S SYNC DOCTRINE!** 🔄

## Overview

Our WebDAV implementation uses **MD5 content hashing** for reliable file synchronization and conflict detection. This is more accurate than timestamp-based approaches.

## How Sync Clients Determine File Changes

### 1. **ETag (Entity Tag) - Primary Method**

**What We Provide:**
- **MD5 hash of file content** (for files < 50MB)
- **mtime-size combination** (for large files > 50MB, for performance)
- Cached per file to avoid repeated hashing

**Format:** `"5d41402abc4b2a76b9719d911017c592"` (MD5 hash)

**How Clients Use It:**
```
1. Client requests file metadata (PROPFIND)
2. Server returns ETag + Last-Modified + Content-Length
3. Client compares server ETag with local cached ETag
4. Decision:
   - ETags match → File unchanged, skip
   - ETags differ → File changed, sync needed
```

### 2. **Last-Modified Timestamp - Secondary Method**

**What We Provide:**
- Filesystem modification time (`mtime`)
- Format: `Thu, 26 Oct 2025 14:23:45 GMT`

**How Clients Use It:**
- Determine sync direction (which version is newer)
- Fallback when ETag not supported
- Used for "last write wins" conflict resolution

### 3. **Content-Length - Verification**

**What We Provide:**
- File size in bytes
- Used for integrity checks and progress tracking

## Sync Scenarios

### ✅ Scenario 1: File Edited in App

```
1. User edits file in web app
2. Server updates file on disk
3. File watcher detects change
4. Client syncs:
   - Requests metadata (PROPFIND)
   - Sees different ETag (content changed!)
   - Sees newer Last-Modified timestamp
   - Decision: DOWNLOAD server version
   - Result: Client file updated with app edits
```

### ✅ Scenario 2: File Edited in Client

```
1. User edits file in local editor
2. Sync client detects local change
3. Client syncs:
   - Compares local/server Last-Modified
   - Local is newer
   - Decision: UPLOAD to server
   - Server file replaced
   - File watcher detects change
   - App database updated
   - Result: App sees client edits
```

### ⚠️ Scenario 3: Conflict (Both Changed)

```
1. File edited in app (server newer)
2. File edited locally (client newer)
3. Client syncs:
   - Sees different ETag (content conflict!)
   - Sees BOTH timestamps changed
   - Decision: CONFLICT HANDLING
   
Conflict Resolution Options (client-dependent):
- "Last write wins" - newest timestamp overwrites
- "Ask user" - manual conflict resolution
- "Create copy" - keep both versions
- "Merge" - advanced clients may attempt merge
```

### ✅ Scenario 4: File Copied (Not Edited)

**Problem with Timestamp-Only Sync:**
- Copied files might have old timestamps
- Sync client might think "already synced"
- Content actually different!

**Our MD5 Solution:**
- Content hash detects actual content difference
- Even if timestamps match, ETag differs
- Sync happens correctly!

## Implementation Details

### Custom MD5FileResource Class

Located in `backend/webdav/simple_filesystem_provider.py`:

```python
class MD5FileResource(FileResource):
    """
    Generates ETags based on MD5 content hash instead of mtime+size.
    
    Benefits:
    - Accurate change detection (content-based)
    - Prevents false "unchanged" when content differs
    - Better conflict resolution
    
    Trade-offs:
    - Slower ETag generation (must read file)
    - Cached after first read for performance
    - Falls back to mtime for files > 50MB
    """
    
    def get_etag(self):
        # Check cache (reuse if mtime unchanged)
        # Hash file content with MD5
        # Return MD5 hash as ETag
        # Fall back to mtime for large files
```

### Performance Optimizations

1. **ETag Caching:**
   - Cache MD5 hash per file
   - Invalidate cache when `mtime` changes
   - Avoids re-hashing on every request

2. **Large File Handling:**
   - Files > 50MB use `mtime-size` ETag
   - Avoids performance hit on big files
   - Most documents are < 50MB

3. **Chunked Reading:**
   - Read files in 8KB chunks
   - Prevents memory issues with large files

## Sync Client Behavior

### Common Clients

**WinSCP:**
- Supports ETags
- "Last write wins" by default
- Can be configured for manual conflict resolution

**Cyberduck:**
- Full ETag support
- Checksum verification
- Conflict detection with user prompts

**Windows Network Drive:**
- Basic WebDAV support
- Relies more on timestamps
- May not use ETags optimally

**macOS Finder:**
- Good WebDAV support
- Uses ETags when available
- "Last write wins" default

### Recommended Client Settings

For best results with our MD5 ETag implementation:

1. **Enable checksum verification** (if available)
2. **Set conflict resolution** to:
   - "Ask me" (safest, manual review)
   - "Last write wins" (automatic, risk of data loss)
3. **Sync frequency:**
   - Real-time monitoring recommended
   - Or manual sync before/after editing

## Testing Sync Behavior

### Test 1: Edit in App, Sync to Client

```bash
1. Edit file in web app editor
2. In sync client: Refresh/Sync
3. Expected: Client downloads newer version
4. Check: File content matches app edits
```

### Test 2: Edit in Client, Sync to Server

```bash
1. Edit file in local editor
2. Save and sync
3. In web app: Refresh file library
4. Expected: App shows client edits
```

### Test 3: Conflict Detection

```bash
1. Edit file in app (don't sync yet)
2. Edit SAME file locally with different content
3. Sync client
4. Expected: Conflict detected
   - Client prompts for resolution OR
   - "Last write wins" applies automatically
```

### Test 4: Copy Detection

```bash
1. Copy file with old timestamp
2. Modify content but preserve timestamp
3. Sync
4. Expected: MD5 detects content change, syncs correctly
```

## Monitoring Sync Operations

### Backend Logs

WebDAV operations are logged:

```
📂 ========== WebDAV Request ==========
📂 User: admin
📂 HTTP Method: PROPFIND
📦 ETag cache hit for: document.md
🔐 Calculating MD5 ETag for: new-file.txt (4096 bytes)
✅ MD5 ETag generated: new-file.txt → "5d41402abc4b2a76b9719d911017c592"
```

### File Watcher Logs

File changes detected and processed:

```
📄 NEW FILE FOUND: Users/admin/document.md
🔄 Processing 1 debounced file events
✅ File added via watcher: document.md
```

## Troubleshooting

### Problem: Client Says "No Changes"

**Possible Causes:**
- Client cached old ETag
- Clock skew between client/server
- Client not using ETags properly

**Solutions:**
- Force refresh in client
- Clear client cache
- Check client ETag support

### Problem: Constant Re-Syncing

**Possible Causes:**
- Timestamp precision mismatch
- Client modifying files on download
- Permission/ownership changes

**Solutions:**
- Check file timestamps match
- Disable "preserve modification time" in client
- Ensure consistent file permissions

### Problem: Conflicts Not Detected

**Possible Causes:**
- Client using only timestamps
- ETag not being sent/received
- Network issues truncating headers

**Solutions:**
- Check client supports ETags
- Verify HTTP headers in logs
- Try different sync client

## Future Enhancements

### 1. Conflict Versioning

Store conflicted versions in database:
```
document.md
document.md.conflict-2025-10-26-14-23-45
```

### 2. Merge Support

For text files, attempt automatic merge:
- Three-way merge (base, client, server)
- Git-style conflict markers
- Present merge result to user

### 3. Selective Sync

Allow users to configure:
- Which folders to sync
- File size limits
- File type filters

### 4. Delta Sync

For large files:
- Only transfer changed portions
- Rsync-style binary diff
- Reduces bandwidth usage

## Conclusion

**BY GEORGE!** Our MD5-based ETag system provides:

✅ **Accurate conflict detection** - content-based, not timestamp guessing  
✅ **Reliable sync** - detects actual changes, not just metadata  
✅ **Performance optimized** - caching + large file handling  
✅ **Client compatible** - works with standard WebDAV clients  

**BULLY!** Your files stay in sync like a well-organized cavalry charge! 🏇











