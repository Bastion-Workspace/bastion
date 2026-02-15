# Collaborative Editing with Yjs

Overview of Yjs-based collaborative editing: benefits for team documents vs individual documents, and implementation recommendations.

## Current State

- **Editor**: CodeMirror 6 (`@uiw/react-codemirror`) for Markdown and Org files.
- **WebSocket infrastructure**: Exists for notifications (`/api/ws/folders`, etc.) but not for real-time co-editing.
- **Document storage**: Files on disk; metadata in PostgreSQL (`document_metadata`, `collection_type`: `user`, `global`, `team`).
- **Team system**: Teams with roles; `team_id` on documents and folders.

No real-time multi-user editing exists today.

---

## Benefits by Context

### Team Documents (Shared Libraries)

For documents in a team document library (`collection_type = 'team'`), Yjs provides:

| Benefit | Description |
|--------|-------------|
| **Real-time multi-user editing** | Multiple users edit the same document; changes sync character-by-character with conflict-free merging. |
| **Multi-cursor awareness** | See other users’ cursors and selections in the document. |
| **Presence** | Know who else is viewing/editing. |
| **Conflict-free merging** | CRDT-based sync; no manual conflict resolution for concurrent edits. |
| **Per-user undo** | Each user has their own undo stack; no cross-user undo confusion. |
| **Cross-tab sync** | Same user in multiple tabs sees consistent state. |

**Relevant packages**: `yjs`, `y-codemirror.next`, `y-websocket` (or custom WebSocket provider). A small y-websocket server (or equivalent) is required for sync.

---

### Individual Documents (My Documents)

For personal documents (`collection_type = 'user'`), Yjs adds complexity with limited payoff:

| Potential benefit | Relevance for solo editing |
|-------------------|----------------------------|
| **Undo/redo with full history** | Nice-to-have; CodeMirror’s built-in undo is usually sufficient. |
| **Offline-first + sync** | Only relevant if offline editing and reconnection sync are a product goal. |
| **Cross-tab sync** | Useful if the same user edits the same doc in two tabs; otherwise rare. |
| **Agent + human co-editing** | Could let the AI write into the shared Yjs doc instead of the current accept/reject diff overlay; optional design choice. |
| **Multi-cursor / presence** | Not applicable with a single user. |

**Conclusion**: For pure solo editing in My Documents, the current flow (CodeMirror + localStorage drafts + PUT to server) is sufficient. Yjs does not provide essential benefits for that path.

---

## Recommendations

1. **Use Yjs only for team documents**  
   Enable Yjs (and the sync server) when the document belongs to a team library. Keep the existing simple flow for user-owned documents.

2. **Conditional editor setup**  
   In the document editor (e.g. `MarkdownCMEditor`), branch on document context:
   - **Team document** → Initialize Yjs provider + `y-codemirror.next` extension and connect to the sync server.
   - **Personal document** → Use current CodeMirror setup (no Yjs, no sync server).

3. **Sync server scope**  
   Run the Yjs WebSocket server (or equivalent) only for team documents; no need to sync personal documents through it.

4. **Persistence**  
   Keep writing the final content to disk (and your existing APIs) so that non-Yjs clients and backups still see the canonical file. Use Yjs for real-time state; use file/API for persistence and compatibility.

---

## Architecture Sketch (Team Documents Only)

```
User A (team doc)          User B (team doc)
CodeMirror 6               CodeMirror 6
+ y-codemirror             + y-codemirror
        \                         /
         \    WebSocket          /
          \                     /
           y-websocket server (or custom provider)
                    |
           Persist to disk on save / disconnect
```

---

## Summary

| Context | Use Yjs? | Reason |
|--------|----------|--------|
| **Team document library** | Yes | Real-time collaboration, presence, conflict-free merge. |
| **My Documents (solo)** | No | Current flow is adequate; Yjs adds complexity without clear benefit. |

Document annotations (inline notes, @mentions) are a separate feature and are described in `DOCUMENT_ANNOTATIONS_PLAN.md`.
