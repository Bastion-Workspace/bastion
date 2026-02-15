# Document Annotations Plan

Plan for inline notes and comments on documents: annotations that can reference a location in the document, mention team members, and live outside the document body (e.g. in metadata or a dedicated store).

## Goal

Support annotations (notes, todos, @mentions) at specific points in a document without embedding them in the document content. Annotations should:

- Be anchored to a place in the document.
- Support @mentions of other team members.
- Support different types (e.g. comment, todo, question).
- Be queryable and notify mentioned users.

Markdown does not define inline annotations; HTML comments are part of the content and have no notion of authorship or resolution. Therefore annotations are stored separately and anchored by position.

---

## Current State

- **Document metadata**: `document_metadata.metadata_json` (JSONB) exists but is not ideal for annotations (see below).
- **Team comments**: `post_comments` exist for team posts, not for document text.
- **Image annotations**: `user_object_annotations` exist for image bounding boxes, not for document text.
- **Editor**: CodeMirror 6; extension pattern already used for diffs, ghost text, etc.

There is no document-level commenting or annotation system today.

---

## Recommended Approach: Dedicated Annotation Table + Text Anchors

### 1. Storage: Dedicated Tables (Not metadata_json)

Use dedicated tables so that:

- “All annotations where I’m @mentioned” can be queried and indexed.
- Annotations can be paginated, filtered, and used for notification feeds.
- Referential integrity and RLS can be applied.

Storing annotations only in `metadata_json` would require scanning JSONB across documents and is harder to scale and query.

### 2. Position: Text Anchors (Not Line/Offset)

Line numbers and character offsets change on every edit. Anchor annotations using **text plus context** instead:

- **anchor_text**: Exact snippet of text being annotated (e.g. a sentence or phrase).
- **anchor_context_before** / **anchor_context_after**: Short surrounding context (~50 chars) to disambiguate when the same text appears multiple times.
- **anchor_line_hint**: Optional line number hint; advisory only, not authoritative.

When the document is edited, re-locate the annotation by searching for the anchor text and context. If the text is gone or changed, mark the annotation as orphaned and optionally surface it in a “floating” or “unresolved location” list.

### 3. Schema Sketch

```sql
CREATE TABLE document_annotations (
    annotation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,  -- FK to document_metadata(document_id)
    team_id UUID,                       -- FK to teams(team_id); NULL for user docs if needed
    author_user_id VARCHAR(255) NOT NULL,

    -- Position (text-based anchoring)
    anchor_text TEXT NOT NULL,
    anchor_context_before TEXT,
    anchor_context_after TEXT,
    anchor_line_hint INTEGER,

    -- Content
    content TEXT NOT NULL,
    annotation_type VARCHAR(50) DEFAULT 'comment',  -- comment, todo, question, suggestion

    -- @mentions
    mentioned_user_ids TEXT[],

    -- Resolution
    is_resolved BOOLEAN DEFAULT FALSE,
    resolved_by VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE annotation_replies (
    reply_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    annotation_id UUID NOT NULL REFERENCES document_annotations(annotation_id) ON DELETE CASCADE,
    author_user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    mentioned_user_ids TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

Indexes and RLS should align with existing document/team access (e.g. by `document_id`, `team_id`, `user_id` as appropriate).

---

## Frontend: CodeMirror 6 Integration

- **Extension**: New CodeMirror 6 extension (similar to `liveEditDiffExtension.js` / `ghostTextExtension.js`) that:
  - Fetches annotations for the current document.
  - Resolves anchors (find `anchor_text` + context in the current content).
  - Renders **gutter markers** and/or **inline decorations** (e.g. highlight + icon) at those positions.
  - On click, opens a popover or sidebar with the annotation thread, @mentions, and resolution status.
- **Creating annotations**: On user selection + hotkey (or toolbar action), create an annotation whose `anchor_text` (and context) is derived from the selection.
- **@mention UI**: When the user types `@` in the annotation/reply input, show team member autocomplete (reuse existing team member data).

This keeps annotations out of the document buffer and uses the same extension patterns as the rest of the editor.

---

## Notifications

When an annotation is created or replied to with `@mentions`:

1. Insert/update rows in `document_annotations` (and `annotation_replies`) with `mentioned_user_ids`.
2. Create notification records (or integrate with existing notification system).
3. Optionally broadcast over WebSocket (e.g. `/api/ws/folders` or a dedicated annotations channel) for real-time badges.
4. Mentioned users can open the document and jump to the annotation (using the same anchor resolution logic).

---

## Scope: Team vs Individual Documents

Annotations can be defined for:

- **Team documents**: Annotations and @mentions are natural; use `team_id` and team membership for access and notifications.
- **Personal documents**: Optional; if supported, `team_id` can be NULL and only the owner (and optionally collaborators) see annotations.

Implementation can start with team documents only and extend to user documents later if needed.

---

## Relationship to Collaborative Editing

- **Collaborative editing** (Yjs): Real-time sync of document content; see `COLLABORATIVE_EDITING_YJS.md`.
- **Annotations**: Stored in the database and anchored to the document by text; independent of how the document is edited (with or without Yjs).

When content is edited (solo or via Yjs), re-run anchor resolution so that annotations stay attached to the right place or are marked orphaned. No change to the core annotation design.

---

## Summary

| Aspect | Choice |
|--------|--------|
| **Storage** | Dedicated `document_annotations` (and `annotation_replies`) tables, not `metadata_json`. |
| **Position** | Text anchors (snippet + context); re-resolve on load/edit; support orphaned state. |
| **UI** | CodeMirror 6 decoration extension: gutter/decoration + popover/sidebar + @mention autocomplete. |
| **Notifications** | Use `mentioned_user_ids` plus existing (or new) notification and WebSocket channels. |
| **Scope** | Team documents first; individual documents optional. |

This document is independent of the Yjs collaborative-editing design; the two can be implemented and evolved separately.
