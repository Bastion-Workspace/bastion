# Publish Targets Architecture

Design plan for publishing Bastion documents (markdown + images) to external static site generators (Hugo, Jekyll, Astro, MkDocs, etc.) using the existing data connector infrastructure.

## Core Concept

Reuse `data_source_connectors` as publish targets. A connector tagged/categorized as `publish` appears in a frontend "Publish to..." dropdown. Each publish target has its own credentials, path templates, frontmatter mapping, and image strategy.

## Existing Infrastructure (reuse)

### Data Source Connectors (`data_source_connectors` table)
- `name`, `description`, `connector_type` — identifies the target
- `definition JSONB` — base_url, auth, endpoints (already supports POST/PUT)
- `tags TEXT[]` — can hold `["publish"]` for filtering
- `category VARCHAR(100)` — can be `"publish"`
- `requires_auth`, `auth_fields`, credentials via `agent_data_sources` or inline

### Connector Executor (`connections-service/service/connector_executor.py`)
- Dispatches on `definition.connector_type` (and DB `connector_type` merged by the backend client when absent from JSON): `rest`, `web_fetch`, `sftp`, `s3`, `webdav`
- REST: POST, PUT, PATCH via `httpx`; auth headers (Bearer, API key), body templates, pagination
- **SFTP** (`service/sftp_executor.py`): `asyncssh` — operations `list`, `read`, `write`, `delete`, `mkdir`; credentials: `username`, `password`, optional `private_key` PEM, `passphrase`
- **S3** (`service/s3_executor.py`): `aioboto3` — `list`, `read`, `write`, `delete`; optional `endpoint_url` for MinIO/R2/etc.; credentials: `access_key_id`, `secret_access_key`, optional `session_token`
- **WebDAV** (`service/webdav_executor.py`): `httpx` — PROPFIND/GET/PUT/DELETE/MKCOL; credentials: `username`, `password` (Basic)
- Templates: `backend/services/connector_templates.py` — “SFTP Server”, “S3 / S3-Compatible Storage”, “WebDAV Server”
- Dependencies: `connections-service/requirements.txt` adds `asyncssh`, `aioboto3`

### Export Services
- `epub_export_service.py` — strips/merges frontmatter, resolves cover images
- `pdf_export_service.py` — markdown → HTML → PDF with image embedding
- Image sidecar service + folder service resolve image references to bytes on disk

### Image Storage
- Images on disk with sidecar `.metadata.json` files
- Paths resolvable from document IDs via `folder_service.get_document_file_path`
- Served at `/api/images/...` (internal only for most deployments)

## Industry Patterns (validated by research)

### Pattern 1: Headless CMS → Webhook → Build (Strapi, Contentful)
- Centralized content, API-exposed
- Webhook fires on publish → triggers SSG rebuild (Netlify/Vercel build hook)
- Images via CDN / object storage, not in Git
- Multiple channels consume same content differently (omnichannel)

### Pattern 2: Vault → Git Push → SSG (Obsidian Enveloppe, Digital Garden)
- Frontmatter flag (`share: true`) marks publishable content
- GitHub API pushes files to repo (branch → push → merge/PR)
- Images pushed alongside content as embedded files
- Frontmatter transformed to match target SSG
- Link conversion (wikilinks → standard markdown links)
- Tracks published state for cleanup of deleted/depublished files

### Our plan synthesizes both:
- Like Strapi: centralized content, multiple targets, credential isolation
- Like Enveloppe: Git API as transport, frontmatter transform, image push
- Unlike both: generic connector abstraction instead of per-platform hardcoding

## Architecture

### Publish Flow

```
User clicks "Publish to My Hugo Blog"
         │
         ▼
   ┌─────────────┐
   │ Publish API  │  POST /api/publish {document_id, connector_id, options}
   └──────┬──────┘
          │
          ▼
   ┌─────────────────┐
   │ Publish Service  │  (new, backend/services/publish_service.py)
   │                  │
   │ 1. Load document │  content + frontmatter from DB/disk
   │ 2. Transform FM  │  Bastion fields → target fields via mapping
   │ 3. Resolve images│  walk markdown, resolve to bytes
   │ 4. Rewrite links │  internal refs → published paths (from publish_records)
   │ 5. Push content  │  via connector executor (multi-step for Git targets)
   │ 6. Push images   │  alongside content or to CDN per config
   │ 7. Record state  │  publish_records table
   │ 8. Fire webhook  │  post_publish_webhook (build trigger)
   └─────────────────┘
          │
          ▼
   ┌──────────────────┐
   │ Connector Executor│  existing, extended for multi-step sequences
   │ (connections-svc) │  Git targets: blob → tree → commit → update ref
   └──────────────────┘
```

### Connector Definition Shape (publish target)

```json
{
  "base_url": "https://api.github.com",
  "auth": { "type": "bearer", "credentials_key": "github_token" },
  "publish_config": {
    "content_path_template": "content/posts/{slug}/index.md",
    "image_path_template": "content/posts/{slug}/{image_filename}",
    "image_strategy": "page_bundle",
    "frontmatter_mapping": {
      "title": "title",
      "date": "date",
      "tags": "tags",
      "draft": "draft",
      "author": "author"
    },
    "image_ref_style": "relative",
    "post_publish_webhook": "https://api.netlify.com/build_hooks/abc123"
  },
  "endpoints": {
    "create_or_update_file": {
      "path": "/repos/{owner}/{repo}/contents/{file_path}",
      "method": "PUT",
      "body_template": {
        "message": "Publish: {slug}",
        "content": "{base64_content}",
        "sha": "{existing_sha}"
      }
    },
    "get_file_sha": {
      "path": "/repos/{owner}/{repo}/contents/{file_path}",
      "method": "GET"
    }
  }
}
```

### Publish State Table (new migration)

```sql
CREATE TABLE IF NOT EXISTS publish_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    document_id VARCHAR(255) NOT NULL,
    connector_id UUID NOT NULL REFERENCES data_source_connectors(id) ON DELETE CASCADE,
    remote_path TEXT NOT NULL,
    remote_ref TEXT,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    frontmatter_snapshot JSONB,
    status VARCHAR(50) DEFAULT 'published',
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_publish_records_user ON publish_records(user_id);
CREATE INDEX IF NOT EXISTS idx_publish_records_document ON publish_records(document_id);
CREATE INDEX IF NOT EXISTS idx_publish_records_connector ON publish_records(connector_id);
CREATE INDEX IF NOT EXISTS idx_publish_records_status ON publish_records(status);
```

Fields:
- `remote_path` — where the file lives on the target (e.g. `content/posts/my-post/index.md`)
- `remote_ref` — SHA, etag, or version ID for idempotent updates
- `frontmatter_snapshot` — what frontmatter was sent (for diff detection)
- `status` — `published`, `unpublished`, `failed`
- `metadata` — target-specific data (commit SHA, PR URL, CDN URLs for images)

## Image Handling Strategies

| Strategy | Config value | Best for | Behavior |
|----------|-------------|----------|----------|
| Page bundle | `page_bundle` | Hugo, personal blogs | Push images next to `index.md` in same directory |
| Static folder | `static_folder` | Jekyll, simple sites | Push images to `static/images/...` or similar |
| CDN upload | `cdn` | High-traffic sites | Upload to S3/R2, rewrite markdown URLs to CDN paths |
| External URL | `external_url` | When Bastion is public | Leave `/api/images/...` URLs as-is (rare) |

Configured per connector in `publish_config.image_strategy`. The publish service reads this to decide whether to push image files via the connector or upload to a separate storage endpoint.

## Key Design Decisions

### Multi-Step Publish Orchestration

The connector executor today handles single request-response cycles. Git-based publishing requires a sequence:

1. Create blobs (one per file — markdown + each image)
2. Create a tree referencing all blobs
3. Create a commit pointing to the tree
4. Update the branch ref

**Options:**
- **A. Publish service orchestrates multiple executor calls.** The connector definition has multiple endpoints; the publish service calls them in sequence. More flexible, keeps the executor simple.
- **B. Specialized `connector_type = "git"` with its own executor.** The executor natively understands blob→tree→commit→ref. Cleaner for Git targets but less generic.
- **C. Enveloppe pattern: branch → push files individually → merge.** Simpler but not atomic (partial publishes possible).

Recommended: **Option A** for maximum reuse of existing infrastructure. The publish service owns the orchestration logic; the connector executor stays generic.

### Frontmatter Mapping

The `publish_config.frontmatter_mapping` dict maps Bastion frontmatter keys to target keys:

```json
{
  "title": "title",
  "date": "date",
  "tags": "tags",
  "status": "draft",
  "author": "author",
  "description": "summary"
}
```

Keys not in the mapping are either dropped or passed through (configurable). Target-specific fields can have static defaults:

```json
{
  "frontmatter_defaults": {
    "layout": "post",
    "type": "blog"
  }
}
```

### Cross-Document Link Rewriting

When document A links to document B and both are published to the same target:
1. Publish service looks up B's `publish_records` entry for the same connector
2. If found, rewrite the internal link to B's `remote_path` (converted to a relative URL)
3. If not found, leave the link as-is or strip it (configurable)

This enables publishing interconnected documents (e.g., a series of blog posts that reference each other).

### Build Trigger Webhook

After all files are pushed, the publish service POSTs to `publish_config.post_publish_webhook` (if configured). This triggers SSG rebuild on Netlify, Vercel, Cloudflare Pages, or any CI/CD system.

Webhook payload is minimal: `{ "source": "bastion", "document_id": "...", "slug": "..." }`.

### Unpublish / Cleanup

To unpublish a document:
1. Look up `publish_records` for the document + connector
2. Delete the remote file via the connector's delete endpoint
3. Delete associated image files
4. Update `publish_records.status = 'unpublished'`
5. Optionally fire the build trigger webhook

## Frontend UX

### Publish Dropdown
- Query `GET /api/agent-factory/connectors?category=publish` (or filter by tags)
- Show connectors by `name` in a dropdown: "My Hugo Blog", "Company Docs", etc.
- Each entry shows last publish time (from `publish_records`) if available

### Publish Action
- User opens a document, clicks "Publish to..." → selects target
- Confirmation dialog shows: target name, mapped frontmatter preview, image count
- On confirm: `POST /api/publish` with `{ document_id, connector_id }`
- Progress indicator → success/error toast
- Document shows "Published to X — 2 hours ago" badge

### Settings / Connection Setup
- New publish connector created via existing Agent Factory connector UI
- Template connectors for common targets: "GitHub Pages (Hugo)", "GitLab Pages", "Netlify"
- User fills in: repo URL, branch, token, content path pattern

## What's New vs. Reuse

| Piece | Status |
|-------|--------|
| Connector definition schema (JSONB) | **Extend** — add `publish_config` section |
| Connector HTTP execution (POST/PUT) | **Exists** — connector_executor |
| Tag/category filtering | **Exists** — `tags TEXT[]` and `category` columns |
| Frontend publish dropdown | **New** — small UI component |
| Frontmatter transform service | **New** — dict mapping layer (~50 lines) |
| Image resolution to bytes | **Exists** — folder_service + disk reads |
| Image URL rewriting in markdown | **New** — regex walk of `![]()` references (~80 lines) |
| Cross-document link rewriting | **New** — lookup + rewrite pass (~60 lines) |
| Publish state table + migration | **New** — SQL migration + small repository |
| Publish API endpoint | **New** — `POST /api/publish`, `DELETE /api/publish` |
| Publish orchestration service | **New** — ~300 lines tying it all together |
| Build trigger webhook call | **New** — ~20 lines (POST to URL after publish) |
| Connector templates for common targets | **New** — JSON template files |

## Open Questions

1. **Batch publish**: Should "Publish folder" be supported (publish all documents in a Bastion folder to a target)? Useful for publishing an entire blog section at once.

2. **Draft vs. live**: Should the publish service respect a `draft` frontmatter flag, or always publish whatever the user selects? Hugo supports `draft: true` in frontmatter.

3. **Conflict detection**: If someone edits the file on the Git side directly, the SHA won't match. Should we detect and warn, or force-overwrite?

4. **Scheduled publishing**: Could a Celery task publish on a schedule (e.g., publish drafts at 9am Monday)? The `publish_records` table could hold a `scheduled_at` field.

5. **Preview**: Should there be a "Preview publish" that shows the transformed markdown + frontmatter without actually pushing? Useful for verifying the mapping is correct.

6. **Multi-file documents**: For documents that reference sub-documents (e.g., a book with chapters), should publish bundle them into one page or publish each separately with navigation links?
