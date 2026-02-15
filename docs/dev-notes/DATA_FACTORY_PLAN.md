# Data Factory - User-Crafted Scraper Platform

## Executive Summary

**Core Principle**: Give users a Cursor-like iterative AI experience inside Bastion to write, test, and deploy deterministic Python scrapers that run on a schedule with no LLM involvement at runtime.

**Key Insight**: Every website is different. Declarative scraper configs are too rigid. The right approach is to let users iterate with the AI to craft real Python code, then deploy that code as a managed, scheduled Celery task with resume capability, output file management, and metadata sidecar generation.

**What exists today**: The RSS task system is a hardcoded version of this pattern - scheduled polling, per-item processing, file output with markdown formatting, cleanup tasks, health checks. The Data Factory generalizes this into a user-configurable system.

## Architecture Overview

```
User pastes URL into Chat Sidebar
    |
    v
Data Factory Agent (LLM Orchestrator)
    |
    v
+---------------------------------------------+
|  Phase 1: URL Analysis                       |
|  - Crawl4AI fetches page (markdown + HTML)   |
|  - LLM analyzes structure, pagination, etc.  |
|  - Proposes initial scraper approach          |
+---------------------------------------------+
    |
    v
+---------------------------------------------+
|  Phase 2: Iterative Development              |
|  - User and AI refine the scraper code       |
|  - User runs test executions from the UI     |
|  - Preview output files + sidecars           |
|  - Adjust selectors, pagination, naming      |
+---------------------------------------------+
    |
    v
+---------------------------------------------+
|  Phase 3: Deploy                             |
|  - Scraper code stored in database           |
|  - Schedule configured (interval, resume)    |
|  - Output directory + sidecar format set     |
|  - Celery task registered                    |
+---------------------------------------------+
    |
    v
+---------------------------------------------+
|  Runtime: Deterministic Execution            |
|  - Celery Beat triggers on schedule          |
|  - ScraperContext provides sandboxed API     |
|  - Files + sidecars written to disk          |
|  - Cursor state persisted for resume         |
|  - No LLM calls - pure Python execution      |
+---------------------------------------------+
    |
    v
File Watcher picks up new files + sidecars
    |
    v
Documents indexed, embedded, searchable
```

## The ScraperContext Interface

Every user-written scraper receives a `ScraperContext` object that provides sandboxed access to platform capabilities. The scraper code itself is a single async function.

### Context API

```python
class ScraperContext:
    """Provided to user scraper code at runtime. This is the sandbox boundary."""

    # --- Web Fetching (via Crawl4AI) ---
    async def crawl(self, url: str, wait_for: str = None) -> CrawlResult:
        """Fetch a URL via Crawl4AI. Returns markdown + raw HTML."""

    async def crawl_many(self, urls: list[str]) -> list[CrawlResult]:
        """Fetch multiple URLs concurrently."""

    # --- File Output ---
    async def write_markdown(self, filename: str, content: str,
                             frontmatter: dict = None,
                             sidecar: dict = None) -> str:
        """Write a markdown file + optional metadata sidecar.
        Returns the written file path."""

    async def write_image(self, filename: str, image_data: bytes,
                          sidecar: dict = None) -> str:
        """Write an image file + optional metadata sidecar.
        Returns the written file path."""

    async def write_file(self, filename: str, content: bytes | str,
                         sidecar: dict = None) -> str:
        """Write an arbitrary file + optional metadata sidecar."""

    # --- Resume State ---
    @property
    def cursor_state(self) -> dict:
        """Persistent dict that survives between runs.
        Store last page number, last date, processed URLs, etc.
        Automatically saved after each run."""

    # --- Run Metadata ---
    @property
    def output_dir(self) -> str:
        """Base output directory for this scraper's files."""

    @property
    def run_id(self) -> str:
        """Unique ID for this execution run."""

    @property
    def scraper_name(self) -> str:
        """Human-readable name of this scraper."""

    # --- Logging and Progress ---
    def log(self, message: str) -> None:
        """Log a message visible in the run history."""

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress for monitoring UI."""

    # --- Utilities ---
    def should_stop(self) -> bool:
        """Check if the user has requested a stop. Scrapers should
        check this periodically in loops and exit gracefully."""
```

### CrawlResult

```python
class CrawlResult:
    url: str
    markdown: str          # Clean extracted text
    html: str              # Raw HTML for selector work
    status_code: int
    title: str
    links: list[str]       # All links found on page
    images: list[str]      # All image URLs found on page
    success: bool
    error: str | None
```

### Example: Article Scraper

```python
async def scrape(ctx: ScraperContext):
    """Scrape paginated articles from breakpoint.org"""
    from bs4 import BeautifulSoup

    base_url = "https://breakpoint.org/category/articles/"
    last_page = ctx.cursor_state.get("last_page", 0)
    processed_urls = set(ctx.cursor_state.get("processed_urls", []))

    page = last_page + 1
    while not ctx.should_stop():
        url = f"{base_url}page/{page}/" if page > 1 else base_url
        result = await ctx.crawl(url)

        if not result.success or result.status_code == 404:
            ctx.log(f"Reached end at page {page}")
            break

        soup = BeautifulSoup(result.html, "html.parser")
        articles = soup.select("article.post-summary")

        if not articles:
            break

        ctx.set_progress(page, 0, f"Processing page {page}")

        for article in articles:
            link = article.select_one("a.read-more")
            if not link:
                continue

            article_url = link["href"]
            if article_url in processed_urls:
                continue

            article_result = await ctx.crawl(article_url)
            if not article_result.success:
                ctx.log(f"Failed to fetch {article_url}")
                continue

            article_soup = BeautifulSoup(article_result.html, "html.parser")

            title = article_soup.select_one("h1.entry-title")
            title_text = title.get_text(strip=True) if title else "Untitled"

            author_el = article_soup.select_one("span.author-name")
            author = author_el.get_text(strip=True) if author_el else "Unknown"

            date_el = article_soup.select_one("time.published")
            date = date_el.get("datetime", "")[:10] if date_el else ""

            content = article_result.markdown

            slug = article_url.rstrip("/").split("/")[-1]
            filename = f"{date}_{slug}.md" if date else f"{slug}.md"

            frontmatter = {
                "title": title_text,
                "author": author,
                "date": date,
                "source_url": article_url,
                "category": "articles"
            }

            sidecar = {
                "schema_type": "document",
                "schema_version": "1.0",
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "tags": ["breakpoint", "articles"],
                "custom_fields": {
                    "source_domain": "breakpoint.org",
                    "scraped_by": ctx.scraper_name
                }
            }

            await ctx.write_markdown(filename, content,
                                     frontmatter=frontmatter, sidecar=sidecar)
            processed_urls.add(article_url)
            ctx.log(f"Saved: {filename}")

        ctx.cursor_state["last_page"] = page
        ctx.cursor_state["processed_urls"] = list(processed_urls)
        page += 1

    ctx.log(f"Complete. Processed {len(processed_urls)} articles total.")
```

### Example: Date-Range Image Scraper

```python
async def scrape(ctx: ScraperContext):
    """Scrape images from a date-indexed archive"""
    from bs4 import BeautifulSoup
    from datetime import date, timedelta

    start_date = date(2020, 1, 1)
    end_date = date(2025, 12, 31)

    last_date_str = ctx.cursor_state.get("last_date")
    if last_date_str:
        current = date.fromisoformat(last_date_str) + timedelta(days=1)
    else:
        current = start_date

    while current <= end_date and not ctx.should_stop():
        url = f"https://example.com/archive/{current.isoformat()}"
        result = await ctx.crawl(url)

        if not result.success:
            ctx.log(f"Skipping {current}: fetch failed")
            current += timedelta(days=1)
            continue

        soup = BeautifulSoup(result.html, "html.parser")
        images = soup.select("img.archive-photo")

        days_done = (current - start_date).days
        total_days = (end_date - start_date).days
        ctx.set_progress(days_done, total_days, f"Processing {current}")

        for idx, img in enumerate(images):
            img_url = img.get("src", "")
            if not img_url:
                continue

            ext = img_url.rsplit(".", 1)[-1].split("?")[0] if "." in img_url else "jpg"
            filename = f"{current.isoformat()}_{idx:03d}.{ext}"

            img_result = await ctx.crawl(img_url)
            if not img_result.success:
                continue

            alt_text = img.get("alt", "")
            caption_el = img.find_next("div", class_="caption")
            caption = caption_el.get_text(strip=True) if caption_el else ""

            sidecar = {
                "image_filename": filename,
                "type": "photo",
                "title": alt_text or f"Archive photo {current}",
                "content": caption or alt_text,
                "date": current.isoformat(),
                "tags": ["archive", str(current.year), f"{current.month:02d}"],
                "custom_fields": {
                    "source_url": url,
                    "image_index": idx
                }
            }

            await ctx.write_image(filename, img_result.html.encode(),
                                  sidecar=sidecar)

        ctx.cursor_state["last_date"] = current.isoformat()
        current += timedelta(days=1)

    ctx.log(f"Complete. Processed through {current - timedelta(days=1)}")
```

---

## Database Schema

### scraper_definitions

Stores the user's scraper code and configuration.

```sql
CREATE TABLE scraper_definitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- The actual scraper code (Python source)
    scraper_code TEXT NOT NULL,

    -- Configuration
    output_directory VARCHAR(500) NOT NULL,
    schedule_interval INTEGER,
    resume_strategy VARCHAR(50) NOT NULL DEFAULT 'by_cursor',

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'draft',

    -- Metadata
    target_urls TEXT[],
    tags VARCHAR(100)[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_run_at TIMESTAMP WITH TIME ZONE,
    created_from_conversation_id UUID
);

CREATE INDEX idx_scraper_definitions_user ON scraper_definitions(user_id);
CREATE INDEX idx_scraper_definitions_status ON scraper_definitions(status);
```

**Status values**: `draft` (still being developed in chat), `active` (scheduled and running), `paused` (temporarily stopped), `disabled` (permanently stopped).

**Resume strategies**: `by_cursor` (use cursor_state to pick up where left off), `skip_existing` (check output dir for existing files), `full_rescrape` (start from scratch every time).

### scraper_runs

Tracks each execution with cursor state for resume.

```sql
CREATE TABLE scraper_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scraper_id UUID NOT NULL REFERENCES scraper_definitions(id) ON DELETE CASCADE,

    -- Execution state
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    cursor_state JSONB DEFAULT '{}',
    items_processed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    files_written INTEGER DEFAULT 0,

    -- Timing
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Logging
    log_entries JSONB DEFAULT '[]',
    error TEXT,

    -- Celery task tracking
    celery_task_id VARCHAR(255)
);

CREATE INDEX idx_scraper_runs_scraper ON scraper_runs(scraper_id);
CREATE INDEX idx_scraper_runs_status ON scraper_runs(status);
CREATE INDEX idx_scraper_runs_started ON scraper_runs(started_at DESC);
```

**Run status values**: `pending`, `running`, `completed`, `failed`, `stopped`, `paused`.

---

## Phase 1: Core Infrastructure

**Goal**: Build the execution engine and storage layer. Scrapers can be created and run manually via API.

### Tasks

1. **Database migration** - Add `scraper_definitions` and `scraper_runs` tables.
   - File: `backend/sql/migrations/add_scraper_tables.sql`

2. **Pydantic models** - Define request/response models for scraper CRUD and run management.
   - File: `backend/models/scraper_models.py`
   - Models: `ScraperDefinitionCreate`, `ScraperDefinitionUpdate`, `ScraperDefinitionResponse`, `ScraperRunResponse`, `ScraperTestRequest`

3. **ScraperContext implementation** - The sandboxed runtime context provided to user code.
   - File: `backend/services/scraper_engine/scraper_context.py`
   - `crawl()` delegates to the existing Crawl4AI gRPC client
   - `write_markdown()` / `write_image()` / `write_file()` write to `output_dir` under `UPLOAD_DIR` and generate compliant metadata sidecars
   - `cursor_state` loaded from last run's state, saved after execution
   - `should_stop()` checks a Redis flag set by the stop API

4. **Scraper executor** - Loads user code, provides context, runs in a safe manner.
   - File: `backend/services/scraper_engine/scraper_executor.py`
   - Loads scraper code from database
   - Creates `ScraperContext` with run-specific config
   - Executes the user's `scrape()` function with the context
   - Captures exceptions, enforces time limits
   - Saves final cursor_state and run results

5. **Scraper repository** - Database access layer for scraper definitions and runs.
   - File: `backend/repositories/scraper_repository.py`
   - CRUD for `scraper_definitions` and `scraper_runs`
   - Queries: active scrapers due for execution, run history, latest cursor state

6. **Scraper service** - Business logic layer.
   - File: `backend/services/scraper_service.py`
   - Create / update / delete definitions
   - Trigger manual runs
   - Start / stop / pause scrapers
   - Get run history and logs
   - Test execution (limited to N items)

7. **Scraper API** - REST endpoints.
   - File: `backend/api/scraper_api.py`
   - `POST /api/scrapers` - Create scraper definition
   - `GET /api/scrapers` - List user's scrapers
   - `GET /api/scrapers/{id}` - Get scraper details + recent runs
   - `PUT /api/scrapers/{id}` - Update definition
   - `DELETE /api/scrapers/{id}` - Delete scraper
   - `POST /api/scrapers/{id}/run` - Trigger manual run
   - `POST /api/scrapers/{id}/stop` - Stop current run
   - `POST /api/scrapers/{id}/pause` - Pause scheduling
   - `POST /api/scrapers/{id}/activate` - Resume scheduling
   - `POST /api/scrapers/{id}/test` - Test run (limited scope)
   - `GET /api/scrapers/{id}/runs` - Run history
   - `GET /api/scrapers/{id}/runs/{run_id}` - Run details + logs
   - `GET /api/scrapers/{id}/runs/{run_id}/logs` - Stream logs (SSE)

### Sandboxing Considerations

The user's scraper code runs inside the Celery worker process. The `ScraperContext` is the trust boundary.

**Provided via context (safe)**:
- Web fetching through Crawl4AI (rate-limited, logged)
- File writing to the scraper's designated output directory only
- Cursor state persistence
- Logging

**Allowed in user code (acceptable risk for admin/authorized users)**:
- `BeautifulSoup`, `lxml`, `re`, `json`, `datetime` - standard parsing libraries
- String manipulation, data transformation logic

**Not provided (no access path)**:
- Direct database access
- Direct filesystem access outside output directory
- Network access outside of `ctx.crawl()`
- Access to other users' data

**Future hardening**: If scrapers are opened to untrusted users, execute them in isolated containers or use RestrictedPython for AST-level import restrictions.

---

## Phase 2: Celery Integration

**Goal**: Scrapers run on schedule via Celery Beat. Start, stop, and resume work reliably.

### Tasks

1. **Scraper Celery tasks** - Following the RSS task pattern.
   - File: `backend/services/celery_tasks/scraper_tasks.py`

   | Task | Purpose | Trigger |
   |------|---------|---------|
   | `scheduled_scraper_poll_task` | Check which scrapers are due to run | Beat: configurable |
   | `run_scraper_task(scraper_id)` | Execute a single scraper | On-demand or from poll |
   | `stop_scraper_task(scraper_id)` | Set stop flag in Redis | API call |
   | `cleanup_stuck_scrapers_task` | Reset scrapers stuck in running state | Beat: every 15 min |
   | `scraper_health_check_task` | Verify scraper targets are reachable | Beat: every 6 hours |

2. **Celery app registration** - Add scraper tasks to Celery.
   - File: `backend/services/celery_app.py` (modify)
   - Add `services.celery_tasks.scraper_tasks` to task includes
   - Add `scrapers` queue to queue definitions
   - Add Beat schedule entries for poll and cleanup tasks

3. **Resume logic** - Cursor state flows between runs.
   ```
   run_scraper_task(scraper_id):
     1. Load scraper definition from DB
     2. Load latest successful run's cursor_state (or {} for first run)
     3. Create ScraperContext with loaded cursor_state
     4. Execute scraper code
     5. Save cursor_state from context to scraper_runs row
     6. On next run, step 2 loads this state
   ```

4. **Stop mechanism** - Graceful stopping via Redis flag.
   ```
   Stop flow:
     1. API sets Redis key: scraper:{scraper_id}:stop = true
     2. ScraperContext.should_stop() checks this key
     3. User code checks ctx.should_stop() in loops
     4. After scraper exits, cursor_state is saved (can resume later)
     5. Redis key is cleared
   ```

5. **Celery worker queue** - Add `scrapers` queue to docker-compose worker command.
   - File: `docker-compose.yml` (modify celery_worker command)
   - Add `scrapers` to `--queues=orchestrator,agents,rss,default,scrapers`

---

## Phase 3: AI-Assisted Scraper Development

**Goal**: Users can build scrapers conversationally in the Chat Sidebar, with the AI analyzing URLs and generating scraper code.

### Tasks

1. **Data Factory Agent** - New agent in the LLM orchestrator.
   - File: `llm-orchestrator/orchestrator/agents/data_factory_agent.py`
   - Workflow nodes:
     - `analyze_url_node` - Crawls the target URL via Crawl4AI, extracts markdown and HTML, presents structure analysis
     - `generate_scraper_node` - Produces scraper code based on user requirements and URL analysis
     - `refine_scraper_node` - Iterates on the code based on user feedback
     - `deploy_scraper_node` - Saves finalized scraper to the database via gRPC tool call
   - Key behaviors:
     - When user provides a URL, the agent fetches it and describes what it sees: pagination pattern, content structure, available selectors, data fields
     - Proposes an initial scraper, explains the approach
     - User can ask for changes: different selectors, different output format, different naming, different sidecar fields
     - Agent knows the ScraperContext API and generates code that uses it correctly
     - Agent knows the sidecar schemas (image and document) and generates compliant metadata

2. **Agent registration** - Register in agent capabilities.
   - File: `llm-orchestrator/orchestrator/services/agent_capabilities.py` (modify)
   - Add `data_factory` agent type
   - Route queries that mention scraping, data collection, web harvesting, data factory

3. **Scraper management tools** - gRPC tools for the agent to save/update scrapers.
   - File: `llm-orchestrator/orchestrator/tools/scraper_tools.py`
   - `save_scraper_definition` - Save scraper code + config to backend
   - `test_scraper` - Trigger a test run and return results
   - `list_user_scrapers` - Show existing scrapers

4. **Backend gRPC handlers** - Tool service handlers for scraper operations.
   - File: `backend/services/grpc_tool_service.py` (modify)
   - Add handlers for scraper CRUD operations called by the orchestrator

5. **System prompt engineering** - The Data Factory agent's system prompt must include:
   - The full ScraperContext API reference
   - Both sidecar schemas (image and document) with all fields
   - Example scrapers for common patterns (paginated articles, date ranges, image galleries)
   - Guidance on resume strategies and when to use each
   - Common pitfalls (rate limiting, anti-bot, dynamic content)

---

## Phase 4: Frontend - Data Factory UI

**Goal**: A dedicated page for managing scrapers with integrated chat for building them.

### Page Layout

```
+----------------------------------------------------------------+
|  Data Factory                                           [+ New] |
+--------------+-------------------------------------------------+
|              |                                                  |
|  Scraper     |   Selected Scraper Detail                       |
|  List        |                                                  |
|              |   Name: Breakpoint Articles                      |
|  * Active    |   Status: Active    [Pause] [Run Now] [Edit]    |
|  o Paused    |   Schedule: Every 24 hours                       |
|  o Draft     |   Resume: by_cursor                              |
|              |   Output: web_sources/breakpoint_articles/        |
|  [Breakpoint |   Last Run: 2026-02-10 03:00 UTC                 |
|   Articles]  |   Items: 342 articles, 0 failures                |
|              |                                                  |
|  [Historical |   +-------------------------------------------+  |
|   Archive]   |   |  Run History                              |  |
|              |   |  Feb 10 03:00  Complete    12 new items    |  |
|  [Product    |   |  Feb 09 03:00  Complete     8 new items   |  |
|   Catalog]   |   |  Feb 08 03:00  Failed      Connection err |  |
|              |   |  Feb 07 03:00  Complete     5 new items    |  |
|              |   +-------------------------------------------+  |
|              |                                                  |
|              |   +-------------------------------------------+  |
|              |   |  Scraper Code                  [View/Edit]|  |
|              |   |  async def scrape(ctx):                   |  |
|              |   |      ...                                  |  |
|              |   +-------------------------------------------+  |
|              |                                                  |
+--------------+-------------------------------------------------+
```

### Components

1. **DataFactoryPage.js** - Main page with scraper list + detail panel.
   - File: `frontend/src/components/DataFactoryPage.js`

2. **ScraperList.js** - Sidebar list of scrapers with status indicators.
   - File: `frontend/src/components/data-factory/ScraperList.js`

3. **ScraperDetail.js** - Detail view with status, controls, run history, code preview.
   - File: `frontend/src/components/data-factory/ScraperDetail.js`

4. **ScraperRunHistory.js** - Run history table with expandable log entries.
   - File: `frontend/src/components/data-factory/ScraperRunHistory.js`

5. **ScraperCodeEditor.js** - Code editor panel (read-only view + link to chat for editing).
   - File: `frontend/src/components/data-factory/ScraperCodeEditor.js`

6. **DataFactoryService.js** - API client for scraper endpoints.
   - File: `frontend/src/services/data-factory/DataFactoryService.js`

7. **Route and Navigation** - Add Data Factory to the app navigation.
   - Files: `frontend/src/App.js`, `frontend/src/components/Navigation.js` (modify)

### Chat Integration

The Chat Sidebar already works on every page. When on the Data Factory page:
- The agent detects the page context (data factory)
- User can say "Help me build a scraper for [URL]" and the agent enters the workflow
- When the scraper is finalized, the agent calls `save_scraper_definition` and it appears in the list
- User can select an existing scraper and say "modify this scraper to also extract images" - the agent loads the current code and iterates

---

## Phase 5: Advanced Features

**Goal**: Quality-of-life improvements after the core system is working.

1. **Test execution with preview** - Run scraper with `max_items=3`, show output files and sidecars in the UI before committing to a full run.

2. **Webhook notifications** - Notify on run completion or failure via the existing email service or a webhook URL.

3. **Rate limiting configuration** - Per-scraper rate limits (requests per minute) enforced by the context's `crawl()` method.

4. **Scraper templates** - Pre-built scraper code templates for common patterns:
   - Paginated article list to markdown files
   - Date-range archive to image files
   - API endpoint with pagination to JSON/markdown files
   - RSS-like feed (non-standard) to markdown files

5. **Dry run mode** - Execute scraper but don't write files, just log what would be written. Useful for validating changes before deploying.

6. **Import/Export** - Export scraper definition as a JSON file, import on another instance.

7. **Dependency management** - Allow scrapers to specify additional pip packages beyond the standard set (BeautifulSoup, lxml, re, json). Install into a virtualenv per scraper or maintain a shared set.

---

## Existing Infrastructure Leverage

| Component | How Data Factory Uses It |
|---|---|
| `crawl4ai-service` | All web fetching goes through the existing Crawl4AI gRPC service. No new crawling infrastructure. |
| Celery Beat / Worker | Scraper scheduling uses the same Beat + Worker infrastructure as RSS tasks. Add a `scrapers` queue. |
| Celery Flower | Scraper task monitoring works out of the box via Flower on port 5555. |
| `file_manager_service.py` | File writing patterns (mkdir, write, sidecar generation) are reused in ScraperContext. |
| Metadata sidecars | ScraperContext generates `{stem}.metadata.json` compliant with image and document sidecar schemas. |
| `file_watcher_service.py` | Automatically picks up new files + sidecars written by scrapers, indexes and embeds them. |
| Chat Sidebar | No changes needed. Data Factory agent is just another agent the orchestrator routes to. |
| Admin role gating | Data Factory page visibility gated by role/capability, same pattern as existing admin tabs. |
| Redis | Used for stop signals and potentially scraper-level rate limiting. |
| PostgreSQL | Two new tables, same database, same connection patterns. |

---

## Security Considerations

### Code Execution Risk

User-written Python code runs in the Celery worker. Mitigations:

1. **Access control** - Only admin or users with explicit `data_factory` capability can create scrapers.
2. **ScraperContext boundary** - The context object is the only interface. No direct DB, filesystem, or network access is provided.
3. **Import restrictions (future)** - Use RestrictedPython or AST analysis to limit imports to an allowlist.
4. **Resource limits** - Celery's `soft_time_limit` and `time_limit` prevent runaway scrapers.
5. **Rate limiting** - `crawl()` enforces per-scraper request rate limits.
6. **Output directory isolation** - Each scraper writes only to its designated directory.

### Progression Path

- Phase 1: Trust boundary is admin-only access + ScraperContext API surface
- Future: If opened to non-admin users, add RestrictedPython sandboxing or container isolation

---

## Implementation Order

1. **Phase 1 + 2** (Core + Celery) should be built together as they are tightly coupled
2. **Phase 3** (Agent) can begin in parallel once the API is defined
3. **Phase 4** (Frontend) can begin once the API endpoints are functional
4. **Phase 5** (Advanced) is iterative improvement after launch

**Estimated scope**:
- Phase 1 + 2: ~8-10 new files, ~1500-2000 lines
- Phase 3: ~3-4 new files, ~600-800 lines
- Phase 4: ~6-7 new files, ~1200-1500 lines
- Phase 5: Incremental additions

**Dependencies**: No new external services or Docker containers required. Everything runs on existing infrastructure.
