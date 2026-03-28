# Future Tooling — Skill Integrations Roadmap

Assessment of external skill/tool integrations that would extend Bastion's capabilities, organized by zone placement and implementation priority. Source material reviewed from the [Anthropic Skills repository](https://github.com/anthropics/skills/tree/main/skills) (official document-format skills), the [OpenClaw skills ecosystem](https://github.com/VoltAgent/awesome-openclaw-skills) (3,000+ community skills), and filtered for relevance to Bastion's architecture.

## Placement Rules Recap

| Zone | Container | What goes here | Extra deps OK? |
|------|-----------|----------------|----------------|
| Zone 1 | llm-orchestrator | In-memory/string-only tools (math, parsing, LLM-in-process) | **NO** — no pip packages, no system packages |
| Zone 2 | tools-service | Bastion's own DB, vector store, files, heavy processing | Yes — pip and apt in tools-service Dockerfile |
| Zone 3 | connections-service | External messaging/provider APIs with persistent connections | Yes — pip in connections-service requirements |
| Zone 4 | llm-orchestrator (plugins/) | Third-party SaaS REST/GraphQL (request/response, no extra deps) | Only if stdlib or already in orchestrator requirements (httpx, pydantic) |

**Key constraint:** The LLM Orchestrator container stays lean. Plugins (Zone 4) use only `httpx` for HTTP calls and `pydantic` for models — both already in the orchestrator's requirements. Any tool that requires additional Python packages or system-level software MUST go in tools-service or connections-service.

---

## Tier 1 — High Value, Strong Fit

### 1. Slack Integration

**Zone:** 3 (Connections Service) + 4 (Plugin)

**Why:** Slack is the dominant business communication tool. Telegram and Discord are supported; Slack is the most-requested gap. Already planned in `CHANNELS_ROADMAP_AND_DEPLOYMENT.md` Phase 1.

**Two-zone split (same pattern as the Slack entry in the channels roadmap):**

- **Zone 3 — Always-on bot (connections-service):**
  - `SlackProvider` extending `BaseMessagingProvider`
  - Socket Mode or Events API for inbound messages
  - Same message flow as Telegram/Discord: inbound → backend → reply
  - Normalize to `InboundMessage` (sender_id, chat_id, text, images, platform)

- **Zone 4 — On-demand API tools (plugin):**
  - `slack_plugin.py` extending `BasePlugin`
  - Tools: `slack_send_message`, `slack_list_channels`, `slack_read_channel_history`, `slack_set_topic`
  - For Agent Factory playbooks that need to post to Slack as a step
  - Uses `httpx` against Slack Web API (no extra deps in orchestrator)

**Connections-service deps:** `slack-bolt>=1.20.0` or `slack-sdk>=3.30.0`

**Implementation steps:**
1. Add `slack-bolt` to `connections-service/requirements.txt`
2. Create `connections-service/providers/slack_provider.py` implementing `BaseMessagingProvider`
3. Register in `provider_router.py`
4. Add Slack connection type to external connections UI/API
5. Create `llm-orchestrator/orchestrator/plugins/integrations/slack_plugin.py` (httpx only)
6. Register plugin tools with Action I/O Registry

**Best-of-breed:** Slack Bolt SDK (Python) for the bot; raw `httpx` + Slack Web API for the plugin.

---

### 2. Google Sheets Integration

**Zone:** 4 (Plugin)

**Why:** The Data Workspace handles SQL, but most users have data in spreadsheets. Google Sheets is the most common cloud spreadsheet. Bridges a major gap in data ingestion workflows.

**Tools:**
- `sheets_read_range` — Read cells from a named range or A1 notation
- `sheets_write_range` — Write data to a range
- `sheets_append_rows` — Append rows to the end of a sheet
- `sheets_list_sheets` — List all sheets in a workbook
- `sheets_create_sheet` — Create a new sheet in a workbook

**Auth:** OAuth2 (Google API) stored in external connections. The plugin receives credentials via `configure()`.

**Implementation steps:**
1. Create `llm-orchestrator/orchestrator/plugins/integrations/google_sheets_plugin.py`
2. All API calls via `httpx` against Google Sheets API v4 (REST, no SDK needed)
3. OAuth token refresh handled by the plugin's `configure()` method
4. Define Pydantic I/O models with `formatted` field per tool-io-contracts rule
5. Add Google OAuth connection type to external connections API if not already present

**Best-of-breed:** Direct REST calls via `httpx` to Sheets API v4. No `google-api-python-client` needed — the REST API is clean enough to call directly, keeping the orchestrator container free of extra deps.

---

### 3. Linear Issue Tracking

**Zone:** 4 (Plugin)

**Why:** Project/issue tracking is a gap. The org-mode todos handle personal tasks; Linear covers team-oriented issue tracking. Linear has a cleaner API than Jira and is increasingly popular.

**Tools:**
- `linear_list_issues` — List issues with filters (project, status, assignee, label)
- `linear_create_issue` — Create an issue with title, description, project, assignee, priority
- `linear_update_issue` — Update status, assignee, priority, labels
- `linear_search_issues` — Full-text search across issues
- `linear_list_projects` — List projects and teams
- `linear_get_issue` — Get full issue detail with comments

**Auth:** API key or OAuth token stored in external connections.

**Implementation steps:**
1. Create `llm-orchestrator/orchestrator/plugins/integrations/linear_plugin.py`
2. GraphQL calls via `httpx` to `https://api.linear.app/graphql`
3. Define Pydantic I/O models (issues, projects, etc.)
4. Register with Action I/O Registry

**Best-of-breed:** Direct GraphQL via `httpx`. Linear's schema is well-documented and stable. No SDK required.

**Note:** Jira can be added later as a separate plugin. Jira's REST API is more complex (pagination, JQL) but the same pattern applies. Both should be separate plugins — users choose which they need.

---

### 4. Google Calendar (Native API)

**Zone:** 4 (Plugin)

**Why:** The CalDAV plugin covers basic calendar access, but a native Google Calendar plugin provides richer features: Meet link generation, attendee management, free/busy queries, and proper notification settings. Most users are on Google Workspace.

**Tools:**
- `gcal_list_events` — List events in a time range with calendar filter
- `gcal_create_event` — Create event with attendees, Meet link, reminders
- `gcal_update_event` — Reschedule, add attendees, change details
- `gcal_delete_event` — Delete/cancel an event
- `gcal_find_free_time` — Query free/busy across calendars
- `gcal_list_calendars` — List user's calendars

**Auth:** OAuth2 (Google API). Same OAuth flow as Google Sheets — can share connection.

**Implementation steps:**
1. Create `llm-orchestrator/orchestrator/plugins/integrations/google_calendar_plugin.py`
2. REST calls via `httpx` to Google Calendar API v3
3. Share Google OAuth connection with Sheets plugin (same token, different scopes)
4. Keep existing CalDAV plugin for Nextcloud/iCloud users

**Best-of-breed:** Direct REST via `httpx`. The CalDAV plugin stays for non-Google calendars.

---

### 5. PDF Processing Enhancement

**Zone:** 2 (Tools Service) — **already partially present**

**Why:** The tools-service container already has PyMuPDF, PyPDF2, pdfplumber, reportlab, and ocrmypdf. The missing piece is *exposing these as first-class gRPC tool handlers* that the orchestrator can call, and adding structured I/O contracts.

**Current state:** PDF libraries are installed (`backend/requirements.txt` lines 37–48) but there are no dedicated gRPC handlers for PDF operations beyond basic document ingestion.

**Tools to add (gRPC handlers in `grpc_tool_service.py`):**
- `ExtractPDFText` — Full text extraction with page numbers (PyMuPDF)
- `ExtractPDFTables` — Table extraction (pdfplumber)
- `CreatePDFFromMarkdown` — Markdown → styled PDF (reportlab)
- `MergePDFs` — Combine multiple PDFs
- `OCRDocument` — OCR a scanned PDF (ocrmypdf, already installed)
- `ExtractPDFMetadata` — Author, title, page count, etc.

**Orchestrator wrappers:** Thin gRPC wrappers in a new `llm-orchestrator/orchestrator/tools/pdf_tools.py` that call `backend_tool_client`. These wrappers use only `httpx`/gRPC (already in orchestrator deps) — no PDF libraries in the orchestrator.

**Implementation steps:**
1. Add gRPC handler methods to `grpc_tool_service.py` (or a new `pdf_tool_handlers.py` if it gets large)
2. Add proto definitions to `tool_service.proto`
3. Create `llm-orchestrator/orchestrator/tools/pdf_tools.py` (thin gRPC wrappers only)
4. Register with Action I/O Registry and tool pack registry
5. No new container deps — everything already installed

**Best-of-breed:** PyMuPDF for text/metadata extraction (fastest), pdfplumber for tables, reportlab for creation, ocrmypdf for OCR. All already in requirements.

---

### 6. Audio Transcription Enhancement

**Zone:** 2 (Tools Service)

**Why:** `TranscribeAudio` exists as a gRPC handler, but the implementation may rely on external APIs. Adding local Whisper transcription makes it work offline and free for users.

**Suggested enhancement:**
- `faster-whisper` for local GPU/CPU transcription (much faster than original Whisper)
- Speaker diarization via `pyannote.audio` (optional, heavier)
- Batch transcription support for long recordings

**Tools (gRPC handlers):**
- `TranscribeAudio` — Enhanced with model selection (tiny/base/small/medium)
- `TranscribeWithTimestamps` — Timestamped segments for subtitle generation
- `DetectLanguage` — Language detection from audio clip

**Deps to add to tools-service:**
- `faster-whisper>=1.0.0` (wraps CTranslate2, much lighter than full Whisper)
- System dep: may need `libcudnn` if GPU acceleration desired (or CPU-only mode)

**Implementation steps:**
1. Add `faster-whisper` to `backend/requirements.txt`
2. Potentially add system deps to `tools-service/Dockerfile` for audio codec support
3. Enhance existing `TranscribeAudio` handler or add new handlers
4. Orchestrator wrappers remain thin gRPC calls (no new orchestrator deps)

**Best-of-breed:** `faster-whisper` — 4x faster than OpenAI Whisper, lower memory, CTranslate2 backend. Falls back to CPU gracefully.

---

### 7. DOCX / Word Document Creation & Editing

**Zone:** 2 (Tools Service)

**Why:** Bastion agents produce markdown, but users need polished Word documents for reports, proposals, memos, and letters. Document format production is the single most common capability gap identified across both the Anthropic skills repo and the OpenClaw ecosystem. The [Anthropic DOCX skill](https://github.com/anthropics/skills/tree/main/skills/docx) provides a mature reference implementation.

**Two approaches (both should be supported):**

- **Create from structured content:** Generate DOCX programmatically with headings, tables, TOC, images, headers/footers, page numbers, and professional styling. Uses `python-docx` for Python-native creation.
- **Convert from Markdown:** Convert existing markdown output (which agents already produce) into styled DOCX. Uses `pandoc` for high-fidelity conversion with custom templates.

**Tools (gRPC handlers):**
- `CreateDocxFromMarkdown` — Convert markdown content to a styled DOCX file with configurable template (margins, fonts, headers/footers)
- `CreateDocxFromStructure` — Create DOCX from structured JSON (sections, tables, images) for when agents need precise layout control
- `EditDocx` — Unpack existing DOCX → edit XML → repack (for modifying uploaded documents)
- `DocxToMarkdown` — Extract markdown from uploaded DOCX for agent processing
- `DocxAddTrackedChanges` — Add tracked changes/comments to existing DOCX (for review workflows)

**Key patterns from Anthropic reference:**
- DOCX is a ZIP of XML files — editing means unpacking, modifying XML, repacking
- `python-docx` for creation, `pandoc` for markdown conversion, direct XML manipulation for advanced editing
- Tracked changes use OOXML `w:ins`/`w:del` elements with author attribution
- Always validate output with schema checks before returning to user

**Deps to add to tools-service:**
- `python-docx>=1.1.0` (DOCX creation/editing)
- `pandoc` system package (already commonly available; markdown → DOCX conversion)

**Implementation steps:**
1. Add `python-docx` to `backend/requirements.txt`
2. Add `pandoc` to `tools-service/Dockerfile` (apt install)
3. Create gRPC handlers in `grpc_tool_service.py` (or dedicated `docx_tool_handlers.py`)
4. Add proto definitions to `tool_service.proto`
5. Create `llm-orchestrator/orchestrator/tools/docx_tools.py` (thin gRPC wrappers)
6. Register with Action I/O Registry
7. Store generated documents via existing document service (returns `document_id` + download URL)

**Best-of-breed:** `python-docx` for programmatic creation (pure Python, no system deps), `pandoc` for markdown conversion (most faithful rendering). The Anthropic skill also uses `docx-js` (Node) for creation but Python-native is preferable for Bastion's stack.

---

### 8. XLSX / Spreadsheet Creation & Editing

**Zone:** 2 (Tools Service)

**Why:** The Data Workspace handles SQL queries, but users need formatted Excel output — financial models, data exports with charts, formatted reports. Same gap pattern as DOCX. The [Anthropic XLSX skill](https://github.com/anthropics/skills/tree/main/skills/xlsx) provides reference patterns.

**Critical design principle:** Always use Excel formulas, not hardcoded computed values. The spreadsheet must remain dynamic and updateable when source data changes.

**Tools (gRPC handlers):**
- `CreateXlsxFromData` — Create formatted XLSX from structured data (rows, columns, types, formulas, styling)
- `CreateXlsxFromDataWorkspace` — Export Data Workspace query results as a formatted XLSX with headers, column widths, and number formatting
- `EditXlsx` — Modify cells, add sheets, update formulas in existing XLSX
- `ReadXlsx` — Extract data and formulas from uploaded XLSX (returns structured JSON)
- `RecalculateXlsx` — Recalculate all formulas in an XLSX (requires LibreOffice)

**Key patterns from Anthropic reference:**
- `openpyxl` for creation/editing with formulas and formatting
- `pandas` for data-driven bulk operations
- LibreOffice headless for formula recalculation (formulas created by openpyxl are strings until recalculated)
- Financial model conventions: blue text for inputs, black for formulas, yellow background for assumptions
- Always use `WidthType.DXA` for table widths (percentages break in Google Docs)

**Deps to add to tools-service:**
- `openpyxl>=3.1.0` (XLSX creation/editing — may already be a transitive dep via pandas)
- LibreOffice headless (apt package, for formula recalculation)

**Implementation steps:**
1. Add `openpyxl` to `backend/requirements.txt` (if not already present)
2. Ensure LibreOffice headless is in `tools-service/Dockerfile`
3. Create gRPC handlers for XLSX operations
4. Add proto definitions
5. Create `llm-orchestrator/orchestrator/tools/xlsx_tools.py` (thin gRPC wrappers)
6. Special integration: `CreateXlsxFromDataWorkspace` calls Data Workspace query tools internally, then formats the result
7. Register with Action I/O Registry

**Best-of-breed:** `openpyxl` for formula-aware creation (the standard), `pandas` for data-driven exports. LibreOffice headless for recalculation. No lighter alternative exists for proper Excel formula support.

---

### 9. PPTX / Presentation Creation

**Zone:** 2 (Tools Service)

**Why:** Research agents produce detailed findings, but presenting them requires manual slide creation. Auto-generating presentations from research output is a high-value workflow: "Research X and create a presentation." The [Anthropic PPTX skill](https://github.com/anthropics/skills/tree/main/skills/pptx) provides design guidelines and tooling.

**Tools (gRPC handlers):**
- `CreatePptxFromOutline` — Generate a presentation from a structured outline (title, slides with content/layout/speaker notes)
- `CreatePptxFromMarkdown` — Convert markdown with headers into slides (each H1/H2 becomes a slide)
- `EditPptxSlide` — Modify content of specific slides in an existing PPTX
- `ReadPptxContent` — Extract text and speaker notes from all slides
- `PptxToImages` — Convert slides to images for preview/QA (LibreOffice → PDF → images)

**Key design principles from Anthropic reference:**
- Pick a bold, content-informed color palette (not default blue)
- Every slide needs a visual element — text-only slides are forgettable
- Vary layouts across slides: two-column, icon grids, stat callouts, timelines
- Large stat callouts (60-72pt numbers with small labels) for impact
- Never use accent lines under titles (hallmark of AI-generated slides)
- Font pairing matters: header font with personality + clean body font

**Deps to add to tools-service:**
- `python-pptx>=0.6.23` (PPTX creation/editing)
- LibreOffice headless (shared with XLSX — for PDF/image conversion)
- `poppler-utils` system package (for `pdftoppm` PDF → image conversion)

**Implementation steps:**
1. Add `python-pptx` to `backend/requirements.txt`
2. Ensure LibreOffice + poppler-utils in `tools-service/Dockerfile`
3. Create gRPC handlers for PPTX operations
4. Include a library of default slide layouts and color palettes as internal templates
5. Create `llm-orchestrator/orchestrator/tools/pptx_tools.py` (thin gRPC wrappers)
6. Register with Action I/O Registry
7. Research agent integration: add a "presentation" output format option to research skills

**Best-of-breed:** `python-pptx` for creation (the standard Python PPTX library). The Anthropic skill also uses `pptxgenjs` (Node) for scratch creation, but `python-pptx` is more natural for Bastion's Python stack.

---

### 10. Chart & Data Visualization Generation

**Zone:** 2 (Tools Service)

**Why:** Bastion can query data (Data Workspace, document search) but cannot produce visual charts. Agents describe data in text when a chart would be 10x more effective. Chart generation pairs naturally with Data Workspace queries and XLSX/PPTX creation.

**Tools (gRPC handlers):**
- `CreateChart` — Generate a chart image (PNG/SVG) from structured data and chart spec (type, labels, colors, title)
- `CreateChartFromQuery` — Execute a Data Workspace SQL query and render the results as a chart
- `ListChartTypes` — Return supported chart types with example configurations

**Supported chart types:**
- Bar / Grouped Bar / Stacked Bar
- Line / Multi-line
- Pie / Donut
- Scatter
- Area
- Histogram
- Heatmap
- Box plot

**Deps to add to tools-service:**
- `matplotlib>=3.8.0` (may already be a transitive dep)
- `seaborn>=0.13.0` (statistical visualization, built on matplotlib)
- Optional: `plotly>=5.18.0` for interactive HTML charts (export as static images)

**Implementation steps:**
1. Add `matplotlib` and `seaborn` to `backend/requirements.txt` (if not present)
2. Create chart generation handlers in `grpc_tool_service.py`
3. Charts are returned as base64-encoded images or saved as documents (reuse image storage)
4. Create `llm-orchestrator/orchestrator/tools/chart_tools.py` (thin gRPC wrappers)
5. Integration with Data Workspace: `CreateChartFromQuery` chains SQL execution → chart rendering
6. Integration with PPTX: charts can be embedded in generated presentations
7. Register with Action I/O Registry

**Best-of-breed:** `matplotlib` for static charts (most flexible, best for PDF/PPTX embedding), `seaborn` for statistical plots. `plotly` for interactive HTML output if needed later.

---

### 11. Mermaid Diagram Rendering

**Zone:** 2 (Tools Service)

**Why:** Research and project agents frequently need to express relationships, workflows, and architectures visually. Mermaid is a text-to-diagram language that LLMs can generate natively. Converting Mermaid syntax to SVG/PNG images fills the gap between text-based analysis and visual communication.

**Tools (gRPC handlers):**
- `RenderMermaidDiagram` — Convert Mermaid syntax to SVG or PNG image
- `RenderMermaidToDocument` — Render and save as a document in user's files

**Supported diagram types (all native to Mermaid):**
- Flowcharts, Sequence diagrams, Class diagrams, State diagrams
- Entity-relationship diagrams, Gantt charts, Pie charts
- Git graphs, Mindmaps, Timeline diagrams

**Deps to add to tools-service:**
- `mermaid-cli` (`mmdc`) via npm (or Puppeteer-based renderer)
- Alternative: Use the Mermaid.ink API (`https://mermaid.ink/img/...`) for zero-dep rendering (sends diagram text, gets back PNG)

**Implementation steps:**
1. Evaluate: Mermaid.ink API (no deps, external call) vs local `mmdc` (needs Node + Puppeteer in tools-service container)
2. Mermaid.ink API is simplest: `httpx.get(f"https://mermaid.ink/img/{base64_encoded_mermaid}")` — returns PNG
3. Create gRPC handler for diagram rendering
4. Create `llm-orchestrator/orchestrator/tools/diagram_tools.py` (thin gRPC wrappers)
5. Register with Action I/O Registry
6. Integration: research agents and project planning agents can generate Mermaid in their output and have it rendered automatically

**Best-of-breed:** Mermaid.ink API for zero-dep rendering. Fall back to local `mmdc` only if offline rendering is required.

---

### 12. Markdown → Office Format Converter

**Zone:** 2 (Tools Service)

**Why:** The lowest-friction path to document format support. Bastion agents already produce markdown. A converter turns existing markdown output into polished DOCX, PDF, or PPTX without agents needing to learn new output formats. Inspired by the `mxe` skill pattern from the OpenClaw ecosystem.

**Tools (gRPC handlers):**
- `ConvertMarkdownToDocx` — Markdown → styled DOCX with configurable template
- `ConvertMarkdownToPdf` — Markdown → styled PDF with headers/footers/page numbers
- `ConvertMarkdownToHtml` — Markdown → self-contained HTML with CSS styling
- `ConvertMarkdownToPptx` — Markdown → slides (H1/H2 headers become slide breaks)

**Implementation approach:**
- All conversions powered by `pandoc` with custom templates
- Templates stored in `tools-service/templates/` — DOCX reference template, LaTeX/HTML templates for PDF
- User-configurable: margins, fonts, header/footer text, logo placement
- The converter wraps existing tool outputs: any tool that produces markdown can have its output converted

**Deps:** `pandoc` (system package, shared with DOCX tools above). For PDF: `texlive-xetex` or `wkhtmltopdf` (pandoc PDF backends).

**Implementation steps:**
1. Ensure `pandoc` and a PDF backend are in `tools-service/Dockerfile`
2. Create default templates in `tools-service/templates/`
3. Create gRPC handlers in `grpc_tool_service.py`
4. Create `llm-orchestrator/orchestrator/tools/format_conversion_tools.py`
5. Register with Action I/O Registry
6. This is the quick-win entry point — can be implemented before the full DOCX/XLSX/PPTX creation tools

**Best-of-breed:** `pandoc` is the undisputed best tool for document format conversion. It handles markdown → DOCX/PDF/HTML/PPTX with high fidelity and supports custom templates.

---

## Composed Workflow Patterns

These aren't individual tools but valuable workflow compositions that leverage existing tools via Agent Factory playbooks or scheduled agents. Identified from recurring patterns across both skill ecosystems.

### Morning Briefing Playbook

**Composes:** Email tools (Zone 3) + Calendar plugin (Zone 4) + Todo tools (Zone 2) + Weather tools (Zone 1) + Notification tools

**Pattern:** Scheduled agent runs daily at a configured time. Gathers unread email summaries, today's calendar events, pending todos, and weather forecast. Synthesizes into a concise briefing. Delivers via notification channel (in-app, Telegram, email).

**Implementation:** Agent Factory scheduled agent with a playbook that chains:
1. `get_emails` (unread, last 12 hours)
2. `gcal_list_events` (today)
3. `list_org_todos` (pending, due today/overdue)
4. `get_weather` (user's location)
5. LLM synthesis step → formatted briefing
6. `send_channel_message` → deliver to user

### Doc Co-Authoring / Reader Testing Pattern

**Source:** [Anthropic doc-coauthoring skill](https://github.com/anthropics/skills/tree/main/skills/doc-coauthoring)

**Pattern:** A three-stage document creation workflow:
1. **Context Gathering** — Structured Q&A to extract all relevant knowledge from the user
2. **Refinement & Structure** — Section-by-section drafting with brainstorming, curation, and iterative editing
3. **Reader Testing** — Feed the finished document to a fresh LLM instance (no conversation context) and ask it questions about the content. If the fresh LLM misinterprets or can't answer, the document has blind spots.

**Value for Bastion:** The "Reader Testing" step is a novel QA pattern. After the proposal generation agent or writing assistant produces a document, a separate LLM call (no shared state) tests whether the document is self-contained and clear. This could be a validation subgraph node in document-producing agents.

### Email Triage & Classification

**Composes:** Email tools (Zone 3) + LLM classification (Zone 1) + Todo/notification tools

**Pattern:** Scheduled agent scans inbox periodically. LLM classifies each email by urgency and topic. High-urgency items become notifications. Action items become org-mode todos. Summaries are batched into a digest.

**Implementation:** Agent Factory scheduled agent with email access + org-capture tools.

---

## Tier 2 — Valuable Additions, Moderate Effort

### 13. Bookmark / Read-Later / Web Clipping

**Zone:** 2 (Tools Service)

**Why:** Users want to save web content for later research. The web crawling tools exist but there's no "save and organize" workflow — crawled content is ephemeral. This gives it persistence.

**Tools (gRPC handlers):**
- `SaveBookmark` — Save URL with auto-extracted title, description, content (uses existing Crawl4AI)
- `ListBookmarks` — List bookmarks with tag/date filters
- `SearchBookmarks` — Full-text search across saved bookmark content
- `TagBookmark` — Add/remove tags
- `GetBookmarkContent` — Retrieve full extracted content
- `DeleteBookmark` — Remove a bookmark

**Database:** New table `user_bookmarks` (user_id, url, title, description, content, tags[], saved_at, source).

**Implementation steps:**
1. SQL migration: `CREATE TABLE user_bookmarks (...)`
2. Add repository: `backend/repositories/bookmark_repository.py`
3. Add gRPC handlers to `grpc_tool_service.py`
4. Reuse existing `CrawlWebContent` for content extraction on save
5. Optional: auto-vectorize bookmark content for semantic search via existing Qdrant
6. Orchestrator wrappers in `llm-orchestrator/orchestrator/tools/bookmark_tools.py`

**Best-of-breed:** Leverage existing Crawl4AI and Qdrant infrastructure. No new deps needed.

---

### 14. Home Assistant Smart Home Control

**Zone:** 4 (Plugin)

**Why:** IoT/smart home control is a natural fit for a personal assistant. Home Assistant is the dominant open-source home automation platform. Its REST API is clean and requires only `httpx`.

**Tools:**
- `ha_list_devices` — List devices/entities with state
- `ha_get_state` — Get current state of a specific entity
- `ha_turn_on` / `ha_turn_off` — Control switches, lights, etc.
- `ha_set_climate` — Set thermostat temperature/mode
- `ha_trigger_automation` — Trigger a named automation
- `ha_call_service` — Generic service call (advanced)

**Auth:** Long-lived access token + Home Assistant URL.

**Implementation steps:**
1. Create `llm-orchestrator/orchestrator/plugins/integrations/homeassistant_plugin.py`
2. REST calls via `httpx` to HA API (`/api/states`, `/api/services/...`)
3. No extra deps — `httpx` is already available
4. Connection type: HA URL + long-lived token

**Best-of-breed:** Direct REST API via `httpx`. Home Assistant's REST API is simple and well-documented. No HA SDK needed.

---

### 15. Todoist Task Management

**Zone:** 4 (Plugin)

**Why:** Many users already have tasks in Todoist. Connecting it enables bidirectional workflows: agent creates Todoist tasks from research, or reads Todoist for daily planning. Complements org-mode (personal) with Todoist (shared/team).

**Tools:**
- `todoist_list_tasks` — List tasks with project/label/priority filters
- `todoist_create_task` — Create task with due date, priority, project, labels
- `todoist_complete_task` — Mark task complete
- `todoist_update_task` — Update task details
- `todoist_list_projects` — List projects
- `todoist_search_tasks` — Filter/search tasks

**Auth:** API token (Todoist REST API v2).

**Implementation steps:**
1. Create `llm-orchestrator/orchestrator/plugins/integrations/todoist_plugin.py`
2. REST calls via `httpx` to Todoist Sync/REST API v2
3. No extra deps
4. Could later build Agent Factory playbook for org-mode ↔ Todoist sync

**Best-of-breed:** Todoist REST API v2. Clean, well-documented, no SDK needed. TickTick could be a separate plugin later (similar API pattern).

---

### 16. IMAP/SMTP Generic Email

**Zone:** 3 (Connections Service)

**Why:** Microsoft Graph covers Outlook/365. Adding IMAP/SMTP covers Gmail, ProtonMail Bridge, Fastmail, and self-hosted email servers. Massively expands email support.

**Provider capabilities:**
- Read inbox, search messages, get threads (IMAP)
- Send and reply to emails (SMTP)
- Folder management
- Attachment handling

**Connections-service deps:**
- `aioimaplib>=1.0.1` (async IMAP)
- `aiosmtplib>=3.0.0` (async SMTP)

**Implementation steps:**
1. Add deps to `connections-service/requirements.txt`
2. Create `connections-service/providers/imap_provider.py` extending `BaseProvider` (not `BaseMessagingProvider` — email is not a chat channel)
3. Implement the same email interface as the Microsoft provider (same gRPC contract)
4. Connection config: IMAP host/port, SMTP host/port, username, password/app-password, TLS settings
5. Register in the connections service gRPC handlers
6. Backend email tools (`GetEmails`, `SearchEmails`, `SendEmail`) route to either Microsoft or IMAP provider based on connection type

**Best-of-breed:** `aioimaplib` for async IMAP (native asyncio), `aiosmtplib` for sending. Both lightweight and well-maintained.

---

### 17. YouTube Data & Transcripts

**Zone:** 2 (Tools Service)

**Why:** YouTube transcript extraction enables research agents to analyze video content. Transcripts are text — no video processing needed, just API/scraping.

**Tools (gRPC handlers):**
- `GetYouTubeTranscript` — Extract transcript/captions from a video URL or ID
- `SearchYouTubeVideos` — Search videos by query (YouTube Data API v3 or scraping)
- `GetVideoMetadata` — Title, channel, description, duration, view count

**Deps to add to tools-service:**
- `youtube-transcript-api>=0.6.0` (lightweight, no Google API key needed for transcripts)

**Implementation steps:**
1. Add `youtube-transcript-api` to `backend/requirements.txt`
2. Add gRPC handlers to `grpc_tool_service.py`
3. Create `llm-orchestrator/orchestrator/tools/youtube_tools.py` (thin gRPC wrappers)
4. Register with Action I/O Registry
5. Integrates with research workflow subgraph as a context source

**Best-of-breed:** `youtube-transcript-api` for transcripts (no API key, scrapes captions directly). For search/metadata, either YouTube Data API v3 (requires key, higher quality) or `httpx` scraping.

---

### 18. Readwise Highlight Sync

**Zone:** 4 (Plugin)

**Why:** Readwise aggregates highlights from Kindle, web, PDFs, podcasts, etc. Syncing highlights into Bastion's document system creates a rich personal knowledge base for research agents.

**Tools:**
- `readwise_list_books` — List books/sources with highlight counts
- `readwise_get_highlights` — Get highlights for a book/source
- `readwise_search_highlights` — Search across all highlights
- `readwise_export_highlights` — Export highlights as a formatted document
- `readwise_sync_to_documents` — Create/update Bastion documents from highlights (calls Zone 2 tools internally)

**Auth:** Readwise API token.

**Implementation steps:**
1. Create `llm-orchestrator/orchestrator/plugins/integrations/readwise_plugin.py`
2. REST calls via `httpx` to Readwise API v2 (`https://readwise.io/api/v2/...`)
3. No extra deps
4. The `sync_to_documents` tool would call document creation tools (Zone 2) via gRPC — making it a composed tool

**Best-of-breed:** Readwise API v2. Simple REST, well-documented. The Reader API (v3) is also available for saved articles.

---

## Tier 3 — Nice-to-Have, Lower Priority

### 19. Spotify Playback Control

**Zone:** 4 (Plugin)

**Why:** Music control is a natural personal assistant feature. Low effort, high "wow factor."

**Tools:** `spotify_now_playing`, `spotify_play_track`, `spotify_pause`, `spotify_search`, `spotify_queue_track`, `spotify_list_playlists`

**Auth:** Spotify OAuth2 (Web API).

**Implementation:** Plugin with `httpx` against Spotify Web API. No extra deps.

---

### 20. Hacker News & Reddit Monitoring

**Zone:** 2 (Tools Service — Connector Templates)

**Why:** HN connector template already exists. Reddit adds another valuable monitoring source. Both feed into Agent Factory monitor mode.

**Enhancement:**
- Enhance existing HN template with filtering/sorting options
- Add Reddit connector template (uses Reddit JSON API, no auth for public data)
- Both usable as monitor-mode data sources for scheduled agents

**Implementation:** New connector template definitions in `connector_templates.py`. Reddit's JSON API (`https://www.reddit.com/r/{sub}.json`) requires no auth for public reads.

---

### 21. Plex / Jellyfin Media Management

**Zone:** 4 (Plugin)

**Why:** Media server search and control for users who self-host media.

**Tools:** `media_search`, `media_now_playing`, `media_play`, `media_list_libraries`, `media_recently_added`

**Implementation:** Plugin with `httpx` against Plex or Jellyfin REST API. Both have clean APIs. Could be two separate plugins or a unified media plugin with provider detection.

---

### 22. Prometheus / Grafana Monitoring

**Zone:** 4 (Plugin)

**Why:** Server health monitoring for self-hosters. Agents can alert on anomalies or answer "how's the server doing?"

**Tools:** `prom_query` (PromQL), `prom_alerts`, `prom_targets_health`, `grafana_list_dashboards`, `grafana_get_panel_data`

**Implementation:** Plugin with `httpx` against Prometheus HTTP API and Grafana API. Both well-documented REST APIs.

---

### 23. Recipe / Meal Planning

**Zone:** 1 (Orchestrator — LLM-in-process) + 2 (Tools Service for storage)

**Why:** Meal planning with grocery list generation. Primarily an LLM task with storage.

**Approach:** This is more of an Agent Factory playbook than a tool. An agent with access to web search + document creation + org-mode todos can do meal planning today. A dedicated "recipe" data model could be added to documents if there's demand.

**Implementation:** No new tools needed — compose existing tools via Agent Factory workflow.

---

### 24. Bitwarden / Password Lookup

**Zone:** 4 (Plugin) — **requires careful security design**

**Why:** Agents that need to log into services on behalf of users need credential access.

**Security considerations:**
- Credentials should never be logged or stored in agent state
- Access should require explicit per-lookup user approval (HITL)
- Bitwarden CLI (`bw`) would need to be in a container — this makes it Zone 2, not Zone 4
- Alternatively, use Bitwarden's REST API (self-hosted Vaultwarden) via `httpx`

**Implementation:** Defer until there's a concrete use case. The security design needs to be right.

---

## Summary Table

| # | Skill | Zone | Container | New Deps | Effort | Priority |
|---|-------|------|-----------|----------|--------|----------|
| 1 | Slack | 3 + 4 | connections-service + orchestrator plugin | `slack-bolt` in conn-svc | Medium | Tier 1 |
| 2 | Google Sheets | 4 | orchestrator plugin | None (httpx) | Medium | Tier 1 |
| 3 | Linear | 4 | orchestrator plugin | None (httpx) | Low-Med | Tier 1 |
| 4 | Google Calendar | 4 | orchestrator plugin | None (httpx) | Low | Tier 1 |
| 5 | PDF Enhancement | 2 | tools-service | None (already installed) | Medium | Tier 1 |
| 6 | Audio Transcription | 2 | tools-service | `faster-whisper` | Medium | Tier 1 |
| 7 | **DOCX Creation** | 2 | tools-service | `python-docx`, `pandoc` (apt) | Medium | **Tier 1** |
| 8 | **XLSX Creation** | 2 | tools-service | `openpyxl`, LibreOffice (apt) | Medium | **Tier 1** |
| 9 | **PPTX Creation** | 2 | tools-service | `python-pptx`, LibreOffice (apt) | Medium | **Tier 1** |
| 10 | **Chart Generation** | 2 | tools-service | `matplotlib`, `seaborn` | Low-Med | **Tier 1** |
| 11 | **Mermaid Diagrams** | 2 | tools-service | None (Mermaid.ink API) or `mmdc` | Low | **Tier 1** |
| 12 | **Markdown → Office** | 2 | tools-service | `pandoc` (shared with #7) | Low | **Tier 1** |
| 13 | Bookmarks | 2 | tools-service | None (DB migration) | Low-Med | Tier 2 |
| 14 | Home Assistant | 4 | orchestrator plugin | None (httpx) | Low | Tier 2 |
| 15 | Todoist | 4 | orchestrator plugin | None (httpx) | Low | Tier 2 |
| 16 | IMAP/SMTP Email | 3 | connections-service | `aioimaplib`, `aiosmtplib` | Medium | Tier 2 |
| 17 | YouTube Transcripts | 2 | tools-service | `youtube-transcript-api` | Low | Tier 2 |
| 18 | Readwise | 4 | orchestrator plugin | None (httpx) | Low | Tier 2 |
| 19 | Spotify | 4 | orchestrator plugin | None (httpx) | Low | Tier 3 |
| 20 | HN + Reddit Monitor | 2 | tools-service (templates) | None | Low | Tier 3 |
| 21 | Plex/Jellyfin | 4 | orchestrator plugin | None (httpx) | Low | Tier 3 |
| 22 | Prometheus/Grafana | 4 | orchestrator plugin | None (httpx) | Low | Tier 3 |
| 23 | Meal Planning | — | Compose existing tools | None | Low | Tier 3 |
| 24 | Bitwarden | TBD | TBD (security design needed) | TBD | Medium | Tier 3 |

## Composed Workflows (No New Tools Required)

| Workflow | Composes | Trigger | Source |
|----------|----------|---------|--------|
| Morning Briefing | Email + Calendar + Todos + Weather | Scheduled agent | OpenClaw `morning-briefing`, `daily-briefing` |
| Doc Co-Authoring | Writing assistant + Reader Testing subgraph | User-initiated | Anthropic `doc-coauthoring` |
| Email Triage | Email + LLM classification + Todos + Notifications | Scheduled agent | OpenClaw `email-triage`, `email-daily-summary` |

## Dependency Impact by Container

### tools-service (Dockerfile + backend/requirements.txt)
- `faster-whisper>=1.0.0` (Tier 1 — audio transcription)
- `python-docx>=1.1.0` (Tier 1 — DOCX creation)
- `openpyxl>=3.1.0` (Tier 1 — XLSX creation, may already be transitive via pandas)
- `python-pptx>=0.6.23` (Tier 1 — PPTX creation)
- `matplotlib>=3.8.0` (Tier 1 — chart generation)
- `seaborn>=0.13.0` (Tier 1 — statistical charts)
- `youtube-transcript-api>=0.6.0` (Tier 2 — YouTube)
- System packages: `pandoc`, `libreoffice-nogui` (headless), `poppler-utils`, `texlive-xetex` (PDF backend)
- System deps for audio codecs if GPU transcription desired
- DB migration for bookmarks table

### connections-service (requirements.txt)
- `slack-bolt>=1.20.0` or `slack-sdk>=3.30.0` (Tier 1 — Slack bot)
- `aioimaplib>=1.0.1` (Tier 2 — IMAP email)
- `aiosmtplib>=3.0.0` (Tier 2 — SMTP email)

### llm-orchestrator
- **No new dependencies.** All plugins use `httpx` (already installed) for REST/GraphQL calls and `pydantic` (already installed) for I/O models. This is by design.

## Source Material

This roadmap was compiled from analysis of two external skill ecosystems, cross-referenced against Bastion's existing capabilities and architecture:

- **[Anthropic Skills Repository](https://github.com/anthropics/skills/tree/main/skills)** — Official skills including document format production (DOCX, XLSX, PDF, PPTX), doc co-authoring, web artifacts builder, canvas design, algorithmic art. The document format skills provided the most detailed reference implementations for items #7–#9 and #12.
- **[OpenClaw Skills Ecosystem](https://github.com/VoltAgent/awesome-openclaw-skills)** — 3,000+ community-contributed skills. The `mxe` (Markdown → Office) pattern inspired item #12. The `chart-image`, `beautiful-mermaid`, `morning-briefing`, `email-triage`, and `daily-briefing` skills confirmed value for items #10–#11 and the composed workflow patterns.
- **Gap analysis performed against:** Bastion's `file_creation_tools.py` (currently plain text only), `document_processor.py` (ingestion but not production), `image_generation_tools.py` (AI images but not charts/diagrams), and the Data Workspace (SQL queries but no visual output).

## Implementation Pattern for Zone 4 Plugins

All new plugins follow the established pattern in `llm-orchestrator/orchestrator/plugins/integrations/`:

```
class MyPlugin(BasePlugin):
    plugin_name = "my_service"
    plugin_version = "0.1.0"

    def get_tools(self) -> List[PluginToolSpec]:
        return [PluginToolSpec(
            name="my_tool",
            category="my_category",
            description="...",
            inputs_model=MyInputs,
            outputs_model=MyOutputs,   # MUST include 'formatted: str'
            tool_function=self._my_tool,
        )]

    def get_connection_requirements(self) -> Dict[str, str]:
        return {"api_key": "Service API Key", "base_url": "Service URL (optional)"}

    async def _my_tool(self, ...) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(...)
        return {"data": ..., "formatted": "Human-readable summary"}
```

All tools return a dict with typed fields + a `formatted` string field per the tool-io-contracts rule.

## What Was Explicitly Excluded

- Agent-to-agent social networks (Moltbook, MoltCasino, etc.)
- Crypto/blockchain/DeFi tools
- AI model routing (Bastion handles this internally via `_get_llm()`)
- Custom memory systems (Bastion has PostgreSQL checkpointing + Qdrant)
- Desktop automation (out of scope for Docker server)
- CRM systems (HubSpot, Salesforce) — too enterprise-specific
- Social media posting automation — adds complexity without core value
- Security scanning of third-party skills — not applicable to Bastion's plugin architecture
- Mobile app control (iOS Shortcuts, Android ADB)
- Gaming and entertainment (chess bots, virtual pets, etc.)
