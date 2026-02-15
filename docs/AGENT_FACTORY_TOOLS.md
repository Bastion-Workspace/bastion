# Agent Factory: Tool Catalog & Registry

**Document Version:** 1.0
**Last Updated:** February 14, 2026
**Companion to:** `AGENT_FACTORY.md`, `AGENT_FACTORY_TECHNICAL_GUIDE.md`, `AGENT_FACTORY_EXAMPLES.md`

---

## Purpose

This document catalogs every tool category available to Agent Factory custom agents, documents their I/O contracts, and defines the modular registry architecture that allows tools to be added, updated, and composed without code changes.

Tools come from three sources:
1. **Native tools** — Built into the orchestrator (`orchestrator/tools/`)
2. **Connector-generated tools** — Dynamically generated from connector YAML definitions
3. **Plugin tools** — Loaded at runtime from a plugin directory or external package

All tools are registered in the **Action I/O Registry** (see `AGENT_FACTORY_TECHNICAL_GUIDE.md`) which provides typed input/output contracts for the Workflow Composer UI.

---

## Tool Categories

### Overview

| Category | Tools | Status | Source |
|----------|-------|--------|--------|
| [File Operations](#1-file-operations) | 11 | Mostly exists | Native |
| [Search & Discovery](#2-search--discovery) | 6 | Exists | Native |
| [Task Management](#3-task-management) | 5 | Partial (org-mode) | Native |
| [Notifications & Messaging](#4-notifications--messaging) | 3 | Partial | Native + Connections Service |
| [Knowledge Graph](#5-knowledge-graph) | 5 | Documented in tech guide | Native |
| [Data Workspace](#6-data-workspace) | 4 | Exists | Native |
| [Text & Content Processing](#7-text--content-processing) | 5 | Exists | Native |
| [Web & Crawling](#8-web--crawling) | 3 | Exists | Native |
| [Email](#9-email) | 6 | Exists | Native (via connections) |
| [Monitor Detection](#10-monitor-detection) | 7 | New (Agent Factory) | Native |
| [Agent Internal](#11-agent-internal) | 7 | New (Agent Factory) | Native |
| [External Integrations](#12-external-integrations) | Variable | New (plugins) | Plugin / Connector |

---

## 1. File Operations

Granular document and file manipulation. These tools operate on the user's document library with scope-aware access control.

**Access Scope Rules:**

| Scope | Read | Write/Edit | Create | Delete/Move |
|-------|------|-----------|--------|-------------|
| **My Docs** (user's own) | Yes | Yes | Yes | Yes |
| **Team Docs** (shared team folders) | Yes (if team member) | Yes (if team_file_access) | Yes (if team_file_access) | Owner only |
| **Global Docs** (system-wide) | Yes | No | No | No |

### Existing Tools (wrapped for Agent Factory)

These tools already exist in the orchestrator. The Agent Factory exposes them with standardized I/O contracts.

```yaml
read_file:
  description: Read a document's full content by ID or path
  maps_to: get_document_content_tool  # Existing native tool
  inputs:
    document_id: string         # Document ID (required if no path)
    path: string                # Folder path + filename (required if no ID)
    scope: string               # "my_docs", "team_docs", "global_docs" (default: "my_docs")
    team_id: string             # Required when scope is "team_docs"
  params:
    max_length: integer         # Truncate if too long (optional)
    include_frontmatter: boolean # Include YAML frontmatter (default: true)
  outputs:
    content: string
    title: string
    file_type: string           # md, org, txt, pdf, etc.
    frontmatter: object         # Parsed frontmatter (if present)
    metadata: object            # created_at, modified_at, size, folder_path

find_file_by_name:
  description: Search for files by title or filename pattern
  maps_to: search_documents_tool (with title filter)
  inputs:
    name: string                # Filename or title to search for (required)
    scope: string               # "my_docs", "team_docs", "global_docs", "all" (default: "all")
    team_id: string             # Required when scope is "team_docs"
  params:
    exact_match: boolean        # Exact vs. fuzzy match (default: false)
    file_types: string[]        # Filter by extension
    folder_id: string           # Restrict to folder
  outputs:
    files:                      # type: file_ref[]
      fields: [document_id, title, filename, file_type, folder_path, scope, modified_at]
    count: integer

find_file_by_vectors:
  description: Semantic search for files by meaning (not exact text)
  maps_to: search_documents_tool
  inputs:
    query: string               # Natural language search query (required)
    scope: string               # "my_docs", "team_docs", "global_docs", "all" (default: "all")
    team_id: string             # Required when scope is "team_docs"
  params:
    max_results: integer        # Default: 10
    file_types: string[]        # Filter by extension
    folder_id: string           # Restrict to folder
    min_score: number           # Minimum similarity score (0-1)
  outputs:
    documents:                  # type: document[]
      fields: [document_id, title, snippet, score, folder_path, scope]
    count: integer

search_within_file:
  description: Search for specific text or patterns within a single document
  maps_to: search_within_document_tool
  inputs:
    document_id: string         # Document to search within (required)
    query: string               # Search query (required)
  params:
    context_lines: integer      # Lines of context around matches (default: 3)
  outputs:
    matches:                    # type: match[]
      fields: [line_number, content, context_before, context_after]
    count: integer

get_file_metadata:
  description: Get document metadata without reading full content
  maps_to: get_document_metadata_tool
  inputs:
    document_id: string         # Document ID (required)
  outputs:
    title: string
    file_type: string
    folder_path: string
    scope: string
    created_at: date
    modified_at: date
    size_bytes: integer
    frontmatter: object
    tags: string[]

create_file:
  description: Create a new document in user's or team's document library
  maps_to: create_document_tool / create_user_file_tool
  inputs:
    content: string             # File content (required)
    title: string               # Document title (required)
  params:
    folder_id: string           # Target folder (optional — uses root if omitted)
    file_type: string           # md, org, txt (default: "md")
    scope: string               # "my_docs" or "team_docs" (default: "my_docs")
    team_id: string             # Required when scope is "team_docs"
    frontmatter: object         # YAML frontmatter to prepend (optional)
    tags: string[]              # Tags to apply (optional)
  outputs:
    document_id: string
    title: string
    folder_path: string
  scope_restriction: no global_docs writes

patch_file:
  description: Edit a specific section of an existing document
  maps_to: propose_document_edit_tool / update_document_content_tool
  inputs:
    document_id: string         # Document to edit (required)
    edits: object[]             # List of edit operations (required)
    # Each edit: {operation, target, content}
    # Operations:
    #   "insert_after_heading" — Insert content below a heading
    #   "replace_range" — Replace text matching original_text
    #   "append" — Append to end of document
    #   "prepend" — Prepend to start of document (after frontmatter)
    #   "insert_at_line" — Insert at specific line number
  params:
    create_if_missing: boolean  # Create file if it doesn't exist (default: false)
  outputs:
    success: boolean
    operations_applied: integer
    new_content_length: integer
  scope_restriction: no global_docs writes

append_to_file:
  description: Append content to the end of an existing document
  maps_to: append_to_document_tool
  inputs:
    document_id: string         # Document to append to (required)
    content: string             # Content to append (required)
  params:
    separator: string           # Separator before appended content (default: "\n\n")
    heading: string             # Optional heading for the appended section
  outputs:
    success: boolean
    new_content_length: integer
  scope_restriction: no global_docs writes
```

### New Tools (to be built)

```yaml
delete_file:
  description: Delete a document from the library
  inputs:
    document_id: string         # Document to delete (required)
  params:
    confirm: boolean            # Require confirmation (default: true)
  outputs:
    success: boolean
    deleted_title: string
  scope_restriction: my_docs only; team_docs owner only

move_file:
  description: Move a document to a different folder
  inputs:
    document_id: string         # Document to move (required)
    target_folder_id: string    # Destination folder (required)
  outputs:
    success: boolean
    new_folder_path: string
  scope_restriction: my_docs and team_docs (if team_file_access)

copy_file:
  description: Copy a document to a different folder (creates a new document)
  inputs:
    document_id: string         # Document to copy (required)
    target_folder_id: string    # Destination folder (required)
  params:
    new_title: string           # Title for the copy (optional — defaults to "Copy of {title}")
  outputs:
    new_document_id: string
    new_title: string
    new_folder_path: string
  scope_restriction: source can be any scope; target must be my_docs or team_docs
```

### File Operations Tool Pack

```yaml
tool_pack: file_operations
tools:
  - read_file
  - find_file_by_name
  - find_file_by_vectors
  - search_within_file
  - get_file_metadata
  - create_file
  - patch_file
  - append_to_file
  - delete_file
  - move_file
  - copy_file
```

---

## 2. Search & Discovery

Tools for finding information across the user's document library, knowledge graph, and external sources.

```yaml
# All of these exist today as native tools

search_documents:
  maps_to: search_documents_tool
  # Semantic search across all documents
  # See find_file_by_vectors above for Agent Factory wrapper

search_segments:
  maps_to: search_segments_across_documents_tool
  inputs:
    query: string
  params:
    max_segments: integer
    document_ids: string[]      # Restrict to specific documents
  outputs:
    segments: segment[]         # [{document_id, title, content, score, line_range}]
    count: integer

search_knowledge_graph:
  # Documented in AGENT_FACTORY_TECHNICAL_GUIDE.md action I/O registry
  # Searches Neo4j for entities and relationships

search_web:
  maps_to: search_web_tool
  # Web search via SearXNG

crawl_url:
  maps_to: crawl_web_content_tool
  # Fetch and extract content from URLs

expand_query:
  maps_to: expand_query_tool
  inputs:
    query: string
  outputs:
    variations: string[]
    count: integer
```

---

## 3. Task Management

Tools for creating, managing, and querying tasks/todos.

The existing system uses **org-mode** as the native task format. These tools wrap org-mode operations with a task-oriented interface that Agent Factory playbooks can use.

### Existing Tools (wrapped)

```yaml
create_todo:
  description: Create a new todo item in the user's inbox
  maps_to: add_org_inbox_item_tool
  inputs:
    title: string               # Task title (required)
    body: string                # Task description/details (optional)
  params:
    priority: string            # "A", "B", "C" (optional)
    state: string               # "TODO", "NEXT", "WAITING" (default: "TODO")
    deadline: date              # Due date (optional)
    scheduled: date             # Scheduled date (optional)
    tags: string[]              # Tags (optional)
    target_file: string         # Org file to add to (default: inbox)
  outputs:
    success: boolean
    todo_id: string
    file_path: string

list_todos:
  description: List current todo items, filterable by state, tag, or date
  maps_to: list_org_todos_tool
  inputs:
    state: string               # Filter by state: "TODO", "DONE", "all" (optional)
  params:
    tags: string[]              # Filter by tags
    deadline_before: date       # Due before date
    file_id: string             # From specific file
    limit: integer              # Max results (default: 50)
  outputs:
    todos:                      # type: todo[]
      fields: [todo_id, title, state, priority, deadline, scheduled, tags, file_path]
    count: integer
```

### New Tools (to be built)

```yaml
update_todo:
  description: Update a todo item's state, priority, deadline, or content
  inputs:
    todo_id: string             # Todo to update (required)
  params:
    state: string               # New state (TODO, NEXT, STARTED, WAITING, DONE, CANCELED)
    priority: string            # New priority (A, B, C, or null to remove)
    deadline: date              # New deadline
    scheduled: date             # New scheduled date
    add_tags: string[]          # Tags to add
    remove_tags: string[]       # Tags to remove
    add_note: string            # Append a note/comment to the todo body
  outputs:
    success: boolean
    updated_fields: string[]

complete_todo:
  description: Mark a todo as DONE
  inputs:
    todo_id: string             # Todo to complete (required)
  params:
    completion_note: string     # Note about completion (optional)
  outputs:
    success: boolean
    completed_at: date

search_todos:
  description: Full-text search across all todo items
  inputs:
    query: string               # Search query (required)
  params:
    states: string[]            # Filter by states
    include_done: boolean       # Include completed items (default: false)
  outputs:
    todos: todo[]
    count: integer
```

### Task Management Tool Pack

```yaml
tool_pack: task_management
tools:
  - create_todo
  - list_todos
  - update_todo
  - complete_todo
  - search_todos
```

---

## 4. Notifications & Messaging

Tools for sending notifications to the user through in-app channels and external messaging platforms.

The connections-service already supports Telegram and Discord, with Slack, Mattermost, Signal, and Matrix planned. These tools abstract the messaging infrastructure so playbooks can send messages without knowing which provider is configured.

### New Tools (to be built)

```yaml
send_notification:
  description: Send an in-app notification to the user (appears in notification center)
  inputs:
    message: string             # Notification text (required)
  params:
    title: string               # Notification title (optional)
    priority: string            # "low", "normal", "high", "urgent" (default: "normal")
    action_url: string          # Deep link when notification is clicked (optional)
    action_label: string        # Label for the action (optional)
  outputs:
    notification_id: string
    success: boolean

send_channel_message:
  description: Send a message through a configured messaging channel (Telegram, Discord, Slack, etc.)
  inputs:
    message: string             # Message content (required)
  params:
    channel: string             # "telegram", "discord", "slack", "default" (default: user's preferred)
    chat_id: string             # Specific chat/channel ID (optional — uses default if omitted)
    format: string              # "markdown", "plain", "html" (default: "markdown")
    silent: boolean             # Send without notification sound (default: false)
  outputs:
    message_id: string
    channel: string             # Which channel was used
    success: boolean
  note: |
    Uses the connections-service provider infrastructure. The user must have
    the target channel configured in their external connections. Falls back to
    in-app notification if no channel is configured.

broadcast_to_team:
  description: Send a message to all members of a team via their preferred channels
  inputs:
    message: string             # Message content (required)
    team_id: string             # Team to broadcast to (required)
  params:
    channel: string             # Force specific channel or "preferred" (default: "preferred")
  outputs:
    delivered_to: integer       # Number of team members reached
    failures: integer           # Number of delivery failures
    success: boolean
  requires: team_post_access
```

### Notifications & Messaging Tool Pack

```yaml
tool_pack: notifications
tools:
  - send_notification
  - send_channel_message
  - broadcast_to_team
```

---

## 5. Knowledge Graph

Tools for interacting with the Neo4j knowledge graph. Documented in full in `AGENT_FACTORY_TECHNICAL_GUIDE.md` action I/O registry.

```yaml
tool_pack: knowledge_graph
tools:
  - search_knowledge_graph
  - extract_entities
  - resolve_entities
  - cross_reference
  - analyze_graph
```

---

## 6. Data Workspace

Tools for querying and managing Data Workspace tables.

```yaml
# All exist today as native tools

list_workspaces:
  maps_to: list_data_workspaces_tool
  outputs:
    workspaces: workspace[]     # [{id, name, database_count, created_at}]

get_schema:
  maps_to: get_workspace_schema_tool
  inputs:
    workspace_id: string
  outputs:
    tables: table_schema[]      # [{name, columns: [{name, type}], row_count}]

query_workspace:
  maps_to: query_data_workspace_tool
  inputs:
    query: string               # SQL or natural language query
    workspace_id: string
  outputs:
    rows: record[]
    columns: string[]
    row_count: integer

save_to_workspace:
  # Documented in tech guide action I/O registry
  inputs:
    data: record[]
  params:
    table_name: string
    create_if_missing: boolean
  outputs:
    rows_inserted: integer

tool_pack: data_workspace
tools:
  - list_workspaces
  - get_schema
  - query_workspace
  - save_to_workspace
```

---

## 7. Text & Content Processing

Tools for transforming, analyzing, and formatting text content. All exist as native tools.

```yaml
tool_pack: text_processing
tools:
  - summarize_text        # maps_to: summarize_text_tool
  - extract_structured    # maps_to: extract_structured_data_tool
  - transform_format      # maps_to: transform_format_tool (markdown ↔ org ↔ HTML ↔ plain)
  - merge_texts           # maps_to: merge_texts_tool
  - compare_texts         # maps_to: compare_texts_tool
```

---

## 8. Web & Crawling

```yaml
tool_pack: web
tools:
  - search_web            # maps_to: search_web_tool (SearXNG)
  - crawl_url             # maps_to: crawl_web_content_tool (Crawl4AI)
  - search_web_structured # maps_to: search_web_structured
```

---

## 9. Email

Tools for reading and sending email. Requires user to have email connection configured.

```yaml
# All exist today via connections-service + email_tools.py

tool_pack: email
tools:
  - get_emails            # maps_to: get_emails_tool (list inbox)
  - search_emails         # maps_to: search_emails_tool
  - get_email_thread      # maps_to: get_email_thread_tool
  - get_email_stats       # maps_to: get_email_statistics_tool
  - send_email            # maps_to: send_email_tool (with confirmation)
  - reply_to_email        # maps_to: reply_to_email_tool (with confirmation)
```

---

## 10. Monitor Detection

Tools for change-aware polling in monitor-mode agents. Documented in full in `AGENT_FACTORY_TECHNICAL_GUIDE.md` action I/O registry.

```yaml
tool_pack: monitor_detection
tools:
  - detect_new_files
  - detect_folder_changes
  - detect_new_data
  - detect_new_team_posts
  - detect_new_entities
  - set_monitor_watermark
  - get_monitor_watermark
```

---

## 11. Agent Internal

Tools automatically available to all custom agents for journaling and team interaction. Documented in full in `AGENT_FACTORY_TECHNICAL_GUIDE.md` action I/O registry.

```yaml
tool_pack: agent_journal
tools:
  - write_journal_entry
  - query_journal

tool_pack: team_interaction    # Conditionally available based on team_config
tools:
  - search_team_files
  - read_team_file
  - search_team_posts
  - write_team_post
  - summarize_team_thread
```

---

## 12. External Integrations (Plugins)

External integrations connect Agent Factory agents to third-party services. They can be implemented as:
- **Connector YAML definitions** (for REST/GraphQL APIs) — declared in the Agent Factory UI
- **Plugin tool packs** (for services needing complex logic) — installed as packages

### Trello

Task and project management. Inspired by [OpenClaw's Trello integration](https://clawhub.ai/steipete/trello).

```yaml
plugin: trello
connection_type: api_key       # Trello API key + token
registration: plugin           # Loaded from plugin directory

tools:
  list_boards:
    description: List all Trello boards accessible to the user
    inputs: {}
    outputs:
      boards: board[]           # [{id, name, url, member_count}]

  list_cards:
    description: List cards on a board or in a specific list
    inputs:
      board_id: string          # Board to list from (required)
    params:
      list_name: string         # Filter to specific list (optional)
      labels: string[]          # Filter by labels (optional)
      members: string[]         # Filter by assigned members (optional)
    outputs:
      cards: card[]             # [{id, name, description, list_name, labels, due_date, members, url}]
      count: integer

  create_card:
    description: Create a new Trello card
    inputs:
      name: string              # Card title (required)
      board_id: string          # Board (required)
      list_name: string         # List to place card in (required)
    params:
      description: string
      labels: string[]
      due_date: date
      members: string[]
      checklist: string[]       # Items for a checklist
    outputs:
      card_id: string
      url: string

  move_card:
    description: Move a card to a different list
    inputs:
      card_id: string           # Card to move (required)
      target_list: string       # Destination list name (required)
    outputs:
      success: boolean

  update_card:
    description: Update card fields (description, due date, labels, etc.)
    inputs:
      card_id: string           # Card to update (required)
    params:
      name: string
      description: string
      due_date: date
      labels: string[]
      add_comment: string
    outputs:
      success: boolean

  add_checklist_item:
    description: Add an item to a card's checklist
    inputs:
      card_id: string           # Card (required)
      item: string              # Checklist item text (required)
    params:
      checklist_name: string    # Which checklist (default: first)
      checked: boolean          # Already done? (default: false)
    outputs:
      success: boolean

tool_pack: trello
```

### Notion

Knowledge base and database management. Inspired by [OpenClaw's Notion integration](https://clawhub.ai/steipete/notion).

```yaml
plugin: notion
connection_type: oauth         # Notion OAuth integration
registration: plugin

tools:
  search_pages:
    description: Search Notion pages by title or content
    inputs:
      query: string             # Search query (required)
    params:
      page_type: string         # "page", "database", "all" (default: "all")
    outputs:
      pages: page[]             # [{id, title, url, type, parent, last_edited}]
      count: integer

  read_page:
    description: Read a Notion page's content
    inputs:
      page_id: string           # Page to read (required)
    outputs:
      title: string
      content: string           # Rendered as markdown
      properties: object        # Page properties
      children: object[]        # Child blocks

  create_page:
    description: Create a new Notion page
    inputs:
      title: string             # Page title (required)
      content: string           # Page content in markdown (required)
    params:
      parent_id: string         # Parent page or database (optional)
      properties: object        # Page properties (optional)
    outputs:
      page_id: string
      url: string

  update_page:
    description: Update a Notion page's content or properties
    inputs:
      page_id: string           # Page to update (required)
    params:
      title: string
      content: string           # Replace content
      append_content: string    # Append to existing content
      properties: object        # Update properties
    outputs:
      success: boolean

  query_database:
    description: Query a Notion database with filters and sorts
    inputs:
      database_id: string       # Database to query (required)
    params:
      filter: object            # Notion filter object
      sorts: object[]           # Sort rules
      limit: integer            # Max results
    outputs:
      rows: record[]
      count: integer

  create_database_entry:
    description: Add a new row to a Notion database
    inputs:
      database_id: string       # Database (required)
      properties: object        # Row data (required — keys match database columns)
    outputs:
      page_id: string
      url: string

tool_pack: notion
```

### Slack

Team communication. Uses the existing connections-service provider architecture.

```yaml
plugin: slack
connection_type: oauth         # Slack Bot OAuth
registration: provider         # Extends connections-service BaseMessagingProvider

tools:
  send_slack_message:
    description: Send a message to a Slack channel or DM
    inputs:
      message: string           # Message content (required)
      channel: string           # Channel name or ID (required)
    params:
      thread_ts: string         # Reply in thread (optional)
      blocks: object[]          # Slack Block Kit blocks (optional)
    outputs:
      message_ts: string
      channel_id: string
      success: boolean

  read_channel:
    description: Read recent messages from a Slack channel
    inputs:
      channel: string           # Channel name or ID (required)
    params:
      limit: integer            # Max messages (default: 20)
      since: date               # Messages after date (optional)
    outputs:
      messages: message[]       # [{ts, user, text, thread_ts, reactions}]
      count: integer

  list_channels:
    description: List accessible Slack channels
    inputs: {}
    params:
      types: string[]           # "public", "private", "dm" (default: ["public"])
    outputs:
      channels: channel[]       # [{id, name, topic, member_count, is_member}]
      count: integer

tool_pack: slack
```

### CalDAV (Calendar)

Calendar management for any CalDAV-compatible service (Google Calendar, Nextcloud, iCloud, etc.).

```yaml
plugin: caldav
connection_type: caldav_credentials  # URL + username + password (or OAuth)
registration: plugin

tools:
  list_events:
    description: List calendar events in a date range
    inputs:
      date_from: date           # Start date (required)
      date_to: date             # End date (required)
    params:
      calendar_name: string     # Specific calendar (optional — defaults to primary)
    outputs:
      events: event[]           # [{id, title, start, end, location, description, attendees, recurrence}]
      count: integer

  create_event:
    description: Create a calendar event
    inputs:
      title: string             # Event title (required)
      start: datetime           # Start time (required)
      end: datetime             # End time (required)
    params:
      location: string
      description: string
      attendees: string[]       # Email addresses
      reminder_minutes: integer # Reminder before event
      calendar_name: string
    outputs:
      event_id: string
      success: boolean

  update_event:
    description: Update an existing calendar event
    inputs:
      event_id: string          # Event to update (required)
    params:
      title: string
      start: datetime
      end: datetime
      location: string
      description: string
    outputs:
      success: boolean

  delete_event:
    description: Delete a calendar event
    inputs:
      event_id: string          # Event to delete (required)
    outputs:
      success: boolean

tool_pack: caldav
```

### Additional Integrations (Planned)

| Integration | Type | Priority | Key Tools |
|-------------|------|----------|-----------|
| **Mattermost** | provider | Phase 2 | send_message, read_channel, list_channels |
| **Signal** | provider | Phase 3 | send_message, read_messages |
| **Matrix** | provider | Phase 4 | send_message, read_room, list_rooms |
| **GitHub** | plugin | Future | list_issues, create_issue, list_prs, create_pr, read_file |
| **Jira** | plugin | Future | list_issues, create_issue, transition_issue, add_comment |
| **Google Sheets** | plugin | Future | read_sheet, write_sheet, create_sheet |
| **Airtable** | plugin | Future | query_base, create_record, update_record |

---

## Modular Tool Registry

### Architecture

The tool registry has three layers, checked in order:

```
Tool Resolution Order:
  1. Agent Profile tools (connector-generated + explicitly assigned)
  2. Plugin tools (dynamically loaded from plugin directory)
  3. Native tool packs (built-in orchestrator tools)
  4. Dynamic registry (user-defined via Agent Factory UI)
```

### Plugin Registration

Plugins are self-contained tool packages that register themselves with the action I/O registry on load. Each plugin provides:

1. **Tool functions** — Async Python functions with typed signatures
2. **I/O contracts** — YAML or Pydantic definitions of inputs, params, outputs
3. **Connection requirements** — What credentials/config the plugin needs
4. **Tool pack definition** — Name and list of tools for assignment to agents

```python
# Example: plugin registration interface
# llm-orchestrator/orchestrator/plugins/base_plugin.py

class BaseToolPlugin:
    """Base class for external integration plugins."""

    plugin_name: str            # "trello", "notion", "slack", etc.
    plugin_version: str
    connection_type: str        # "api_key", "oauth", "caldav_credentials", etc.
    required_config: List[str]  # Config keys needed from user's connection

    def get_tools(self) -> List[ToolDefinition]:
        """Return list of tool definitions with I/O contracts."""
        ...

    def get_tool_pack(self) -> ToolPack:
        """Return the tool pack for this plugin."""
        ...

    async def initialize(self, connection_config: Dict) -> None:
        """Initialize the plugin with user's connection credentials."""
        ...
```

### Dynamic Tool Loading

```python
# llm-orchestrator/orchestrator/plugins/plugin_loader.py

class PluginLoader:
    """
    Discovers and loads tool plugins at startup and on-demand.

    Plugins are discovered from:
    1. Built-in plugins directory: orchestrator/plugins/integrations/
    2. Installed packages with entry point: bastion.plugins
    3. User-uploaded plugin packages (future)
    """

    async def load_all_plugins(self) -> Dict[str, BaseToolPlugin]:
        """Load all available plugins and register their tools."""
        ...

    async def load_plugin(self, plugin_name: str, config: Dict) -> BaseToolPlugin:
        """Load a specific plugin with user's connection config."""
        ...

    async def register_with_io_registry(self, plugin: BaseToolPlugin) -> None:
        """Register plugin's tool I/O contracts in the action registry."""
        ...
```

### How It Works at Runtime

1. **Agent profile is loaded** — `prepare_context_node` reads the profile's assigned tool packs
2. **Native tools resolved** — Built-in tool packs are resolved from `orchestrator/tools/`
3. **Connector tools generated** — Connector YAML definitions generate tool functions
4. **Plugin tools loaded** — Plugin tool packs are loaded and initialized with user's connection config
5. **I/O contracts merged** — All tool contracts are merged into a single registry for the Workflow Composer
6. **Tools bound to agent** — Final tool list is assembled and bound to the LangGraph workflow

### Adding a New Integration

To add a new external integration (e.g., Jira):

1. Create `orchestrator/plugins/integrations/jira_plugin.py`
2. Extend `BaseToolPlugin`
3. Define tool functions with typed signatures
4. Define I/O contracts (inputs, params, outputs)
5. Register the plugin's entry point
6. The plugin appears automatically in the Agent Factory UI's tool browser
7. Users can assign the Jira tool pack to their agents

No orchestrator code changes needed. No registry updates needed. The plugin system discovers and registers tools automatically.

---

## Summary: Tool Inventory

| Category | Existing | New | Total |
|----------|---------|-----|-------|
| File Operations | 8 | 3 | 11 |
| Search & Discovery | 6 | 0 | 6 |
| Task Management | 2 | 3 | 5 |
| Notifications & Messaging | 0 | 3 | 3 |
| Knowledge Graph | 0* | 5 | 5 |
| Data Workspace | 4 | 0 | 4 |
| Text & Content | 5 | 0 | 5 |
| Web & Crawling | 3 | 0 | 3 |
| Email | 6 | 0 | 6 |
| Monitor Detection | 0 | 7 | 7 |
| Agent Internal | 0 | 7 | 7 |
| Trello | 0 | 6 | 6 |
| Notion | 0 | 6 | 6 |
| Slack | 0 | 3 | 3 |
| CalDAV | 0 | 4 | 4 |
| **Total** | **34** | **47** | **81** |

*Knowledge graph tools exist conceptually but need Agent Factory wrappers with I/O contracts.

New tools to build: 47 (across native tools and plugin integrations)
Existing tools to wrap: 34 (add I/O contracts for the Workflow Composer)
