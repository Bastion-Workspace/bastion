"""
Built-in skill definitions for Agent Factory dynamic skill discovery.

Pure data module -- no logic, no imports. Each dict is a skill definition
keyed by slug. The list is consumed by ``seed_builtin_skills()`` in
``agent_skills_service.py``.

Built-in skills default to ``is_core=True`` (always shown in the condensed
skill catalog).  Set ``"is_core": False`` explicitly on entries that should
only be discoverable via vector search.
"""

BUILTIN_SKILL_DEFINITIONS: list[dict] = [
    # ---- Automation ----
    {
        "slug": "weather",
        "name": "Weather",
        "description": "Get current weather conditions, forecasts, and historical weather data for any location.",
        "category": "automation",
        "procedure": (
            "You are a weather assistant. ALWAYS use get_weather to fetch real weather data - never make up weather information. "
            "Parameters: location (required - city name, ZIP code, or place name like 'New York' or '14532'), "
            "data_types (comma-separated: 'current' for now, 'forecast' for upcoming days, 'history' for past weather; default 'current'), "
            "date_str (for history only: 'YYYY-MM-DD' for a specific day, 'YYYY-MM' for monthly average, or 'YYYY-MM to YYYY-MM' for date ranges up to 24 months). "
            "If the user does not specify a location, ask them for one before calling the tool. "
            "Present temperatures in both Fahrenheit and Celsius. Include practical recommendations based on conditions."
        ),
        "required_tools": ["get_weather"],
        "optional_tools": [],
        "tags": ["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": [],
    },
    {
        "slug": "email",
        "name": "Email",
        "description": "Read, search, send, draft, and manage email. Folder browsing, threading, move, mark read/flagged, and connection routing.",
        "category": "email",
        "procedure": (
            "You are an email assistant. Use the available tools for all operations. "
            "list_email_folders: discover folder names (inbox, sent, drafts, etc.) before listing. "
            "list_emails: folder (inbox/sent/drafts), top (limit), unread_only (bool). search_emails: query, top. "
            "read_email: message_id for full body and headers. get_email_thread: conversation_id from a previous list. get_email_statistics: inbox/unread counts. "
            "For sending: ALWAYS call send_email with confirmed=False first to show the draft; only call with confirmed=True after the user explicitly approves (e.g. yes, send, approve). "
            "For replying: ALWAYS call reply_to_email with confirmed=False first to show the draft; only call with confirmed=True after the user approves. "
            "send_email: to (comma-separated), subject, body, confirmed (bool, default False). "
            "reply_to_email: message_id from thread/list, body, reply_all (bool), confirmed (bool, default False). "
            "create_draft: to, subject, body — saves a draft without sending. "
            "move_email: message_id, destination_folder. update_email: message_id, mark read/unread/flagged. "
            "When replying, use the same connection_id as the source message. "
            "For new threads, use the connection_id from the agent's configured email binding. Include clear subject lines and avoid stripping reply threading headers."
        ),
        "required_tools": [
            "list_emails", "search_emails", "get_email_thread", "read_email",
            "get_email_statistics", "send_email", "reply_to_email",
            "create_draft", "move_email", "update_email", "list_email_folders",
        ],
        "optional_tools": [],
        "tags": ["email", "inbox", "mail", "send email", "reply", "read my email", "check email", "search email", "unread", "draft", "compose", "move email", "folders"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["email"],
    },
    {
        "slug": "calendar",
        "name": "Calendar",
        "description": "List calendars, view events in a date range, get event details, create/update/delete events (O365).",
        "category": "calendar",
        "procedure": (
            "You are a calendar assistant. Use the available tools for all operations. "
            "list_calendars: list user's calendars. get_calendar_events: start_datetime and end_datetime required (ISO 8601); optional calendar_id, top. "
            "get_event_by_id: event_id from a previous list. "
            "For create_event: ALWAYS call with confirmed=False first to show the draft; only call with confirmed=True after the user explicitly approves. "
            "For update_event and delete_event: ALWAYS call with confirmed=False first; only call with confirmed=True after the user approves. "
            "create_event: subject, start_datetime, end_datetime (ISO 8601), confirmed (bool), optional location, body, attendee_emails (comma-separated), is_all_day. "
            "update_event: event_id, confirmed (bool), optional subject, start_datetime, end_datetime, location, body, attendee_emails, is_all_day. delete_event: event_id, confirmed (bool)."
        ),
        "required_tools": ["list_calendars", "get_calendar_events", "get_event_by_id", "create_event", "update_event", "delete_event"],
        "optional_tools": [],
        "tags": ["calendar", "schedule", "meeting", "event", "appointment", "create event", "delete event", "update event"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["calendar"],
    },
    {
        "slug": "contacts",
        "name": "Contacts",
        "description": "List, get, create, update, and delete contacts (O365 and org-mode unified when include_org).",
        "category": "contacts",
        "procedure": (
            "You are a contacts assistant. Use the available tools for all operations. "
            "list_contacts: list contacts (O365 + org-mode when include_org=True); optional folder_id, top. "
            "get_contact_by_id: get a single contact by contact_id (from a previous list). "
            "create_contact: display_name, given_name, surname, optional company_name, job_title, birthday, notes, email_addresses, phone_numbers. "
            "update_contact: contact_id (required), optional display_name, given_name, surname, company_name, job_title, birthday, notes. delete_contact: contact_id (required)."
        ),
        "required_tools": ["list_contacts", "get_contact_by_id", "create_contact", "update_contact", "delete_contact"],
        "optional_tools": [],
        "tags": ["contact", "contacts", "people", "address book", "look up contact", "create contact", "add contact", "update contact", "delete contact"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["email"],
    },
    {
        "slug": "m365-todo",
        "name": "Microsoft To Do",
        "description": "List, create, update, and delete tasks in Microsoft To Do (Microsoft 365 / Graph), not org-mode files.",
        "category": "productivity",
        "procedure": (
            "You manage Microsoft To Do via Graph. Use tools for all operations; do not confuse with org-mode todos (different skill). "
            "Flow: list_todo_lists first; for get_todo_tasks / create / update / delete always pass the list's id from that response (not the display name). "
            "If the user says 'flagged emails' you may use list_id 'flagged' as a shortcut. "
            "create_todo_task: list_id, title; optional body, due_datetime (ISO 8601), importance (low, normal, high). "
            "update_todo_task: list_id, task_id; optional title, body, status, due_datetime, importance. "
            "delete_todo_task: list_id, task_id."
        ),
        "required_tools": [
            "list_todo_lists",
            "get_todo_tasks",
            "create_todo_task",
            "update_todo_task",
            "delete_todo_task",
        ],
        "optional_tools": [],
        "tags": [
            "microsoft to do",
            "ms to do",
            "to do app",
            "graph to do",
            "outlook tasks",
            "my tasks microsoft",
            "list tasks to do",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["todo"],
    },
    {
        "slug": "m365-onedrive",
        "name": "OneDrive",
        "description": "Browse, search, read, upload, and manage files and folders in Microsoft OneDrive.",
        "category": "productivity",
        "procedure": (
            "You manage OneDrive via Graph. Use tools for all operations. "
            "list_drive_items: optional parent_item_id (empty for root), top. get_drive_item: item_id. "
            "search_drive: query, optional top. get_onedrive_file_content: item_id (returns base64; suitable for small files). "
            "upload_onedrive_file: name, content_base64, optional parent_item_id, mime_type. "
            "create_drive_folder: name, optional parent_item_id. move_drive_item: item_id, new_parent_item_id. delete_drive_item: item_id. "
            "List or search before reading or moving items so you have correct ids."
        ),
        "required_tools": [
            "list_drive_items",
            "get_drive_item",
            "search_drive",
            "get_onedrive_file_content",
            "upload_onedrive_file",
            "create_drive_folder",
            "move_drive_item",
            "delete_drive_item",
        ],
        "optional_tools": [],
        "tags": [
            "onedrive",
            "one drive",
            "microsoft files",
            "cloud files",
            "upload to onedrive",
            "search my onedrive",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["files"],
    },
    {
        "slug": "m365-onenote",
        "name": "OneNote",
        "description": "List OneNote notebooks, sections, and pages; read page HTML; create pages.",
        "category": "notes",
        "procedure": (
            "You work with OneNote via Graph. Traverse top-down: list_onenote_notebooks, then list_onenote_sections with notebook_id, "
            "then list_onenote_pages with section_id (optional top). get_onenote_page_content: page_id. "
            "create_onenote_page: section_id, html (HTML body), optional title."
        ),
        "required_tools": [
            "list_onenote_notebooks",
            "list_onenote_sections",
            "list_onenote_pages",
            "get_onenote_page_content",
            "create_onenote_page",
        ],
        "optional_tools": [],
        "tags": [
            "onenote",
            "one note",
            "microsoft notes",
            "notebook",
            "onenote page",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["onenote"],
    },
    {
        "slug": "m365-planner",
        "name": "Microsoft Planner",
        "description": "List plans and tasks in Microsoft Planner; create, update, and delete planner tasks.",
        "category": "productivity",
        "procedure": (
            "You manage Microsoft Planner via Graph (team/task boards, not Microsoft To Do lists). "
            "list_planner_plans first for plan ids, then get_planner_tasks with plan_id. "
            "create_planner_task: plan_id, title; optional bucket_id. "
            "update_planner_task: task_id; optional title, percent_complete (0-100), due_datetime. "
            "delete_planner_task: task_id."
        ),
        "required_tools": [
            "list_planner_plans",
            "get_planner_tasks",
            "create_planner_task",
            "update_planner_task",
            "delete_planner_task",
        ],
        "optional_tools": [],
        "tags": [
            "microsoft planner",
            "planner",
            "planner tasks",
            "office planner",
            "team tasks planner",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["planner"],
    },
    {
        "slug": "navigation",
        "name": "Navigation",
        "description": "Manage saved locations, plan routes, and get directions.",
        "category": "navigation",
        "procedure": (
            "You are a navigation assistant. Multi-step flow: (1) list_locations to see saved locations and their IDs; create_location to add a new one (name, address). "
            "(2) compute_route with from_location_id and to_location_id (from list_locations), or use coordinates string. profile: driving, walking, cycling. "
            "save_route to save a computed route (pass waypoints, geometry, steps, distance_meters, duration_seconds from compute_route). "
            "list_saved_routes to list saved routes. delete_location to remove a location by location_id."
        ),
        "required_tools": ["create_location", "list_locations", "delete_location", "compute_route", "save_route", "list_saved_routes"],
        "optional_tools": [],
        "tags": ["create location", "list locations", "saved locations", "delete location", "route", "directions", "navigate", "map", "turn by turn", "save route"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "rss",
        "name": "RSS",
        "description": "Subscribe to, read, search, star, and manage RSS feeds and articles. Unread counts, starring, mark read/unread, feed pause/resume.",
        "category": "rss",
        "procedure": (
            "You are an RSS feed assistant. "
            "list_rss_feeds: scope 'user' or 'global'; includes unread_count per feed. "
            "add_rss_feed: feed_url (required), feed_name, category, is_global (bool). delete_rss_feed: feed_id. "
            "refresh_rss_feed: feed_name or feed_id to trigger refresh. toggle_feed_active: feed_id to pause/resume polling. "
            "list_rss_articles: feed_id, optional unread_only (bool), starred_only (bool), limit. search_rss: query across all feeds. "
            "list_starred_rss_articles: starred articles across all feeds. get_unread_counts: per-feed unread summary. "
            "mark_article_read: article_id. mark_article_unread: article_id. set_article_starred: article_id, starred (bool). "
            "Use list_rss_feeds first to discover feed IDs, then list_rss_articles to browse articles within a feed."
        ),
        "required_tools": [
            "add_rss_feed", "list_rss_feeds", "refresh_rss_feed",
            "list_rss_articles", "search_rss", "list_starred_rss_articles",
            "delete_rss_feed", "mark_article_read", "mark_article_unread",
            "set_article_starred", "get_unread_counts", "toggle_feed_active",
        ],
        "optional_tools": [],
        "tags": ["rss", "feed", "feeds", "subscribe", "news", "articles", "unread", "starred", "read articles", "mark read"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Org-mode ----
    {
        "slug": "org-capture",
        "name": "Org Capture",
        "description": "Quick capture items to the org-mode inbox. Use for 'capture to inbox', 'add to inbox', 'quick capture'.",
        "category": "org",
        "procedure": (
            "You help capture items to the user's org-mode inbox. "
            "ALWAYS call add_org_inbox_item to add each item - never just describe what you would do. "
            "\n\n"
            "SCOPE: Only capture items requested in the CURRENT user message (the last message). "
            "You may see conversation history for context — it helps interpret references like 'capture that' "
            "or 'add what you just said'. NEVER independently scan history for additional items to capture. "
            "If the current message does not ask to capture something, do not capture anything from history."
            "\n\n"
            "PRIOR STEP CONTENT: When the user message includes 'Content to capture (from a previous step):', "
            "that content is the main text to capture. Use it as the inbox item: use a concise title/summary for the heading "
            "if the content is long, and put the full content or a clear summary in the text parameter. "
            "Do not capture only the 'User request' line; capture the prior-step content. "
            "\n\n"
            "ORG FORMATTING: When capturing research, articles, or multi-paragraph content, convert the content to org-mode "
            "syntax so the inbox stays neat and readable. Output a single text value: first line = short title (used as the "
            "headline), then a blank line, then the body in org format. Conversion rules: "
            "Markdown ## H2 → ** H2, ### H3 → *** H3 (subheadings in body). "
            "Markdown **bold** → *bold*, *italic* → /italic/. "
            "Tables: use org table form | col1 | col2 | with a |-| separator row after the header. "
            "Lists: use - for unordered, 1. 2. for ordered; indent continuation lines with 2 spaces. "
            "Links: use [[url][description]] or plain URL. "
            "Keep paragraphs separated by blank lines; do not leave raw markdown (e.g. ## or **) in the captured text. "
            "\n\n"
            "CALL RULES: Call add_org_inbox_item exactly once per distinct inbox entry. "
            "One user-provided item → one tool call. Multiple items in one message → one tool call per item. "
            "Never call the tool multiple times with the same text for the same logical entry. "
            "\n\n"
            "REPLY RULES: After calling the tool, reply with a short confirmation only (e.g. 'Added X to your inbox.'). "
            "Do not mention deduplication, 'you provided it twice', 'I captured it just once', or how many times the tool was called. "
            "\n\n"
            "KIND: Use kind='todo' for explicit tasks or short action items (e.g. 'remind me to X', '* TODO X'). "
            "Use kind='note' for longer-form notes, ideas, or when the user says 'note' or 'capture this' without implying a task. "
            "Use kind='checkbox' for '- [ ] Item' or checklist items; kind='event' for calendar; kind='contact' for contacts. "
            "Org headlines: '* TODO Title' or '* Title' (note); extract title as text. Strip prefixes like 'capture to inbox:', 'add to inbox:'. "
            "\n\n"
            "TAGS: If the user asks to tag the item (e.g. 'tag with work', 'with tag urgent', 'add tag @project'), "
            "pass those tags in the tags parameter (comma-separated string or list). Never put :tag: in the title text — "
            "always use the tags parameter so tags are stored correctly and not duplicated. "
            "\n\n"
            "Parameters: text (required; title only, no tags or priority in the text), kind ('todo', 'note', 'checkbox', 'event', 'contact'; default 'todo'), "
            "schedule (optional org timestamp like '<2026-02-05 Thu>'), tags (optional; use when user specifies tags)."
        ),
        "required_tools": ["add_org_inbox_item"],
        "optional_tools": [],
        "tags": ["capture", "inbox", "capture to inbox", "add to inbox", "quick capture"],
        "evidence_metadata": {"engine_type": "automation", "stateless": True},
    },
    {
        "slug": "org-journal",
        "name": "Org Journal",
        "description": "Read and search the user's org-mode journal (diary entries). Use for 'what did I write', 'read my journal', 'search journal'.",
        "category": "org",
        "procedure": (
            "You help the user read and search their org-mode journal. ALWAYS call the appropriate tool - never make up journal content. "
            "get_journal_entry for a single date (date='today' or YYYY-MM-DD). get_journal_entries for a date range with full content. "
            "list_journal_entries for counts or metadata only. search_journal for keyword search. "
            "When the user does not specify a date, use 'today' for get_journal_entry. For ranges, use start_date and end_date in YYYY-MM-DD format."
        ),
        "required_tools": ["get_journal_entry", "get_journal_entries", "list_journal_entries", "search_journal"],
        "optional_tools": [],
        "tags": ["journal", "journal entry", "read my journal", "search my journal", "diary", "journal entries"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "org-content",
        "name": "Org Content",
        "description": "Parse and query the org file that is currently open in the editor. Only works when an org file is active in the editor pane.",
        "category": "org",
        "procedure": (
            "You help the user query their org-mode file. The full file content is included in the user message between === FILE ... === and === END FILE === markers. READ IT CAREFULLY before answering. "
            "You have tools: parse_org_structure to get the full outline; search_org_headings with search_term to find headings or content; get_org_statistics for counts and completion rate. "
            "When the user says 'today', use the YYYY-MM-DD date from the datetime context as search_term. "
            "Base your answers ONLY on the actual file content provided. Never fabricate data. Read-only; do not modify the file."
        ),
        "required_tools": ["parse_org_structure", "search_org_headings", "get_org_statistics"],
        "optional_tools": [],
        "tags": ["org file", "org structure", "open file", "file statistics", "headings", "outline", "org-mode editor"],
        "evidence_metadata": {"engine_type": "automation", "requires_editor": True, "editor_types": ["org"], "editor_preference": "require", "context_boost": 15},
    },
    {
        "slug": "task-management",
        "name": "Task Management",
        "description": "List, view, and manage todos across all org files and inbox without requiring an editor open. Use for 'show my todos', 'what are my tasks', create/update/toggle/delete/refile/archive.",
        "category": "org",
        "procedure": (
            "You help manage the user's todos across all org files in Bastion (org-mode), not Microsoft To Do or Microsoft Planner—those use separate skills. "
            "Org-mode todos: Put tags in the tags parameter (or add_tags/remove_tags for update_todo). "
            "Put priority in the priority parameter (A, B, or C). Do not embed :tag: or [#A] in the title text. "
            "Apply TODO updates (state, tags, priority) with update_todo or toggle_todo only - do not use propose_document_edit or patch_file; the todo API applies directly. Effort and category are not settable via these tools. "
            "list_todos: scope 'all', 'inbox', or a file path; optional states, tags, query, limit (0 = return all matches; use a positive limit only to cap huge lists). Results include file_path, line_number (0-based), heading, todo_state, tags, scheduled, deadline. "
            "create_todo: text = title only (required). Use tags parameter for tags and priority parameter for A/B/C. Optional: body, deadline, scheduled, file_path, insert_after_line_number. "
            "update_todo: file_path, line_number (0-based). Optional: new_state, new_text, add_tags/remove_tags, priority, scheduled, deadline, new_body. "
            "toggle_todo: file_path, line_number to toggle TODO <-> DONE. delete_todo: file_path, line_number. "
            "refile_todo: move a todo to another file or heading. Provide file_path + line_number of the todo, target_file_path and optionally target_heading. "
            "discover_refile_targets: find valid refile destinations (files and headings) for a todo. "
            "archive_done: single entry (file_path, line_number) or bulk (omit line_number, provide file_path or omit for inbox). Use list_todos first when the user asks to see or manage todos."
        ),
        "required_tools": [
            "list_todos", "create_todo", "update_todo", "toggle_todo",
            "delete_todo", "archive_done", "refile_todo", "discover_refile_targets",
        ],
        "optional_tools": [],
        "tags": ["todo", "todos", "task", "tasks", "list my todos", "show my todos", "how are my todos", "what are my todos", "create todo", "add todo", "mark done", "toggle todo", "update todo", "delete todo", "archive done", "inbox", "org", "refile", "move todo"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Agent / memory ----
    {
        "slug": "agent-run-history",
        "name": "Agent Run History",
        "description": "Report on this agent's or another agent's recent run history, status, and performance.",
        "category": "agent",
        "procedure": (
            "You report on agent execution (run) history. ALWAYS call get_agent_run_history to fetch real data - never make up run counts, dates, or statuses. "
            "When the user asks about 'your' history or 'this agent', omit agent_profile_id so the tool returns this agent's runs. "
            "Parameters: agent_profile_id (omit for self), limit (default 10, max 50), status ('completed', 'failed', 'running' to filter), start_date and end_date (YYYY-MM-DD). "
            "Summarize patterns: e.g. 'You have run 47 times this month with a 94% success rate.' Use status='failed' when the user asks about failures."
        ),
        "required_tools": ["get_agent_run_history"],
        "optional_tools": [],
        "tags": ["last run", "run history", "when did you last run", "agent log", "execution history", "recent runs"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "save-user-fact",
        "name": "Save User Fact",
        "description": "Save a fact about the user to their persistent fact store for future conversations.",
        "category": "memory",
        "procedure": (
            "You help users store facts about themselves for future conversations. When the user shares information they want remembered "
            "(e.g. 'remember I'm a vegetarian', 'save that I prefer Python'), ALWAYS call save_user_fact to store it. "
            "fact_key: short snake_case label (e.g. job_title, preferred_language, dietary_restriction, city). value: the actual information. "
            "category: 'work', 'preferences', 'personal', or 'general'. After saving, confirm naturally (e.g. 'Got it, I'll remember that [value].')."
        ),
        "required_tools": ["save_user_fact"],
        "optional_tools": [],
        "tags": ["remember", "save fact", "note that", "store preference", "remember that", "save that"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Image ----
    {
        "slug": "image-generation",
        "name": "Image Generation",
        "description": "Generate images from text descriptions. Use for 'draw', 'create image', 'make a picture'.",
        "category": "image",
        "procedure": (
            "You are an image generation assistant. Use generate_image: prompt (required), size (e.g. 1024x1024, 512x512), num_images (1-4), optional negative_prompt, model. "
            "Set check_reference_first=True to return a reference image from the user's library if one exists for the prompt/object name before generating."
        ),
        "required_tools": ["generate_image"],
        "optional_tools": [],
        "tags": ["create image", "generate image", "draw", "visualize", "image", "picture", "photo", "create a picture", "make an image"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "image-processing",
        "name": "Image Processing",
        "description": "Convert images between formats and optimize for size. Optional: render Graphviz DOT source to PNG, SVG, or PDF as a new document.",
        "category": "image",
        "procedure": (
            "You help process and convert raster images. "
            "convert_image: convert between formats (PNG/JPEG/WebP/GIF/BMP); input document_id + target_format, optional resize. "
            "optimize_image: reduce image file size while preserving quality (document_id, optional quality 1-100). "
            "Optional render_diagram: only for Graphviz DOT source (not Mermaid or PlantUML); input dot_content + output_format (png/svg/pdf). "
            "For Mermaid or other diagram text shown in chat, use create_artifact (artifact_diagrams / artifact_generation skills), not this skill."
        ),
        "required_tools": ["convert_image", "optimize_image"],
        "optional_tools": ["render_diagram"],
        "tags": ["convert image", "resize image", "optimize image", "graphviz", "dot"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Reference / calculation ----
    {
        "slug": "reference",
        "name": "Reference",
        "description": "Calculate, convert units, and visualize data from a reference file currently open in the editor. Only works when a reference-type file is active in the editor pane.",
        "category": "reference",
        "procedure": (
            "You help with reference documents, journals, and logs. The file content is included in the user message between === FILE ... === and === END FILE === markers. "
            "Read it carefully and base your answers on the actual content. You can run calculations (calculate_expression, evaluate_formula), convert units (convert_units), and create interactive chart artifacts from file data (create_chart — the chart appears in the artifact panel). "
            "Use the available tools. Never fabricate data that is not in the file."
        ),
        "required_tools": ["calculate_expression", "evaluate_formula", "convert_units", "create_chart"],
        "optional_tools": [],
        "tags": ["reference file", "open file", "calculation", "visualize", "chart", "graph", "editor file", "convert units"],
        "evidence_metadata": {"engine_type": "automation", "requires_editor": True, "editor_types": ["reference"], "context_boost": 20},
    },
    # ---- Documents ----
    {
        "slug": "document-creator",
        "name": "Document Creator",
        "description": "Create new files or documents in a folder (not for editing the current open document; use editor skills for that).",
        "category": "documents",
        "procedure": (
            "You are a document creation assistant. You create files and folders in the user's document tree. "
            "WORKFLOW: (1) If the user specifies a folder by name, call list_folders first to find its folder_id or verify it exists. "
            "Optionally call find_document_by_path with the intended filename or path (e.g. 'Reference/notes.md') to check whether a file already exists before creating. "
            "(2) If the folder does not exist, call create_user_folder to create it first. "
            "(3) Call create_typed_document with content, filename, and folder_path (e.g. 'Reference'). "
            "PARAMETERS for create_typed_document: filename (with extension), content (document body - clean markdown with a title heading), folder_path (auto-create if missing). "
            "If context from a prior step is available (prior_step_*_response keys), use it as the document body. If the user doesn't specify a filename, generate a descriptive one from the content."
        ),
        "required_tools": ["list_folders", "create_typed_document", "create_user_folder"],
        "optional_tools": ["find_document_by_path"],
        "tags": ["create file", "create document", "save to folder", "new file", "put in folder", "reference file", "save as"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "document-reading",
        "name": "Document Reading",
        "description": "Read documents by ID or path, search within a document, and cross-document segment search. Use when you already know which file to open; to discover documents by topic in the library, use document-search first.",
        "category": "documents",
        "procedure": (
            "You help read and search within documents the user or team already has. "
            "If you only have a topic and no document_id, run document-search first to find candidates. "
            "get_document_content: document_id (from search results or known ID) to read full content. "
            "find_document_by_path: filename or path (e.g. 'project-plan.md', 'Reference/notes.md') to resolve document_id. "
            "search_within_document: document_id + query to find specific text within one document. "
            "search_segments_across_documents: query across multiple documents for relevant passages. "
            "Typical flow: use find_document_by_path if you have a name, then get_document_content to read it. "
            "Use search_within_document when looking for specific content in a known file."
        ),
        "required_tools": ["get_document_content", "find_document_by_path", "search_within_document", "search_segments_across_documents"],
        "optional_tools": [],
        "tags": ["read document", "open file", "get content", "find document", "search in document", "read file", "document content"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "file-management",
        "name": "File Management",
        "description": "Create files and folders, and patch or append to existing files. Use for saving outputs and organizing content.",
        "category": "documents",
        "procedure": (
            "You help create and organize files. list_folders to see folder structure. create_user_folder to create a folder. "
            "create_typed_document: use for new documents with a filename, content, and folder_path (creates folder if missing). "
            "create_user_file: use for raw files (e.g. plain text, CSV) with filename, content, and folder_path. "
            "Before patching an existing file: use find_document_by_path if you have a filename or path, then get_document_content to read current content. "
            "patch_file: apply edits to an existing file. Each edit in 'edits' MUST include 'operation' (one of: replace, delete, insert_after_heading, append). "
            "Optional 'section': heading line that scopes the edit (e.g. '## Chapter 3', '## Personality') when the file repeats the same sub-headings in multiple places. "
            "For replace/delete, also include 'target' with verbatim text from the document (2-3 surrounding context lines). "
            "For replace, insert_after_heading, and append, include 'content' with the new text. append_to_file: add content to the end of a file. "
            "Always confirm the target path or folder with the user when it is ambiguous."
        ),
        "required_tools": [
            "create_typed_document", "create_user_file", "create_user_folder", "list_folders",
            "find_document_by_path", "get_document_content", "patch_file", "append_to_file",
        ],
        "optional_tools": [],
        "tags": [
            "create file",
            "create folder",
            "save file",
            "append to file",
            "patch file",
            "file management",
            "organize files",
            "edit document",
            "update document",
            "add to document",
            "modify file",
            "write to file",
            "put into document",
            "edit file",
        ],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "document-processing",
        "name": "Document Processing",
        "description": "Convert documents between formats, OCR images, extract text from PDFs, split/merge/compress PDFs.",
        "category": "documents",
        "procedure": (
            "You help process documents and PDFs. "
            "convert_document: convert between formats (docx/html/md/txt to PDF, etc.); input document_id + target_format. "
            "ocr_image: extract text from an image (document_id of the image). "
            "extract_pdf_text: extract all text from a PDF (document_id). "
            "split_pdf: split a PDF into individual pages or page ranges (document_id, page_ranges). "
            "merge_pdfs: combine multiple PDFs into one (list of document_ids). "
            "render_pdf_pages: render specific PDF pages as images (document_id, pages). "
            "compress_pdf: reduce PDF file size (document_id). "
            "convert_pdfa: convert a PDF to PDF/A for archival compliance (document_id)."
        ),
        "required_tools": [
            "convert_document", "ocr_image", "extract_pdf_text",
            "split_pdf", "merge_pdfs", "render_pdf_pages",
            "compress_pdf", "convert_pdfa",
        ],
        "optional_tools": [],
        "tags": ["convert to PDF", "OCR", "extract text", "merge PDFs", "split PDF", "compress PDF", "document conversion", "PDF"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Search / research ----
    {
        "slug": "document-search",
        "name": "Document Search",
        "description": "Search the user's and team's knowledge base: documents and segments (uploads, library files, shared content). Use FIRST for user content, reference PDFs in the library, and 'do we have' before web search. For how the Bastion product works, use help-docs, not this skill.",
        "category": "search",
        "procedure": (
            "Comprehensive local document and image search. The search_documents tool runs hybrid semantic (vector) plus keyword full-text matching by default, so natural-language and exact-term queries both work. "
            "Start with limit=10, scope=my_docs. If fewer than 3 relevant results, call enhance_query and retry with the enhanced query. "
            "Escalate to team_docs or global scope if needed. "
            "Use file_types filter when the user asks for a specific document type. Covers factual lookups ('what is X'), 'do we have' queries, and image/comic/photo collection searches. "
            "Search results include short content previews only; to read the full body of a found document, call get_document_content with the document_id from the result."
        ),
        "required_tools": ["search_documents", "enhance_query"],
        "optional_tools": ["get_document_content"],
        "tags": ["search", "find", "document", "look up", "do we have", "find me", "in our collection", "comic", "photo", "image", "knowledge base"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "web-search",
        "name": "Web Search",
        "description": "Multi-query web search with synthesis and source citation. Use when you must discover pages online (no usable URLs yet). If the user already gave one or more http(s) page links, prefer the web-crawl skill instead—do not run search_web first just to rediscover those same URLs.",
        "category": "search",
        "procedure": (
            "Multi-query web search with synthesis and source citation. "
            "When the user message is primarily explicit http(s) URLs and the task is to read or summarize those pages, load the web-crawl skill and crawl directly; skip search_web unless you need additional pages beyond the links provided. "
            "STRATEGY: Formulate all 2-4 focused search queries UPFRONT, run them, review results across all queries, "
            "then select the best pages to crawl. Avoid search-crawl-search-crawl loops. "
            "Use search_web with limit 10-15. For key results needing full content, use crawl_web_content with exact URLs. "
            "IMPORTANT: Do NOT crawl URLs ending in .pdf, .doc, .docx, or other binary formats -- the crawler cannot extract content from these. Prefer HTML pages. "
            "When crawling multiple pages, pass them as a list in the `urls` parameter of a single crawl_web_content call instead of one call per URL. "
            "Synthesize findings across all queries and crawled pages; cite sources (title, URL). Note when sources disagree or when information is uncertain or missing. "
            "Run follow-up searches only to fill specific gaps. For fact-checking and verification, cross-reference multiple sources."
        ),
        "required_tools": ["search_web", "crawl_web_content"],
        "optional_tools": [],
        "tags": ["web search", "search the web", "look up online", "find online", "research", "fact check", "verify", "what is", "how to", "investigate"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "security-analysis",
        "name": "Security Analysis",
        "description": "Security and vulnerability scanning for URLs and websites.",
        "category": "research",
        "procedure": (
            "Security and vulnerability scanning for URLs and websites. Use search_web to find security-related information and crawl_web_content to fetch and analyze page content, headers, and exposed resources. "
            "Report on security headers, exposed files, and common vulnerability indicators. Do not perform active exploitation."
        ),
        "required_tools": ["search_web", "crawl_web_content"],
        "optional_tools": [],
        "tags": ["security scan", "vulnerability scan", "security analysis", "check for vulnerabilities", "security audit", "pen test", "security assessment", "website security"],
        "evidence_metadata": {"engine_type": "research"},
    },
    {
        "slug": "web-crawl",
        "name": "Web Crawl",
        "description": "Fetch and extract content from specific page URLs the user provides (paste, cite, or list). Best when the message already contains http(s) links; use web-search when you still need a search engine to find relevant pages.",
        "category": "research",
        "procedure": (
            "Fetch page content by URL. If the user pasted one or more http(s) links, call crawl_web_content immediately with url or urls (batch multiple links in one call). "
            "Do not call search_web first when those links are already the targets. "
            "For a single URL, call crawl_web_content directly. "
            "For deeper site ingestion (multiple pages, recursive crawl), use search_web to discover additional page URLs within the domain, then crawl each. "
            "Use for one-off content extraction, site scraping, or downloading website content for processing or storage."
        ),
        "required_tools": ["crawl_web_content"],
        "optional_tools": ["search_web"],
        "tags": [
            "crawl",
            "crawl site",
            "crawl website",
            "ingest",
            "scrape",
            "download website",
            "url",
            "domain crawl",
            "capture website",
            "read this link",
            "summarize this url",
            "pasted url",
            "fetch page",
        ],
        "evidence_metadata": {"engine_type": "research"},
    },
    # ---- Knowledge graph ----
    {
        "slug": "knowledge-graph-traversal",
        "name": "Knowledge Graph Traversal",
        "description": "Search entities by name or type in the knowledge graph, and explore relationship patterns between them. Use for 'what is connected to X', 'find entities related to Y', or entity lookups.",
        "category": "knowledge-graph",
        "procedure": (
            "Search entities by name or type first. Use relationship depth to expand connections. "
            "When resolving identities across sources, prefer entity_search then follow relationship types. "
            "Limit relationship depth to 2 unless the user asks for deep traversal."
        ),
        "required_tools": ["search_knowledge_graph", "get_entity_details"],
        "optional_tools": [],
        "tags": ["knowledge graph", "entities", "relationships", "entity search", "graph traversal", "connections", "linked data"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Zettelkasten ----
    {
        "slug": "zettelkasten-traversal",
        "name": "Zettelkasten Note Traversal",
        "description": (
            "Navigate the note link graph for the open markdown document. "
            "Find what the current note links to, load linked note content, "
            "and traverse the note graph up to depth 2. "
            "Use for 'what does this note connect to', 'load my linked notes', "
            "'find all notes about X that I have linked here', or 'build context from my ZK'."
        ),
        "category": "knowledge-graph",
        "procedure": (
            "Prefer the link graph when you have a document_id (e.g. from editor_document_id or a prior tool result).\n"
            "Step 0: Call get_document_links with that document_id; use direction='outgoing' for notes this file links to, "
            "direction='incoming' for backlinks (what links TO this note), or direction='both'. "
            "Optional link_types filter e.g. ['wikilink'] to focus on Zettelkasten wikilinks.\n"
            "The prompt variable editor_linked_notes lists [[Title]] wikilinks parsed from the open file — use as a hint when the graph is stale or for unresolved titles.\n"
            "When you need fuzzy discovery or editor_linked_notes titles without graph rows, call search_documents with the title (limit 1–3) then get_document_content.\n"
            "After resolving neighbours, call get_document_content for each document_id you need to read.\n"
            "Limit graph traversal to depth 2 unless the user explicitly asks for deeper. Summarise the note neighbourhood before answering.\n"
            "If editor_linked_notes is empty and the user has no open note, use search_documents on the user's query."
        ),
        "required_tools": ["search_documents", "get_document_content", "get_document_links"],
        "optional_tools": ["search_within_document"],
        "tags": [
            "zettelkasten",
            "wikilinks",
            "backlinks",
            "note graph",
            "linked notes",
            "note neighbourhood",
            "knowledge base",
            "note traversal",
            "my notes",
            "connected notes",
            "network of notes",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "is_core": True,
    },
    # ---- Data ----
    {
        "slug": "data-workspace",
        "name": "Data Workspace",
        "description": "Query and inspect tabular data in Data Workspaces. List workspaces, get schema, run SQL or natural language queries.",
        "category": "data",
        "procedure": (
            "You help the user query tabular data in Data Workspaces. "
            "First call list_data_workspaces to see available workspaces and their IDs. "
            "Use get_workspace_schema with workspace_id to inspect tables and columns. "
            "Then use query_data_workspace with workspace_id and either a SQL query or a natural language question. "
            "Present results in a clear table or summary. For large result sets, summarize or paginate."
        ),
        "required_tools": ["query_data_workspace", "list_data_workspaces", "get_workspace_schema"],
        "optional_tools": [],
        "tags": ["data workspace", "query data", "sql", "tabular", "dataset", "spreadsheet", "table data"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "data-management",
        "name": "Data Management",
        "description": "Create tables and insert, update, or delete rows in Data Workspaces. For querying, use the Data Workspace skill.",
        "category": "data",
        "procedure": (
            "You help manage tabular data in Data Workspaces. "
            "list_data_workspaces: discover available workspaces and their IDs. get_workspace_schema: inspect tables and columns. "
            "create_workspace_table: workspace_id, table_name, columns (list of {name, type, nullable, default}). "
            "insert_workspace_rows: workspace_id, table_name, rows (list of row dicts). "
            "update_workspace_rows: workspace_id, table_name, updates (list of {filter, set} dicts). "
            "delete_workspace_rows: workspace_id, table_name, filter (WHERE condition dict). "
            "Always check get_workspace_schema before modifying to understand the current structure."
        ),
        "required_tools": [
            "create_workspace_table", "insert_workspace_rows",
            "update_workspace_rows", "delete_workspace_rows",
            "list_data_workspaces", "get_workspace_schema",
        ],
        "optional_tools": [],
        "tags": ["create table", "insert data", "import data", "update rows", "delete rows", "data management", "data entry"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Messaging / notifications ----
    {
        "slug": "notifications",
        "name": "Notifications",
        "description": "Send messages and schedule reminders via in-app, Telegram, Discord, or email. High-level notify_user respects user channel preferences.",
        "category": "messaging",
        "procedure": (
            "You help send notifications and reminders. "
            "notify_user: high-level send that respects the user's configured channel preferences (recommended for most cases). "
            "send_channel_message: send to a specific channel_id or channel type (in_app, telegram, discord); provide subject (optional) and body. "
            "schedule_reminder: schedule a future notification; provide message, remind_at (ISO 8601 datetime or relative like '2h', '30m', 'tomorrow 9am'). "
            "Confirm with the user before sending if the request is ambiguous. After sending or scheduling, confirm delivery briefly."
        ),
        "required_tools": ["notify_user", "send_channel_message", "schedule_reminder"],
        "optional_tools": [],
        "tags": ["notify", "notification", "send message", "alert", "telegram", "discord", "in-app message", "remind me", "set reminder", "reminder", "notify me later"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Text transforms ----
    {
        "slug": "text-transforms",
        "name": "Text Transforms",
        "description": "Summarize, extract structured data, convert formats, merge, and compare texts.",
        "category": "text",
        "procedure": (
            "You help transform text. "
            "summarize_text: text + optional max_length, style (brief/detailed/bullets). "
            "extract_structured_data: text + schema description to extract JSON/tables from prose. "
            "transform_format: text + target_format (markdown/json/csv/yaml/html) to convert between formats. "
            "merge_texts: list of texts + optional separator/strategy to combine multiple texts into one. "
            "compare_texts: text_a + text_b to find differences and similarities."
        ),
        "required_tools": ["summarize_text", "extract_structured_data", "transform_format", "merge_texts", "compare_texts"],
        "optional_tools": [],
        "tags": ["summarize", "extract data", "convert format", "merge texts", "compare", "diff", "reformat", "transform"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Planning ----
    {
        "slug": "planning",
        "name": "Planning",
        "description": "Create and manage multi-step plans. Track progress, adapt steps, and mark completion.",
        "category": "automation",
        "procedure": (
            "You help create and manage plans. "
            "create_plan: title (required) + steps (list of step descriptions). Creates a tracked plan. "
            "get_plan: retrieve current plan status and step states. "
            "update_plan_step: step_index + status (done/failed/skipped) + optional notes. "
            "add_plan_step: description to append a new step to the current plan. "
            "Typical flow: create_plan with the high-level steps, work through each step, update_plan_step as you go."
        ),
        "required_tools": ["create_plan", "get_plan", "update_plan_step", "add_plan_step"],
        "optional_tools": [],
        "tags": ["plan", "make a plan", "break down", "step by step", "project plan", "organize", "outline steps"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Browser automation ----
    {
        "slug": "browser-automation",
        "name": "Browser Automation",
        "description": "Control a browser: navigate, click, fill forms, extract content, take screenshots.",
        "category": "automation",
        "procedure": (
            "You help automate browser tasks. browser_navigate: open a URL. browser_click: click an element (selector or coordinates). "
            "browser_fill: fill form fields by selector. browser_extract: extract text or attributes from the page. browser_screenshot: capture a screenshot. "
            "Use browser_inspect when you need to discover selectors or page structure. For multi-step flows, navigate first, then click or fill as needed. "
            "Confirm destructive or high-impact actions with the user when appropriate."
        ),
        "required_tools": ["browser_navigate", "browser_click", "browser_fill", "browser_extract", "browser_screenshot"],
        "optional_tools": [],
        "tags": ["browser", "automation", "navigate", "click", "fill form", "screenshot", "scrape", "extract content"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Local device / code workspace ----
    {
        "slug": "local-device",
        "name": "Local Device",
        "description": "Interact with the user's local machine via Bastion Local Proxy: screenshots, clipboard, shell, files, processes, and more.",
        "category": "automation",
        "procedure": (
            "You interact with the user's local machine through the Bastion Local Proxy. "
            "local_screenshot: capture the current screen. local_clipboard_read / local_clipboard_write: read or set clipboard content. "
            "local_system_info: OS, CPU, memory, disk info. local_desktop_notify: show a desktop notification. "
            "local_shell_execute: run a shell command (use with care; confirm destructive commands with the user). "
            "local_read_file / local_write_file / local_patch_file: read, overwrite, or search-replace in local files. local_list_directory: list directory contents. "
            "local_list_processes: list running processes. local_open_url: open a URL in the default browser. "
            "Always confirm potentially destructive operations (write, shell) with the user before executing."
        ),
        "required_tools": [
            "local_screenshot", "local_clipboard_read", "local_clipboard_write",
            "local_system_info", "local_desktop_notify", "local_shell_execute",
            "local_read_file", "local_list_directory", "local_write_file", "local_patch_file",
            "local_list_processes", "local_open_url",
        ],
        "optional_tools": [],
        "tags": ["screenshot", "clipboard", "system info", "run command", "shell", "local file", "processes", "desktop", "open url"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "code-workspace",
        "name": "Code Workspace",
        "description": "Browse, search, index, and edit code in local workspaces via Bastion Local Proxy; supports multiple saved workspaces per user.",
        "category": "code",
        "procedure": (
            "You help the user work with local code workspaces via Bastion Local Proxy. "
            "ALWAYS start by calling code_list_workspaces to list saved workspaces (id, name, path, device_id). "
            "If there are none, tell the user to create one in the Bastion UI (Code Workspaces) or provide an absolute path and device. "
            "If exactly one workspace exists and the user did not name another, call code_get_workspace then code_open_workspace with that workspace_path and device_id. "
            "If multiple exist, show name + path + id and ask which workspace_id to use before calling code_open_workspace. "
            "After opening: call code_file_tree to orient (pass the same device_id from code_get_workspace so the right proxy handles paths when multiple devices are connected). "
            "Use code_search_files for exact regex/symbol search on the device (same device_id). "
            "For intent-based questions over indexed code, call code_semantic_search with the chosen workspace_id (user must run code_index_workspace first to populate the index). "
            "code_index_workspace: walks the repo on the device and upserts chunks for semantic search (may run in multiple batches if truncated). "
            "code_git_info: branch, status, diff, log. "
            "Follow any project rules in the system context (e.g. code_workspace_rules) for this workspace. "
            "**Editing — two paths:** (1) **Files on disk** (normal case): use local_read_file to read source; prefer **local_patch_file** "
            "(exact old_string → new_string, unique match) for small edits; use local_write_file for full-file rewrites. "
            "If a write fails, the Local Proxy config may have write_file disabled or the path outside allowed_paths — fix config on the device. "
            "For non-trivial or destructive edits, summarize the change and get explicit user confirmation before writing. "
            "Read enough context (full file or surrounding lines) so replacements are accurate. "
            "(2) **Bastion documents** when you have a document_id (imported notes, specs, or doc-linked content): use patch_file for batched "
            "replace/delete/insert_after_heading/append proposals, or append_to_file for appends — these create editor proposals the user accepts in the UI. "
            "That patch_file is for document_id workflows only; for repo files on disk use local_patch_file / local_write_file. "
            "**Shell (local_shell_execute):** Commands may be allowed, blocked by user shell policy (deny), or require human approval (require_approval). "
            "If the tool reports approval required, stop and wait — the user approves in chat (e.g. yes/approve) or via the notifications queue; then retry the same command. "
            "Always confirm destructive or irreversible operations (rm -rf, force push, piping to sh, etc.) with the user before running, even when policy would allow. "
            "Run builds/tests/lints from workspace_path as cwd when helpful. Never invent paths — derive from code_get_workspace and code_file_tree."
        ),
        "required_tools": [
            "code_list_workspaces",
            "code_get_workspace",
            "code_open_workspace",
            "code_file_tree",
            "code_search_files",
            "code_git_info",
            "local_read_file",
            "local_write_file",
            "local_patch_file",
            "local_list_directory",
            "local_shell_execute",
        ],
        "optional_tools": [
            "code_index_workspace",
            "code_semantic_search",
            "patch_file",
            "append_to_file",
        ],
        "tags": ["code", "workspace", "file tree", "search code", "git status", "coding", "source code", "codebase", "semantic search", "index"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Code platforms ----
    {
        "slug": "github",
        "name": "GitHub",
        "description": "Browse repos, issues, PRs, diffs, commits, branches, and code on GitHub via OAuth. Create issues, comment, and review PRs.",
        "category": "code",
        "procedure": (
            "You are a GitHub assistant. Use the available tools for all operations. "
            "github_list_repos: list repositories for the connected account; optional org, sort, per_page. "
            "github_get_repo: owner, repo for details. github_list_branches: owner, repo. "
            "github_list_issues: owner, repo; optional state (open/closed/all), labels, per_page. github_get_issue: owner, repo, issue_number. "
            "github_list_issue_comments: owner, repo, issue_number. github_create_issue: owner, repo, title, body. github_create_issue_comment: owner, repo, issue_number, body. "
            "github_list_pulls: owner, repo; optional state, per_page. github_get_pull: owner, repo, pull_number. github_get_pull_diff: owner, repo, pull_number. "
            "github_list_pull_reviews: owner, repo, pull_number. github_list_pull_comments: owner, repo, pull_number. "
            "github_create_pr_review: owner, repo, pull_number, body, event (APPROVE/REQUEST_CHANGES/COMMENT). "
            "github_list_commits: owner, repo; optional sha (branch), per_page. github_get_commit: owner, repo, ref. github_compare_refs: owner, repo, base, head. "
            "github_get_file_content: owner, repo, path; optional ref (branch/tag). github_search_code: query, optional owner, repo, per_page."
        ),
        "required_tools": [
            "github_list_repos", "github_get_repo", "github_list_issues", "github_get_issue",
            "github_list_issue_comments", "github_list_pulls", "github_get_pull", "github_get_pull_diff",
            "github_list_pull_reviews", "github_list_pull_comments", "github_list_commits", "github_get_commit",
            "github_compare_refs", "github_get_file_content", "github_list_branches", "github_search_code",
            "github_create_issue", "github_create_issue_comment", "github_create_pr_review",
        ],
        "optional_tools": [],
        "tags": ["github", "pull request", "PR", "issues", "code review", "commits", "repo", "repository", "branches", "diff", "merge"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["code_platform"],
    },
    {
        "slug": "gitea",
        "name": "Gitea",
        "description": "Browse repos, issues, PRs, diffs, commits, branches, and code on self-hosted Gitea instances via personal access token.",
        "category": "code",
        "procedure": (
            "You are a Gitea assistant. The tool names are the same as the GitHub tools (github_*) but are scoped to your Gitea connection. "
            "github_list_repos: list repositories. github_get_repo: owner, repo. github_list_branches: owner, repo. "
            "github_list_issues / github_get_issue: browse issues. github_list_issue_comments / github_create_issue / github_create_issue_comment: issue management. "
            "github_list_pulls / github_get_pull / github_get_pull_diff: pull request browsing. "
            "github_list_pull_reviews / github_list_pull_comments / github_create_pr_review: PR review. "
            "github_list_commits / github_get_commit / github_compare_refs: commit history. "
            "github_get_file_content: read file by path. github_search_code: search code. "
            "All tools accept the same parameters as the GitHub equivalents."
        ),
        "required_tools": [
            "github_list_repos", "github_get_repo", "github_list_issues", "github_get_issue",
            "github_list_issue_comments", "github_list_pulls", "github_get_pull", "github_get_pull_diff",
            "github_list_pull_reviews", "github_list_pull_comments", "github_list_commits", "github_get_commit",
            "github_compare_refs", "github_get_file_content", "github_list_branches", "github_search_code",
            "github_create_issue", "github_create_issue_comment", "github_create_pr_review",
        ],
        "optional_tools": [],
        "tags": ["gitea", "pull request", "issues", "code review", "commits", "repo", "self-hosted git"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["code_platform"],
    },
    # ---- Media processing ----
    {
        "slug": "media-processing",
        "name": "Media Processing",
        "description": "Transcode video/audio, extract audio tracks, trim clips, get metadata, download media, and burn subtitles.",
        "category": "media",
        "procedure": (
            "You help process media files. "
            "transcode_media: convert video/audio to another format (document_id + target_format e.g. mp4/mp3/wav/webm). "
            "extract_audio: extract the audio track from a video (document_id, optional output_format). "
            "trim_media: trim a clip by start/end timestamps (document_id, start_time, end_time in HH:MM:SS or seconds). "
            "get_media_info: get duration, codec, resolution, bitrate, and other metadata (document_id). "
            "download_media: download media from a URL and store as a document (url, optional filename). "
            "burn_subtitles: burn a subtitle file into a video (document_id of video, subtitle_document_id). "
            "read_media_metadata: read embedded metadata tags (title, artist, album, etc.) from a media file (document_id)."
        ),
        "required_tools": [
            "transcode_media", "extract_audio", "trim_media",
            "get_media_info", "download_media", "burn_subtitles",
            "read_media_metadata",
        ],
        "optional_tools": [],
        "tags": ["convert video", "extract audio", "trim video", "media info", "download video", "subtitles", "transcode", "audio"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Help / support ----
    {
        "slug": "help-docs",
        "name": "Help Docs",
        "description": "Search Bastion's shipped help documentation (product how-tos, features, settings). Not for searching the user's own uploaded files or library; use document-search for the knowledge base.",
        "category": "support",
        "procedure": (
            "You answer from Bastion's built-in help docs only. For material the user stored in Bastion (notes, PDFs, manuals in their folders), acquire document-search instead—this skill does not search their library. "
            "search_help_docs: query (natural language question or keywords). Returns relevant documentation sections with titles and content. "
            "Use when the user asks how to use or configure Bastion itself, what a feature does, or how to troubleshoot the app. "
            "Present the relevant documentation clearly; quote the key parts and provide context."
        ),
        "required_tools": ["search_help_docs"],
        "optional_tools": [],
        "tags": ["help", "how do I", "documentation", "guide", "instructions", "manual", "support"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Utility / state management ----
    {
        "slug": "utility-tools",
        "name": "Utility Tools",
        "description": "State management and data manipulation helpers: increment/decrement counters, adjust dates, parse/compare dates, toggle booleans, append to lists.",
        "category": "automation",
        "procedure": (
            "You help with data manipulation and state management. "
            "adjust_number: increment/decrement a numeric value (value, delta). "
            "adjust_date: shift a date by days/weeks/months (date, delta, unit). "
            "parse_date: parse a date string into structured components. "
            "compare_dates: compare two dates (date_a, date_b) and return which is earlier/later and the difference. "
            "set_value: store a named value for later retrieval (key, value). "
            "toggle_boolean: flip a boolean value (key). "
            "append_to_list: add an item to a named list (key, item). "
            "get_list_length: get the current length of a named list (key)."
        ),
        "required_tools": [
            "adjust_number", "adjust_date", "parse_date", "compare_dates",
            "set_value", "toggle_boolean", "append_to_list", "get_list_length",
        ],
        "optional_tools": [],
        "tags": ["counter", "date math", "toggle", "state", "list", "parse date", "compare dates"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    {
        "slug": "math-tools",
        "name": "Math Tools",
        "description": "Mathematical calculations, named formula evaluation (HVAC, electrical, construction), and unit conversions.",
        "category": "reference",
        "procedure": (
            "You help with math. "
            "calculate_expression: evaluate a mathematical expression (e.g. '2+3*4', 'sqrt(144)', 'sin(pi/4)'). "
            "evaluate_formula: run a named formula with parameters. Call list_available_formulas to see all formulas and their required parameters. "
            "convert_units: convert between units (value, from_unit, to_unit). Supports length, weight, temperature, area, volume, speed, and more. "
            "list_available_formulas: show all named formulas with descriptions and parameters."
        ),
        "required_tools": ["calculate_expression", "evaluate_formula", "convert_units", "list_available_formulas"],
        "optional_tools": [],
        "tags": ["math", "calculate", "formula", "convert units", "unit conversion", "expression", "HVAC", "electrical"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Session memory ----
    {
        "slug": "session-memory",
        "name": "Session Memory",
        "description": "Ephemeral clipboard for passing data between steps or storing intermediate results within a session.",
        "category": "automation",
        "procedure": (
            "You help store and retrieve temporary data within a session. "
            "clipboard_store: store a value under a named key (key, value). "
            "clipboard_get: retrieve a previously stored value by key (key). "
            "Use for passing intermediate results between steps or caching data within a conversation."
        ),
        "required_tools": ["clipboard_store", "clipboard_get"],
        "optional_tools": [],
        "tags": ["clipboard", "store", "retrieve", "temporary", "cache", "session", "pass data"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Data connection builder ----
    {
        "slug": "data-connection-builder",
        "name": "Data Connection Builder",
        "description": "Analyze APIs and websites, build and test data connectors, bulk scrape URLs, and manage control panes for agent status bars.",
        "category": "data",
        "procedure": (
            "You help build data connections and connectors. "
            "probe_api_endpoint: test an API URL and inspect the response. "
            "analyze_openapi_spec: parse an OpenAPI/Swagger spec to extract endpoints. "
            "draft_connector_definition: generate a connector definition from probed endpoints. "
            "validate_connector_definition: check a connector definition for errors. "
            "test_connector_endpoint: test a specific endpoint in a connector. "
            "create_data_connector / update_data_connector: persist a connector definition. "
            "list_data_connectors: list existing connectors. "
            "bulk_scrape_urls / get_bulk_scrape_status: scrape multiple URLs for content/images. "
            "bind_data_source_to_agent: link a connector to an agent profile. "
            "Control panes: list_control_panes, create_control_pane, update_control_pane, delete_control_pane, execute_control_action. "
            "get_connector_endpoints: list endpoints for a connector. "
            "Typical flow: probe API → analyze spec → draft → validate → create → bind to agent."
        ),
        "required_tools": [
            "probe_api_endpoint", "analyze_openapi_spec",
            "draft_connector_definition", "validate_connector_definition",
            "test_connector_endpoint", "create_data_connector",
            "list_data_connectors", "update_data_connector",
            "bulk_scrape_urls", "get_bulk_scrape_status",
            "bind_data_source_to_agent",
            "list_control_panes", "get_connector_endpoints",
            "create_control_pane", "update_control_pane",
            "delete_control_pane", "execute_control_action",
            "crawl_web_content", "search_web",
        ],
        "optional_tools": [],
        "tags": ["connector", "API", "data source", "scrape", "control pane", "build connector", "openapi"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Team collaboration ----
    {
        "slug": "team-collaboration",
        "name": "Team Collaboration",
        "description": "Agent line messaging, tasks, goals, governance, and workspace tools. Auto-injected when an agent runs in a team context.",
        "category": "agent",
        "procedure": (
            "You collaborate with other agents on a team (agent line). "
            "Messaging: send_to_agent (direct message), start_agent_conversation / halt_agent_conversation (multi-agent threads). "
            "read_team_timeline: see recent team activity. read_my_messages: check messages sent to you. "
            "get_team_status_board: see team member statuses. "
            "Workspace: write_to_workspace / read_workspace for shared documents. "
            "Tasks: create_task_for_agent, check_my_tasks, update_task_status, escalate_task. "
            "Goals: list_team_goals, report_goal_progress, delegate_goal_to_tasks. "
            "Governance: propose_hire, propose_strategy_change, propose_action, vote_on_proposal, tally_proposals. "
            "get_agent_run_history: review your own or another agent's recent runs."
        ),
        "required_tools": [
            "send_to_agent", "start_agent_conversation", "halt_agent_conversation",
            "read_team_timeline", "read_my_messages", "get_team_status_board",
            "write_to_workspace", "read_workspace",
            "create_task_for_agent", "check_my_tasks", "update_task_status", "escalate_task",
            "list_team_goals", "report_goal_progress", "delegate_goal_to_tasks",
            "propose_hire", "propose_strategy_change",
            "get_agent_run_history",
            "propose_action", "vote_on_proposal", "tally_proposals",
        ],
        "optional_tools": [],
        "tags": ["team", "collaborate", "agent line", "messaging", "tasks", "goals", "governance", "workspace"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Agent Factory builder (meta-agent) ----
    {
        "slug": "agent-factory-builder",
        "name": "Agent Factory Builder",
        "description": "Create and manage agent profiles, playbooks, and skills. Meta-agent capability for building other agents.",
        "category": "agent",
        "procedure": (
            "You help build and manage agents in the Agent Factory. "
            "Discovery: list_available_actions for tool I/O contracts; list_agent_profiles, list_playbooks, list_skills for summaries. "
            "Read before edit: get_agent_profile_detail (agent_id), get_playbook_detail (playbook_id), get_skill_detail (skill_id) — always fetch full definitions before update or delete. "
            "Playbooks: call validate_playbook_wiring on the step definition before create_playbook or update_playbook. "
            "create_playbook / update_playbook: use same step shape as get_playbook_detail returns; confirmed=False preview, confirmed=True apply. "
            "Agents: create_agent_profile; update_agent_profile to change settings; delete_agent_profile when removal is requested (confirmed flow). "
            "assign_playbook_to_agent links default playbook; set_agent_profile_status to pause or activate. "
            "Skills: create_skill; propose_skill_update to revise procedure (get_skill_detail first). "
            "Before create_agent_schedule, call list_agent_schedules for that agent to avoid duplicates. "
            "list_agent_data_sources shows connector bindings for an agent profile. "
            "Typical new build: list_available_actions, validate_playbook_wiring, create_playbook, create_agent_profile, assign_playbook_to_agent."
        ),
        "required_tools": [
            "list_available_actions", "validate_playbook_wiring",
            "list_agent_profiles", "get_agent_profile_detail",
            "create_agent_profile", "update_agent_profile", "delete_agent_profile",
            "set_agent_profile_status", "assign_playbook_to_agent",
            "list_playbooks", "get_playbook_detail",
            "create_playbook", "update_playbook", "delete_playbook",
            "list_skills", "get_skill_detail", "create_skill", "propose_skill_update",
            "list_agent_schedules", "list_agent_data_sources",
        ],
        "optional_tools": [],
        "tags": ["create agent", "build agent", "new agent", "create playbook", "list agents", "list playbooks", "agent factory", "build a skill"],
        "evidence_metadata": {"engine_type": "automation"},
    },
    # ---- Editor navigation (non-core, discoverable) ----
    {
        "slug": "editor-navigation",
        "name": "Editor Navigation",
        "description": "Navigate and extract sections from the active editor document and its references. List headings, pull specific sections by name or index, search for terms, and retrieve reference file sections on demand.",
        "category": "editor",
        "is_core": False,
        "procedure": (
            "You have tools to navigate the document currently open in the editor. "
            "Use them when you need content beyond what is already in your context.\n\n"
            "**Strategy:**\n"
            "1. editor_list_sections — see the full document structure (headings + char counts)\n"
            "2. editor_get_section — pull one section by heading name or index\n"
            "3. editor_get_sections — pull multiple sections at once (batch, more efficient)\n"
            "4. editor_search_content — find where a term or phrase appears in the document\n"
            "5. editor_get_ref_section — pull a section from a referenced file (outline, rules, style, etc.)\n\n"
            "**When to use:** When you need to check continuity with distant chapters, "
            "verify foreshadowing, confirm character details, or find where something was established. "
            "You already have the current and adjacent sections in your context — only pull more when needed.\n\n"
            "**When NOT to use:** For the current section and its neighbors (already provided). "
            "Do not pull the entire document section by section — be selective."
        ),
        "required_tools": [
            "editor_list_sections",
            "editor_get_section",
            "editor_get_sections",
            "editor_search_content",
            "editor_get_ref_section",
        ],
        "optional_tools": [],
        "tags": [
            "editor", "navigation", "sections", "chapters", "headings",
            "search", "manuscript", "outline", "reference", "context",
            "fiction", "document", "browse", "table of contents",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": [],
    },
    # ---- Story outline editing (fiction outline documents) ----
    {
        "slug": "outline-editing",
        "name": "Outline Editing",
        "description": (
            "Edit story outlines: chapters, beats, summaries, status, pacing, and synopsis. "
            "Uses patch_file for proposals and editor navigation tools for large documents."
        ),
        "category": "editor",
        "is_core": False,
        "depends_on": ["editor-navigation"],
        "procedure": (
            "You edit only the outline file the user has open. Reference files (rules, style, characters) "
            "are read-only.\n\n"
            "### CRITICAL: patch_file for edits\n"
            "When the user wants changes, you MUST call patch_file (once per turn, batch all edits in one "
            "edits list). Do not describe edits without calling the tool. If you catch yourself writing "
            "\"I've proposed...\" without having called patch_file, stop and call it. Summarize in text "
            "after the tool call.\n\n"
            "### When to edit vs answer\n"
            "- Information or analysis only (e.g. unresolved plot points, list chapters) -> reply in text; "
            "do not call patch_file.\n"
            "- User wants changes -> use the outline content in the user message as the primary source for "
            "verbatim targets. If you must verify a distant chapter before building a target, call "
            "editor_get_section first, then patch_file.\n"
            "- If a question implies edits you can do, call patch_file and answer in text.\n\n"
            "### Navigating long outlines\n"
            "Follow the included Editor Navigation skill for editor_list_sections, editor_get_section, "
            "editor_get_sections, editor_search_content, editor_get_ref_section. The outline in the user "
            "message is the source of truth; use navigation when you need verification beyond context, "
            "continuity search, or ref sections. Prefer editor_get_section over ad-hoc full-document fetches "
            "unless your step also exposes a full-document tool and content seems stale.\n\n"
            "### patch_file shape\n"
            "- document_id: from the Document context in the user message.\n"
            "- edits: non-empty list of objects. Each MUST have operation: replace | delete | "
            "insert_after_heading | append.\n"
            "- target: required for replace and delete: exact verbatim text from the outline (no "
            "paraphrasing). Prefer 10-20+ words. For insert_after_heading, target is the exact heading line "
            "(e.g. ### Beats) or anchor line.\n"
            "- content: required for replace, insert_after_heading, append.\n"
            "- section: optional but strongly recommended: exact chapter heading (e.g. ## Chapter 3) when "
            "### Beats, ### Summary, ### Status, or ### Pacing repeat across chapters.\n"
            "Never send empty edit objects. patch_file creates a proposal for review, not immediate apply.\n\n"
            "**Section scoping patterns:**\n"
            "- New beats under ### Beats: prefer insert_after_heading with section = ## Chapter N, "
            "target = ### Beats, content = new - lines only.\n"
            "- Replace/delete inside a chapter: replace or delete with section = that ## Chapter N and "
            "verbatim target from that chapter only.\n"
            "- Append at end of a chapter: append with section = ## Chapter N and content to insert before "
            "the next chapter heading.\n"
            "- New chapter at file end: append with no section; content starts with ## Chapter N and "
            "optional subsections.\n\n"
            "### Outline structure (no frontmatter tracking)\n"
            "**Top-level (required; add or fill when missing):**\n"
            "- # Overall Synopsis -- high-level story; update when the story changes.\n"
            "- # Notes -- rules, themes, worldbuilding, tone.\n"
            "- # Characters -- every named character as name + role only (e.g. Protagonist: Alex). Details "
            "live in character reference files. Add anyone named in the story or shown in reference "
            "character content if missing from # Characters.\n\n"
            "**Per chapter:** use exactly ## Chapter 1, ## Chapter 2, etc. (no titles in the heading).\n"
            "- ### Status (optional Ch 1; recommended Ch 2+): immediately after ## Chapter N, before "
            "Pacing/Summary. A **bulleted list** of who matters at chapter open and their situation, carried "
            "from **previous chapters only** (never from this chapter's beats).\n"
            "  Format each line: - **Name** -- where they are, what they are doing or their situation as this "
            "chapter opens.\n"
            "  Include only characters **relevant** going into this chapter (active or consequential from "
            "the prior chapter or still carrying forward). A minor character from an early chapter need not "
            "appear in Chapter 5+ unless still relevant.\n"
            "  Optionally add bullets for key items (who has them, where they are) if needed for continuity.\n"
            "  Example:\n"
            "  ### Status\n"
            "  - **Alex** -- at the safehouse, recovering after the ambush.\n"
            "  - **Mara** -- en route to the capital, unseen since she slipped away.\n"
            "- ### Pacing (optional; recommended Ch 2+): FROM / TO / TECHNIQUE; after Status if present, "
            "before Summary.\n"
            "- ### Summary: one brief paragraph (3-5 sentences); details go in beats.\n"
            "- ### Beats: bullet events only, each line starting with - . Max 150 beats per chapter.\n"
            "- ### Scenes (optional): group beats for generation.\n\n"
            "### Beats style\n"
            "Beats are shortened events, perceptions, feelings; no standalone direct dialogue quotes "
            "(paraphrase or pair a short line with an action). Example lines: emotional beats, discussion "
            "topics, procedural actions.\n\n"
            "### 150-beat limit\n"
            "Count beats before adding. If over 150, replace/delete less important beats first (always set "
            "section to that chapter). Then add.\n\n"
            "### Maintaining tracking\n"
            "Use the outline as source of truth. Keep Overall Synopsis, Notes, and Characters updated; keep "
            "each chapter's Status list aligned with the **previous** chapter's outcomes; keep Pacing/Summary "
            "consistent with beats; ensure every named character appears under # Characters.\n\n"
            "### Adding content\n"
            "- New beats: insert_after_heading (section + target ### Beats) preferred; else replace tail with "
            "section set -- never unscoped replace when ### Beats repeats.\n"
            "- New chapter at end: append without section; content is all-new events (do not duplicate prior "
            "chapter's closing beats).\n"
            "- Empty heading: insert_after_heading with section if needed; target = heading; content = "
            "paragraph/bullets. If content already exists under the heading, use replace to extend.\n"
            "- Synopsis/Notes with existing body: replace extending the last paragraph; omit section for "
            "true top-level sections. Do not insert_after_heading on a heading that already has body text "
            "(splits the section).\n"
            "- Empty outline (frontmatter only): insert_after_heading with target \"\" and content starting "
            "## Chapter 1 with Summary and Beats.\n\n"
            "### Surgical edits\n"
            "Verbatim targets; section when ambiguous. New chapter content must be new prose, not copied "
            "existing beats.\n\n"
            "### Output\n"
            "Edits: patch_file once, then brief summary. Questions only: text only, no patch_file. If vague, "
            "you may ask and still apply partial edits."
        ),
        "required_tools": ["patch_file"],
        "optional_tools": [],
        "tags": [
            "outline", "fiction", "chapters", "beats", "summary", "story structure",
            "patch_file", "manuscript", "writing", "plot",
        ],
        "evidence_metadata": {
            "engine_type": "automation",
            "requires_editor": True,
            "editor_types": ["outline"],
            "context_boost": 20,
        },
        "required_connection_types": [],
    },
    # ---- Azure DevOps ----
    {
        "slug": "azure-devops-reader",
        "name": "Azure DevOps Reader",
        "description": "Read-only access to Azure DevOps: projects, teams, work items, sprints, boards, repos, PRs, and pipelines.",
        "category": "devops",
        "procedure": (
            "You are an Azure DevOps read-only assistant. Use available tools to list and inspect DevOps data. "
            "list_devops_projects: discover available projects. "
            "list_devops_teams: list teams in a project (provide project name). "
            "list_devops_team_members: list members of a team (project + team name). "
            "query_devops_work_items: run a WIQL query against work items. The wiql parameter must be a valid WIQL string, e.g. "
            "'SELECT [System.Id],[System.Title],[System.State] FROM WorkItems WHERE [System.TeamProject] = @project AND [System.State] <> \"Closed\" ORDER BY [System.ChangedDate] DESC'. "
            "get_devops_work_item: get full details for a single work item by ID. "
            "list_devops_iterations: list sprints/iterations for a project (optionally scoped to a team). "
            "get_devops_iteration_work_items: get work items assigned to a specific sprint. "
            "list_devops_boards / get_devops_board_columns: view board layout and columns. "
            "list_devops_repos: list Git repositories. "
            "list_devops_pull_requests: list PRs (default: active; also 'completed', 'abandoned', 'all'). "
            "list_devops_pipelines / get_devops_pipeline_runs: view CI/CD pipelines and recent run results. "
            "You may NOT create, update, or comment on work items with this skill."
        ),
        "required_tools": [
            "list_devops_projects",
            "list_devops_teams",
            "list_devops_team_members",
            "query_devops_work_items",
            "get_devops_work_item",
            "list_devops_iterations",
            "get_devops_iteration_work_items",
            "list_devops_boards",
            "get_devops_board_columns",
            "list_devops_repos",
            "list_devops_pull_requests",
            "list_devops_pipelines",
            "get_devops_pipeline_runs",
        ],
        "optional_tools": [],
        "tags": [
            "devops", "azure devops", "work items", "sprints", "iterations",
            "boards", "pipelines", "pull requests", "repos", "teams",
            "backlog", "kanban", "scrum", "project management",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["devops"],
    },
    {
        "slug": "azure-devops-manager",
        "name": "Azure DevOps Manager",
        "description": "Full access to Azure DevOps: read all data plus create/update work items and add comments.",
        "category": "devops",
        "procedure": (
            "You are an Azure DevOps assistant with full read and write access. "
            "You can do everything the Azure DevOps Reader can, plus create and modify work items. "
            "create_devops_work_item: create a new work item. Required: project, work_item_type (Bug, Task, User Story, Feature, Epic), title. "
            "Optional: description, assigned_to, iteration_path, area_path, priority (1-4), tags. "
            "update_devops_work_item: modify an existing work item. Required: project, work_item_id. "
            "Optional: title, description, state, assigned_to, iteration_path, area_path, priority, tags. "
            "add_devops_work_item_comment: add a discussion comment. Required: project, work_item_id, text. "
            "For WIQL queries, always provide a valid WIQL string in the wiql parameter. "
            "IMPORTANT: For create and update operations, ALWAYS describe what you will do and ask for confirmation before executing. "
            "Do not create or modify work items without the user's explicit approval."
        ),
        "required_tools": [
            "list_devops_projects",
            "list_devops_teams",
            "list_devops_team_members",
            "query_devops_work_items",
            "get_devops_work_item",
            "list_devops_iterations",
            "get_devops_iteration_work_items",
            "list_devops_boards",
            "get_devops_board_columns",
            "list_devops_repos",
            "list_devops_pull_requests",
            "list_devops_pipelines",
            "get_devops_pipeline_runs",
            "create_devops_work_item",
            "update_devops_work_item",
            "add_devops_work_item_comment",
        ],
        "optional_tools": [],
        "tags": [
            "devops", "azure devops", "work items", "sprints", "iterations",
            "boards", "pipelines", "pull requests", "repos", "teams",
            "backlog", "kanban", "scrum", "project management",
            "create work item", "update work item", "comment",
        ],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": ["devops"],
    },
    {
        "slug": "artifact_generation",
        "name": "Artifact Generation",
        "description": "Produce rich visual outputs (HTML, charts, Mermaid diagrams, SVG, React/JSX components) in the chat artifact panel.",
        "category": "output",
        "procedure": (
            "When a visualization, interactive layout, diagram, or UI component would help more than plain text, call create_artifact. "
            "Always include a normal text explanation in your reply as well.\n\n"
            "Parameters:\n"
            "- artifact_type: html | mermaid | chart | svg | react\n"
            "- title: short user-facing title\n"
            "- code: the full content (for html/chart: self-contained HTML with inline CSS/JS as needed, e.g. Plotly from CDN; "
            "for mermaid: diagram source only; for svg: SVG markup; "
            "for react: one file only — root component named App or a single export default; no import/require; "
            "React and ReactDOM come from CDN globals; full rules in artifact_react skill)\n"
            "- language (optional): hint for code view (html, javascript, jsx, mermaid, svg)\n\n"
            "Live data in html, chart, or react previews: the UI injects window.bastion.query(path, params) — the parent proxies "
            "allowlisted GET requests with the user session (no raw fetch to /api/* from the iframe). "
            "CRITICAL for generated code: every path passed to bastion.query MUST start with the literal prefix /api/ "
            "(e.g. '/api/todos', not '/todos' or 'todos'). Paths without /api/ are rejected by the bridge. "
            "See artifact_react skill for the exact allowed paths, limits, and usage. "
            "Shared/public artifact links have no session; live queries fail there — handle errors or show static fallback.\n\n"
            "Prefer artifact_type chart for Plotly-style interactive charts (HTML snippet). "
            "Use mermaid for flowcharts, sequence diagrams, and ERD-style views. "
            "Use react for interactive React/JSX in a sandboxed iframe (React 18 + Babel from CDN; not a real bundler or Node project). "
            "Keep code within reasonable size; avoid embedding huge base64 blobs.\n\n"
            "Artifact lifecycle and surfaces:\n"
            "Artifacts start in the chat sidebar drawer, but the user can save them to an artifact library for reuse. Once saved, an artifact can be:\n"
            "- Embedded as a home dashboard widget (always visible, resizable card).\n"
            "- Embedded as a control pane in the status bar (compact popover, may run in the background).\n"
            "- Shared via public link (no auth session — bastion.query calls will fail; show a static fallback).\n"
            "Design implications: keep layouts responsive (artifacts may render in a narrow control pane or a wide dashboard card). "
            "For control pane use, compact layouts work best. Use cooperative state for anything stateful so instances stay in sync. "
            "Handle the no-auth case for public links.\n\n"
            "Cooperative state (when the artifact may appear in multiple places — dashboard widget and control pane): "
            "The SDK provides bastion.setState(key, value), bastion.getState(key), and bastion.onStateChange(callback). "
            "When an artifact calls bastion.setState('timer', 120), every other instance of the same artifact "
            "receives the update via the onStateChange callback (argument: {key, value, state}). "
            "Use this to keep counters, timers, toggles, or any mutable state in sync across instances. "
            "Always initialize from bastion.getState(key) on mount to pick up the current shared value. "
            "bastion.notify({ badge: true, text: '...' }) can surface a badge on the control pane icon in the status bar. "
            "These APIs are available when the artifact has an artifact_id (i.e. it was saved to the library "
            "and embedded in a dashboard widget or control pane). In one-off chat previews they are inert (no-op). "
            "When building artifacts that manage ongoing state (timers, counters, toggles, live dashboards), "
            "ALWAYS use bastion.setState / onStateChange to make them cooperative-ready, even for initial creation."
        ),
        "required_tools": ["create_artifact"],
        "optional_tools": [],
        "tags": [
            "artifact", "chart", "diagram", "visualization", "html", "mermaid",
            "interactive", "plotly", "svg", "react", "jsx", "component",
        ],
        "evidence_metadata": {"engine_type": "output"},
        "required_connection_types": [],
        "is_core": True,
    },
    {
        "slug": "artifact_charts",
        "name": "Artifact Charts",
        "description": "Plotly-oriented chart artifacts; extends Artifact Generation.",
        "category": "output",
        "depends_on": ["artifact_generation"],
        "procedure": (
            "For numeric or time-series data you want to chart, choose the right tool:\n\n"
            "- create_chart: preferred when the user supplies structured data (numbers, labels, series). "
            "Pass chart_type (bar, line, pie, scatter, area, heatmap, box_plot, histogram), title, and data. "
            "The tool generates valid Plotly HTML automatically and emits it as an interactive artifact. "
            "Use this path when reliability matters more than custom styling.\n\n"
            "- create_artifact (artifact_type=chart): preferred for custom layouts, combination charts, "
            "annotations, or chart types not covered by create_chart. "
            "Provide a self-contained HTML document that loads Plotly from CDN and renders in the sandboxed panel. "
            "Include clear axis labels and a legend when applicable. "
            "For data that must stay current (e.g. todos, RSS counts), load JSON via window.bastion.query (allowlisted GET only; "
            "see artifact_react skill). In HTML/JS you emit, the first argument MUST be a path starting with /api/ "
            "(e.g. bastion.query('/api/todos', {...})); never use '/todos' alone. Do not use fetch() to /api/* from the iframe. "
            "Otherwise rely on CDN scripts only.\n\n"
            "Always include a brief text explanation in your reply alongside the artifact."
        ),
        "required_tools": ["create_artifact"],
        "optional_tools": ["create_chart"],
        "tags": ["artifact", "chart", "plotly", "visualization", "graph"],
        "evidence_metadata": {"engine_type": "output"},
        "required_connection_types": [],
        "is_core": False,
    },
    {
        "slug": "artifact_diagrams",
        "name": "Artifact Diagrams",
        "description": "Mermaid diagram artifacts; extends Artifact Generation.",
        "category": "output",
        "depends_on": ["artifact_generation"],
        "procedure": (
            "For workflows, architecture, or relationships, use create_artifact with artifact_type mermaid. "
            "Pass valid Mermaid syntax only in code (no HTML wrapper). "
            "Common diagram types: flowchart, sequenceDiagram, classDiagram, erDiagram, stateDiagram-v2, gantt. "
            "Keep diagrams readable: limit nodes and use short labels."
        ),
        "required_tools": ["create_artifact"],
        "optional_tools": [],
        "tags": ["artifact", "mermaid", "diagram", "flowchart", "sequence", "erd"],
        "evidence_metadata": {"engine_type": "output"},
        "required_connection_types": [],
        "is_core": False,
    },
    {
        "slug": "artifact_react",
        "name": "Artifact React Components",
        "description": "Live React/JSX component previews in the artifact panel.",
        "category": "output",
        "depends_on": ["artifact_generation"],
        "procedure": (
            "For interactive UI demos, form prototypes, or component previews, "
            "use create_artifact with artifact_type react.\n\n"
            "What the preview actually is:\n"
            "- A single script evaluated in a sandboxed iframe: React 18 and ReactDOM are globals; "
            "Babel (CDN) transpiles JSX. There is no bundler, no node_modules, and no ES module loader.\n\n"
            "Live data (bastion bridge):\n"
            "- CRITICAL — path format: endpointPath MUST begin with /api/ exactly. The bridge normalizes and allowlists only "
            "application API routes under /api/. Short paths like '/todos', '/rss/feeds', or '/folders/tree' are REJECTED "
            "with 'Endpoint not allowed'. Correct: bastion.query('/api/todos', {...}). Wrong: bastion.query('/todos', {...}). "
            "Never omit the /api/ segment when calling bastion.query from generated React, HTML, or chart code.\n"
            "- The preview injects window.bastion.query(endpointPath, queryParams) — returns a Promise that resolves to the JSON body "
            "of an allowlisted GET. The parent attaches the user session; do not use fetch() or XMLHttpRequest to /api/* from the iframe.\n"
            "- endpointPath is the path only (no host, no origin URL), always starting with /api/, e.g. '/api/todos' or "
            "'/api/rss/feeds/your-feed-id/articles'. queryParams is a plain object turned into a query string (omit or {} if none).\n"
            "- Allowed paths only (anything else is rejected): /api/todos, /api/org/todos (deprecated list shape); /api/org/agenda, /api/org/tags, /api/org/search, "
            "/api/org/clock/active, /api/org/journal/entry; /api/calendar/events, /api/calendar/calendars; "
            "/api/rss/feeds, /api/rss/unread-count, /api/rss/feeds/{feed_id}/articles; /api/folders/tree, /api/document-pins, "
            "/api/folders/{folder_id}/contents; /api/status-bar/data.\n"
            "- Limits: read-only GET; roughly 10 requests per 10 seconds and 3 concurrent; ~15s timeout per call. "
            "Use modest polling (e.g. 30–60s) if refreshing. Always .catch() errors.\n"
            "- Public/shared artifact URLs have no login cookie path for the bridge; bastion.query fails — use static fallback or a clear message.\n\n"
            "The artifact may later be saved and embedded as a dashboard widget or control pane (compact popover in the status bar). "
            "Design for variable viewport sizes and use cooperative state for mutable data.\n\n"
            "Cooperative state:\n"
            "When the artifact is saved and embedded in multiple places (dashboard + control pane), the SDK "
            "provides synchronization. Always structure mutable state through the bridge:\n"
            "  const [count, setCount] = React.useState(() => bastion.getState('count') ?? 0);\n"
            "  React.useEffect(() => {\n"
            "    bastion.onStateChange(({key, value}) => { if (key === 'count') setCount(value); });\n"
            "  }, []);\n"
            "  function increment() {\n"
            "    const next = count + 1;\n"
            "    setCount(next);\n"
            "    bastion.setState('count', next);\n"
            "  }\n"
            "Initialize from bastion.getState on mount; update via bastion.setState on change; listen with "
            "bastion.onStateChange for external updates. Use bastion.notify({ badge: true, text: 'Done' }) to badge the "
            "control pane icon. These are no-ops in one-off chat previews (no artifact_id), so always safe to include. "
            "When the use case involves ongoing state (timers, counters, toggles, live status), always wire up "
            "cooperative state so the artifact works correctly across dashboard and control pane instances.\n\n"
            "Do:\n"
            "- One entry component only: name it App (function App, const App = …, class App), "
            "or export default exactly one component (including export default function App).\n"
            "- Use React.useState, React.useEffect, React.useMemo, etc. on the React global.\n"
            "- Style with inline style objects or a <style> tag inside your JSX tree.\n"
            "- Prefer plain JavaScript; strip TypeScript types if you would otherwise emit TS.\n\n"
            "Do not:\n"
            "- import / require / dynamic import(), multiple files, or export maps (e.g. export { X as default }).\n"
            "- Rely on npm packages, React Router, or other frameworks unless you inline minimal code yourself.\n"
            "- Use fetch() to this app’s /api/* APIs — use bastion.query for the allowlisted paths above.\n"
            "- Pass a bastion.query path that does not start with /api/ (e.g. '/todos' or '/folders/tree' — wrong; use '/api/todos', '/api/folders/tree').\n\n"
            "Example (counter):\n"
            "function App() {\n"
            "  const [count, setCount] = React.useState(0);\n"
            "  return React.createElement('button', { type: 'button', onClick: () => setCount((c) => c + 1) }, 'Clicks: ' + count);\n"
            "}\n\n"
            "Example (live todos): in useEffect, call bastion.query('/api/todos', { scope: 'all' }).then(...).catch(...); "
            "map results to UI. The string '/api/todos' is mandatory — '/todos' will fail. JSX in return is supported. "
            "Keep everything self-contained."
        ),
        "required_tools": ["create_artifact"],
        "optional_tools": [],
        "tags": ["artifact", "react", "jsx", "component", "interactive", "ui", "preview"],
        "evidence_metadata": {"engine_type": "output"},
        "required_connection_types": [],
        "is_core": False,
    },
    {
        "slug": "read-scratchpad",
        "name": "Read Scratch Pad",
        "description": "Read the contents of the user's dashboard scratch pad (up to 4 named pads). Use to understand what the user has noted for today, their current tasks, or any staging content.",
        "category": "user",
        "procedure": (
            "You can read the user's scratch pad to understand their current notes and staging content. "
            "ALWAYS call read_scratchpad_tool before answering questions about what the user has in their scratch pad. "
            "Parameters: pad_index (optional integer 0-3 to read a single pad; omit or pass -1 to read all pads). "
            "The pads are labeled by the user (default: Pad 1–4). "
            "Do not fabricate pad content; always call the tool to get the real data."
        ),
        "required_tools": ["read_scratchpad"],
        "optional_tools": [],
        "tags": ["scratchpad", "scratch pad", "notes", "dashboard notes", "my notes", "what did i note", "my pad"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": [],
    },
    {
        "slug": "write-scratchpad",
        "name": "Write Scratch Pad",
        "description": "Write or update a named pad on the user's dashboard scratch pad. Overwrites the body of the specified pad. Use only when explicitly instructed by the user.",
        "category": "user",
        "is_core": False,
        "procedure": (
            "You can write content to one of the user's four dashboard scratch pads. "
            "Parameters: pad_index (required integer 0-3), body (required string — the new content), "
            "label (optional — rename the pad tab). "
            "IMPORTANT: This OVERWRITES the existing content of the pad. "
            "ALWAYS confirm with the user which pad to use and what content to write BEFORE calling write_scratchpad_pad_tool. "
            "Do not call this tool on your own initiative; only when the user explicitly asks you to update their scratch pad."
        ),
        "required_tools": ["write_scratchpad_pad"],
        "optional_tools": ["read_scratchpad"],
        "tags": ["scratchpad", "scratch pad", "write notes", "update pad", "save to dashboard", "put in scratch pad"],
        "evidence_metadata": {"engine_type": "automation"},
        "required_connection_types": [],
    },
]
