"""
Automation engine skill definitions.

Skills: weather, dictionary, help, email, navigation, rss, entertainment,
org_capture, image_generation, image_description.
"""

from orchestrator.skills.skill_schema import EngineType, Skill

AUTOMATION_SKILLS = [
    Skill(
        name="weather",
        description="Get current weather conditions, forecasts, and historical weather data for any location.",
        engine=EngineType.AUTOMATION,
        domains=["weather", "forecast", "climate"],
        actions=["query", "observation"],
        keywords=["weather", "temperature", "forecast", "rain", "snow", "sunny", "cloudy"],
        priority=90,
        tools=["get_weather_tool"],
        system_prompt=(
            "You are a weather assistant. ALWAYS use get_weather_tool to fetch real weather data - never make up weather information. "
            "Parameters: location (required - city name, ZIP code, or place name like 'New York' or '14532'), "
            "data_types (comma-separated: 'current' for now, 'forecast' for upcoming days, 'history' for past weather; default 'current'), "
            "date_str (for history only: 'YYYY-MM-DD' for a specific day, 'YYYY-MM' for monthly average, or 'YYYY-MM to YYYY-MM' for date ranges up to 24 months like '2024-01 to 2024-12'). "
            "If the user does not specify a location, ask them for one before calling the tool. "
            "Present temperatures in both Fahrenheit and Celsius. Include practical recommendations based on conditions."
        ),
    ),
    Skill(
        name="dictionary",
        description="Look up word definitions, pronunciation, etymology, and synonyms.",
        engine=EngineType.AUTOMATION,
        domains=["general", "reference"],
        actions=["query"],
        keywords=["define", "definition", "meaning", "etymology", "pronunciation", "synonym", "antonym", "word"],
        priority=95,
        tools=[],
        system_prompt="You are a dictionary assistant. Look up words and present definitions clearly with pronunciation and etymology when available.",
    ),
    Skill(
        name="help",
        description="Answer questions about how to use the system, available features, and documentation.",
        engine=EngineType.AUTOMATION,
        domains=["general", "help", "documentation"],
        actions=["query"],
        keywords=[
            "help", "documentation", "features", "getting started", "agents available",
            "user guide", "feature guide", "tutorial", "how to use", "what can i do",
        ],
        priority=85,
        tools=[],
        system_prompt="You are a helpful assistant for this application. Answer questions about features, agents, and how to use the system. Be concise and accurate.",
    ),
    Skill(
        name="email",
        description="Read, search, send, and manage email.",
        engine=EngineType.AUTOMATION,
        domains=["email", "inbox", "mail"],
        actions=["query", "observation", "generation", "modification"],
        keywords=[
            "email", "inbox", "emails", "mail", "message", "messages",
            "send email", "reply to", "read my email", "check email", "search email",
            "unread", "draft", "compose", "forward",
        ],
        priority=90,
        tools=[
            "get_emails_tool",
            "search_emails_tool",
            "get_email_thread_tool",
            "get_email_statistics_tool",
            "send_email_tool",
            "reply_to_email_tool",
        ],
        system_prompt=(
            "You are an email assistant. Use the available tools for all operations. "
            "get_emails_tool: folder (inbox/sent/drafts), top (limit), unread_only (bool). "
            "search_emails_tool: query, top. get_email_thread_tool: conversation_id from a previous list. "
            "get_email_statistics_tool: inbox/unread counts. "
            "For sending: ALWAYS call send_email_tool with confirmed=False first to show the draft; only call with confirmed=True after the user explicitly approves (e.g. says yes, send, approve). "
            "For replying: ALWAYS call reply_to_email_tool with confirmed=False first to show the draft; only call with confirmed=True after the user approves. "
            "send_email_tool: to (comma-separated), subject, body, confirmed (bool, default False). "
            "reply_to_email_tool: message_id from thread/list, body, reply_all (bool), confirmed (bool, default False)."
        ),
    ),
    Skill(
        name="navigation",
        description="Manage saved locations, plan routes, and get directions.",
        engine=EngineType.AUTOMATION,
        domains=["navigation", "locations", "routes", "maps"],
        actions=["observation", "query", "management"],
        keywords=[
            "create location", "save location", "add location", "new location at",
            "list locations", "show my locations", "saved locations", "my locations",
            "delete location", "remove location",
            "route from", "route to", "directions", "how do i get", "navigate",
            "drive from", "walk from", "map", "turn by turn", "save route",
        ],
        priority=90,
        tools=[
            "create_location_tool",
            "list_locations_tool",
            "delete_location_tool",
            "compute_route_tool",
            "save_route_tool",
            "list_saved_routes_tool",
        ],
        system_prompt=(
            "You are a navigation assistant. Multi-step flow: (1) list_locations_tool to see saved locations and their IDs; "
            "create_location_tool to add a new one (name, address). (2) compute_route_tool with from_location_id and to_location_id "
            "(from list_locations), or use coordinates string. profile: driving, walking, cycling. "
            "save_route_tool to save a computed route (pass waypoints, geometry, steps, distance_meters, duration_seconds from compute_route). "
            "list_saved_routes_tool to list saved routes. delete_location_tool to remove a location by location_id."
        ),
    ),
    Skill(
        name="rss",
        description="Manage RSS feeds, list feeds, and fetch recent articles.",
        engine=EngineType.AUTOMATION,
        domains=["general", "rss", "news"],
        actions=["query", "management"],
        keywords=["rss", "feed", "feeds", "subscribe", "news", "articles"],
        priority=85,
        tools=["add_rss_feed_tool", "list_rss_feeds_tool", "refresh_rss_feed_tool"],
        system_prompt=(
            "You are an RSS feed assistant. add_rss_feed_tool: feed_url (required), feed_name, category, is_global (bool). "
            "list_rss_feeds_tool: scope 'user' or 'global'. refresh_rss_feed_tool: feed_name or feed_id from list to trigger refresh."
        ),
    ),
    Skill(
        name="entertainment",
        description=(
            "Recommend movies, books, music, and other entertainment based on preferences. "
            "Does NOT search local collections — only provides recommendations and suggestions."
        ),
        engine=EngineType.AUTOMATION,
        domains=["general", "entertainment"],
        actions=["query", "observation"],
        keywords=["movie", "film", "book", "music", "recommend", "entertainment", "watch", "read", "listen"],
        priority=85,
        tools=[],
        system_prompt="You are an entertainment recommendation assistant. Suggest movies, books, music, and similar based on user preferences.",
    ),
    Skill(
        name="org_capture",
        description="Quick capture items to the org-mode inbox.",
        engine=EngineType.AUTOMATION,
        domains=["general", "management"],
        actions=["generation", "management"],
        keywords=["capture", "inbox", "capture to inbox", "for my inbox", "add to inbox", "quick capture"],
        priority=95,
        tools=["add_org_inbox_item_tool"],
        stateless=True,
        system_prompt=(
            "You help capture items to the user's org-mode inbox. "
            "ALWAYS call add_org_inbox_item_tool to add each item - never just describe what you would do. "
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
            "CALL RULES: Call add_org_inbox_item_tool exactly once per distinct inbox entry. "
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
            "Parameters: text (required), kind ('todo', 'note', 'checkbox', 'event', 'contact'; default 'todo'), "
            "schedule (optional org timestamp like '<2026-02-05 Thu>'), tags (optional comma-separated)."
        ),
    ),
    Skill(
        name="org_content",
        description="Read-only queries about org-mode files, todos, and tasks.",
        engine=EngineType.AUTOMATION,
        domains=["general"],
        actions=["query", "observation"],
        editor_types=["org"],
        context_boost=15,
        keywords=[
            "org", "org-mode", "orgmode", "todo", "task",
            "project", "show", "list", "find", "tagged", "tag", "org file",
        ],
        priority=70,
        tools=[
            "parse_org_structure_tool",
            "list_org_todos_tool",
            "search_org_headings_tool",
            "get_org_statistics_tool",
        ],
        system_prompt=(
            "You help the user query their org-mode file. The file content is provided above (=== FILE ... ===). "
            "Use the tools to answer: parse_org_structure_tool to get the full outline; list_org_todos_tool with "
            "state_filter (TODO, DONE, NEXT, etc.) and optional tag_filter to list TODOs; search_org_headings_tool "
            "with search_term to find headings or content; get_org_statistics_tool for counts and completion rate. "
            "When the user says 'today' or asks about the current day, use the YYYY-MM-DD date from the datetime "
            "context as search_term (e.g. 2026-02-10 for today) so you find today's heading or entry. Call the "
            "relevant tool(s) then summarize the results in plain language. Read-only; do not modify the file."
        ),
    ),
    Skill(
        name="image_generation",
        description="Generate images from text descriptions.",
        engine=EngineType.AUTOMATION,
        domains=["general", "image", "visual", "art"],
        actions=["generation"],
        keywords=[
            "create image", "generate image", "generate picture", "draw", "visualize",
            "image", "picture", "photo", "photography", "create a picture",
            "make an image", "generate a photo", "create a photo", "draw a picture",
            "create an image", "make a picture", "generate an image", "create picture",
        ],
        priority=90,
        tools=["generate_image_tool", "get_reference_image_tool"],
        system_prompt=(
            "You are an image generation assistant. Use generate_image_tool: prompt (required), size (e.g. 1024x1024, 512x512), "
            "num_images (1-4), optional negative_prompt, model. get_reference_image_tool: object_name to check for a reference image "
            "from the user's library (e.g. for accuracy); then describe that object in the prompt when generating."
        ),
    ),
    Skill(
        name="image_description",
        description="Describe or analyze attached images.",
        engine=EngineType.AUTOMATION,
        domains=["general", "image", "vision"],
        actions=["observation", "query"],
        keywords=[
            "describe this image", "describe the image", "what is in this image",
            "what's in this image", "what does this image show", "describe this picture",
            "what do you see", "analyze this image", "what is this image", "caption this",
            "describe this photo", "what's in the image", "tell me about this image",
        ],
        priority=92,
        requires_image_context=True,
        tools=[],
        system_prompt="You describe and analyze images. Use the provided image context to answer the user's question.",
    ),
    Skill(
        name="reference",
        description="Query and analyze reference documents, journals, and logs; run calculations and visualizations.",
        engine=EngineType.AUTOMATION,
        domains=["general", "reference", "journal", "log"],
        actions=["query", "analysis", "observation", "generation"],
        editor_types=["reference"],
        requires_editor=True,
        context_boost=20,
        keywords=[
            "journal", "log", "record", "tracking", "diary", "food log", "weight log", "mood log",
            "graph", "chart", "visualize",
            "calculate", "calculation", "compute", "math", "formula", "btu", "heat loss", "heat losses",
            "manual j", "hvac", "electrical", "ohms law", "convert units", "unit conversion",
        ],
        priority=85,
        tools=["calculate_expression_tool", "evaluate_formula_tool", "convert_units_tool", "create_chart_tool"],
        system_prompt="You help with reference documents, journals, and logs. You can run calculations, convert units, and create visualizations. Use the available tools.",
    ),
    Skill(
        name="document_creator",
        description="Create new documents or reference files and place them in a specific folder.",
        engine=EngineType.AUTOMATION,
        domains=["general", "documents", "files", "management"],
        actions=["generation", "management"],
        keywords=[
            "create file", "create document", "save to folder", "new file",
            "put in folder", "reference file", "save as",
        ],
        priority=80,
        tools=["list_folders_tool", "create_user_file_tool", "create_user_folder_tool"],
        system_prompt=(
            "You are a document creation assistant. You create files and folders in the user's document tree. "
            "WORKFLOW: "
            "1. If the user specifies a folder by name, call list_folders_tool first to find its folder_id or verify it exists. "
            "2. If the folder does not exist, call create_user_folder_tool to create it first. "
            "3. Call create_user_file_tool with the content, filename, and folder_path (e.g., 'Reference'). "
            "PARAMETERS for create_user_file_tool: "
            "filename (with extension, e.g., 'uranium_deposits.md'), "
            "content (the document body - format as clean markdown with a title heading), "
            "folder_path (e.g., 'Reference' or 'Projects/Notes' - will auto-create if missing). "
            "If context from a prior step is available (check for prior_step_*_response keys in the conversation), "
            "use it as the document body. Format it as clean, well-structured markdown. "
            "If the user doesn't specify a filename, generate a descriptive one from the content."
        ),
    ),
]
