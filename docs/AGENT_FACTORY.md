# Agent Factory: User-Defined Research & Intelligence Agents

**Document Version:** 1.0
**Last Updated:** February 14, 2026
**Status:** Planning Phase
**Supersedes:** DATA_WORKSPACE_ANALYTICS_ENHANCEMENTS.md
**Companion docs:** [Technical Guide](./AGENT_FACTORY_TECHNICAL_GUIDE.md), [Tool Catalog](./AGENT_FACTORY_TOOLS.md), [Examples](./AGENT_FACTORY_EXAMPLES.md)

---

## Executive Summary

The Agent Factory is a GUI-driven system that enables users to assemble custom intelligent agents from modular building blocks — data source connectors, workflow steps (tools, LLM tasks, and approval gates), knowledge graph configurations, and output destinations — assisted by an AI-powered Assembly Agent that helps non-technical users build sophisticated research and intelligence capabilities.

A custom agent IS a workflow. Users compose that workflow from three types of steps — **deterministic tool calls** (no LLM, fully predictable), **LLM tasks** (analysis, synthesis, classification), and **approval gates** (human-in-the-loop checkpoints) — mixed in any combination. An agent can be a pure data pipeline with no LLM involvement, a fully LLM-driven research workflow, or a hybrid that uses tools for data collection, LLM for analysis, and approval gates for human oversight. Any workflow can run interactively in chat, as a background job, or on a cron schedule.

Every custom agent has a **unique @handle** (e.g., `@nonprofit-investigator`) that users type in the chat sidebar to invoke it directly — no auto-routing, no trigger ambiguity. Agents maintain a **work journal** that tracks every execution, enabling users to ask "What have you done today?" and get a meaningful answer. Agents can be **shared with teams**, accessing team files and team conversation threads when the user explicitly grants permission.

This system transforms Bastion from a platform where agents are code-defined and developer-maintained into one where domain experts (investigative journalists, researchers, analysts, nonprofit auditors) can create purpose-built research agents through a visual interface. Each custom agent feeds discoveries back into a shared knowledge graph, meaning the entire system gets smarter with every question asked of any agent.

The Agent Factory subsumes and extends the previously-planned Data Workspace Analytics Enhancements. The Data Workspace becomes one of several output destinations for custom agents, and its analytical capabilities (statistical functions, pattern detection, clustering) become tools that custom agents can leverage rather than standalone features.

---

## Motivation: The Intelligence Gap

### The Problem

Current open-source agent frameworks (OpenClaw, LangFlow, Flowise) offer two extremes:

1. **Code-level extensibility** — Write TypeScript/Python plugins to add tools. Powerful but requires developers. (OpenClaw model)
2. **No-code node editors** — Drag LLM nodes on a canvas. Accessible but shallow — no knowledge graphs, no entity resolution, no compound learning. (LangFlow/Flowise model)

Neither enables a domain expert to build something like: "A research agent that pulls FEC campaign contribution records, cross-references them with IRS 990 nonprofit filings, resolves entities across data sources, builds a knowledge graph of funding relationships, and gets smarter with every query."

### The Opportunity

Bastion already has the infrastructure pieces:

- **Neo4j** knowledge graph with entity extraction
- **Qdrant** vector store with semantic search
- **LangGraph** multi-agent runtime with checkpointing
- **Data Workspace** with import, SQL queries, and visualization
- **Web crawling** (Crawl4AI) and **web search** (SearXNG)
- **Typed skill system** with engine routing
- **Multi-round research** with gap analysis

What's missing is the **user-facing composition layer** — the ability for users to wire these pieces together into custom agents, and an AI assistant to help them do it.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     AGENT FACTORY UI                              │
│                                                                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐  │
│  │   Data      │ │  Workflow   │ │   Output   │ │    Test      │  │
│  │   Sources   │ │  Composer   │ │ Destinations│ │    Panel     │  │
│  │   Panel     │ │   Panel    │ │   Panel     │ │              │  │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              Assembly Agent (AI Assistant)                     │ │
│  │  Helps configure connectors, suggests skills, tests outputs   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                          Agent Profile
                        (stored in PostgreSQL)
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
     │  Connector    │  │  Workflow     │  │     Output       │
     │  Registry     │  │  Engine       │  │   Router         │
     │  (dynamic)    │  │  (dynamic)   │  │   (configurable) │
     └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
            │                 │                    │
            ▼                 ▼                    ▼
     ┌──────────────────────────────────────────────────────┐
     │           WORKFLOW RUNTIME (LangGraph)                  │
     │                                                        │
     │   Execution Modes:                                     │
     │   • Deterministic — tool pipelines, no LLM             │
     │   • LLM-Augmented — LLM decides tool use               │
     │   • Hybrid — mix tools, LLM tasks, & approval gates    │
     │                                                        │
     │   Run Contexts:                                        │
     │   • Interactive — streams in chat                      │
     │   • Background  — runs outside chat session            │
     │   • Scheduled   — triggered by cron                    │
     │   • Monitor     — periodic change-aware polling        │
     └──────────────────────┬───────────────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────────────┐
     │          KNOWLEDGE ACCUMULATION LOOP                   │
     │   Entity Extract → Resolve → Store in Neo4j            │
     │   Embeddings → Store in Qdrant                         │
     │   Structured data → Route to output destinations       │
     └──────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Agent Profile

An Agent Profile is a stored configuration that fully defines a custom agent. It is the primary artifact produced by the Agent Factory UI.

**Schema:**

```
AgentProfile
├── id: UUID
├── user_id: UUID
├── name: str                        ("Nonprofit Investigator")
├── handle: str                      ("nonprofit-investigator" — unique per user, used as @mention)
├── description: str                 ("Tracks money flows through nonprofits...")
├── icon: str                        (emoji or icon reference)
├── created_at: datetime
├── updated_at: datetime
├── is_active: bool
│
├── data_sources: List[DataSourceBinding]
│   ├── connector_id: UUID           (which connector to use)
│   ├── config_overrides: Dict       (API keys, base URLs, etc.)
│   └── permissions: Dict            (rate limits, allowed endpoints)
│
├── skills: List[SkillBinding]
│   ├── skill_id: UUID               (built-in or user-defined skill)
│   ├── priority: int                (execution order preference)
│   └── parameters: Dict             (skill-specific config)
│
├── tool_packs: List[str]            (tool pack names to include)
│
├── knowledge_config: KnowledgeConfig
│   ├── read_collections: List[str]  (Qdrant collections to search)
│   ├── write_collection: str        (where to store discoveries)
│   ├── graph_namespaces: List[str]  (Neo4j label prefixes to query)
│   ├── auto_enrich: bool            (feed discoveries back to graph)
│   └── entity_resolution: bool      (enable cross-source entity matching)
│
├── output_config: OutputConfig
│   ├── destinations: List[OutputDestination]
│   ├── default_format: str          (markdown, json, csv, org)
│   └── auto_save: bool             (save results automatically)
│
├── system_prompt_additions: str     (domain-specific instructions)
├── model_preference: str            (preferred LLM model)
├── max_research_rounds: int         (depth of multi-round research)
│
├── team_config: TeamConfig
│   ├── shared_with_teams: List[UUID]  (team IDs this agent is shared with)
│   ├── team_file_access: bool         (can access team folders/documents)
│   ├── team_post_access: bool         (can read/write team conversation threads)
│   └── team_permissions: Dict         (per-team permission overrides)
│
└── journal_config: JournalConfig
    ├── auto_journal: bool             (log every execution automatically)
    ├── journal_detail_level: str      (summary, detailed, verbose)
    └── journal_retention_days: int    (how long to keep entries, default: 90)
```

### 2. Data Source Connector

A Data Source Connector is a declarative definition of an external data source. Connectors are templates; Agent Profiles bind to connector instances with specific credentials and configuration.

**Connector Types:**

| Type | Description | Example |
|------|-------------|---------|
| `rest_api` | REST API with structured endpoints | FEC API, ProPublica Nonprofit API |
| `graphql` | GraphQL API with schema introspection | GitHub API, Hasura endpoints |
| `web_scraper` | Structured web scraping with selectors | State corporate registries |
| `file_parser` | File format parsing (CSV, XML, PDF) | IRS 990 XML bulk data |
| `database` | External database connection | Existing Data Workspace DBs |
| `rss_feed` | RSS/Atom feed monitoring | News feeds, government updates |
| `existing_tool` | Wraps an existing Bastion tool | SearXNG search, Crawl4AI |

**REST API Connector Definition (Example — FEC):**

```yaml
id: fec-contributions
name: FEC Campaign Contributions
type: rest_api
version: "1.0"
description: Search Federal Election Commission campaign contribution records

connection:
  base_url: https://api.open.fec.gov/v1
  auth:
    type: api_key
    key_param: api_key
    env_var: FEC_API_KEY
  rate_limit:
    requests_per_second: 10
    daily_limit: 1000
  timeout_seconds: 30

endpoints:
  search_contributions:
    method: GET
    path: /schedules/schedule_a/
    description: Search individual campaign contributions
    parameters:
      - name: contributor_name
        type: string
        description: Name of the contributor
        required: false
      - name: committee_id
        type: string
        description: FEC committee ID
        required: false
      - name: min_date
        type: date
        description: Start date (YYYY-MM-DD)
        required: false
      - name: max_date
        type: date
        description: End date (YYYY-MM-DD)
        required: false
      - name: min_amount
        type: number
        description: Minimum contribution amount
        required: false
      - name: max_amount
        type: number
        description: Maximum contribution amount
        required: false
      - name: per_page
        type: integer
        description: Results per page (max 100)
        default: 20
    pagination:
      type: cursor
      cursor_param: last_index
      results_path: results
      has_more_path: pagination.pages
    response_mapping:
      entity_type: contribution
      fields:
        donor_name: contributor_name
        donor_employer: contributor_employer
        donor_occupation: contributor_occupation
        donor_city: contributor_city
        donor_state: contributor_state
        recipient_name: committee.name
        recipient_id: committee_id
        amount: contribution_receipt_amount
        date: contribution_receipt_date
        filing_type: receipt_type_full
      entity_extraction:
        - field: donor_name
          entity_type: PERSON
        - field: donor_employer
          entity_type: ORG
        - field: recipient_name
          entity_type: ORG
    output_schema:                    # Typed output for workflow wiring
      type: record_set
      fields:
        - name: donor_name
          type: string
        - name: donor_employer
          type: string
        - name: amount
          type: number
        - name: date
          type: date
        - name: recipient_name
          type: string
        - name: recipient_id
          type: string
        - name: filing_type
          type: string
      metadata_fields:
        - name: count
          type: integer
        - name: has_more
          type: boolean

  search_committees:
    method: GET
    path: /committees/
    description: Search FEC-registered committees
    parameters:
      - name: q
        type: string
        description: Committee name search
        required: false
      - name: committee_type
        type: string
        description: Committee type code
        required: false
      - name: state
        type: string
        description: State abbreviation
        required: false
    response_mapping:
      entity_type: committee
      fields:
        name: name
        id: committee_id
        type: committee_type_full
        party: party_full
        state: state
        treasurer: treasurer_name
      entity_extraction:
        - field: name
          entity_type: ORG
        - field: treasurer
          entity_type: PERSON

  get_committee_detail:
    method: GET
    path: /committee/{committee_id}/
    description: Get detailed committee information
    parameters:
      - name: committee_id
        type: string
        description: FEC committee ID
        required: true
        in: path
    response_mapping:
      entity_type: committee_detail
      fields:
        name: name
        type: committee_type_full
        designation: designation_full
        filing_frequency: filing_frequency
        organization_type: organization_type_full
        connected_orgs: connected_organization_name
```

**Web Scraper Connector Definition (Example — State Corporate Registry):**

```yaml
id: state-corp-registry-ny
name: New York Corporate Registry
type: web_scraper
version: "1.0"
description: Search NY Department of State corporate filings

connection:
  base_url: https://appext20.dos.ny.gov/corp_public
  rate_limit:
    requests_per_second: 2
    delay_between_pages: 3
  user_agent: "Bastion Research Agent"

endpoints:
  search_corporations:
    url_template: "{base_url}/CORPSEARCH.ENTITY_SEARCH_ENTRY"
    method: POST
    description: Search for corporate entities by name
    parameters:
      - name: entity_name
        type: string
        description: Corporation name
        required: true
        form_field: p_search_str
      - name: search_type
        type: string
        description: Search type (BEGINS/CONTAINS/EXACT)
        default: CONTAINS
        form_field: p_search_type
    extraction:
      type: html_table
      selector: "table.searchResults"
      columns:
        - header: "DOS ID"
          field: dos_id
        - header: "Current Entity Name"
          field: entity_name
        - header: "Initial DOS Filing Date"
          field: filing_date
        - header: "County"
          field: county
        - header: "Jurisdiction"
          field: jurisdiction
        - header: "Entity Type"
          field: entity_type
        - header: "Status"
          field: status
      entity_extraction:
        - field: entity_name
          entity_type: ORG
```

### 3. Skill / Playbook

A Skill (or Playbook for multi-step sequences) defines a research strategy. Skills are reusable recipes that chain together data sources, analysis steps, and output formatting.

**Built-in Skills** are the existing Bastion skills (research, knowledge_builder, site_crawl, etc.) that can be assigned to custom agents.

**User-Defined Playbooks** are multi-step workflows that compose tools, LLM tasks, and approval gates into a single invocation. A playbook can be fully deterministic (no LLM), fully LLM-driven, or a hybrid that mixes both — the user decides.

```yaml
id: nonprofit-investigation
name: Nonprofit Investigation Playbook
description: Comprehensive investigation of a nonprofit organization
version: "1.0"
execution_mode: hybrid             # "deterministic", "llm_augmented", or "hybrid"
run_context: interactive           # "interactive" (chat), "background", or "scheduled"

triggers:
  - "investigate nonprofit"
  - "look into [org] nonprofit"
  - "what do we know about [org] foundation"

steps:
  - name: initial_search
    description: Search existing knowledge graph for the organization
    step_type: tool                # Deterministic tool call — no LLM
    action: search_knowledge_graph
    inputs:
      entity_name: "{entity_name}" # From playbook input variable
    params:                         # Static config (not wired from upstream)
      entity_types: [ORG]
      include_related: true
      max_hops: 2
    output_key: existing_knowledge
    # outputs: entities[], relationships[], count

  - name: pull_990_data
    description: Get IRS 990 filing data
    step_type: tool
    action: call_connector
    connector: propublica-nonprofit
    endpoint: search_organizations
    inputs:
      q: "{entity_name}"           # From playbook input variable
    output_key: nonprofit_data
    # outputs: record_set (from connector output_schema)

  - name: pull_fec_data
    description: Search FEC for related political contributions
    step_type: tool
    action: call_connector
    connector: fec-contributions
    endpoint: search_contributions
    inputs:
      contributor_name: "{entity_name}"
    output_key: fec_contributions
    condition: "{nonprofit_data.count} > 0"
    # outputs: record_set with donor_name, amount, date, etc.

  - name: extract_officers
    description: Extract officers and board members from 990 data
    step_type: tool
    action: extract_entities
    inputs:
      data: "{nonprofit_data.results}"  # Wired to upstream step output field
      entity_types: [PERSON]
    params:
      relationship_type: OFFICER_OF
    output_key: officers
    # outputs: entities[], count

  - name: cross_reference_officers
    description: Check if officers appear in FEC data or other nonprofits
    step_type: tool
    action: cross_reference
    inputs:
      dataset_a: "{officers.entities}"          # Wired: entities from extraction
      dataset_b: "{fec_contributions.results}"  # Wired: records from FEC
    params:
      match_strategy: fuzzy_name
    output_key: officer_connections
    # outputs: matches[], unmatched[], count

  - name: review_connections
    description: User reviews discovered connections before graph enrichment
    step_type: approval             # Pauses workflow for human review
    preview_from: officer_connections
    preview_limit: 10
    prompt: "Found {officer_connections.count} officer-to-contribution links. Enrich knowledge graph?"
    timeout_minutes: 1440           # Auto-skip after 24h if unreviewed

  - name: synthesize
    description: Compile findings into structured report
    step_type: llm_task             # LLM analysis / synthesis step
    action: synthesize_report
    inputs:                          # Explicit wiring from all upstream steps
      existing_knowledge: "{existing_knowledge}"
      nonprofit_data: "{nonprofit_data}"
      fec_contributions: "{fec_contributions}"
      officers: "{officers}"
      officer_connections: "{officer_connections}"
    params:
      template: nonprofit_investigation_report
    output_schema:                    # LLM steps declare their output shape
      type: object
      fields:
        - name: report
          type: string
          description: Full investigation report in markdown
        - name: key_findings
          type: string[]
          description: Bullet-point summary of findings
        - name: risk_indicators
          type: string[]
          description: Identified risk flags
    output_key: report

output:
  format: markdown
  sections:
    - title: Organization Overview
      source: nonprofit_data
    - title: Officers & Board Members
      source: officers
    - title: Political Connections
      source: officer_connections
    - title: Related Entities
      source: existing_knowledge
  auto_enrich_graph: true
  save_to: document
```

### 4. Workflow Composition Patterns

The Agent Factory's key architectural insight is that an agent IS a workflow, and users should be able to compose that workflow from three building blocks mixed in any combination:

| Step Type | What It Does | LLM Involved? | Example |
|-----------|-------------|----------------|---------|
| `tool` | Calls a connector, tool, or data operation with specified parameters | No — fully deterministic | Call FEC API, filter results, save to table |
| `llm_task` | Sends data to an LLM for analysis, classification, or synthesis | Yes — structured JSON output | Analyze funding patterns, generate report, classify entities |
| `approval` | Pauses the workflow and shows a preview; resumes on user confirmation | No — human decision | Review connections before graph enrichment, confirm email send |

These compose into five natural workflow patterns:

#### Pattern A: Deterministic Pipeline (No LLM)

A sequence of tool calls that runs without any LLM mediation. The playbook defines exactly which tools to call, with what parameters, in what order. Data flows between steps via variable interpolation.

```yaml
execution_mode: deterministic

steps:
  - name: fetch_contributions
    step_type: tool
    action: call_connector
    connector: fec-contributions
    endpoint: search_contributions
    params:
      contributor_name: "{entity_name}"
      min_amount: 1000
    output_key: contributions

  - name: filter_recent
    step_type: tool
    action: transform_data
    input_key: contributions
    operations:
      - type: filter
        field: date
        operator: gte
        value: "{30_days_ago}"
      - type: sort
        field: amount
        direction: desc
    output_key: recent_contributions

  - name: save_to_workspace
    step_type: tool
    action: save_to_workspace
    input_key: recent_contributions
    config:
      table_name: recent_fec_contributions
      create_if_missing: true
    output_key: save_result
```

This pattern is ideal for **data extraction jobs** — pull from API, transform, load into a table. No LLM tokens consumed. Fully predictable, testable, and auditable.

#### Pattern B: LLM-Augmented Workflow (LLM Decides)

A workflow where an LLM receives the user's query, the available tools, and decides which tools to call and how to interpret results. This is how existing Bastion skills work — the skill defines the boundary (available tools, system prompt) and the LLM operates within it.

```yaml
execution_mode: llm_augmented

steps:
  - name: research
    step_type: llm_task
    action: research_with_tools
    tools:
      - search_web_tool
      - search_documents_tool
      - crawl_web_content_tool
      - fec_search_contributions   # connector-generated tool
      - propublica_search_orgs     # connector-generated tool
    system_prompt: |
      You are a nonprofit research specialist. Use the available tools
      to investigate the organization. Check multiple sources and
      cross-reference findings.
    max_rounds: 3
    output_key: research_findings

  - name: format_report
    step_type: llm_task
    action: synthesize_report
    inputs:
      - research_findings
    template: investigation_report
```

This pattern is ideal for **open-ended research** — the LLM's judgment determines which tools to call and how to synthesize findings.

#### Pattern C: Hybrid Workflow (Mixed)

The most powerful pattern: deterministic steps for predictable data collection, LLM steps for analysis and synthesis, and approval gates for human oversight. The nonprofit investigation playbook above is a hybrid workflow.

This pattern reflects how a human analyst actually works: gather data systematically (deterministic), interpret it (LLM), and get sign-off before acting on it (approval).

#### Pattern D: Background Job (Scheduled / Automated)

Any workflow pattern (A, B, or C) can run as a background job instead of in the chat. Background jobs execute outside the chat session and deliver results to configured output destinations when complete.

```yaml
execution_mode: deterministic      # or hybrid, or llm_augmented
run_context: background            # Not in chat — runs as a background job

steps:
  - name: daily_fec_check
    step_type: tool
    action: call_connector
    connector: fec-contributions
    endpoint: search_contributions
    params:
      min_date: "{yesterday}"
      max_date: "{today}"
      contributor_name: "{monitored_entity}"
    output_key: new_contributions

  - name: check_for_activity
    step_type: tool
    action: transform_data
    input_key: new_contributions
    operations:
      - type: filter
        field: amount
        operator: gt
        value: 0
    output_key: activity

  - name: notify_if_found
    step_type: tool
    action: route_output
    condition: "{activity}.count > 0"
    destinations:
      - type: append_to_existing
        document_id: "{monitoring_log_id}"
      - type: notification
        message: "New FEC activity for {monitored_entity}: {activity.count} contributions"
```

Background jobs can also include approval gates. When a background workflow reaches an `approval` step, it pauses and surfaces the approval request in the user's notification queue. The workflow resumes when the user approves or rejects, even if that happens hours or days later. This is possible because LangGraph checkpointing persists the full workflow state.

#### Pattern E: Monitor Mode (Change-Aware Polling)

A monitor is a specialized background agent that **polls for changes** at a regular interval and **only acts when something new is detected**. Unlike a scheduled job (which runs its full playbook on every trigger), a monitor checks a watermark (last-checked state), detects deltas, and conditionally executes downstream steps only on the new items.

This is the equivalent of [OpenClaw's heartbeat](https://docs.openclaw.ai/automation/cron-vs-heartbeat) concept — periodic awareness with smart suppression.

```yaml
execution_mode: hybrid
run_context: monitor
monitor_config:
  interval: 30m                   # Check every 30 minutes
  active_hours:                   # Optional: only poll during work hours
    start: "08:00"
    end: "22:00"
    timezone: "America/New_York"
  suppress_if_empty: true         # Don't journal or notify if nothing new

steps:
  - name: check_new_files
    step_type: tool
    action: detect_new_files
    inputs:
      folder_id: "{watched_folder_id}"
    params:
      file_types: [jpg, png, webp, pdf]
    output_key: new_files
    # outputs: files[], count, watermark
    # Automatically compares against last watermark and returns only new items

  - name: classify_images
    step_type: llm_task
    action: llm_analyze
    condition: "{new_files.count} > 0"
    inputs:
      files: "{new_files.files}"
    params:
      instructions: |
        For each image, classify it into one of: receipt, invoice, screenshot,
        photo, diagram, other. Return structured classification.
    output_schema:
      type: record_set
      fields:
        - name: file_id
          type: string
        - name: classification
          type: string
        - name: confidence
          type: number
        - name: description
          type: string
    output_key: classifications

  - name: organize_files
    step_type: tool
    action: batch_move_files
    condition: "{classifications.count} > 0"
    inputs:
      files: "{classifications}"
    params:
      routing_rules:
        receipt: "{receipts_folder_id}"
        invoice: "{invoices_folder_id}"
        screenshot: "{screenshots_folder_id}"
    output_key: move_results

  - name: update_watermark
    step_type: tool
    action: set_monitor_watermark
    inputs:
      watermark: "{new_files.watermark}"
```

**How monitors differ from scheduled jobs:**

| | Scheduled (cron) | Monitor |
|---|---|---|
| **Trigger** | Exact time (cron expression) | Interval-based polling (every N minutes) |
| **State awareness** | Stateless — runs full playbook every time | Stateful — tracks watermark, only processes deltas |
| **Empty runs** | Runs and reports "nothing found" | Suppressed silently (no journal, no notification) |
| **Cost** | Full playbook execution on every trigger | Minimal cost when nothing changed (only detection step runs) |
| **Use case** | "Generate Monday report at 9am" | "Watch for new files and classify them" |
| **Active hours** | Runs at the scheduled time regardless | Can be restricted to active hours |

**What monitors can watch:**

| Detection Action | What It Watches | Watermark Type |
|---|---|---|
| `detect_new_files` | Folder for new/modified files since last check | File timestamp cursor |
| `detect_new_data` | Connector endpoint for new records since last check | API cursor / date watermark |
| `detect_new_team_posts` | Team threads for new messages since last check | Post timestamp cursor |
| `detect_folder_changes` | Folder for any changes (adds, edits, deletes) | Folder hash / changelog cursor |
| `detect_new_entities` | Knowledge graph for new entities matching criteria | Entity creation timestamp |

Monitors are ideal for:
- **File inbox processing** — Watch a folder, classify/tag/organize new uploads
- **Data feed monitoring** — Poll APIs for new records, alert on matching criteria
- **Team awareness** — Summarize new team posts, flag action items
- **Knowledge graph maintenance** — Detect new entities that need resolution or linking
- **Research alerting** — Watch for new SEC filings, FEC contributions, or news mentions

See `AGENT_FACTORY_EXAMPLES.md` for detailed monitor use cases.

**Run contexts:**

| Run Context | Behavior | Approval Gates | Results Delivered To |
|-------------|----------|----------------|---------------------|
| `interactive` | Runs in chat, streams progress | Inline in chat conversation | Chat response + configured destinations |
| `background` | Runs outside chat session | Surfaced in notification queue | Configured output destinations + notification |
| `scheduled` | Triggered by cron schedule, runs as background | Surfaced in notification queue (or auto-approve) | Configured output destinations + execution log |
| `monitor` | Periodic polling; runs only when changes detected | Same as scheduled (notification queue or auto-approve) | Configured output destinations + execution log |

Scheduled and monitor workflows have an additional option: `approval_policy: auto_approve` — which allows them to skip approval gates and proceed automatically. This is useful for trusted, well-tested pipelines where human review would just slow down automated monitoring.

### 5. Step I/O Contracts

For users to compose workflows visually, every step must declare what it **accepts** (input schema) and what it **produces** (output schema). These typed contracts are what makes the Workflow Composer usable — instead of guessing what `{pull_fec_data.results[0].amount}` means, the UI shows users exactly which fields are available from each upstream step and lets them wire connections visually.

#### Every Step Has a Contract

Each tool action, connector endpoint, and LLM task has a typed I/O contract:

```
┌─────────────────────────┐       ┌─────────────────────────┐
│  search_contributions    │       │  extract_entities        │
│  (FEC connector)         │       │  (built-in tool)         │
│                          │       │                          │
│  INPUTS:                 │       │  INPUTS:                 │
│    contributor_name: str │──┐    │    data: record[]  ◄─────┤
│    min_date: date        │  │    │    entity_types: str[]   │
│    max_date: date        │  │    │                          │
│    min_amount: number    │  │    │  OUTPUTS:                │
│                          │  │    │    entities: entity[]     │
│  OUTPUTS:                │  │    │    count: integer         │
│    results: record[]  ───┼──┘    │                          │
│      .donor_name: str    │       └─────────────────────────┘
│      .donor_employer: str│
│      .amount: number     │
│      .date: date         │
│      .recipient_name: str│
│    count: integer        │
│    has_more: boolean     │
└─────────────────────────┘
```

#### Connector Endpoints Define Output Schemas

Connectors already define `parameters` (inputs) and `response_mapping` (field names). The output schema makes the shape explicit so the UI can present it:

```yaml
# In a connector endpoint definition
endpoints:
  search_contributions:
    description: Search individual campaign contributions
    parameters:               # ← Input schema (already exists)
      - name: contributor_name
        type: string
        required: false
      - name: min_amount
        type: number
        required: false

    output_schema:             # ← Output schema (NEW)
      type: record_set
      fields:
        - name: donor_name
          type: string
          description: Name of the contributor
        - name: donor_employer
          type: string
          description: Contributor's employer
        - name: amount
          type: number
          description: Contribution amount in USD
        - name: date
          type: date
          description: Date of contribution
        - name: recipient_name
          type: string
          description: Committee receiving the contribution
      metadata_fields:
        - name: count
          type: integer
          description: Total result count
        - name: has_more
          type: boolean
          description: Whether more pages are available
```

#### Built-In Actions Have Standard Contracts

Every built-in action (search_knowledge_graph, extract_entities, cross_reference, transform_data, etc.) publishes a typed I/O contract. These are registered centrally so the Workflow Composer can look up any action's inputs and outputs:

| Action | Input Types | Output Type |
|--------|------------|-------------|
| `call_connector` | Connector-defined parameters | Connector-defined `output_schema` |
| `search_knowledge_graph` | entity_types: `string[]`, max_hops: `integer` | entities: `entity[]`, relationships: `relationship[]` |
| `extract_entities` | data: `record[]`, entity_types: `string[]` | entities: `entity[]`, count: `integer` |
| `cross_reference` | datasets: `record[][]`, match_strategy: `string` | matches: `match[]`, unmatched: `record[]` |
| `transform_data` | data: `record[]`, operations: `operation[]` | Same shape as input (filtered/sorted/aggregated) |
| `search_documents` | query: `string`, max_results: `integer` | documents: `document[]`, count: `integer` |
| `search_web` | query: `string`, max_results: `integer` | results: `web_result[]`, count: `integer` |
| `save_to_workspace` | data: `record[]`, table_name: `string` | rows_inserted: `integer`, table_name: `string` |

LLM task steps define their output schema explicitly in the step definition (since the LLM's output structure varies by task):

```yaml
- name: analyze_patterns
  step_type: llm_task
  action: llm_analyze
  inputs:
    contributions: "{fec_data}"        # Wired to upstream output
  output_schema:                        # Declares what this step produces
    type: object
    fields:
      - name: patterns
        type: string[]
        description: Identified funding patterns
      - name: risk_score
        type: number
        description: Risk assessment (0-1)
      - name: summary
        type: string
        description: Natural language analysis
  output_key: analysis
```

#### Wiring Steps Together in the Workflow Composer

In the UI, users connect steps by linking an upstream step's output fields to a downstream step's input parameters. The Workflow Composer:

1. **Shows available outputs** — When the user selects a step's input field, a dropdown lists all compatible output fields from upstream steps (filtered by type)
2. **Type-checks connections** — A `string[]` output can connect to a `string[]` input but not to a `number` input. Record sets can feed into actions that accept records.
3. **Auto-suggests wiring** — When adding a new step, the Assembly Agent suggests connections based on field names and types (e.g., "connect `fec_data.results` to `extract_entities.data` since both are record arrays")
4. **Validates at save time** — All required inputs must be wired to either an upstream output, a playbook input variable, or a literal value. Unresolved references are flagged as errors.

In the playbook YAML, wiring is expressed as explicit input mappings rather than string-template interpolation:

```yaml
steps:
  - name: pull_fec_data
    step_type: tool
    action: call_connector
    connector: fec-contributions
    endpoint: search_contributions
    inputs:                              # Explicit typed wiring
      contributor_name: "{entity_name}"  # From playbook input variable
    output_key: fec_data
    # Output schema inherited from connector endpoint definition

  - name: filter_large
    step_type: tool
    action: transform_data
    inputs:
      data: "{fec_data.results}"         # Wired to upstream step's output field
    operations:
      - type: filter
        field: amount
        operator: gt
        value: 10000
    output_key: large_contributions
    # Output schema: same shape as input (filtered record set)

  - name: extract_people
    step_type: tool
    action: extract_entities
    inputs:
      data: "{large_contributions}"      # Wired to filtered results
      entity_types: [PERSON]
    output_key: people
    # Output schema: standard extract_entities contract

  - name: analyze
    step_type: llm_task
    action: llm_analyze
    inputs:
      contributions: "{large_contributions}"
      entities: "{people.entities}"
    output_schema:
      type: object
      fields:
        - name: patterns
          type: string[]
        - name: summary
          type: string
    output_key: analysis
```

The `inputs` map replaces the older pattern of burying references inside action-specific fields like `input_key` or `params`. Every step uses `inputs` for its data dependencies, making the wiring explicit and inspectable. The `params` field is still available for static configuration values that aren't wired from upstream steps.

**Why this matters:** Without typed I/O contracts, building a workflow requires knowing the internal structure of every tool's return value. With them, the UI can present a Zapier-like experience where users see "Step A produces these fields → connect them to Step B's inputs" without reading documentation.

### 6. Output Destinations

Custom agents must be able to route their output to multiple destinations. This is critical — research results are only useful if they land somewhere actionable.

**Output Destination Types:**

| Destination | Description | Format Support | Use Case |
|-------------|-------------|----------------|----------|
| **Chat Response** | Inline in the conversation | Markdown, images, charts | Interactive Q&A |
| **Document** | Save as a document in the user's library | Markdown, Org-mode, HTML | Reports, dossiers |
| **Folder** | Save into a specific folder hierarchy | Any document format | Organized research projects |
| **Data Workspace Table** | Insert rows into a Data Workspace table | Structured rows (JSON) | Tabular data, records, timelines |
| **Data Workspace Database** | Create/update tables in a workspace DB | Schema + rows | Full datasets |
| **Knowledge Graph** | Create/update entities and relationships | Neo4j entities + edges | Persistent intelligence |
| **File Export** | Generate downloadable files | CSV, JSON, XLSX, PDF | External sharing |
| **Append to Existing** | Append findings to an existing document | Matching source format | Ongoing investigations |

**Output Configuration Schema:**

```
OutputConfig
├── destinations: List[OutputDestination]
│   ├── type: enum (chat, document, folder, data_workspace_table,
│   │              data_workspace_db, knowledge_graph, file_export,
│   │              append_to_existing)
│   ├── config: Dict
│   │   ├── folder_id: UUID          (for folder destination)
│   │   ├── document_id: UUID        (for append destination)
│   │   ├── workspace_id: UUID       (for data workspace destinations)
│   │   ├── database_id: UUID        (for data workspace table)
│   │   ├── table_name: str          (for data workspace table)
│   │   ├── create_if_missing: bool  (auto-create table/folder)
│   │   ├── schema_mapping: Dict     (field → column mapping)
│   │   └── filename_template: str   (for file export: "investigation_{entity}_{date}.md")
│   ├── format: str                  (markdown, json, csv, org, xlsx, pdf)
│   ├── condition: str               (optional: only route here if condition met)
│   └── transforms: List[Transform]  (optional: reshape data before output)
│
├── default_format: str              (fallback format)
├── auto_save: bool                  (save without asking)
├── auto_enrich_graph: bool          (always feed entities to Neo4j)
└── deduplication: bool              (avoid duplicate entries)
```

**Output Routing Examples:**

```yaml
# Example 1: Investigation that saves a report AND populates a database
output_config:
  auto_enrich_graph: true
  destinations:
    - type: chat
      format: markdown
      config:
        summary_only: true

    - type: document
      format: markdown
      config:
        folder_id: "research-reports"
        filename_template: "investigation_{entity_name}_{date}.md"
        create_if_missing: true

    - type: data_workspace_table
      format: structured
      config:
        workspace_id: "nonprofit-research"
        database_id: "investigations"
        table_name: contributions
        create_if_missing: true
        schema_mapping:
          donor_name: TEXT
          recipient_name: TEXT
          amount: REAL
          date: TIMESTAMP
          source: TEXT
          filing_type: TEXT

    - type: knowledge_graph
      config:
        auto_enrich_graph: true
        entity_types: [PERSON, ORG]
        relationship_types: [FUNDS, OFFICER_OF, CONTROLS]
```

```yaml
# Example 2: Ongoing monitoring that appends to a living document
output_config:
  destinations:
    - type: append_to_existing
      format: org
      config:
        document_id: "my-investigation-log"
        heading_level: 2
        timestamp: true

    - type: data_workspace_table
      format: structured
      config:
        table_name: daily_findings
        deduplication_key: [entity_name, date, source]
```

```yaml
# Example 3: Data extraction that produces downloadable files
output_config:
  destinations:
    - type: chat
      format: markdown
      config:
        summary_only: true
        include_download_link: true

    - type: file_export
      format: csv
      config:
        filename_template: "fec_contributions_{query}_{date}.csv"

    - type: file_export
      format: xlsx
      config:
        filename_template: "fec_contributions_{query}_{date}.xlsx"
        include_charts: true
```

### 7. Knowledge Accumulation Loop

The Knowledge Accumulation Loop is what makes custom agents get smarter over time. Every query result, regardless of the agent that produced it, feeds back into the shared knowledge infrastructure.

**Loop Flow:**

```
Agent Query Result
       │
       ▼
┌─────────────────────┐
│  Entity Extraction    │  spaCy NER + connector-defined entity mappings
│  (automatic)         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Entity Resolution    │  Fuzzy match against existing graph entities
│  (automatic)         │  Canonicalize names, merge duplicates
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Relationship        │  Extract typed relationships from context
│  Extraction          │  FUNDS, CONTROLS, OFFICER_OF, etc.
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Graph Enrichment    │  Store new entities + relationships in Neo4j
│                      │  Store embeddings in Qdrant
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Discovery           │  Run graph algorithms on affected subgraph
│  Propagation         │  Community detection, centrality updates
│                      │  Flag new patterns for user attention
└─────────────────────┘
```

**What This Enables:**

- **Cross-agent learning**: The "Nonprofit Investigator" agent discovers that Person X is an officer of Foundation Y. Later, the "Political Tracker" agent is asked about Person X and already knows about the foundation connection.
- **Pattern emergence**: After 50 queries across different agents, the graph reveals that 8 seemingly-unrelated foundations share 3 board members — a cluster the user never explicitly asked about.
- **Diminishing effort**: Early queries require full API calls and web scraping. Later queries find most entities already in the graph and only need to fill gaps.

### 8. @Mention Invocation Model

Custom agents are **not auto-routed** like built-in skills. Built-in skills use trigger keywords and semantic matching to auto-route queries — this works well for general-purpose capabilities but creates confusion when multiple custom agents overlap in domain. Instead, custom agents use an explicit **@mention** invocation model.

#### How It Works

Every custom agent has a unique **handle** — a short, kebab-case identifier like `nonprofit-investigator` or `fec-tracker`. When the user types `@` in the chat sidebar, an autocomplete dropdown shows their available agents (personal and team-shared), each with its name, icon, and description.

```
User types: @non
            ┌────────────────────────────────────────────┐
            │ 🔍 @nonprofit-investigator                  │
            │    Tracks money flows through nonprofits    │
            │                                             │
            │ 📊 @nonprofit-990-tracker                    │
            │    Monitors IRS 990 filings quarterly       │
            │                                             │
            │ 👥 @team: @political-tracker    (shared)     │
            │    Tracks campaign contributions             │
            └────────────────────────────────────────────┘

User selects: @nonprofit-investigator Investigate the Omidyar Foundation
```

The @mention is parsed from the message and routed directly to the specified agent's `CustomAgentEngine` — no trigger matching, no intent classification, no routing ambiguity. The user knows exactly which agent will handle the request.

#### Why Not Auto-Routing?

| Concern | Auto-routing (built-in skills) | @mention (custom agents) |
|---------|-------------------------------|--------------------------|
| Routing confidence | High — small, curated set of skills | Low — N user-created agents with overlapping domains |
| User mental model | "Bastion figures it out" | "I'm asking this specific agent" |
| Accountability | Opaque — which skill handled this? | Explicit — the agent I mentioned |
| Disambiguation | Requires ranked fallback stack | Not needed — explicit selection |
| Scheduled/background use | N/A — always in chat context | Agent referenced by profile ID (no @mention needed) |

**Trigger keywords are still available** as an optional feature. If a user wants their `@nonprofit-investigator` to also activate on "investigate nonprofit", they can configure triggers. But triggers are opt-in for custom agents, not the default invocation path.

#### Chat Sidebar Integration

The chat sidebar gains a new section: **My Agents**. This lists the user's custom agents (and team-shared agents) with quick-action buttons:

```
┌──────────────────────────┐
│  MY AGENTS                │
│                           │
│  🔍 Nonprofit Investigator│  [@] [▶ Run] [📋 Journal]
│  📊 FEC Tracker           │  [@] [▶ Run] [📋 Journal]
│  🌐 SEC Filing Monitor    │  [@] [▶ Run] [📋 Journal]
│                           │
│  TEAM: Research Unit       │
│  👥 Political Tracker     │  [@] [📋 Journal]
│                           │
│  [+ Create New Agent]     │
└──────────────────────────┘
```

- **[@]** — Inserts `@handle ` into the chat input, ready for the user to type their query
- **[Run]** — Triggers the agent's default playbook (useful for scheduled-style agents that the user wants to run on-demand)
- **[Journal]** — Opens the agent's work journal (see next section)

### 9. Work Journal & Activity Awareness

Custom agents maintain a **work journal** — a persistent, human-readable log of everything the agent has done. The journal is distinct from the raw execution log (which is for system monitoring): the journal is written by the agent in natural language, summarizing its work for the user.

#### What Gets Journaled

Every agent execution (interactive, background, or scheduled) automatically produces a journal entry:

```
┌─────────────────────────────────────────────────────────────┐
│  📋 WORK JOURNAL — @nonprofit-investigator                   │
│                                                              │
│  Today, Feb 14 2026                                          │
│  ─────────────────                                           │
│  14:32  Investigated Omidyar Foundation (interactive)         │
│         • Pulled 23 FEC contributions, 4 IRS 990 filings     │
│         • Extracted 12 officers, cross-referenced with FEC    │
│         • Found 3 officer-contribution links (high confidence)│
│         • Report saved to Research Reports/omidyar_2026.md    │
│         • 5 new entities added to knowledge graph             │
│                                                              │
│  08:00  Daily FEC monitoring run (scheduled)                  │
│         • Checked 5 monitored entities for new contributions  │
│         • Found 2 new contributions for Entity: Koch Network  │
│         • Results appended to daily_findings table             │
│         • No approval gates triggered                         │
│                                                              │
│  Yesterday, Feb 13 2026                                       │
│  ─────────────────────                                        │
│  16:45  Investigated Open Society Foundations (interactive)    │
│         • ...                                                 │
└─────────────────────────────────────────────────────────────┘
```

#### Asking Agents About Their Work

Users can ask an agent about its own activity using natural language:

```
User: @nonprofit-investigator What have you done today?

Agent: Today I've completed 2 tasks:

1. **Omidyar Foundation Investigation** (2:32 PM, interactive)
   - Pulled 23 FEC contributions and 4 IRS 990 filings
   - Identified 3 officer-to-contribution connections
   - Saved report to Research Reports/omidyar_2026.md
   - Added 5 new entities to the knowledge graph

2. **Daily FEC Monitoring** (8:00 AM, scheduled)
   - Checked 5 monitored entities
   - Found 2 new Koch Network contributions
   - Results appended to daily_findings table
```

This works because the agent has a dedicated `query_journal` tool that searches its own journal entries. The journal is queryable by date range, task type, entities mentioned, and status.

**Example queries:**
- `@fec-tracker What did you find last week?`
- `@nonprofit-investigator How many entities have you added to the graph?`
- `@sec-monitor When was your last successful run?`
- `@political-tracker Show me everything you've found about Koch Industries`

#### Journal Tools

Agents get two new tools automatically:

| Tool | Description | When Used |
|------|-------------|-----------|
| `write_journal_entry` | Write a structured journal entry summarizing work done | Automatically called at the end of every execution |
| `query_journal` | Search the agent's journal by date, keyword, entity, status | When users ask the agent about its past work |

The `write_journal_entry` tool is called automatically by the workflow engine after each execution completes. It receives the execution context (steps run, entities found, outputs produced, errors encountered) and generates a human-readable summary. For LLM-augmented workflows, the LLM generates the summary; for deterministic pipelines, the engine generates a structured summary from step results.

### 10. Team Integration

Custom agents can be **shared with teams** the user belongs to. When shared, team members can invoke the agent via @mention, view its journal, and benefit from its knowledge graph contributions. The agent owner controls what level of access the agent has to team resources.

#### Sharing Model

```
Agent Owner (creates and configures the agent)
       │
       ├── Shares with Team A
       │   ├── Team members can @mention the agent
       │   ├── Team members see the agent in their sidebar
       │   ├── Agent can access Team A's shared folders (if enabled)
       │   └── Agent can read/write Team A's conversation threads (if enabled)
       │
       └── Shares with Team B
           ├── Team members can @mention the agent
           ├── Agent has read-only access to Team B's files (per config)
           └── No access to Team B's conversation threads
```

#### Team Resource Access

When an agent is shared with a team, the owner configures what team resources the agent can access:

| Access Level | What It Grants | Use Case |
|---|---|---|
| **Invoke only** (default) | Team members can @mention the agent and see its journal. Agent has no access to team files or posts. | Personal agent shared for others to use |
| **Team file access** | Agent can search and read documents in the team's shared folders. | Research agent that needs team context |
| **Team post access** | Agent can read team conversation threads and write responses/summaries to team channels. | Monitoring agent that reports to the team |
| **Full team access** | Both file and post access. | Team research assistant |

#### Team Interaction Tools

When team access is enabled, the agent gains additional tools:

| Tool | Description | Requires |
|------|-------------|----------|
| `search_team_files` | Search documents in the team's shared folders | `team_file_access: true` |
| `read_team_file` | Read a specific document from team folders | `team_file_access: true` |
| `search_team_posts` | Search team conversation threads by keyword/date/author | `team_post_access: true` |
| `write_team_post` | Write a message to a team conversation thread | `team_post_access: true` |
| `summarize_team_thread` | Generate a summary of a team conversation thread | `team_post_access: true` |

These tools are only available when the agent's `team_config` grants the appropriate access. The tool pack is dynamically assembled based on the agent's permissions — an agent with `team_file_access: true` but `team_post_access: false` will have file tools but not post tools.

#### Team Journal Visibility

When an agent is shared with a team:
- **Team members can view the agent's journal** — providing transparency into what the shared agent has been doing
- **Journal entries are scoped** — a team member sees entries from their own invocations plus entries from scheduled/background runs, but not entries from other team members' private invocations (unless the owner enables full journal visibility)
- **Team-context entries are tagged** — journal entries from team-scoped work (accessing team files, writing team posts) are marked with the team name for auditability

#### Security & Permissions

- **Owner controls sharing** — Only the agent owner can share with teams or modify team permissions
- **Team membership required** — An agent can only be shared with teams the owner belongs to
- **Credential isolation** — Connector credentials (API keys) are NEVER shared. The agent uses the owner's credentials regardless of who invokes it. If the team needs different credentials, the team should create their own agent.
- **RLS enforcement** — All team file/post access goes through existing Row-Level Security policies on the teams infrastructure
- **Audit trail** — All team resource access is logged in the agent's journal and the team's activity log

---

## Entity Resolution Service

Entity resolution is the critical capability that makes cross-source intelligence possible. Without it, "Pierre Omidyar", "P. Omidyar", "Omidyar Network", and "The Omidyar Group" are four unrelated entities. With it, they're one family of connected entities.

### Resolution Pipeline

```
Raw Entity (from any source)
       │
       ▼
┌─────────────────────┐
│  1. Normalization     │  Strip Inc/LLC/Foundation, normalize whitespace,
│                      │  handle name variants (Robert/Bob, William/Bill)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Candidate Search  │  Fuzzy search existing graph entities
│                      │  Jaro-Winkler + Levenshtein + phonetic encoding
│                      │  Search by name, aliases, external IDs
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Scoring           │  Score each candidate:
│                      │  - Name similarity (0-1)
│                      │  - Type match (PERSON↔PERSON = 1, PERSON↔ORG = 0)
│                      │  - Context overlap (shared documents, relationships)
│                      │  - External ID match (FEC ID, EIN, etc.)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Decision          │  Score > 0.95: Auto-merge (same entity)
│                      │  Score 0.7-0.95: Flag for user review
│                      │  Score < 0.7: Create new entity
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Merge / Create    │  Merge: update canonical entity, add alias
│                      │  Create: new entity node with provenance
└─────────────────────┘
```

### External ID Linking

When connectors return external IDs (FEC committee ID, IRS EIN, SEC CIK), these are stored as properties on entity nodes and enable exact cross-source matching:

```
Entity: "Americans for Prosperity"
  ├── type: ORG
  ├── canonical_name: "Americans for Prosperity"
  ├── aliases: ["AFP", "Americans For Prosperity Inc"]
  ├── external_ids:
  │   ├── fec_committee_id: "C00523514"
  │   ├── irs_ein: "04-3568784"
  │   └── opensecrets_id: "AFP"
  └── relationships:
      ├── FUNDED_BY → "Koch Industries" (confidence: 0.92)
      ├── OFFICER: "Tim Phillips" (role: president, source: 990)
      └── RELATED_TO → "Americans for Prosperity Foundation" (confidence: 0.98)
```

---

## Assembly Agent

The Assembly Agent is an AI assistant that helps users build custom agents through the Agent Factory UI. It understands available connectors, skills, and output formats, and can guide a non-technical user through the process of creating a sophisticated research agent.

### Capabilities

1. **Connector Discovery**: "I want to look up campaign contributions" → suggests FEC API connector, helps configure API key
2. **Skill Recommendation**: "I need to investigate nonprofits" → suggests relevant playbooks, offers to create custom ones
3. **Output Planning**: "I want a spreadsheet of all contributions" → configures Data Workspace table output with appropriate schema
4. **Testing**: Executes sample queries against configured connectors to verify they work
5. **Iterative Refinement**: "The results don't include employer information" → adjusts response field mappings

### Interaction Pattern

```
User: "I want to build an agent that can research dark money in politics"

Assembly Agent: "I can help with that. Here's what I'd recommend:

Data Sources:
  ✅ FEC Contributions API (campaign finance records)
  ✅ ProPublica Nonprofit API (990 tax filings)
  ⬜ OpenCorporates (corporate registry — needs API key)
  ⬜ SEC EDGAR (corporate filings — free, no key needed)

Skills:
  ✅ Multi-round research (built-in)
  ✅ Entity resolution (built-in)
  📝 Nonprofit Investigation Playbook (I can create this)
  📝 Money Flow Tracing Playbook (I can create this)

Output:
  Where should results go?
  • Chat (always)
  • Save reports to a folder?
  • Build a database of contributions?
  • Feed into knowledge graph? (recommended)

Shall I set this up? I'll need your FEC API key to get started."
```

### Agent Definition (LangGraph)

The Assembly Agent is itself a LangGraph agent that:

- Has access to the Connector Registry (read-only) to know what connectors exist
- Has access to the Skill Registry (read-only) to know what skills/playbooks exist
- Can create and validate connector configurations
- Can create playbook definitions
- Can test connector endpoints with sample queries
- Stores the resulting Agent Profile in PostgreSQL
- Has access to output destination configuration tools

---

## Graph Intelligence Capabilities

Custom agents benefit from graph algorithms that run on the shared Neo4j knowledge graph. These algorithms surface patterns that no single query would reveal.

### Available Algorithms

| Algorithm | Purpose | Agent Factory Use |
|-----------|---------|-------------------|
| **PageRank** | Find most influential entities | Identify root funders, key power brokers |
| **Community Detection** (Louvain) | Group related entities | Discover funding networks, family foundations |
| **Betweenness Centrality** | Find pass-through entities | Detect shell organizations, intermediaries |
| **Shortest Path** | Trace connections between entities | "How is Foundation A connected to Politician B?" |
| **Weakly Connected Components** | Find isolated clusters | Identify independent funding networks |
| **Node Similarity** | Find similar entities | "What other orgs look like this one?" |

### Pass-Through Detection

The signature capability described in the motivating use case: identifying entities that exist primarily to obscure funding sources.

**Detection Heuristics:**

1. **High betweenness, low degree**: Entity sits on many paths but has few direct connections → likely a pass-through
2. **Asymmetric flow**: Receives from few sources, distributes to many (or vice versa) → likely a conduit
3. **Timing correlation**: Created shortly before a major funding event → possible shell entity
4. **Name patterns**: Generic names ("Americans for Progress Fund") with minimal public presence
5. **Shared infrastructure**: Same registered agent, address, or officers as known pass-throughs

**Network Simplification:**

Once pass-throughs are identified, the graph can be simplified:

```
Full graph: Donor A → Shell 1 → Shell 2 → PAC → Candidate
Simplified: Donor A → Candidate (via 2 intermediaries)
```

This is the "80% edge/node reduction" capability — collapsing known family-of-foundations into single nodes while preserving the detail for drill-down.

---

## Data Workspace Integration

The Data Workspace remains a critical component, serving as both an output destination and an analytical engine for custom agents.

### As Output Destination

Custom agents can route structured data directly into Data Workspace tables:

- **Contribution records** → Workspace table with donor, recipient, amount, date columns
- **Officer lists** → Workspace table with name, title, organization, term columns
- **Entity timelines** → Workspace table with entity, event, date, source columns

### As Analytical Engine

The Data Workspace's analytical capabilities (planned and existing) become tools available to custom agents:

- **Statistical analysis**: Correlation between donation amounts and election outcomes
- **Anomaly detection**: Unusual contribution patterns (spikes, round numbers)
- **Clustering**: Group donors by contribution patterns (K-means on amount/frequency/timing)
- **Time-series analysis**: Detect seasonal patterns in political giving
- **Visualization**: Generate charts and graphs from extracted data

### Schema Auto-Generation

When a connector returns structured data, the system can automatically generate a Data Workspace table schema:

```
FEC API returns: {contributor_name, contributor_employer, amount, date, committee_name}
                            ↓
Auto-generates table schema:
  contributor_name  TEXT
  contributor_employer  TEXT
  amount  REAL
  date  TIMESTAMP
  committee_name  TEXT
  _source  TEXT (auto-added: which connector produced this row)
  _query  TEXT (auto-added: what query produced this row)
  _timestamp  TIMESTAMP (auto-added: when this was retrieved)
```

---

## Implementation Phases

### Phase 0: Foundation — Agent Profile Schema & CRUD

**Estimated Complexity:** Medium
**Prerequisites:** None
**Expected Impact:** Enables all subsequent phases

**Deliverables:**
- PostgreSQL schema for Agent Profiles, Connector definitions, Skill/Playbook definitions
- Backend CRUD API endpoints for all three
- Basic frontend list/detail views for Agent Profiles
- Migration path from existing static skill definitions

**Database Tables:**
```sql
-- Core tables
agent_profiles          -- User-created agent configurations
agent_data_sources      -- Connector bindings per agent
agent_skills            -- Skill/playbook bindings per agent
agent_output_configs    -- Output routing per agent

-- Connector registry
data_source_connectors  -- Connector definitions (YAML stored as JSONB)
connector_templates     -- Pre-built connector templates (FEC, 990, etc.)

-- Skill registry
custom_playbooks        -- User-defined multi-step playbooks (YAML as JSONB)
playbook_steps          -- Individual steps within playbooks

-- Execution tracking
agent_execution_log     -- History of agent runs
agent_discoveries       -- Entities/relationships discovered per run
```

---

### Phase 1: Data Source Connector Framework

**Estimated Complexity:** High
**Prerequisites:** Phase 0
**Expected Impact:** Very High — enables external data access

**Deliverables:**
- Connector definition parser (YAML → tool generation)
- REST API connector runtime (handles auth, pagination, rate limiting, response mapping)
- Web scraper connector runtime (Crawl4AI integration with selector-based extraction)
- File parser connector runtime (CSV, JSON, XML with schema mapping)
- Connector testing framework (validate endpoints, check auth, sample data)
- Frontend connector configuration UI
- 3-5 pre-built connector templates:
  - FEC Campaign Finance API
  - ProPublica Nonprofit Explorer API
  - SEC EDGAR Full-Text Search
  - OpenCorporates API
  - IRS 990 Bulk Data (XML file parser)

**Technical Architecture:**

The Connector Framework generates LangGraph tools dynamically from connector definitions:

```
Connector YAML Definition
         │
         ▼
┌─────────────────────┐
│  Connector Parser     │  Validate YAML, extract endpoints
└──────────┬──────────┘
         │
         ▼
┌─────────────────────┐
│  Tool Generator       │  For each endpoint, generate:
│                      │  - Tool function with typed parameters
│                      │  - Response parser with field mapping
│                      │  - Entity extraction from response
│                      │  - Pagination handler
└──────────┬──────────┘
         │
         ▼
┌─────────────────────┐
│  Dynamic Tool         │  Register generated tools in the
│  Registry             │  tool pack for this agent profile
└─────────────────────┘
```

---

### Phase 2: Dynamic Tool & Skill Registry

**Estimated Complexity:** High
**Prerequisites:** Phase 0, Phase 1
**Expected Impact:** High — enables runtime extensibility

**Deliverables:**
- Database-backed tool registry (supplements static `TOOL_PACKS`)
- Database-backed skill registry (supplements static skill definitions)
- Runtime tool registration without restart
- Tool vectorization for dynamic tools (integrate with existing `tool_vector_store.py`)
- Skill hot-reload when playbooks are created/modified
- Per-agent tool access control
- Frontend skill/playbook builder UI

**Key Change:** The existing `SkillRegistry` and `TOOL_PACKS` become the "built-in" layer, and the dynamic registry sits on top:

```
Tool Resolution Order:
  1. Agent Profile tool packs (from connector-generated tools)
  2. Dynamic tool registry (user-defined)
  3. Static TOOL_PACKS (built-in)
  
Skill Resolution Order:
  1. Agent Profile playbooks (user-defined multi-step)
  2. Dynamic skill registry (user-defined single skills)
  3. Static skill definitions (built-in)
```

---

### Phase 3: Entity Resolution Service

**Estimated Complexity:** High
**Prerequisites:** Existing Neo4j infrastructure
**Expected Impact:** Very High — enables cross-source intelligence

**Deliverables:**
- Entity Resolution Service (`backend/services/entity_resolution_service.py`)
- Name normalization (strip suffixes, handle aliases, phonetic encoding)
- Fuzzy matching (Jaro-Winkler, Levenshtein, n-gram)
- External ID linking (FEC IDs, EINs, CIK numbers)
- Confidence scoring and threshold configuration
- Human-in-the-loop review UI for ambiguous matches
- Entity merge operations in Neo4j (canonical entity + alias tracking)
- Integration with connector response_mapping entity_extraction

**This phase builds on the existing Neo4j Knowledge Graph Improvements plan** (see `backend/docs/NEO4J_KNOWLEDGE_GRAPH_IMPROVEMENTS.md`) and extends it with cross-source resolution capabilities.

---

### Phase 4: Knowledge Accumulation Loop

**Estimated Complexity:** Medium-High
**Prerequisites:** Phase 1 (connectors), Phase 3 (entity resolution)
**Expected Impact:** Very High — makes the system self-improving

**Deliverables:**
- Post-research enrichment pipeline (runs after every agent query)
- Automatic entity extraction from connector results
- Entity resolution against existing graph
- Typed relationship extraction (FUNDS, CONTROLS, OFFICER_OF)
- Provenance tracking (which source, which query, when)
- Staleness detection (flag data older than N days for refresh)
- Discovery log (what new entities/relationships were found per query)

---

### Phase 5: Assembly Agent

**Estimated Complexity:** High
**Prerequisites:** Phases 0-2 (profile schema, connectors, dynamic registry)
**Expected Impact:** High — makes the system accessible to non-developers

**Deliverables:**
- Assembly Agent (new LangGraph agent extending BaseAgent)
- Connector discovery and recommendation logic
- Skill/playbook recommendation logic
- Interactive connector configuration (test endpoints, validate auth)
- Playbook generation from natural language descriptions
- Output destination configuration assistance
- Integration with Agent Factory UI

---

### Phase 6: Agent Factory UI

**Estimated Complexity:** High
**Prerequisites:** All previous phases
**Expected Impact:** Very High — the complete user experience

**Deliverables:**
- Agent Factory page in frontend (new top-level section)
- Data Sources panel (browse, configure, test connectors)
- Skills & Playbooks panel (browse, create, edit playbooks)
- Output Destinations panel (configure where results go)
- Test Panel (interactive testing with live queries)
- Agent Profile management (create, edit, duplicate, archive)
- Execution history and discovery log viewer
- Knowledge graph visualization for agent-specific discoveries

---

### Phase 7: Graph Intelligence & Multi-Agent Collaboration

**Estimated Complexity:** Very High
**Prerequisites:** Phases 3-4 (entity resolution, knowledge loop)
**Expected Impact:** Revolutionary — autonomous pattern detection

**Deliverables:**
- Graph algorithm integration (PageRank, community detection, centrality)
- Pass-through entity detection
- Network simplification and visualization
- Money flow path queries
- Cross-agent discovery sharing
- Agent-to-agent handoff ("the nonprofit investigator found this person, hand to political tracker")
- Automated pattern alerting ("new cluster detected in funding network")

---

## Relationship to Existing Capabilities

### What Gets Enhanced

| Existing Feature | Enhancement |
|---|---|
| Research Agent (multi-round) | Becomes the default runtime for custom agent research playbooks |
| Data Workspace | Becomes a primary output destination + analytical engine for custom agents |
| Neo4j Knowledge Graph | Receives automatic enrichment from every custom agent query |
| Qdrant Vector Store | Stores embeddings for connector results, enabling semantic search across external data |
| Web Crawling (Crawl4AI) | Becomes the runtime for web_scraper connector type |
| Web Search (SearXNG) | Available as an existing_tool connector for any custom agent |
| Tool Discovery (vector) | Extended to include dynamically-registered connector tools |
| Skill System | Extended with database-backed skills and user-defined playbooks |
| LangGraph Runtime | Executes custom agent profiles with dynamic tool/skill binding |

### What Gets Replaced

| Replaced | By |
|---|---|
| Static `TOOL_PACKS` (as sole source) | Dynamic Tool Registry (with static as built-in layer) |
| Static skill definitions (as sole source) | Dynamic Skill Registry (with static as built-in layer) |
| Code-defined agents only | Agent Profiles (database-backed, user-created) |
| Data Workspace Analytics Enhancements (standalone doc) | Integrated as analytical tools within Agent Factory |

### What Stays Unchanged

- Core LangGraph agent architecture (BaseAgent, workflows, checkpointing)
- gRPC microservice communication pattern
- Frontend React architecture
- Docker Compose deployment model
- Authentication and authorization
- Document management system
- Chat interface (extended with agent profile selection)

---

## Security Considerations

### Data Source Access Control

- Connector credentials (API keys) stored encrypted in PostgreSQL
- Per-agent credential isolation (agent A cannot use agent B's API keys)
- Rate limiting enforced at connector level
- User must explicitly grant each connector access to their agent

### Output Destination Permissions

- Agents can only write to folders/workspaces owned by the user
- Data Workspace RLS (Row-Level Security) enforced on all writes
- Knowledge graph writes scoped to user's namespace
- File exports respect user's storage quotas

### Entity Resolution Safety

- Auto-merge only above high confidence threshold (0.95+)
- Ambiguous matches flagged for human review
- Merge operations are reversible (audit log)
- Entity provenance always tracked

### Assembly Agent Guardrails

- Cannot execute connectors — only configure them
- Cannot access other users' agent profiles
- Cannot modify built-in skills or connectors
- All generated configurations validated before storage

---

## Success Metrics

### User Adoption

- Number of custom agent profiles created per user
- Number of distinct connector types configured
- Playbook creation rate (user-defined vs. built-in usage)
- Agent profile duplication/sharing rate

### Intelligence Quality

- Entity resolution accuracy (precision/recall on known entity pairs)
- Knowledge graph growth rate (new entities/relationships per week)
- Cross-source entity linking rate (entities matched across 2+ sources)
- Graph algorithm insight rate (patterns detected automatically)

### System Performance

- Connector query latency (p50, p95, p99)
- Entity resolution throughput (entities resolved per second)
- Knowledge accumulation overhead (time added per query for enrichment)
- Dynamic tool registration latency (time from profile save to tool availability)

### Research Effectiveness

- Queries answered from existing graph vs. requiring fresh API calls (cache hit rate)
- Research depth (average graph hops used in answers)
- Discovery compounding (new entities found per query over time — should increase then plateau)
- User time-to-insight (time from question to actionable answer)

---

## Conclusion

The Agent Factory transforms Bastion from a developer-maintained AI platform into a user-extensible intelligence system. By making data source connectors declarative, workflows composable from deterministic tools, LLM tasks, and approval gates, output destinations configurable, and the whole process AI-assisted, we enable domain experts to build research agents that rival what previously required custom engineering.

The key differentiator is the Knowledge Accumulation Loop: every agent, every query, every result enriches a shared knowledge graph that benefits all agents. This compound learning effect means the system's value grows superlinearly with usage — the more questions you ask, the smarter every future question becomes.

Combined with entity resolution, graph algorithms, and pass-through detection, this system enables the kind of investigative intelligence work (tracking money flows through shell organizations, identifying root funders behind networks of foundations, mapping political influence networks) that currently requires either expensive commercial tools or significant custom engineering.

The Data Workspace, rather than being a standalone analytical engine, becomes one of several powerful output destinations — and its analytical capabilities (statistical functions, clustering, anomaly detection, visualization) become tools that custom agents can leverage as part of their research workflows.

This is not incremental improvement. This is a new category of capability.
