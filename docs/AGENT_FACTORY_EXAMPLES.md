# Agent Factory: Use Case Examples

**Document Version:** 1.0
**Last Updated:** February 14, 2026
**Companion to:** `AGENT_FACTORY.md`, `AGENT_FACTORY_TECHNICAL_GUIDE.md`, `AGENT_FACTORY_TOOLS.md`, `AGENT_FACTORY_TOOLS.md` (tool catalog)

---

This document provides concrete, end-to-end examples of custom agents built with the Agent Factory. Each example shows the playbook YAML, the run context, the invocation model, and the expected behavior. Examples are grouped by pattern type and escalate in complexity.

---

## Pattern A: Deterministic Pipelines (No LLM)

### Example 1: FEC Contribution Extractor

**What it does:** Pulls FEC campaign contribution records for a given entity, filters by amount, saves to a Data Workspace table. Zero LLM tokens consumed.

**Handle:** `@fec-extractor`

```yaml
id: fec-contribution-extractor
name: FEC Contribution Extractor
handle: fec-extractor
execution_mode: deterministic
run_context: interactive

inputs:
  - name: entity_name
    type: entity_name
    description: Person or organization to search
    required: true
    extract_from: query
  - name: min_amount
    type: number
    description: Minimum contribution amount
    required: false
    default: 1000

steps:
  - name: pull_contributions
    step_type: tool
    action: call_connector
    connector: fec-contributions
    endpoint: search_contributions
    inputs:
      contributor_name: "{entity_name}"
      min_amount: "{min_amount}"
    output_key: fec_data

  - name: filter_recent
    step_type: tool
    action: transform_data
    inputs:
      data: "{fec_data.results}"
    operations:
      - type: sort
        field: amount
        direction: desc
    output_key: sorted_contributions

  - name: save_to_table
    step_type: tool
    action: save_to_workspace
    inputs:
      data: "{sorted_contributions}"
    params:
      table_name: fec_contributions
      create_if_missing: true
    output_key: save_result

output:
  format: markdown
  auto_enrich_graph: true
  destinations:
    - type: chat
      format: markdown
    - type: data_workspace_table
      config:
        table_name: fec_contributions
```

**Invocation:**
```
User: @fec-extractor Koch Industries contributions over $5000
```

**Journal entry:**
> Extracted 47 FEC contributions for Koch Industries (min $5,000). Sorted by amount. Saved to fec_contributions table (47 rows inserted). 3 new entities added to knowledge graph.

---

### Example 2: Document Deduplication Pipeline

**What it does:** Scans a folder for duplicate documents using content hashing, reports duplicates, and optionally moves them to a "duplicates" folder.

**Handle:** `@dedup`

```yaml
id: document-deduplicator
name: Document Deduplicator
handle: dedup
execution_mode: deterministic
run_context: interactive

inputs:
  - name: folder_id
    type: string
    description: Folder to scan for duplicates
    required: true

steps:
  - name: list_files
    step_type: tool
    action: detect_folder_changes
    inputs:
      folder_id: "{folder_id}"
    params:
      include_subfolders: true
    output_key: all_files

  - name: find_duplicates
    step_type: tool
    action: transform_data
    inputs:
      data: "{all_files.changes}"
    operations:
      - type: aggregate
        field: new_hash
        count_field: duplicate_count
      - type: filter
        field: duplicate_count
        operator: gt
        value: 1
    output_key: duplicates

output:
  format: markdown
  destinations:
    - type: chat
```

---

## Pattern B: LLM-Augmented Workflows

### Example 3: Open-Ended Research Agent

**What it does:** User asks a research question; the LLM decides which tools to use and synthesizes findings.

**Handle:** `@researcher`

```yaml
id: general-researcher
name: Research Agent
handle: researcher
execution_mode: llm_augmented
run_context: interactive

steps:
  - name: research
    step_type: llm_task
    action: research_with_tools
    params:
      tools:
        - search_documents_tool
        - search_web_tool
        - crawl_web_content_tool
        - search_knowledge_graph_tool
      system_prompt: |
        You are a thorough research assistant. Use available tools to
        investigate the user's question. Cross-reference multiple sources.
        Always cite your sources. If information conflicts, note the
        discrepancy and explain which source you trust more and why.
      max_rounds: 5
    output_key: research_findings

  - name: format_report
    step_type: llm_task
    action: llm_analyze
    inputs:
      findings: "{research_findings}"
    params:
      instructions: |
        Synthesize the research findings into a clear, well-structured
        report with sections, citations, and a summary of key findings.
    output_schema:
      type: object
      fields:
        - name: report
          type: string
        - name: sources
          type: string[]
        - name: confidence
          type: number
    output_key: report

output:
  format: markdown
  auto_enrich_graph: true
  destinations:
    - type: chat
    - type: document
      config:
        folder_id: research-reports
        filename_template: "research_{date}.md"
```

**Invocation:**
```
User: @researcher What is the relationship between the Koch Network and
      the American Legislative Exchange Council (ALEC)?
```

---

## Pattern C: Hybrid Workflows (Mixed)

### Example 4: Nonprofit Investigation with Approval Gate

**What it does:** Collects data from multiple sources (deterministic), cross-references entities (deterministic), pauses for human review (approval), then synthesizes findings (LLM).

**Handle:** `@nonprofit-investigator`

This is the canonical example from `AGENT_FACTORY.md`. See the full playbook in that document under "Skill / Playbook" → Nonprofit Investigation Playbook.

**Key points:**
- Steps 1-5: Deterministic tool calls (FEC API, ProPublica, entity extraction, cross-reference)
- Step 6: Approval gate — user reviews discovered connections before proceeding
- Step 7: LLM synthesis — generates structured investigation report

**Invocation:**
```
User: @nonprofit-investigator Investigate the Omidyar Foundation
```

**Approval gate preview:**
> Found 3 officer-to-contribution links:
> 1. Pierre Omidyar → $50,000 to Senate Leadership Fund
> 2. Matt Stinchcomb → $10,000 to DCCC
> 3. Ellen Pao → $2,500 to ActBlue
>
> Enrich knowledge graph with these connections? [Approve] [Reject]

---

### Example 5: SEC Filing Analyzer

**What it does:** Pulls SEC filings for a company, extracts key financial metrics (deterministic), analyzes for notable changes (LLM), and alerts if thresholds are exceeded.

**Handle:** `@sec-analyzer`

```yaml
id: sec-filing-analyzer
name: SEC Filing Analyzer
handle: sec-analyzer
execution_mode: hybrid
run_context: interactive

inputs:
  - name: company
    type: entity_name
    required: true
    extract_from: query
  - name: filing_type
    type: string
    default: "10-K"

steps:
  - name: search_filings
    step_type: tool
    action: call_connector
    connector: sec-edgar
    endpoint: full_text_search
    inputs:
      q: "{company}"
      forms: "{filing_type}"
    output_key: filings

  - name: extract_financials
    step_type: tool
    action: extract_entities
    inputs:
      data: "{filings.results}"
    params:
      entity_types: [FINANCIAL_METRIC, ORG, PERSON]
    output_key: financials

  - name: analyze_trends
    step_type: llm_task
    action: llm_analyze
    inputs:
      filings: "{filings}"
      financials: "{financials}"
    params:
      instructions: |
        Analyze the SEC filings for notable changes, red flags, or
        significant financial trends. Compare year-over-year where
        possible. Flag any unusual executive compensation, related-party
        transactions, or material weaknesses in internal controls.
    output_schema:
      type: object
      fields:
        - name: summary
          type: string
        - name: red_flags
          type: string[]
        - name: key_metrics
          type: object
        - name: recommendations
          type: string[]
    output_key: analysis

output:
  format: markdown
  auto_enrich_graph: true
  destinations:
    - type: chat
    - type: document
      config:
        filename_template: "sec_analysis_{company}_{date}.md"
```

---

## Pattern D: Background / Scheduled Jobs

### Example 6: Daily News Briefing (Cron)

**What it does:** Runs every morning at 7am, searches for news about monitored entities, summarizes, and posts to a team thread.

**Handle:** `@morning-brief`

```yaml
id: daily-news-briefing
name: Daily News Briefing
handle: morning-brief
execution_mode: hybrid
run_context: scheduled

inputs:
  - name: monitored_entities
    type: list
    default: ["Koch Industries", "Omidyar Network", "Open Society Foundations"]

steps:
  - name: search_news
    step_type: tool
    action: search_web
    inputs:
      query: "{monitored_entities} news today"
    params:
      max_results: 20
      categories: [news]
    output_key: news

  - name: filter_relevant
    step_type: llm_task
    action: llm_analyze
    inputs:
      articles: "{news}"
      entities: "{monitored_entities}"
    params:
      instructions: |
        Filter these news articles to only those directly relevant to
        the monitored entities. For each relevant article, extract:
        entity mentioned, headline, source, and a 1-sentence summary.
    output_schema:
      type: record_set
      fields:
        - name: entity
          type: string
        - name: headline
          type: string
        - name: source
          type: string
        - name: summary
          type: string
        - name: url
          type: string
    output_key: relevant_news

  - name: generate_briefing
    step_type: llm_task
    action: llm_analyze
    condition: "{relevant_news.count} > 0"
    inputs:
      news: "{relevant_news}"
    params:
      instructions: |
        Generate a concise morning briefing organized by entity.
        Include headline, source, and 1-sentence summary for each item.
        Lead with the most significant stories.
    output_schema:
      type: object
      fields:
        - name: briefing
          type: string
        - name: story_count
          type: integer
    output_key: briefing

  - name: post_to_team
    step_type: tool
    action: write_team_post
    condition: "{briefing.story_count} > 0"
    inputs:
      content: "{briefing.briefing}"
      team_id: "{research_team_id}"
    params:
      thread_id: "{briefing_thread_id}"
    output_key: post_result

output:
  auto_enrich_graph: true
  destinations:
    - type: append_to_existing
      config:
        document_id: "{briefing_log_id}"
        timestamp: true
```

**Schedule:** `0 7 * * *` (7:00 AM daily)

**Journal entry:**
> Daily News Briefing (7:00 AM, scheduled). Found 20 articles, 8 relevant to monitored entities. Generated briefing with 8 stories. Posted to Research Unit team thread.

---

### Example 7: Weekly Knowledge Graph Health Check (Cron)

**What it does:** Runs weekly, analyzes the knowledge graph for stale data, orphaned entities, and potential merge candidates. Reports findings.

**Handle:** `@graph-health`

```yaml
id: graph-health-check
name: Knowledge Graph Health Check
handle: graph-health
execution_mode: hybrid
run_context: scheduled

steps:
  - name: find_stale_entities
    step_type: tool
    action: search_knowledge_graph
    inputs:
      entity_names: ["*"]
    params:
      entity_types: [PERSON, ORG]
      max_hops: 0
    output_key: all_entities

  - name: detect_anomalies
    step_type: llm_task
    action: llm_analyze
    inputs:
      entities: "{all_entities}"
    params:
      instructions: |
        Analyze the knowledge graph entities for:
        1. Stale entities (not updated in 30+ days)
        2. Low-confidence entities (confidence < 0.5)
        3. Potential duplicates (similar names that may need merging)
        4. Orphaned entities (no relationships)
        Report each category with counts and top examples.
    output_schema:
      type: object
      fields:
        - name: stale_count
          type: integer
        - name: low_confidence_count
          type: integer
        - name: potential_duplicates
          type: string[]
        - name: orphan_count
          type: integer
        - name: report
          type: string
    output_key: health_report

output:
  destinations:
    - type: document
      config:
        filename_template: "graph_health_{date}.md"
    - type: notification
      config:
        message: "Weekly graph health: {health_report.stale_count} stale, {health_report.potential_duplicates} potential duplicates"
```

**Schedule:** `0 9 * * 1` (9:00 AM every Monday)

---

## Pattern E: Monitor Mode (Change-Aware Polling)

### Example 8: Document Inbox Classifier (Folder Monitor)

**What it does:** Watches an "inbox" folder every 15 minutes. When new files appear, classifies them by content type using LLM vision/analysis, tags them, and moves them to the appropriate folder.

**Handle:** `@inbox-sorter`

```yaml
id: document-inbox-classifier
name: Document Inbox Classifier
handle: inbox-sorter
execution_mode: hybrid
run_context: monitor
monitor_config:
  interval: 15m
  active_hours:
    start: "07:00"
    end: "23:00"
    timezone: "America/New_York"
  suppress_if_empty: true

inputs:
  - name: inbox_folder_id
    type: string
    required: true
  - name: receipts_folder_id
    type: string
    required: true
  - name: research_folder_id
    type: string
    required: true
  - name: misc_folder_id
    type: string
    required: true

steps:
  - name: check_inbox
    step_type: tool
    action: detect_new_files
    inputs:
      folder_id: "{inbox_folder_id}"
    params:
      file_types: [pdf, jpg, png, docx, md, txt]
    output_key: new_files

  - name: classify_documents
    step_type: llm_task
    action: llm_analyze
    condition: "{new_files.count} > 0"
    inputs:
      files: "{new_files.files}"
    params:
      instructions: |
        For each document, determine its category:
        - receipt: Financial receipts, invoices, purchase records
        - research: Academic papers, reports, articles, analysis
        - personal: Personal correspondence, notes
        - reference: Reference material, manuals, guides
        - misc: Anything that doesn't fit above categories
        Also generate a brief 1-line description for tagging.
    output_schema:
      type: record_set
      fields:
        - name: file_id
          type: string
        - name: filename
          type: string
        - name: category
          type: string
        - name: description
          type: string
        - name: confidence
          type: number
    output_key: classifications

  - name: route_receipts
    step_type: tool
    action: batch_move_files
    condition: "{classifications.count} > 0"
    inputs:
      files: "{classifications}"
    params:
      routing_rules:
        receipt: "{receipts_folder_id}"
        research: "{research_folder_id}"
        misc: "{misc_folder_id}"
    output_key: move_results

  - name: update_watermark
    step_type: tool
    action: set_monitor_watermark
    inputs:
      watermark: "{new_files.watermark}"

output:
  destinations:
    - type: notification
      condition: "{new_files.count} > 0"
      config:
        message: "Sorted {new_files.count} new files from inbox: {move_results.summary}"
```

**Behavior:**
- Every 15 minutes (during active hours), checks the inbox folder
- If no new files → suppressed silently, no cost
- If 3 new PDFs found → classifies via LLM, moves to correct folders, notifies user
- Watermark ensures files are only processed once

**Journal query:**
```
User: @inbox-sorter How many files have you sorted this week?
Agent: This week I've sorted 23 files across 8 monitor runs:
       - 12 research papers → Research folder
       - 6 receipts → Receipts folder
       - 5 misc items → Misc folder
       Last run: 14 minutes ago (2 files sorted)
```

---

### Example 9: FEC Contribution Monitor (API Monitor)

**What it does:** Polls the FEC API every 4 hours for new contributions to monitored committees. Alerts on large contributions.

**Handle:** `@fec-monitor`

```yaml
id: fec-contribution-monitor
name: FEC Contribution Monitor
handle: fec-monitor
execution_mode: deterministic
run_context: monitor
monitor_config:
  interval: 4h
  suppress_if_empty: true
approval_policy: auto_approve

inputs:
  - name: committee_ids
    type: list
    default: ["C00573519", "C00580100"]
  - name: alert_threshold
    type: number
    default: 10000

steps:
  - name: check_new_contributions
    step_type: tool
    action: detect_new_data
    inputs:
      connector: fec-contributions
      endpoint: search_contributions
    params:
      extra_params:
        committee_id: "{committee_ids}"
    output_key: new_contributions

  - name: filter_large
    step_type: tool
    action: transform_data
    condition: "{new_contributions.count} > 0"
    inputs:
      data: "{new_contributions.results}"
    operations:
      - type: filter
        field: amount
        operator: gte
        value: "{alert_threshold}"
    output_key: large_contributions

  - name: save_all
    step_type: tool
    action: save_to_workspace
    condition: "{new_contributions.count} > 0"
    inputs:
      data: "{new_contributions.results}"
    params:
      table_name: fec_monitored_contributions
      create_if_missing: true
      upsert_key: [contributor_name, date, amount]
    output_key: save_result

  - name: update_watermark
    step_type: tool
    action: set_monitor_watermark
    inputs:
      watermark: "{new_contributions.watermark}"

output:
  destinations:
    - type: notification
      condition: "{large_contributions.count} > 0"
      config:
        message: "Alert: {large_contributions.count} contributions over ${alert_threshold} detected"
```

**Behavior:**
- Every 4 hours, checks FEC API for contributions since last check
- If nothing new → silent suppression
- If new contributions found → saves all to Data Workspace table
- If any exceed $10,000 → sends alert notification
- Fully deterministic: zero LLM tokens consumed

---

### Example 10: Team Discussion Summarizer (Team Post Monitor)

**What it does:** Monitors team conversation threads every hour. When significant new discussion is detected, generates a summary and posts it to a "team-digest" thread.

**Handle:** `@team-digest`

```yaml
id: team-discussion-summarizer
name: Team Discussion Summarizer
handle: team-digest
execution_mode: hybrid
run_context: monitor
monitor_config:
  interval: 1h
  active_hours:
    start: "08:00"
    end: "20:00"
  suppress_if_empty: true

inputs:
  - name: team_id
    type: string
    required: true
  - name: digest_thread_id
    type: string
    required: true
  - name: min_posts_for_summary
    type: number
    default: 5

steps:
  - name: check_new_posts
    step_type: tool
    action: detect_new_team_posts
    inputs:
      team_id: "{team_id}"
    params:
      min_length: 50
    output_key: new_posts

  - name: summarize_discussion
    step_type: llm_task
    action: llm_analyze
    condition: "{new_posts.count} >= {min_posts_for_summary}"
    inputs:
      posts: "{new_posts.posts}"
    params:
      instructions: |
        Summarize the team discussion. Extract:
        1. Key topics discussed
        2. Decisions made
        3. Action items assigned (with person responsible)
        4. Open questions that need resolution
        Be concise but complete.
    output_schema:
      type: object
      fields:
        - name: summary
          type: string
        - name: topics
          type: string[]
        - name: decisions
          type: string[]
        - name: action_items
          type: string[]
        - name: open_questions
          type: string[]
    output_key: digest

  - name: post_digest
    step_type: tool
    action: write_team_post
    condition: "{new_posts.count} >= {min_posts_for_summary}"
    inputs:
      content: "{digest.summary}"
      team_id: "{team_id}"
    params:
      thread_id: "{digest_thread_id}"
    output_key: post_result

  - name: update_watermark
    step_type: tool
    action: set_monitor_watermark
    inputs:
      watermark: "{new_posts.watermark}"

output:
  destinations:
    - type: notification
      condition: "{digest.action_items} is not empty"
      config:
        message: "Team digest: {digest.action_items.length} action items identified"
```

---

### Example 11: Image Classification Monitor (Folder + LLM Vision)

**What it does:** Watches a folder for new images, uses LLM vision to classify and describe each image, tags them with metadata, and adds to knowledge graph.

**Handle:** `@image-tagger`

```yaml
id: image-classification-monitor
name: Image Classification Monitor
handle: image-tagger
execution_mode: hybrid
run_context: monitor
monitor_config:
  interval: 30m
  suppress_if_empty: true

inputs:
  - name: watched_folder_id
    type: string
    required: true

steps:
  - name: detect_images
    step_type: tool
    action: detect_new_files
    inputs:
      folder_id: "{watched_folder_id}"
    params:
      file_types: [jpg, jpeg, png, webp, gif, tiff]
      include_subfolders: true
    output_key: new_images

  - name: classify_images
    step_type: llm_task
    action: llm_analyze
    condition: "{new_images.count} > 0"
    inputs:
      images: "{new_images.files}"
    params:
      instructions: |
        For each image, provide:
        1. Classification: photo, screenshot, diagram, chart, document_scan,
           artwork, map, infographic, other
        2. Brief description (1-2 sentences)
        3. Entities visible (people, organizations, locations, objects)
        4. Suggested tags for organization
    output_schema:
      type: record_set
      fields:
        - name: file_id
          type: string
        - name: classification
          type: string
        - name: description
          type: string
        - name: entities
          type: string[]
        - name: tags
          type: string[]
    output_key: classifications

  - name: extract_entities
    step_type: tool
    action: extract_entities
    condition: "{classifications.count} > 0"
    inputs:
      data: "{classifications}"
    params:
      entity_types: [PERSON, ORG, LOCATION]
    output_key: image_entities

  - name: update_watermark
    step_type: tool
    action: set_monitor_watermark
    inputs:
      watermark: "{new_images.watermark}"

output:
  auto_enrich_graph: true
  destinations:
    - type: notification
      condition: "{new_images.count} > 0"
      config:
        message: "Classified {new_images.count} new images. Found {image_entities.count} entities."
```

---

## Team Integration Examples

### Example 12: Shared Research Agent with Team Context

**What it does:** A research agent shared with the "Investigations" team. It can search team files for prior research and post findings to team threads.

**Handle:** `@team-researcher`

```yaml
id: team-research-agent
name: Team Research Agent
handle: team-researcher
execution_mode: llm_augmented
run_context: interactive

team_config:
  shared_with_teams: ["{investigations_team_id}"]
  team_file_access: true
  team_post_access: true

steps:
  - name: research
    step_type: llm_task
    action: research_with_tools
    params:
      tools:
        - search_documents_tool
        - search_web_tool
        - search_knowledge_graph_tool
        - search_team_files          # Has access because team_file_access: true
        - search_team_posts          # Has access because team_post_access: true
      system_prompt: |
        You are a research agent for the Investigations team. You have
        access to the team's shared files and conversation history.
        Always check team files first for prior research before searching
        external sources. Cite team members' prior work when relevant.
      max_rounds: 5
    output_key: findings

output:
  format: markdown
  auto_enrich_graph: true
  destinations:
    - type: chat
```

**Invocation (by team member):**
```
User: @team-researcher What do we know about the Sackler family's
      philanthropic giving patterns?

Agent: Based on team files and external research:

      **Prior Team Research:**
      Sarah's investigation (Research Reports/sackler_philanthropy.md)
      identified 12 major donations to universities between 2010-2020...

      **New Findings:**
      ...
```

---

## Journal & Activity Awareness Examples

### Example 13: Querying an Agent's Work History

A user interacts with their `@fec-monitor` agent about its activity:

```
User: @fec-monitor What have you done this week?

Agent: This week (Feb 10-14) I completed 42 monitor checks:

       Monday-Friday, every 4 hours:
       - 35 checks: No new contributions (suppressed)
       - 7 checks: New activity detected

       Key findings:
       - Tue 12:00: 3 new contributions to C00573519 (largest: $15,000)
       - Wed 08:00: 1 large contribution ($25,000) to C00580100 — alert sent
       - Thu 16:00: 8 new contributions across both committees
       - Fri 04:00: 2 new contributions, both under alert threshold

       Total: 14 new contributions saved to fec_monitored_contributions
       Alerts triggered: 1 (Wed, $25,000 contribution)

User: @fec-monitor Show me the $25,000 contribution details

Agent: From my journal entry on Wed Feb 12 at 08:00:

       Contributor: Apex Holdings LLC
       Amount: $25,000.00
       Committee: C00580100 (Senate Leadership Fund)
       Date: 2026-02-11
       Filing Type: Individual contribution

       This was saved to fec_monitored_contributions table and
       triggered an alert notification.
```

---

### Example 14: Cross-Agent Activity Summary

A user asks about activity across multiple agents:

```
User: @nonprofit-investigator @fec-monitor @sec-analyzer
      Give me a status update

(Each agent responds to the @mention independently)

@nonprofit-investigator: Last run was yesterday at 2:32 PM
(interactive). Investigated Omidyar Foundation — found 3 new
officer-contribution links, saved report, added 5 entities to graph.

@fec-monitor: Active — running every 4 hours. Last check: 23 minutes
ago (no new activity). This week: 14 new contributions tracked,
1 alert triggered.

@sec-analyzer: Last run was Monday at 10:15 AM (interactive).
Analyzed Tesla 10-K filing — identified 3 red flags, saved analysis
report. No scheduled runs configured.
```

---

## Combining Patterns

### Example 15: Full Investigation Pipeline (Interactive + Monitor + Scheduled)

A complete investigation setup using three agents that work together:

**Agent 1: `@investigate`** — Interactive hybrid agent for on-demand deep investigation
- Run context: interactive
- Execution mode: hybrid (tool collection → LLM synthesis)
- Triggered by @mention when user wants to dig into an entity

**Agent 2: `@watch-fec`** — Monitor agent polling FEC for new activity
- Run context: monitor (every 4 hours)
- Execution mode: deterministic
- Silently watches for new contributions, alerts on large ones

**Agent 3: `@weekly-report`** — Scheduled agent generating weekly summary
- Run context: scheduled (Mondays at 9am)
- Execution mode: hybrid
- Reads from knowledge graph + Data Workspace tables, generates report, posts to team

**How they work together:**
1. User runs `@investigate Koch Industries` — deep investigation, entities added to graph
2. `@watch-fec` detects new Koch-related contributions at next 4-hour check (because Koch entities are now in the monitored set from the graph)
3. `@weekly-report` includes both the investigation findings and the monitoring alerts in Monday's summary
4. All three agents share the same knowledge graph — discoveries compound across agents

### Example 16: Meeting Minutes Task Extractor (Scheduled + Notifications)

**What it does:** Every day at 5 PM, scans a "Meeting Minutes" folder for new files added that day, reads each one, uses an LLM to extract tasks assigned to the user, creates org-mode TODO items in the user's inbox, and sends a Telegram summary notification.

**Handle:** `@meeting-tasks`

```yaml
id: meeting-minutes-task-extractor
name: Meeting Minutes Task Extractor
handle: meeting-tasks
execution_mode: hybrid
run_context: scheduled
schedule: "0 17 * * *"     # Daily at 5 PM

journal_config:
  auto_journal: true
  detail_level: detailed

inputs:
  - name: folder_name
    type: string
    description: Folder containing meeting minutes
    required: false
    default: "Meeting Minutes"
  - name: assignee_name
    type: string
    description: Name to match when extracting assigned tasks
    required: true

steps:
  # Step 1: Find the meeting minutes folder
  - name: find_folder
    step_type: tool
    action: find_file_by_name
    inputs:
      name: "{folder_name}"
    params:
      scope: "my_docs"
    output_key: folder_match

  # Step 2: Detect files added or modified today
  - name: detect_new_minutes
    step_type: tool
    action: detect_new_files
    inputs:
      folder_id: "{folder_match.files[0].document_id}"
    params:
      file_types: ["md", "org", "txt", "pdf"]
    output_key: new_files

  # Step 3: Read each new file's content
  - name: read_minutes
    step_type: tool
    action: read_file
    inputs:
      document_id: "{new_files.new_items}"
    params:
      batch: true               # Process each item in the list
    output_key: minutes_content

  # Step 4: LLM extracts tasks assigned to the user
  - name: extract_tasks
    step_type: llm_task
    prompt: |
      You are reviewing meeting minutes to extract action items.

      The user's name is: {assignee_name}

      For each set of meeting minutes below, extract ONLY tasks/action items
      assigned to {assignee_name}. Return a JSON array of tasks.

      Each task should have:
      - "title": concise task description (imperative voice)
      - "source_meeting": filename the task came from
      - "deadline": deadline if mentioned (ISO format), or null
      - "priority": "A" if urgent/ASAP, "B" if this week, "C" if no urgency mentioned
      - "context": one-sentence context for why this task exists

      Meeting minutes:
      {minutes_content}
    inputs:
      minutes_content: "{minutes_content}"
    output_schema:
      type: object
      fields:
        tasks:
          type: array
          items:
            type: object
            fields:
              title: string
              source_meeting: string
              deadline: date
              priority: string
              context: string
        task_count: integer
    output_key: extracted_tasks

  # Step 5: Create a TODO for each extracted task
  - name: create_todos
    step_type: tool
    action: create_todo
    inputs:
      title: "{extracted_tasks.tasks}"    # Iterated over task list
    params:
      batch: true
      batch_mapping:
        title: "{item.title}"
        body: "Source: {item.source_meeting}\n\n{item.context}"
        priority: "{item.priority}"
        deadline: "{item.deadline}"
        tags: ["meeting-action"]
    output_key: todo_results

  # Step 6: Send Telegram notification with summary
  - name: notify_user
    step_type: tool
    action: send_channel_message
    inputs:
      message: |
        📋 Meeting Task Extraction — {extracted_tasks.task_count} new tasks

        {extracted_tasks.tasks | format_list:
          "• [{item.priority}] {item.title} (from {item.source_meeting})"}

        All tasks added to your inbox. Open Bastion to review.
    params:
      channel: "telegram"
      format: "markdown"
    output_key: notification_result

  # Step 7: Handle case where no new minutes were found
  - name: no_new_minutes
    step_type: tool
    action: noop
    condition: "{new_files.count} == 0"
    skip_remaining: true        # End workflow early

output:
  format: markdown
  destinations:
    - type: journal             # Log to agent journal
      config:
        summary: "Extracted {extracted_tasks.task_count} tasks from {new_files.count} meeting minutes"
```

**Invocation:**

```
# Runs automatically at 5 PM daily (scheduled)

# Can also be triggered manually:
@meeting-tasks Check for tasks in today's meetings

# Query past activity:
@meeting-tasks What tasks did you find this week?
```

**Typical journal entry:**

```
@meeting-tasks — Feb 14, 2026 at 5:00 PM (scheduled)
Scanned "Meeting Minutes" folder. Found 2 new files:
  • 2026-02-14_product_sync.md
  • 2026-02-14_leadership_standup.md
Extracted 4 tasks assigned to user:
  • [A] Finalize Q1 budget proposal by Friday
  • [B] Review competitor analysis doc
  • [B] Schedule 1:1 with design lead
  • [C] Update team wiki with new process
All tasks created in inbox. Telegram notification sent.
```

**Tools used:** `find_file_by_name`, `detect_new_files`, `read_file`, `create_todo`, `send_channel_message`

---

## Decision Guide: Choosing the Right Pattern

| I want to... | Pattern | Run Context | Execution Mode |
|---|---|---|---|
| Pull data from an API and save it | A: Deterministic | interactive | deterministic |
| Ask a question and get a researched answer | B: LLM-Augmented | interactive | llm_augmented |
| Investigate an entity with human review | C: Hybrid | interactive | hybrid |
| Generate a daily/weekly report | D: Scheduled | scheduled | hybrid |
| Watch for new files and process them | E: Monitor | monitor | hybrid |
| Watch an API for new records | E: Monitor | monitor | deterministic |
| Summarize team discussions periodically | E: Monitor | monitor | hybrid |
| Run a pure data pipeline on a schedule | D: Scheduled | scheduled | deterministic |
| Alert me when something specific happens | E: Monitor | monitor | deterministic or hybrid |
| Extract tasks from docs and notify me | D: Scheduled | scheduled | hybrid |
| Create todos + send Telegram alerts | D: Scheduled + notifications | scheduled | hybrid |

| I want the agent to... | Use |
|---|---|
| Run when I ask it | `run_context: interactive` with @mention |
| Run at specific times | `run_context: scheduled` with cron |
| Watch for changes and react | `run_context: monitor` with detection tools |
| Run in the background right now | `run_context: background` |
| Be usable by my team | `team_config.shared_with_teams` |
| Track its own activity | `journal_config.auto_journal: true` (default) |
| Create todos from extracted data | `create_todo` tool in playbook steps |
| Send me Telegram/Slack/Discord alerts | `send_channel_message` tool with channel param |
