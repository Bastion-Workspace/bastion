# Agent Factory UX Design

## Overview

The Agent Factory is a top-level page (permission-gated) where users build, configure, manage, and observe custom AI agents. The design follows a **module-card + configuration-drawer** pattern — no visual node-graph (React Flow). Workflows are represented as vertical step cards; detailed configuration happens in a slide-out drawer.

This document covers the complete frontend UX: page layout, interaction patterns, component hierarchy, and integration with the existing Bastion UI framework.

### Design Principles

| Principle | Meaning |
|---|---|
| **Cards, not canvas** | Workflows are vertical step cards, not a 2D node graph. Sequential reading, drag-to-reorder. |
| **Drawer, not modal** | Configuration opens in a slide-out drawer alongside the workflow, not a blocking dialog. Context is preserved. |
| **Type-aware dropdowns** | Input wiring uses dropdowns populated by the I/O registry — only compatible upstream outputs shown. No manual typing of field references. |
| **Chat co-pilot** | The Assembly Agent in the chat sidebar can see and modify the agent being edited. Building an agent is a conversation. |
| **Progressive disclosure** | Summary cards show just enough. Drawer reveals full configuration. Advanced options hidden behind expandable sections. |
| **Consistent with Bastion** | Uses MUI components, follows existing sidebar + main area + chat sidebar layout, matches existing card/drawer patterns. |

### Why Not React Flow

React Flow is already in the dependency tree but is not the right tool for this job:

- **Canvas management tax** — zooming, panning, and wire management distract from configuration
- **Spaghetti problem** — 6+ step workflows become visually noisy with crossing wires
- **Tiny config boxes** — node UIs squeeze configuration into small cards, forcing modals anyway
- **Mobile/tablet hostile** — canvas UIs are desktop-only
- **Accessibility gap** — screen readers cannot navigate a spatial graph
- **Overkill for linearity** — 90%+ of workflows are sequential chains. A vertical card stack is simpler and more readable.

React Flow may be useful for a read-only visualization of complex agent-to-agent relationships or knowledge graph views, but the primary builder UI uses the card + drawer pattern.

---

## Page Layout

### Navigation Integration

Agent Factory appears as a top-level route in `Navigation.js`, gated by user permissions:

```
/agent-factory         -> AgentFactoryPage (main page)
/agent-factory/:id     -> AgentFactoryPage (with agent selected for editing)
```

The navigation item is conditionally rendered based on user permissions (similar to how other features are gated). Users without Agent Factory access do not see the nav item.

### Three-Panel Layout

The Agent Factory page follows Bastion's standard three-panel layout:

```
+----------------------------------------------------------------------+
|  Navigation (AppBar)                                                  |
|  [Dashboard] [Documents] [Agent Factory] [Data Workspace] [...]      |
+-----------+----------------------------------+-----------------------+
|           |                                  |                       |
|  AGENT    |   AGENT EDITOR                   |  CHAT SIDEBAR         |
|  LIST     |   (Main Content Area)            |                       |
|  PANEL    |                                  |  @assembly-agent      |
|           |                                  |  active when on       |
|  (Left)   |                                  |  Agent Factory page   |
|           |                                  |                       |
|  240px    |   Fluid                          |  400px (resizable)    |
|  resize-  |                                  |                       |
|  able     |                                  |                       |
|           |                                  |                       |
+-----------+----------------------------------+-----------------------+
|  StatusBar                                                            |
+----------------------------------------------------------------------+
```

- **Left panel**: Agent list sidebar (replaces `FileTreeSidebar` on this page)
- **Main area**: Agent editor (section cards) or agent list grid (when no agent selected)
- **Right panel**: Standard `ChatSidebar` — automatically activates the Assembly Agent context when on the Agent Factory page

On mobile, the left panel becomes a slide-out drawer (consistent with `FileTreeSidebar` mobile behavior).

---

## Left Panel: Agent List Sidebar

### Layout

```
+- MY AGENTS --------------------------+
|  [Search agents...]                  |
|                                      |
|  +--------------------------------+  |
|  | (icon) @nonprofit-investigator |  |
|  |    Active  -  Last: 2h ago     |  |
|  +--------------------------------+  |
|  +--------------------------------+  |
|  | (icon) @fec-tracker            |  |
|  |    Scheduled  -  Daily 8am     |  |
|  +--------------------------------+  |
|  +--------------------------------+  |
|  | (icon) @sec-monitor            |  |
|  |    Monitoring  -  15m          |  |
|  +--------------------------------+  |
|  +--------------------------------+  |
|  | (icon) Draft: Email Responder  |  |
|  |    Draft  -  Not activated     |  |
|  +--------------------------------+  |
|                                      |
|  -- TEAM AGENTS -------------------- |
|  +--------------------------------+  |
|  | (icon) @political-tracker      |  |
|  |    Shared by: Research Unit    |  |
|  +--------------------------------+  |
|                                      |
|  [+ New Agent]                       |
+--------------------------------------+
```

### Agent Card States

| State | Indicator | Meaning |
|---|---|---|
| Active | green dot | Agent is activated and can be invoked |
| Scheduled | clock icon, blue | Agent runs on a cron schedule |
| Monitoring | pulsing green | Agent is actively polling on a monitor interval |
| Draft | gray circle | Agent is being configured, not yet activated |
| Error | red dot | Last execution failed — tap to see journal |
| Paused | yellow pause | Agent is activated but temporarily paused |

### Context Menu (Right-Click or kebab button)

- Edit
- Duplicate
- Test Run
- View Journal
- Share with Team...
- Pause / Resume
- Delete

### New Agent Flow

Clicking `[+ New Agent]` creates a draft agent with a default name and opens it in the editor. The Assembly Agent in the chat sidebar proactively offers: "I see you're creating a new agent. What should it do?"

---

## Main Area: Agent Editor

When an agent is selected, the main area shows the **Agent Editor** — a vertically scrolling page of collapsible section cards. Each section represents a configuration domain.

### Section Order

```
v IDENTITY & BASICS
v DATA SOURCES
v WORKFLOW STEPS
v OUTPUT DESTINATIONS
v SCHEDULE & TRIGGERS
> SHARING & PERMISSIONS        (collapsed by default)
> ADVANCED                     (collapsed by default)
```

Sections use MUI `Accordion` (or a custom collapsible card) with a consistent header pattern:

```
+- SECTION NAME ----------------------- [status badge] -+
|                                                        |
|  Section content (cards, forms, etc.)                  |
|                                                        |
+--------------------------------------------------------+
```

The status badge shows validation state: checkmark (valid), warning (warnings), X (errors), or a count (e.g., "3 steps").

### Bottom Action Bar

A sticky bottom bar with primary actions:

```
+--------------------------------------------------------------+
|  [Test Run]  [Save Draft]  [Activate]   Last saved: 2m ago   |
+--------------------------------------------------------------+
```

- **Test Run**: Runs the agent with a test query (opens a test panel or streams results into the chat sidebar)
- **Save Draft**: Saves configuration without activating
- **Activate**: Saves and makes the agent live (available for @mention invocation, schedules start, monitors begin polling)

---

## Section 1: Identity & Basics

```
v IDENTITY & BASICS                                        (valid)

  Name            [@nonprofit-investigator         ]
  Handle          @[nonprofit-investigator         ]
                  (i) Unique per user. Used for @mention.

  Icon            [(emoji) v]    (emoji picker)

  Description     [Investigates nonprofit organizations   ]
                  [using FEC and IRS 990 data             ]

  Execution Mode  (*) Hybrid   ( ) Deterministic
                  ( ) LLM-Augmented
                  (i) Hybrid: mix tool steps and LLM tasks.
                     Deterministic: tool-only, zero LLM.
                     LLM-Augmented: LLM decides tool use.

  Default Model   [anthropic/claude-sonnet-4-20250514 v]
                  (i) Default for LLM task steps that don't
                     specify their own model.
```

The model selector dropdown shows globally available models with pricing tiers, matching the model selector in the AI Chat sidebar settings.

---

## Section 2: Data Sources

```
v DATA SOURCES                                        2 connectors

  +----------------------------------------------+
  | (icon) FEC Campaign Finance API     [gear][X] |
  |    Endpoints: search_contributions,           |
  |               get_committee_details           |
  |    Rate limit: 10 req/s (shared w/ 2 agents) |
  |    Schema: v2 (latest) - 8 output fields     |
  +----------------------------------------------+

  +----------------------------------------------+
  | (icon) ProPublica Nonprofit Explorer [gear][X]|
  |    Endpoints: search_orgs, get_filing         |
  |    Rate limit: 5 req/s                        |
  |    Schema: v1 - 12 output fields              |
  +----------------------------------------------+

  [+ Add Connector]
```

### Add Connector Flow

Clicking `[+ Add Connector]` shows an inline selector (not a modal):

```
+- ADD DATA SOURCE -----------------------------------------+
|                                                           |
|  [Search connectors...]                                   |
|                                                           |
|  -- YOUR CONNECTORS ------------------------------------  |
|  (icon) FEC Campaign Finance API          [Add]           |
|  (icon) SEC EDGAR Full-Text Search        [Add]           |
|  (icon) Custom: State Lobbying DB         [Add]           |
|                                                           |
|  -- TEMPLATES ------------------------------------------  |
|  (icon) ProPublica Nonprofit Explorer     [Use Template]  |
|  (icon) OpenCorporates API                [Use Template]  |
|  (icon) IRS 990 Bulk Data (XML)           [Use Template]  |
|  (icon) Federal Register RSS              [Use Template]  |
|                                                           |
|  -- CREATE NEW -----------------------------------------  |
|  [+ Create Custom Connector]                              |
|                                                           |
|  (i) Ask @assembly-agent to help build a connector        |
|     from an API you want to use.                          |
+-----------------------------------------------------------+
```

"Your Connectors" shows connectors the user has already created (shared across all their agents). "Templates" shows system-provided connector definitions the user can clone and customize.

### Connector Configuration Drawer

Clicking the gear icon on a connector card opens a drawer with:

- Connection settings (auth, base URL, rate limits)
- Available endpoints (list with toggle to enable/disable per agent)
- Schema version management (current version, history, field list)
- Test endpoint (make a sample API call and see response)

---

## Section 3: Workflow Steps (Core Builder)

This is the primary builder interface — the section that replaces what a React Flow canvas would have been.

### Step Card Stack

```
v WORKFLOW STEPS                                         5 steps

  +----------------------------------------------------+
  | (drag) 1  pull_fec_data               [config] [X] |
  |   TOOL - FEC Contributions ->                      |
  |        search_contributions                        |
  |   Input: entity_name <- {query.entity_name}        |
  |   Output: fec_data (record_set, ~23 records)       |
  +----------------------------------------------------+
                          |
                          v
  +----------------------------------------------------+
  | (drag) 2  filter_recent               [config] [X] |
  |   TOOL - transform_data -> filter                  |
  |   Input: data <- {pull_fec_data.results}           |
  |   Output: recent_data                              |
  +----------------------------------------------------+
                          |
                          v
  +----------------------------------------------------+
  | (drag) 3  synthesize_findings         [config] [X] |
  |   LLM TASK - synthesize_report                     |
  |   Model: claude-sonnet - ~$0.03/run                |
  |   Input: data <- {recent_data} + {fec_data}        |
  |   Output: report {summary, risk_score}             |
  +----------------------------------------------------+
                          |
                          v
  +----------------------------------------------------+
  | (drag) 4  Review findings             [config] [X] |
  |   APPROVAL GATE                                    |
  |   Preview: {synthesize_findings.summary}           |
  |   On reject: stop                                  |
  +----------------------------------------------------+
                          |
                          v
  +----------------------------------------------------+
  | (drag) 5  hand_to_analyst             [config] [X] |
  |   TOOL - send_to_agent -> @political-analyst       |
  |   Condition: {report.risk_score} > 7               |
  |   Mode: fire-and-forget                            |
  +----------------------------------------------------+

                 [+ Add Step]

  -- Cost Summary ------------------------------------------
  Tool steps (4): $0.00  -  LLM steps (1): ~$0.03
  Est. per run: ~$0.03   -  Daily (scheduled): ~$0.90
```

### Step Card Anatomy

Each step card has a consistent structure:

```
+------------------------------------------------------+
| (drag)  N  step_name                  [config] [X]   |
|    ICON  STEP_TYPE - ACTION_SUMMARY                  |
|    Input: input_name <- {source_step.field}          |
|    Output: output_key (type_summary)                 |
|    [Condition badge if present]                      |
+------------------------------------------------------+

(drag) = Drag handle (drag to reorder)
[config] = Open configuration drawer
[X] = Delete step (with confirmation)
N = Step number (auto-calculated from position)
```

**Step type indicators:**
- TOOL = Tool step (deterministic)
- LLM = LLM Task step
- APPROVAL = Approval gate
- AGENT = Send to agent
- PARALLEL = Parallel sub-steps

**Color coding:**
- Tool steps: neutral/default card background
- LLM steps: subtle blue tint (indicating model usage + cost)
- Approval gates: subtle yellow tint (indicating human action needed)
- Conditional steps: dashed border (may be skipped)

### Drag-to-Reorder

Steps support drag-to-reorder using MUI's drag-and-drop or a library like `@dnd-kit`. When dragging:

- A ghost card follows the cursor
- Drop zones appear between cards
- After drop, the system validates the new order: if moving a step breaks an input reference (step references an output_key that now comes AFTER it), a warning appears: "Step 3 references {fec_data} which is now at step 4. Adjust input wiring?"

### Add Step Selector

Clicking `[+ Add Step]` shows an inline panel (not a modal):

```
+- ADD STEP -----------------------------------------------+
|                                                           |
|  [Tool Step]      [LLM Task]      [Approval]             |
|                                                           |
|  -- TOOL ACTIONS ------------------------------------     |
|  Call a data source connector                             |
|  Search knowledge graph                                   |
|  Search documents                                         |
|  Web search                                               |
|  Transform / filter data                                  |
|  Send to another agent                                    |
|  Send notification                                        |
|  Save to Data Workspace                                   |
|  Analyze graph (PageRank, community detection)            |
|  Route to output destination                              |
|  Parallel sub-steps                                       |
|  ---------------------------------------------------------|
|  LLM TASK ACTIONS                                         |
|  Synthesize report                                        |
|  Analyze / classify data                                  |
|  Research with tools (LLM decides tool use)               |
|  ---------------------------------------------------------|
|  Approval gate                                            |
|                                                           |
|  [Search all actions...]                                  |
|                                                           |
|  (i) Ask @assembly-agent to suggest the next step         |
+-----------------------------------------------------------+
```

Selecting an action creates a new step card and immediately opens the configuration drawer for it.

### Step Configuration Drawer

Clicking a step card (or its config button) slides open a configuration drawer from the right side. The drawer pushes the chat sidebar or overlays it (user preference).

#### Drawer Layout

```
+- STEP CONFIGURATION -----------------------------------+
|                                              [X Close] |
|                                                        |
|  Step 3: synthesize_findings                           |
|  Type: [LLM Task v]                                   |
|  Action: [synthesize_report v]                         |
|                                                        |
|  -- MODEL -------------------------------------------- |
|  [anthropic/claude-sonnet-4-20250514 v]                       |
|  Est. cost: ~$0.03/run (based on avg input size)       |
|  (i) Override the agent's default model for this step. |
|                                                        |
|  -- INPUTS ------------------------------------------- |
|  data:                                                 |
|    Source: [step 2 filter_recent -> results       v]   |
|                                                        |
|  additional_context:                                   |
|    Source: [step 1 pull_fec_data -> results       v]   |
|                                                        |
|  [+ Add Input Mapping]                                 |
|                                                        |
|  -- OUTPUT ------------------------------------------- |
|  Output key: [synthesize_findings     ]                |
|  Schema type: [object v]                               |
|                                                        |
|  Fields:                                               |
|  +----------------------------------------------+      |
|  |  summary      [string  v]  Report summary    |      |
|  |  risk_score   [number  v]  Risk 1-10         |      |
|  |  entities     [string[] v] Entities found     |      |
|  |  [+ Add Field]                                |      |
|  +----------------------------------------------+      |
|                                                        |
|  -- INSTRUCTIONS ------------------------------------- |
|  +----------------------------------------------+      |
|  | Synthesize the FEC contribution data         |      |
|  | into a risk assessment report. Focus on      |      |
|  | connections between officers and donors.     |      |
|  +----------------------------------------------+      |
|                                                        |
|  > ERROR HANDLING (collapsed)                          |
|  > CONDITION (collapsed)                               |
|                                                        |
|  [Test This Step]                   [Apply Changes]    |
+---------------------------------------------------------+
```

#### Input Wiring Dropdowns (Key UX)

The input wiring dropdowns are the most critical UX element — they replace visual wires. Each input field shows a dropdown with:

```
Source: [                                    v]
+---------------------------------------------------+
|  -- UPSTREAM STEP OUTPUTS -----------------------  |
|  step 1: pull_fec_data                             |
|    -> results         (record_set)     (compat)    |
|    -> count           (number)                     |
|    -> query_used      (text)                       |
|  step 2: filter_recent                             |
|    -> results         (record_set)     (compat)    |
|    -> count           (number)                     |
|  -- PLAYBOOK INPUTS ----------------------------   |
|    -> entity_name     (text)                       |
|    -> start_date      (date)                       |
|  -- RUNTIME VARIABLES --------------------------   |
|    -> {today}         (date)                       |
|    -> {execution_id}  (text)                       |
+---------------------------------------------------+

(compat) = type-compatible with this input (highlighted)
```

The dropdown is organized into sections: upstream step outputs, playbook input variables, and runtime variables. Items are tagged with their type. **Compatible types are highlighted** (using the I/O registry's type compatibility rules). Incompatible types are grayed out but still selectable (with a coercion warning).

This is where the typed I/O contracts deliver their value — the user never types `{pull_fec_data.results}` manually. They select from a dropdown that shows what's available and what's compatible.

#### Parallel Sub-Steps

For the `parallel` action, the drawer shows a nested step card list:

```
-- PARALLEL SUB-STEPS ----------------------------------
  +----------------------------------------------+
  | A  search_fec                    [config] [X] |
  |    TOOL - call_connector -> FEC               |
  +----------------------------------------------+
  +----------------------------------------------+
  | B  search_990                    [config] [X] |
  |    TOOL - call_connector -> IRS 990           |
  +----------------------------------------------+
  +----------------------------------------------+
  | C  search_graph                  [config] [X] |
  |    TOOL - search_knowledge_graph              |
  +----------------------------------------------+
  [+ Add Parallel Step]

  (i) All sub-steps run simultaneously. Each produces
    its own output_key. Downstream steps can reference
    any sub-step output.
```

---

## Section 4: Output Destinations

```
v OUTPUT DESTINATIONS                                 2 destinations

  +----------------------------------------------+
  | (icon) Document                     [gear][X] |
  |    Format: Markdown                           |
  |    Folder: Research Reports/                  |
  |    Filename: {entity_name}_{today}.md         |
  +----------------------------------------------+

  +----------------------------------------------+
  | (icon) Data Workspace Table         [gear][X] |
  |    Table: contributions                       |
  |    Mode: Append rows                          |
  +----------------------------------------------+

  [+ Add Destination]

  [x] Auto-enrich knowledge graph
  (i) Discovered entities and relationships will be
    added to the knowledge graph automatically.
```

### Add Destination Selector

```
+- ADD DESTINATION ----------------------------------------+
|                                                          |
|  Document           Save results as a document           |
|  Folder             Save to a specific folder            |
|  Data Workspace     Insert into a table                  |
|  Knowledge Graph    Write entities/relationships         |
|  File Export        Export as CSV/JSON/PDF                |
|  Append to Existing Append to an existing document       |
|  Chat               Display in chat (always on)          |
|  Notification       Send via Telegram/Discord/etc.       |
|                                                          |
+----------------------------------------------------------+
```

Each destination type has its own configuration drawer with type-specific options (folder picker, table selector, format options, etc.).

---

## Section 5: Schedule & Triggers

```
v SCHEDULE & TRIGGERS                               Scheduled, Daily

  Run Context:
  (*) Interactive (chat)    ( ) Background
  ( ) Scheduled             ( ) Monitor

  -- SCHEDULE (when run_context = scheduled) --
  Frequency:    [Daily v]
  Time:         [08:00 v]  Timezone: [America/New_York v]
  Active hours: [08:00] to [22:00] (optional)

  Query template:
  +----------------------------------------------+
  | Check for new contributions to monitored     |
  | entities since {last_run_date}               |
  +----------------------------------------------+

  -- MONITOR (when run_context = monitor) --
  Poll interval: [15 minutes v]
  Suppress if empty: [x]
  Active hours: [08:00] to [22:00] (optional)

  -- TRIGGERS (for interactive mode) --
  +----------------------------------------------+
  | Pattern: "investigate *"  Type: keyword  [X]  |
  | Pattern: "fec|campaign"   Type: regex    [X]  |
  +----------------------------------------------+
  [+ Add Trigger Pattern]

  Approval policy:
  (*) Require approval at gates
  ( ) Auto-approve (for trusted scheduled pipelines)
```

---

## Section 6: Sharing & Permissions

```
> SHARING & PERMISSIONS                            Not shared

  -- TEAM SHARING --
  Not shared with any teams.
  [Share with Team...]

  When shared, configure per team:
  [ ] Agent can search/read team files
  [ ] Agent can read/write team posts
  Journal visibility: [Own + Scheduled v]
```

---

## Section 7: Advanced

```
> ADVANCED

  -- AGENT-TO-AGENT --
  Allow other agents to send messages to this agent: [x]
  Max daily received messages: [100   ]

  -- CONVERSATION MODE --
  [ ] Enable as conversational participant
  Role label: [                    ]
  Max turns per conversation: [10  ]

  -- RESOURCE LIMITS --
  Max execution time: [300] seconds
  Max retries: [2]
```

---

## Assembly Agent Integration

The AI chat sidebar becomes a co-pilot for building agents. When the user navigates to the Agent Factory page, the chat context automatically includes the Assembly Agent awareness.

### How It Works

1. **Active editor context**: The Agent Factory page sends the current agent profile state to the chat sidebar via `ChatSidebarContext` / `EditorContext`, the same pattern used for document editing. The Assembly Agent sees the agent's current configuration.

2. **Proactive assistance**: When the user creates a new agent, the Assembly Agent offers guidance. When the user adds a connector, it suggests relevant steps. When the user configures an LLM step, it suggests model choices.

3. **Apply suggestions**: The Assembly Agent can propose configuration changes that appear as inline suggestions in the chat. The user clicks "Apply" to have the change reflected in the editor (similar to how the fiction editing agent proposes edits).

4. **Validation on demand**: The user can ask "Is my workflow valid?" and the Assembly Agent checks I/O wiring, type compatibility, missing connections, and reports issues.

### Example Interaction

```
User: I want to build an agent that tracks new SEC filings
      for companies I'm monitoring.

Assembly Agent: I can help with that. Here's what I'd suggest:

  1. Start with the SEC EDGAR connector template (I see it's
     available in your templates)
  2. Use monitor mode with a 1-hour polling interval
  3. Add a detection step for new filings, then a filter step
     for your monitored companies, then a notification step

  Want me to set this up? I'll create the steps and you can
  adjust the details.

User: Yes, set it up.

Assembly Agent: Done. I've created 4 steps:
  1) detect_new_filings (monitor detection)
  2) filter_monitored_companies (transform_data)
  3) summarize_findings (LLM task, gemini-2.5-flash)
  4) notify_user (send notification)

  The workflow is in monitor mode, polling every hour.
  Open step 2 to configure which companies to monitor.
  Open step 3 to adjust the summary instructions.
```

---

## Agent Journal View

Accessible from the agent list context menu ("View Journal") or from a tab in the agent editor. Shows the agent's work history in reverse chronological order.

```
+- JOURNAL: @nonprofit-investigator ---------------------+
|                                                        |
|  Filter: [All v]  [This week v]  [Search...]           |
|                                                        |
|  -- Today, Feb 15 2026 --                              |
|                                                        |
|  +----------------------------------------------------+|
|  | FAILED  09:15  Daily FEC monitoring                 ||
|  |    Scheduled - 2/3 steps completed                  ||
|  |    Error: KeyError 'contribution_receipt_date'      ||
|  |    [View Diagnostic Details v]                      ||
|  |    +----------------------------------------------+ ||
|  |    | Step trace:                                  | ||
|  |    |  1) pull_fec_data        OK (450ms)          | ||
|  |    |  2) filter_recent        FAIL KeyError       | ||
|  |    |  3) notify_if_found      skipped             | ||
|  |    |                                              | ||
|  |    | Connector: FEC API responded normally         | ||
|  |    | Cause: API field renamed                     | ||
|  |    | Suggestion: Update connector response        | ||
|  |    |   mapping for 'contribution_date'            | ||
|  |    +----------------------------------------------+ ||
|  +----------------------------------------------------+|
|                                                        |
|  +----------------------------------------------------+|
|  | OK  08:00  Omidyar Foundation investigation         ||
|  |    Interactive - 3/3 steps completed                ||
|  |    - 23 FEC contributions, 12 officers              ||
|  |    - 3 officer-contribution links found             ||
|  |    - Report: Research Reports/omidyar_2026.md       ||
|  |    - 5 entities added to knowledge graph            ||
|  +----------------------------------------------------+|
|                                                        |
+--------------------------------------------------------+
```

Failed entries are visually distinct (red accent, expanded diagnostic section). The diagnostic details are collapsible — showing the step-by-step execution trace, error context, and suggestions.

---

## Agent-to-Agent Conversation View

When agents are engaged in a conversation (conversational loop pattern), the user can observe in real-time through the chat sidebar. The conversation appears as a special conversation thread:

```
+- AGENT CONVERSATION ------------------------------------+
|  @research-agent <-> @skeptic-agent                     |
|  Topic: Funding connections for Koch Network            |
|  Turn 3 of 10                             [Stop]        |
|                                                         |
|  +---------------------------------------------------+ |
|  | @research-agent (Turn 1):                          | |
|  | I found 23 FEC contributions linked to Koch        | |
|  | Industries subsidiaries...                         | |
|  +---------------------------------------------------+ |
|                                                         |
|  +---------------------------------------------------+ |
|  | @skeptic-agent (Turn 2):                           | |
|  | 3 of those contributions may be coincidental.      | |
|  | Can you verify the officer connections?             | |
|  +---------------------------------------------------+ |
|                                                         |
|  +---------------------------------------------------+ |
|  | @research-agent (Turn 3):                          | |
|  | Cross-referencing with 990 officer data...         | |
|  | (Processing...)                                    | |
|  +---------------------------------------------------+ |
|                                                         |
|  -- USER INTERVENTION --                                |
|  [Type a message to intervene in this conversation...]  |
|                                                         |
+---------------------------------------------------------+
```

The user can intervene by typing a message, which is injected into the conversation context. They can also halt the conversation at any time.

---

## No-Agent-Selected State

When no agent is selected (fresh navigation to Agent Factory), the main area shows a grid of agents:

```
+- AGENT FACTORY ------------------------------------------+
|                                                          |
|  Your Agents (4)                      [+ New Agent]      |
|                                                          |
|  +------------------+  +------------------+              |
|  | (icon)           |  | (icon)           |              |
|  | @nonprofit-      |  | @fec-tracker     |              |
|  | investigator     |  |                  |              |
|  | Active           |  | Daily 8am        |              |
|  | Last: 2h ago     |  | Last: 6h ago     |              |
|  | 12 runs - 0 err  |  | 89 runs - 2 err  |              |
|  +------------------+  +------------------+              |
|                                                          |
|  +------------------+  +------------------+              |
|  | (icon)           |  | (icon)           |              |
|  | @sec-monitor     |  | Email Responder  |              |
|  | Monitoring       |  | Draft            |              |
|  | Every 15m        |  | Not activated    |              |
|  | 230 runs - 0 err |  |                  |              |
|  +------------------+  +------------------+              |
|                                                          |
|  Team Agents (1)                                         |
|  +------------------+                                    |
|  | (icon)           |                                    |
|  | @political-      |                                    |
|  | tracker          |                                    |
|  | Research Unit    |                                    |
|  | Active           |                                    |
|  +------------------+                                    |
|                                                          |
+----------------------------------------------------------+
```

---

## Component Hierarchy

### New Components

```
frontend/src/components/agent_factory/
+-- AgentFactoryPage.js              # Top-level page (route handler)
+-- AgentListSidebar.js              # Left panel: agent list with search
+-- AgentListGrid.js                 # Main area: grid view when no agent selected
+-- AgentCard.js                     # Card component (used in sidebar and grid)
+-- AgentEditor.js                   # Main area: section-based editor
|
+-- sections/                        # Editor sections (collapsible cards)
|   +-- IdentitySection.js           # Name, handle, icon, mode, model
|   +-- DataSourcesSection.js        # Connector list + add connector
|   +-- WorkflowStepsSection.js      # Step card stack + add step
|   +-- OutputDestinationsSection.js # Destination list + add destination
|   +-- ScheduleTriggersSection.js   # Run context, schedule, monitor, triggers
|   +-- SharingPermissionsSection.js # Team sharing config
|   +-- AdvancedSection.js           # Agent-to-agent, resource limits
|
+-- workflow/                        # Workflow step components
|   +-- StepCard.js                  # Summary card for a single step
|   +-- StepCardStack.js             # Orderable list of StepCards
|   +-- AddStepSelector.js           # Inline step action picker
|   +-- ParallelStepGroup.js         # Nested parallel sub-steps
|   +-- CostSummaryBar.js            # Per-run and monthly cost estimates
|
+-- drawers/                         # Configuration drawers
|   +-- StepConfigDrawer.js          # Full step configuration
|   +-- ConnectorConfigDrawer.js     # Connector settings + test
|   +-- DestinationConfigDrawer.js   # Output destination config
|   +-- InputWiringDropdown.js       # Type-aware input source selector
|
+-- selectors/                       # Inline selectors (not modals)
|   +-- AddConnectorSelector.js      # Connector library + templates
|   +-- AddDestinationSelector.js    # Destination type picker
|   +-- ModelSelector.js             # Model picker with pricing
|   +-- OutputSchemaEditor.js        # Field list editor for LLM step outputs
|
+-- journal/                         # Journal and execution history
|   +-- AgentJournalView.js          # Journal entry list with filters
|   +-- JournalEntry.js              # Single entry (success/failure)
|   +-- DiagnosticDetails.js         # Expandable failure diagnostic trace
|   +-- ExecutionHistoryView.js      # Raw execution log (for advanced users)
|
+-- conversations/                   # Agent-to-agent conversation UI
|   +-- AgentConversationView.js     # Real-time conversation observer
|   +-- ConversationTurn.js          # Single turn in a conversation
|
+-- shared/                          # Shared sub-components
    +-- SectionCard.js               # Collapsible section wrapper
    +-- StatusBadge.js               # Agent status indicator
    +-- TypeTag.js                   # I/O type label (text, number, etc.)
    +-- ActionBar.js                 # Bottom sticky action bar
```

### New Context

```
frontend/src/contexts/
  AgentFactoryContext.js              # State for current agent profile being edited
                                     # Provides: currentAgent, updateAgent, saveAgent,
                                     # activateAgent, testRun, validationErrors
                                     # Shares agent state with ChatSidebar for
                                     # Assembly Agent awareness
```

### New Service

```
frontend/src/services/
  agentFactoryService.js             # API methods for all Agent Factory endpoints
                                     # (profiles, connectors, playbooks, schedules,
                                     # journal, approvals, agent messages, schemas)
```

---

## Integration with Existing Components

### App.js Changes

Add route for Agent Factory (permission-gated):

```jsx
<Route path="/agent-factory" element={
  <ProtectedRoute requiredPermission="agent_factory">
    <AgentFactoryPage />
  </ProtectedRoute>
} />
<Route path="/agent-factory/:agentId" element={
  <ProtectedRoute requiredPermission="agent_factory">
    <AgentFactoryPage />
  </ProtectedRoute>
} />
```

### Navigation.js Changes

Add Agent Factory nav item (conditionally rendered based on user permissions):

```jsx
{user.permissions?.agent_factory && (
  <NavItem icon={<FactoryIcon />} label="Agent Factory" to="/agent-factory" />
)}
```

### ChatSidebar.js Changes

When on the Agent Factory page, the chat sidebar:
- Shows the Assembly Agent as the active assistant context
- Receives the current agent profile from `AgentFactoryContext`
- Can apply Assembly Agent suggestions to the editor

### TabbedContentManager.js Changes

Add `agent-journal` as a supported tab type, allowing journal views to open in tabs alongside documents.

---

## Responsive Behavior

### Desktop (> 1200px)

Full three-panel layout: agent list sidebar + editor + chat sidebar.

### Tablet (768px - 1200px)

- Agent list sidebar collapses to icons-only (expandable on tap)
- Chat sidebar defaults to collapsed (expandable on tap)
- Editor uses full available width

### Mobile (< 768px)

- Agent list becomes a top-level drawer (swipe from left)
- Editor is full-width
- Chat sidebar is a bottom sheet or overlay drawer
- Configuration drawers are full-screen overlays
- Step cards are simplified (one line each, tap to expand)

---

## Accessibility

- All step cards are keyboard-navigable (Tab to focus, Enter to open drawer, Arrow keys to reorder)
- Drag-to-reorder has keyboard alternative (select card, then use Shift+Arrow to move)
- Input wiring dropdowns are standard MUI Select components (full keyboard + screen reader support)
- Color coding is supplemented by icons and text (not color-only indicators)
- Agent status uses both color and icon (Active, Error, etc.)
- All drawers are focus-trapped and Escape-dismissible

---

## State Management

### AgentFactoryContext

```
AgentFactoryContext provides:
  currentAgent              # Full agent profile object being edited
  isDirty                   # Has unsaved changes
  validationErrors          # Per-section validation state
  updateSection(name, data) # Update a section of the profile
  saveAgent()               # Save to backend
  activateAgent()           # Save + activate
  testRun(query)            # Run test query
  addStep(stepConfig)       # Add a workflow step
  removeStep(index)         # Remove a workflow step
  reorderSteps(from, to)    # Reorder steps
  addConnector(id)          # Bind a connector
  removeConnector(id)       # Unbind a connector
```

### Persistence

- Agent profile edits auto-save to localStorage as drafts (like document editing)
- Explicit save persists to backend via `agentFactoryService`
- Sidebar collapse/width states saved to localStorage (consistent with existing pattern)
- Last-selected agent remembered in localStorage

---

## Future Considerations

### Workflow Visualization (Read-Only)

A React Flow-based read-only visualization could be added as an optional "View as Graph" toggle on the Workflow Steps section. This would render the step cards as a flow diagram for users who want a visual overview — but editing always happens through the card + drawer pattern.

### Template Marketplace

A future "Community Templates" section could allow users to browse and clone agent configurations shared by other users (with appropriate privacy controls).

### Drag-From-Sidebar

A future enhancement could allow dragging connectors or tools from a sidebar palette directly into the step card stack, as an alternative to the [+ Add Step] selector.
