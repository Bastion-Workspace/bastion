# Skills-Based Architecture Design

## Phased Integration Plan

### Phase 1: Define the Skill Schema

**Goal**: Establish the foundational data model that replaces per-agent capability declarations with a unified, declarative skill registry.

**Tasks**:
1. Create a `Skill` Pydantic model (or dataclass) capturing:
   - `name`, `description` (used for LLM-native discovery)
   - `engine` (which execution engine runs this skill)
   - `domains`, `actions`, `keywords` (migrated from `AGENT_CAPABILITIES`)
   - `editor_types`, `requires_editor`, `editor_preference` (editor gating)
   - `tools` (list of tool names the skill needs)
   - `subgraphs` (optional list of subgraph builders for complex skills)
   - `system_prompt` (skill-specific system prompt or prompt builder function)
   - `context_loader` (optional function ref for loading domain-specific context)
   - `priority` (for disambiguation when multiple skills match)
2. Create a `SkillRegistry` class that loads skill definitions and provides:
   - `discover(query, editor_context, conversation_context) -> List[SkillCandidate]`
   - `get_skill(name) -> Skill`
   - `get_skills_for_engine(engine_type) -> List[Skill]`
3. Write skill definitions for 3-4 simple agents first (weather, dictionary, help) to validate the schema.

**Files to create/modify**:
- `llm-orchestrator/orchestrator/skills/skill_schema.py` (new)
- `llm-orchestrator/orchestrator/skills/skill_registry.py` (new)
- `llm-orchestrator/orchestrator/skills/definitions/` (new directory for skill defs)

**Validation**: Skill definitions for weather, dictionary, and help agents should fully describe their behavior with no agent-class code needed beyond the engine.

---

### Phase 2: Build the Automation Engine

**Goal**: Replace all "simple" agents (thin wrappers around tools + system prompt) with a single generic execution engine.

**Agents to consolidate**:
- `weather_agent` → skill definition
- `dictionary_agent` → skill definition
- `help_agent` → skill definition
- `email_agent` → skill definition
- `navigation_agent` → skill definition
- `rss_agent` → skill definition
- `entertainment_agent` → skill definition
- `org_capture_agent` → skill definition
- `image_generation_agent` → skill definition
- `image_description_agent` → skill definition

**Tasks**:
1. Build `AutomationEngine` extending `BaseAgent` with a generic LangGraph workflow:
   - `load_skill` → `prepare_context` → `execute_with_tools` → `format_response`
   - Tool binding happens dynamically based on the skill's `tools` list
   - System prompt composed from skill definition + base instructions
2. Write skill definitions for all simple agents listed above.
3. Validate that each skill produces identical output to its predecessor agent.
4. Update `agent_capabilities.py` routing to dispatch simple-domain queries to `AutomationEngine` + skill.

**Estimated reduction**: ~10 agent classes (2000-4000 lines) replaced by ~10 skill definitions (200-400 lines total) + 1 engine (~300 lines).

---

### Phase 3: Build the Editor Engine

**Goal**: Consolidate all editor-gated agents into a single Editor Engine that loads domain-specific editing skills.

**Agents to consolidate**:
- `fiction_editing_agent` → skill definition + subgraph references
- `writing_assistant_agent` → skill definition (outline, character, rules, style, article, series modes)
- `rules_editing_agent` → skill definition
- `series_editing_agent` → skill definition
- `podcast_script_agent` → skill definition
- `electronics_agent` → skill definition
- `general_project_agent` → skill definition
- `proofreading_agent` → skill definition

**Tasks**:
1. Build `EditorEngine` extending `BaseAgent` with a workflow:
   - `load_skill` → `load_editor_context` → `load_references` → `route_execution` → `resolve_operations` → `format_response`
   - `route_execution` dispatches to the skill's declared subgraphs
   - Editor context loading (frontmatter, cursor position, content) is handled once by the engine
   - Reference file loading uses skill's `context_loader` config
2. Extract domain-specific system prompts and validation rules from existing agents into skill definitions.
3. Subgraphs (fiction_generation, fiction_validation, fiction_resolution, etc.) remain as-is — skills reference them by name.
4. HITL patterns remain in the engine, triggered by skill configuration flags.

**Key challenge**: The fiction editing agent has 2300+ lines of domain logic. Much of this lives in subgraphs already, but the agent-level orchestration (chapter detection, outline sync analysis, operation resolution) needs to be either:
- (a) Moved into the engine as generic "editing orchestration" patterns, or
- (b) Encapsulated in a `fiction_context_loader` function that the skill references

Option (b) is recommended — keeps the engine generic while allowing complex skills to bring their own preparation logic.

---

### Phase 4: Simplify Routing with Skill Discovery (LLM-Primary)

**Goal**: Replace intent-classifier + keyword scoring with a hard-gate filter plus a single LLM selection step.

**Implemented approach**:
1. **Hard gates (deterministic, no keywords for routing)**:
   - Greeting → instant route to `chat` (no LLM).
   - Filter by eligibility only: `internal_only`, `requires_editor` (no editor → exclude), `editor_types` (editor active but type mismatch → exclude), `requires_image_context` (no image → exclude).
   - Output: `(eligible_skills, instant_route)`. No scoring; no domain detection; keywords are not used for routing.
2. **LLM skill selection**:
   - If `instant_route` is set, use it. Otherwise, one fast-model LLM call with prompt: user query, editor context, continuity hint, and **AVAILABLE SKILLS** (name + description only).
   - Structured output: `{ "skill": "<name>", "confidence": 0.0-1.0, "reason": "brief" }`. If confidence &lt; threshold or parse failure → fallback to `chat`.
3. Intent classifier and keyword-based domain/action scoring are removed from the skill-dispatch path.

**Expected improvement**: One LLM call for routing (or zero for greetings), correct routing for natural-language queries (e.g. "What is the weather?" → weather skill) without maintaining brittle keyword lists. See [SKILLS_ARCHITECTURE_PROGRESS.md](./SKILLS_ARCHITECTURE_PROGRESS.md) for current file layout and config.

---

### Phase 5: Research Engine Refinement

**Goal**: Keep the Research Engine as a dedicated agent (it's genuinely complex) but have it accept skill-level configuration for different research modes.

**Skills within the Research Engine**:
- `deep-research` — Multi-round with gap analysis, web search, synthesis
- `quick-lookup` — Single-round, local search only, fast path
- `content-analysis` — Document comparison, no web search
- `knowledge-building` — Fact-checking, distillation, knowledge graph
- `security-analysis` — Vulnerability scanning, security-focused research
- `site-crawl` — Domain crawling and content extraction

**Tasks**:
1. Parameterize the existing `FullResearchAgent` to accept a skill configuration that controls:
   - Research depth (quick vs. deep)
   - Tool availability (web search enabled/disabled)
   - Synthesis style (academic, conversational, analytical)
   - Output format (prose, structured, tabular)
2. Consolidate `content_analysis_agent`, `knowledge_builder_agent`, `security_analysis_agent`, `site_crawl_agent`, `website_crawler_agent` into research skill variants.

---

### Phase 6: Deprecate Legacy Agents

**Goal**: Remove old agent classes once engines + skills are validated.

**Tasks**:
1. Feature-flag the new skill-based routing alongside legacy agent routing.
2. Run both paths in parallel, log discrepancies.
3. Once validated, remove legacy agent classes and update imports.
4. Clean up `AGENT_CAPABILITIES` dict (replaced by skill registry).
5. Clean up `intent_classifier.py` (replaced by skill discovery).

---

## Architecture Overview

### Current State: Many-Agent Architecture

```
User Query
    → Intent Classifier (7-node LangGraph workflow, 2 LLM calls)
        → Domain Detection (LLM)
        → Action Intent Classification (LLM)
        → Capability Scoring (deterministic)
    → Agent Selection (1 of 30+ agents)
    → Agent Execution (dedicated LangGraph workflow per agent)
    → Response
```

**Strengths**: Domain isolation, clear ownership, sophisticated editor gating.
**Weaknesses**: 30+ agent classes with duplicated boilerplate, expensive routing, hard to add new capabilities, state preservation overhead across every agent.

### Target State: Tiered Skill Architecture

```
User Query
    → Hard Gates (instant)
        → Greeting? → chat (skip LLM)
        → Else: filter_eligible() → (eligible_skills, instant_route)
    → LLM Skill Selection (if no instant_route; 1 fast-model call)
        → Prompt: query + editor context + continuity + skill name + description per eligible skill
        → Returns skill name (or "chat" on failure / low confidence)
    → Engine Dispatch (1 of 4 engines by skill.engine)
    → Skill Loading (engine loads skill definition)
    → Engine Execution (engine's LangGraph workflow, parameterized by skill)
    → Response
```

**Strengths**: Fewer execution codepaths, shared state management, declarative skill definitions (descriptions drive routing), one or zero LLM calls for routing, easy to add new skills without tuning keywords.

---

## Execution Engines

### Conversational Engine

**Purpose**: General chat, Q&A, simple tasks that don't need editor context or complex workflows.

**Workflow**:
```
prepare_context → check_handoff → generate_response → END
                      ↓
              (handoff to research engine if needed)
```

**Skills served**: General chat (default/fallback), org content queries.

**Basis**: Current `ChatAgent` with handoff-to-research logic preserved.

### Automation Engine

**Purpose**: Tool-calling tasks with structured I/O against external services.

**Workflow**:
```
load_skill → prepare_context → execute_tools → format_response → END
```

**Skills served**: Weather, email, RSS, navigation, entertainment, dictionary, help, image generation, image description, org capture.

**Key property**: The engine doesn't need domain knowledge — the skill's system prompt and tool list fully define behavior. The engine handles tool calling, error handling, and response formatting generically.

### Editor Engine

**Purpose**: All document editing operations that require active editor context.

**Workflow**:
```
load_skill → load_editor_context → load_references
    → route_execution (conditional based on skill subgraphs)
        → [skill-specific subgraphs: generation, validation, resolution, etc.]
    → resolve_operations → format_response → END
```

**Skills served**: Fiction editing, outline editing, character development, rules editing, series editing, electronics projects, general projects, podcast scripts, proofreading, article writing.

**Key property**: Handles frontmatter awareness, cursor position, editor operations (`insert_after_heading`, `replace_range`, `delete_range`), HITL confirmation, and reference file loading as engine-level concerns. Skills provide domain-specific system prompts, validation rules, and subgraph references.

### Research Engine

**Purpose**: Complex multi-round research with gap analysis, synthesis, and diverse source types.

**Workflow**: Retains current `FullResearchAgent` workflow structure with skill-level parameterization for depth, tool selection, and output format.

**Skills served**: Deep research, quick lookup, content analysis, knowledge building, security analysis, site crawling.

---

## Skill Definition Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Any
from enum import Enum

class EngineType(str, Enum):
    CONVERSATIONAL = "conversational"
    AUTOMATION = "automation"
    EDITOR = "editor"
    RESEARCH = "research"

class Skill(BaseModel):
    """Declarative skill definition — replaces per-agent classes for simple skills"""

    # Identity
    name: str = Field(description="Unique skill identifier (kebab-case)")
    description: str = Field(description="Human-readable description for LLM skill selection")

    # Routing
    engine: EngineType = Field(description="Which execution engine runs this skill")
    domains: List[str] = Field(default_factory=list, description="Matching domains")
    actions: List[str] = Field(default_factory=list, description="Supported action intents")
    keywords: List[str] = Field(default_factory=list, description="Trigger keywords for deterministic matching")
    priority: int = Field(default=50, description="Disambiguation priority (higher wins)")

    # Editor gating
    editor_types: List[str] = Field(default_factory=list, description="Supported editor document types")
    requires_editor: bool = Field(default=False, description="Hard requirement for active editor")
    editor_preference: str = Field(default="none", description="'prefer' boosts when editor present, 'require' is hard gate")

    # Execution
    tools: List[str] = Field(default_factory=list, description="Tool names this skill needs from the registry")
    subgraphs: List[str] = Field(default_factory=list, description="Subgraph builder names for complex skills")
    system_prompt: Optional[str] = Field(default=None, description="Skill-specific system prompt (simple skills)")
    context_loader: Optional[str] = Field(default=None, description="Dotted path to context loader function (complex skills)")

    # Behavior flags
    override_continuity: bool = Field(default=False, description="Explicit requests override conversation continuity")
    requires_explicit_keywords: bool = Field(default=False, description="Only match on explicit keyword presence")
    supports_hitl: bool = Field(default=False, description="Skill may trigger human-in-the-loop confirmation")
```

### Example Skill Definitions

**Simple skill (weather)**:
```python
Skill(
    name="weather",
    description="Get current weather conditions, forecasts, and historical weather data for any location",
    engine=EngineType.AUTOMATION,
    domains=["weather"],
    actions=["query", "observation"],
    keywords=["weather", "temperature", "forecast", "rain", "snow", "humidity"],
    tools=["weather_conditions", "weather_forecast", "weather_history"],
    system_prompt=(
        "You are a weather assistant. Use the available tools to answer weather queries. "
        "Always include both Fahrenheit and Celsius. Provide forecast details when relevant."
    ),
    priority=90,
)
```

**Complex skill (fiction editing)**:
```python
Skill(
    name="fiction-editing",
    description="Edit fiction manuscripts: write and rewrite chapters, continue prose, sync with outline, maintain voice and character consistency",
    engine=EngineType.EDITOR,
    domains=["fiction", "writing"],
    actions=["modification", "generation"],
    keywords=["write", "rewrite", "chapter", "continue", "prose", "manuscript"],
    editor_types=["fiction"],
    requires_editor=True,
    editor_preference="require",
    tools=["search_documents", "get_document_content", "search_segments"],
    subgraphs=["fiction_context", "fiction_generation", "fiction_validation", "fiction_resolution"],
    context_loader="orchestrator.skills.loaders.fiction_context_loader",
    supports_hitl=True,
    priority=90,
)
```

---

## Comparison with Industry Approaches

### Cursor Agent Skills (2026)

Cursor uses `SKILL.md` files with YAML frontmatter. Skills are instructions that a single agent reads and follows. Discovery is based on the agent reading skill descriptions and deciding relevance.

**What we adopt**: Declarative skill definitions, LLM-native skill selection, progressive loading (only load skill details when selected).

**What we diverge on**: Cursor has one agent; we need multiple execution engines because our workflows (editor operations, multi-round research, HITL) are too diverse for a single agent to handle generically. Cursor has no editor context, state persistence, or multi-turn conversation awareness.

### OpenClaw Skills

OpenClaw uses `SKILL.md` with richer frontmatter gating (`requires.bins`, `requires.env`, `requires.config`). Skills are filtered at load time and injected into the system prompt.

**What we adopt**: Load-time gating (our deterministic pre-filter), environment-aware skill eligibility, skill metadata for discovery.

**What we diverge on**: OpenClaw skills are instruction-only (no workflow orchestration). Our complex skills need to reference subgraphs and context loaders. OpenClaw has no concept of editor-gated skills or document-type awareness.

### Our Hybrid Advantage

Our app has capabilities that neither Cursor nor OpenClaw address:

1. **In-editor editing** — Active editor context with frontmatter, cursor position, editor operations, and HITL confirmation. No other skill system handles this.
2. **Multi-turn state persistence** — PostgreSQL checkpointing for conversation continuity across turns. Skills-based systems typically start fresh each session.
3. **Multi-model support** — User-selected LLM preferences that flow through metadata. Skills must not interfere with model selection.
4. **gRPC tool ecosystem** — Rich tool access through gRPC backends (vector search, document management, web crawling). Skills declare tool needs; engines handle the gRPC plumbing.
5. **Subgraph composition** — Complex skills compose behavior from reusable subgraph workflows (fiction generation, validation, resolution, research, gap analysis). This is execution-level sophistication that pure instruction-based skills can't express.

### OpenClaw Gateway Architecture (2026)

OpenClaw's gateway architecture adds significant capabilities worth learning from:

1. **Multi-session orchestration** — `sessions_spawn` creates sub-agent runs; `sessions_send` enables agent-to-agent ping-pong with reply-back. This is true multi-agent collaboration within a single chat thread.
2. **Tool profiles** — Per-agent tool allowlists (`minimal`, `coding`, `messaging`, `full`) replace agent routing with tool-set configuration. Instead of "which agent?", it's "which tool profile?"
3. **Plugin-extensible tools** — External tools register via plugins, extending the core set without modifying the system.
4. **Cross-surface messaging** — One gateway owns all surfaces (WhatsApp, Discord, Slack, Telegram, etc.), so agents can send messages across channels.
5. **Cron and automation** — Built-in `cron` tool for scheduled agent actions, and `gateway` tool for self-restart and config updates.

**What we adopt for Phase 7-9**: The compound query planner borrows from OpenClaw's multi-session concept (steps as lightweight sub-sessions). Tool packs borrow from their tool profiles concept. The context bridge is our equivalent of their session result forwarding.

**What we diverge on**: OpenClaw's single-agent-with-all-tools model works for general-purpose chat but doesn't handle domain-specific workflows (fiction editing pipelines, multi-round research with gap analysis). Our curated skill + engine model produces higher-quality domain-specific output.

---

## Current Implementation Status

See [SKILLS_ARCHITECTURE_PROGRESS.md](./SKILLS_ARCHITECTURE_PROGRESS.md) for up-to-date status.

**Done**: Skill schema and registry; all four engines (Automation, Conversational, Editor, Research); unified dispatch and `grpc_service` integration; LLM-primary routing (filter_eligible + llm_select_skill; no keyword scoring). Skill routing uses the fast model from settings. **Skill dispatch is the only path** — the `SKILL_DISPATCH_ENABLED` flag has been removed. Phase 7 Compound Query Planner (plan_models, plan_engine, context_bridge, llm_select_skill_or_plan) built and wired into grpc_service; P0 bridging tools (summarize_text_tool, extract_structured_data_tool, clipboard_store_tool, clipboard_get_tool) implemented and registered.

**Agent reduction (completed)**: ~26 redundant agent files have been deleted. The Automation Engine fully replaces the former dedicated automation agents (weather, dictionary, help, email, navigation, rss, entertainment, org_capture, image_generation, image_description, reference, technical_hyperspace, learning, org_content, etc.). Editor/research/conversational flows are handled by EditorEngine (WritingAssistantAgent), ResearchEngine (FullResearchAgent), and ConversationalEngine (ChatAgent). Only base_agent, chat_agent, full_research_agent, writing_assistant_agent, and proposal_generation_agent remain. The legacy `elif agent_type ==` chain and intent-classifier fallback have been removed from `grpc_service.py`; checkpoint/shared_memory loading uses the chat agent from the unified dispatcher. Short-circuits (/help, /define, explicit agent_type) set `discovered_skill` from the registry; /hyperspace short-circuit removed. All EDITOR skills (including former electronics, general_project, podcast_script) route through EditorEngine. Email HITL addressed via `confirmed` param on send/reply tools; image_description addressed via automation engine image injection for `requires_image_context` skills and attachment_analysis_subgraph inline vision LLM.

**Recently completed (Feb 2026):** `document_creator` automation skill and supporting tooling. Backend: `GetFolderTree` RPC and handler in `grpc_tool_service.py` (uses `folder_service.get_folder_tree()`, flattens tree to proto `FolderInfo`). Orchestrator: `get_folder_tree()` in `backend_tool_client.py`; `list_folders_tool` in `file_creation_tools.py` (formats folder tree for LLM); `create_user_file_tool` and `create_user_folder_tool` now return human-readable strings and are registered in `tools/__init__.py`. Skill: `document_creator` in `automation_skills.py` with workflow (list_folders → create folder if needed → create_user_file), use of `prior_step_*_response` for compound plans. Enables compound flows like "Research X and put results in folder Y" (research step → document_creator step with context bridge).

**Phase 5 (Research Engine Parameterization) complete:** Full Research Agent split into `research/` package; skill config for all 7 research skills; nodes and research_workflow_subgraph respect skill_config; five synthesis prompt variants plus comprehensive default; state preservation (skill_config and critical 5) in all node returns.

**Remaining**: Optional deepening of engine consolidation; Phase 8 (Tool Packs and Mid-Execution Augmentation) next; then Phase 9 (Hierarchical Skill Selection); Phase 10–12 (Inter-Agent Collaboration, Structured Protocols, A2A Interoperability) as designed.

---

## Phase 7: Compound Query Planner

**Goal**: Enable a single user message to invoke multiple skills in sequence or parallel, with context flowing between them.

**Problem**: The current system maps one query to one skill to one engine. Compound requests like "Research X, capture key points to my inbox, and update chapter 12" require three skills (research → org_capture → fiction_editing) but only one gets selected.

### Design: Plan-Then-Execute

```
User Query
    → Hard Gates (same as Phase 4)
    → LLM Compound Detection (fast model)
        → Single intent? → Normal skill dispatch (Phases 1-6)
        → Compound intent? → Generate execution plan
    → Plan Execution Loop
        → For each step: dispatch(skill, sub_query, accumulated_context)
        → Context bridge: pass results forward as shared_memory
    → Aggregate Response
```

### Planner Output Schema

```python
class PlanStep(BaseModel):
    """A single step in a compound execution plan."""
    step_id: int
    skill_name: str
    sub_query: str = Field(description="Focused query for this skill")
    depends_on: List[int] = Field(default_factory=list, description="Step IDs that must complete first")
    context_keys: List[str] = Field(default_factory=list, description="Keys from prior steps to inject")

class ExecutionPlan(BaseModel):
    """Compound query execution plan."""
    is_compound: bool = Field(description="True if query needs multiple skills")
    steps: List[PlanStep] = Field(default_factory=list)
    reasoning: str = Field(description="Why this plan was chosen")
```

### Execution Semantics

- Steps with no `depends_on` can run in parallel
- Steps with dependencies wait for predecessors; results injected via `context_keys`
- Each step calls `UnifiedDispatcher.dispatch()` with accumulated context in `shared_memory`
- The planner is an LLM call only when the initial skill selector signals compound intent (e.g., confidence split across multiple skills, or explicit multi-verb query)
- Single-intent queries (80%+ of traffic) bypass the planner entirely — zero added latency

### Context Bridge

The context bridge is the mechanism for passing results between plan steps:

```python
class ContextBridge:
    """Accumulates results across plan steps for injection into subsequent steps."""
    
    def __init__(self):
        self.step_results: Dict[int, Dict[str, Any]] = {}
    
    def store_result(self, step_id: int, result: Dict[str, Any]):
        self.step_results[step_id] = {
            "response_text": result.get("response", ""),
            "structured_data": result.get("structured_data", {}),
            "skill_name": result.get("agent_type", ""),
        }
    
    def build_context_for_step(self, step: PlanStep) -> Dict[str, Any]:
        """Build shared_memory injection for a step from its dependencies."""
        context = {}
        for dep_id in step.depends_on:
            dep_result = self.step_results.get(dep_id, {})
            for key in step.context_keys:
                if key in dep_result:
                    context[f"step_{dep_id}_{key}"] = dep_result[key]
            # Always inject prior response text for LLM context
            context[f"prior_step_{dep_id}_response"] = dep_result.get("response_text", "")
        return context
```

### Key Design Decisions

1. **The planner does NOT decompose within skills** — a fiction editing pipeline stays intact. The planner composes between skills.
2. **Planner is optional** — single-intent queries (the vast majority) skip it entirely.
3. **Fail-safe** — if the planner fails or produces a bad plan, fall back to single-skill dispatch on the highest-confidence skill.
4. **Maximum plan length** — cap at 3-4 steps to prevent runaway plans. Longer sequences are better handled as multi-turn conversations.

### Implementation Location

- `llm-orchestrator/orchestrator/engines/plan_engine.py` (new)
- `llm-orchestrator/orchestrator/engines/context_bridge.py` (new)
- Modifications to `unified_dispatch.py` to support plan-mode dispatch

---

## Phase 8: Tool Packs and Mid-Execution Augmentation

**Goal**: Allow the planner (or an engine mid-execution) to pull in tools from outside the skill's declared tool set.

**Problem**: A weather skill declares `["get_weather_tool"]`. But a user might ask "What's the weather, and capture that to my inbox." If the planner decomposes this into two steps, step 2 needs `add_org_inbox_item_tool` — but what if a simpler query just needs one tool from another domain mid-conversation?

### Tool Packs

Tool packs are named groups of related tools that can be requested by the planner or by an engine node:

```python
class ToolPack(BaseModel):
    """A named, requestable group of tools for mid-execution augmentation."""
    name: str
    description: str
    tools: List[str]
    
TOOL_PACKS = {
    "capture": ToolPack(name="capture", description="Capture items to org inbox", tools=["add_org_inbox_item_tool"]),
    "search_local": ToolPack(name="search_local", description="Search local documents and images", tools=["search_documents_tool", "search_images_tool"]),
    "search_web": ToolPack(name="search_web", description="Search the web", tools=["search_web_tool", "crawl_web_content_tool"]),
    "email_read": ToolPack(name="email_read", description="Read and search email", tools=["get_emails_tool", "search_emails_tool", "get_email_thread_tool"]),
    "email_send": ToolPack(name="email_send", description="Send and reply to email", tools=["send_email_tool", "reply_to_email_tool"]),
    "visualization": ToolPack(name="visualization", description="Create charts and diagrams", tools=["create_chart_tool"]),
    "math": ToolPack(name="math", description="Calculations and unit conversion", tools=["calculate_expression_tool", "evaluate_formula_tool", "convert_units_tool"]),
    "document_ops": ToolPack(name="document_ops", description="Read and search within documents", tools=["get_document_content_tool", "search_within_document_tool"]),
    "navigation": ToolPack(name="navigation", description="Locations and route planning", tools=["list_locations_tool", "compute_route_tool"]),
}
```

### Augmentation Flow

1. Planner step declares `tool_packs: ["capture", "visualization"]` in addition to the skill's tools
2. Engine resolves the union of skill tools + requested tool pack tools
3. LLM sees the augmented tool set for that step only
4. No change to the skill definition itself — augmentation is per-plan-step

### Skill Fragments: Partial Skill Access

For the case where a planner needs ONE part of a complex skill (e.g., just the "search local documents" capability from the research skill, without the full multi-round gap analysis workflow):

- **Tool packs** handle the simple case: just need `search_documents_tool`? Request the `search_local` tool pack.
- **Subgraph invocation** handles the complex case: need intelligent document retrieval with image search? The planner can reference `intelligent_document_retrieval_subgraph` as a step type.

This introduces a new plan step type:

```python
class PlanStep(BaseModel):
    step_id: int
    # Choose one: skill_name for full skill, or subgraph_name for a fragment
    skill_name: Optional[str] = None
    subgraph_name: Optional[str] = None
    sub_query: str
    depends_on: List[int] = Field(default_factory=list)
    context_keys: List[str] = Field(default_factory=list)
    tool_packs: List[str] = Field(default_factory=list, description="Additional tool packs to augment")
```

This is the key insight: **subgraphs become first-class plan step targets**. The planner can say "run the gap_analysis subgraph" or "run the visualization subgraph" without invoking the full research agent. The subgraphs already exist and are already tested — they just need to be invocable from outside their parent skill.

---

## Phase 9: Hierarchical Skill Selection

**Goal**: Scale skill selection beyond 30-40 skills without prompt bloat.

**Problem**: The current `llm_select_skill` puts all eligible skill descriptions into one prompt. At 30 skills this works. At 100+ skills (as tool packs and fragments are added), the prompt becomes unwieldy and selection accuracy drops.

### Two-Stage Selection

```
Stage 1: Domain Router (deterministic + fast LLM)
    → Input: query, editor context
    → Output: 1-3 domain categories (e.g., "research", "editing", "automation")
    → Method: Keyword pre-filter + fast LLM if ambiguous

Stage 2: Skill Selector (within domain)
    → Input: query, editor context, skills filtered to selected domains
    → Output: skill name
    → Method: Current llm_select_skill, but with 5-10 skills instead of 30
```

This keeps selection prompts focused and allows the system to scale to 100+ skills without degradation.

---

## Phase 10: Inter-Agent Collaboration Sessions

**Goal**: Enable multiple skills to collaborate in a supervised loop -- exchanging messages, critiquing each other's output, and converging on a result -- rather than only passing snapshots forward sequentially.

**Problem**: The Compound Query Planner (Phase 7) runs skills sequentially or in parallel with one-way context passing. This handles "Research X then save to Y" but cannot handle patterns where one skill's output needs to be evaluated, critiqued, and revised by another skill before proceeding. Examples: research → fact-check → revise; draft email → tone review → revise; generate chapter → continuity check → revise. The fiction pipeline already does this as hardcoded subgraph composition (generation → validation → resolution), but the pattern isn't generic.

### Design: Supervisor-Worker Collaboration

```
Supervisor Node
    → Select next worker (LLM or rule-based)
    → Worker executes (skill invocation via UnifiedDispatcher)
    → Result enters shared session thread
    → Check convergence
        → Not converged? → Back to supervisor
        → Converged? → Format final response → END
```

### Collaboration Session Model

```python
class AgentMessage(BaseModel):
    """Structured message exchanged between collaborating skills."""
    sender: str              # Skill name
    recipient: str           # Skill name or "supervisor"
    message_type: str        # "proposal", "critique", "revision", "approval", "request_info"
    content: str             # Natural language content
    structured_data: Dict[str, Any] = {}  # Typed payload (e.g., extracted facts, confidence scores)
    references: List[str] = []  # Document IDs, URLs, or step IDs referenced
    status: str              # "pending", "accepted", "rejected", "needs_revision"

class CollaborationSession(BaseModel):
    """Shared state for a multi-skill collaboration."""
    session_id: str
    participants: List[str]  # Skill names involved
    messages: List[AgentMessage]
    convergence_criteria: Dict[str, Any]  # e.g., {"max_turns": 6, "require_approval": True}
    current_turn: int
    status: str  # "active", "converged", "failed", "max_turns_reached"
```

### Convergence Criteria

Sessions terminate when any of:
- A designated reviewer skill emits `message_type: "approval"` with `status: "accepted"`
- `max_turns` is reached (fail-safe against infinite loops)
- A confidence score from the supervisor exceeds a configured threshold
- All participants signal "no further changes needed"

### Integration with Compound Planner

`PlanStep` gains a new step type:

```python
class PlanStep(BaseModel):
    step_id: int
    skill_name: Optional[str] = None        # Single-skill step
    subgraph_name: Optional[str] = None     # Fragment step
    collaboration: Optional[CollaborationConfig] = None  # Multi-skill collaboration step
    sub_query: str
    depends_on: List[int] = Field(default_factory=list)
    context_keys: List[str] = Field(default_factory=list)
    tool_packs: List[str] = Field(default_factory=list)

class CollaborationConfig(BaseModel):
    """Configuration for a collaboration plan step."""
    participants: List[str]  # Skill names (e.g., ["research", "fact_checker"])
    supervisor_strategy: str  # "llm" (LLM picks next speaker) or "round_robin" or "role_based"
    convergence: Dict[str, Any]  # {"max_turns": 6, "require_approval": True}
    roles: Dict[str, str] = {}  # Optional role assignments: {"research": "generator", "fact_checker": "critic"}
```

The planner decides when to use collaboration vs. sequential: simple "research then save" stays sequential; "research, verify facts, then save" uses a collaboration session for the research + verification step.

### Use Cases Unlocked

| Pattern | Participants | Flow |
|---------|-------------|------|
| Generator + Critic | research + fact_checker | Research synthesizes → critic evaluates → research revises → critic approves |
| Draft + Review | email + tone_reviewer | Draft email → reviewer checks tone/clarity → revise → approve → send |
| Multi-Perspective | research (optimist) + research (pessimist) | Two research invocations with different system prompts → supervisor synthesizes balanced view |
| Write + Verify | fiction_editing + proofreading | Generate chapter → proofread → revise → approve (generic version of current hardcoded fiction pipeline) |
| Research + Organize | research + document_creator | Research produces findings → document_creator structures them → research validates completeness → save |

### Implementation Location

- `llm-orchestrator/orchestrator/engines/collaboration_engine.py` (new) — Supervisor StateGraph with worker dispatch and convergence checking
- `llm-orchestrator/orchestrator/engines/collaboration_session.py` (new) — `CollaborationSession`, `AgentMessage`, turn management
- Modifications to `plan_models.py` — `CollaborationConfig` and step type extension
- Modifications to `plan_engine.py` — Dispatch collaboration steps to `CollaborationEngine`

### Why This Matters

The fiction pipeline proves that generator→validator→revision loops produce better output than single-pass generation. Making this pattern generic means *every* domain benefits: research gets fact-checked, emails get tone-reviewed, documents get verified before saving. The collaboration engine is the generalization of what's already proven in the editor pipeline.

---

## Phase 11: Structured Inter-Agent Protocols

**Goal**: Formalize the message contracts between collaborating skills so that inter-agent conversations are debuggable, auditable, and type-safe.

**Problem**: If collaboration sessions use free-form text passing, debugging failures becomes opaque ("why did the critic reject this?"). Structured protocols make every exchange traceable.

### Message Type Taxonomy

```python
class MessageType(str, Enum):
    PROPOSAL = "proposal"        # Initial or revised output from a generator
    CRITIQUE = "critique"        # Evaluation with specific issues identified
    REVISION = "revision"        # Updated output addressing critique
    APPROVAL = "approval"        # Reviewer accepts the current state
    REJECTION = "rejection"      # Reviewer rejects with reasons (triggers revision or termination)
    REQUEST_INFO = "request_info"  # Skill requests additional context from another skill
    INFO_RESPONSE = "info_response"  # Response to an info request
    DELEGATION = "delegation"    # Supervisor delegates a sub-task to a specific skill
```

### Protocol Rules (Enforced by Supervisor)

1. A session must start with a `PROPOSAL` or `DELEGATION`.
2. A `CRITIQUE` must follow a `PROPOSAL` or `REVISION` (can't critique nothing).
3. A `REVISION` must follow a `CRITIQUE` (can't revise without feedback).
4. An `APPROVAL` terminates the session (convergence).
5. A `REJECTION` after max revision attempts terminates with failure.
6. `REQUEST_INFO` / `INFO_RESPONSE` can occur at any point (information gathering).

### Collaboration Trace

Every collaboration session produces an auditable trace:

```python
class CollaborationTrace(BaseModel):
    """Audit trail for a completed collaboration session."""
    session_id: str
    participants: List[str]
    total_turns: int
    outcome: str  # "converged", "max_turns", "failed"
    messages: List[AgentMessage]
    duration_ms: int
    model_calls: int  # Total LLM invocations across all participants
    convergence_reason: str  # "approval received", "max turns reached", etc.
```

This trace can be:
- Logged for debugging ("why did this research+fact-check collaboration take 6 turns?")
- Stored for audit compliance (EU AI Act Article 14 traceability)
- Returned to the user as optional transparency ("Here's how the agents collaborated on your request")

### Implementation Notes

- Message type validation is a simple state machine in the supervisor node -- low overhead, high value.
- The structured messages are Pydantic models, consistent with the project's structured-output philosophy (no string matching).
- Collaboration traces are stored alongside the normal response in checkpoint state for persistence across sessions.

---

## Phase 12: External Agent Interoperability (A2A Protocol)

**Goal**: Expose skills as discoverable, invocable agents that external systems can collaborate with, following Google's Agent2Agent (A2A) open standard.

**Problem**: The system is self-contained -- skills can only be invoked internally. A2A interoperability enables two things: (1) external systems (other LangGraph deployments, AutoGen clusters, Claude Code instances) can discover and use your skills as services, and (2) your system can delegate to external agents it doesn't control.

### Agent Cards from Skill Definitions

Skill definitions already contain the metadata A2A Agent Cards require. Generation is automatic:

```python
# Skill definition (already exists):
Skill(
    name="research",
    description="Multi-round deep research with gap analysis and synthesis",
    engine=EngineType.RESEARCH,
    tools=["search_documents_tool", "search_web_tool", ...],
)

# Auto-generated A2A Agent Card:
{
    "name": "bastion-research",
    "description": "Multi-round deep research with gap analysis and synthesis",
    "url": "https://your-system/a2a/skills/research",
    "version": "1.0.0",
    "capabilities": {
        "streaming": true,
        "pushNotifications": false,
        "stateTransitionHistory": true
    },
    "skills": [
        {
            "id": "deep-research",
            "name": "Deep Research",
            "description": "Multi-round research with gap analysis, web search, and synthesis"
        }
    ],
    "authentication": {
        "schemes": ["bearer"]
    }
}
```

### A2A Task Lifecycle

Following the A2A spec, external agents interact with skills through a stateful task lifecycle:

```
External Agent → POST /a2a/tasks/send (creates task)
    → Task status: "submitted" → "working" → "completed"
    → External Agent polls or receives SSE updates
    → Artifacts (research results, created documents, etc.) returned on completion
    → If HITL needed: task status → "input-required" → external agent provides input → resumes
```

### Inbound vs. Outbound

**Inbound (external → your skills):**
- A2A HTTP endpoint accepts tasks, maps them to skill invocations via `UnifiedDispatcher.dispatch()`.
- Authentication, rate limiting, and cost controls at the gateway level.
- Results returned as A2A `Artifact` objects (text, files, structured data).

**Outbound (your skills → external agents):**
- A new tool: `delegate_to_external_agent_tool` that discovers external Agent Cards, sends tasks, and awaits results.
- The collaboration engine can include external agents as participants (they just have higher latency).
- Planner can include external delegation as a plan step type.

### When to Build This

A2A is relevant when:
- You want external systems to use your research, document creation, or editing capabilities as a service.
- You want your system to delegate specialized tasks (e.g., code generation, data analysis) to external agents.
- You're operating in a multi-system environment where different teams run different agent frameworks.

For a single-system deployment, Phases 10-11 (internal collaboration + structured protocols) provide 90%+ of the value. A2A adds the final 10% for multi-system interoperability.

### Implementation Location

- `llm-orchestrator/orchestrator/a2a/agent_card_generator.py` (new) — Generate Agent Cards from skill registry
- `llm-orchestrator/orchestrator/a2a/a2a_endpoint.py` (new) — HTTP endpoint handling A2A task lifecycle
- `llm-orchestrator/orchestrator/a2a/external_agent_client.py` (new) — Outbound client for delegating to external A2A agents
- `llm-orchestrator/orchestrator/tools/external_delegation_tools.py` (new) — `delegate_to_external_agent_tool`

---

## New Tools, Skills, and Fragments Roadmap

This section catalogs tools, skills, and subgraph fragments beyond what currently exists that would enable maximum flexibility for the planner and mid-execution augmentation.

### Cross-Cutting Utility Tools (High Priority)

These are "glue" tools that the planner needs to bridge between skill steps.

| Tool | Description | Enables |
|------|-------------|---------|
| `summarize_text_tool` | Distill long text to N sentences/bullets with configurable style | Planner passing research results to capture or editing steps without overwhelming context |
| `extract_structured_data_tool` | Extract key-value pairs, named entities, dates, lists from text | Planner parsing research results for downstream tools (e.g., extract URLs for bookmarking) |
| `transform_format_tool` | Convert between formats (markdown → org, bullets → prose, JSON → table, etc.) | Bridging output of one skill to input format of another (research prose → org inbox item) |
| `clipboard_store_tool` | Store/retrieve named values in session-scoped memory | Multi-step workflows referencing earlier results by name (e.g., "the URL from step 1") |
| `compare_texts_tool` | Diff two text blocks, highlight additions/removals/changes | Proofreading → editing verification, before/after comparison |
| `merge_texts_tool` | Combine multiple text blocks with configurable strategy (concatenate, interleave, deduplicate) | Aggregating results from parallel plan steps |

### Document Management Tools (Medium Priority)

Currently tools can search and read documents but have limited creation/mutation capability outside the editor engine.

| Tool | Description | Enables |
|------|-------------|---------|
| `create_document_tool` | Create a new document with content, frontmatter type, and folder placement | Research → document creation pipeline; planner creating output artifacts |
| `append_to_document_tool` | Append content to an existing document (without full editor context) | Incremental capture, journal-style workflows, log appending |
| `list_documents_by_type_tool` | List documents filtered by frontmatter type (fiction, outline, reference, etc.) | Planner discovering what documents exist for editing; context gathering |
| `get_document_metadata_tool` | Get frontmatter and metadata without loading full content | Lightweight context for planning decisions (does this doc exist? what type is it?) |
| `tag_document_tool` | Add/remove tags on a document's frontmatter | Post-research organization, automated categorization |
| `move_document_tool` | Move a document to a different folder | Organizational workflows, inbox triage |

### Context and Memory Tools (High Priority)

These enable the planner and skills to maintain awareness across turns and sessions.

| Tool | Description | Enables |
|------|-------------|---------|
| `get_conversation_summary_tool` | Summarize the current conversation so far (last N turns) | Long conversations needing context compression for new skill steps |
| `store_fact_tool` | Store a persistent fact in user's knowledge base (key-value with optional expiry) | Research → knowledge capture; learning workflows; user preference accumulation |
| `recall_facts_tool` | Retrieve stored facts by topic or key | Any skill needing user-specific context across sessions |
| `get_user_preferences_tool` | Retrieve user preferences for a domain (writing style, temperature units, etc.) | Personalization: planner choosing skill parameters based on preferences |
| `get_session_plan_tool` | Retrieve the current execution plan (if compound query) | Mid-execution skills knowing what step they're in, what's next |

### Notification and Scheduling Tools (Medium Priority)

| Tool | Description | Enables |
|------|-------------|---------|
| `notify_user_tool` | Send a real-time notification/toast to the frontend | Long-running plan steps reporting progress; background task completion |
| `schedule_reminder_tool` | Schedule a future notification or prompt | Weather + calendar integration; deadline tracking; follow-up reminders |
| `schedule_skill_run_tool` | Schedule a skill to run at a future time | "Research X every Monday and capture to my inbox" |

### Export and Delivery Tools (Lower Priority)

| Tool | Description | Enables |
|------|-------------|---------|
| `export_to_format_tool` | Export content as PDF, DOCX, HTML, or plain text | Research → deliverable pipeline; document distribution |
| `email_content_tool` | Package content and send via email (combines formatting + send) | Research → email report pipeline |
| `bookmark_url_tool` | Save a URL with tags, notes, and optional summary | Research → bookmarking; web crawl → save interesting links |

### Subgraph Fragments for Planner Access

These are **existing subgraphs** that should be exposed as standalone plan step targets. The planner can invoke these without running their parent skill's full workflow.

| Subgraph | Current Owner | Standalone Use Case |
|----------|---------------|---------------------|
| `intelligent_document_retrieval` | Research Agent | Planner needs local doc + image search without full multi-round research |
| `image_search` | Research Agent | Planner needs image search in a non-research context (e.g., "show me X then edit Y") |
| `gap_analysis` | Research Agent | Planner wants to assess knowledge gaps before committing to full research |
| `web_research` | Research Agent | Planner needs a single focused web search round, not multi-round with gap analysis |
| `visualization` | Research Agent | Any skill wanting a chart from data (e.g., reference → chart, weather history → chart) |
| `proofreading` | Editor Engine | Planner wants a quick proofread without full editor context loading ceremony |
| `fact_verification` | Knowledge Builder | Planner needs to verify a claim mid-conversation before acting on it |
| `data_formatting` | Research Agent | Format results into structured output (tables, lists, comparison matrices) |
| `attachment_analysis` | Chat Agent | Analyze an uploaded file mid-conversation without rerouting to a different agent |
| `collection_search` | Research Agent | Search user's media collection (comics, photos) as a standalone step |
| `entity_relationship` | Research Agent | Build entity relationship maps from content for visualization or analysis |

**Implementation pattern for fragment access**: Each fragment becomes invocable via the plan engine with a standard state interface:

```python
# Fragment invocation via plan engine
class FragmentStep(PlanStep):
    """A plan step that runs a subgraph fragment instead of a full skill."""
    subgraph_name: str
    input_mapping: Dict[str, str]  # Maps context bridge keys to subgraph state keys
    output_mapping: Dict[str, str]  # Maps subgraph output keys to context bridge keys
```

The plan engine wraps the subgraph invocation, handles state setup/teardown, and maps inputs/outputs through the context bridge.

### New Skill Candidates

Skills that don't exist yet but would unlock planner flexibility (or recently added):

| Skill | Engine | Description | Key Tools/Fragments | Status |
|-------|--------|-------------|---------------------|--------|
| `document_creator` | Automation | Create a new document from research/conversation results; list folders, create folder/file | `list_folders_tool`, `create_user_file_tool`, `create_user_folder_tool` | **Done** (Feb 2026) |
| `daily_briefing` | Research | Compile weather + email + org todos into a morning briefing | Compound: weather, email_read, org_content tools | Planned |
| `meeting_notes` | Editor | Structure meeting notes from a conversation or transcript | `summarize_text_tool`, `extract_structured_data_tool` | Planned |
| `learning_review` | Conversational | Review and quiz on stored learning items | `recall_facts_tool`, lesson tools | Planned |
| `project_status` | Research | Compile status across org todos, documents, and recent activity | `list_org_todos_tool`, `list_documents_by_type_tool` | Planned |
| `comparison_report` | Research | Compare multiple documents/data sources into a structured report | `compare_texts_tool`, `data_formatting` subgraph | Planned |
| `inbox_triage` | Automation | Review and categorize org inbox items with AI suggestions | `list_org_todos_tool`, `tag_document_tool` | Planned |
| `template_fill` | Automation | Fill a document template with provided data | `get_document_content_tool`, `create_document_tool` | Planned |
| `research_to_document` | Research | Full pipeline: research → synthesize → create document | `web_research` fragment, `create_document_tool` | Planned |
| `digest_emails` | Automation | Summarize unread emails into a digest | `get_emails_tool`, `summarize_text_tool` | Planned |

### Tool Pack Definitions (for Phase 8)

Canonical tool packs that the planner can request to augment any skill:

| Pack Name | Tools | Use Case |
|-----------|-------|----------|
| `capture` | `add_org_inbox_item_tool` | Quick capture to inbox from any context |
| `search_local` | `search_documents_tool`, `search_images_tool` | Local document/image search |
| `search_web` | `search_web_tool`, `crawl_web_content_tool` | Web search and content extraction |
| `email_read` | `get_emails_tool`, `search_emails_tool`, `get_email_thread_tool`, `get_email_statistics_tool` | Read and search email |
| `email_send` | `send_email_tool`, `reply_to_email_tool` | Send and reply (HITL gated) |
| `visualization` | `create_chart_tool` | Chart generation from any data |
| `math` | `calculate_expression_tool`, `evaluate_formula_tool`, `convert_units_tool` | Calculations and conversions |
| `document_ops` | `get_document_content_tool`, `search_within_document_tool`, `get_document_metadata_tool` | Document reading and inspection |
| `document_write` | `create_document_tool`, `append_to_document_tool`, `tag_document_tool` | Document creation and modification |
| `navigation` | `list_locations_tool`, `compute_route_tool` | Location and routing |
| `org_tools` | `parse_org_structure_tool`, `list_org_todos_tool`, `search_org_headings_tool`, `get_org_statistics_tool` | Org-mode file queries |
| `memory` | `store_fact_tool`, `recall_facts_tool`, `clipboard_store_tool` | Persistent and session memory |
| `formatting` | `summarize_text_tool`, `extract_structured_data_tool`, `transform_format_tool`, `merge_texts_tool` | Text transformation and bridging |

---

## Reliability Analysis

### Why Fewer Engines Are More Reliable Than Many Agents

1. **Fewer execution codepaths to test**: 3-4 engine workflows vs. 30+ agent workflows. Each engine is tested once, thoroughly.
2. **Shared state management**: The "critical 5 keys" preservation pattern (metadata, user_id, shared_memory, messages, query) is implemented once per engine, not 30+ times.
3. **Hard gates plus LLM selection**: Eligibility filter (editor gating, requires_editor, requires_image_context) narrows candidates; one fast LLM call selects by intent from descriptions, avoiding brittle keyword-based routing.
4. **Skill definitions are data**: They can be validated, tested, and versioned independently of execution logic. A bad skill definition is caught at load time, not at runtime.
5. **Progressive loading reduces prompt bloat**: Only the selected skill's tools and context are loaded, keeping the LLM's context focused.
6. **Subgraph reuse is proven**: The existing subgraphs (fiction_generation, research_workflow, etc.) are already well-tested and shared across agents.

### Risk Areas

1. **Editor Engine complexity**: Must handle all editing skill variants without becoming a monolith. Mitigated by keeping domain logic in skill context loaders and subgraphs.
2. **Skill definition correctness**: A misconfigured skill (wrong tools, wrong editor type) could cause subtle failures. Mitigated by schema validation and integration tests per skill.
3. **Migration period**: Running old agents and new engines simultaneously requires careful feature flagging. Mitigated by phased rollout (simple skills first, complex skills last).

---

## Estimated Impact

### Phases 1-6: Engine Consolidation (Done/In Progress)

| Metric | Before (Many-Agent) | After (Skill-Engine) |
|--------|---------------------|----------------------|
| Agent classes | 30+ | 3-4 engines |
| Skill definitions | 0 | ~30 |
| Total agent code lines | ~15,000+ | ~3,000 (engines) + ~1,000 (skill defs) |
| Routing LLM calls | 2 per query | 0-1 per query |
| State preservation implementations | 30+ (per agent) | 3-4 (per engine) |
| Time to add new simple capability | New agent class (~200-400 lines) | New skill definition (~20-40 lines) |
| Time to add new editing capability | New agent class (~500-1000+ lines) | New skill def + context loader (~100-200 lines) |

### Phases 7-9: Planner and Augmentation (Planned)

| Metric | Current (Single-Skill) | After (Planner + Tool Packs) |
|--------|------------------------|------------------------------|
| Queries handleable per turn | 1 skill | 1-4 skills (compound plans) |
| Cross-domain tasks | Manual multi-turn | Single-turn with context bridge |
| Tool availability per step | Skill-declared only | Skill tools + requested tool packs |
| Subgraph reuse | Within parent skill only | Any subgraph as plan step fragment |
| New tool packs (estimated) | 0 | ~14 packs covering all tool domains |
| New utility tools needed | 0 | ~15 cross-cutting tools |
| New skill candidates | 0 | ~10 compound/pipeline skills |
| Skill selection scaling | 30 skills in one prompt | 5-10 per domain via hierarchical selection |

### Phases 10-12: Inter-Agent Collaboration and Interoperability (Planned)

| Metric | Current (Sequential/Parallel Only) | After (Collaboration + Protocols + A2A) |
|--------|------------------------------------|-----------------------------------------|
| Skill interaction model | One-way context passing (upstream → downstream) | Bidirectional: skills critique, revise, and converge in loops |
| Generator + critic pattern | Hardcoded per domain (fiction subgraphs only) | Generic: any skill pair can collaborate with configurable convergence |
| Audit/debug for multi-skill | Log inspection only | Structured collaboration traces with typed messages and turn history |
| External agent interop | None (closed system) | A2A Agent Cards auto-generated from skills; inbound/outbound task delegation |
| Output quality for complex tasks | Single-pass per skill | Multi-pass with verification loops (research → fact-check → revise) |
| Time to add collaboration pattern | New subgraph composition (~200-400 lines) | Collaboration config on plan step (~10-20 lines) |
| Compliance readiness (EU AI Act Art. 14) | Partial (checkpoint state) | Full: typed message traces, decision attribution, convergence audit trails |
