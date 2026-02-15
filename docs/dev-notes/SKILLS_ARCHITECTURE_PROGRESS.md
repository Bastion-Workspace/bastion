# Skills Architecture – Progress and Status

Status as of February 2026. For design and phased plan, see [SKILLS_ARCHITECTURE_DESIGN.md](./SKILLS_ARCHITECTURE_DESIGN.md).

## Current Status

### Completed Phases (Engine Consolidation)

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Skill schema, registry, definitions for all engines | Done |
| 2 | Automation Engine (generic tool-calling engine) | Done |
| 3 | Unified dispatch + `grpc_service` integration | Done |
| 4 | Editor Engine (wrapper around WritingAssistantAgent) | Done |
| 5 | Research Engine (wrapper around FullResearchAgent) | Done |
| 6 | Conversational Engine (wrapper around ChatAgent) | Done |
| 7 | LLM-primary routing (filter_eligible + llm_select_skill; no keyword scoring) | Done |
| 8 | gRPC tool wrappers (weather, email, navigation, rss, org_capture, image_generation) | Done |
| 9 | Skill dispatch as only path; legacy removed; ~26 agent files deleted; flag removed | Done |
| 10 | document_creator skill + folder/file tools (GetFolderTree RPC, list_folders_tool, create_user_file_tool/create_user_folder_tool registered; skill in automation_skills.py) | Done |
| 11 | P0 bridging tools (summarize_text_tool, extract_structured_data_tool, clipboard_store_tool, clipboard_get_tool) implemented and registered | Done |
| 12 | Compound Query Planner (plan_models, plan_engine, context_bridge, llm_select_skill_or_plan) built and wired into grpc_service | Done |

### Remaining Work (Current → Next)

| Phase | Description | Status | Priority |
|-------|-------------|--------|----------|
| 11 (Design Phase 7) | Compound Query Planner | Done | — |
| 12 (Design Phase 8) | Tool Packs and Mid-Execution Augmentation | Done; tool_pack_registry redesigned (Pydantic ToolPack), PlanStep.tool_packs, engine merge, P1 tools, vector store and discovery updated | — |
| 13 (Design Phase 9) | Hierarchical Skill Selection | Design complete | Low (scales at 50+ skills) |
| 14 (Design Phase 10) | Inter-Agent Collaboration Sessions (supervisor-worker loops, convergence criteria, generic generator+critic pattern) | Design complete | Medium |
| 15 (Design Phase 11) | Structured Inter-Agent Protocols (typed messages, protocol rules, collaboration traces for audit/debug) | Design complete | Medium |
| 16 (Design Phase 12) | External Agent Interoperability via A2A (Agent Cards from skills, inbound/outbound task delegation) | Design complete | Low (multi-system only) |

**Where we are now:** All requests flow through skill dispatch. The legacy `elif agent_type ==` chain and intent-classifier fallback have been removed. The `SKILL_DISPATCH_ENABLED` flag has been removed; skill dispatch is the only path. ~26 redundant agent files have been deleted; only `base_agent`, `chat_agent`, `full_research_agent`, `writing_assistant_agent`, and `proposal_generation_agent` remain (plus `agents/__init__.py` updated). Checkpoint/shared_memory loading uses the chat agent from the unified dispatcher; `grpc_service` no longer holds agent instances or `_ensure_agents_loaded()`. Phase 7 compound planner is live; P0 bridging tools are implemented and registered. Phase 8 (Tool Pack redesign, P1 tools, vector store and discovery updates) is complete. Subgraph fragment exposure is done for **12 fragments** (document_retrieval, web_research, visualization, data_formatting, image_search, diagramming, full_document_analysis, gap_analysis, assessment, fact_verification, entity_relationship, knowledge_document); compound steps can use `fragment_name` and PlanEngine invokes them via `invoke_fragment()`. P1 gRPC document tools (append_to_document_tool, get_document_metadata_tool, create_document_tool) are implemented and registered. See [SUBGRAPH_FRAGMENT_REFERENCE.md](./SUBGRAPH_FRAGMENT_REFERENCE.md) for the full fragment and subgraph catalog. Next actionable work is Phase 13/14 (hierarchical selection, inter-agent collaboration).

**Recently completed (Feb 2026):** Expanded fragment registry and P1 gRPC tools. Eight new fragments added to `fragment_registry.py`: image_search, diagramming, full_document_analysis, gap_analysis, assessment, fact_verification, entity_relationship, knowledge_document. `invoke_fragment` adapted to handle dict outputs (JSON-serialized) and to wrap string `research_findings` as `{combined_results: s}` for fact_verification. Three P1 document tools in `document_tools.py`: `append_to_document_tool`, `get_document_metadata_tool`, `create_document_tool`; registered in `tools/__init__.py`, DOCUMENT_TOOLS, and new `document_management` tool pack. Created [SUBGRAPH_FRAGMENT_REFERENCE.md](./SUBGRAPH_FRAGMENT_REFERENCE.md) cataloging all subgraphs and fragment interface details.

**Previously completed (Feb 2026):** Subgraph fragment exposure. `fragment_registry.py` with `FragmentDef`, `FRAGMENT_REGISTRY` (document_retrieval, web_research, visualization, data_formatting), and `invoke_fragment()`. `PlanStep` extended with `fragment_name`; plan engine runs fragment steps via `_run_fragment_step_collect_chunks` and `invoke_fragment()`; skill_llm_selector lists fragments in prompt and parses `fragment_name` in compound steps. Results stored in ContextBridge same as skill steps.

**Previously completed (Feb 2026):** Phase 8 Tool Packs and Mid-Execution Augmentation. `tool_pack_registry.py` redesigned: Pydantic `ToolPack(name, description, tools)` and `TOOL_PACKS` dict with 7 packs; `resolve_pack_tools(pack_names)` for engine use. `PlanStep` extended with `tool_packs: List[str]`; `skill_llm_selector.py` prompt includes available tool packs and step schema with `tool_packs`; `plan_engine.py` injects `step.tool_packs` into step_metadata; `automation_engine.py` merges skill tools with pack tools from metadata. P1 tools: `transform_format_tool`, `merge_texts_tool`, `compare_texts_tool` in `text_transform_tools.py`, registered in `tools/__init__.py`. `tool_vector_store.py` and `tool_discovery.py` updated to new registry (string pack names; vectorize_all_tools builds protos from TOOL_PACKS and docstrings).

**Previously completed (Feb 2026):** Compound Query Planner and P0 bridging tools. Plan engine (`plan_engine.py`), plan models (`plan_models.py`), and context bridge (`context_bridge.py`) are built; `llm_select_skill_or_plan` in `skill_llm_selector.py` returns single-skill or multi-step plans; `grpc_service.py` wires compound plans to `PlanEngine.execute_plan()`. P0 tools: `summarize_text_tool`, `extract_structured_data_tool`, `clipboard_store_tool`, `clipboard_get_tool` implemented in `text_transform_tools.py` and `session_memory_tools.py`, registered in `tools/__init__.py`. Compound flows like "Research X and put results in folder Y" now run end-to-end (research step → document_creator step with `prior_step_*_response`).

**Previously completed (Feb 2026):** `document_creator` automation skill and supporting tooling. New gRPC: `GetFolderTree` (proto + backend handler + client `get_folder_tree()`). New orchestrator tool: `list_folders_tool` (formats folder tree for LLM). File creation tools `create_user_file_tool` and `create_user_folder_tool` now return human-readable strings and are registered in `tools/__init__.py`. Skill definition in `automation_skills.py` with workflow: list_folders → create folder if needed → create_user_file; uses `prior_step_*_response` for compound plans.

**Research Engine Parameterization (Design Phase 5):** Full Research Agent refactored into `research/` package (research_state, research_helpers, research_routing, research_skill_config, research_attachment_nodes, research_fast_path_nodes, research_core_nodes, research_synthesis_nodes); re-export shim at `full_research_agent.py`. `SKILL_CONFIGS` for all 7 research skills; `metadata["skill_name"]` drives config; fast path classification only for default `"research"` skill. Nodes and research_workflow_subgraph gate on skill_config (tier, full_doc_analysis, gap_analysis/round2, web_search, synthesis_style). Five synthesis prompt variants (analytical, verification, security_report, extraction, ingestion) plus comprehensive default. State preservation (skill_config and critical 5) in all node returns.

---

## Routing: LLM-Primary (Current)

Routing no longer uses keyword-based scoring or domain detection.

1. **Hard gates (instant)**  
   `SkillRegistry.filter_eligible(query, editor_context, conversation_context, …)` returns `(eligible_skills, instant_route)`.
   - **Greeting** → `instant_route = "chat"`; caller skips LLM.
   - **Eligibility filter**: drop `internal_only`, `requires_editor` without editor, `editor_types` mismatch when editor active, `requires_image_context` without image.
2. **LLM selection**  
   If `instant_route` is `None` and there are eligible skills, `llm_select_skill(eligible, query, …)` runs. It uses the **fast model** (from settings `FAST_MODEL`). Prompt lists skill **name + description** only. Returns skill name or `"chat"` on failure or low confidence.
3. **Dispatch**  
   `UnifiedDispatcher.dispatch(skill_name, …)` → engine by `skill.engine` (AUTOMATION, CONVERSATIONAL, EDITOR, RESEARCH).

Keywords in skill definitions are **not** used for routing; they remain for reference and possible future use. Descriptions drive LLM selection.

---

## File Layout

```
llm-orchestrator/orchestrator/
  engines/
    __init__.py
    automation_engine.py     # Generic tool-calling engine (~4 nodes); replaces 12 dedicated agents when skill dispatch is on
    conversational_engine.py # Wraps ChatAgent; accepts skill_name
    editor_engine.py         # Wraps WritingAssistantAgent; accepts skill_name
    research_engine.py       # Wraps FullResearchAgent; accepts skill_name
    unified_dispatch.py      # skill_name → engine.process → ChatChunk stream
  skills/
    __init__.py
    skill_schema.py          # Skill, EngineType; ScoredSkill (compatibility)
    skill_registry.py        # SkillRegistry, filter_eligible(), get(), get_for_engine()
    skill_llm_selector.py    # llm_select_skill(eligible, query, …) → skill name
    definitions/
      automation_skills.py   # weather, dictionary, help, email, navigation, rss, entertainment, org_capture, image_*, reference, technical_hyperspace, document_creator
      conversational_skills.py  # chat, org_content, story_analysis
      editor_skills.py      # fiction_editing, outline_editing, character_*, rules_editing, style_editing, series_editing, electronics, general_project, podcast_script, proofreading, article_writing
      research_skills.py    # research, content_analysis, knowledge_builder, security_analysis, site_crawl, website_crawler
```

**Config** (`llm-orchestrator/config/settings.py`): Skill dispatch is the only path (no feature flag). Skill routing uses `FAST_MODEL` for the LLM selection call.

---

## Agent Reduction: What’s Done vs. What’s Left

**Current state (Feb 2026): Completed.** Skill dispatch is the only path; the flag has been removed. ~26 agent files deleted; only base_agent, chat_agent, full_research_agent, writing_assistant_agent, proposal_generation_agent remain. grpc_service uses the unified dispatcher for checkpoint loading; legacy chain and intent-classifier fallback removed. Short-circuits set discovered_skill; /hyperspace removed. All EDITOR skills go through EditorEngine. Email HITL and image_description injection addressed (see Quality Gaps).

### When skill dispatch is ON (historical; dispatch is now always on)

- **Automation Engine** fully replaces these **12** dedicated agent classes (no agent code runs):
  - weather, dictionary, help, email, navigation, rss, entertainment, org_capture, image_generation, image_description, reference, technical_hyperspace  
  So 12 agent files (~7,000+ lines) are redundant when the flag is on; they are still present for the legacy path.

- **Conversational, Editor, Research engines** are thin wrappers:
  - ConversationalEngine → ChatAgent  
  - EditorEngine → WritingAssistantAgent  
  - ResearchEngine → FullResearchAgent  
  So 3 “core” agent classes remain in use.

### When skill dispatch is OFF (obsolete; legacy path removed)

- ~~The legacy `elif agent_type == …` chain…~~ Removed; skill dispatch is the only path.

### Path to full agent reduction

1. **Enable skill dispatch**  
   Set `SKILL_DISPATCH_ENABLED=True` (e.g. in docker-compose or env). Validate that automation skills (weather, help, etc.) and editor/research/conversational flows behave correctly.
2. **Remove legacy dispatch**  
   Delete the legacy `elif agent_type ==` block in `grpc_service.py` and the 12 automation agent files that are fully replaced by AutomationEngine + skill definitions.
3. **Optional later**  
   Fold remaining editor/research/conversational behavior more deeply into engines so that the 3 core agents can be simplified or inlined; that’s a larger refactor.

---

## What to Expect When You Enable `SKILL_DISPATCH_ENABLED` (obsolete: skill dispatch is always on)

Enable via environment (no longer used; flag removed):

```bash
SKILL_DISPATCH_ENABLED=true
```

### Flow when enabled

1. Short-circuits for explicit `request.agent_type` and (if still present) `/help`, `/define`, `/hyperspace` are unchanged.
2. **Skill filter**  
   `registry.filter_eligible(query, editor_context, conversation_context, …)` → `(eligible, instant_route)`.
3. **Skill selection**  
   If `instant_route` (e.g. greeting → chat), use it. Else if `eligible` non-empty, `llm_select_skill(eligible, query, …)` returns skill name (or `"chat"`). Else fallback to `"chat"`.
4. **Unified dispatcher**  
   `dispatch(skill_name, query, metadata, messages)` routes by `skill.engine`.

### What works when the flag is on

| Engine | Skills | Behavior |
|--------|--------|----------|
| **AUTOMATION** | weather, dictionary, help, email, navigation, rss, entertainment, org_capture, image_generation, image_description, reference, technical_hyperspace, document_creator | AutomationEngine runs with skill’s system_prompt and tools. Replaces legacy agents for these. document_creator uses list_folders_tool, create_user_file_tool, create_user_folder_tool for compound flows (e.g. research then save to folder). |
| **EDITOR** | fiction_editing, outline_editing, character_development, rules_editing, style_editing, series_editing, electronics, general_project, podcast_script, proofreading, article_writing | EditorEngine (WritingAssistantAgent). Same subgraphs and routing by `active_editor.frontmatter.type`. |
| **RESEARCH** | research, content_analysis, knowledge_builder, security_analysis, site_crawl, website_crawler | ResearchEngine runs FullResearchAgent with `skill_name` in metadata. |
| **CONVERSATIONAL** | chat, org_content, story_analysis | ConversationalEngine runs ChatAgent with `skill_name` in metadata. |

### Writing assistant

Writing assistant works when skill dispatch is on if the LLM selects an editor skill (e.g. fiction_editing, outline_editing). That depends on editor context (e.g. `active_editor` with type/filename) being sent and the query clearly implying editing. If the frontend sends editor context and the user asks for edits (“rewrite this paragraph”, “add a scene”), the LLM should pick an editor skill and EditorEngine (WritingAssistantAgent) runs as before.

### Recommendation for testing

1. Enable the flag in dev/staging.
2. Test clear intents: doc open + edit request, “what’s the weather”, “research X”.
3. If the writing assistant doesn’t run when expected, confirm editor context is in the request and check logs for “Skill discovery: &lt;skill_name&gt;”.

---

## Quality Gaps: Automation Engine vs Legacy Agents

Automation skills now use **modular gRPC tool wrappers** in `orchestrator/tools/` (weather_tools, email_tools, navigation_tools, rss_tools, org_capture_tools, image_generation_tools). The AutomationEngine resolves tools by name and the LLM + system prompt handle orchestration. The following gaps exist compared to the legacy dedicated agents; document these for future fixes.

| Skill | Gap | Severity |
|-------|-----|----------|
| **weather** | Legacy agent aggregated 12 monthly gRPC calls for year-range historical queries; automation engine makes one call per tool use. LLM may not loop 12 times for “weather in 2024”. | Medium |
| **email** | ~~Legacy agent had HITL… automation engine invokes send_email_tool directly with no safety gate.~~ **Addressed:** send_email_tool and reply_to_email_tool have `confirmed=False` (draft first, then confirm); email skill prompt instructs LLM to use that pattern. | — |
| **email** | Legacy agent composed draft JSON with structured parsing; automation engine relies on LLM to pass correct params (to, subject, body, message_id). | Medium |
| **navigation** | Legacy agent used subgraphs to resolve location names to IDs then compute route; automation engine relies on LLM to call list_locations_tool then compute_route_tool with IDs. | Medium |
| **image_generation** | Legacy agent loaded reference image bytes from get_reference_image_for_object and passed them into generate_image; tool wrapper returns a string only, so reference image bytes are not passed into generate_image in the same turn. | Low |
| **image_description** | ~~Requires engine-level support to inject attached image context.~~ **Addressed:** Automation engine injects image from shared_memory/metadata when skill has `requires_image_context`; attachment_analysis_subgraph uses inline vision LLM (no deleted agent). | — |
| **rss** | Legacy agent used regex-based command parsing; automation engine relies on LLM to extract feed_url, feed_name, scope, etc. from natural language. | Low |

**Fixes to consider in agents/engines**

- **Weather**: If year-range history is needed, either extend get_weather_tool to accept a range and do 12 calls inside the tool, or document “single date/month only” and add a separate “weather history range” tool that aggregates.
- **Email**: Add HITL (interrupt_before or confirmation step) for send_email_tool and reply_to_email_tool in the automation flow, or a dedicated “confirm send” node that requires user approval before calling the tool.
- **Navigation**: No change required if LLM reliably calls list then compute_route; otherwise add a small “resolve location name to ID” helper or document that users should use exact names/IDs from list.
- **Image generation**: If reference images must be used, the backend or a dedicated tool could accept “use reference for object X” and perform get_reference + generate in one RPC; or the engine could run a two-step flow (get_reference_image_tool then generate_image_tool with a backend that accepts a reference_id from the first call).
- **Image description**: Ensure ConversationalEngine or a dedicated image_description path injects `metadata.shared_memory.attached_images` / `image_base64` into the LLM message as vision content when the skill is image_description.
- **RSS**: No change required; LLM extraction is preferred over regex.

---

## Next Steps

### Completed (Phase 9: Skill Dispatch as Only Path)

1. ~~Validate with `SKILL_DISPATCH_ENABLED=true`~~ — Done; skill dispatch was validated, then made the only path.
2. ~~Remove legacy dispatch and flag~~ — Legacy `elif agent_type ==` chain and intent-classifier fallback removed from `grpc_service.py`; `SKILL_DISPATCH_ENABLED` removed from settings and docker-compose.
3. ~~Delete redundant agent files~~ — ~26 agent files deleted; only base, chat, full_research, writing_assistant, proposal_generation remain. Email HITL and image-description injection added; attachment_analysis_subgraph updated to use inline vision LLM.

### Completed (Phase 7: Compound Query Planner)

1. ~~Implement `PlanStep` and `ExecutionPlan` Pydantic models~~ — Done in `orchestrator/engines/plan_models.py`.
2. ~~Build `PlanEngine`~~ — Done in `orchestrator/engines/plan_engine.py`; compound detection via `llm_select_skill_or_plan`; single-intent bypass; 2-4 step decomposition.
3. ~~Build `ContextBridge`~~ — Done in `orchestrator/engines/context_bridge.py`; step_results, shared_memory injection from `depends_on` + `context_keys`, prior step response text.
4. ~~Integrate with UnifiedDispatcher~~ — Done; `grpc_service.py` wires compound plans to `PlanEngine.execute_plan()`; parallel steps via `asyncio.gather()`; aggregated response.
5. ~~Compound detection~~ — Done; `llm_select_skill_or_plan` in `skill_llm_selector.py` returns `ExecutionPlan` (single or compound).
6. ~~Fail-safe~~ — Done; fallback to single-skill dispatch on parse failure or low confidence.

### Completed (Phase 8: Tool Packs + Augmentation)

1. ~~Redefine `ToolPack` model and `TOOL_PACKS` registry~~ — Done in `orchestrator/tools/tool_pack_registry.py` (Pydantic `ToolPack(name, description, tools)`, 7 packs, `resolve_pack_tools()`).
2. ~~Extend `PlanStep` to include optional `tool_packs: List[str]`~~ — Done in `plan_models.py`.
3. ~~Modify engine tool resolution to merge skill tools + requested tool pack tools~~ — Done in `automation_engine.py` _load_skill_node; plan_engine injects `step.tool_packs` into step_metadata; skill_llm_selector prompt includes tool packs and step schema.
4. ~~Build remaining cross-cutting utility tools~~ — Done: `transform_format_tool`, `merge_texts_tool`, `compare_texts_tool` in `text_transform_tools.py`; P0 tools already done.
5. ~~Expose subgraph fragments as plan step targets~~ — Done for 4 fragments. `fragment_registry.py` defines `FragmentDef`, `FRAGMENT_REGISTRY` (document_retrieval, web_research, visualization, data_formatting), and `invoke_fragment()`. `PlanStep` has optional `fragment_name`; plan engine calls `invoke_fragment()` for fragment steps; LLM selector lists fragments and parses `fragment_name`.

### Longer-Term (Phase 13: Hierarchical Selection)

1. Implement domain router as first stage (deterministic domain detection + fast LLM for ambiguous cases).
2. Skill selector runs within domain (5-10 skills per domain instead of 30+).
3. Only needed when skill count exceeds ~50 — current 30 works fine with single-stage LLM selection.

### Longer-Term (Phase 14: Inter-Agent Collaboration Sessions)

1. Define `AgentMessage` and `CollaborationSession` Pydantic models in `orchestrator/engines/collaboration_session.py`.
2. Build `CollaborationEngine` as a StateGraph with supervisor → worker dispatch → convergence check loop:
   - Supervisor node selects next speaker (LLM-based or round-robin).
   - Worker nodes invoke skills via `UnifiedDispatcher.dispatch()`, results enter shared session thread.
   - Convergence node checks criteria: approval received, max turns, confidence threshold.
3. Add `CollaborationConfig` to `PlanStep` as a new step type (alongside single-skill and fragment steps).
4. Planner decides when to use collaboration vs. sequential: simple handoffs stay sequential; tasks needing critique/revision use collaboration sessions.
5. Implement convergence strategies: approval gate (reviewer skill approves), max-turn cap (fail-safe), confidence threshold.
6. Use cases to validate: research + fact-checker, draft + tone reviewer, generate + proofread.

### Longer-Term (Phase 15: Structured Inter-Agent Protocols)

1. Define `MessageType` enum (proposal, critique, revision, approval, rejection, request_info, info_response, delegation).
2. Add protocol rule enforcement to supervisor node (state machine: proposal → critique → revision → approval).
3. Build `CollaborationTrace` model for audit trails (session_id, participants, messages, outcome, duration, model_calls).
4. Store traces in checkpoint state for persistence and debugging.
5. Optional: expose trace to user as transparency ("Here's how the agents collaborated").

### Future (Phase 16: A2A External Interoperability)

1. Build `AgentCardGenerator` that auto-generates A2A Agent Cards from skill registry (name, description, capabilities, endpoint, auth).
2. Add A2A HTTP endpoint (`/a2a/tasks/send`, `/a2a/tasks/{id}`) that maps inbound tasks to `UnifiedDispatcher.dispatch()`.
3. Build outbound client for delegating tasks to external A2A agents.
4. Add `delegate_to_external_agent_tool` so skills and the planner can invoke external agents.
5. Only needed for multi-system deployments; internal collaboration (Phases 14-15) provides 90%+ of value for single-system use.

---

## New Tools Needed (Prioritized)

### Completed: Document / Folder Tools (Feb 2026)

| Tool | Type | Location | Status |
|------|------|----------|--------|
| `list_folders_tool` | External (gRPC) | `orchestrator/tools/file_creation_tools.py` | Done |
| `create_user_file_tool` | External (gRPC) | `orchestrator/tools/file_creation_tools.py` | Done (registered; returns str) |
| `create_user_folder_tool` | External (gRPC) | `orchestrator/tools/file_creation_tools.py` | Done (registered; returns str) |

Backend: `GetFolderTree` RPC in `protos/tool_service.proto`; handler in `backend/services/grpc_tool_service.py`; client method `get_folder_tree()` in `backend_tool_client.py`. Used by `document_creator` skill for compound plans (e.g. research → put results in folder).

### Completed: P0 Bridging Tools (Planner Context Bridge)

These tools support the context bridge between plan steps. All implemented and registered in `orchestrator/tools/__init__.py`.

| Tool | Type | Location | Status |
|------|------|----------|--------|
| `summarize_text_tool` | Local (LLM) | `orchestrator/tools/text_transform_tools.py` | Done |
| `extract_structured_data_tool` | Local (LLM) | `orchestrator/tools/text_transform_tools.py` | Done |
| `clipboard_store_tool` | Local (in-memory) | `orchestrator/tools/session_memory_tools.py` | Done |
| `clipboard_get_tool` | Local (in-memory) | `orchestrator/tools/session_memory_tools.py` | Done |

### P1: Required for Tool Packs (Phase 8)

| Tool | Type | Location | Status |
|------|------|----------|--------|
| `transform_format_tool` | Local (string) | `orchestrator/tools/text_transform_tools.py` | Done |
| `merge_texts_tool` | Local (string) | `orchestrator/tools/text_transform_tools.py` | Done |
| `compare_texts_tool` | Local (string/LLM) | `orchestrator/tools/text_transform_tools.py` | Done |
| `create_document_tool` | External (gRPC) | `orchestrator/tools/document_tools.py` | Done |
| `append_to_document_tool` | External (gRPC) | `orchestrator/tools/document_tools.py` | Done |
| `get_document_metadata_tool` | External (gRPC) | `orchestrator/tools/document_tools.py` | Done |

### P2: Valuable for Compound Skills

| Tool | Type | Location | Status |
|------|------|----------|--------|
| `store_fact_tool` | External (gRPC) | `orchestrator/tools/memory_tools.py` | Not started |
| `recall_facts_tool` | External (gRPC) | `orchestrator/tools/memory_tools.py` | Not started |
| `notify_user_tool` | External (gRPC) | `orchestrator/tools/notification_tools.py` | Not started |
| `schedule_reminder_tool` | External (gRPC) | `orchestrator/tools/notification_tools.py` | Not started |
| `export_to_format_tool` | External (gRPC) | `orchestrator/tools/export_tools.py` | Not started |
| `list_documents_by_type_tool` | External (gRPC) | `orchestrator/tools/document_tools.py` | Not started |

---

## Subgraph Fragment Exposure Status

Subgraphs invocable as standalone plan step targets (fragment exposure). Full catalog: [SUBGRAPH_FRAGMENT_REFERENCE.md](./SUBGRAPH_FRAGMENT_REFERENCE.md).

| Subgraph | Build Function | Fragment-Ready | Notes |
|----------|---------------|----------------|-------|
| `document_retrieval` | `build_intelligent_document_retrieval_subgraph()` | Yes | In FRAGMENT_REGISTRY |
| `web_research` | `build_web_research_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `visualization` | `build_visualization_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `data_formatting` | `build_data_formatting_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `image_search` | `build_image_search_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `diagramming` | `build_diagramming_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `full_document_analysis` | `build_full_document_analysis_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `gap_analysis` | `build_gap_analysis_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY; pipeline (needs results) |
| `assessment` | `build_assessment_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY; pipeline (needs results) |
| `fact_verification` | `build_fact_verification_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY; pipeline (needs research_findings) |
| `entity_relationship` | `build_entity_relationship_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY |
| `knowledge_document` | `build_knowledge_document_subgraph(checkpointer)` | Yes | In FRAGMENT_REGISTRY; pipeline |
| `intelligent_document_retrieval` | (same as document_retrieval) | Yes | Registered as document_retrieval |
| `proofreading` | `build_proofreading_subgraph()` | No | Needs llm_factory, editor content; internal to EditorEngine |
| `collection_search` | `execute_collection_search()` | Partial | Callable function, not subgraph |
| `attachment_analysis` | `build_attachment_analysis_subgraph()` | No | Requires binary attachments; not in registry |

**Standard fragment interface** — implemented in `orchestrator/engines/fragment_registry.py`: `invoke_fragment(fragment_name, query, metadata, messages, prior_context)` builds subgraph from registry, maps state, runs `ainvoke`, returns `{"response": text, "agent_type": fragment_name}` for ContextBridge. Dict outputs are JSON-serialized.
