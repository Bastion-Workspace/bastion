# Tool I/O Contract Migration Tracker

**Created:** February 14, 2026
**Rule:** `.cursor/rules/tool-io-contracts.mdc`
**Registry:** `llm-orchestrator/orchestrator/utils/action_io_registry.py` (created)
**Shared types:** `llm-orchestrator/orchestrator/utils/tool_type_models.py` (created)

**Recent migration (Feb 2026):** Prerequisites P1–P4 and four tools completed: `calculate_expression_tool`, `search_documents_tool`, `search_web_tool`, `get_document_content_tool`. All use unified dict return with `formatted`; `search_documents_structured` and `search_web_structured` removed. Automation engine and all known direct callers updated.

**Post-enrichment (Feb 2026):** `list_available_formulas_tool` — full output model (formulas, count, success, error, formatted) and `register_action`. Data workspace outputs enriched: `list_data_workspaces` adds `count`; `get_workspace_schema` adds `workspace_id`, `workspace_name`; `query_data_workspace` adds `rows`, `columns`. RSS `refresh_rss_feed` adds `new_items`, `count`. Navigation tools (55–60) and image tools (`analyze_image_query`, `detect_faces_in_image`, `identify_faces_in_image`) confirmed: all return dict with `formatted` and are registered (navigation uses `NavigationOutputs`; image tools have granular output models). Duplicate gRPC methods (StartTask, GetTaskStatus, ApprovePermission, GetPendingPermissions, HealthCheck) removed from `grpc_service.py`.

---

## Strategy

### In-Place Migration (No Variants)

Every tool is modified in-place to return a **unified dict** with typed fields + a `formatted` field. No `_structured` variants. The existing function name stays the same.

### Prerequisites (do these FIRST)

| # | Task | Status | Notes |
|---|------|--------|-------|
| P1 | Create `action_io_registry.py` | `[x]` | Registry + coercion logic + `register_action()` + `is_type_compatible()` |
| P2 | Create `tool_type_models.py` | `[x]` | Shared types: `DocumentRef`, `WebResult`, `FileRef` (TodoItem, EmailRef, NotificationResult can be added as needed) |
| P3 | Update `automation_engine.py` ToolMessage line | `[x]` | **Line ~381**: read `result["formatted"]` for dict returns |
| P4 | Update `automation_engine.py` StructuredTool `_run` wrapper | `[x]` | **Line ~332**: read `out["formatted"]` for dict returns in `_run` closure |
| P5 | Update `ContextBridge` for structured data passthrough | `[x]` | Done. `context_bridge.py` stores `typed_outputs`; `build_context_for_step()` exposes `step_{dep_id}_typed_outputs`. Plan engine passes `typed_outputs` in `store_result()`. |
| P6 | Enrich `Skill` schema with optional I/O contract refs | `[x]` | Done. `skill_schema.py`: added `tool_io_map: Dict[str, str]`. Backfilled on research, document_creator, reference, fiction_editing. |
| P7 | Enrich `ToolPack` with aggregate I/O | `[x]` | Done. `tool_pack_registry.py`: `ToolPack.get_aggregate_outputs()` derives output fields from Action I/O Registry. |
| P8 | Build Agent Factory pipeline executor | `[x]` | Done. `orchestrator/engines/pipeline_executor.py`: resolves `{step_name.field}`, type coercion, executes tool steps; `execute_pipeline()` runs sequence. |

### Per-Tool Migration Steps

For each tool:
1. Add Pydantic I/O models (inputs, params, outputs with `formatted: str`)
2. Modify tool to return dict matching output model (move string formatting to `formatted`)
3. Register with `register_action()` at module level
4. Update any direct callers that expected `str`
5. Mark complete in this tracker

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `[ ]` | Not started — returns `str`, no models, no registry |
| `[d]` | Already returns `dict` — needs Pydantic models + registry + `formatted` field |
| `[r]` | Registered but shallow — returns dict with `formatted`, registered; output model needs granular fields for Workflow Composer wiring |
| `[~]` | In progress |
| `[x]` | Complete — unified return, granular typed outputs, Pydantic models, registered |
| `[-]` | Skipped — internal-only, not needed for Agent Factory |

---

## Summary

| Category | Total | Complete [x] | Registered [r] | Remaining |
|----------|-------|--------------|----------------|-----------|
| File & Document Ops | 17 | 2 | 10 | 5 |
| Search & Discovery | 6 | 0 | 5 | 1 |
| Task Management (Org) | 5 | 0 | 5 | 0 |
| Email | 6 | 0 | 6 | 0 |
| Text & Content | 5 | 0 | 5 | 0 |
| Data Workspace | 3 | 0 | 3 | 0 |
| Math & Formulas | 4 | 2 | 2 | 0 |
| Web & Crawling | 3 | 2 | 0 | 1 |
| Image & Vision | 5 | 0 | 5 | 0 |
| Navigation | 6 | 0 | 6 | 0 |
| RSS | 3 | 0 | 3 | 0 |
| Weather | 1 | 1 | 0 | 0 |
| Session Memory | 2 | 0 | 2 | 0 |
| System Modeling | 3 | 0 | 3 | 0 |
| Visualization | 1 | 0 | 1 | 0 |
| Lessons | 3 | 0 | 3 | 0 |
| Analysis (Internal) | 5 | 0 | 2 | 3 |
| Project Structure | 3 | 0 | 0 | 3 |
| **Total** | **81** | **7** | **58** | **16** |

**Reconciled (Feb 2026):** Tier 1 [x] = 7 tools (granular outputs + formatted + registered; includes `list_available_formulas`). Tier 2 [r] = 58 tools (registered, return dict with formatted; many output models enriched for field-level wiring — data_workspace count/rows/columns/workspace_id, rss new_items/count; navigation and image tools use unified dict + NavigationOutputs / granular image outputs). Tier 3 = 16 tools (internal/skipped or not yet registered). P1–P8 and Phase 4 (GET /api/agent-factory/actions) in place. Duplicate gRPC methods removed from `grpc_service.py`.

---

## Priority 1: Agent Factory Critical + Infrastructure

### Prerequisites

| # | Task | File | Status | Notes |
|---|------|------|--------|-------|
| P1 | Action I/O Registry | `orchestrator/utils/action_io_registry.py` | `[x]` | Core registry with `register_action()`, coercion logic, field introspection |
| P2 | Shared Type Models | `orchestrator/utils/tool_type_models.py` | `[x]` | `DocumentRef`, `WebResult`, `FileRef` |
| P3 | Automation Engine update | `orchestrator/engines/automation_engine.py` | `[x]` | Read `result["formatted"]` from dict returns in ToolMessage creation |
| P4 | StructuredTool compat | `orchestrator/engines/automation_engine.py` | `[x]` | `_run` wrapper reads `out["formatted"]` for dict returns |

### File & Document Operations

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 1 | `search_documents_tool` | document_tools.py | `dict` | `[x]` | Migrated. Unified return: `documents`, `count`, `query_used`, `formatted`. `search_documents_structured` removed; callers use `search_documents_tool` and `result["documents"]`/`result["count"]`. |
| 2 | ~~`search_documents_structured`~~ | document_tools.py | — | `[-]` | Removed; merged into `search_documents_tool`. |
| 3 | `get_document_content_tool` | document_tools.py | `dict` | `[x]` | Migrated. Outputs: `content`, `document_id`, `word_count`, `formatted`. All direct callers updated to use `result["content"]` (or backward-compat pattern). |
| 4 | `search_within_document_tool` | document_tools.py | `dict` | `[d]` | Add `formatted` + models + registry. Outputs: `matches: list[record]`, `count: number`. Not yet registered. |
| 5 | `get_document_metadata_tool` | document_tools.py | `dict` | `[r]` | Registered. Returns dict with document_id, title, filename, content_type, metadata, formatted. Enrich output model for field-level wiring. |
| 6 | `create_document_tool` | document_tools.py | `dict` | `[r]` | Registered. Returns dict with success, document_id, formatted. Enrich output model for field-level wiring. |
| 7 | `append_to_document_tool` | document_tools.py | `dict` | `[r]` | Registered. Returns dict with success, content_length, formatted. Enrich output model for field-level wiring. |
| 8 | `search_by_tags_tool` | document_tools.py | `dict` | `[r]` | Registered. Returns dict with documents, count, formatted. Enrich output model for field-level wiring. |
| 9 | `find_documents_by_tags_tool` | document_tools.py | `list[dict]` | `[d]` | Returns list directly — wrap in `{results: list, count: int, formatted: str}`; not registered. |
| 10 | `list_folders_tool` | file_creation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (folders, count) for field-level wiring. |
| 11 | `create_user_file_tool` | file_creation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (document_id, title, success) for field-level wiring. |
| 12 | `create_user_folder_tool` | file_creation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (folder_id, name, success) for field-level wiring. |
| 13 | `update_document_content_tool` | document_editing_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 14 | `update_document_metadata_tool` | document_editing_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 15 | `propose_document_edit_tool` | document_editing_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 16 | `apply_operations_directly_tool` | document_editing_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 17 | `apply_document_edit_proposal_tool` | document_editing_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |

### Task Management — Org Mode

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 18 | `add_org_inbox_item_tool` | org_capture_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (success, todo_id, file_path) for field-level wiring. |
| 19 | `list_org_todos_tool` | org_content_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (todos, count) for field-level wiring. |
| 20 | `parse_org_structure_tool` | org_content_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (structure, heading_count) for field-level wiring. |
| 21 | `search_org_headings_tool` | org_content_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (headings, count) for field-level wiring. |
| 22 | `get_org_statistics_tool` | org_content_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (statistics) for field-level wiring. |

### Search & Discovery

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 23 | `search_segments_across_documents_tool` | segment_search_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 24 | `expand_query_tool` | enhancement_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (variations, count) for field-level wiring. |
| 25 | `search_conversation_cache_tool` | enhancement_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 26 | `analyze_information_needs_tool` | information_analysis_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 27 | `generate_project_aware_queries_tool` | information_analysis_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |

### Data Workspace

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 28 | `list_data_workspaces_tool` | data_workspace_tools.py | `dict` | `[r]` | Registered. Output model includes workspaces, count, total_count (enriched). |
| 29 | `get_workspace_schema_tool` | data_workspace_tools.py | `dict` | `[r]` | Registered. Output model includes tables, workspace_id, workspace_name (enriched). |
| 30 | `query_data_workspace_tool` | data_workspace_tools.py | `dict` | `[r]` | Registered. Output model includes results, rows, columns, row_count (enriched). |

---

## Priority 2: Agent Factory Supporting

### Email

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 31 | `get_emails_tool` | email_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted, content (shallow). Enrich output model (emails, count) for field-level wiring. |
| 32 | `search_emails_tool` | email_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted, content (shallow). Enrich output model (emails, count, query_used) for field-level wiring. |
| 33 | `get_email_thread_tool` | email_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted, content (shallow). Enrich output model (messages, subject, participants) for field-level wiring. |
| 34 | `get_email_statistics_tool` | email_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted, content (shallow). Enrich output model (statistics) for field-level wiring. |
| 35 | `send_email_tool` | email_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted, content (shallow). Enrich output model (success, message_id) for field-level wiring. |
| 36 | `reply_to_email_tool` | email_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted, content (shallow). Enrich output model (success, message_id) for field-level wiring. |

### Web & Crawling

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 37 | `search_web_tool` | web_tools.py | `dict` | `[x]` | Migrated. Unified return: `results`, `count`, `query_used`, `formatted`. `search_web_structured` removed; callers use `result["results"]`. |
| 38 | ~~`search_web_structured`~~ | web_tools.py | — | `[-]` | Removed; merged into `search_web_tool`. |
| 39 | `crawl_web_content_tool` | web_tools.py | `dict` | `[x]` | Migrated. Unified return: `results`, `count`, `formatted`. Registered. |

### Text & Content Processing

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 40 | `summarize_text_tool` | text_transform_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (summary, original_length, summary_length) for field-level wiring. |
| 41 | `extract_structured_data_tool` | text_transform_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (extracted, source_text_length) for field-level wiring. |
| 42 | `transform_format_tool` | text_transform_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (content, source_format, target_format) for field-level wiring. |
| 43 | `merge_texts_tool` | text_transform_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (merged, source_count) for field-level wiring. |
| 44 | `compare_texts_tool` | text_transform_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (comparison, similarity_score, differences) for field-level wiring. |

### Math & Formulas

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 45 | `calculate_expression_tool` | math_tools.py | `dict` | `[x]` | Migrated. Outputs: `result`, `expression`, `success`, `error`, `steps`, `variables_used`, `formatted`. Registered. |
| 46 | `list_available_formulas_tool` | math_tools.py | `dict` | `[x]` | Migrated. Outputs: `formulas`, `count`, `success`, `error`, `formatted`. Registered. |
| 47 | `evaluate_formula_tool` | math_formulas.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (result, formula_name, variables_used) for field-level wiring. |
| 48 | `convert_units_tool` | unit_conversions.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (result, from_unit, to_unit) for field-level wiring. |

---

## Priority 3: Nice to Have

### Image & Vision

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 49 | `search_images_tool` | image_search_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 50 | `analyze_image_query` | image_query_analyzer.py | `dict` | `[x]` | Migrated. Outputs: `query`, `series`, `author`, `date`, `image_type`, `formatted`. Registered. |
| 51 | `generate_image_tool` | image_generation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (image_url, prompt_used, success) for field-level wiring. |
| 52 | `get_reference_image_tool` | image_generation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (image_data, found) for field-level wiring. |
| 53 | `detect_faces_in_image` | face_analysis_tools.py | `dict` | `[x]` | Migrated. Outputs: `success`, `faces`, `face_count`, `image_width`, `image_height`, `error`, `formatted`. Registered. |
| 54 | `identify_faces_in_image` | face_analysis_tools.py | `dict` | `[x]` | Migrated. Outputs: `success`, `face_count`, `identified_count`, `identified_faces`, `error`, `formatted`. Registered. |

### Navigation

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 55 | `create_location_tool` | navigation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted; NavigationOutputs (location_id, locations, count, routes, total, route_id, distance_meters, duration_seconds). |
| 56 | `list_locations_tool` | navigation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted; NavigationOutputs. |
| 57 | `delete_location_tool` | navigation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted; NavigationOutputs. |
| 58 | `compute_route_tool` | navigation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted; NavigationOutputs. |
| 59 | `save_route_tool` | navigation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted; NavigationOutputs. |
| 60 | `list_saved_routes_tool` | navigation_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted; NavigationOutputs. |

### RSS

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 61 | `add_rss_feed_tool` | rss_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (feed_id, title, success) for field-level wiring. |
| 62 | `list_rss_feeds_tool` | rss_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model (feeds, count) for field-level wiring. |
| 63 | `refresh_rss_feed_tool` | rss_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Output model includes new_items, count (enriched). |

### Weather

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 64 | `get_weather_tool` | weather_tools.py | `dict` | `[x]` | Migrated. Returns dict with formatted, data. Registered. Enrich output model (temperature, conditions, location, forecast) for field-level wiring optional. |

### Session Memory

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 65 | `clipboard_store_tool` | session_memory_tools.py | `dict` | `[r]` | Registered. Returns dict with success, key, formatted. Enrich output model for field-level wiring. |
| 66 | `clipboard_get_tool` | session_memory_tools.py | `dict` | `[r]` | Registered. Returns dict with value, key, found, formatted. Enrich output model for field-level wiring. |

### Visualization

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 67 | `create_chart_tool` | visualization_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |

### System Modeling

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 68 | `design_system_component_tool` | system_modeling_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 69 | `simulate_system_failure_tool` | system_modeling_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 70 | `get_system_topology_tool` | system_modeling_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |

### Lessons

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 71 | `resolve_lesson_images` | lesson_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 72 | `search_lessons` | lesson_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |
| 73 | `generate_lesson` | lesson_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Enrich output model for field-level wiring. |

---

## Priority 4: Internal / Evaluate for Skip

| # | Tool | File | Currently | Status | Migration notes |
|---|------|------|-----------|--------|-----------------|
| 74 | `analyze_tool_needs_for_research` | dynamic_tool_analyzer.py | `dict` | `[d]` | Internal routing tool — not registered; evaluate if Agent Factory needs it |
| 75 | `analyze_text_metrics` | file_analysis_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Analytics tool — useful for monitor agents. |
| 76 | `analyze_document_metrics` | file_analysis_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Analytics tool. |
| 77 | `analyze_active_editor_metrics` | file_analysis_tools.py | `dict` | `[r]` | Registered. Returns dict with formatted. Editor-specific — may skip for Agent Factory. |
| 78 | `plan_project_structure` | project_structure_tools.py | `dict` | `[d]` | Internal to writing agents — not registered |
| 79 | `execute_project_structure_plan` | project_structure_tools.py | `dict` | `[d]` | Internal to writing agents — not registered |
| 80 | `load_referenced_context` | project_structure_tools.py | `dict` | `[d]` | Internal context loader — not registered |
| 81 | `load_file_by_path` | reference_file_loader.py | `dict` | `[d]` | Internal file loader — not registered |
| 82 | `load_referenced_files` | reference_file_loader.py | `dict` | `[d]` | Internal batch loader — not registered |

---

## Caller Migration Tracker

Tools are consumed in two ways. Both must be updated.

### Automation Engine (one-time, applies to ALL tools)

| Task | File | Status | Notes |
|------|------|--------|-------|
| Fix `_run` wrapper `str()` | `automation_engine.py` ~line 332 | `[x]` | Reads `out["formatted"]` for dict returns |
| Fix ToolMessage creation `str()` | `automation_engine.py` ~line 381 | `[x]` | Reads `result["formatted"]` for dict returns |
| Fix ContextBridge store_result | `context_bridge.py` ~line 29 | `[x]` | P5: Preserve `typed_outputs` from tool return dicts. Done. |
| Fix plan_engine result passing | `plan_engine.py` ~line 228 | `[x]` | P5: Pass typed tool outputs into `store_result()`. Done. |

### Direct Callers (per-tool, tracked as found)

These agents call tools directly and may expect `str` returns. Update each to read from the dict:

| Caller File | Tools Called | Status | Notes |
|-------------|-------------|--------|-------|
| `research/research_fast_path_nodes.py` | `search_documents_tool`, `get_document_content_tool` | `[x]` | Uses `result["documents"]`, `result["content"]` |
| `subgraphs/intelligent_document_retrieval_subgraph.py` | `search_documents_tool`, `get_document_content_tool` | `[x]` | Uses `result["documents"]`, `result["content"]` |
| `subgraphs/factual_query_subgraph.py` | `search_documents_tool`, `search_web_tool`, `get_document_content_tool` | `[x]` | Uses `documents`/`results`/`content` from dicts |
| `subgraphs/research_workflow_subgraph.py` | `search_web_tool` | `[x]` | Uses `raw.get("results", [])` |
| `subgraphs/web_research_subgraph.py` | `search_web_tool` | `[x]` | Uses `raw.get("results", [])` |
| `tools/segment_search_tools.py` | `search_documents_tool`, `get_document_content_tool` | `[x]` | Uses `result["documents"]`, `result["content"]` |
| `tools/lesson_tools.py` | `search_documents_tool`, `get_document_content_tool` | `[x]` | Uses `result["documents"]`, `result["content"]` |
| `tools/reference_file_loader.py` | `get_document_content_tool` | `[x]` | Uses `result["content"]` |
| `tools/document_tools.py` | `get_document_content_tool` (search_within_document) | `[x]` | Uses `result["content"]` |
| `tools/project_content_tools.py` | `get_document_content_tool` | `[x]` | Multiple call sites updated to `result["content"]` |
| `agents/electronics_agent.py` | `search_documents_tool`, `search_web_tool`, `get_document_content_tool` | `[x]` | Uses `results`/`documents`/`content` from dicts |
| `agents/proposal_generation_agent.py` | `get_document_content_tool` | `[x]` | Uses `_r.get("content", _r)` pattern |
| `agents/electronics_nodes/*.py` | `get_document_content_tool` | `[x]` | Updated |
| `agents/general_project_nodes/*.py` | `get_document_content_tool` | `[x]` | Updated |
| `subgraphs/entity_relationship_subgraph.py` | `get_document_content_tool` | `[x]` | Extracts `content` from gathered results |
| `subgraphs/full_document_analysis_subgraph.py` | `get_document_content_tool` | `[x]` | Updated |
| `utils/document_batch_editor.py` | `get_document_content_tool` | `[x]` | Updated |
| `engines/plan_engine.py` | Various tools via automation engine | `[x]` | Covered by automation engine update |

**Discovery method:** When migrating a tool, grep for its function name across the codebase to find all callers. Update each.

---

## Consolidation Plan (Post-Migration)

After all tools return unified dicts:

1. **Remove `search_documents_structured`** — merge into `search_documents_tool`
2. **Remove `search_web_structured`** — merge into `search_web_tool`
3. **Remove any remaining `_structured` variants** — the base tool IS the structured version now
4. **Delete dual-mode documentation** — update the rule to remove migration guidance
5. **Verify all skills work** — run through each skill's tool list and confirm compatibility
6. **Expose I/O registry via API** — `GET /api/agent-factory/actions` returns all contracts for the Workflow Composer — **Done (Phase 4)**.
7. **Backfill `tool_io_map` on all skill definitions** — link each skill's tools to their registry entries (P6) — **Done for skills with migrated tools**.
8. **Add `get_aggregate_outputs()` to ToolPack** — so Workflow Composer can show pack outputs (P7) — **Done**.
9. **Build pipeline executor** — typed data flow between Agent Factory playbook steps (P8) — **Done**.

### Execution Order

```
Phase 1: Foundation
  P1 action_io_registry.py
  P2 tool_type_models.py
  P3 + P4 automation_engine.py fixes (two str() conversions)

Phase 2: Tool Migration (P1 tools, then P2, P3, P4)
  Migrate tools in priority order
  Update direct callers as each tool changes

Phase 3: Infrastructure Enrichment
  P5 ContextBridge structured data
  P6 Skill schema I/O awareness
  P7 ToolPack aggregate I/O

Phase 4: Agent Factory Build
  P8 Pipeline executor (depends on P1 registry)
  Workflow Composer UI (depends on P1 registry + P7 ToolPack)
  Playbook validation (depends on P1 coercion + P6 Skill I/O)

Phase 5: Consolidation
  Remove _structured variants
  Backfill tool_io_map on skills
  Expose registry via API
```

---

## Infrastructure Gaps (discovered via execution path analysis)

These gaps were identified by tracing every execution path through the codebase — automation engine, plan engine, direct calls, skill definitions, and tool pack resolution. Each must be addressed for the typed I/O system to work end-to-end.

### Gap 1: Dual `str()` Conversion in Automation Engine

**Covered by:** P3 + P4 prerequisites above.

There are TWO places in `automation_engine.py` that stringify tool results:
1. **Line ~332** — The `_run` closure inside `_wrap_async_tool()` does `return str(out)`. This is what `StructuredTool` returns to LangChain.
2. **Line ~381** — The tool execution loop does `result_str = str(result)` before creating `ToolMessage`.

Both must check for dict returns and read `result["formatted"]`. Fix P4 first (the wrapper), then P3 (the ToolMessage creation).

### Gap 2: Context Bridge Structured Data

**Covered by:** P5 prerequisite above.

**Problem:** `context_bridge.py` captures only `response_text` (a string extracted from ChatChunk content) when plan engine steps complete. The `structured_data` field exists but is never populated. When plan step A produces typed output (e.g., `{documents: [...], count: 3}`), step B only receives the formatted text summary — the typed fields are lost at the ChatChunk serialization boundary.

**Impact on current skills:** Low — existing skills don't wire typed data between plan steps. They receive prior step context as text in `shared_memory`.

**Impact on Agent Factory:** High — playbook steps that wire `{step_a.documents}` into step B's input need typed data, not text.

**Fix:**
```python
# context_bridge.py — store_result should preserve structured tool outputs
def store_result(self, step_id: int, result: Dict[str, Any]) -> None:
    self.step_results[step_id] = {
        "response_text": _extract_response_text(result),
        "structured_data": result.get("structured_data", {}),
        # NEW: preserve full typed outputs from tool returns
        "typed_outputs": result.get("typed_outputs", {}),
        "skill_name": result.get("agent_type", ""),
    }
```

The plan engine must pass tool return dicts (minus `formatted`) as `typed_outputs` when storing results. Dependent steps can then access `step_{id}_typed_outputs` in shared_memory alongside the existing `prior_step_{id}_response` text.

**Note:** The Agent Factory pipeline executor (P8) bypasses ContextBridge entirely — it passes typed dicts directly between steps without going through ChatChunk streaming. But fixing ContextBridge benefits compound plans that mix skills with tool-returning steps.

### Gap 3: Skill Schema I/O Awareness

**Covered by:** P6 prerequisite above.

**Problem:** `skill_schema.py` defines `Skill.tools` as `List[str]` — just function names. The skill has no knowledge of what its tools accept or produce. This means:
- No design-time validation of tool chains within a skill
- The Workflow Composer can't show what a skill's tools collectively output
- Adding an incompatible tool to a skill produces no warning

**Impact on current skills:** Low — the LLM figures out tool usage from docstrings at runtime. Works fine for LLM-driven execution.

**Impact on Agent Factory:** Medium — when users compose playbooks that reference skills, they need to know what data the skill's tools produce.

**Fix (additive, non-breaking):**
```python
# skill_schema.py — add optional I/O contract references
class Skill(BaseModel):
    # ... existing fields ...
    tools: List[str] = Field(default_factory=list)

    # NEW: optional mapping of tool names to I/O registry action names
    # Allows design-time validation and Workflow Composer introspection
    tool_io_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional mapping: tool function name → action_io_registry name. "
                    "When present, enables design-time type validation of tool chains."
    )
```

This is purely additive. Existing skills with empty `tool_io_map` work exactly as before. As tools are migrated, their skill definitions can optionally add the mapping.

### Gap 4: ToolPack I/O Awareness

**Covered by:** P7 prerequisite above.

**Problem:** `tool_pack_registry.py` defines `ToolPack` as `{name, description, tools: List[str]}`. When the plan engine attaches tool packs to steps, there's no way to know what outputs those tools collectively provide for downstream wiring.

**Impact on current system:** None — tool packs just augment a skill's tool list.

**Impact on Agent Factory:** Medium — the Workflow Composer needs to show "if you attach this tool pack, these output types become available."

**Fix (additive, non-breaking):**
```python
# tool_pack_registry.py — add I/O awareness
class ToolPack(BaseModel):
    name: str
    description: str
    tools: List[str]

    def get_aggregate_outputs(self) -> Dict[str, str]:
        """
        Derive combined output fields from all tools in this pack
        by looking up their contracts in the Action I/O Registry.
        Returns {field_name: field_type} for all outputs across all tools.
        """
        from orchestrator.utils.action_io_registry import get_action
        outputs = {}
        for tool_name in self.tools:
            contract = get_action(tool_name)
            if contract:
                for field in contract.get_output_fields():
                    outputs[f"{tool_name}.{field['name']}"] = field["type"]
        return outputs
```

### Gap 5: Pipeline Executor for Agent Factory

**Covered by:** P8 prerequisite above.

**Problem:** No existing execution path passes typed data between steps. The automation engine stringifies everything for LLM consumption. The plan engine passes text between steps via ContextBridge. Agent Factory playbooks need a third execution path that:

1. Resolves step inputs from the `inputs` map (e.g., `{step_a.documents}`)
2. Looks up the upstream step's typed output by field name
3. Applies type coercion using `is_type_compatible()` from the I/O registry
4. Passes the typed value directly to the downstream tool function
5. Never serializes to string in between

**Impact on current system:** None — this is new infrastructure for Agent Factory only.

**Impact on Agent Factory:** Critical — the entire deterministic pipeline and hybrid workflow patterns depend on this.

**Already designed in:**
- `AGENT_FACTORY_TECHNICAL_GUIDE.md` §1 — `DeterministicPipelineExecutor` class
- `AGENT_FACTORY_TECHNICAL_GUIDE.md` §5 — Playbook step `inputs` map specification
- `AGENT_FACTORY_TECHNICAL_GUIDE.md` §11 — Listed as `pipeline_executor.py` in new files

**Depends on:** P1 (action_io_registry.py) for type coercion, P2 (tool_type_models.py) for shared types.

---

## Quick Reference: Migration Effort by Current Return Type

### Tools returning `dict` (36 tools) — EASIER

These already return structured data. Migration:
1. Define Pydantic output model matching current dict shape
2. Add `formatted` field to the return dict (build human-readable string)
3. Define Pydantic input model
4. Register with `register_action()`
5. No caller changes needed (callers already handle dicts)

**Estimated effort:** ~15 min per tool

### Tools returning `str` (45 tools) — MODERATE

These need full restructuring. Migration:
1. Identify what structured data the tool actually produces (look at the string formatting logic)
2. Define Pydantic output model with typed fields
3. Refactor tool to build dict with typed fields
4. Move existing string formatting into `formatted` field
5. Define Pydantic input model
6. Register with `register_action()`
7. Update direct callers

**Estimated effort:** ~30-45 min per tool

### Tools returning `str` (JSON) (4 org tools) — MODERATE

These return JSON strings (already structured data, just serialized). Migration:
1. Return the parsed dict instead of `json.dumps()`
2. Add `formatted` field (can be the JSON string or a human-readable version)
3. Define Pydantic models
4. Register

**Estimated effort:** ~20 min per tool
