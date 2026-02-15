# Subgraph and Fragment Reference

Catalog of subgraphs in `llm-orchestrator/orchestrator/subgraphs/` and which are exposed as **fragments** (invocable as compound plan steps). See [SKILLS_ARCHITECTURE_PROGRESS.md](./SKILLS_ARCHITECTURE_PROGRESS.md) for overall skills architecture.

---

## Fragment Quick Reference

All fragments are defined in `orchestrator/engines/fragment_registry.py` and invoked via `invoke_fragment(fragment_name, query, metadata, messages, prior_context)`.

| Fragment name | Description | Output key | Checkpointer |
|---------------|-------------|------------|--------------|
| document_retrieval | Search and retrieve documents from local knowledge base with intelligent chunking | formatted_context | No |
| web_research | Search the web and crawl top results for current information | search_results | Yes |
| visualization | Generate charts and data visualizations from research data | static_visualization_data | Yes |
| data_formatting | Transform data into structured formats: tables, lists, timelines, comparisons | formatted_output | Yes |
| image_search | Search local image/comic collection by visual content, series, author, or date | image_search_results | No |
| diagramming | Generate Mermaid diagrams, ASCII circuit diagrams, or pin tables from a description | diagram_content | Yes |
| full_document_analysis | Deep-read up to 2 full documents and synthesize answers to specific questions | synthesis | Yes |
| gap_analysis | Analyze research results for information gaps, missing entities, and suggest follow-up queries | gap_analysis (dict) | Yes |
| assessment | Assess whether research results are sufficient to answer a query, with confidence score | assessment (dict) | Yes |
| fact_verification | Cross-reference and verify factual claims from research with web sources | consensus_findings (dict) | Yes |
| entity_relationship | Discover entity relationships via knowledge graph (Neo4j) from document corpus | kg_formatted_results | No |
| knowledge_document | Assemble verified research into a structured markdown knowledge document with citations | markdown_content | Yes |

Pipeline fragments (gap_analysis, assessment, fact_verification, knowledge_document) expect prior step outputs via `depends_on` and `context_keys`; see input_mapping in the registry.

---

## How to Use Fragments in Compound Plans

- A **PlanStep** can target either a **skill** (`skill_name`) or a **fragment** (`fragment_name`); the two are mutually exclusive.
- The LLM planner (skill_llm_selector) sees the list of fragments from `get_all_fragments()` and can emit steps with `fragment_name` set.
- **ContextBridge** injects prior step results into `prior_context` using `depends_on` (step IDs or skill/fragment names) and `context_keys`. Fragment input_mapping maps `prior_context` keys to subgraph state keys.
- For pipeline fragments, set `depends_on` to the step that produced the research/results, and use `context_keys` such as `["results"]`, `["research_findings"]`, `["verified_claims", "contradictions", "uncertainties"]` so the fragment receives the right inputs.

---

## How to Add a New Fragment

1. **Define a FragmentDef** in `orchestrator/engines/fragment_registry.py`:
   - `name`: canonical fragment name (snake_case)
   - `description`: short text for the LLM planner prompt
   - `input_mapping`: dict from context_bridge_key (e.g. `"query"`, `"results"`) to subgraph state key
   - `output_key`: subgraph state key for the main text result (string or dict; dicts are JSON-serialized)
   - `needs_checkpointer`: True if the subgraph is compiled with a checkpointer

2. **Add a build function** in the same file’s lazy `build_fns` dict (inside `invoke_fragment`): map fragment name to a lambda that takes a checkpointer and returns the compiled subgraph.

3. **Import** the subgraph’s `build_*_subgraph` in the same block and add the lambda.

4. **Fragment-specific state**: If the subgraph expects extra initial state (e.g. `document_ids`, `limit`), add an `elif fragment_name == "..."` block to set `base_state.setdefault(...)` after the generic input_mapping loop.

5. **Input transforms**: If a prior step yields a string but the subgraph expects a dict (e.g. `research_findings` with `combined_results`), handle it in the input_mapping loop (see fact_verification).

6. **Output**: `invoke_fragment` already turns dict outputs into JSON strings; string outputs are passed through.

---

## Subgraph Catalog

### Utility / Research Subgraphs (fragment candidates)

| Subgraph module | Build function | Fragment? | Notes |
|-----------------|----------------|-----------|--------|
| intelligent_document_retrieval_subgraph | build_intelligent_document_retrieval_subgraph() | Yes (document_retrieval) | No checkpointer |
| web_research_subgraph | build_web_research_subgraph(cp) | Yes | |
| visualization_subgraph | build_visualization_subgraph(cp) | Yes | |
| data_formatting_subgraph | build_data_formatting_subgraph(cp) | Yes | |
| image_search_subgraph | build_image_search_subgraph(cp) | Yes | Optional cp |
| diagramming_subgraph | build_diagramming_subgraph(cp) | Yes | |
| full_document_analysis_subgraph | build_full_document_analysis_subgraph(cp) | Yes | Needs document_ids, analysis_queries |
| gap_analysis_subgraph | build_gap_analysis_subgraph(cp) | Yes | Pipeline: needs results |
| assessment_subgraph | build_assessment_subgraph(cp) | Yes | Pipeline: needs results |
| fact_verification_subgraph | build_fact_verification_subgraph(cp) | Yes | Pipeline: needs research_findings.combined_results |
| entity_relationship_subgraph | build_entity_relationship_subgraph(cp) | Yes | Optional cp; can use vector_results from prior step |
| knowledge_document_subgraph | build_knowledge_document_subgraph(cp) | Yes | Pipeline: verified_claims, contradictions, uncertainties |
| attachment_analysis_subgraph | build_attachment_analysis_subgraph() | No | Requires binary attachments in state; not in registry |

### Research Pipeline Subgraphs (internal to ResearchEngine)

| Subgraph module | Build / entry | Fragment? | Notes |
|-----------------|---------------|-----------|--------|
| research_workflow_subgraph | build_* | No | Orchestrates rounds, document retrieval, web research, gap analysis, synthesis |
| collection_search_subgraph | execute_collection_search() | No | Callable function, not a StateGraph builder |
| factual_query_subgraph | build_* | No | Used inside research workflow |

### Editor Subgraphs (internal to EditorEngine / WritingAssistant)

Used by WritingAssistantAgent based on document type and skill. Not exposed as standalone fragments.

| Subgraph module | Notes |
|-----------------|--------|
| fiction_editing_subgraph | Manuscript edits, editor operations |
| fiction_generation_subgraph | Generate new fiction content |
| fiction_resolution_subgraph | Resolve editor operations |
| fiction_validation_subgraph | Consistency checks |
| fiction_context_subgraph | Load fiction context |
| fiction_book_generation_subgraph | Book-level generation |
| outline_editing_subgraph | Outline/lesson editing |
| rules_editing_subgraph | Rules document editing |
| style_editing_subgraph | Style guide editing |
| character_development_subgraph | Character docs |
| proofreading_subgraph | Proofread content |
| article_writing_subgraph | Article drafting |
| nonfiction_outline_subgraph | Nonfiction outline |
| podcast_script_subgraph | Podcast script |

### Navigation Subgraphs (internal to navigation skills)

| Subgraph module | Notes |
|-----------------|--------|
| route_planning_subgraph | Route planning flow |
| location_management_subgraph | Location CRUD |

### Other / Nested

| Location | Notes |
|----------|--------|
| org/parsing_subgraph, org/query_subgraph, org/synthesis_subgraph | Org-mode flows |
| proposal/*_subgraph | Proposal generation and validation |

---

## Fragment input_mapping reference

- **document_retrieval**: query, user_id  
- **web_research**: query  
- **visualization**: query  
- **data_formatting**: query  
- **image_search**: query, user_id  
- **diagramming**: query  
- **full_document_analysis**: query → original_query, document_ids → document_ids  
- **gap_analysis**: query, results  
- **assessment**: query, results  
- **fact_verification**: query, research_findings (string from prior step wrapped as {combined_results: s})  
- **entity_relationship**: query, user_id  
- **knowledge_document**: query, verified_claims, contradictions, uncertainties  

ContextBridge stores step results by step_id and optional context_keys; the key used in `prior_context` for a step is typically the step’s stored response or keys like `prior_step_1_response`, `results`, etc., depending on how the planner sets `context_keys`.
