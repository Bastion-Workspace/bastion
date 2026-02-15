# Agent Factory: Technical Architecture & Implementation Guide

**Document Version:** 1.0
**Last Updated:** February 14, 2026
**Status:** Planning Phase
**Companion to:** [AGENT_FACTORY.md](./AGENT_FACTORY.md) (product design), [AGENT_FACTORY_TOOLS.md](./AGENT_FACTORY_TOOLS.md) (tool catalog), [AGENT_FACTORY_EXAMPLES.md](./AGENT_FACTORY_EXAMPLES.md) (use cases)

---

## Purpose

This document covers the technical implementation details for the Agent Factory system. It is the "how" companion to `AGENT_FACTORY.md` (the "what" and "why"). Read that document first for context on the product vision, core concepts, and implementation phases.

---

## Table of Contents

1. [Runtime Execution Model](#1-runtime-execution-model)
2. [Database Schema](#2-database-schema)
3. [API Endpoint Specification](#3-api-endpoint-specification)
4. [Connector YAML Specification](#4-connector-yaml-specification)
5. [Playbook YAML Specification](#5-playbook-yaml-specification)
6. [Neo4j Graph Schema](#6-neo4j-graph-schema)
7. [Connector Template Library](#7-connector-template-library)
8. [Pre-Built Agent Profile Templates](#8-pre-built-agent-profile-templates)
9. [Scheduling & Monitoring](#9-scheduling--monitoring)
10. [Error Handling & Resilience](#10-error-handling--resilience)
11. [Integration Points](#11-integration-points)
12. [Modular Tool & Plugin Registry](#12-modular-tool--plugin-registry)
13. [Migration Strategy](#13-migration-strategy)

---

## 1. Runtime Execution Model

The central technical question: when a user activates a custom agent profile and sends a query, how does it actually execute?

### Overview

Custom agents do NOT get their own hard-coded LangGraph `StateGraph`. Instead, a single **Custom Agent Runner** dynamically assembles a workflow at execution time by reading the Agent Profile and binding the appropriate connectors, skills, tools, and output routes.

A custom agent's workflow is composed from three types of steps, mixed in any combination:

- **`tool` steps** — Deterministic: call a connector, tool, or data operation with specified parameters. No LLM involved.
- **`llm_task` steps** — LLM-driven: send data to an LLM for analysis, classification, or synthesis. Returns structured JSON.
- **`approval` steps** — Human-in-the-loop: pause workflow, show a preview of pending actions, resume on user confirmation or rejection.

This composition supports three execution modes:

| Execution Mode | Description | LLM Token Cost |
|---|---|---|
| `deterministic` | All steps are `tool` type. Pure data pipeline — no LLM tokens consumed. | Zero |
| `llm_augmented` | LLM receives tools and decides what to call (existing skill pattern). | Variable |
| `hybrid` | Mix of `tool`, `llm_task`, and `approval` steps. Deterministic data collection, LLM analysis, human oversight. | Targeted |

And three run contexts:

| Run Context | Behavior | Approval Handling |
|---|---|---|
| `interactive` | Runs in chat, streams progress to user | Inline in chat conversation |
| `background` | Runs outside chat session | Surfaced in notification queue |
| `scheduled` | Triggered by cron, runs as background job | Notification queue or `auto_approve` policy |
| `monitor` | Periodic polling; runs only when changes detected | Notification queue or `auto_approve` policy |

```
User sends query with agent_profile_id
         │
         ▼
┌─────────────────────────┐
│  Unified Dispatch         │  Detects agent_profile_id in metadata
│  (unified_dispatch.py)   │  Routes to CustomAgentEngine
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Custom Agent Engine      │  New engine type alongside
│  (custom_agent_engine.py)│  research, automation, editor, conversational
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Profile Loader           │  Load AgentProfile from DB
│                          │  Resolve connector bindings → tool functions
│                          │  Resolve skill bindings → playbook steps
│                          │  Resolve output config → output router
│                          │  Determine execution_mode + run_context
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Custom Agent Runner      │  BaseAgent subclass with dynamic workflow
│  (custom_agent_runner.py)│
└───────────┬─────────────┘
            │
            ▼
┌───────────────────────────────────────────────────┐
│  Dynamic LangGraph Workflow                         │
│                                                    │
│  prepare_context ──→ select_strategy               │
│       │                    │                       │
│       │         ┌──────────┼──────────┐            │
│       │         │          │          │            │
│       ▼         ▼          ▼          ▼            │
│  [playbook]  [research] [direct] [pipeline]        │
│  (hybrid)    (llm_aug)  (llm_aug) (deterministic)  │
│       │         │          │          │            │
│       └─────────┴──────────┴──────────┘            │
│                        │                           │
│                        ▼                           │
│              enrich_knowledge                      │
│                        │                           │
│                        ▼                           │
│              route_output                          │
│                        │                           │
│                        ▼                           │
│              format_response                       │
└───────────────────────────────────────────────────┘
```

### Custom Agent Runner (LangGraph Agent)

```python
class CustomAgentState(TypedDict):
    """State for custom agent LangGraph workflow"""
    # Standard fields
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]

    # Profile-specific
    agent_profile: Dict[str, Any]      # Loaded profile config
    bound_tools: List[Any]             # Resolved tool functions
    bound_playbooks: List[Dict]        # Resolved playbook definitions
    output_router: Dict[str, Any]      # Output routing config

    # Workflow composition
    execution_mode: str                 # "deterministic", "llm_augmented", "hybrid"
    run_context: str                    # "interactive", "background", "scheduled", "monitor"
    approval_policy: str                # "require", "auto_approve"

    # Execution state
    strategy: str                       # "playbook", "research", "direct", "pipeline"
    active_playbook: Optional[Dict]     # Currently executing playbook
    playbook_state: Dict[str, Any]      # Step results keyed by output_key (typed per action I/O contract)
    step_io_schemas: Dict[str, Any]     # Per-step resolved I/O schemas for runtime validation
    connector_results: List[Dict]       # Raw results from connectors
    extracted_entities: List[Dict]      # Entities found in results
    resolved_entities: List[Dict]       # After entity resolution

    # Approval gate state
    pending_approval: Optional[Dict]    # {step_name, preview_data, prompt, resume_token}
    approval_response: Optional[str]    # "approved", "rejected", or None

    # Journal state
    journal_entries: List[Dict]         # Accumulated journal entries for this execution
    journal_config: Dict[str, Any]      # Auto-journal, detail level, retention

    # Monitor state (for run_context="monitor")
    monitor_config: Optional[Dict]      # {interval, active_hours, suppress_if_empty}
    monitor_watermarks: Dict[str, Any]  # Per-step watermarks from previous run
    changes_detected: bool              # Whether any detection step found new items

    # Team context
    invoked_by: str                     # User ID of whoever triggered this run
    team_context: Optional[Dict]        # {team_id, team_file_access, team_post_access} if team-scoped
    team_tools: List[Any]               # Dynamically assembled team tools (based on permissions)

    # Invocation context
    agent_handle: str                   # @handle used to invoke this agent
    invocation_source: str              # "mention", "sidebar", "scheduled", "background"

    # Output
    response: Dict[str, Any]
    task_status: str
    error: str


class CustomAgentRunner(BaseAgent):
    """
    Executes user-defined agent profiles.

    Unlike other agents that have fixed workflows, this agent
    dynamically binds tools and skills based on the loaded profile.
    Invoked via @mention in chat, sidebar button, or scheduled execution.

    Supports three execution modes:
    - deterministic: Pure tool pipeline, no LLM. Steps run sequentially
      with data flowing between them via playbook_state output_keys.
    - llm_augmented: LLM receives bound tools and decides what to call.
      This is the traditional agent pattern.
    - hybrid: Mix of tool, llm_task, and approval steps. The playbook
      defines which steps are deterministic and which need LLM/human.

    After execution, automatically writes a journal entry summarizing
    work done. Journal and query_journal tools are always available.
    Team tools (search_team_files, write_team_post, etc.) are
    dynamically added based on agent's team_config permissions.

    And three run contexts:
    - interactive: Runs in chat, streams progress.
    - background: Runs outside chat session, delivers results to output
      destinations when complete.
    - scheduled: Triggered by cron, runs as background.
    """

    def __init__(self):
        super().__init__(agent_type="custom_agent")
        self.connector_runtime = ConnectorRuntime()
        self.entity_resolver = EntityResolutionService()
        self.output_router = OutputRouter()
        self.profile_cache = {}  # agent_profile_id → loaded profile

    def _build_workflow(self, checkpointer) -> StateGraph:
        workflow = StateGraph(CustomAgentState)

        # Core nodes (always present)
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("select_strategy", self._select_strategy_node)
        workflow.add_node("enrich_knowledge", self._enrich_knowledge_node)
        workflow.add_node("route_output", self._route_output_node)
        workflow.add_node("format_response", self._format_response_node)

        # Strategy-specific nodes
        workflow.add_node("execute_playbook", self._execute_playbook_node)
        workflow.add_node("execute_research", self._execute_research_node)
        workflow.add_node("execute_direct", self._execute_direct_node)
        workflow.add_node("execute_pipeline", self._execute_pipeline_node)

        # Approval gate node (for hybrid/deterministic workflows with approval steps)
        workflow.add_node("handle_approval", self._handle_approval_node)

        # Flow
        workflow.set_entry_point("prepare_context")
        workflow.add_edge("prepare_context", "select_strategy")

        # Conditional routing based on strategy
        workflow.add_conditional_edges(
            "select_strategy",
            self._route_strategy,
            {
                "playbook": "execute_playbook",
                "research": "execute_research",
                "direct": "execute_direct",
                "pipeline": "execute_pipeline",
            }
        )

        # All strategies converge to enrichment (or approval gate)
        workflow.add_conditional_edges(
            "execute_playbook",
            self._check_pending_approval,
            {"approval_needed": "handle_approval", "continue": "enrich_knowledge"}
        )
        workflow.add_conditional_edges(
            "execute_pipeline",
            self._check_pending_approval,
            {"approval_needed": "handle_approval", "continue": "enrich_knowledge"}
        )
        workflow.add_edge("execute_research", "enrich_knowledge")
        workflow.add_edge("execute_direct", "enrich_knowledge")

        # Approval gate can resume back into execution or proceed
        workflow.add_conditional_edges(
            "handle_approval",
            self._route_after_approval,
            {"resume_playbook": "execute_playbook", "resume_pipeline": "execute_pipeline",
             "rejected": "format_response", "continue": "enrich_knowledge"}
        )

        # Enrichment → output routing → response formatting
        workflow.add_edge("enrich_knowledge", "route_output")
        workflow.add_edge("route_output", "format_response")
        workflow.add_edge("format_response", END)

        # Compile with interrupt_before for approval gates
        return workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["handle_approval"]
        )
```

### Node Details

**prepare_context_node:**
1. Load Agent Profile from DB (or cache), resolve `agent_handle` and `invocation_source` from metadata
2. Resolve connector bindings → instantiate connector tool functions
3. Resolve skill bindings → load playbook definitions
4. Build output routing configuration
5. Determine `execution_mode` from profile/playbook (`deterministic`, `llm_augmented`, `hybrid`)
6. Determine `run_context` from invocation metadata (`interactive`, `background`, `scheduled`)
7. Resolve step I/O schemas: load each action's I/O contract from the action_io_registry, resolve connector output_schemas for `call_connector` steps, validate all step input wiring at startup. Store in `step_io_schemas`.
8. Assemble team tools: if `team_config` grants file or post access, dynamically add `search_team_files`, `read_team_file`, `search_team_posts`, `write_team_post`, `summarize_team_thread` to the agent's bound tools. Store team context in `team_context`.
9. Add journal tools: `write_journal_entry` and `query_journal` are always added to bound tools. Load `journal_config` from profile.
10. Inject profile's `system_prompt_additions` into messages
9. Search knowledge graph for entities mentioned in query (pre-existing context)

**select_strategy_node:**
1. If playbook's `execution_mode` is `deterministic` → `pipeline` strategy (no LLM)
2. Check if query matches any playbook triggers → `playbook` strategy (hybrid execution)
3. Check if query requires multi-round research → `research` strategy
4. Otherwise → `direct` strategy (single connector call or knowledge graph query)

**execute_pipeline_node (NEW — deterministic execution):**
1. Iterate through playbook steps in order — no LLM involved
2. For each step based on `step_type`:
   - `tool`: Resolve `inputs` map by evaluating variable references against `playbook_state`. Type-check resolved values against the action's I/O contract from `step_io_schemas`. Call connector/tool with resolved inputs + static params. Store result in `playbook_state[output_key]`.
   - `approval`: Set `pending_approval` in state with preview data and prompt. Return to let the graph route to `handle_approval`.
3. Input resolution: `{step_name.field}` references in `inputs` are resolved from `playbook_state`, with type coercion applied per the I/O contract
4. Handle conditionals (skip step if condition not met)
5. Handle `on_error` per step (skip / stop / retry)
6. Collect all connector_results for later enrichment

This is the key differentiator from `execute_playbook_node`: **no LLM calls are made**. The playbook YAML is the complete execution specification. This makes deterministic pipelines predictable, testable, auditable, and zero-cost in LLM tokens.

**execute_playbook_node (updated — hybrid execution):**
1. Iterate through playbook steps in order
2. For each step based on `step_type`:
   - `tool`: Call connector/tool with resolved parameters (same as pipeline — no LLM)
   - `llm_task`: Send accumulated context to LLM for analysis. LLM returns structured JSON. Actions: `synthesize_report`, `llm_analyze`, `research_with_tools`
   - `approval`: Set `pending_approval` in state with preview data and prompt. Return to let the graph route to `handle_approval`.
3. Pass output_key results between steps via `playbook_state`
4. Handle conditionals (skip step if condition not met)
5. Collect all connector_results for later enrichment

The difference from `execute_pipeline_node`: hybrid playbooks can include `llm_task` steps that invoke the LLM for analysis, synthesis, or tool-use reasoning.

**handle_approval_node (NEW — human-in-the-loop gate):**
1. Read `pending_approval` from state (set by playbook/pipeline execution)
2. Based on `run_context`:
   - `interactive`: Stream the approval preview to the chat. LangGraph's `interrupt_before` pauses the workflow. User responds in chat, which resumes the graph with `approval_response`.
   - `background`/`scheduled`: Push approval request to notification queue (PostgreSQL `agent_approval_queue` table). Workflow remains checkpointed. User approves via UI notification, which resumes the graph.
3. If `approval_policy` is `auto_approve` (scheduled workflows), skip the gate and proceed.
4. On resume: set `approval_response` in state and route accordingly.

**execute_research_node:**
1. Delegate to existing research workflow subgraph
2. But with connector-generated tools bound alongside standard search tools
3. The research agent can call FEC API, ProPublica, etc. as naturally as it calls SearXNG

**execute_direct_node:**
1. Single connector call or knowledge graph query
2. For simple questions that don't need multi-step research

**enrich_knowledge_node:**
1. Extract entities from all connector_results (using connector-defined entity_extraction + spaCy NER)
2. Run entity resolution against existing graph
3. Extract typed relationships from context
4. Store new entities and relationships in Neo4j
5. Store embeddings in Qdrant
6. Log discoveries in agent_discoveries table

**route_output_node:**
1. For each destination in output_config:
   - Transform data to destination format
   - Write to destination (document, folder, Data Workspace table, etc.)
   - Track what was written where
2. Handle auto_save vs. confirmation

**format_response_node:**
1. Build chat response (always present for interactive; notification for background)
2. Include links to saved documents/files
3. Include summary of knowledge graph enrichment ("Found 3 new entities, 5 new relationships")
4. Include download links for file exports
5. For background jobs: generate completion notification with result summary
6. **Write journal entry:** Call `write_journal_entry` with a summary of the execution — steps completed, entities found, outputs produced, errors encountered. For LLM-augmented workflows, the LLM generates the natural-language summary. For deterministic pipelines, the engine generates a structured summary from step results. Journal entry is linked to the execution_id for drill-down.

### Dynamic Tool Binding

When connectors are resolved, each endpoint becomes a LangGraph tool:

```python
class ConnectorRuntime:
    """Generates callable tool functions from connector definitions."""

    async def bind_connector(
        self,
        connector_def: Dict,
        credentials: Dict
    ) -> List[Callable]:
        """
        Turn a connector definition into callable tool functions.

        Each endpoint becomes one tool function that the LLM can call.
        """
        tools = []
        connector_type = connector_def["type"]

        for endpoint_name, endpoint_def in connector_def["endpoints"].items():
            if connector_type == "rest_api":
                tool_fn = self._create_rest_api_tool(
                    connector_def["connection"],
                    endpoint_name,
                    endpoint_def,
                    credentials
                )
            elif connector_type == "web_scraper":
                tool_fn = self._create_scraper_tool(
                    connector_def["connection"],
                    endpoint_name,
                    endpoint_def,
                    credentials
                )
            elif connector_type == "graphql":
                tool_fn = self._create_graphql_tool(
                    connector_def["connection"],
                    endpoint_name,
                    endpoint_def,
                    credentials
                )
            elif connector_type == "file_parser":
                tool_fn = self._create_file_parser_tool(
                    endpoint_name,
                    endpoint_def,
                    credentials
                )
            elif connector_type == "database":
                tool_fn = self._create_database_tool(
                    connector_def["connection"],
                    endpoint_name,
                    endpoint_def,
                    credentials
                )
            # ... etc

            tools.append(tool_fn)

        return tools

    def _create_rest_api_tool(self, connection, name, endpoint, creds):
        """Generate a tool function for a REST API endpoint."""

        # Build tool description from endpoint metadata
        description = endpoint["description"]
        params = endpoint.get("parameters", [])

        # Build parameter schema for the LLM
        param_schema = {}
        for p in params:
            param_schema[p["name"]] = {
                "type": p["type"],
                "description": p["description"],
                "required": p.get("required", False),
            }

        async def tool_function(**kwargs) -> Dict[str, Any]:
            """Dynamically generated REST API tool."""
            url = connection["base_url"] + endpoint["path"]

            # Handle path parameters
            for key, value in kwargs.items():
                if f"{{{key}}}" in url:
                    url = url.replace(f"{{{key}}}", str(value))

            # Build query parameters
            query_params = {
                k: v for k, v in kwargs.items()
                if f"{{{k}}}" not in endpoint["path"]
            }

            # Add auth
            auth_config = connection.get("auth", {})
            if auth_config.get("type") == "api_key":
                query_params[auth_config["key_param"]] = creds.get("api_key")

            # Rate limiting
            await self._rate_limit(connection.get("rate_limit", {}))

            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    endpoint.get("method", "GET"),
                    url,
                    params=query_params,
                    timeout=aiohttp.ClientTimeout(
                        total=connection.get("timeout_seconds", 30)
                    )
                ) as resp:
                    data = await resp.json()

            # Apply response mapping
            mapped_results = self._apply_response_mapping(
                data, endpoint.get("response_mapping", {})
            )

            # Handle pagination if configured
            pagination = endpoint.get("pagination", {})
            if pagination and self._has_more_pages(data, pagination):
                mapped_results["_has_more"] = True
                mapped_results["_next_cursor"] = self._get_cursor(
                    data, pagination
                )

            return mapped_results

        # Set tool metadata for LLM
        tool_function.__name__ = f"{name}"
        tool_function.__doc__ = description
        tool_function._parameter_schema = param_schema

        return tool_function
```

### Output Router

```python
class OutputRouter:
    """Routes agent results to configured destinations."""

    async def route(
        self,
        results: Dict[str, Any],
        output_config: Dict[str, Any],
        user_id: str,
        agent_profile_id: str,
        execution_id: str
    ) -> Dict[str, Any]:
        """Route results to all configured destinations."""
        routing_report = []

        for destination in output_config.get("destinations", []):
            # Check condition
            condition = destination.get("condition")
            if condition and not self._evaluate_condition(condition, results):
                continue

            # Apply transforms
            transformed = self._apply_transforms(
                results, destination.get("transforms", [])
            )

            # Format for destination
            formatted = self._format_for_destination(
                transformed,
                destination["format"],
                destination["type"]
            )

            # Route to destination
            dest_type = destination["type"]
            config = destination.get("config", {})

            try:
                if dest_type == "document":
                    result = await self._save_to_document(
                        formatted, config, user_id
                    )
                elif dest_type == "folder":
                    result = await self._save_to_folder(
                        formatted, config, user_id
                    )
                elif dest_type == "data_workspace_table":
                    result = await self._insert_to_workspace_table(
                        formatted, config, user_id
                    )
                elif dest_type == "data_workspace_db":
                    result = await self._create_workspace_tables(
                        formatted, config, user_id
                    )
                elif dest_type == "knowledge_graph":
                    result = await self._enrich_graph(
                        formatted, config, user_id
                    )
                elif dest_type == "file_export":
                    result = await self._generate_export_file(
                        formatted, config, user_id
                    )
                elif dest_type == "append_to_existing":
                    result = await self._append_to_document(
                        formatted, config, user_id
                    )
                elif dest_type == "chat":
                    result = {"type": "chat", "included_in_response": True}

                routing_report.append({
                    "destination": dest_type,
                    "status": "success",
                    "details": result
                })

            except Exception as e:
                routing_report.append({
                    "destination": dest_type,
                    "status": "error",
                    "error": str(e)
                })

        return {"routing_report": routing_report}

    async def _save_to_document(self, content, config, user_id):
        """Save formatted content as a new document."""
        filename = self._render_template(
            config.get("filename_template", "agent_output_{date}.md"),
            content
        )
        folder_id = config.get("folder_id")

        if config.get("create_if_missing") and folder_id:
            folder_id = await self._ensure_folder_exists(folder_id, user_id)

        doc_id = await create_document_tool(
            title=filename,
            content=content["text"],
            folder_id=folder_id,
            user_id=user_id
        )

        return {"document_id": doc_id, "filename": filename}

    async def _insert_to_workspace_table(self, data, config, user_id):
        """Insert structured rows into a Data Workspace table."""
        workspace_id = config["workspace_id"]
        database_id = config["database_id"]
        table_name = config["table_name"]
        schema_mapping = config.get("schema_mapping", {})

        if config.get("create_if_missing"):
            await self._ensure_table_exists(
                workspace_id, database_id, table_name,
                schema_mapping, user_id
            )

        # Insert rows
        rows_inserted = 0
        dedup_key = config.get("deduplication_key")

        for row in data.get("rows", []):
            # Map fields to columns
            mapped_row = {}
            for field, value in row.items():
                if field in schema_mapping:
                    mapped_row[field] = value

            # Add metadata columns
            mapped_row["_source"] = data.get("_source", "unknown")
            mapped_row["_query"] = data.get("_query", "")
            mapped_row["_timestamp"] = datetime.now().isoformat()

            # Deduplication check
            if dedup_key:
                if await self._row_exists(
                    workspace_id, database_id, table_name,
                    {k: mapped_row.get(k) for k in dedup_key}
                ):
                    continue

            await self._insert_row(
                workspace_id, database_id, table_name, mapped_row
            )
            rows_inserted += 1

        return {"table": table_name, "rows_inserted": rows_inserted}
```

### Deterministic Pipeline Executor

The pipeline executor handles `tool` and `approval` steps without any LLM involvement. It is the core runtime for `execution_mode: deterministic` workflows and handles `tool` steps within hybrid workflows.

```python
class PipelineExecutor:
    """
    Executes deterministic playbook steps (tool and approval only).

    This executor never calls an LLM. The playbook YAML is the complete
    execution specification. Data flows between steps via output_keys
    stored in playbook_state.

    For hybrid workflows, the CustomAgentRunner calls PipelineExecutor
    for tool steps and LLMTaskExecutor for llm_task steps.
    """

    def __init__(self, connector_runtime: ConnectorRuntime):
        self.connector_runtime = connector_runtime
        self.action_handlers = {
            "call_connector": self._execute_connector_call,
            "search_knowledge_graph": self._execute_graph_search,
            "search_documents": self._execute_document_search,
            "search_web": self._execute_web_search,
            "crawl_url": self._execute_crawl,
            "extract_entities": self._execute_entity_extraction,
            "resolve_entities": self._execute_entity_resolution,
            "cross_reference": self._execute_cross_reference,
            "analyze_graph": self._execute_graph_algorithm,
            "transform_data": self._execute_data_transform,
            "save_to_workspace": self._execute_workspace_save,
            "route_output": self._execute_output_route,
        }

    async def execute_step(
        self,
        step: Dict[str, Any],
        playbook_state: Dict[str, Any],
        inputs: Dict[str, Any],
        bound_tools: List[Any],
    ) -> Dict[str, Any]:
        """
        Execute a single deterministic step.

        Returns the step result (stored in playbook_state[output_key]).
        Raises StepError on failure (caller handles on_error policy).
        """
        # Check condition
        condition = step.get("condition")
        if condition and not self._evaluate_condition(condition, playbook_state, inputs):
            return {"_skipped": True, "_reason": f"Condition not met: {condition}"}

        # Resolve variable interpolation in params
        resolved_params = self._resolve_variables(
            step.get("params", {}), playbook_state, inputs
        )

        # Execute action
        action = step["action"]
        handler = self.action_handlers.get(action)
        if not handler:
            raise StepError(f"Unknown tool action: {action}")

        result = await handler(step, resolved_params, playbook_state, bound_tools)
        return result

    async def execute_pipeline(
        self,
        steps: List[Dict[str, Any]],
        playbook_state: Dict[str, Any],
        inputs: Dict[str, Any],
        bound_tools: List[Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict]]:
        """
        Execute all tool steps in sequence.

        Returns (updated_playbook_state, pending_approval_or_none).
        If an approval step is encountered, execution pauses and
        returns the approval request. The caller checkpoints state
        and resumes after user responds.
        """
        for step in steps:
            step_type = step.get("step_type", "tool")

            if step_type == "approval":
                # Build approval preview from referenced step output
                preview_data = self._build_approval_preview(
                    step, playbook_state
                )
                return playbook_state, {
                    "step_name": step["name"],
                    "preview_data": preview_data,
                    "prompt": self._resolve_template(
                        step["prompt"], playbook_state, inputs
                    ),
                    "timeout_minutes": step.get("timeout_minutes"),
                    "on_reject": step.get("on_reject", "stop"),
                }

            if step_type == "llm_task":
                # Hybrid workflow: caller handles llm_task steps separately
                raise StepTypeError(
                    f"Step '{step['name']}' is llm_task — use HybridExecutor"
                )

            # Execute tool step
            try:
                result = await self.execute_step(
                    step, playbook_state, inputs, bound_tools
                )
                if step.get("output_key"):
                    playbook_state[step["output_key"]] = result
            except Exception as e:
                on_error = step.get("on_error", "skip")
                if on_error == "stop":
                    raise
                elif on_error == "retry":
                    try:
                        result = await self.execute_step(
                            step, playbook_state, inputs, bound_tools
                        )
                        if step.get("output_key"):
                            playbook_state[step["output_key"]] = result
                    except Exception:
                        pass  # Skip on retry failure
                # else: skip — continue to next step

        return playbook_state, None  # No pending approval
```

### Approval Gate Handler

The approval gate integrates with LangGraph's `interrupt_before` mechanism for interactive workflows and a PostgreSQL notification queue for background/scheduled workflows.

```python
class ApprovalGateHandler:
    """
    Manages approval gates across all run contexts.

    Interactive: Uses LangGraph interrupt_before to pause the graph.
                 User responds in chat. Graph resumes.
    Background:  Writes approval request to agent_approval_queue table.
                 User approves via notification UI. Celery task resumes graph.
    Scheduled:   Same as background, OR auto_approve if policy allows.
    """

    async def handle_approval(
        self,
        state: CustomAgentState,
    ) -> Dict[str, Any]:
        """Process an approval gate based on run context and policy."""
        pending = state.get("pending_approval")
        if not pending:
            return {"approval_response": "approved"}  # No-op

        approval_policy = state.get("approval_policy", "require")
        run_context = state.get("run_context", "interactive")

        # Auto-approve for scheduled workflows with auto_approve policy
        if approval_policy == "auto_approve":
            return {
                "approval_response": "approved",
                "pending_approval": None,
            }

        if run_context == "interactive":
            # For interactive: LangGraph's interrupt_before pauses here.
            # The frontend shows the approval preview in chat.
            # User's next message resumes the graph.
            # This node reads the user's response from state.
            user_response = state.get("approval_response")
            if user_response == "approved":
                return {"pending_approval": None}
            else:
                on_reject = pending.get("on_reject", "stop")
                if on_reject == "stop":
                    return {
                        "task_status": "rejected",
                        "pending_approval": None,
                        "error": f"User rejected: {pending['step_name']}"
                    }
                else:  # skip
                    return {"pending_approval": None}

        else:
            # Background/scheduled: write to approval queue
            await self._queue_approval_request(
                user_id=state["user_id"],
                agent_profile_id=state["agent_profile"].get("id"),
                step_name=pending["step_name"],
                preview_data=pending["preview_data"],
                prompt=pending["prompt"],
                timeout_minutes=pending.get("timeout_minutes"),
                # Store enough info to resume the LangGraph workflow
                thread_id=state["metadata"].get("thread_id"),
                checkpoint_ns=state["metadata"].get("checkpoint_ns"),
            )
            # Workflow stays checkpointed. Celery watcher resumes on approval.
            return {"task_status": "awaiting_approval"}
```

### Background Job Execution

Background and scheduled workflows execute through Celery but use the same CustomAgentRunner. The key difference is how results are delivered and how approval gates are handled.

```python
@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
async def run_agent_background(self, agent_profile_id: str, query: str,
                                user_id: str, schedule_id: str = None):
    """
    Execute an agent workflow as a background job.

    Same execution path as interactive, but:
    - No streaming to chat
    - Results delivered to configured output destinations
    - Approval gates surface in notification queue
    - Completion triggers a notification
    """
    metadata = {
        "run_context": "scheduled" if schedule_id else "background",
        "schedule_id": schedule_id,
        "agent_profile_id": agent_profile_id,
    }

    result = await execute_custom_agent(
        agent_profile_id=agent_profile_id,
        query=query,
        user_id=user_id,
        metadata=metadata,
    )

    # Send completion notification
    await send_user_notification(
        user_id=user_id,
        notification_type="agent_job_complete",
        title=f"Agent completed: {result.get('agent_name', 'Custom Agent')}",
        summary=result.get("summary", "Job finished."),
        details={
            "execution_id": result.get("execution_id"),
            "entities_discovered": result.get("entities_discovered", 0),
            "output_destinations": result.get("routing_report", []),
        }
    )

    return result
```

---

## 2. Database Schema

### Core Tables

```sql
-- ============================================================
-- AGENT PROFILES
-- ============================================================

CREATE TABLE agent_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    handle VARCHAR(100) NOT NULL,
    -- Unique @mention handle (kebab-case, e.g. "nonprofit-investigator")
    -- Used as @handle in chat. Unique per user (different users can have same handle).
    description TEXT,
    icon VARCHAR(50),  -- emoji or icon name
    is_active BOOLEAN DEFAULT true,
    model_preference VARCHAR(255),  -- e.g. "anthropic/claude-sonnet-4-20250514"
    max_research_rounds INTEGER DEFAULT 3,
    system_prompt_additions TEXT,  -- domain-specific instructions appended to system prompt
    knowledge_config JSONB DEFAULT '{}',
    -- {
    --   "read_collections": ["user_123_default"],
    --   "write_collection": "user_123_research",
    --   "graph_namespaces": ["user_123"],
    --   "auto_enrich": true,
    --   "entity_resolution": true
    -- }
    output_config JSONB DEFAULT '{}',
    -- Full OutputConfig as described in AGENT_FACTORY.md
    default_execution_mode VARCHAR(50) DEFAULT 'hybrid',
    -- "deterministic", "llm_augmented", "hybrid"
    -- Can be overridden per-playbook
    default_run_context VARCHAR(50) DEFAULT 'interactive',
    -- "interactive", "background", "scheduled"
    default_approval_policy VARCHAR(50) DEFAULT 'require',
    -- "require", "auto_approve"
    journal_config JSONB DEFAULT '{"auto_journal": true, "detail_level": "summary", "retention_days": 90}',
    -- Work journal settings
    team_config JSONB DEFAULT '{}',
    -- {
    --   "shared_with_teams": ["team-uuid-1"],
    --   "team_file_access": false,
    --   "team_post_access": false,
    --   "team_permissions": {"team-uuid-1": {"file_access": true, "post_access": false}}
    -- }
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Handle uniqueness: unique per user (different users can reuse handles)
    UNIQUE (user_id, handle)
);

CREATE INDEX idx_agent_profiles_user ON agent_profiles(user_id);
CREATE INDEX idx_agent_profiles_active ON agent_profiles(user_id, is_active);
CREATE INDEX idx_agent_profiles_handle ON agent_profiles(user_id, handle);

-- RLS: owner can see their own agents; team members can see agents shared with their teams
ALTER TABLE agent_profiles ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_profiles_user_policy ON agent_profiles
    USING (
        user_id = current_setting('app.user_id')::UUID
        OR team_config->'shared_with_teams' ?| (
            SELECT array_agg(team_id::text)
            FROM team_members
            WHERE user_id = current_setting('app.user_id')::UUID
        )
    );


-- ============================================================
-- DATA SOURCE CONNECTORS (Template Definitions)
-- ============================================================

CREATE TABLE data_source_connectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    -- NULL user_id = system-provided template (FEC, ProPublica, etc.)
    -- Non-NULL user_id = user-created custom connector
    name VARCHAR(255) NOT NULL,
    description TEXT,
    connector_type VARCHAR(50) NOT NULL,
    -- "rest_api", "graphql", "web_scraper", "file_parser",
    -- "database", "rss_feed", "existing_tool"
    version VARCHAR(20) DEFAULT '1.0',
    definition JSONB NOT NULL,
    -- Full YAML connector definition stored as JSONB
    -- Includes: connection, endpoints, response_mapping, entity_extraction
    is_template BOOLEAN DEFAULT false,
    -- true = system-provided, immutable by users
    -- false = user-created or user-customized copy of template
    requires_auth BOOLEAN DEFAULT false,
    auth_fields JSONB DEFAULT '[]',
    -- [{"name": "api_key", "label": "API Key", "type": "password", "env_var": "FEC_API_KEY"}]
    icon VARCHAR(50),
    category VARCHAR(100),
    -- "government", "finance", "corporate", "nonprofit", "general"
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_connectors_user ON data_source_connectors(user_id);
CREATE INDEX idx_connectors_type ON data_source_connectors(connector_type);
CREATE INDEX idx_connectors_template ON data_source_connectors(is_template);
CREATE INDEX idx_connectors_category ON data_source_connectors(category);

ALTER TABLE data_source_connectors ENABLE ROW LEVEL SECURITY;
CREATE POLICY connectors_access_policy ON data_source_connectors
    USING (
        is_template = true  -- Everyone can see templates
        OR user_id = current_setting('app.user_id')::UUID  -- Own connectors
    );


-- ============================================================
-- AGENT DATA SOURCE BINDINGS
-- ============================================================

CREATE TABLE agent_data_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    connector_id UUID NOT NULL REFERENCES data_source_connectors(id),
    credentials_encrypted JSONB,
    -- Encrypted API keys, tokens, etc.
    -- Encryption at application level before storage
    config_overrides JSONB DEFAULT '{}',
    -- Override base_url, rate limits, etc.
    permissions JSONB DEFAULT '{}',
    -- {"allowed_endpoints": ["search_contributions"], "rate_limit_override": 5}
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_sources_profile ON agent_data_sources(agent_profile_id);

ALTER TABLE agent_data_sources ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_sources_policy ON agent_data_sources
    USING (
        agent_profile_id IN (
            SELECT id FROM agent_profiles
            WHERE user_id = current_setting('app.user_id')::UUID
        )
    );


-- ============================================================
-- CUSTOM PLAYBOOKS
-- ============================================================

CREATE TABLE custom_playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    -- NULL = system-provided playbook
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(20) DEFAULT '1.0',
    definition JSONB NOT NULL,
    -- Full playbook YAML stored as JSONB
    -- Includes: triggers, steps, output config
    triggers JSONB DEFAULT '[]',
    -- Extracted from definition for fast trigger matching
    -- [{"pattern": "investigate nonprofit", "type": "keyword"}]
    is_template BOOLEAN DEFAULT false,
    category VARCHAR(100),
    tags TEXT[] DEFAULT '{}',
    required_connectors TEXT[] DEFAULT '{}',
    -- Connector IDs or types that this playbook needs
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_playbooks_user ON custom_playbooks(user_id);
CREATE INDEX idx_playbooks_triggers ON custom_playbooks USING GIN (triggers);

ALTER TABLE custom_playbooks ENABLE ROW LEVEL SECURITY;
CREATE POLICY playbooks_access_policy ON custom_playbooks
    USING (
        is_template = true
        OR user_id = current_setting('app.user_id')::UUID
    );


-- ============================================================
-- AGENT SKILL BINDINGS
-- ============================================================

CREATE TABLE agent_skills (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    skill_type VARCHAR(50) NOT NULL,
    -- "built_in" (references static skill name) or "playbook" (references custom_playbooks.id)
    skill_reference VARCHAR(255) NOT NULL,
    -- For built_in: skill name (e.g., "research", "knowledge_builder")
    -- For playbook: custom_playbooks.id as string
    priority INTEGER DEFAULT 0,
    parameters JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_agent_skills_profile ON agent_skills(agent_profile_id);

ALTER TABLE agent_skills ENABLE ROW LEVEL SECURITY;
CREATE POLICY agent_skills_policy ON agent_skills
    USING (
        agent_profile_id IN (
            SELECT id FROM agent_profiles
            WHERE user_id = current_setting('app.user_id')::UUID
        )
    );


-- ============================================================
-- EXECUTION TRACKING
-- ============================================================

CREATE TABLE agent_execution_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID REFERENCES agent_profiles(id) ON DELETE SET NULL,
    user_id UUID NOT NULL REFERENCES users(id),
    query TEXT NOT NULL,
    strategy VARCHAR(50),  -- "playbook", "research", "direct"
    playbook_id UUID REFERENCES custom_playbooks(id),
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    duration_ms INTEGER,
    status VARCHAR(50) DEFAULT 'running',
    -- "running", "complete", "error", "timeout"
    connectors_called JSONB DEFAULT '[]',
    -- [{"connector": "fec", "endpoint": "search_contributions", "latency_ms": 450, "results_count": 23}]
    entities_discovered INTEGER DEFAULT 0,
    relationships_discovered INTEGER DEFAULT 0,
    output_destinations JSONB DEFAULT '[]',
    -- [{"type": "document", "id": "doc_123"}, {"type": "data_workspace_table", "rows": 23}]
    error_details TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_execution_log_profile ON agent_execution_log(agent_profile_id);
CREATE INDEX idx_execution_log_user ON agent_execution_log(user_id);
CREATE INDEX idx_execution_log_time ON agent_execution_log(started_at DESC);

ALTER TABLE agent_execution_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY execution_log_policy ON agent_execution_log
    USING (user_id = current_setting('app.user_id')::UUID);


-- ============================================================
-- DISCOVERY LOG (Entities/Relationships Found Per Execution)
-- ============================================================

CREATE TABLE agent_discoveries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES agent_execution_log(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    discovery_type VARCHAR(50) NOT NULL,
    -- "new_entity", "new_relationship", "entity_merge",
    -- "new_external_id", "pattern_detected"
    entity_name VARCHAR(500),
    entity_type VARCHAR(50),
    entity_neo4j_id VARCHAR(255),
    relationship_type VARCHAR(100),
    related_entity_name VARCHAR(500),
    source_connector VARCHAR(255),
    source_endpoint VARCHAR(255),
    confidence REAL,
    details JSONB DEFAULT '{}',
    discovered_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_discoveries_execution ON agent_discoveries(execution_id);
CREATE INDEX idx_discoveries_user ON agent_discoveries(user_id);
CREATE INDEX idx_discoveries_entity ON agent_discoveries(entity_name);
CREATE INDEX idx_discoveries_type ON agent_discoveries(discovery_type);
CREATE INDEX idx_discoveries_time ON agent_discoveries(discovered_at DESC);

ALTER TABLE agent_discoveries ENABLE ROW LEVEL SECURITY;
CREATE POLICY discoveries_policy ON agent_discoveries
    USING (user_id = current_setting('app.user_id')::UUID);


-- ============================================================
-- ENTITY RESOLUTION AUDIT LOG
-- ============================================================

CREATE TABLE entity_resolution_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    execution_id UUID REFERENCES agent_execution_log(id),
    action VARCHAR(50) NOT NULL,
    -- "auto_merge", "user_merge", "user_split", "create_new", "flag_review"
    source_entity_name VARCHAR(500) NOT NULL,
    source_entity_type VARCHAR(50),
    target_entity_name VARCHAR(500),  -- canonical entity (for merges)
    target_entity_neo4j_id VARCHAR(255),
    confidence REAL,
    match_method VARCHAR(100),
    -- "exact_name", "fuzzy_name", "external_id", "context_overlap"
    match_details JSONB DEFAULT '{}',
    -- {"jaro_winkler": 0.93, "shared_documents": 3, "external_id_match": "irs_ein"}
    is_reversible BOOLEAN DEFAULT true,
    reversed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_resolution_log_user ON entity_resolution_log(user_id);
CREATE INDEX idx_resolution_log_action ON entity_resolution_log(action);
CREATE INDEX idx_resolution_log_entity ON entity_resolution_log(source_entity_name);

ALTER TABLE entity_resolution_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY resolution_log_policy ON entity_resolution_log
    USING (user_id = current_setting('app.user_id')::UUID);


-- ============================================================
-- PENDING ENTITY REVIEWS (HITL for ambiguous merges)
-- ============================================================

CREATE TABLE entity_review_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    execution_id UUID REFERENCES agent_execution_log(id),
    source_entity_name VARCHAR(500) NOT NULL,
    source_entity_type VARCHAR(50),
    candidate_entity_name VARCHAR(500) NOT NULL,
    candidate_entity_neo4j_id VARCHAR(255),
    confidence REAL NOT NULL,
    match_details JSONB DEFAULT '{}',
    context JSONB DEFAULT '{}',
    -- {"source_documents": [...], "shared_relationships": [...]}
    status VARCHAR(50) DEFAULT 'pending',
    -- "pending", "approved_merge", "rejected", "skipped"
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_review_queue_user ON entity_review_queue(user_id, status);

ALTER TABLE entity_review_queue ENABLE ROW LEVEL SECURITY;
CREATE POLICY review_queue_policy ON entity_review_queue
    USING (user_id = current_setting('app.user_id')::UUID);


-- ============================================================
-- APPROVAL QUEUE (for background/scheduled workflow gates)
-- ============================================================

CREATE TABLE agent_approval_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    agent_profile_id UUID REFERENCES agent_profiles(id),
    execution_id UUID REFERENCES agent_execution_log(id),
    step_name VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    preview_data JSONB DEFAULT '{}',
    -- Snapshot of data shown to user for the approval decision
    status VARCHAR(50) DEFAULT 'pending',
    -- "pending", "approved", "rejected", "expired"
    thread_id VARCHAR(500),
    -- LangGraph thread_id for resuming the checkpointed workflow
    checkpoint_ns VARCHAR(255),
    timeout_at TIMESTAMPTZ,
    -- Auto-expire after this time (NULL = no timeout)
    responded_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_approval_queue_user ON agent_approval_queue(user_id, status);
CREATE INDEX idx_approval_queue_timeout ON agent_approval_queue(timeout_at)
    WHERE status = 'pending';

ALTER TABLE agent_approval_queue ENABLE ROW LEVEL SECURITY;
CREATE POLICY approval_queue_policy ON agent_approval_queue
    USING (user_id = current_setting('app.user_id')::UUID);


-- ============================================================
-- SCHEDULED AGENT RUNS
-- ============================================================

CREATE TABLE agent_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    cron_expression VARCHAR(100) NOT NULL,
    -- Standard cron: "0 8 * * 1" = every Monday at 8am
    query_template TEXT NOT NULL,
    -- The query to run, may include {date}, {last_run_date} variables
    is_enabled BOOLEAN DEFAULT true,
    last_run_at TIMESTAMPTZ,
    last_run_status VARCHAR(50),
    next_run_at TIMESTAMPTZ,
    max_retries INTEGER DEFAULT 2,
    timeout_seconds INTEGER DEFAULT 300,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_schedules_profile ON agent_schedules(agent_profile_id);
CREATE INDEX idx_schedules_user ON agent_schedules(user_id);
CREATE INDEX idx_schedules_next_run ON agent_schedules(next_run_at)
    WHERE is_enabled = true;

ALTER TABLE agent_schedules ENABLE ROW LEVEL SECURITY;
CREATE POLICY schedules_policy ON agent_schedules
    USING (user_id = current_setting('app.user_id')::UUID);


-- ============================================================
-- AGENT WORK JOURNAL
-- ============================================================
-- Human-readable activity log written by agents after each execution.
-- Queryable by users ("What have you done today?") via query_journal tool.

CREATE TABLE agent_journal (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    execution_id UUID REFERENCES agent_execution_log(id),
    -- Links to the raw execution log for details

    -- Journal content
    summary TEXT NOT NULL,
    -- Human-readable summary: "Investigated Omidyar Foundation. Pulled 23 FEC
    -- contributions, found 3 officer-contribution links, saved report."
    detail_entries JSONB DEFAULT '[]',
    -- Structured detail: [{"action": "pull_fec_data", "result": "23 contributions",
    --   "entities_found": 12}, {"action": "cross_reference", ...}]

    -- Metadata for querying
    run_context VARCHAR(50),     -- "interactive", "background", "scheduled"
    invoked_by UUID REFERENCES users(id),
    -- Who triggered this run (owner or team member)
    team_id UUID,               -- If invoked in team context
    entities_mentioned JSONB DEFAULT '[]',
    -- Entity names mentioned in this run, for search: ["Omidyar Foundation", "Koch Network"]
    steps_completed INTEGER DEFAULT 0,
    entities_discovered INTEGER DEFAULT 0,
    outputs_produced JSONB DEFAULT '[]',
    -- [{"type": "document", "path": "Research Reports/omidyar_2026.md"},
    --  {"type": "data_workspace_table", "table": "daily_findings", "rows": 2}]
    task_status VARCHAR(50) DEFAULT 'complete',
    -- "complete", "error", "partial", "awaiting_approval"
    error_summary TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_journal_agent ON agent_journal(agent_profile_id, created_at DESC);
CREATE INDEX idx_journal_user ON agent_journal(user_id, created_at DESC);
CREATE INDEX idx_journal_entities ON agent_journal USING GIN (entities_mentioned);
CREATE INDEX idx_journal_status ON agent_journal(agent_profile_id, task_status);

ALTER TABLE agent_journal ENABLE ROW LEVEL SECURITY;
CREATE POLICY journal_policy ON agent_journal
    USING (
        -- Owner sees all entries
        user_id = current_setting('app.user_id')::UUID
        -- Team members see entries from team-context runs and scheduled runs
        OR (team_id IS NOT NULL AND team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.user_id')::UUID
        ))
    );


-- ============================================================
-- AGENT TEAM SHARING
-- ============================================================
-- Explicit sharing records for agents shared with teams.
-- Supplements the team_config JSONB on agent_profiles with a
-- normalized table for efficient team-member lookups.

CREATE TABLE agent_team_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    team_id UUID NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    shared_by UUID NOT NULL REFERENCES users(id),
    -- The agent owner who shared it

    -- Access levels
    team_file_access BOOLEAN DEFAULT false,
    -- Can the agent search/read the team's shared folders?
    team_post_access BOOLEAN DEFAULT false,
    -- Can the agent read/write team conversation threads?
    journal_visibility VARCHAR(50) DEFAULT 'own_and_scheduled',
    -- "own_only": team members see only their own invocation entries
    -- "own_and_scheduled": own entries + scheduled/background entries (default)
    -- "full": all entries visible to all team members

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (agent_profile_id, team_id)
);

CREATE INDEX idx_team_shares_agent ON agent_team_shares(agent_profile_id);
CREATE INDEX idx_team_shares_team ON agent_team_shares(team_id);

ALTER TABLE agent_team_shares ENABLE ROW LEVEL SECURITY;
CREATE POLICY team_shares_policy ON agent_team_shares
    USING (
        -- Agent owner can manage shares
        shared_by = current_setting('app.user_id')::UUID
        -- Team members can see shares for their teams
        OR team_id IN (
            SELECT team_id FROM team_members
            WHERE user_id = current_setting('app.user_id')::UUID
        )
    );


-- ============================================================
-- MONITOR WATERMARKS (for run_context="monitor" agents)
-- ============================================================
-- Persists the last-checked state for each detection step in a
-- monitor workflow. Allows delta detection across polling intervals.

CREATE TABLE agent_monitor_watermarks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_profile_id UUID NOT NULL REFERENCES agent_profiles(id) ON DELETE CASCADE,
    step_name VARCHAR(255) NOT NULL,
    -- Which detection step this watermark belongs to
    watermark_type VARCHAR(50) NOT NULL,
    -- "timestamp", "cursor", "hash", "offset"
    watermark_value TEXT NOT NULL,
    -- The actual watermark value (ISO timestamp, cursor string, hash, etc.)
    metadata JSONB DEFAULT '{}',
    -- Additional context: {folder_id, connector_id, endpoint, entity_types}
    last_checked_at TIMESTAMPTZ DEFAULT NOW(),
    items_found_last INTEGER DEFAULT 0,
    -- How many new items were found last check (0 = suppressed)
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (agent_profile_id, step_name)
);

CREATE INDEX idx_watermarks_agent ON agent_monitor_watermarks(agent_profile_id);

ALTER TABLE agent_monitor_watermarks ENABLE ROW LEVEL SECURITY;
CREATE POLICY watermarks_policy ON agent_monitor_watermarks
    USING (
        agent_profile_id IN (
            SELECT id FROM agent_profiles
            WHERE user_id = current_setting('app.user_id')::UUID
        )
    );
```

---

## 3. API Endpoint Specification

### Agent Profile Endpoints

```
POST   /api/agent-factory/profiles              Create agent profile
GET    /api/agent-factory/profiles              List user's agent profiles
GET    /api/agent-factory/profiles/{id}         Get profile detail
PUT    /api/agent-factory/profiles/{id}         Update profile
DELETE /api/agent-factory/profiles/{id}         Delete profile
POST   /api/agent-factory/profiles/{id}/duplicate  Clone a profile
POST   /api/agent-factory/profiles/{id}/test    Run test query against profile
```

### Connector Endpoints

```
GET    /api/agent-factory/connectors            List available connectors (templates + user's)
GET    /api/agent-factory/connectors/{id}       Get connector detail
POST   /api/agent-factory/connectors            Create custom connector
PUT    /api/agent-factory/connectors/{id}       Update connector
DELETE /api/agent-factory/connectors/{id}       Delete user's connector
POST   /api/agent-factory/connectors/{id}/test  Test connector endpoint
POST   /api/agent-factory/connectors/{id}/validate  Validate connector YAML
GET    /api/agent-factory/connectors/templates  List system connector templates
POST   /api/agent-factory/connectors/from-template/{template_id}  Create from template
```

### Playbook Endpoints

```
GET    /api/agent-factory/playbooks             List available playbooks
GET    /api/agent-factory/playbooks/{id}        Get playbook detail
POST   /api/agent-factory/playbooks             Create playbook
PUT    /api/agent-factory/playbooks/{id}        Update playbook
DELETE /api/agent-factory/playbooks/{id}        Delete playbook
POST   /api/agent-factory/playbooks/{id}/test   Test playbook with sample data
POST   /api/agent-factory/playbooks/{id}/validate  Validate playbook YAML
```

### Execution & Discovery Endpoints

```
GET    /api/agent-factory/executions            List execution history
GET    /api/agent-factory/executions/{id}       Get execution detail
GET    /api/agent-factory/executions/{id}/discoveries  Get discoveries from execution
GET    /api/agent-factory/discoveries           List all discoveries (filterable)
GET    /api/agent-factory/discoveries/stats     Discovery statistics
```

### Entity Resolution Endpoints

```
GET    /api/agent-factory/entity-reviews        List pending entity reviews
POST   /api/agent-factory/entity-reviews/{id}/approve   Approve merge
POST   /api/agent-factory/entity-reviews/{id}/reject    Reject merge
GET    /api/agent-factory/entity-resolution/log         Resolution audit log
POST   /api/agent-factory/entity-resolution/reverse/{id}  Reverse a merge
```

### Approval Queue Endpoints

```
GET    /api/agent-factory/approvals              List pending approvals for current user
GET    /api/agent-factory/approvals/{id}         Get approval detail with preview data
POST   /api/agent-factory/approvals/{id}/approve  Approve and resume workflow
POST   /api/agent-factory/approvals/{id}/reject   Reject and stop/skip workflow
```

### Schedule Endpoints

```
GET    /api/agent-factory/schedules             List schedules
POST   /api/agent-factory/schedules             Create schedule
PUT    /api/agent-factory/schedules/{id}        Update schedule
DELETE /api/agent-factory/schedules/{id}        Delete schedule
POST   /api/agent-factory/schedules/{id}/run-now  Trigger immediate run
```

### Monitor Endpoints

```
POST   /api/agent-factory/monitors                Create monitor (agent_profile_id + monitor_config)
GET    /api/agent-factory/monitors                List user's active monitors
PUT    /api/agent-factory/monitors/{id}           Update monitor config (interval, active_hours)
DELETE /api/agent-factory/monitors/{id}           Delete monitor
POST   /api/agent-factory/monitors/{id}/pause     Pause monitor polling
POST   /api/agent-factory/monitors/{id}/resume    Resume monitor polling
POST   /api/agent-factory/monitors/{id}/run-now   Trigger immediate check (ignores interval)
GET    /api/agent-factory/monitors/{id}/watermarks  Get current watermarks for all detection steps
POST   /api/agent-factory/monitors/{id}/reset-watermarks  Reset watermarks (re-process all items)
```

### Journal Endpoints

```
GET    /api/agent-factory/profiles/{id}/journal            List journal entries (paginated, filterable by date/status/entity)
GET    /api/agent-factory/profiles/{id}/journal/summary    Get summary of recent activity (for sidebar display)
GET    /api/agent-factory/journal/{entry_id}               Get specific journal entry detail
```

### Team Sharing Endpoints

```
POST   /api/agent-factory/profiles/{id}/share              Share agent with a team
DELETE /api/agent-factory/profiles/{id}/share/{team_id}    Unshare from team
PUT    /api/agent-factory/profiles/{id}/share/{team_id}    Update team access level
GET    /api/agent-factory/profiles/{id}/shares             List teams this agent is shared with
GET    /api/agent-factory/teams/{team_id}/agents           List agents shared with this team
```

### @Mention Resolution Endpoint

```
GET    /api/agent-factory/resolve-handle?handle={handle}   Resolve @handle to agent_profile_id
  Response: {
    "agent_profile_id": "uuid",
    "name": "Nonprofit Investigator",
    "icon": "🔍",
    "owner": "user" | "team",
    "team_id": "uuid-or-null"
  }

GET    /api/agent-factory/available-agents                  List all agents available to current user
  Response: [
    {"handle": "nonprofit-investigator", "name": "...", "icon": "🔍", "source": "own"},
    {"handle": "political-tracker", "name": "...", "icon": "👥", "source": "team", "team_name": "Research Unit"}
  ]
  // Used by the @mention autocomplete dropdown in the chat sidebar
```

### Chat Integration

The existing chat/orchestrator endpoint gains awareness of custom agents via **@mention parsing**:

```
POST /api/async/orchestrator/stream
  Body: {
    "message": "@nonprofit-investigator Investigate the Omidyar Foundation",
    "conversation_id": "...",
    // agent_profile_id resolved server-side from @handle in message
  }
```

**@Mention Parsing Flow:**

1. Frontend detects `@handle` at start of message (via autocomplete selection or manual typing)
2. Frontend resolves handle to `agent_profile_id` via `/resolve-handle` endpoint
3. Frontend sends both the original message (with @mention stripped) and the `agent_profile_id`
4. Orchestrator routes to `CustomAgentEngine` when `agent_profile_id` is present
5. Standard skill-based dispatch is skipped entirely — no trigger matching, no intent classification

**Alternative invocation:** The sidebar's [Run] button and scheduled executions pass `agent_profile_id` directly without @mention parsing. The @mention model is for chat-based invocation only.

**Handle uniqueness:** Handles are unique per user (`UNIQUE (user_id, handle)`). When a team member @mentions an agent, the system resolves in this order:
1. User's own agents (by handle)
2. Agents shared with the user's teams (by handle, with team prefix shown in autocomplete)

---

## 4. Connector YAML Specification

### Common Fields (All Connector Types)

```yaml
id: string                    # Unique identifier (kebab-case)
name: string                  # Human-readable name
type: string                  # rest_api | graphql | web_scraper | file_parser | database | rss_feed | existing_tool
version: string               # Semantic version
description: string           # What this connector accesses

connection:                   # Type-specific connection config
  # ... varies by type

endpoints:                    # Named operations
  endpoint_name:
    description: string
    parameters:               # INPUT schema — what this endpoint accepts
      - name: string
        type: string          # string | number | integer | date | boolean | object | array
        description: string
        required: boolean
        default: any

    # ... type-specific fields (method, path, query, extraction, etc.)

    response_mapping:         # How to interpret results
      entity_type: string     # Label for this data type
      fields:                 # output_field: source_path (dot notation for nested)
        local_name: api_field_path
      entity_extraction:      # Auto-extract entities from results
        - field: string       # Which result field
          entity_type: string # PERSON, ORG, LOCATION, etc.
          relationship_type: string  # Optional: OFFICER_OF, FUNDS, etc.

    output_schema:            # OUTPUT schema — what this endpoint produces
      type: string            # "record_set" (array of records), "object", "scalar"
      fields:                 # Fields in each result record
        - name: string        # Field name (matches response_mapping local name)
          type: string        # string | number | integer | date | boolean | string[] | object
          description: string # Human-readable description for UI
      metadata_fields:        # Envelope-level fields (count, has_more, etc.)
        - name: string
          type: string
          description: string
```

**Why output_schema matters:** The `response_mapping` tells the *runtime* how to transform API responses into normalized records. The `output_schema` tells the *Workflow Composer UI* what fields downstream steps can wire to. Both are needed: response_mapping for execution, output_schema for composition.

When a connector is generated from an API with known response structures, the output_schema can be auto-inferred from the response_mapping. For user-created connectors, the Assembly Agent helps derive the output_schema from a sample API call during the test phase.


### Type: `graphql`

```yaml
id: github-repos
name: GitHub Repository Search
type: graphql
version: "1.0"
description: Search GitHub repositories and organizations

connection:
  endpoint: https://api.github.com/graphql
  auth:
    type: bearer_token
    env_var: GITHUB_TOKEN
  rate_limit:
    requests_per_minute: 30
  timeout_seconds: 15

endpoints:
  search_repos:
    description: Search repositories by keyword
    query: |
      query SearchRepos($query: String!, $first: Int) {
        search(query: $query, type: REPOSITORY, first: $first) {
          repositoryCount
          edges {
            node {
              ... on Repository {
                nameWithOwner
                description
                stargazerCount
                primaryLanguage { name }
                owner { login }
              }
            }
          }
        }
      }
    variables:
      - name: query
        type: string
        description: Search query
        required: true
      - name: first
        type: integer
        description: Number of results
        default: 10
    response_mapping:
      entity_type: repository
      results_path: data.search.edges
      fields:
        name: node.nameWithOwner
        description: node.description
        stars: node.stargazerCount
        language: node.primaryLanguage.name
        owner: node.owner.login
      entity_extraction:
        - field: owner
          entity_type: ORG

  get_org_members:
    description: List organization members
    query: |
      query OrgMembers($org: String!, $first: Int) {
        organization(login: $org) {
          membersWithRole(first: $first) {
            edges {
              node { login, name }
              role
            }
          }
        }
      }
    variables:
      - name: org
        type: string
        required: true
      - name: first
        type: integer
        default: 50
    response_mapping:
      entity_type: org_member
      results_path: data.organization.membersWithRole.edges
      fields:
        username: node.login
        name: node.name
        role: role
      entity_extraction:
        - field: name
          entity_type: PERSON
          relationship_type: MEMBER_OF
```

### Type: `file_parser`

```yaml
id: irs-990-xml
name: IRS 990 XML Parser
type: file_parser
version: "1.0"
description: Parse IRS 990 tax filing XML documents

connection:
  source: upload  # "upload", "url", or "directory"
  accepted_formats: [xml]
  max_file_size_mb: 50

endpoints:
  parse_990:
    description: Extract key fields from IRS Form 990
    parser: xml
    root_element: Return/ReturnData/IRS990
    fields:
      - xpath: //BusinessName/BusinessNameLine1Txt
        field: organization_name
        type: string
      - xpath: //EIN
        field: ein
        type: string
      - xpath: //TaxYr
        field: tax_year
        type: string
      - xpath: //GrossReceiptsAmt
        field: gross_receipts
        type: number
      - xpath: //TotalRevenueGrp/TotalRevenueColumnAmt
        field: total_revenue
        type: number
      - xpath: //TotalExpensesGrp/TotalExpensesColumnAmt
        field: total_expenses
        type: number
      - xpath: //TotalAssetsGrp/EOYAmt
        field: total_assets
        type: number
    response_mapping:
      entity_type: nonprofit_filing
      fields:
        organization_name: organization_name
        ein: ein
        tax_year: tax_year
        gross_receipts: gross_receipts
        total_revenue: total_revenue
        total_expenses: total_expenses
        total_assets: total_assets
      entity_extraction:
        - field: organization_name
          entity_type: ORG

  parse_990_officers:
    description: Extract officers and directors from 990
    parser: xml
    root_element: Return/ReturnData/IRS990
    list_element: //OfficerDirectorTrusteeEmplGrp
    fields:
      - xpath: PersonNm
        field: name
        type: string
      - xpath: TitleTxt
        field: title
        type: string
      - xpath: AverageHoursPerWeekRt
        field: hours_per_week
        type: number
      - xpath: ReportableCompFromOrgAmt
        field: compensation
        type: number
    response_mapping:
      entity_type: nonprofit_officer
      fields:
        name: name
        title: title
        hours_per_week: hours_per_week
        compensation: compensation
      entity_extraction:
        - field: name
          entity_type: PERSON
          relationship_type: OFFICER_OF

  parse_990_grants:
    description: Extract grants made (Schedule I)
    parser: xml
    root_element: Return/ReturnData/IRS990ScheduleI
    list_element: //RecipientTable
    fields:
      - xpath: RecipientBusinessName/BusinessNameLine1Txt
        field: recipient_name
        type: string
      - xpath: RecipientEIN
        field: recipient_ein
        type: string
      - xpath: CashGrantAmt
        field: grant_amount
        type: number
      - xpath: PurposeOfGrantTxt
        field: purpose
        type: string
    response_mapping:
      entity_type: grant
      fields:
        recipient_name: recipient_name
        recipient_ein: recipient_ein
        grant_amount: grant_amount
        purpose: purpose
      entity_extraction:
        - field: recipient_name
          entity_type: ORG
          relationship_type: FUNDED_BY
```

### Type: `database`

```yaml
id: external-postgres
name: External PostgreSQL Connection
type: database
version: "1.0"
description: Query an external PostgreSQL database

connection:
  driver: postgresql
  auth:
    type: connection_string
    env_var: EXTERNAL_DB_URL
  # Or individual fields:
  # host: env_var: DB_HOST
  # port: 5432
  # database: env_var: DB_NAME
  # user: env_var: DB_USER
  # password: env_var: DB_PASSWORD
  pool_size: 3
  timeout_seconds: 30
  read_only: true  # Safety: prevent writes to external DB

endpoints:
  query:
    description: Execute a read-only SQL query
    type: sql
    parameters:
      - name: sql
        type: string
        description: SQL query to execute
        required: true
      - name: params
        type: object
        description: Query parameters (for parameterized queries)
        required: false
    max_rows: 1000
    response_mapping:
      entity_type: query_result
      # Fields are dynamic based on query columns
```

### Type: `rss_feed`

```yaml
id: federal-register
name: Federal Register RSS
type: rss_feed
version: "1.0"
description: Monitor Federal Register for new rules and notices

connection:
  feeds:
    - url: https://www.federalregister.gov/documents/search.rss?conditions%5Btype%5D%5B%5D=RULE
      name: rules
      description: New federal rules
    - url: https://www.federalregister.gov/documents/search.rss?conditions%5Btype%5D%5B%5D=NOTICE
      name: notices
      description: Federal notices
  poll_interval_minutes: 60
  max_items_per_poll: 50

endpoints:
  get_recent:
    description: Get recent items from monitored feeds
    parameters:
      - name: feed_name
        type: string
        description: Which feed to query (or "all")
        default: all
      - name: since_hours
        type: integer
        description: Only items from last N hours
        default: 24
    response_mapping:
      entity_type: federal_register_entry
      fields:
        title: title
        summary: description
        published: pubDate
        link: link
        category: category
      entity_extraction:
        - field: title
          entity_type: auto  # Use spaCy NER on full text
```

### Type: `existing_tool`

```yaml
id: bastion-web-search
name: Bastion Web Search (SearXNG)
type: existing_tool
version: "1.0"
description: Wraps Bastion's built-in web search for use in custom agents

connection:
  tool_module: orchestrator.tools.web_tools
  tool_function: search_web_tool

endpoints:
  search:
    description: Search the web using SearXNG
    parameters:
      - name: query
        type: string
        description: Search query
        required: true
      - name: max_results
        type: integer
        description: Maximum results to return
        default: 10
    # No response_mapping needed — uses existing tool's output format
    passthrough: true
```

---

## 5. Playbook YAML Specification

### Full Specification

```yaml
# ============================================================
# PLAYBOOK DEFINITION
# ============================================================

id: string                      # Unique identifier (kebab-case)
name: string                    # Human-readable name
description: string             # What this playbook does
version: string                 # Semantic version
category: string                # "investigation", "monitoring", "analysis", "extraction"
required_connectors: [string]   # Connector IDs or types needed

# ============================================================
# EXECUTION CONFIGURATION
# ============================================================

execution_mode: string          # "deterministic", "llm_augmented", or "hybrid"
  # deterministic: All steps are tool-type. No LLM calls. Pure data pipeline.
  # llm_augmented: LLM receives tools and decides what to call (traditional agent pattern).
  # hybrid: Mix of tool, llm_task, and approval steps. Default if not specified.

run_context: string             # "interactive", "background", "scheduled", or "monitor"
  # interactive: Runs in chat, streams progress. Default.
  # background: Runs outside chat session. Delivers results to output destinations.
  # scheduled: Triggered by cron schedule. Runs as background.
  # monitor: Periodic polling. Runs only when detection steps find changes.

# Monitor configuration (only for run_context: monitor)
monitor_config:
  interval: string              # Polling interval: "5m", "15m", "30m", "1h", "4h", "12h", "24h"
  active_hours:                 # Optional: restrict polling to a time window
    start: string               # "08:00" (HH:MM, 24h format)
    end: string                 # "22:00"
    timezone: string            # "America/New_York", "UTC", etc.
  suppress_if_empty: boolean    # Skip journal/notify when no changes (default: true)

approval_policy: string         # "require" or "auto_approve". Default: "require".
  # require: Approval steps pause and wait for human confirmation.
  # auto_approve: Approval steps are skipped (useful for trusted scheduled/monitor pipelines).

# ============================================================
# TRIGGERS - When should this playbook activate?
# ============================================================

triggers:
  - pattern: string             # Keyword or phrase pattern
    type: string                # "keyword", "regex", "semantic"
    # keyword: exact substring match (case-insensitive)
    # regex: regular expression match
    # semantic: vector similarity match (requires embedding)
    priority: integer           # Higher = preferred when multiple match (default: 0)

# ============================================================
# INPUT VARIABLES - User-provided parameters
# ============================================================

inputs:
  - name: string                # Variable name (referenced as {var_name} in steps)
    type: string                # "string", "number", "date", "entity_name", "list"
    description: string
    required: boolean
    default: any                # Default value if not provided
    extract_from: string        # "query" = auto-extract from user's message
    # For entity_name type, the system uses NER to extract from query

# ============================================================
# STEPS - Sequential execution plan
# ============================================================
#
# Each step has a step_type that determines how it executes:
#
#   tool       — Deterministic tool/connector call. No LLM involved.
#                The step definition IS the execution specification.
#                Data flows between steps via playbook_state[output_key].
#
#   llm_task   — LLM-driven step. Sends accumulated context to an LLM
#                for analysis, classification, synthesis, or tool-use
#                reasoning. Returns structured JSON.
#
#   approval   — Human-in-the-loop gate. Pauses the workflow, shows a
#                preview of data/pending actions, resumes on user
#                confirmation or rejection.
#
# These three step types compose into any workflow pattern:
#   - All tool steps → deterministic pipeline (zero LLM cost)
#   - All llm_task steps → LLM-driven workflow (traditional agent)
#   - Mix of all three → hybrid workflow (most common)

steps:
  - name: string                # Step identifier (used as reference)
    description: string         # What this step does

    # Step type (required — determines execution semantics)
    step_type: string           # "tool", "llm_task", or "approval"

    # ── STEP I/O (all step types except approval) ─────────────
    #
    # inputs: is the PRIMARY mechanism for wiring data between steps.
    # It is a map of input_name → value, where values can be:
    #   - Literal values: 42, "hello", [1,2,3]
    #   - Playbook input variables: "{entity_name}"
    #   - Upstream step output references: "{step_name.field}"
    #   - Nested field access: "{step_name.results[0].name}"
    #
    # The Workflow Composer UI uses the action's I/O contract (see
    # Built-In Action I/O Registry below) to present a dropdown of
    # compatible upstream outputs for each input field.

    inputs:                     # Data dependencies — wired from upstream steps
      input_name: value         # "{upstream_step.field}" or literal

    # params: is for STATIC configuration that is NOT wired from upstream.
    # Think of inputs as "data in" and params as "settings".
    params:                     # Static configuration
      param_name: value         # Literal values only (no upstream wiring)

    # output_key: names this step's output so downstream steps can reference it.
    # The shape of the output is determined by the action's I/O contract
    # (for tool steps) or by output_schema (for llm_task steps).
    output_key: string          # Key to store results under in playbook_state

    # ── TOOL STEPS (step_type: tool) ──────────────────────────
    # Deterministic execution. No LLM. Action + inputs + params fully
    # specify behavior. Output shape is defined by the action's I/O contract.

    action: string
    # Available tool actions (see Built-In Action I/O Registry for contracts):
    #   call_connector        - Call a data source connector endpoint
    #   search_knowledge_graph - Search Neo4j for existing entities/relationships
    #   search_documents      - Search Qdrant for relevant documents
    #   search_web            - Web search via SearXNG
    #   crawl_url             - Crawl a URL via Crawl4AI
    #   extract_entities      - Run entity extraction on data
    #   resolve_entities      - Run entity resolution on extracted entities
    #   cross_reference       - Match entities across multiple datasets
    #   analyze_graph         - Run graph algorithms on subgraph
    #   transform_data        - Data transformation (filter, sort, aggregate)
    #   save_to_workspace     - Save structured data to Data Workspace table
    #   route_output          - Send data to output destinations mid-workflow
    #   parallel              - Execute sub-steps in parallel

    # Action-specific fields (in addition to inputs/params)
    # -- call_connector --
    connector: string           # Connector ID
    endpoint: string            # Endpoint name within connector
    # inputs come from connector's parameter definitions
    # output_schema inherited from connector endpoint definition

    # -- transform_data --
    operations:                 # Transform operations (applied to inputs.data)
      - type: string            # "filter", "sort", "aggregate", "deduplicate", "limit"
        field: string
        operator: string        # For filter: "eq", "gt", "lt", "contains", "in"
        value: any
        direction: string       # For sort: "asc", "desc"

    # -- parallel (sub-steps) --
    parallel_steps:
      - name: string
        step_type: string       # Each sub-step also has a step_type
        action: string
        inputs: {}
        params: {}
        output_key: string

    # ── LLM TASK STEPS (step_type: llm_task) ──────────────────
    # LLM-driven execution. Sends context to LLM, returns structured output.
    # LLM task steps MUST declare output_schema so downstream steps
    # know what fields are available for wiring.

    action: string
    # Available llm_task actions:
    #   synthesize_report     - LLM synthesis of collected data into report
    #   llm_analyze           - LLM analysis/classification of data
    #   research_with_tools   - LLM receives tools and decides what to call

    # -- synthesize_report / llm_analyze --
    # inputs: wired from upstream steps (the data to analyze/synthesize)
    # params:
    #   template: string        # Report template name
    #   instructions: string    # LLM instructions

    output_schema:              # REQUIRED for llm_task — declares output shape
      type: string              # "object", "record_set", "scalar"
      fields:
        - name: string
          type: string          # string | number | integer | date | boolean | string[] | object
          description: string

    # -- research_with_tools --
    # params:
    #   tools: [string]         # Tool names the LLM can call
    #   system_prompt: string   # Research instructions
    #   max_rounds: integer     # Maximum tool-calling rounds (default: 3)

    # ── APPROVAL STEPS (step_type: approval) ──────────────────
    # Human-in-the-loop gate. Pauses workflow for user confirmation.
    # Approval steps have no inputs/outputs — they gate the flow.

    preview_from: string        # Step output_key to show as preview
    preview_limit: integer      # Max items to show in preview (default: 10)
    prompt: string              # Question to ask the user
    timeout_minutes: integer    # Auto-skip after N minutes (default: none)
    on_reject: string           # "stop" or "skip" (default: "stop")

    # ── COMMON FIELDS (all step types) ────────────────────────

    condition: string           # Skip if condition is false
    # Condition syntax:
    #   "{step_name.count} > 0"
    #   "{input_var} is defined"
    on_error: string            # "skip" (continue), "stop" (halt playbook), "retry" (retry once)
    timeout_seconds: integer    # Per-step timeout (for tool and llm_task steps)
    retry_count: integer        # Number of retries on transient errors

# ============================================================
# OUTPUT - How to format and route results
# ============================================================

output:
  format: string                # "markdown", "json", "csv", "structured"
  template: string              # Named template for formatting
  sections:                     # For structured reports
    - title: string
      source: string            # Step output_key
      format: string            # Override format for this section
  auto_enrich_graph: boolean    # Feed entities to Neo4j
  save_to: string               # "document", "workspace", "both", "none"
  summary_prompt: string        # LLM instructions for generating chat summary
```

### Variable Interpolation

The `inputs` field on each step uses variable interpolation to wire upstream data.
Variables can also appear in `params`, `condition`, and `prompt` fields.

```yaml
# Playbook input variables (from user query or explicit input)
inputs:
  query: "{entity_name}"              # From playbook inputs section
  since: "{start_date}"

# Upstream step output references
inputs:
  data: "{pull_990_data.results}"     # The results array from step "pull_990_data"
  committee_id: "{pull_990_data.results[0].ein}"  # Nested field access
  officer_names: "{extract_officers.entities[*].name}"  # Array projection

# Built-in runtime variables (available in any field)
inputs:
  since: "{today}"
  range_start: "{30_days_ago}"
  run_id: "{execution_id}"
  last_run: "{last_execution_date}"

# Nested access with dot notation and array indexing
inputs:
  name: "{cross_ref.matches[0].canonical_name}"
  all_donors: "{fec_data.results[*].donor_name}"
```

**inputs vs params:** Use `inputs` for data dependencies (values wired from upstream steps or playbook variables). Use `params` for static configuration that doesn't come from upstream (e.g., `entity_types: [PERSON]`, `match_strategy: fuzzy_name`). The Workflow Composer treats `inputs` as wireable connections and `params` as form fields.

### Condition Syntax

```yaml
# Check if previous step produced results
condition: "{pull_990_data.count} > 0"

# Check if input was provided
condition: "{committee_id} is defined"

# Compound conditions
condition: "{pull_990_data.count} > 0 AND {entity_name} is defined"

# Check result field value
condition: "{initial_search.entities[0].type} == 'ORG'"
```

### Built-In Action I/O Registry

Every built-in action has a typed I/O contract that the Workflow Composer uses to present wireable inputs and available outputs. These contracts are registered centrally in `action_io_registry.py` and are also available via API for the frontend.

```yaml
# ============================================================
# ACTION I/O CONTRACTS
# ============================================================
# Each action defines:
#   inputs:  Named parameters the action accepts (with types)
#   params:  Static configuration fields (not wired from upstream)
#   outputs: Named fields the action produces (with types)
#
# The Workflow Composer uses these contracts to:
#   1. Show available input slots when configuring a step
#   2. Offer a dropdown of type-compatible upstream outputs for each input
#   3. Show available output fields for downstream steps to reference
#   4. Validate that all required inputs are wired at save time

# ── CONNECTOR ACTIONS ─────────────────────────────────────────

call_connector:
  description: Call a data source connector endpoint
  inputs:
    # Dynamically determined by the connector endpoint's `parameters` list.
    # Each endpoint parameter becomes a wireable input slot.
    # Example for fec-contributions/search_contributions:
    #   contributor_name: string (optional)
    #   min_date: date (optional)
    #   min_amount: number (optional)
  params:
    connector: string           # Connector ID (required)
    endpoint: string            # Endpoint name (required)
  outputs:
    # Determined by the connector endpoint's `output_schema`.
    # Example for fec-contributions/search_contributions:
    #   results: record[]  (with fields: donor_name, amount, date, ...)
    #   count: integer
    #   has_more: boolean

# ── KNOWLEDGE GRAPH ACTIONS ───────────────────────────────────

search_knowledge_graph:
  description: Search Neo4j for existing entities and relationships
  inputs:
    entity_name: string         # Entity to search for (optional)
    entity_names: string[]      # Multiple entities (optional)
  params:
    entity_types: string[]      # Filter by type [PERSON, ORG, ...]
    relationship_types: string[] # Filter relationships
    include_related: boolean    # Include N-hop neighbors
    max_hops: integer           # Neighbor depth (default: 2)
  outputs:
    entities:                   # type: entity[]
      fields: [name, type, subtype, aliases, external_ids, confidence]
    relationships:              # type: relationship[]
      fields: [source, target, type, properties, confidence]
    count: integer

# ── DOCUMENT & WEB SEARCH ────────────────────────────────────

search_documents:
  description: Semantic search over user's document library (Qdrant)
  inputs:
    query: string               # Search query (required)
  params:
    max_results: integer        # Default: 10
    folder_id: string           # Restrict to folder
    file_types: string[]        # Filter by type
  outputs:
    documents:                  # type: document[]
      fields: [document_id, title, snippet, score, folder_path]
    count: integer

search_web:
  description: Web search via SearXNG
  inputs:
    query: string               # Search query (required)
  params:
    max_results: integer        # Default: 10
    categories: string[]        # SearXNG categories
  outputs:
    results:                    # type: web_result[]
      fields: [title, url, snippet, source]
    count: integer

crawl_url:
  description: Fetch and extract content from a URL
  inputs:
    url: string                 # URL to crawl (required)
  params:
    extract_mode: string        # "text", "markdown", "structured"
  outputs:
    content: string             # Extracted page content
    title: string
    metadata: object            # Page metadata (author, date, etc.)
    links: string[]             # Extracted links

# ── ENTITY OPERATIONS ─────────────────────────────────────────

extract_entities:
  description: Extract named entities from data records
  inputs:
    data: record[]              # Records to process (required)
  params:
    entity_types: string[]      # Types to extract [PERSON, ORG, ...]  ("auto" for all)
    relationship_type: string   # Default relationship to assign
  outputs:
    entities:                   # type: entity[]
      fields: [name, type, source_field, source_record_index, confidence]
    count: integer

resolve_entities:
  description: Resolve entity duplicates and merge aliases
  inputs:
    entities: entity[]          # Entities to resolve (required)
  params:
    strategy: string            # "exact", "fuzzy", "ml"  (default: "fuzzy")
    threshold: number           # Minimum similarity (0-1, default: 0.85)
  outputs:
    resolved:                   # type: entity[]
      fields: [canonical_name, type, aliases, merge_count, confidence]
    merge_count: integer
    ambiguous: entity[]         # Entities that need human review

cross_reference:
  description: Match entities across two or more datasets
  inputs:
    dataset_a: record[]         # First dataset (required)
    dataset_b: record[]         # Second dataset (required)
  params:
    match_strategy: string      # "exact", "fuzzy_name", "external_id"
    match_threshold: number     # Minimum similarity (0-1, default: 0.8)
  outputs:
    matches:                    # type: match[]
      fields: [entity_a, entity_b, match_score, match_method]
    unmatched_a: record[]       # Records from A with no match
    unmatched_b: record[]       # Records from B with no match
    count: integer

# ── GRAPH ANALYSIS ────────────────────────────────────────────

analyze_graph:
  description: Run graph algorithms on entity subgraph
  inputs:
    entities: entity[]          # Starting entities (optional — omit for full graph)
  params:
    algorithm: string           # "pagerank", "community_detection", "betweenness",
                                # "shortest_path", "connected_components"
    scope: string               # "full_graph" or "subgraph_from_entities"
    max_nodes: integer          # Limit for performance
  outputs:
    results:                    # type: varies by algorithm
      pagerank: [entity_name, score]
      community_detection: [entity_name, community_id]
      betweenness: [entity_name, centrality_score]
      shortest_path: [path_nodes[], distance]
      connected_components: [component_id, entities[]]
    node_count: integer
    edge_count: integer

# ── DATA TRANSFORMATION ───────────────────────────────────────

transform_data:
  description: Filter, sort, aggregate, or reshape data
  inputs:
    data: record[]              # Data to transform (required)
  params:
    operations:                 # List of operations (applied sequentially)
      - type: string            # "filter", "sort", "aggregate", "deduplicate", "limit"
        field: string
        operator: string        # For filter: "eq", "gt", "lt", "gte", "lte", "contains", "in"
        value: any
        direction: string       # For sort: "asc", "desc"
  outputs:
    # Same shape as input — filtered/sorted/aggregated records
    data: record[]
    count: integer

# ── PERSISTENCE ───────────────────────────────────────────────

save_to_workspace:
  description: Save structured data to a Data Workspace table
  inputs:
    data: record[]              # Data to save (required)
  params:
    table_name: string          # Target table
    workspace_id: string        # Workspace ID (optional — uses default)
    create_if_missing: boolean  # Auto-create table from data shape
    upsert_key: string          # Field to use for upsert (optional)
  outputs:
    rows_inserted: integer
    rows_updated: integer
    table_name: string
    workspace_id: string

route_output:
  description: Send data to output destinations mid-workflow
  inputs:
    data: any                   # Data to route (required)
  params:
    destinations: object[]      # Output destination configs (same as playbook output)
  outputs:
    routed_to: string[]         # List of destination types that received data
    success: boolean

# ── MONITOR DETECTION ──────────────────────────────────────────
# These tools are used in monitor-mode workflows (run_context: monitor).
# Each detection tool compares current state against a persisted watermark
# and returns only items that are new since the last check.
# The watermark is automatically persisted after successful processing.

detect_new_files:
  description: Check a folder for files added or modified since last check
  inputs:
    folder_id: string           # Folder to watch (required)
  params:
    file_types: string[]        # Filter by extension (optional)
    include_subfolders: boolean # Recurse into subfolders (default: false)
  outputs:
    files:                      # type: file_ref[]
      fields: [file_id, filename, file_type, size_bytes, created_at, modified_at, folder_path]
    count: integer
    watermark: string           # New watermark to persist (timestamp cursor)
    previous_watermark: string  # What was checked last time (for debugging)
  watermark_type: timestamp

detect_folder_changes:
  description: Detect any changes in a folder (adds, edits, deletes) since last check
  inputs:
    folder_id: string           # Folder to watch (required)
  params:
    include_subfolders: boolean # Recurse (default: false)
    change_types: string[]      # Filter: ["added", "modified", "deleted"] (default: all)
  outputs:
    changes:                    # type: change_event[]
      fields: [file_id, filename, change_type, old_hash, new_hash, timestamp]
    added_count: integer
    modified_count: integer
    deleted_count: integer
    watermark: string           # Folder state hash
  watermark_type: hash

detect_new_data:
  description: Check a connector endpoint for new records since last check
  inputs:
    connector: string           # Connector ID (required)
    endpoint: string            # Endpoint name (required)
  params:
    date_field: string          # Which response field to use as cursor (default: auto-detect)
    extra_params: object        # Additional endpoint parameters
  outputs:
    # Inherits output_schema from the connector endpoint definition
    # Plus:
    count: integer
    watermark: string           # Date cursor or API pagination cursor
  watermark_type: cursor

detect_new_team_posts:
  description: Check team threads for new messages since last check
  inputs:
    team_id: string             # Team to monitor (required)
  params:
    thread_ids: string[]        # Specific threads (optional — omit for all)
    min_length: integer         # Ignore short messages (optional)
  outputs:
    posts:                      # type: post[]
      fields: [post_id, thread_id, author, content, created_at, reply_count]
    count: integer
    watermark: string           # Post timestamp cursor
  watermark_type: timestamp
  requires: team_post_access

detect_new_entities:
  description: Check knowledge graph for new entities matching criteria since last check
  inputs:
    entity_types: string[]      # Entity types to watch (required)
  params:
    min_confidence: number      # Minimum confidence (default: 0.5)
    source_filter: string       # Only entities from specific source (optional)
  outputs:
    entities:                   # type: entity[]
      fields: [name, type, subtype, confidence, source, created_at]
    count: integer
    watermark: string           # Entity creation timestamp cursor
  watermark_type: timestamp

set_monitor_watermark:
  description: Explicitly set/update a monitor watermark (called at end of successful processing)
  inputs:
    watermark: string           # New watermark value (required)
  params:
    # step_name is auto-populated from the calling step
  outputs:
    success: boolean
    previous_watermark: string

get_monitor_watermark:
  description: Get the current watermark for a detection step
  inputs:
    step_name: string           # Which step's watermark to retrieve (required)
  outputs:
    watermark: string
    last_checked_at: date
    items_found_last: integer

# ── AGENT JOURNAL ─────────────────────────────────────────────
# These tools are automatically available to all custom agents.
# write_journal_entry is called by the workflow engine after execution;
# query_journal is available for user queries about agent activity.

write_journal_entry:
  description: Write a structured journal entry summarizing work done
  inputs:
    summary: string             # Human-readable summary (required)
    detail_entries: object[]    # Structured step-by-step details (optional)
    entities_mentioned: string[] # Entity names for search indexing (optional)
  params:
    # Automatically populated by workflow engine:
    # agent_profile_id, execution_id, run_context, invoked_by, team_id,
    # steps_completed, entities_discovered, outputs_produced, task_status
  outputs:
    journal_entry_id: string
    success: boolean

query_journal:
  description: Search the agent's work journal by date, keyword, entity, or status
  inputs:
    query: string               # Natural language query (optional - "What did you do today?")
    date_from: date             # Filter by start date (optional)
    date_to: date               # Filter by end date (optional)
    entity_name: string         # Filter by entity mentioned (optional)
    status: string              # Filter by status (optional)
  params:
    limit: integer              # Max entries to return (default: 20)
  outputs:
    entries:                    # type: journal_entry[]
      fields: [summary, created_at, run_context, steps_completed,
               entities_discovered, outputs_produced, task_status]
    count: integer
    date_range: object          # {from, to} of matching entries

# ── TEAM INTERACTION ──────────────────────────────────────────
# These tools are CONDITIONALLY available — only when the agent's
# team_config grants the appropriate access level.
# Assembled dynamically by CustomAgentRunner based on team_config.

search_team_files:
  description: Search documents in a team's shared folders
  inputs:
    query: string               # Search query (required)
    team_id: string             # Team to search (required — from team_config)
  params:
    max_results: integer        # Default: 10
    folder_path: string         # Restrict to subfolder (optional)
    file_types: string[]        # Filter by type (optional)
  outputs:
    documents:                  # type: document[]
      fields: [document_id, title, snippet, score, folder_path, author]
    count: integer
  requires: team_file_access

read_team_file:
  description: Read a specific document from team shared folders
  inputs:
    document_id: string         # Document to read (required)
    team_id: string             # Team context (required)
  params:
    max_length: integer         # Truncate if too long (optional)
  outputs:
    content: string
    title: string
    metadata: object            # Author, last_modified, folder_path
  requires: team_file_access

search_team_posts:
  description: Search team conversation threads by keyword, date, or author
  inputs:
    query: string               # Search query (required)
    team_id: string             # Team to search (required)
  params:
    date_from: date             # Filter by start date (optional)
    date_to: date               # Filter by end date (optional)
    author: string              # Filter by author (optional)
    max_results: integer        # Default: 20
  outputs:
    posts:                      # type: post[]
      fields: [post_id, thread_id, author, content_preview, created_at, reply_count]
    count: integer
  requires: team_post_access

write_team_post:
  description: Write a message to a team conversation thread
  inputs:
    content: string             # Message content (required)
    team_id: string             # Team to post to (required)
  params:
    thread_id: string           # Reply to existing thread (optional — creates new if omitted)
    format: string              # "markdown" (default), "plain"
  outputs:
    post_id: string
    thread_id: string
    success: boolean
  requires: team_post_access

summarize_team_thread:
  description: Generate a summary of a team conversation thread
  inputs:
    thread_id: string           # Thread to summarize (required)
    team_id: string             # Team context (required)
  params:
    max_length: integer         # Summary length limit (optional)
    focus: string               # What to focus on: "decisions", "action_items", "key_points" (optional)
  outputs:
    summary: string
    key_points: string[]
    action_items: string[]
    participants: string[]
  requires: team_post_access

# ── NOTIFICATIONS & MESSAGING ────────────────────────────────

send_notification:
  inputs:
    message: string
  params:
    title: string
    priority: string            # "low", "normal", "high", "urgent" (default: "normal")
    action_url: string
    action_label: string
  outputs:
    notification_id: string
    success: boolean

send_channel_message:
  inputs:
    message: string
  params:
    channel: string             # "telegram", "discord", "slack", "default"
    chat_id: string
    format: string              # "markdown", "plain", "html"
    silent: boolean
  outputs:
    message_id: string
    channel: string
    success: boolean

broadcast_to_team:
  inputs:
    message: string
    team_id: string
  params:
    channel: string             # Force specific channel or "preferred"
  outputs:
    delivered_to: integer
    failures: integer
    success: boolean
  requires: team_post_access

# ── TASK MANAGEMENT ──────────────────────────────────────────

create_todo:
  inputs:
    title: string
    body: string
  params:
    priority: string            # "A", "B", "C"
    state: string               # "TODO", "NEXT", "WAITING" (default: "TODO")
    deadline: date
    scheduled: date
    tags: string[]
    target_file: string
  outputs:
    success: boolean
    todo_id: string
    file_path: string

list_todos:
  inputs:
    state: string               # "TODO", "DONE", "all"
  params:
    tags: string[]
    deadline_before: date
    file_id: string
    limit: integer
  outputs:
    todos: todo[]
    count: integer

update_todo:
  inputs:
    todo_id: string
  params:
    state: string
    priority: string
    deadline: date
    scheduled: date
    add_tags: string[]
    remove_tags: string[]
    add_note: string
  outputs:
    success: boolean
    updated_fields: string[]

complete_todo:
  inputs:
    todo_id: string
  params:
    completion_note: string
  outputs:
    success: boolean
    completed_at: date

search_todos:
  inputs:
    query: string
  params:
    states: string[]
    include_done: boolean
  outputs:
    todos: todo[]
    count: integer

# ── FILE OPERATIONS (GRANULAR) ───────────────────────────────

read_file:
  inputs:
    document_id: string
    path: string
    scope: string               # "my_docs", "team_docs", "global_docs"
    team_id: string
  params:
    max_length: integer
    include_frontmatter: boolean
  outputs:
    content: string
    title: string
    file_type: string
    frontmatter: object
    metadata: object

find_file_by_name:
  inputs:
    name: string
    scope: string               # "my_docs", "team_docs", "global_docs", "all"
    team_id: string
  params:
    exact_match: boolean
    file_types: string[]
    folder_id: string
  outputs:
    files: file_ref[]
    count: integer

find_file_by_vectors:
  inputs:
    query: string
    scope: string
    team_id: string
  params:
    max_results: integer
    file_types: string[]
    folder_id: string
    min_score: number
  outputs:
    documents: document[]
    count: integer

patch_file:
  inputs:
    document_id: string
    edits: object[]             # [{operation, target, content}]
  params:
    create_if_missing: boolean
  outputs:
    success: boolean
    operations_applied: integer
    new_content_length: integer

delete_file:
  inputs:
    document_id: string
  params:
    confirm: boolean
  outputs:
    success: boolean
    deleted_title: string

move_file:
  inputs:
    document_id: string
    target_folder_id: string
  outputs:
    success: boolean
    new_folder_path: string

copy_file:
  inputs:
    document_id: string
    target_folder_id: string
  params:
    new_title: string
  outputs:
    new_document_id: string
    new_title: string
    new_folder_path: string
```

### Step Linking Validation

At save time (and in real-time in the Workflow Composer), the system validates that all step wiring is correct:

**Validation Rules:**

1. **Required inputs resolved** — Every required input field (per the action's I/O contract) must be wired to either an upstream step output, a playbook input variable, a literal value, or a default.

2. **Type compatibility** — The referenced upstream field's type must be compatible with the input's expected type. Compatibility follows these rules:
   - Exact match: `string` → `string`, `number` → `number`
   - Coercion: `integer` → `number`, `string` → `string[]` (single element)
   - Record access: `record_set` → `record[]` (via `.results`), `record` → field type (via `.field_name`)
   - Array projection: `record[].field` → `type_of_field[]`

3. **Reference existence** — `{step_name.field}` must reference a step that (a) exists, (b) appears before the current step in the sequence, and (c) has the named field in its output schema.

4. **Cycle detection** — No circular dependencies between steps.

5. **Connector binding** — `call_connector` steps must reference a connector that is bound to the agent profile.

6. **LLM task output_schema** — Every `llm_task` step must declare an `output_schema` so downstream steps can wire to its outputs.

**Validation Response:**

```python
class StepValidationResult(BaseModel):
    step_name: str
    is_valid: bool
    errors: List[StepValidationError]
    warnings: List[StepValidationWarning]

class StepValidationError(BaseModel):
    field: str                    # "inputs.data", "connector", etc.
    error_type: str               # "unresolved_reference", "type_mismatch",
                                  # "missing_required", "nonexistent_step"
    message: str                  # Human-readable error
    suggested_fix: Optional[str]  # Assembly Agent suggestion

class StepValidationWarning(BaseModel):
    field: str
    warning_type: str             # "loose_type_coercion", "unused_output", "long_chain"
    message: str
```

**API Endpoints for I/O Contracts:**

```
GET  /api/agent-factory/actions                    List all available actions with I/O contracts
GET  /api/agent-factory/actions/{action_name}      Get specific action's I/O contract
GET  /api/agent-factory/connectors/{id}/schema     Get connector endpoints with output_schemas
POST /api/agent-factory/playbooks/validate         Validate playbook wiring (returns validation results)
GET  /api/agent-factory/playbooks/{id}/graph       Get step dependency graph with types
```

---

## 6. Neo4j Graph Schema

### Enhanced Entity Model

```cypher
-- ============================================================
-- NODE LABELS
-- ============================================================

-- Core entity types
(:Entity {
    name: String,               -- Display name
    canonical_name: String,     -- Normalized canonical name
    type: String,               -- PERSON, ORG, LOCATION, EVENT, PRODUCT, etc.
    subtype: String,            -- Optional: NONPROFIT, PAC, CORPORATION, FOUNDATION, etc.
    aliases: [String],          -- Known alternative names
    external_ids: Map,          -- {fec_id: "C00523514", ein: "04-3568784", cik: "0001234567"}
    confidence: Float,          -- Confidence in entity identity (0-1)
    first_seen: DateTime,       -- When first discovered
    last_seen: DateTime,        -- Most recent reference
    source_count: Integer,      -- Number of distinct sources
    user_id: String,            -- Owning user (for namespace isolation)
    created_at: DateTime,
    updated_at: DateTime
})

-- Document nodes (existing, enhanced)
(:Document {
    id: String,                 -- Bastion document_id
    title: String,
    doc_type: String,
    user_id: String,
    created_at: DateTime
})

-- Data source provenance
(:DataSource {
    connector_id: String,
    connector_name: String,
    endpoint: String,
    query: String,              -- The query that produced this data
    execution_id: String,
    retrieved_at: DateTime,
    user_id: String
})

-- Cluster/Community nodes (auto-generated by graph algorithms)
(:EntityCluster {
    cluster_id: Integer,
    algorithm: String,          -- "louvain", "label_propagation"
    member_count: Integer,
    label: String,              -- Auto-generated or user-assigned label
    computed_at: DateTime,
    user_id: String
})


-- ============================================================
-- RELATIONSHIP TYPES
-- ============================================================

-- Existing relationships (enhanced with properties)
(Entity)-[:MENTIONED_IN {
    count: Integer,             -- How many times mentioned
    context: String,            -- Surrounding text snippet
    section: String,            -- Document section
    confidence: Float
}]->(Document)

-- Funding relationships
(Entity)-[:FUNDS {
    amount: Float,
    currency: String,
    date: Date,
    date_range_start: Date,
    date_range_end: Date,
    filing_type: String,        -- "990", "fec", "sec"
    source_connector: String,
    source_execution_id: String,
    confidence: Float
}]->(Entity)

(Entity)-[:FUNDED_BY {
    -- Same properties as FUNDS (inverse direction)
}]->(Entity)

-- Organizational relationships
(Entity)-[:OFFICER_OF {
    title: String,              -- "President", "Board Member", "Treasurer"
    start_date: Date,
    end_date: Date,
    compensation: Float,
    hours_per_week: Float,
    source_connector: String,
    confidence: Float
}]->(Entity)

(Entity)-[:BOARD_MEMBER_OF {
    start_date: Date,
    end_date: Date,
    role: String,               -- "Chair", "Vice Chair", "Member"
    source_connector: String,
    confidence: Float
}]->(Entity)

(Entity)-[:CONTROLS {
    ownership_percentage: Float,
    control_type: String,       -- "majority_owner", "sole_owner", "subsidiary"
    source_connector: String,
    confidence: Float
}]->(Entity)

(Entity)-[:SUBSIDIARY_OF {
    -- Same as CONTROLS but inverse
}]->(Entity)

(Entity)-[:RELATED_TO {
    relationship_type: String,  -- "family", "business_partner", "affiliated"
    description: String,
    source_connector: String,
    confidence: Float
}]->(Entity)

-- Political relationships
(Entity)-[:DONATED_TO {
    amount: Float,
    date: Date,
    filing_id: String,
    employer: String,
    occupation: String,
    source_connector: String,
    confidence: Float
}]->(Entity)

-- Location relationships
(Entity)-[:LOCATED_IN {
    address: String,
    is_registered_address: Boolean,
    is_headquarters: Boolean,
    source_connector: String,
    confidence: Float
}]->(Entity)

(Entity)-[:OPERATES_IN]->(Entity)  -- Geographic areas of operation

-- Employment
(Entity)-[:WORKS_FOR {
    title: String,
    start_date: Date,
    end_date: Date,
    source_connector: String,
    confidence: Float
}]->(Entity)

-- Provenance
(Entity)-[:DISCOVERED_BY]->(DataSource)
(Relationship)-[:SOURCED_FROM]->(DataSource)

-- Cluster membership
(Entity)-[:MEMBER_OF_CLUSTER]->(EntityCluster)


-- ============================================================
-- INDEXES AND CONSTRAINTS
-- ============================================================

CREATE CONSTRAINT entity_canonical IF NOT EXISTS
    FOR (e:Entity) REQUIRE (e.canonical_name, e.user_id) IS UNIQUE;

CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_subtype IF NOT EXISTS FOR (e:Entity) ON (e.subtype);
CREATE INDEX entity_user IF NOT EXISTS FOR (e:Entity) ON (e.user_id);
CREATE INDEX entity_external_ids IF NOT EXISTS FOR (e:Entity) ON (e.external_ids);

CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id);
CREATE INDEX document_user IF NOT EXISTS FOR (d:Document) ON (d.user_id);

CREATE INDEX datasource_execution IF NOT EXISTS FOR (ds:DataSource) ON (ds.execution_id);
CREATE INDEX datasource_connector IF NOT EXISTS FOR (ds:DataSource) ON (ds.connector_id);

-- Full-text search on entity names and aliases
CALL db.index.fulltext.createNodeIndex(
    'entityFullText',
    ['Entity'],
    ['name', 'canonical_name'],
    {analyzer: 'standard-folding'}  -- Case-insensitive, accent-folding
);
```

---

## 7. Connector Template Library

Priority data sources for pre-built connector templates, listed by implementation priority:

### Tier 1: Government & Nonprofit (Core for investigative use cases)

| Connector | API | Auth | Rate Limits | Key Endpoints |
|-----------|-----|------|-------------|---------------|
| **FEC Campaign Finance** | `api.open.fec.gov/v1` | API key (free) | 1000/day, 10/sec | contributions, committees, candidates, filings |
| **ProPublica Nonprofit Explorer** | `projects.propublica.org/nonprofits/api/v2` | API key (free) | 5/sec | org search, filing details, officers |
| **SEC EDGAR** | `efts.sec.gov/LATEST` | None (public) | 10/sec | full-text search, company filings, ownership |
| **OpenSecrets (CRP)** | `www.opensecrets.org/api` | API key (free) | Varies | donors, recipients, lobbying, PACs |
| **USASpending** | `api.usaspending.gov/api/v2` | None (public) | 50/sec | federal awards, contracts, grants |

### Tier 2: Corporate & Business

| Connector | API | Auth | Rate Limits | Key Endpoints |
|-----------|-----|------|-------------|---------------|
| **OpenCorporates** | `api.opencorporates.com/v0.4` | API key (free tier) | 500/month free | company search, officers, filings |
| **Wikidata / SPARQL** | `query.wikidata.org/sparql` | None (public) | 5/sec | entity lookup, relationships, IDs |
| **DUNS / D&B** | Varies | API key (paid) | Varies | company profiles, hierarchies |

### Tier 3: Monitoring & News

| Connector | API | Auth | Rate Limits | Key Endpoints |
|-----------|-----|------|-------------|---------------|
| **Federal Register** | RSS + `federalregister.gov/api/v1` | None (public) | 10/sec | rules, notices, executive orders |
| **Congress.gov** | `api.congress.gov/v3` | API key (free) | 5000/hr | bills, members, votes, committees |
| **RECAP (Court Records)** | `www.courtlistener.com/api/rest/v4` | API key (free) | Varies | opinions, dockets, parties |

### Tier 4: State-Level (Web Scraper Connectors)

| Connector | Source | Method | Key Data |
|-----------|--------|--------|----------|
| **State Corp Registries** | Per-state websites | Web scraper | Business filings, officers, registered agents |
| **State Campaign Finance** | Per-state websites | Web scraper / API | State-level contributions |
| **State Lobbying Disclosures** | Per-state websites | Web scraper | Lobbyist registrations, expenditures |

---

## 8. Pre-Built Agent Profile Templates

### Nonprofit Investigator

```yaml
name: Nonprofit Investigator
description: Investigate nonprofit organizations — finances, officers, political connections, and funding networks
icon: "🔍"

data_sources:
  - connector: propublica-nonprofit
  - connector: fec-contributions
  - connector: irs-990-xml
  - connector: opensecrets

skills:
  - skill: research (built-in)
  - playbook: nonprofit-investigation
  - playbook: officer-cross-reference

knowledge_config:
  auto_enrich: true
  entity_resolution: true
  graph_namespaces: ["nonprofit_research"]

output_config:
  auto_enrich_graph: true
  destinations:
    - type: chat
      format: markdown
    - type: document
      format: markdown
      config:
        folder_id: nonprofit-investigations
        filename_template: "nonprofit_{entity_name}_{date}.md"
        create_if_missing: true
    - type: data_workspace_table
      config:
        workspace_id: research
        table_name: nonprofit_filings
        create_if_missing: true
    - type: knowledge_graph

system_prompt_additions: |
  You are a nonprofit research specialist. When investigating an organization:
  1. Always check 990 filings for officer lists and grant recipients
  2. Cross-reference officers with FEC contribution records
  3. Look for connections between the nonprofit and political committees
  4. Track the money: who funds this org, and who does this org fund?
  5. Note any officers who appear across multiple related organizations
  6. Flag organizations that appear to be pass-throughs (high betweenness, few public activities)
```

### Political Finance Tracker

```yaml
name: Political Finance Tracker
description: Track campaign contributions, PAC activity, and political spending
icon: "🏛️"

data_sources:
  - connector: fec-contributions
  - connector: opensecrets
  - connector: congress-gov

skills:
  - skill: research (built-in)
  - playbook: contribution-analysis
  - playbook: pac-network-mapping

knowledge_config:
  auto_enrich: true
  entity_resolution: true

output_config:
  auto_enrich_graph: true
  destinations:
    - type: chat
    - type: data_workspace_table
      config:
        table_name: contributions
        create_if_missing: true
        schema_mapping:
          donor_name: TEXT
          donor_employer: TEXT
          recipient: TEXT
          amount: REAL
          date: TIMESTAMP
          type: TEXT
    - type: knowledge_graph

system_prompt_additions: |
  You are a campaign finance analyst. When researching political money:
  1. Track both individual contributions and PAC/committee donations
  2. Identify employer patterns (many employees of same company donating = bundling)
  3. Note the maximum contribution limits and flag unusually structured donations
  4. Look for connections between corporate PACs and individual officers
  5. Map committee-to-committee transfers (these often obscure the original source)
```

### Corporate Intelligence Analyst

```yaml
name: Corporate Intelligence Analyst
description: Research corporate structures, ownership, board connections, and regulatory filings
icon: "🏢"

data_sources:
  - connector: sec-edgar
  - connector: opencorporates
  - connector: wikidata

skills:
  - skill: research (built-in)
  - playbook: corporate-structure-analysis
  - playbook: board-interlock-mapping

knowledge_config:
  auto_enrich: true
  entity_resolution: true

output_config:
  auto_enrich_graph: true
  destinations:
    - type: chat
    - type: document
      config:
        folder_id: corporate-research
        filename_template: "corp_{entity_name}_{date}.md"
    - type: knowledge_graph

system_prompt_additions: |
  You are a corporate intelligence analyst. When researching companies:
  1. Map the corporate hierarchy (parent, subsidiaries, affiliates)
  2. Identify board members and look for board interlocks (same person on multiple boards)
  3. Check SEC filings for ownership changes, insider trading, and proxy statements
  4. Cross-reference with nonprofit connections (corporate foundations, board overlap)
  5. Note the registered agent — shared agents often indicate related entities
```

---

## 9. Scheduling & Monitoring

### Celery Integration

Custom agent schedules and background jobs integrate with the existing Celery infrastructure. Scheduled and background workflows use the same `CustomAgentRunner` as interactive workflows — the `run_context` field determines how results are delivered and how approval gates behave.

```python
# backend/services/celery_tasks/agent_factory_tasks.py

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
async def run_scheduled_agent(self, schedule_id: str):
    """Execute a scheduled agent run."""
    schedule = await get_schedule(schedule_id)
    if not schedule or not schedule.is_enabled:
        return

    profile = await get_agent_profile(schedule.agent_profile_id)
    if not profile or not profile.is_active:
        return

    # Render query template
    query = render_query_template(
        schedule.query_template,
        last_run_date=schedule.last_run_at,
        current_date=datetime.now()
    )

    # Execute through the standard orchestrator path
    # run_context is set to "scheduled", which affects:
    #   - No streaming to chat
    #   - Approval gates use notification queue (or auto_approve)
    #   - Results go to configured output destinations
    #   - Completion triggers a notification
    result = await execute_custom_agent(
        agent_profile_id=schedule.agent_profile_id,
        query=query,
        user_id=schedule.user_id,
        metadata={
            "run_context": "scheduled",
            "schedule_id": schedule_id,
            "approval_policy": schedule.approval_policy or "require",
        }
    )

    # Handle approval pauses
    if result.get("task_status") == "awaiting_approval":
        # Workflow is checkpointed. A Celery watcher task monitors
        # the approval queue and resumes workflows when approved.
        return {"status": "awaiting_approval", "schedule_id": schedule_id}

    # Update schedule status
    await update_schedule_status(
        schedule_id,
        last_run_at=datetime.now(),
        last_run_status=result.get("task_status", "unknown")
    )


@celery_app.task(bind=True)
async def resume_approved_workflow(self, approval_id: str):
    """
    Resume a workflow after approval gate is resolved.

    Called by the approval queue watcher when a user approves or
    rejects a pending approval. Loads the checkpointed LangGraph
    state and continues execution.
    """
    approval = await get_approval(approval_id)
    if not approval or approval.status != "approved":
        return

    # Resume the checkpointed workflow
    result = await resume_custom_agent(
        thread_id=approval.thread_id,
        approval_response=approval.status,
        user_id=approval.user_id,
    )

    return result
```

### Monitor Mode (Celery Beat Integration)

Monitor-mode agents use Celery Beat for interval-based polling. Unlike cron jobs (which fire at exact times), monitors run on a repeating interval and are change-aware.

```python
# backend/services/celery_tasks/agent_factory_tasks.py

@celery_app.task(bind=True, max_retries=1, default_retry_delay=30)
async def run_monitor_check(self, monitor_id: str):
    """
    Execute a single monitor polling cycle.

    1. Load monitor config and agent profile
    2. Check active_hours — skip if outside window
    3. Load persisted watermarks for all detection steps
    4. Run the playbook with watermarks injected into state
    5. If no detection step found new items AND suppress_if_empty:
       - Skip downstream steps, don't journal, don't notify
       - Update last_checked_at on watermarks
       - Return early
    6. If changes detected:
       - Execute full playbook (detection → processing → output)
       - Persist new watermarks from detection step outputs
       - Write journal entry
       - Deliver results to configured destinations
    """
    monitor = await get_monitor(monitor_id)
    if not monitor or not monitor.is_active:
        return

    # Active hours check
    if monitor.active_hours and not within_active_hours(monitor.active_hours):
        return {"status": "outside_active_hours"}

    profile = await get_agent_profile(monitor.agent_profile_id)

    # Load existing watermarks
    watermarks = await get_monitor_watermarks(monitor.agent_profile_id)

    result = await execute_custom_agent(
        agent_profile_id=monitor.agent_profile_id,
        query=monitor.query_template or "Monitor check",
        user_id=monitor.user_id,
        metadata={
            "run_context": "monitor",
            "monitor_id": monitor_id,
            "monitor_watermarks": watermarks,
            "suppress_if_empty": monitor.suppress_if_empty,
            "approval_policy": monitor.approval_policy or "auto_approve",
        }
    )

    # Persist updated watermarks
    if result.get("new_watermarks"):
        await save_monitor_watermarks(
            monitor.agent_profile_id, result["new_watermarks"]
        )

    return result
```

**Celery Beat schedule registration:**

When a monitor is created or updated, its interval is registered with Celery Beat dynamically:

```python
# Register monitor with Celery Beat
celery_app.conf.beat_schedule[f"monitor-{monitor_id}"] = {
    "task": "run_monitor_check",
    "schedule": timedelta(minutes=monitor_config["interval_minutes"]),
    "args": [monitor_id],
}
```

**Suppression logic:**

The key differentiator from scheduled jobs: when `suppress_if_empty: true` (the default), the pipeline executor checks whether any detection step returned `count > 0`. If all detection steps found zero new items:
- Downstream steps are **skipped entirely** (no LLM calls, no tool calls)
- No journal entry is written
- No notification is sent
- Only the watermark timestamps are updated
- The execution log records a "suppressed" status (for monitoring dashboard)

This makes monitors extremely cost-efficient — a monitor checking 5 sources every 30 minutes costs almost nothing when nothing has changed (just the detection queries), versus a cron job that would run the full playbook every time.

### Monitoring Dashboard Data

The execution log and discovery log provide data for monitoring:

- **Agent Performance**: Average execution time, success rate, connector call counts
- **Discovery Rate**: Entities and relationships found per execution, per day, per agent
- **Knowledge Graph Growth**: Total entities, relationships, clusters over time
- **Connector Health**: Latency, error rates, rate limit utilization per connector
- **Entity Resolution Stats**: Auto-merges, flagged reviews, user decisions

---

## 10. Error Handling & Resilience

### Connector Failures

| Failure Mode | Handling | User Experience |
|---|---|---|
| API returns 401/403 | Log, mark connector as "auth_failed", notify user | "FEC connector needs re-authentication" |
| API returns 429 (rate limit) | Exponential backoff, retry up to 3x | Transparent delay, falls back to cached data |
| API returns 500+ | Retry once, then skip connector | "FEC API temporarily unavailable, using cached data" |
| Timeout | Retry with doubled timeout, then skip | "FEC API slow, partial results returned" |
| Network error | Retry once, then skip | "Could not reach FEC API" |
| Malformed response | Log, return empty result for endpoint | "Unexpected response from FEC, skipping" |

### Playbook Step Failures

Controlled by `on_error` per step:

- `skip` (default): Log error, continue to next step. Missing output_key returns empty.
- `stop`: Halt playbook, return partial results collected so far.
- `retry`: Retry step once with same parameters. If still fails, behave as `skip`.

### Entity Resolution Failures

- Entity extraction failure: Fall back to basic NER (spaCy) if connector-defined extraction fails
- Resolution timeout: Queue for background resolution, return unresolved entities immediately
- Merge conflict: Never auto-merge when confidence < 0.7; queue for human review

### Output Routing Failures

Each destination is independent. Failure in one destination does not block others:

- Document save fails: Log error, include in routing_report, chat response still delivered
- Data Workspace insert fails: Log error, suggest user check table schema
- Knowledge graph write fails: Queue for retry, log to discovery log with status "pending"
- File export fails: Log error, offer retry link in chat response

### Circuit Breaker Pattern

For external connectors that fail repeatedly:

```python
class ConnectorCircuitBreaker:
    """
    Circuit breaker for external API connectors.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Connector failing, reject requests immediately
    - HALF_OPEN: Testing if connector recovered
    """

    def __init__(self, failure_threshold=5, recovery_timeout_seconds=300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_try_recovery():
                self.state = "HALF_OPEN"
            else:
                raise ConnectorUnavailableError(
                    f"Connector circuit breaker OPEN, retry after "
                    f"{self.recovery_timeout}s"
                )

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise
```

---

## 11. Integration Points

### Existing Files That Change

| File | Change | Reason |
|---|---|---|
| `llm-orchestrator/orchestrator/engines/unified_dispatch.py` | Add `CUSTOM_AGENT` engine type, route when `agent_profile_id` present | Entry point for custom agents |
| `llm-orchestrator/orchestrator/tools/__init__.py` | Export dynamic tool registration functions | Dynamic tools need registration |
| `llm-orchestrator/orchestrator/tools/tool_pack_registry.py` | Add dynamic tool pack loading from DB | Connector-generated tools |
| `llm-orchestrator/orchestrator/skills/skill_registry.py` | Add dynamic skill loading from DB | User-defined playbooks |
| `llm-orchestrator/orchestrator/utils/tool_vector_store.py` | Support vectorizing dynamic tools | Semantic discovery of connector tools |
| `backend/api/async_orchestrator_api.py` | Pass `agent_profile_id` to orchestrator | Frontend → backend → orchestrator |
| `backend/services/knowledge_graph_service.py` | Add typed relationship storage, entity resolution hooks | Graph enrichment |
| `frontend/src/services/apiService.js` | Add Agent Factory API methods | Frontend data access |
| `frontend/src/App.js` | Add Agent Factory route/navigation | UI navigation |
| `docker-compose.yml` | No change (all runs in existing containers) | — |
| `protos/tool_service.proto` | Add connector execution RPC methods | gRPC tool execution |

### New Files to Create

| File | Purpose |
|---|---|
| **Backend** | |
| `backend/api/agent_factory_api.py` | REST endpoints for profiles, connectors, playbooks, schedules |
| `backend/services/agent_factory_service.py` | Business logic for Agent Factory operations |
| `backend/services/entity_resolution_service.py` | Entity resolution pipeline |
| `backend/services/connector_runtime_service.py` | Execute connectors (REST, scraper, etc.) |
| `backend/services/output_router_service.py` | Route results to destinations |
| `backend/sql/migrations/XXX_agent_factory.sql` | Database schema migration |
| **LLM Orchestrator** | |
| `llm-orchestrator/orchestrator/engines/custom_agent_engine.py` | Custom agent execution engine |
| `llm-orchestrator/orchestrator/engines/pipeline_executor.py` | Deterministic step executor (no LLM) |
| `llm-orchestrator/orchestrator/engines/approval_gate_handler.py` | Approval gate handler (interactive + background) |
| `llm-orchestrator/orchestrator/agents/custom_agent_runner.py` | LangGraph agent for executing profiles |
| `llm-orchestrator/orchestrator/agents/assembly_agent.py` | AI assistant for building agents |
| `llm-orchestrator/orchestrator/tools/connector_tools.py` | Dynamic tool generation from connectors |
| `llm-orchestrator/orchestrator/subgraphs/playbook_execution_subgraph.py` | Hybrid playbook step execution |
| `llm-orchestrator/orchestrator/subgraphs/knowledge_enrichment_subgraph.py` | Post-query graph enrichment |
| `llm-orchestrator/orchestrator/utils/connector_parser.py` | Parse and validate connector YAML |
| `llm-orchestrator/orchestrator/utils/playbook_parser.py` | Parse and validate playbook YAML |
| `llm-orchestrator/orchestrator/utils/action_io_registry.py` | Central registry of action I/O contracts |
| `llm-orchestrator/orchestrator/utils/step_link_validator.py` | Validate step wiring (type checks, reference resolution) |
| `llm-orchestrator/orchestrator/tools/journal_tools.py` | `write_journal_entry` and `query_journal` tools |
| `llm-orchestrator/orchestrator/tools/team_tools.py` | `search_team_files`, `read_team_file`, `search_team_posts`, `write_team_post`, `summarize_team_thread` |
| `llm-orchestrator/orchestrator/tools/monitor_tools.py` | `detect_new_files`, `detect_folder_changes`, `detect_new_data`, `detect_new_team_posts`, `detect_new_entities`, watermark tools |
| `llm-orchestrator/orchestrator/utils/mention_parser.py` | Parse @handle from chat messages, resolve to agent_profile_id |
| `llm-orchestrator/orchestrator/tools/notification_tools.py` | `send_notification`, `send_channel_message`, `broadcast_to_team` |
| `llm-orchestrator/orchestrator/tools/task_management_tools.py` | `update_todo`, `complete_todo`, `search_todos` (wrapping org-mode) |
| `llm-orchestrator/orchestrator/tools/file_operation_tools.py` | `delete_file`, `move_file`, `copy_file` (new granular file ops) |
| `llm-orchestrator/orchestrator/plugins/base_plugin.py` | Base class for external integration plugins |
| `llm-orchestrator/orchestrator/plugins/plugin_loader.py` | Plugin discovery, loading, and registry integration |
| `llm-orchestrator/orchestrator/plugins/integrations/trello_plugin.py` | Trello integration tools |
| `llm-orchestrator/orchestrator/plugins/integrations/notion_plugin.py` | Notion integration tools |
| `llm-orchestrator/orchestrator/plugins/integrations/slack_plugin.py` | Slack integration tools (extends connections-service) |
| `llm-orchestrator/orchestrator/plugins/integrations/caldav_plugin.py` | CalDAV calendar integration tools |
| **Frontend** | |
| `frontend/src/components/agent_factory/AgentFactoryPage.js` | Main Agent Factory UI |
| `frontend/src/components/agent_factory/AgentProfileEditor.js` | Profile create/edit form |
| `frontend/src/components/agent_factory/ConnectorBrowser.js` | Browse and configure connectors |
| `frontend/src/components/agent_factory/WorkflowComposer.js` | Visual step wiring and I/O linking UI |
| `frontend/src/components/agent_factory/PlaybookEditor.js` | Create/edit playbooks |
| `frontend/src/components/agent_factory/OutputConfigEditor.js` | Configure output destinations |
| `frontend/src/components/agent_factory/TestPanel.js` | Interactive agent testing |
| `frontend/src/components/agent_factory/ExecutionHistory.js` | Execution log viewer |
| `frontend/src/components/agent_factory/EntityReviewQueue.js` | HITL entity resolution UI |
| `frontend/src/components/agent_factory/ApprovalQueue.js` | Workflow approval gate UI (notifications + inline) |
| `frontend/src/components/agent_factory/AgentJournal.js` | Agent work journal viewer |
| `frontend/src/components/agent_factory/TeamShareManager.js` | Team sharing configuration UI |
| `frontend/src/components/chat/AgentMentionAutocomplete.js` | @mention autocomplete dropdown in chat input |
| `frontend/src/components/sidebar/MyAgentsPanel.js` | Chat sidebar panel listing user's and team-shared agents |
| `frontend/src/services/agentFactoryService.js` | API service for Agent Factory |

---

## 12. Modular Tool & Plugin Registry

The tool system is designed for modularity. Native tools, connector-generated tools, and external plugin tools all register through the same Action I/O Registry, making them equally available to the Workflow Composer UI and playbook runtime.

For the **complete tool catalog** with I/O contracts, scope rules, and implementation status, see [AGENT_FACTORY_TOOLS.md](./AGENT_FACTORY_TOOLS.md).

### Registry Architecture

```
                  ┌─────────────────────────────────────┐
                  │       Action I/O Registry            │
                  │  (Unified tool catalog with typed    │
                  │   input/output contracts)            │
                  └──────────┬──────────┬──────────┬────┘
                             │          │          │
              ┌──────────────┘          │          └──────────────┐
              ▼                         ▼                         ▼
  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
  │   Native Tools   │    │ Connector Tools  │    │  Plugin Tools    │
  │                  │    │                  │    │                  │
  │ orchestrator/    │    │ Generated from   │    │ Loaded from      │
  │ tools/*.py       │    │ connector YAML   │    │ plugins/         │
  │                  │    │ at runtime       │    │ integrations/    │
  │ • file_ops       │    │                  │    │                  │
  │ • task_mgmt      │    │ • call_connector │    │ • trello         │
  │ • notifications  │    │ • (per-endpoint  │    │ • notion         │
  │ • search         │    │    wrappers)     │    │ • slack          │
  │ • email          │    │                  │    │ • caldav          │
  │ • monitor        │    │                  │    │ • github (future) │
  └──────────────────┘    └──────────────────┘    └──────────────────┘
```

### Tool Resolution Order

When a custom agent runs, tools are resolved in this order:

1. **Agent Profile tools** — Explicitly assigned tool packs + connector-generated tools
2. **Plugin tools** — Loaded from the plugin directory, filtered by user's active connections
3. **Native tool packs** — Built-in orchestrator tools (always available)
4. **Dynamic registry** — User-defined via Agent Factory UI

### Plugin Interface

All external integration plugins extend `BaseToolPlugin`:

```python
# llm-orchestrator/orchestrator/plugins/base_plugin.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ToolIOContract(BaseModel):
    """Typed I/O contract for a single tool."""
    name: str
    description: str
    inputs: Dict[str, Any]       # Required inputs with types
    params: Dict[str, Any]       # Optional parameters with types and defaults
    outputs: Dict[str, Any]      # Output fields with types
    requires: Optional[str]      # Connection type required (e.g., "trello_api_key")


class ToolPack(BaseModel):
    """A named group of related tools."""
    name: str                    # e.g., "trello"
    description: str
    tools: List[str]             # Tool names in this pack


class BaseToolPlugin(ABC):
    """
    Base class for external integration plugins.

    Plugins are self-contained packages that register tools
    with the Action I/O Registry. Each plugin provides:
    - Tool functions (async Python callables)
    - I/O contracts (typed input/output definitions)
    - Connection requirements (what credentials are needed)
    - A tool pack definition (for assignment to agents)
    """

    plugin_name: str             # "trello", "notion", "slack", etc.
    plugin_version: str          # Semantic version
    connection_type: str         # "api_key", "oauth", "caldav_credentials", etc.
    required_config: List[str]   # Config keys needed from user's connection

    @abstractmethod
    def get_tool_contracts(self) -> List[ToolIOContract]:
        """Return typed I/O contracts for all tools in this plugin."""
        ...

    @abstractmethod
    def get_tool_pack(self) -> ToolPack:
        """Return the tool pack definition for this plugin."""
        ...

    @abstractmethod
    async def initialize(self, connection_config: Dict[str, Any]) -> None:
        """
        Initialize the plugin with user's connection credentials.
        Called once per agent execution before any tools are invoked.
        """
        ...

    @abstractmethod
    def get_tool_functions(self) -> Dict[str, callable]:
        """
        Return a mapping of tool_name -> async callable.
        Each callable must match the I/O contract signature.
        """
        ...
```

### Plugin Loader

```python
# llm-orchestrator/orchestrator/plugins/plugin_loader.py

class PluginLoader:
    """
    Discovers and loads tool plugins at startup and on-demand.

    Discovery sources (checked in order):
    1. Built-in: orchestrator/plugins/integrations/*.py
    2. Entry points: Python packages with 'bastion.plugins' entry point
    3. Config: Plugins listed in ENABLED_PLUGINS env var
    """

    def __init__(self):
        self._plugins: Dict[str, BaseToolPlugin] = {}
        self._initialized: Dict[str, bool] = {}

    async def discover_plugins(self) -> List[str]:
        """Scan all discovery sources and return available plugin names."""
        ...

    async def load_plugin(
        self,
        plugin_name: str,
        connection_config: Dict[str, Any]
    ) -> BaseToolPlugin:
        """
        Load and initialize a specific plugin with user's connection config.
        Returns the initialized plugin instance.
        """
        ...

    async def register_with_io_registry(
        self,
        plugin: BaseToolPlugin,
        registry: ActionIORegistry
    ) -> None:
        """
        Register all of a plugin's tool I/O contracts in the action registry.
        After registration, tools appear in the Workflow Composer UI.
        """
        for contract in plugin.get_tool_contracts():
            registry.register_action(
                name=contract.name,
                category=f"plugin:{plugin.plugin_name}",
                contract=contract,
                callable=plugin.get_tool_functions()[contract.name]
            )

    async def get_tools_for_agent(
        self,
        agent_profile: Dict[str, Any],
        user_connections: List[Dict[str, Any]]
    ) -> Dict[str, callable]:
        """
        Resolve which plugin tools are available for an agent execution.
        Filters by agent's assigned tool packs AND user's active connections.
        """
        ...
```

### Runtime Integration

At custom agent execution time:

```python
# In CustomAgentRunner.prepare_context_node:

async def _prepare_context_node(self, state: CustomAgentState) -> Dict[str, Any]:
    # 1. Load agent profile
    profile = state["agent_profile"]

    # 2. Resolve native tool packs
    native_tools = resolve_native_tools(profile["tool_packs"])

    # 3. Generate connector tools
    connector_tools = generate_connector_tools(profile["connectors"])

    # 4. Load plugin tools (filtered by user's connections)
    plugin_loader = PluginLoader()
    user_connections = await get_user_connections(state["user_id"])
    plugin_tools = await plugin_loader.get_tools_for_agent(
        agent_profile=profile,
        user_connections=user_connections
    )

    # 5. Merge all tools
    all_tools = {**native_tools, **connector_tools, **plugin_tools}

    return {
        "available_tools": all_tools,
        # ... preserve critical state ...
    }
```

### Adding a New Integration

To add a new external integration (e.g., Jira):

1. **Create plugin file:** `orchestrator/plugins/integrations/jira_plugin.py`
2. **Extend `BaseToolPlugin`** with Jira-specific tool functions
3. **Define I/O contracts** for each tool (list_issues, create_issue, etc.)
4. **Add connection type** to `external_connections` supported providers
5. **Plugin auto-discovered** — appears in Agent Factory UI's tool browser
6. **Users assign the Jira tool pack** to their agents via the Workflow Composer

No orchestrator engine changes. No registry code changes. No deployment changes beyond adding the plugin file.

### Database: Plugin Registry Table

```sql
CREATE TABLE agent_plugin_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plugin_name VARCHAR(100) NOT NULL UNIQUE,
    plugin_version VARCHAR(20) NOT NULL,
    connection_type VARCHAR(100) NOT NULL,
    required_config JSONB NOT NULL DEFAULT '[]',
    tool_contracts JSONB NOT NULL DEFAULT '[]',   -- Cached I/O contracts
    tool_pack_name VARCHAR(100) NOT NULL,
    is_builtin BOOLEAN DEFAULT false,             -- Built-in vs. user-installed
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Track which users have configured which plugins
CREATE TABLE agent_plugin_user_config (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    plugin_name VARCHAR(100) NOT NULL REFERENCES agent_plugin_registry(plugin_name),
    connection_id UUID REFERENCES external_connections(id), -- Link to user's connection
    custom_config JSONB DEFAULT '{}',             -- Plugin-specific overrides
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (user_id, plugin_name)
);
```

### API Endpoints: Plugin Management

```
GET    /api/agent-factory/plugins              — List available plugins
GET    /api/agent-factory/plugins/:name        — Get plugin details + I/O contracts
POST   /api/agent-factory/plugins/:name/enable — Enable plugin for current user
DELETE /api/agent-factory/plugins/:name/enable — Disable plugin for current user
GET    /api/agent-factory/plugins/:name/tools  — List tools with I/O contracts
POST   /api/agent-factory/plugins/:name/test   — Test plugin connection
```

---

## 13. Migration Strategy

### Backward Compatibility

The Agent Factory is purely additive. No existing functionality changes:

- Static agents continue to work unchanged
- Static skills continue to work unchanged
- Static tool packs continue to work unchanged
- Existing chat/orchestrator endpoints accept `agent_profile_id` as optional

### Migration Path for Existing Skills

Existing static skills can optionally be surfaced in the Agent Factory UI as "built-in" skills:

```python
# During Phase 2, load static skills into the dynamic registry

async def migrate_static_skills():
    """
    Make existing static skills visible in Agent Factory
    without changing how they work.
    """
    static_skills = load_all_skills()  # Existing function

    for skill in static_skills:
        # Create a read-only registry entry
        await create_skill_registry_entry(
            name=skill.name,
            description=skill.description,
            skill_type="built_in",
            is_template=True,  # Cannot be modified by users
            definition={
                "engine": skill.engine.value,
                "domains": skill.domains,
                "tools": skill.tools,
            }
        )
```

### Data Migration

No data migration needed. All Agent Factory tables are new. The existing `data_source_connectors` concept (from Data Workspace's `external_db_connections`) is similar but different enough to warrant separate tables rather than migration.

### Feature Flag

The Agent Factory can be gated behind a feature flag during development:

```python
# backend/config.py
AGENT_FACTORY_ENABLED = os.getenv("AGENT_FACTORY_ENABLED", "false").lower() == "true"
```

This allows incremental development and testing without affecting existing users.

---

## Appendix: Technology Choices

### Entity Resolution Libraries

| Library | Strengths | Consideration |
|---|---|---|
| **python-Levenshtein** | Fast string distance | Good for name matching |
| **jellyfish** | Phonetic encoding (Soundex, Metaphone) | Good for name variants |
| **recordlinkage** | Full deduplication pipeline | May be too heavy |
| **dedupe** | ML-based entity matching | Active learning model |
| **spaCy** | Already in stack, NER + similarity | Extend existing usage |

Recommendation: Start with jellyfish (phonetic) + python-Levenshtein (string distance) + spaCy (NER), evaluate dedupe for ML-based matching in Phase 3.

### Neo4j Graph Data Science

The Neo4j GDS library provides all needed graph algorithms. Ensure the GDS plugin is installed on the Neo4j instance:

```cypher
-- Verify GDS is available
RETURN gds.version()

-- Required algorithms:
-- gds.pageRank
-- gds.louvain (community detection)
-- gds.betweenness
-- gds.shortestPath
-- gds.wcc (weakly connected components)
-- gds.nodeSimilarity
```

### Connector HTTP Client

Use `aiohttp` (already in stack) for REST API connectors. For web scrapers, delegate to the existing Crawl4AI gRPC service.

### YAML Parsing

Use `pyyaml` (already in stack) for connector and playbook definition parsing. Validate against JSON Schema generated from the YAML specifications defined in this document.
