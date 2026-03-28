# Complementary LLM Libraries for Agent Factory

LangGraph is the execution backbone for all agent workflows. This document evaluates libraries that **complement** LangGraph by filling capability gaps — prompt optimization, multi-agent delegation, and advanced retrieval. None of these replace LangGraph; they slot in as specialized capabilities that feed into or wrap around LangGraph nodes.

## Architecture Principle

```
UnifiedDispatcher.dispatch(skill_name, ...)
  └─ LangGraph StateGraph (execution engine — nodes, edges, state, checkpointing)
       ├─ Nodes use DSPy-optimized prompts (compiled at build/warmup time)
       ├─ Subgraphs follow CrewAI-inspired delegation patterns (design-time influence)
       └─ Retrieval nodes can use Haystack pipelines (specialized RAG)
```

| Library       | Role                                     | When It Runs              |
|---------------|------------------------------------------|---------------------------|
| LangGraph     | Executes the agent workflow              | Every request             |
| DSPy          | Optimizes prompts for agents             | Build / warmup time       |
| CrewAI        | Informs multi-agent subgraph design      | Design time (patterns)    |
| Haystack      | Advanced retrieval pipelines             | Within LangGraph nodes    |
| Autogen       | Multi-agent conversation patterns        | Within LangGraph nodes    |
| Semantic Kernel | Auto-planning from goals + tools       | Within LangGraph nodes    |

---

## 1. DSPy — Prompt Optimization and Compilation

### What It Provides

DSPy (Stanford) treats prompts as programs that can be optimized. Instead of hand-tuning system prompts and few-shot examples, you define input/output signatures and DSPy compiles optimal prompts by running against evaluation datasets.

### How It Integrates

DSPy runs **at build time or agent warmup**, not at request time. It optimizes the prompts that LangGraph agents already use. The output is a better system prompt or few-shot examples injected into existing `_get_llm()` calls. LangGraph never knows the difference.

```
Agent Factory "Compile" step:
  1. User defines agent with input/output examples
  2. DSPy optimizes the system prompt + few-shot selection
  3. Optimized prompt stored in agent_profiles.system_prompt
  4. LangGraph agents use the optimized prompt at runtime via _get_llm()
```

### Integration Points

- `CustomAgentRunner` — playbook `llm_task` steps could use DSPy-compiled prompts
- `BaseAgent._get_llm()` — system prompts injected at the existing prompt layer
- Agent Factory UI — "Optimize" button that runs DSPy compilation against user-provided examples

### Pros

- Dramatically improves agent quality without prompt engineering expertise
- Systematic rather than trial-and-error prompt tuning
- Works with any LLM provider (OpenAI, Anthropic, open-source)
- Composable modules (ChainOfThought, ReAct, etc.) that compile to optimized prompts
- No runtime overhead — optimization happens offline

### Cons

- Requires evaluation datasets (users must provide input/output examples)
- Compilation can be slow and expensive (many LLM calls during optimization)
- Optimized prompts can be opaque — hard to debug why a compiled prompt works
- Version management needed — recompile when model or task changes
- Relatively new ecosystem; API still evolving

### Priority: **High**

Agent Factory custom agents would benefit most. Users define what they want, DSPy figures out how to prompt for it.

---

## 2. CrewAI — Multi-Agent Delegation Patterns

### What It Provides

CrewAI's core strength is agent-to-agent delegation with explicit role hierarchies (manager, researcher, writer, reviewer). Agents delegate subtasks, critique each other's output, and iterate autonomously.

### How It Integrates

CrewAI patterns are implemented **as LangGraph nodes and subgraphs**, not as a runtime dependency. A "crew manager" node decides which sub-agents to invoke, then routes to those agents' subgraphs via conditional edges. The delegation logic comes from CrewAI's patterns; execution is LangGraph state transitions.

```
LangGraph StateGraph:
  ├─ crew_manager_node (decides delegation)
  │    ├─ conditional_edge → research_subgraph
  │    ├─ conditional_edge → writing_subgraph
  │    └─ conditional_edge → review_subgraph
  ├─ review_subgraph (critiques output)
  │    └─ conditional_edge → back to writing if needs revision
  └─ END
```

### Integration Points

- Planned `send_to_agent` and `start_agent_conversation` tools
- New `EngineType` or subgraph pattern within `CUSTOM_AGENT` engine
- Agent Factory UI for defining crew compositions (which agents collaborate)

### Pros

- Enables complex multi-agent workflows (research → write → review → revise)
- Role-based delegation is intuitive for users designing agents
- "Critique and revise" loops improve output quality
- Can be built incrementally on existing LangGraph subgraph patterns
- No new runtime dependency if implemented as LangGraph patterns

### Cons

- Multi-agent loops can be expensive (many LLM calls per request)
- Difficult to predict execution time and cost
- Debugging multi-agent interactions is complex
- Risk of infinite loops without careful termination conditions
- CrewAI as a direct dependency adds coupling; better to adopt patterns only
- Agent-to-agent communication adds state management complexity

### Priority: **Medium**

The planned `send_to_agent` features need a delegation model. Adopt patterns rather than the library itself.

---

## 3. Haystack (deepset) — Advanced Retrieval Pipelines

### What It Provides

Haystack is purpose-built for RAG pipelines — document preprocessing, hybrid search (BM25 + dense retrieval), cross-encoder reranking, and answer extraction. Its pipeline model is specifically optimized for document-heavy workflows.

### How It Integrates

Haystack pipelines run **within LangGraph nodes** as specialized retrieval steps. A retrieval node could use a Haystack pipeline instead of direct Qdrant queries when advanced retrieval is needed.

```
LangGraph Node:
  async def _advanced_retrieval_node(self, state):
      pipeline = HaystackPipeline(...)
      pipeline.add_component("bm25_retriever", ...)
      pipeline.add_component("dense_retriever", ...)
      pipeline.add_component("reranker", ...)
      results = pipeline.run({"query": state["query"]})
      return {"retrieved_documents": results}
```

### Integration Points

- `intelligent_document_retrieval_subgraph` — could use Haystack for hybrid search
- Data source connectors — document ingestion pipelines
- Research agent — advanced retrieval for multi-round research

### Pros

- Hybrid search (BM25 + dense) significantly improves retrieval quality
- Cross-encoder reranking catches results that bi-encoder embeddings miss
- Mature document preprocessing (PDF, DOCX, HTML extraction)
- Composable pipeline components
- Active community and good documentation

### Cons

- Overlaps significantly with existing Qdrant vector search
- Adds dependency complexity for incremental retrieval improvement
- BM25 requires a separate index (Elasticsearch/OpenSearch)
- Reranking adds latency to every retrieval call
- May be overkill if current retrieval quality is acceptable

### Priority: **Low**

Only worth pursuing if retrieval quality becomes a measurable bottleneck. Current vector store service handles most cases.

---

## 4. Autogen (Microsoft) — Multi-Agent Conversation Patterns

### What It Provides

Autogen excels at conversational agent patterns — round-robin debates, group chats between agents, and human-in-the-loop within multi-agent conversations. It treats multi-agent interaction as a first-class primitive rather than a graph.

### How It Integrates

Autogen patterns would be implemented within LangGraph nodes that manage multi-turn agent-to-agent conversations. A "group chat" node would orchestrate several LLM calls in a conversation loop.

### Integration Points

- "Committee review" pattern for research synthesis (multiple perspectives)
- "Devil's advocate" agent that challenges conclusions before presenting to user
- Could inform design of `start_agent_conversation` tool

### Pros

- Natural model for agent discussion and debate
- Good for generating diverse perspectives on a topic
- Built-in conversation management (turn-taking, termination)
- Strong human-in-the-loop patterns

### Cons

- Multi-agent conversations are expensive (many LLM calls)
- Difficult to control quality and relevance of "debates"
- Niche use case — most agent tasks don't benefit from debate
- Significant overlap with CrewAI patterns
- Adding Autogen as a dependency for a narrow use case is expensive

### Priority: **Low**

Only relevant if "committee of agents" patterns become a user-requested feature.

---

## 5. Semantic Kernel (Microsoft) — Auto-Planning from Goals

### What It Provides

Semantic Kernel's planners can automatically compose a sequence of tool calls to achieve a goal, given a set of available functions. Instead of defining explicit playbook steps, you describe what you want and Semantic Kernel figures out how to get there.

### How It Integrates

A Semantic Kernel planner could run as a LangGraph node that dynamically generates a playbook at runtime, which is then executed by the existing `PipelineExecutor`.

```
LangGraph Node:
  async def _auto_plan_node(self, state):
      available_tools = get_tools_for_agent(agent_profile)
      plan = semantic_kernel_planner.create_plan(
          goal=state["query"],
          available_functions=available_tools
      )
      return {"dynamic_playbook": plan.steps}
```

### Integration Points

- `PlaybookGraphBuilder` — auto-generated playbooks as input
- Agent Factory UI — "auto-plan" mode alongside manual playbook design
- `CustomAgentRunner` — dynamic playbook execution

### Pros

- Users describe goals instead of engineering step sequences
- Adapts to available tools dynamically
- Could make Agent Factory accessible to non-technical users
- Natural complement to the existing playbook system

### Cons

- Auto-generated plans can be unpredictable and incorrect
- Hard to debug when the planner makes bad tool choices
- Security risk — planner might invoke tools in unintended combinations
- Explicit playbooks are safer and more auditable
- Adds a Microsoft dependency for a feature that LLMs can approximate natively
- Function-calling models (GPT-4, Claude) already do basic planning via tool use

### Priority: **Low**

Explicit playbooks are safer for production use. The LLM's native tool-use capabilities already handle basic planning within `llm_augmented` execution mode.

---

## Build Toward in Our Own Codebase (Scheduling & Execution)

We keep Celery for scheduled and event-triggered agent execution. The following are **capabilities to improve over time in our own code**, rather than adopting an external durable-workflow engine:

| Capability | Current state | Build toward |
|------------|----------------|--------------|
| **Overlap prevention** | Redis locks per dispatch (e.g. `sched:agent:{id}:lock`, `team_reaction:{id}:{team}:lock`) with manual TTL and cleanup | Shared utility so every dispatch doesn’t copy-paste the pattern; consider workflow-ID–style uniqueness in execution log |
| **Per-user concurrency** | Redis counter `sched:user:{user_id}:running` with manual incr/decr and expiry | Centralized concurrency helper (same pattern, less duplication) |
| **Circuit breaker** | Consecutive-failure count in DB, auto-pause schedule after N failures, WebSocket notify | Keep; optionally expose max failures and backoff in Agent Factory UI |
| **Retries and timeouts** | Celery `soft_time_limit` / `task_time_limit`, `SoftTimeLimitExceeded`; task-level retry settings | Declarative retry/backoff per schedule (e.g. in `agent_schedules`) if we want finer control |
| **Mid-step recovery** | Not applicable today — each run is one gRPC call | If playbooks become multi-step with approval gates: checkpoint progress in DB and resume from last step instead of re-running from scratch |
| **Execution visibility** | `agent_execution_log` + task progress meta | Richer execution detail API and UI (e.g. step-by-step history, duration per step) when we add multi-step playbook execution |
| **Long-running runs** | Hard limit 20 min; most runs are short | If we need longer runs: chunked execution (save state, resume in a new task) or raise limit only where justified |

These stay as incremental improvements to the existing Celery + Redis + Postgres stack rather than a separate orchestration layer.

---

## Implementation Roadmap

### Phase 1 — Near Term (High Priority)

| Library     | Integration                                    | Effort  |
|-------------|------------------------------------------------|---------|
| DSPy        | Prompt compilation for Agent Factory agents   | Medium  |

### Phase 2 — Medium Term

| Library     | Integration                                    | Effort  |
|-------------|------------------------------------------------|---------|
| CrewAI patterns | Multi-agent delegation subgraphs          | Medium  |

### Phase 3 — If Needed

| Library     | Integration                                    | Effort  |
|-------------|------------------------------------------------|---------|
| Haystack    | Hybrid search in retrieval subgraphs           | Medium  |
| Autogen     | Multi-agent debate patterns                    | Low     |
| Semantic Kernel | Auto-planning for dynamic playbooks       | Medium  |

---

## Decision Criteria

Before adding any library, evaluate:

1. **Does it solve a real user problem?** Not a theoretical improvement — an actual bottleneck or missing capability.
2. **Can LangGraph do it natively?** Many patterns can be built with conditional edges and subgraphs without new dependencies.
3. **What's the operational cost?** DSPy adds a build step. Weigh benefit against complexity.
4. **Is the library mature?** Prefer stable, well-maintained libraries with active communities.
5. **Can it be adopted incrementally?** Prefer patterns you can borrow (CrewAI delegation) over hard dependencies you must maintain.
