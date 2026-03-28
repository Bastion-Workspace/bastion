---
title: Playbook steps and flow control
order: 5
---

# Playbook steps and flow control

Playbooks are made of **steps**. Each step has a **step type** that determines what it does and what it outputs. Many steps also support an optional **condition** so they run only when the condition is true (see **Condition on any step**). This page describes the step types you can use and how **branch**, **loop**, and **parallel** steps change the flow. Use this when you’re designing a playbook in the Workflow Composer.

---

## Step types at a glance

| Step type    | What it does |
|-------------|--------------|
| **tool**    | Calls one tool (e.g. search, send email) with fixed inputs. Output is that tool’s result. |
| **llm_task**| Sends a prompt (and context) to the LLM once; returns the model’s response. Good for analysis, summarization, or classification in a fixed order. |
| **llm_agent**| Lets the LLM run a **ReAct loop**: it can call multiple tools, in an order it chooses, for several turns (up to `max_iterations`). Use when the task is exploratory or the sequence isn’t known in advance. |
| **deep_agent**| Runs a **multi-phase workflow**: an ordered list of phases (reason, act, search, evaluate, synthesize, refine) that form a small graph. Use when you want a fixed structure (e.g. plan → gather → synthesize → evaluate → revise) with optional loops and tool use per phase. |
| **approval**| Pauses the workflow and shows you a preview; continues only after you approve (or stops if you reject). |
| **branch**  | Picks one of two paths (`then_steps` or `else_steps`) based on a **branch_condition**. |
| **loop**    | Runs a group of steps repeatedly (e.g. over a list). |
| **parallel**| Runs several steps at the same time; when all finish, their outputs are available to the next step. |

---

## Tool steps

A **tool** step runs a single action: search documents, get weather, send an email, run SQL, and so on. You choose the **action** (tool name) and set its **inputs**. Inputs can be literals, runtime variables like `{query}` or `{today}`, or outputs from earlier steps, e.g. `{search_results.formatted}`.

The step’s **output** is whatever the tool returns (typically a `formatted` string plus typed fields). Give the step an **output_key** so later steps can reference it (e.g. `output_key: weather_data` → `{weather_data.formatted}`).

Use a **tool** step when you know exactly which tool to call and in what order. For a fixed pipeline like “get weather → get calendar → compose briefing → send email,” use one tool step per call.

---

## LLM task steps

An **llm_task** step sends a **prompt** (and any context you wire in) to the LLM **once**. The model returns a response; that response is the step’s output. You can reference it in later steps (e.g. `{briefing.formatted}` or `{classify.label}`).

Use **llm_task** when you want a single LLM call: summarization, classification, rewriting, or composing text from data you’ve already gathered. The flow is fixed: first the tools (or previous steps) run, then this step runs, then the next step.

You can optionally define an **output_schema** so the model returns structured fields (e.g. `summary`, `bullets`) that downstream steps can reference.

---

## LLM agent steps

An **llm_agent** step is different: the LLM runs in a **ReAct loop**. It can call **multiple tools**, in an order it decides, and can do so for several turns (up to **max_iterations**, e.g. 5 or 10). That makes it suitable for tasks like “figure out what to do and use these tools” — research, multi-step document editing, or exploratory workflows.

You must list **available_tools**: the exact tool names the agent is allowed to call (e.g. `patch_file`, `search_documents`, `get_document_content`). You can set **max_iterations** (default is often 3); use a higher value when the task needs many tool calls. You can also set a **prompt** (or prompt template with variables like `{query}`) to guide the agent’s behavior, and optionally attach **skills** on the profile to influence how it reasons.

Use **llm_agent** when the sequence of actions isn’t fixed. Use **tool** + **llm_task** steps when the order is known (e.g. “always search, then summarize, then email”).

---

## Deep agent steps

A **deep_agent** step runs a **multi-phase workflow**: an ordered list of **phases** that form a small graph. Each phase has a **type** that determines what it does:

- **Reason** — One LLM call to plan or analyze (no tools). Use for “plan the approach” or “analyze the context.”
- **Act** — LLM can call tools in a ReAct loop (like a mini llm_agent) for this phase. Use when the phase needs to take actions.
- **Search** — Runs one or more tools (e.g. search_documents, web search), often in **parallel** or **sequential**; results are merged for the next phase. Use for “gather information.”
- **Evaluate** — LLM scores or critiques content against **criteria**; can **pass** (e.g. go to end), **fail** (e.g. go to refine), or **retry** with a limit. Use for quality gates.
- **Synthesize** — One LLM call to produce a summary, draft, or answer from prior phase outputs. Use for “write the final answer” or “combine findings.”
- **Refine** — LLM revises content (e.g. from a prior phase) using **feedback** from an evaluate phase; then routes to the next phase (often back to evaluate). Use for “improve until good enough.”

You define **phases** in order and wire them with **next** (e.g. after “synthesize” go to “critique”; after “revise” go back to “critique”). The Workflow Composer offers **phase list** editing and optional **starter templates** (e.g. Research, Iterative Refinement, Analyze) to get going quickly.

Use **deep_agent** when you want a **fixed multi-step structure** (plan → gather → synthesize → evaluate → maybe revise) with clear phases and optional retry loops, rather than a single free-form ReAct loop. Use **llm_agent** when the agent should fully decide the sequence; use **deep_agent** when you want to design the phases yourself.

---

An **approval** step **pauses** the workflow and shows you a preview (e.g. “About to send this email” or “About to apply these edits”). You choose **Approve** to continue or **Reject** to stop. Optional **approval_message** and **reject_message** customize what the user sees.

Use **approval** before steps that have real-world impact: sending messages, updating documents, or calling external APIs. In scheduled or background runs, the profile can be set to **auto_approve** so approval steps don’t block.

---

## Branch steps

A **branch** step doesn’t run a tool or an LLM. It **chooses a path** based on a condition.

- **branch_condition** — An expression that is evaluated at runtime, e.g. `{editor_document_type} == "fiction"`, `{search.count} > 0`, or `{query} matches "analyze|critique"`. Use runtime variables or outputs from earlier steps.
- **then_steps** — Steps that run when the condition is true.
- **else_steps** — Steps that run when the condition is false.

Only one of the two paths runs. The rest of the playbook continues after the branch (with whatever output the chosen path produced). You can nest branches for "if-then-else-if" chains, but for several cases (e.g. fiction vs outline vs article), see **Condition on any step** below — a flat list of steps each with a **condition** is usually easier to build and maintain.

Common use: route by document type (`{editor_document_type} == "fiction"`), by search results (`{step.count} > 0`), or by any variable you have in context.

---

## Condition on any step

Many step types (**tool**, **llm_task**, **llm_agent**, **deep_agent**) support an optional **condition** field. The condition uses the same expression syntax as **branch_condition** (e.g. `{editor_document_type} == "fiction"` or `{query} matches "proofread|grammar|typo"`). At runtime:

- If the **condition** is **true** — the step runs normally and its output is stored under its **output_key**.
- If the **condition** is **false** — the step is **skipped**. Its output key is set to `{"_skipped": true}` and execution continues to the next step.

This lets you build **multi-way routing** without nesting branch steps. For example, to do different work depending on the open file's frontmatter type:

1. Add several steps in a row (e.g. one **llm_task** or **tool** per document type).
2. Give each step a **condition** that matches one value, e.g. `{editor_document_type} == "fiction"`, `{editor_document_type} == "outline"`, `{editor_document_type} == "article"`.
3. Only the step whose condition is true runs; the others are skipped.

All steps stay at the same level in the playbook, so you can see and edit them easily in the Workflow Composer. Use **branch** when you need two clearly separate sub-workflows (then vs else). Use **condition** on individual steps when you have three or more cases (e.g. by `editor_document_type`) and want to keep the flow flat.

### Regex matching with `matches`

The `matches` operator tests a variable against a **regular expression pattern** (case-insensitive). This lets you route by what the user is asking for, not just what document is open — without needing an LLM classify step.

**Syntax:** `{variable} matches "pattern"`

**Examples:**
- `{query} matches "analyze|critique|summarize"` — true when the user's message contains any of those words.
- `{query} matches "analyze|critique" AND {editor_document_id} is defined` — only when asking for analysis AND a document is open.
- `{editor_document_type} matches "article|blog|substack"` — matches any of several doc types without chaining `==` with `OR`.

The pattern is a standard Python regular expression. If the pattern is invalid, the condition evaluates to `false`. Always quote the pattern: `"edit|fix"`.

---

## Loop steps

A **loop** step runs a group of steps **repeatedly**. You configure what to iterate over (e.g. a list from an earlier step) and which steps run inside the loop. The exact fields depend on the UI; the idea is “for each item, run these steps.”

Use **loop** when you need to do the same sequence for each item in a list (e.g. process each search result, or each row from a connector).

---

## Parallel steps

A **parallel** step runs **several steps at the same time** (e.g. “fetch weather and fetch calendar in parallel”). When all of them finish, their outputs are available to the next step. You define **parallel_steps** as a list of steps; they all start together.

Use **parallel** when you have independent work that doesn’t depend on each other — it can shorten total time and keep the playbook simple.

---

## Choosing the right step type

- **Fixed sequence, one tool per step** → Use **tool** steps and wire outputs to the next step. Add an **llm_task** when you need a single LLM call (e.g. “summarize this”).
- **Agent must decide what to do and call tools multiple times** → Use one **llm_agent** step with **available_tools** and a suitable **max_iterations**.
- **Fixed multi-phase flow (plan → gather → synthesize → evaluate/revise)** → Use one **deep_agent** step with a **phases** list; choose phase types and wire next/retry in the Composer.
- **Different path based on data** → Use a **branch** step with **branch_condition**, **then_steps**, and **else_steps**. For **three or more cases** (e.g. by document type), use a flat list of steps each with a **condition** instead of nesting branches — see **Condition on any step** above.
- **Human must confirm before something runs** → Add an **approval** step before the risky action.
- **Same steps for each item in a list** → Use a **loop**.
- **Independent work that can run at once** → Use **parallel**.

For more on wiring (inputs, output_key, variables), see **Playbooks overview** and **Prompt variables and conditional blocks**.
