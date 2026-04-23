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

### How `output_schema` works in the Composer

In the Workflow Composer you define **field names** and **JSON types** (string, number, boolean, and so on). The system stores a small JSON Schema object of the form `{ "type": "object", "properties": { "field_name": { "type": "string" }, ... } }`.

After the LLM responds, the runtime **extracts JSON** from the reply (including markdown `json` code fences) and parses it into a **dictionary**. Each top-level key from the parsed object becomes a **named output** on the step, so you can wire `{your_output_key.label}`, `{your_output_key.confidence}`, etc., in downstream prompts and inputs. The parser also keeps **`formatted`** (and on failure paths **`raw`**) so you still have text to read or pass along when JSON parsing does not succeed.

**Without** an `output_schema`, treat the step as **plain text** only: **`{step.formatted}`** and **`{step.raw}`** are the portable references. **Named fields exist only when** you declared them via `output_schema` (or they appear because parsing returned keys matching your schema’s properties). The schema object is currently used as **metadata for tooling and future validation**; strict enforcement against the schema at parse time may be added later.

---

## LLM agent steps

An **llm_agent** step is different: the LLM runs in a **ReAct loop**. It can call **multiple tools**, in an order it decides, and can do so for several turns (up to **max_iterations**, e.g. 5 or 10). That makes it suitable for tasks like “figure out what to do and use these tools” — research, multi-step document editing, or exploratory workflows.

The Composer’s **Tools & connections** section builds **available_tools** and **tool_packs** for you (category toggles, pinned tools, external accounts). You can set **max_iterations** (default is often 3); use a higher value when the task needs many tool calls. You can also set a **prompt** (or prompt template with variables like `{query}`) to guide the agent’s behavior, and attach **skills** on the step to influence how it reasons.

Use **llm_agent** when the sequence of actions isn’t fixed. Use **tool** + **llm_task** steps when the order is known (e.g. “always search, then summarize, then email”).

### Subagents (LLM agent and deep agent)

On **LLM agent** and **deep agent** steps you can attach **subagents**: other **agent profiles** the parent model may hand work to. Each subagent uses that profile’s **default playbook** (the same as when you run that profile elsewhere). The Workflow Composer lists subagents on the step and three **delegation modes**:

| Mode | What happens |
|------|----------------|
| **Supervised** | The parent model runs a normal tool loop. Each subagent appears as its own **delegation tool** (e.g. `delegate_subagent_0_…`). The model chooses **when** to call which specialist. |
| **Parallel** | Every subagent runs **once** **before** the parent’s main work. They run **at the same time**. Their outputs are stored in a **shared scratchpad**; the parent then **synthesizes** (LLM agent: supervisor ReAct loop; deep agent: subsequent phases). |
| **Sequential** | Same as parallel, but subagents run **one after another** in list order, then the parent synthesizes. |

**How the parent model knows what each subagent is for**

You do **not** rely only on the step prompt. Each delegation tool gets a **description** built from the Composer fields:

- **Role** — Short label for the specialist (included in the tool description).
- **Accepts** — Shown as **“Best for: …”** so the model knows what kind of request belongs with this subagent.
- **Returns** — Shown as **“Typically returns: …”** so the model knows what output to expect.

In **supervised** mode, the parent uses those descriptions to pick the right tool. On each call it supplies a **`task`** string (required) and optional **`context_json`** / **`output_hint`** as the tool arguments—so the **task** is how you steer that run’s instructions, while Role / Accepts / Returns steer **which** tool to use.

**What subagents receive as their first task (parallel / sequential)**

- **LLM agent step:** The **resolved prompt template** for that step (after wiring `{placeholders}`) is passed to **each** subagent as the initial task before synthesis. Write it as a **shared objective**; use Role / Accepts / Returns so each profile knows its slice of the work.
- **Deep agent step:** The initial task is built from the **user query** plus **prompts and criteria** from the **first few phases** on the step (up to five phases, concatenated)—not from the LLM agent–style prompt field elsewhere in the drawer.

After pre-dispatch, scratchpad contents are injected into context so the parent or later phases can integrate subagent results.

### Tools & connections: scope, packs, individual tools, skills

In the Workflow Composer **Tools & connections** section for **LLM agent** and **deep agent** steps, tools are grouped into **categories** (e.g. Search and discovery, Documents, Email, GitHub). Each row is a logical capability—not a raw pack name.

| UI area | What you configure (saved on the step) | What happens at run time |
|---------|----------------------------------------|---------------------------|
| **Connection scope** | **Inherit** (default), **Restrict** (subset of accounts), or **None** (no email/calendar/code tools on this step). | The orchestrator applies this on top of the agent profile’s connection allowlist. |
| **Tool packs & accounts** (category checkbox) | Enables the underlying **tool packs** for that category (still stored as `tool_packs`). Read/Full appears when any pack in the category has write-capable tools. | **Every tool** in those packs is registered for that run unless you narrow with per-tool checkboxes. |
| **Expand category** | Optional **per-tool** checkboxes. Turning off a tool removes the pack entries for that category and keeps only the selected tool names in **`available_tools`**. | Packs + explicit tools are merged by the orchestrator. |
| **External categories** (Email, Calendar, GitHub, Gitea, MCP) | **Which connected accounts** apply; chosen on the same row. | Accounts are chosen in the UI; the orchestrator exposes **scoped** tools (per connection id) and validates them against the effective allowlist. |
| **Individual tools** (pinned) | Tool names **not** already implied by an enabled category (e.g. only `patch_file` with no document packs on). | Static **`available_tools`** each run—use **Add tool** to pick by name. |
| **Skills** | Which skills are selected | **Pinned** skills are **always** attached each run, regardless of discovery mode. |
| **Skill discovery** | **Off** / **Auto** / **Catalog** / **Full**, plus **max discovered skills** on LLM agent and deep agent steps. **Auto** is recommended for most agents. | **Off**: pinned skills only. **Auto**: pre-run semantic match to the prompt, up to max discovered skills; no injected skills catalog. **Catalog**: injects core skills list into the prompt and adds **acquire_skill** / **search_and_acquire_skills**; no pre-run prompt match (pin skills to preload). **Full**: **Auto** + **Catalog** (pre-run match, catalog, and acquisition tools). |

**LLM task** steps only show **Tools & connections** as **skills and discovery**: pinned skills are fixed; **Auto** can still add matched skills from the prompt before the run. **Catalog** and **Full** are not available on **llm_task** (no injected skills catalog and no mid-run skill search).

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

### Step output (what the user sees)

The step’s main result (for example `{output_key.formatted}` in chat and for downstream wiring) is chosen in this order:

1. **Output template** — If set, the template is resolved the same way as phase prompts: `{phase_name.output}`, `{phase_name.feedback}`, `{query}`, `{editor}`, and other playbook variables. Use this to concatenate a synthesis plus evaluator margin notes, or to pick between `{refine.output}` and `{draft.output}` in prose.
2. **Output phase** — If set to a phase **name**, that phase’s **output** string becomes the step result. Leave unset (**Auto (last non-evaluate phase)** in the Composer) to use the rule below.
3. **Auto default** — The runtime walks phases in **reverse definition order** and picks the first phase with non-empty **output** whose type is **not** **evaluate** (so quality-gate JSON is not shown as the main answer). If no non-evaluate phase produced text, it falls back to the previous behavior: the last phase with any output, **including** evaluate, so older playbooks do not end up empty.

If both **output template** and **output phase** are set, the **template wins** (validation may note this).

**Subagents** on a deep agent step use the same delegation modes and tool-description fields (**Role**, **Accepts**, **Returns**) as on an LLM agent step. For **parallel** / **sequential** pre-dispatch, see **Subagents** above: the task string is query + early phase prompts/criteria, not the single prompt template used by LLM agent steps.

---

An **approval** step **pauses** the workflow and shows you a preview (e.g. “About to send this email” or “About to apply these edits”). You choose **Approve** to continue or **Reject** to stop. Optional **approval_message** and **reject_message** customize what the user sees.

Use **approval** before steps that have real-world impact: sending messages, updating documents, or calling external APIs. In scheduled or background runs, set the agent profile’s **`default_approval_policy`** to **`auto_approve`** so approval steps don’t block unattended jobs; use **`require`** when a human must always confirm (default).

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

### Exclusive (stop after match)

In the step configuration drawer, when a step has a **condition**, you can enable **Exclusive (stop after match)** on that step.

- If the condition is **true** and the step **runs**, the playbook **ends after this step** — no later steps run.
- If the condition is **false** and the step is **skipped**, execution **continues** to the next step as usual.

Use this when you have several **mutually exclusive** conditional steps (e.g. one per document type) followed by a **general** step that should run only when **none** of those conditions matched. Without Exclusive, the playbook keeps running in order after a matched step, so that final “catch-all” step would still run even after a specialist step already ran.

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

## Model override (LLM agent and deep agent)

On **LLM agent** and **deep agent** steps you can set a **model override** so **that step alone** uses a different model than the agent profile’s **model preference** (or the user’s default when the profile does not set one).

- In the Composer, pick a model from the **dropdown** or type a **model id** (deep agent steps accept free-text ids such as `provider/model-name`).
- The override applies **only to that step**; other steps keep the profile default.
- Use a **smaller or cheaper** model for classification or extraction, and a **stronger** model for synthesis, without creating separate agent profiles.

This setting is intended for **`llm_agent`** and **`deep_agent`** steps.

---

## Best-of-N sampling (LLM agent and deep agent)

On **LLM agent** and **deep agent** steps you can enable **Best-of-N sampling** so the runtime runs the step **multiple times independently** (with higher temperature for diversity) and then **selects the best** result before the playbook continues.

| Field | What it does |
|-------|----------------|
| **samples** | Integer **1–5** (default **1** = off). How many parallel runs. Each run uses the same wiring but the model is nudged toward more varied outputs. |
| **selection_strategy** | **`llm_judge`** (default): one extra LLM call reads all candidates and returns the best index. **`highest_score`** (**deep agent** only): uses the **score** from the **last `evaluate` phase** in each run’s `phase_trace`; if no scores exist, the runtime falls back to `llm_judge`. |
| **selection_criteria** | Optional text for the judge when using **`llm_judge`** (e.g. “Pick the most thorough, well-structured answer”). |

**Cost and latency** scale about linearly with **samples**. Prefer **2–3** unless quality demands more.

**In the Workflow Composer:** open the step → expand **Best-of-N sampling** → enable **Run multiple samples and select the best**, set **Samples**, **Selection strategy**, and (for LLM judge) **Selection criteria**.

---

## Dynamic fan-out (LLM agent and deep agent)

**Dynamic fan-out** runs the step **once per element** of a **list** stored in playbook state (from an earlier step), **in parallel** (up to a cap), then **merges** everything into this step’s **output_key**.

| Field | What it does |
|-------|----------------|
| **fan_out.source** | Required. **Dot-path** to the list inside `playbook_state`, e.g. `plan.items` if a prior step wrote `output_key: plan` and a list at `items`. |
| **fan_out.item_variable** | Name injected for each copy (default **`current_item`**). Use **`{current_item}`** in the LLM agent prompt template (or in phase prompts for deep agent). |
| **fan_out.max_items** | Max parallel branches (**1–10**, default **10**). |
| **fan_out.merge** | **`list`** (default): merged output includes **`items`** (array of per-item results) plus a combined **`formatted`** string. **`concat`**: emphasizes sectioned **`formatted`** text (still includes **`items`**). |

If the source list is **empty**, the step still completes: the output records **`_fan_out_count`: 0** and a short **`formatted`** message.

**In the Workflow Composer:** expand **Dynamic fan-out** → enable **Run once per item in a list** → set **Source**, **Item variable**, **Max parallel items**, and **Merge mode**.

---

## Choosing the right step type

- **Fixed sequence, one tool per step** → Use **tool** steps and wire outputs to the next step. Add an **llm_task** when you need a single LLM call (e.g. “summarize this”).
- **Agent must decide what to do and call tools multiple times** → Use one **llm_agent** step with **available_tools** and a suitable **max_iterations**. Tune **skills**, **skill discovery** (Off / Auto / Catalog / Full), and **max discovered skills** when you need pinned procedures or broader skill awareness—see **Tools reference**. Add **subagents** when another profile should do part of the work (supervised delegation tools, or parallel/sequential pre-dispatch + synthesis).
- **Fixed multi-phase flow (plan → gather → synthesize → evaluate/revise)** → Use one **deep_agent** step with a **phases** list; choose phase types and wire next/retry in the Composer.
- **Different path based on data** → Use a **branch** step with **branch_condition**, **then_steps**, and **else_steps**. For **three or more cases** (e.g. by document type), use a flat list of steps each with a **condition** instead of nesting branches — see **Condition on any step** above.
- **Human must confirm before something runs** → Add an **approval** step before the risky action.
- **Same steps for each item in a list** → Use a **loop**.
- **Independent work that can run at once** → Use **parallel**.
- **Same agentic step, pick the best of several tries** → On **llm_agent** or **deep_agent**, use **Best-of-N sampling** (`samples` > 1).
- **Same step for each element of a runtime list** → On **llm_agent** or **deep_agent**, use **Dynamic fan-out** (`fan_out`).

For more on wiring (inputs, output_key, variables), see **Playbooks overview** and **Prompt variables and conditional blocks**.
