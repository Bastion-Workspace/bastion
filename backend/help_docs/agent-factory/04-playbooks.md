---
title: Playbooks overview
order: 4
---

# Playbooks overview

A **playbook** is the workflow that defines what your custom agent does when it runs. It’s a sequence of **steps**: each step runs a tool, asks an LLM to analyze or generate text, waits for your approval, or branches the flow based on a condition. You build playbooks in the **Workflow Composer** by adding steps and wiring data between them. This page explains what playbooks are, how they run, and how they fit into Agent Factory.

---

## What a playbook is

Think of a playbook as a recipe. The steps run in order (unless you use branches or parallel steps). Each step can:

- **Call a tool** — e.g. search documents, get the weather, send an email. The step’s **output** (e.g. search results, formatted text) is available to later steps.
- **Run an LLM task** — Send a prompt and some context to the model; get back analysis, a summary, or structured data. Again, the output can be used by later steps.
- **Run an LLM agent** — Let the model decide which tools to call and in what order, over several turns (a “ReAct” loop). Useful when the task is exploratory or the sequence isn’t fixed.
- **Run a deep agent** — Run a multi-phase workflow (reason, search, synthesize, evaluate, refine) you define; use when you want a fixed structure with optional retry loops.
- **Pause for approval** — Show you a preview of what’s about to happen and wait for you to approve or reject before continuing.
- **Branch, loop, or run in parallel** — Change the path based on a condition, repeat steps, or run several steps at once.

You don’t write code. You pick step types, set **inputs** (often from previous steps or from runtime variables like `{query}` or `{today}`), and give each step an **output key** so downstream steps can refer to its results.

---

## How data flows between steps

Every step can produce **output**. For a tool step, that’s the tool’s return value (e.g. `formatted` text plus typed fields like `documents` or `count`). For an LLM step, it’s the model’s response (e.g. `formatted`, `raw`, or custom fields you define).

When you add a step, you give it an **output_key** (e.g. `search_results` or `briefing`). Later steps then reference that key and a field name, like `{search_results.formatted}` or `{briefing.summary}`. The Composer uses the tool and step contracts to suggest compatible wires and flag invalid references.

You can also use **runtime variables** in step inputs — things like `{query}` (the user’s message), `{today}`, or `{editor}` (the open document). See **Prompt variables and conditional blocks** for the full list and how to hide sections when data is missing.

---

## Where playbooks run

- **In chat (interactive)** — When you @mention the agent or open a conversation with it, the playbook runs in that conversation. You see progress and can approve steps inline.
- **In the background** — A “Run” trigger or a folder watch can start the playbook without an open chat. Results go to the destinations configured on the profile (e.g. document, notification).
- **On a schedule** — The profile can be set to run on a cron schedule (e.g. daily briefing). The playbook runs like a background job; you don’t need to be in chat.

The same playbook definition is used in all cases. The **agent profile** controls run context (interactive vs background vs scheduled) and where results are sent. For **cron**, **interval**, and **event monitors** (lines, email, folders, conversations), see **Schedules and monitors**; for **`{trigger_input}`** and **`default_approval_policy`** (`require` vs `auto_approve`) on unattended runs, see that topic and **Agent profile settings**.

---

## How to build one

1. Open an **agent profile** and go to its **Playbook** section (or create a new playbook and attach it).
2. Add **steps** in the order you want them to run. Choose the step type (tool, LLM task, LLM agent, deep agent, approval, branch, loop, or parallel).
3. For each step, set **inputs**. Use `{output_key.field}` to pull from earlier steps, or `{variable}` for runtime data. Set an **output_key** so later steps can reference this step’s result.
4. For **tool** steps, pick the **action** (tool name) and fill required inputs. For **LLM** steps, write the **prompt** and optionally declare output schema. For **LLM agent** steps, use **Tools & connections** (connection scope, tool packs, individual tools, skill discovery), set **max_iterations**, and optionally add **subagents** (other profiles with a **delegation mode** and the step fields **Role**, **Accepts**, and **Returns** so the parent model knows when to delegate). For **deep agent** steps, define the **phases** list (reason, search, synthesize, evaluate, refine, etc.) and wire next/retry in the phase editor; subagents are optional here too—see **Playbook steps and flow control → Subagents**.
5. Use **branch** steps when you want different paths (e.g. “if fiction doc, do X; else do Y”). Use **approval** steps when you want a human to confirm before something runs.
6. **Validate** the playbook (the UI or the validate tool will report missing inputs or invalid references), then save.

For a deeper dive into step types and flow control (branch, loop, parallel), see **Playbook steps and flow control**.

---

## External connections (profile and steps)

**Agent profile — External connections for tools**

On the agent profile editor, you can **limit** which email, calendar, and code platform accounts this agent may ever use. **All connected accounts** means no profile-level cap (any account you add in Settings remains available until a step narrows it). **Only selected accounts** saves an explicit allowlist. Empty allowlist at the profile means *unrestricted* at the profile layer.

**LLM agent and deep agent steps — Connection scope**

In the Workflow Composer, open an **LLM agent** or **deep agent** step. Under **Tools & connections**, **Connection scope** controls how that step uses external accounts relative to the profile:

- **Inherit** — Use whatever the profile allows (default).
- **Restrict** — Pick a subset of accounts (only those allowed on the profile are listed when the profile has a limit).
- **None** — This step cannot use email, calendar, or code platform tools.

You still choose **tool packs** and **which accounts** apply to external categories (email, calendar, GitHub, etc.) on the same step; connection scope and pack wiring work together at run time.

**Skill discovery** (LLM agent / deep agent steps)

- **Off** — Only skills you attach on the step; no catalog block; no skill-acquisition tools added for discovery.
- **Auto** — Before the run, skills matched to the step prompt are attached (up to **max discovered skills**); no injected skills catalog.
- **Catalog** — Injects a compact **skills catalog** (core skills) into the prompt and adds **`acquire_skill`** / **`search_and_acquire_skills`** for this step. There is **no** automatic pre-run match from the prompt (pin skills if you need them loaded before the first turn).
- **Full** — **Auto** and **Catalog** together: pre-run matching, catalog text, and the same acquisition tools.

**LLM task** steps support **Off** and **Auto** only. **Tools reference → Skill discovery modes** has the full matrix and the difference between the **tool** catalog (*AGENT_FACTORY_TOOLS.md*) and the run-time **skills** catalog.

**Subagent naming fields (step configuration)**

In subagent step configuration, the editable fields are:

- **Role** — Short specialist label.
- **Accepts** — What requests this specialist is best for (rendered to the parent as "Best for").
- **Returns** — Expected output shape or topic (rendered as "Typically returns").

---

## Summary

- A playbook is an ordered sequence of steps that run when the agent is triggered.
- Steps can call tools, run LLM tasks or agents, wait for approval, or branch/loop/parallel.
- Data flows via **output_key** and **inputs**; you wire `{step_name.field}` and runtime variables like `{query}` and `{today}`.
- Playbooks run in chat, in the background, or on a schedule depending on the profile. Build and edit them in the Workflow Composer.
