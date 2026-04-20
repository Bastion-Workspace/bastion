---
title: Tools reference
order: 2
---

# Agent Factory: Tools reference

Agents use **tools** to read files, search, run tasks, send notifications, query the knowledge graph, and call external systems. All tools are registered with typed input/output contracts so the Workflow Composer can wire step outputs to later step inputs.

## Tool categories

The following categories are available. Each category contains multiple tools; the full **tool** catalog (names, parameters, return types) is maintained in the repository (see *AGENT_FACTORY_TOOLS.md*). That is separate from the **skills catalog** used at run time for some discovery modes—see **Skill discovery** below.

| Category | Description |
|----------|-------------|
| **File operations** | Read, search, create, and update documents; scope-aware (my docs, team docs, global). |
| **Search & discovery** | Semantic and keyword search over documents, segments, and the help docs. |
| **Task management** | Org-mode todos: list, create, update, refile, and query by state, priority, and tags. |
| **Notifications & messaging** | Send messages to configured channels (e.g. Telegram, Discord) and in-app notifications. |
| **Knowledge graph** | Query and update the Neo4j graph: entities, relationships, and namespaces. |
| **Data workspace** | Run SQL, list tables/schemas, and work with Data Workspace datasets. |
| **Text & content processing** | Summarization, extraction, and other in-process content tools. |
| **Web & crawling** | Fetch and parse web pages (Crawl4AI-based). |
| **Email** | Read and send email via configured providers (e.g. Microsoft Graph, IMAP). |
| **Monitor detection** | Tools used for monitor-style, change-aware workflows. |
| **Agent internal** | Session memory, clipboard, and other in-agent utilities. |
| **External integrations** | Connector-generated tools and plugins (e.g. Trello, Notion, CalDAV). |

## Tool packs vs Skills

- **Tool packs** — Groups of tool *access*: what the agent **can** do. Think permissions. You attach packs at the team or step level (e.g. discovery, email, data_workspace). Some packs support read-only vs full access.
- **Skills** — Reusable *procedural knowledge* (instructions) plus optional tool requirements: **how** to do it. Think know-how. Skills describe when and how to use tools; their required tools are auto-bound when the skill is used.

Pinned skills are always loaded. **Auto** or **Full** can also attach skills before the run by matching the step prompt (up to **max discovered skills**). **Catalog** or **Full** inject a **skills catalog** and add **acquire_skill** / **search_and_acquire_skills** for on-demand loading. The usual default for LLM agent and deep agent steps is **Auto**.

## Skill discovery modes (LLM agent and deep agent)

On **LLM agent** and **deep agent** steps, **Skill discovery** in **Tools & connections** controls (1) whether skills are **matched to the prompt before the run**, (2) whether a **skills catalog** (compact list of core skills) is injected into the system message, and (3) whether **`acquire_skill`** and **`search_and_acquire_skills`** are available so the model can load procedures during the step.

| Mode | Pre-run match to prompt | Skills catalog in prompt | `acquire_skill` / `search_and_acquire_skills` |
|------|-------------------------|---------------------------|-----------------------------------------------|
| **Off** | No | No | No (unless those tools are included another way) |
| **Auto** | Yes — up to **max discovered skills** | No | No (unless included another way) |
| **Catalog** | No | Yes — core skills listed; non-core still findable via search | Yes — both tools are added for this step |
| **Full** | Yes — same as **Auto**, up to **max discovered skills** | Yes — same as **Catalog** | Yes — same as **Catalog** |

**How this differs in practice**

- **Auto** is the default sweet spot: relevant skills are pulled in before the loop starts from the step prompt, without a long catalog block in the system message.
- **Catalog** is for when you want the model to **see every core skill by name** and explicitly load one with **`acquire_skill(slug)`** or **`search_and_acquire_skills(query)`**. There is **no** automatic pre-run skill matching from the prompt in **Catalog** mode — only skills you **pin** on the step are attached automatically before the model acts.
- **Full** combines **Auto** and **Catalog**: pre-run matching **plus** the catalog text **plus** acquisition tools. Use for rich, open-ended steps where both automatic procedures and on-demand loading help.

**Max discovered skills** — Caps how many skills the **pre-run** matcher may attach for **Auto** and **Full** (default 3, maximum 10). It does not limit pinned skills. It does not apply to **Catalog**-only pre-run (there is no pre-run matcher in **Catalog** mode).

**Two different “catalogs”** — The **tool** catalog (*AGENT_FACTORY_TOOLS.md*) lists callable tools and I/O. The **skills catalog** here is a run-time summary of **skills** (procedures) injected into the prompt when **Catalog** or **Full** is selected.

**LLM task** steps show skills and discovery in a reduced form: pinned skills and **Off** / **Auto** only—no **Catalog** or **Full**, and no mid-run skill acquisition.

## Tool packs, individual tools, and skill search in step configuration

On **LLM agent** and **deep agent** steps, the **Tools & connections** section lets you combine:

- **Tool packs & accounts** — Turn on a capability category (which maps to one or more packs), and choose external accounts where applicable.
- **Individual tools** — Expand a category and keep only specific tools, or add pinned tool names directly. This is useful when you need a narrow toolset.
- **Skills** — Pinned skills are always attached; discovery modes control *additional* skills and catalog visibility as in the table above.

In practice, this means you can start broad with packs, then narrow to individual tools, and choose how aggressively the run time should discover and expose procedural skills.

## Using tools in playbooks

In the Workflow Composer you add **tool steps** to a playbook and connect their outputs to other steps (e.g. “search documents” → “LLM analysis” → “write to file”). Required inputs can be set from constants, from the trigger payload, or from the outputs of previous steps. The full I/O contract for each tool is available in the Composer UI and in *AGENT_FACTORY_TOOLS.md*.
