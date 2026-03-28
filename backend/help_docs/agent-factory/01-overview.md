---
title: Agent Factory overview
order: 1
---

# Agent Factory overview

Agent Factory is where you design and run **custom agents** — assistants that follow workflows you define, use the tools you choose, and can work with your documents, calendar, and external services. You don’t need to write code: you compose **playbooks** (step-by-step workflows), attach **tools** and **skills**, and optionally connect **data sources**. This page introduces the main ideas and points you to the details.

---

## What you can do with it

- **Custom assistants** — Create agents with a clear purpose (e.g. “morning briefing”, “fiction editor”, “research summarizer”) and a unique @handle so you can mention them in chat.
- **Workflows, not scripts** — Define what the agent does as a sequence of steps: run a tool, ask an LLM to analyze or write, maybe pause for your approval, then continue. You wire data between steps in the Composer.
- **Your data in context** — Use the open document, referenced files, conversation history, and date/time as variables in prompts so the agent has the right context every time.
- **One place to manage it** — Agent profiles, playbooks, skills, tools, and data connections are all configured in the Agent Factory UI (Settings or the dedicated Agent Factory area).

---

## The main pieces

### Agent profiles

An **agent profile** is the top-level object. It has:

- A **name** and **@handle** (e.g. `@nonprofit-investigator`) so you can invoke it in chat.
- A **playbook** — the workflow that runs when the agent is triggered.
- Optional **skills** (e.g. multi-round research, entity extraction) that shape how it reasons and which tools it uses.
- Optional **data connections** (connectors) for external APIs, calendars, or databases.
- Settings for where results go (knowledge graph, vector store, documents, etc.) and how the agent is run (interactive, scheduled, or on demand).

You create and edit profiles in the Agent Factory UI. Each profile is a single, coherent “agent” that you can @mention or run on a schedule.

### Playbooks

A **playbook** is the workflow definition: an ordered list of **steps**. Each step does one thing — call a tool, run an LLM task, branch on a condition, or wait for your approval. Steps can pass data to the next: the output of “search documents” can feed into an “LLM analysis” step, whose output can then be sent by email or written to a file.

You build playbooks in the **Workflow Composer** by adding steps, setting their inputs (often from previous steps or from runtime variables like `{query}` or `{today}`), and optionally adding branches or approval gates. The **Prompt variables and conditional blocks** help topic explains which variables exist and how to use conditional blocks so parts of a prompt only appear when relevant (e.g. when a document is open).

### Tools

**Tools** are the actions an agent can perform: search documents, read or update a file, run SQL, send a notification, call an external API, and so on. Each tool has defined inputs and outputs so the Composer can suggest and validate wiring between steps.

You don’t implement tools in the UI — you choose from the registered set and wire their outputs into later steps. The **Tools reference** topic in this section lists categories and how to use tools in playbooks; the full catalog lives in the repository docs (*AGENT_FACTORY_TOOLS.md*).

### Skills

**Skills** are reusable bundles of behavior (e.g. “multi-round research”, “entity extraction”) that you attach to a profile. They influence how the agent reasons and which tools it tends to use. The technical guide describes the skill registry and schema; in the UI you pick skills when editing a profile.

### Data connections (connectors)

**Connectors** are declarative definitions of external data sources: REST APIs, GraphQL, web scrapers, databases. An agent profile can be bound to one or more connector instances (with credentials and config). At runtime, playbook steps can “execute connector” to call those APIs or run those scrapers as part of the workflow.

---

## How agents run

- **Interactive (chat)** — You @mention the agent or open a chat with it. The playbook runs in the context of that conversation; you can stream responses and approve steps when required.
- **Scheduled** — The profile can be set to run on a schedule (e.g. daily briefing). The playbook runs with no open chat; results go to the destinations you configured.
- **On demand / background** — Triggers (e.g. “Run” button, webhook, or folder watch) can start a playbook in the background.

Results can be written to the **Knowledge Graph** (Neo4j) and **vector store** (Qdrant) so the system can reuse that knowledge later. Output can also go to documents, the Data Workspace, or external destinations depending on the profile.

---

## Where to go next

- **Playbooks overview** — What playbooks are, how they run, and how data flows between steps.
- **Playbook steps and flow control** — Step types (tool, llm_task, llm_agent, deep_agent, approval, branch, loop, parallel) and when to use each.
- **Tools reference** — Categories of tools and how to use them in playbooks.
- **Prompt variables and conditional blocks** — What variables you can use in prompts and step inputs (e.g. `{query}`, `{editor}`, `{today}`), how to reference previous steps, and how to hide sections when data is missing using `{{#var}}...{{/var}}`.
- **Data Connectors overview** — What connectors are and when to use them.
- **Creating and configuring connectors** — Base URL, auth, endpoints, and testing.

For architecture, examples, and implementation details, see the repository docs: *AGENT_FACTORY.md*, *AGENT_FACTORY_TECHNICAL_GUIDE.md*, and *AGENT_FACTORY_EXAMPLES.md*.
