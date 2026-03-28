---
title: Tools reference
order: 2
---

# Agent Factory: Tools reference

Agents use **tools** to read files, search, run tasks, send notifications, query the knowledge graph, and call external systems. All tools are registered with typed input/output contracts so the Workflow Composer can wire step outputs to later step inputs.

## Tool categories

The following categories are available. Each category contains multiple tools; the full catalog with parameters and return types is maintained in the repository (see *AGENT_FACTORY_TOOLS.md*).

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

When **Auto-discover skills** is on (the default for LLM Agent and Deep Agent steps), the agent finds relevant skills from the step prompt via semantic search. You can still assign skills manually for overrides or when you want to lock in a specific procedure.

## Using tools in playbooks

In the Workflow Composer you add **tool steps** to a playbook and connect their outputs to other steps (e.g. “search documents” → “LLM analysis” → “write to file”). Required inputs can be set from constants, from the trigger payload, or from the outputs of previous steps. The full I/O contract for each tool is available in the Composer UI and in *AGENT_FACTORY_TOOLS.md*.
