---
title: Tool assignment
order: 7
---

# Tool assignment

Agent Lines support a **three-layer** tool model: tools that always apply in line context, tools you add for the whole line, and tools you add per member. This page explains when to use each and how to configure them.

---

## The three layers

| Layer | Where to set | Applies to | Example |
|-------|----------------|------------|---------|
| **team_tools** | Automatic | Every agent when running in line context | send_to_agent, read_team_timeline, create_task_for_agent, write_to_workspace, etc. |
| **Line tool packs** | Settings > Tools & Skills | All line members | discovery, knowledge, task_management, notifications |
| **Member additional tools** | Settings > Members > expand row | That agent only | search_documents_tool, notify_user_tool |

**team_tools** are always injected when the request has a line context (heartbeat, or a run with line_id). You do not add “team_tools” in the UI for this — it’s automatic.

**Line tool packs** give the whole line a shared capability. For example, add the **discovery** pack so every member can search documents and the web when they run. Add **task_management** if you want every member to create and update todos.

**Member additional tools** are for one agent. For example, give only the CEO the **notifications** pack (or individual notify tools) so only the CEO can send user notifications; or give one specialist **image_generation** while others don’t have it.

---

## Access mode (read vs full)

When you assign a **line tool pack** or a **step-level tool pack** in the Composer, some packs support an access mode:

- **Full** — The agent gets all tools in the pack (read and write). Default.
- **Read** — The agent gets only the read-only subset of that pack (e.g. search, list, get). Write tools (create, update, delete, send) are not included.

Packs that have no write tools (e.g. **discovery**, **knowledge**, **math**) do not show a mode toggle. For packs that mix read and write tools (e.g. **document_management**, **email**), use **Read** when you want the line or step to query/list without modifying data. Example: give the team **read** access to **document_management** so they can read metadata and content but not create or update documents.

---

## Resolution order

Tools are merged in this order (later layers add or override):

1. **Step tools** — `available_tools` and **step-level tool packs** (with optional read/full mode) from the playbook step.
2. **Line tool packs** — Packs assigned to the team (with optional read/full mode), when the agent runs in line context.
3. **team_tools** — Automatically injected when running in line context (messaging, tasks, goals, workspace).
4. **Member additional tools** — Per-member extra tools (in **Advanced: Additional tools** in the member row).

Step-level tool packs are configured in the Workflow Composer on **LLM Agent** and **Deep Agent** steps, in the **Tool packs** section above the individual tool picker. Team tool packs are configured in **Settings > Tools & Skills** for the line.

---

## Where to configure

### Team tool packs

1. Open the line **Settings**.
2. Find **Tools & Skills**.
3. Under **Line tool packs**, click pack names to select or deselect. Selected packs appear as filled chips.
4. For packs that support it, use the **Read** / **Full** toggle to choose access mode (read-only subset vs all tools).
5. Click **Save tools & skills**.

Available packs match the Workflow Composer catalog: text_transforms, session_memory, planning, discovery, knowledge, knowledge_graph, rss, document_management, file_management, org_management, task_management, math, utility, contacts, notifications, email, calendar, navigation, data_workspace, image_generation, visualization, data_connection_builder, browser, local_device, and team_tools (optional extra; normally automatic).

### Step-level tool packs (Composer)

In the Workflow Composer, when editing an **LLM Agent** or **Deep Agent** step, use the **Tool packs** section to attach packs to that step (with optional Read/Full mode). These apply in addition to team packs and step-level **Available tools**.

### Member additional tools

1. Open the line **Settings**.
2. In **Members**, find the member and click the **expand** (chevron) icon.
3. In the expanded section, open **Advanced: Additional tools** to select individual tools (grouped by category). These are added on top of the member’s playbook tools and any team packs. Prefer configuring tools in the playbook or team packs when possible.
4. Optionally change **Role** or **Reports to** in the same section.
5. Click **Save** to persist changes for that member.

---

## When to use which

- **Line packs** — Use when you want the whole line to have a capability (e.g. “everyone can search documents and read the knowledge base”). Avoid giving every member heavy or sensitive packs (e.g. browser, data_connection_builder) unless the whole line needs them.
- **Member tools** — Use when only one or a few agents need a tool (e.g. only the CEO sends notifications, or only one specialist generates images). Reduces noise and cost for other members.
- **team_tools** — Always present in line context; no configuration needed. Use for messaging, tasks, goals, workspace, and governance.

---

## Where to go next

- **Skills and playbooks** — How line-level skills combine with playbook steps.
- **Workspace and communication** — Shared workspace and timeline.
