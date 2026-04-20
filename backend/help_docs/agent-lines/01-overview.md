---
title: Agent Lines overview
order: 1
---

# Agent Lines overview

**Agent Lines** are autonomous groups of your custom agents that work together on goals, tasks, and conversations. Unlike a single agent you @mention in chat, a line has a **CEO agent** (or root member), a **heartbeat** that runs on a schedule, **goals** and **tasks**, and **governance** (e.g. approvals for hiring new agents or changing strategy). This page introduces the main ideas.

---

## What you can do with Agent Lines

- **Autonomous lines** — Add agents to a line, assign a CEO (root in the org chart), and enable a **heartbeat**. The CEO runs on a schedule, sees a summary of tasks, goals, escalations, and messages, and can delegate work, post to the timeline, or request approvals.
- **Goals and tasks** — Define high-level **goals** and break them into **tasks** assigned to line members. Use the **task board** to move tasks through statuses (e.g. todo → in progress → done). The CEO (or you) can use **goal-to-task delegation** to turn a goal into a set of tasks with suggested assignees.
- **Timeline and conversations** — All inter-agent messages, escalations, and summaries appear on the **line timeline**. You can run **moderated conversations** so an LLM steers when to continue, redirect, summarize, or conclude.
- **Budget and safety** — Set a **monthly budget** and optional **emergency stop** so you can cap spend and halt a running heartbeat or agent invocation.

---

## Key concepts

### CEO agent and org chart

- The **org chart** defines who reports to whom. The **root** member is the **CEO** — the agent invoked during a **line heartbeat**.
- The CEO receives a context summary (pending tasks, goal progress, escalations, budget, recent messages) and can delegate, post, or request approvals. Other agents are invoked when the CEO (or governance) assigns work.

### Heartbeat

- A **heartbeat** is a scheduled run of the CEO agent with line context. You choose a **periodic schedule**: **none** (no automatic cadence), **interval** (every *N* seconds, minimum 60), or **cron** (wall-clock times in an **IANA time zone**). The backend stores the next run in **next beat**; a worker checks about every 60 seconds and enqueues a heartbeat when due, then updates **last beat** / **next beat**.
- Heartbeats respect **line budget**. If the monthly limit is exceeded, the heartbeat is skipped and you get a notification.

### Goals and tasks

- **Goals** are hierarchical (parent/child). They can be active, completed, or cancelled. **Tasks** are flat items tied to a goal (or unassigned), with statuses such as todo, in progress, done. The **task board** shows columns by status; you can drag and drop to change status.
- **Goal-to-task delegation** is a tool the CEO (or you) can use: an LLM turns a goal into 2–5 tasks with suggested assignees, and the system creates those tasks and links them to the goal.

### Governance

- **Governance policies** can require **approvals** for certain actions (e.g. hire a new agent, change strategy). When an agent requests such an action, it appears in your **approval queue**; you approve or deny, and the system continues accordingly.

### Three-layer capability model

When an agent runs **in line context** (e.g. during a heartbeat or when the line is set on the request), three layers of tools and skills apply:

1. **Line-level tool packs and skills** — In **Settings > Tools & Skills** you choose which **tool packs** (e.g. discovery, knowledge, task_management) and **skills** apply to **all** line members. These are added on top of each member’s playbook tools and skills.
2. **Member-level additional tools** — In **Settings > Members**, expand a member row to give that agent **extra individual tools** (e.g. a specific search or notification tool). These apply only when that agent runs.
3. **team_tools** — The system always injects the **team_tools** pack in line context: messaging, tasks, goals, workspace, governance. You don’t need to add it manually.

Together, this lets you give the whole line a shared capability set (e.g. “everyone can search documents”) while giving specific members extra tools (e.g. the CEO can also send notifications). See **Tool assignment** and **Skills and playbooks** for details.

---

## How lines differ from single agents

| Single agent | Agent line |
|--------------|------------|
| You @mention or schedule one profile | A CEO runs on a heartbeat; multiple agents collaborate |
| One playbook, one set of tools | Org chart, goals, tasks, timeline, budget |
| Results go to documents, graph, etc. | Results plus inter-agent messages, escalations, approvals |

You still create **agent profiles** and **playbooks** in Agent Factory; lines **use** those profiles as members. Give a line a **handle** (e.g. `political-tracker`) and you can **@political-tracker** in chat to get a summary of that line’s status without opening the dashboard.

---

## Where to go next

- **Creating a line** — Step-by-step: create line, add CEO and members, configure heartbeat, first goal.
- **Dashboard and controls** — Dashboard layout, live execution indicator, emergency stop, budget.
- **Tasks and goals** — Creating goals, task board, drag-and-drop, goal-to-task delegation.
- **Timeline and conversations** — Line timeline, inter-agent messaging, conversation moderator.
- **Tool assignment** — Line-level packs vs member-level tools, when to use each.
- **Skills and playbooks** — Skills vs tools, line-level skill injection, combining with per-step skills.
- **Workspace and communication** — @mentions, line timeline, shared workspace, heartbeat context.
- **Analytics** — Charts for task throughput, cost, goal progress, agent activity, message volume.
