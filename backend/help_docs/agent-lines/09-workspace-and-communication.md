---
title: Workspace and communication
order: 9
---

# Workspace and communication

Agent Lines share a **timeline** of inter-agent messages and a **workspace** (shared key-value scratchpad). You can mention a line in chat with **@line-handle** to get a summary. This page explains how these work and how the heartbeat context uses them.

---

## @line-handle in chat

If you set a **handle** for the line (e.g. `political-tracker`), you can type **@political-tracker** in chat. The system resolves it to that line and injects a **line chat context** summary into the reply: goals, tasks, budget, and recent activity. You don’t open the dashboard; you just ask “what’s new?” or “summarize goals” and the chat agent has the line summary in context.

The line handle must be unique. It is set in **Settings > General > Handle**.

---

## Line timeline

The **line timeline** lists inter-agent messages, escalations, and system events in chronological order. Agents use the **send_to_agent** and related team tools to post messages; the CEO sees recent messages in the **heartbeat context** and can reference them when delegating or reporting.

You can open the **Timeline** tab on the line dashboard to read the full history. Agents can call **read_team_timeline** to fetch recent messages when running in line context.

---

## Shared workspace (Blackboard)

The **workspace** is a shared key-value store for the line. Agents use **write_to_workspace(key, value)** and **read_workspace(key)** (or read all keys) to share artifacts — e.g. a “current_draft” or “research_summary” that another agent can pick up.

- **write_to_workspace** — Sets or updates a key. Optional “updated by agent” is recorded.
- **read_workspace** — With a key, returns that entry’s value; without a key, returns a list of keys (and metadata) so the agent can discover what’s there.

Workspace entries are included in the **heartbeat context** (e.g. “WORKSPACE (shared scratchpad): key1 (updated by AgentName, 5m ago)”). The CEO can read the workspace to coordinate.

---

## Heartbeat context in brief

When the **heartbeat** runs, the CEO agent receives a **status briefing** that includes:

- **Escalations** — Tasks or messages that need action.
- **Stalled tasks** — Tasks in progress with no update for a long time.
- **Active tasks** — Pending and in-progress tasks with assignees and timestamps.
- **Goals** — Goal tree and unassigned active goals.
- **Recent agent activity** — Who did what in the last 24 hours (messages, tasks, executions).
- **Recent messages** — Last N timeline messages (sender, recipient, content, time).
- **Workspace** — Keys and who last updated them.
- **Approvals** — Pending governance approvals.
- **Budget** — Current spend vs limit and warnings.

This briefing is built automatically so the CEO can decide what to do next (delegate, post, escalate, or request approvals) without you configuring it. Team tool packs and team skills apply to the CEO during the heartbeat as well.

---

## Where to go next

- **Dashboard and controls** — Run heartbeat now, emergency stop, status bar.
- **Timeline and conversations** — Full timeline and conversation moderator.
- **Tasks and goals** — Task board and goal-to-task delegation.
