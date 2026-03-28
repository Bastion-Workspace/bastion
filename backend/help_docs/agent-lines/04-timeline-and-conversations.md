---
title: Timeline and conversations
order: 4
---

# Timeline and conversations

The **line timeline** is the log of inter-agent messages, escalations, and system events for a line. **Moderated conversations** let an LLM steer when to continue, redirect, summarize, or conclude a multi-agent exchange. This page describes how to use the timeline and conversation tools.

---

## Line timeline

- **What it shows** — Messages between agents (e.g. CEO posts a summary, another agent replies), **escalations** (agent asks for human help), and **system events** (e.g. heartbeat started, task completed). Messages are ordered by time; you can open a **thread** to see replies under a message.
- **Where to view** — **Timeline** tab on the line dashboard. You can scroll through recent items; the list may support pagination or “load more” for older entries.
- **Message form** — You can **post a message** to the timeline (e.g. an instruction or note to the line). It appears as a human/orchestrator message and agents may react depending on their playbooks.

---

## Inter-agent conversations

- **What they are** — Instead of a single agent responding once, you can run a **conversation** where multiple agents take turns. Each agent sees the previous messages and can post a reply. The conversation continues until a **completion** condition is met or a **moderator** concludes it.
- **Conversation moderator** — When you start an agent conversation with a **moderator** enabled, an LLM periodically reviews the exchange and can:
  - **Continue** — Let the next agent speak.
  - **Redirect** — Send the thread to a different agent or topic.
  - **Summarize and continue** — Add a summary message and then continue.
  - **Conclude** — End the conversation and optionally produce a final summary.
- **Without moderator** — Conversations can run in a simple round-robin or until a heuristic “done” condition; the moderator gives you more control and avoids endless loops.

---

## How to read the timeline

- **Avatar and color** — Each message shows **who** sent it (agent name, avatar). Line members can have **colors** so you can quickly see which agent said what.
- **Content** — Message body (text, markdown, or structured content). Escalations may include a **preview** and a link to the **approval queue** or a refile action.
- **Threads** — Click a message to open its **thread** (replies). Useful for long chains or when an agent responded to a specific post.

---

## Notifications and the bell

- Line events (e.g. **heartbeat_failed**, **team_budget_exceeded**, **team_emergency_stop**) are sent to the **central notification** stream. They appear in the **bell** dropdown and, when relevant, as a **snackbar** on the line dashboard (filtered by line and subtype). So you can stay informed even when not on the dashboard.

---

## Where to go next

- **Analytics** — Message volume over time and other charts.
- **Dashboard and controls** — Emergency stop, budget, live indicator.
