---
title: Dashboard and controls
order: 2
---

# Dashboard and controls

The **line dashboard** is the main view for a single Agent Line. You open it from **Agent Factory > Lines**, then select a line. This page describes the layout and the controls you use to monitor and control the line.

---

## Dashboard layout

- **Header** — Line name, back link to lines list, and shortcuts to **Timeline**, **Tasks**, **Goals**, **Analytics**, and **Settings**.
- **Run heartbeat now** — Button to trigger the CEO heartbeat immediately (in addition to the scheduled heartbeat). Disabled if a heartbeat is already running or budget is exceeded.
- **Live execution indicator** — When a heartbeat or agent invocation is running, a **pulsing green dot** (and optional label) appears so you can see the line is busy. The same status is shown in the **status bar** at the bottom of the app for any line with an active run.
- **Emergency stop** — Button to **halt** the current heartbeat or agent run for this line. Use it when you need to stop the line immediately. Confirmation is required.
- **Budget card** — Shows **monthly limit**, **current spend**, and optional **warning threshold**. If over limit, the card reflects that and heartbeats are skipped until the next period or you raise the limit.
- **Agent health** — Summary of agent activity (e.g. last 7 days): successes, failures, or trends. Helps spot failing or idle agents.
- **Org chart** — Visual graph of the line (who reports to whom). The root is the CEO. You can open **Settings** to add/remove members and change roles.
- **Goals and tasks** — Summaries or links to the full **Goals** and **Task board** views.
- **Recent timeline** — Latest inter-agent messages, escalations, and system events. Full history is on the **Timeline** tab.

---

## Status bar line indicators

When a team’s heartbeat (or an agent invocation in that line) is **running**, the **status bar** at the bottom of the app shows a compact **chip** for that line: a **pulsing dot** and the line name (truncated if long). Clicking the chip opens that line’s dashboard. When no line is running, the chips are hidden so the bar is uncluttered.

---

## Emergency stop

- **What it does** — Cancels the current Celery task for the team’s heartbeat or agent invocation. The run stops; any in-flight LLM or tool call may still complete on the server, but no further steps are executed.
- **When to use** — When you need to stop the line immediately (e.g. runaway loop, wrong configuration, or to free resources).
- **After stop** — The team’s **next beat** remains as scheduled; you can run **Run heartbeat now** again when ready.

---

## Budget monitoring

- **Settings > Budget** — Set **monthly limit** (e.g. USD), **warning threshold** (e.g. 80%), and whether to **enforce** a hard limit. When enforced, heartbeats and line agent invocations are blocked once the limit is reached.
- **Dashboard** — The budget card shows current spend vs limit and warns when near or over. Notifications (bell icon) can alert you to **team_budget_exceeded** or **team_budget_warning** so you can adjust limits or pause the line.

---

## Where to go next

- **Tasks and goals** — Task board, creating goals, delegation.
- **Timeline and conversations** — How to read the timeline and use the conversation moderator.
- **Analytics** — Charts and period selector.
