---
title: Analytics
order: 5
---

# Analytics

The **Analytics** tab for an Agent Line shows charts and metrics for task throughput, cost, goal progress, agent activity, and message volume. You can change the **period** (e.g. last 7, 14, or 30 days) to focus on the range you care about.

---

## Available charts

- **Task throughput** — Over time (e.g. per day): how many tasks were **created** and how many were **completed**. Helps you see if the line is keeping up with new work or if a backlog is growing.
- **Cost over time** — Spend (e.g. USD) per day or per period. If a **monthly budget** is set, the chart can show a **reference line** so you can compare trend vs limit.
- **Goal progress** — Horizontal bar chart of goals and their **progress** (e.g. percentage or completed vs active). Useful to see which goals are on track.
- **Agent activity** — Per-agent counts (e.g. tasks completed, messages sent, or runs) over the selected period. Surfaces which agents are busiest or idle.
- **Message volume** — Timeline message count over time (e.g. per day). Shows conversation and escalation activity.

---

## Period selector

- **Control** — A dropdown or buttons let you choose the **number of days** (e.g. 7, 14, 30) to include in the analytics. The backend returns aggregated data for that range; charts and tables update accordingly.
- **Use case** — Short periods for recent sprint checks; longer periods for monthly or quarterly reviews.

---

## How the data is produced

- **Backend** — The dashboard calls an **analytics** endpoint (e.g. `GET /api/agent-factory/lines/{lineId}/analytics?days=30`). The server queries the **agent_tasks**, **agent_goals**, **agent_messages**, and budget/spend tables and returns pre-aggregated series (e.g. task counts per day, goal progress list, message counts). No raw data is streamed; the frontend just renders the charts.
- **Frontend** — Charts are built with a library such as **Recharts** (area, line, bar). The analytics page is read-only; it does not create or edit goals or tasks.

---

## Where to go next

- **Dashboard and controls** — Budget card, emergency stop, live indicator.
- **Tasks and goals** — Task board and goal-to-task delegation.
- **Timeline and conversations** — Message volume in context.
