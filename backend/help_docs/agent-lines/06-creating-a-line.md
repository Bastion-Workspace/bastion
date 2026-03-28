---
title: Creating a line
order: 6
---

# Creating a line

This page walks through creating an Agent Line from scratch: create the line, add a CEO and members, configure the heartbeat, and create a first goal.

---

## Step 1: Create the team

1. Go to **Agent Factory > Lines**.
2. Click **Create line** (or **Add line**).
3. Enter a **name** (e.g. “Political tracker”) and optionally a **description** and **mission statement**.
4. Set a **handle** (e.g. `political-tracker`) if you want to mention the line in chat with **@political-tracker**.
5. Save. You are taken to the line dashboard or settings.

---

## Step 2: Add members (CEO and others)

1. Open **Settings** for the line.
2. In the **Members** section, use the dropdowns to pick an **Agent** (an existing Agent Factory profile), **Role** (CEO, Manager, Worker, Specialist), and **Reports to** (another member, or “None” for the root).
3. Click **Add** to add the member. The **CEO** is the root of the org chart — the agent that runs on the **heartbeat**. Assign one member as CEO (root) by setting **Reports to** to “— None (root)” and giving that member the **CEO** role if you use roles to distinguish.
4. Add more agents and set **Reports to** so the org chart reflects who reports to whom.

You can later expand any member row to change **role**, **reports to**, and **additional tools** for that agent.

---

## Step 3: Configure heartbeat

1. In **Settings**, find the **Heartbeat** section.
2. Turn **Enable autonomous heartbeat** on.
3. Set **Interval (seconds)** (e.g. 900 for 15 minutes) or a **Cron expression** if you prefer a schedule.
4. Click **Save heartbeat**. The dashboard will show **Last beat** and **Next beat** once the first heartbeat runs.

The CEO agent runs with a **context summary** of the team: pending tasks, goals, escalations, recent messages, and workspace entries. The CEO can delegate work, post to the timeline, or request approvals.

---

## Step 4: Create a first goal

1. Open the **Goals** tab on the line dashboard.
2. Create a **goal** (e.g. “Track election news daily”). You can assign it to a member or leave it unassigned.
3. Use **Goal-to-task delegation** (from the CEO or from the UI) to break the goal into **tasks** with suggested assignees. Tasks appear on the **Task board** where you (or the CEO) can move them through statuses.

---

## Optional: Tools and skills

- In **Settings > Tools & Skills**, add **line tool packs** (e.g. discovery, knowledge) and **line skills** so all members have those capabilities when running in line context.
- Expand a member in **Members** to add **additional tools** for that agent only.

See **Tool assignment** and **Skills and playbooks** for when to use each.

---

## Where to go next

- **Tool assignment** — Line-level packs vs member-level tools.
- **Skills and playbooks** — Skills vs tools, line-level skills.
- **Workspace and communication** — @mentions, timeline, shared workspace.
