---
title: Skills and playbooks
order: 8
---

# Skills and playbooks

Agent Line members are **agent profiles** with their own **playbooks** and optional **skills**. When they run in line context, **line-level skills** are added on top of their step-level skills. This page explains how skills and playbooks work together in lines.

---

## Skills vs tools

- **Tools** — Actions the agent can call (search, read file, send message, etc.). They are attached at the **playbook step** (per-step tools and tool packs) and, in teams, at **line level** (line tool packs) and **member level** (additional tools).
- **Skills** — Reusable **procedural knowledge** (text instructions) and optional tool requirements. Skills shape how the agent reasons and which tools it may use. They are attached at the **playbook step** (step skill_ids) and, in teams, at **line level** (line skills).

So: each member has a playbook with steps; each step can have **tools** and **skills**. When the member runs in line context, the system adds **line tool packs** and **line skills** to what that step already has.

---

## Line-level skills

In **Settings > Tools & Skills**, under **Line skills**, you select skills that apply to **all** line members whenever they run in line context. For example, add a “Multi-round research” skill so every member follows that procedure when researching; or add “Structured reporting” so every member formats outputs consistently.

Team skills are **merged** with the step’s own skill_ids: the agent sees both the step’s skills and the team’s skills. Duplicates are avoided. Order is: step skills first, then team skills.

When you pick skills (in the line settings or in the Composer step config), each skill can show a **+N tools** badge or tooltip listing its **required_tools**. Those tools are auto-bound when the skill is used, so you don't need to add them manually to the step or team.

---

## Playbook steps and team context

When an agent runs as part of a line (heartbeat or team-triggered run):

1. The **playbook step** (e.g. llm_agent) has its own **available_tools**, **tool_packs**, and **skill_ids**.
2. The system injects **team_tools** (always) and any **line tool packs** you configured.
3. The system injects any **member additional tools** for this agent.
4. The system injects **line skills** into the step’s skill list before resolving skills and tools.

So the final set of tools and skills is: step tools + step tool_packs + team_tools + team tool packs + member additional tools; and step skills + team skills.

---

## Best practices

- **Line skills** — Use for behavior you want **every** member to follow (e.g. “always cite sources”, “always summarize in three bullets”). Keep team skills focused so they don’t conflict with step-specific skills.
- **Step skills** — Use for step-specific behavior (e.g. “in this step, do multi-round research”). Step skills are defined in the playbook in the Workflow Composer.
- **Combining** — Line skills are good for cross-cutting concerns; step skills for step-specific ones. Both can be used together.

---

## Where to go next

- **Tool assignment** — Team vs member tools.
- **Workspace and communication** — Shared workspace and timeline.
