---
title: Skills
order: 6
---

# Skills

**Skills** bundle **procedure** text (instructions the model should follow) with optional **tool lists** and metadata. They let you reuse the same “how to do this job” across agents and playbook steps without duplicating long prompts.

---

## What a skill is

A skill typically includes:

- **Procedure** — Multiline instructions (markdown or plain text) injected into the run so the model follows a consistent method.
- **Required tools** — Tool names that are **auto-bound** when the skill is active, so the procedure can rely on those actions being available.
- **Optional tools** — Extra tools the skill may use; resolution depends on how the skill is loaded and what the step allows.
- **Metadata** — **Name**, **slug** (stable id), **category**, **description**, **tags**.
- **Required connection types** — If set, the skill is only auto-suggested when the user has matching connection types (e.g. email, calendar).
- **Depends on** — Other skill **slugs** whose tools and procedures are pulled in automatically at runtime.

---

## Built-in vs user-authored

- **Built-in skills** — Shipped with the product; **read-only** in the Skill Editor. Use **Duplicate to My Skills** to get an editable copy (typically `{slug}-custom`).
- **User skills** — Created in your workspace; fully editable and deletable (subject to locks/shares).

---

## Core skills (`is_core`)

Skills marked **core** appear in the **condensed skills catalog** injected when a step uses **Catalog** or **Full** skill discovery (see **Tools reference → Skill discovery modes**). Non-core skills remain discoverable via semantic search and acquisition tools at runtime.

---

## Browsing and creating skills

- Open **Agent Factory** and use the **skills list** to browse built-in and your skills.
- **Create** a skill from the UI, or use the **`create_skill`** meta-tool from an automation-capable context (preview / confirm flow applies).
- The **slug** is **fixed after creation**; choose it carefully when creating a new skill.

---

## Editing skills (Skill Editor)

For user-owned skills you can edit:

| Area | Purpose |
|------|--------|
| **Name** | Display name |
| **Slug** | Read-only after creation |
| **Category** | Organization (e.g. search, org) |
| **Description** | Short summary |
| **Procedure** | Main instruction body for the model |
| **Required tools** / **Optional tools** | Autocomplete from registered **actions** |
| **Required connection types** | Presets (email, calendar, code platform, …) plus custom types |
| **Depends on** | Other skill slugs |
| **Tags** | Free-form labels |
| **Core skill** toggle | Whether this skill is listed in the injected catalog |

Built-in skills show the same fields **read-only**, with **Duplicate to My Skills** to customize.

---

## Versioning (candidates)

- **Save as Candidate** stores an **A/B candidate** version with an adjustable **weight**.
- Use **Skill Candidate Panel** to **promote** a winning candidate or **reject** it.
- **Metrics** (when available) include total uses, success rate, unique agents, recent use, and last-used time.

---

## Sharing

- **Shared skills** show an **info banner**; use **Make my own copy** to duplicate into your workspace for editing.
- Duplicating a built-in skill creates a **`{slug}-custom`** (or similar) user skill you own.

---

## How skills are used at runtime

- **Pinned on steps** — Set **skill_ids** on **LLM agent**, **deep agent**, or **LLM task** steps so those skills always load for that step.
- **Discovery** — **Off**, **Auto**, **Catalog**, or **Full** controls pre-run matching, catalog injection, and **acquire_skill** / **search_and_acquire_skills** (see **Tools reference**).
- **Required tools** from active skills are **merged** into the effective tool set for the run.

---

## Summary

Skills are reusable **procedure + tools + metadata**. Built-in skills are templates; user skills are yours to edit. Core skills power the **catalog** text; discovery modes control how skills are matched and acquired. Wire skills on steps or rely on discovery; use **Tools reference** for the discovery matrix.

---

## Related

- **Tools reference** — Skill discovery modes and tool categories.
- **Playbook steps and flow control** — **Tools & connections** on LLM and deep agent steps.
- **Profile settings** — Agent-level defaults that interact with skills and tools.
