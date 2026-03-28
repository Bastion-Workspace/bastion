---
title: Org Mode files
order: 5
---

# Org Mode files

**Org Mode** (`.org`) is a structured format for outlines, TODOs, agendas, contacts, and tags. Bastion parses org files and surfaces TODOs, contacts, and tags in dedicated views and in tools that agents and playbooks can use. This page explains how org files work in the Document Library and how TODOs and related data are cataloged and used.

---

## Creating and opening

- **Create** — Right‑click a folder and choose **New Org-Mode**. Name the file (e.g. `tasks.org`). It’s created in that folder (My, Team, or Global).
- **Open** — Click an `.org` file in the tree or from search. It opens in the Org editor. You can fold headings, edit text, and toggle TODOs.
- **Org tools in the sidebar** — When you have at least one org file, the Documents sidebar shows org-related entries: **Search Org Files**, **All TODOs**, **Contacts**, and **Tags Browser**. Use these to work across all your org files without opening each file.

---

## TODOs and how they’re cataloged

Org TODOs are **headings** that start with a keyword like `TODO`, `DONE`, or `NEXT`. For example:

```org
* TODO Write report
* DONE Call client
* NEXT Review pull request
```

Bastion **parses** all your org files (in scope) and extracts:

- **State** — TODO, DONE, NEXT, WAITING, etc.
- **Priority** — e.g. `[#A]`, `[#B]`, `[#C]`.
- **Dates** — DEADLINE, SCHEDULED.
- **Tags** — `:work:urgent:` on the heading.
- **File and location** — Which file and heading the item lives in.

This catalog is what powers **All TODOs** and the **todo tools**.

---

## All TODOs view

Open **All TODOs** from the Documents sidebar (e.g. under the org section). You get a single list of TODO items from **all** your org files (and team org files if you have access). You can:

- **Filter** — By state (e.g. active, done, upcoming), tag, priority, category, or file.
- **Sort** — By file, state, date, or priority.
- **Click an item** — Opens the org file at that heading so you can edit or toggle.
- **Toggle state** — Mark as DONE or back to TODO from the view or in the file.
- **Refile** — Move a TODO to another heading or file (when refile is supported).
- **Bulk archive** — Archive all DONE items in a file.

So you don’t have to open each org file to see or manage tasks — use All TODOs as your task list. The same data is available to **tools** (e.g. list todos, toggle todo, create todo) so agents and playbooks can read and update your tasks.

---

## Contacts view

If your org files contain **contact** entries (e.g. with certain properties or structure), they’re extracted and shown in the **Contacts** org view. Open it from the Documents sidebar. You can browse and open the source org file from there. Contact data can also be used by tools when the instance supports it.

---

## Tags Browser

Headings in org files can have **tags** (e.g. `:work:urgent:`). The **Tags Browser** (Documents sidebar) lets you browse and filter by tag across all org files. Useful to find everything tagged with a given tag without opening each file.

---

## Tools and agents

Tools that work with **todos** operate on the same catalog that All TODOs uses:

- **List todos** — Returns TODOs matching scope, state, tag, etc. (e.g. “all”, “inbox”, or a file path).
- **Toggle todo** — Flips state (e.g. TODO ↔ DONE) for a given item (identified by file and line or heading).
- **Create todo** — Adds a new TODO in a specified org file (and optionally heading).
- **Refile** — Moves a TODO to another heading or file.

When you build **Agent Factory** playbooks or use agents that “manage my tasks”, they use these tools against your org files. So Org Mode is the format that gives you **cataloged** TODOs (and contacts/tags) in both the UI and in automation. Markdown files don’t get this treatment — they’re just text for search and read/write tools.

---

## Summary

- **Org Mode** is for outlines, TODOs, dates, contacts, and tags. Create with **New Org-Mode** from a folder; edit in the Org editor.
- **TODOs** from all org files are cataloged and shown in **All TODOs**. You can filter, sort, toggle, refile, and bulk archive.
- **Contacts** and **Tags** have dedicated views (Contacts, Tags Browser) and can be used by tools.
- **Tools** (list todos, toggle, create, refile) use the same catalog — so agents and playbooks can manage your tasks without opening files manually. For simple notes without task/contact structure, use **Markdown files** instead.
