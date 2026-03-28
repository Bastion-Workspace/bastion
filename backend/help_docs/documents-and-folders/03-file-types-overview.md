---
title: File types: Markdown and Org Mode
order: 3
---

# File types: Markdown and Org Mode

The Document Library treats two text formats as first-class: **Markdown** (`.md`) and **Org Mode** (`.org`). Both can be created from the folder right‑click menu, edited in the app, and searched. They differ in structure and in how they’re used by features like TODOs, agendas, and tools. This page gives an overview; **Markdown files** and **Org Mode files** go into detail.

---

## When to use which

**Markdown** is best when you want:

- Simple formatting — headers, lists, links, bold, italic.
- Notes, documentation, wikis, or any free-form text.
- No built-in notion of “tasks” or “contacts” — just headings and paragraphs.
- Compatibility with many editors and static sites (GitHub, Notion, etc.).

**Org Mode** is best when you want:

- **TODOs** — Task items with states (TODO, DONE, etc.), priorities, and dates. These are cataloged and show up in **All TODOs** and in tools (list todos, toggle, refile).
- **Agenda** — Dates, deadlines, and scheduled items in one place.
- **Contacts** — Structured contact entries, viewable in the **Contacts** org view.
- **Tags** — Per-heading tags and a **Tags Browser** across org files.
- **Outlines** — Strong hierarchy (headings and subheadings) with folding and navigation.

So: use **Markdown** for straightforward docs and notes; use **Org Mode** when you care about tasks, dates, contacts, and tags that the app (and agents) can read and update.

---

## Creating and editing

- **New Markdown** — Right‑click a folder → **New Markdown**. You get an empty `.md` file. Edit in the document viewer; support for frontmatter (YAML at the top) depends on the editor.
- **New Org-Mode** — Right‑click a folder → **New Org-Mode**. You get an empty `.org` file. Edit in the Org editor; you can add headings, TODOs, and properties.

Both file types are stored in the Document Library like any other document. They can live in My Documents, Teams, or Global Documents. They’re indexed for search (unless you set vectorization to “Don’t vectorize”) and can be opened by agents and tools when you reference them (e.g. by path or document ID).

---

## How TODOs and org data are cataloged

Org Mode files are **parsed** for structure:

- **TODOs** — Lines like `* TODO Task name` or `* DONE Done task` are extracted. State, priority, deadline, scheduled date, and tags are stored. They appear in the **All TODOs** view (Documents sidebar → **All TODOs**) and can be filtered by state, tag, priority, and file. You can toggle state, refile, and archive from that view or from the file.
- **Contacts** — Contact entries (e.g. with `:PROPERTIES:` and `:CONTACT:` or similar) are collected and shown in the **Contacts** org view.
- **Tags** — Heading-level tags are indexed and browseable via the **Tags Browser** org view.

Markdown files don’t have this parsing: there’s no “All TODOs” or “Contacts” for plain Markdown. If you need tasks and contacts that the app and tools understand, use Org Mode.

---

## Tools and UI

- **Search** — Both Markdown and Org files are searchable (semantic and keyword) when included in search. Results show the file and relevant snippets.
- **All TODOs** — Lists TODOs from **all** your Org files (and team org files if applicable). Filter by state (e.g. active, done), tag, priority, file; click to open the file at that item. Available from the Documents sidebar.
- **Contacts** — Lists contacts extracted from Org files. Open from the Documents sidebar.
- **Tags Browser** — Browse and filter by tags from Org headings. Open from the Documents sidebar.
- **Agent Factory / tools** — Tools like “list todos”, “toggle todo”, “create todo”, “refile” operate on Org TODOs. When a playbook or agent needs task lists, it uses these tools against your org files. Markdown content can be read and edited as text but doesn’t get structured todo/contact handling.

For full behavior of each format, see **Markdown files** and **Org Mode files**.
