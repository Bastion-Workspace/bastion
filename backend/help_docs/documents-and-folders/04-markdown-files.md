---
title: Markdown files
order: 4
---

# Markdown files

**Markdown** (`.md`) is a simple format for headings, lists, links, and basic formatting. Bastion lets you create and edit Markdown files in the Document Library, use them in search and in agents, and (when frontmatter is supported) use metadata like `type` and `ref_*` for document types and referenced files. This page describes what you can do with Markdown in the app.

---

## Creating and opening

- **Create** — Right‑click a folder in the Document Library and choose **New Markdown**. Name the file (e.g. `notes.md`). It’s created in that folder under the same scope (My, Team, or Global).
- **Open** — Click a `.md` file in the tree or from search. It opens in the document viewer. You can have multiple tabs open.

---

## Editing and structure

- **Editor** — The viewer provides an editor for the raw Markdown. You can use:
  - Headings: `#`, `##`, `###`, etc.
  - Bold, italic, links, lists (bulleted and numbered).
  - Code blocks and inline code.
- **Frontmatter** — Many Markdown files support **YAML frontmatter** at the top (between `---` lines). There you can set:
  - **type** — e.g. `fiction`, `outline`. Used by Agent Factory and editor features to route or scope (e.g. `editor_document_type`).
  - **ref_*** — References to other files (e.g. `ref_rules: ./rules.md`). Those files are loaded and exposed as `editor_refs_rules` (and similar) in Agent Factory prompts when the document is open.
- **Saving** — Changes are saved as you edit or when you save, depending on the editor. The file stays in the same folder unless you move it (right‑click → **Move**).

---

## Search and indexing

If the folder (and file) are **included in search** (default), Markdown content is indexed for semantic and keyword search. You can find the file from the search bar or from agent tools that search documents. If you set **Exempt from search** on the folder or the file, it won’t appear in search results.

---

## Use with agents and Agent Factory

- **Open document** — When a Markdown file is open in the editor, Agent Factory can use variables like `{editor}`, `{editor_document_id}`, `{editor_document_type}`, and `{editor_refs_*}` in playbook prompts. Section-scoping variables (e.g. `editor_current_section`, `editor_adjacent_sections`) are derived from `##` headings and cursor position. For each referenced file, you also get `{editor_refs_<category>_toc}`, `_current`, and `_adjacent`, plus `{editor_refs_<category>_section:Heading}` to pull a named section — see **Prompt variables and conditional blocks** in Agent Factory help.
- **Tools** — Agents can call tools to read, search, and update documents. Markdown is read and written as text; there’s no special “todo” or “contact” structure as in Org Mode. For task-like workflows, use Org files or structure your Markdown in a way the agent can parse from the text.

---

## Summary

- Create with **New Markdown** from a folder’s right‑click menu; edit in the document viewer.
- Use frontmatter for **type** and **ref_*** when you want document type and referenced files in Agent Factory.
- Markdown is searchable when included in search, and can be used by agents via document tools and editor variables. For TODOs, contacts, and tags that are cataloged and available in **All TODOs** and tools, use **Org Mode files** instead.
