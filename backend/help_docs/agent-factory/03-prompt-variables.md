---
title: Prompt variables and conditional blocks
order: 3
---

# Prompt variables and conditional blocks

When you build a playbook, you often want step prompts to use **live data**: the user’s question, the current date, the document they have open, or the output of a previous step. Agent Factory supports this through **runtime variables** and **conditional blocks**. This guide explains what’s available and how to use them so your prompts stay clear and relevant.

---

## What are runtime variables?

**Runtime variables** are placeholders that get replaced with real values when the playbook runs. You don’t set them in the Composer — the system fills them in from the conversation, the calendar, the open file, and so on.

Think of them as “magic” names you can drop into your prompt text or into step inputs. When the step runs, each `{variable_name}` is replaced with the actual value (or left empty if there’s no data).

---

## Variables you can use

Here’s what’s available. Use the exact names below; the system only recognizes these (and a few special patterns we’ll cover).

### Conversation and context

| Variable | What it contains |
|---------|------------------|
| **query** | The user’s current message or request. |
| **history** | Recent conversation turns, formatted as `USER: …` and `ASSISTANT: …` lines. Only populated when **Include history in prompt** is enabled on the agent profile and you use `{history}` in the step prompt; if you don't include `{history}` in the prompt, no history is sent. |
| **query_length** | Character count of the user's message (as a number). Use in expression conditionals (e.g. `{{#query_length > 200}}`) for long-query prompts. |
| **trigger_input** | Payload from the trigger (e.g. a scheduled run or webhook), when present. |

#### How to inject conversation history

1. **On the agent profile** — In the profile editor (Identity or settings), turn on **Include history in prompt** and set **Chat history lookback** (number of recent exchanges, e.g. 10). This controls how many messages are available to fill `{history}`.
2. **In the playbook step** — In the step's prompt (or prompt template), include the variable **`{history}`** where you want the conversation to appear. For example:

   ```text
   === Recent conversation ===
   {history}
   === End conversation ===

   User's latest message: {query}
   ```

   If the profile has history enabled but the step prompt does not contain `{history}`, no conversation context is sent to the model. History is injected only where you place `{history}`.

### Date and time

| Variable | What it contains |
|----------|------------------|
| **today** | Start of today (e.g. `2026-03-06T00:00:00`) in the user’s timezone. |
| **today_end** | End of today (e.g. `2026-03-06T23:59:59`). |
| **tomorrow** | Start of tomorrow. |
| **today_day_of_week** | Day name (Monday, Tuesday, etc.) in the user's timezone. Empty if datetime context is disabled. Useful for weekday-aware scheduled playbooks. |

Use these when you need “today’s date” in prompts or in tool inputs (e.g. calendar or briefing steps). Don’t invent names like `{today_iso}` or `{current_date}` — stick to `{today}`, `{today_end}`, and `{tomorrow}`.

### The open document (editor)

When the user has a document open in the editor, you get a rich set of variables. If nothing is open, the editor-related variables are empty.

| Variable | What it contains |
|---------|------------------|
| **editor** | The full content of the open file (including a “FILE: …” header). Empty if no file is open. |
| **editor_document_id** | The document’s ID. Use this when a tool needs a document ID (e.g. to save edits). |
| **editor_filename** | Base filename of the open file (e.g. `chapter_01.md`). Empty if no file is open. Use in branch conditions or prompts to route by filename. |
| **editor_document_type** | The document “type” from frontmatter (e.g. `fiction`, `outline`), lowercased. Great for branching: “if this is a fiction doc, do X; if outline, do Y.” |
| **editor_length** | Character count of the open file body (as a number). Use in **expression conditionals** (see below) to include different content for short vs long documents (e.g. full file when small, current section when large). |
| **editor_cursor_offset** | Character position of the cursor in the file (-1 if not applicable). |
| **editor_selection** | The text the user has selected, if any. |
| **editor_current_section** | The content of the section (marked by headings up to the configured level, default `#` and `##`) that contains the cursor. |
| **editor_current_heading** | The heading line of that section (e.g. `## Chapter 3`). |
| **editor_previous_section** | The content of the section immediately before the current one. Empty if the cursor is in the first section. |
| **editor_next_section** | The content of the section immediately after the current one. Empty if the cursor is in the last section. |
| **editor_section_index** | Zero-based index of the current section (e.g. 2 for the third section). |
| **editor_adjacent_sections** | Previous, current, and next sections combined. Use this to give the agent “context around the cursor” without sending the whole file. |
| **editor_total_sections** | Total number of sections (at the same heading level) in the file. |
| **editor_toc** | All heading lines from the open file, one per line (e.g. `## Chapter 1\n## Chapter 2\n...`). Useful as a lightweight structural map the agent can scan without loading the full document. Empty if no file is open or the file has no headings. |
| **editor_is_first_section** | `"true"` when the cursor is in the first section; empty otherwise. Use `{{#editor_is_first_section}}` to show content only at the start of the document. |
| **editor_is_last_section** | `"true"` when the cursor is in the last section; empty otherwise. |
| **editor_ref_count** | Number of loaded ref_* files (as a number). Use `{{#editor_ref_count > 0}}` when you want to include a block only when refs exist. |

Section boundaries are determined by Markdown headings. By default the system uses **heading level 2** (`#` and `##`), so each “section” is a block under a top-level or second-level heading. You can override this per playbook step with the **heading_level** step setting (1–6): set it on any step and the same level is used for the whole run for both the open document and all referenced files (so `editor_current_section`, `editor_refs_*_current`, etc. stay in sync).

### Chat artifact (artifact drawer)

When the user has a **chat artifact** open in the sidebar artifact drawer (HTML, Mermaid, chart, SVG, or React preview), the full source of that artifact is available so the agent can refine it iteratively. If the drawer is closed or no artifact is active, these variables are empty.

| Variable | What it contains |
|---------|------------------|
| **previous_artifact** | Full source code of the artifact currently open in the drawer. Empty if none. |
| **previous_artifact_type** | One of: `html`, `mermaid`, `chart`, `svg`, `react`. Empty if none. |
| **previous_artifact_title** | Title shown for the artifact in chat. Empty if none. |
| **previous_artifact_language** | Optional hint for the code view (e.g. `html`, `javascript`, `jsx`, `mermaid`, `svg`). Empty if none. |

The user can open an artifact from a **past message** in the thread and send a follow-up; that artifact becomes `previous_artifact` for that request.

### Referenced files (ref_*)

Many documents reference other files (rules, style guides, character lists, outlines). You can expose those by adding keys in the document’s **frontmatter** with the prefix **ref_**.

- **editor_refs** — All referenced file contents combined into one block. Empty if there are no refs or no file open.
- **editor_refs_*** — One variable per ref category. The part after `editor_refs_` comes from the frontmatter key after `ref_`. Loaded files are returned in full (no character cap); use scoped variables below to keep prompts small.

#### Scoped reference variables (per category)

For every loaded ref category (e.g. `outline`, `style`, `rules`), the system also fills **heading-based** slices so you can avoid sending an entire long file in the prompt. Section boundaries use the same **heading level** as the open document (default 2: `#` and `##`). The “current” slice is the section whose heading matches the open document’s current section heading (`editor_current_heading`).

| Variable | What it contains |
|----------|------------------|
| **editor_refs_CATEGORY_toc** | Replace `CATEGORY` with your ref key (e.g. `outline`). All heading lines in that file, one per line. |
| **editor_refs_CATEGORY_current** | The section whose heading **exactly matches** the open document’s current section heading. Empty if no match. |
| **editor_refs_CATEGORY_adjacent** | Previous + current + next sections around that match. Empty if no match. |
| **editor_refs_CATEGORY_previous** | The section immediately before the cursor-matched section in that ref file. Empty if no match or at start. |
| **editor_refs_CATEGORY_next** | The section immediately after the cursor-matched section in that ref file. Empty if no match or at end. |

**Example:** Frontmatter has `ref_outline: ./outline.md`. You get:

- `{editor_refs_outline}` — full outline content
- `{editor_refs_outline_toc}` — list of headings only
- `{editor_refs_outline_current}` — outline slice for the chapter matching the cursor
- `{editor_refs_outline_adjacent}` — that chapter plus neighbouring outline sections

Use conditionals when a match might not exist: `{{#editor_refs_outline_adjacent}}...{{/editor_refs_outline_adjacent}}`.

#### Named section extraction: `_section:Name`

To always include a specific part of a reference (e.g. a `## Summary` block) regardless of cursor, use this **dynamic** placeholder:

```text
{editor_refs_outline_section:Summary}
```

Replace `outline` with your ref category and `Summary` with a substring of the heading text (match is case-insensitive; first matching section wins). Examples:

- `{editor_refs_outline_section:Theme}` — section whose heading contains “Theme”
- `{editor_refs_style_section:Voice}` — style file section containing “Voice”

Closing conditional blocks must use the **exact** same name:  
`{{#editor_refs_outline_section:Summary}}...{{/editor_refs_outline_section:Summary}}`.

**Example:** In your document’s frontmatter you have:

```yaml
ref_rules: ./rules.md
ref_style: ./style.md
ref_characters: ./characters.md
```

Then at runtime you get:

- **editor_refs_rules** — Contents of `rules.md`
- **editor_refs_style** — Contents of `style.md`
- **editor_refs_characters** — Contents of `characters.md`

You can use any name after `ref_`: `ref_rules`, `ref_style`, `ref_whatever`. Each becomes `editor_refs_whatever`. The system loads the file at the path you give (relative to the current document) and injects its content into that variable.

### Wildcard refs: one variable for many ref_* keys

When you have many ref_* keys that vary by document (e.g. `ref_character_adam`, `ref_character_betty`, and so on), you can use a **wildcard variable** instead of listing each one. Add a trailing `*` to the variable name: the part before `*` is the prefix, and the system replaces it with the **concatenated content** of all loaded refs whose category starts with that prefix.

- **Example:** `{editor_refs_character_*}` — includes content from `editor_refs_character_adam`, `editor_refs_character_betty`, and any other ref whose category starts with `character_`.
- **Without underscore before *:** `{editor_refs_character*}` also matches a single `ref_character` key if present, plus all `ref_character_*` keys.

The same pattern works in **conditional blocks**: `{{#editor_refs_character_*}}...{{/editor_refs_character_*}}` includes the block when *any* matching ref has content. Frontmatter keys must still start with `ref_`; the wildcard only aggregates the per-category variables the system already creates.

### User profile

| Variable | What it contains |
|---------|------------------|
| **profile** | The current user’s profile (name, email, timezone, etc.), when profile loading is enabled. |

### Other

| Variable | What it contains |
|----------|------------------|
| **user_weather_location** | The user’s configured weather location, when set. |


### Other runtime variables

These are filled automatically when the context applies; use them in prompts or conditionals when needed.

| Variable | What it contains |
|----------|------------------|
| **last_tool_results** | JSON of the most recent tool results from the conversation (from the agent's last tool call). Empty if none. |
| **document_context** | Full content of the **pinned** document (when the user has pinned one for the agent). Includes a header with title and ID. Empty if no pin. |
| **pinned_document_id** | Document ID of the pinned document. Empty if no pin. Use when a tool needs to edit or reference the pinned doc. |

Inside a **loop** step, you also get **{_iteration}** — the current 1-based iteration number (1, 2, 3, …). Use it in prompts or conditionals, e.g. `{{#_iteration > 1}}Refining based on previous pass...{{/_iteration > 1}}`.

---

## How to use variables in prompts

Use **one pair of curly braces** around the variable name:

- Right: `The user asked: {query}`
- Right: `Today is {today}.`
- Wrong: `{{query}}` — double braces are reserved for conditional blocks (see below).

You can mix variables with normal text and with outputs from earlier steps. To use the result of a previous step, use its **output_key** and a field name, e.g. `{fetch_cal.formatted}` or `{search_docs.documents}`. The step’s display name doesn’t matter — only the output_key you set in the Composer.

---

## Hiding sections when there’s no data: conditional blocks

Sometimes you only want a part of the prompt to appear when a variable has a value. For example: “If the user has a document open, show it; otherwise don’t mention the document at all.” That keeps the prompt clean and avoids confusing the model with empty or “N/A” blocks.

**Conditional blocks** do exactly that. Syntax:

```text
{{#variable_name}}
  … this text is only included when variable_name is non-empty …
{{/variable_name}}
```

The opening tag is `{{#variable_name}}` and the closing tag is `{{/variable_name}}`. The name must match exactly. If `variable_name` is empty (or missing), the whole block is removed before the prompt is sent to the model.

**Expression conditionals** let you show or hide a block based on a comparison (e.g. document size):

```text
{{#editor_length > 5000}}
  … included only when the open file has more than 5000 characters …
{{/editor_length > 5000}}

{{#editor_length < 5000}}
  … included only when the open file has fewer than 5000 characters …
{{/editor_length < 5000}}
```

Use this to choose **full editor** vs **current section** by size: for short documents include the full file; for long ones include only `editor_adjacent_sections` or `editor_current_section` to keep the prompt small. The closing tag must match the opening expression exactly (e.g. `{{/editor_length > 5000}}`). Supported operators: `>`, `<`, `>=`, `<=`, `==`, `!=`. You can use **editor_length** (character count) or any other variable that resolves to a number or string.

#### Advanced conditionals

In **branch conditions** and in **expression conditionals** you can use:

- **is defined** / **is not defined** — Include a branch or block only when a variable (or step output) has a value: `{editor_selection} is defined`, `{search_docs.formatted} is not defined`.
- **matches** — Case-insensitive regex: `{editor_document_type} matches "^(fiction|novella)$"`.
- **AND** / **OR** — Combine conditions: `{editor_document_type} == "fiction" AND {editor_selection} is defined`, or `{editor_document_type} == "outline" OR {editor_document_type} == "notes"`.
- **Step output expressions** — In prompt conditionals, any step output key and field from the playbook state can be used in an expression. For example, if a previous step has `output_key: search_docs` and returns a `count` field, use `{{#search_docs.count > 0}}...{{/search_docs.count > 0}}` to include a block only when the search found results.

**Example — only show the document block when a file is open:**

```text
{{#editor}}
The user has this document open:
{editor}
Document ID: {editor_document_id}
{{/editor}}

User request: {query}
```

If no file is open, the block from `{{#editor}}` to `{{/editor}}` is stripped out, and the model only sees “User request: …”.

**Example — only show rules when the document references a rules file:**

```text
{{#editor_refs_rules}}
=== Universe rules ===
{editor_refs_rules}
{{/editor_refs_rules}}

{{#editor_adjacent_sections}}
=== Context around cursor ===
{editor_adjacent_sections}
{{/editor_adjacent_sections}}

User: {query}
```

So: use **single braces** `{var}` for substitution, and **double braces** `{{#var}}...{{/var}}` to include a section only when that variable is set. That way your prompts stay focused and easy for the model to follow.

---

## Routing by document type: branch conditions

When you add a **branch** step, you choose which path to take with a **branch_condition**. You can use variables there too.

To route by document type (e.g. “fiction” vs “outline”):

- Condition: `{editor_document_type} == "fiction"`
- Condition: `{editor_document_type} == "outline"`

The value is whatever you put in the document’s frontmatter under `type:` (and it’s lowercased). So a branch can send fiction manuscripts to one set of steps and outlines to another, with no extra code — just playbook configuration.

---

## Quick reference: editor-focused prompts

A common pattern is an “editor” playbook that behaves differently depending on whether a file is open and what type it is. You can combine variables and conditionals like this:

1. **Branch** on `{editor_document_type} == "fiction"` (or `"outline"`, etc.).
2. In the **llm_agent** (or LLM task) prompt:
   - Use `{{#editor_adjacent_sections}}...{{/editor_adjacent_sections}}` to send only the sections around the cursor.
   - Use `{{#editor_refs_rules}}`, `{{#editor_refs_style}}`, etc., to add referenced files only when they exist.
   - Use `{{#editor_selection}}` to mention the current selection only when the user selected something.
   - Use `{editor_document_id}` so the agent (and tools like patch_file) know which document to edit.

That way you get document-aware, context-scoped prompts without pasting the entire file every time, and the system can route and prepare context from the document’s frontmatter and cursor position.

---

## Summary

- **Runtime variables** — Use `{variable_name}` in prompts and step inputs. Names must be from the list above (or `editor_refs_*` for ref_* frontmatter). To give the agent conversation context, enable **Include history in prompt** on the profile and put **`{history}`** in your step prompt (see "How to inject conversation history" above).
- **Conditional blocks** — Use `{{#var}}...{{/var}}` so a block is only included when `var` is non-empty. Use **expression conditionals** like `{{#editor_length > 5000}}...{{/editor_length > 5000}}` or `{{#search_docs.count > 0}}...{{/search_docs.count > 0}}` to include a block only when a comparison is true. Use **advanced conditionals** in branches and prompts: `is defined` / `is not defined`, `matches` (regex), and `AND` / `OR`. Keeps prompts clean when the editor is closed or a ref is missing.
- **Universal ref_*** — Any frontmatter key starting with `ref_` (e.g. `ref_rules`, `ref_style`) becomes a loaded file and is available as `editor_refs_rules`, `editor_refs_style`, etc. Full content is available; use `_toc`, `_current`, `_adjacent`, `_previous`, `_next`, and `_section:Name` to scope long references.
- **Wildcard refs** — Use a trailing `*` (e.g. `{editor_refs_character_*}` or `{{#editor_refs_character_*}}...{{/editor_refs_character_*}}`) to include all refs whose category starts with the prefix.
- **Branch conditions** — Use variables like `{editor_document_type}` in branch conditions to route by document type or other context. Use `{_iteration}` inside loop steps for the current pass number.
- **Other runtime variables** — `last_tool_results`, `document_context`, `pinned_document_id` are available when applicable; see the “Other runtime variables” section.


Stick to these variable names and the single-brace / double-brace rules, and your playbooks will stay valid and easy to maintain.
