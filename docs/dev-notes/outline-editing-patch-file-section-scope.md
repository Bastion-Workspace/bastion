# Outline editing: `patch_file` `section` field (playbook prompt updates)

The orchestrator `patch_file` tool now accepts an optional **`section`** on each edit (stored as `section_scope` in semantic operations). The backend resolver constrains replace/delete matching and insert/append placement to that heading’s section (through the next heading of the same or higher outline level).

Use this note when updating **Agent Factory** playbook steps for outline editing (or any long markdown document with repeated `###` sub-headings).

## Recommended instruction changes

### Adding beats under an existing `### Beats`

Prefer **`insert_after_heading`** with **`section`** set to the chapter heading (verbatim line, e.g. `## Chapter 3`), **`target`** set to `### Beats`, and **`content`** set to the new `- ` beat lines. The resolver inserts at the end of that chapter’s Beats subsection, not the last `### Beats` in the file.

### Replace / delete within a specific chapter or profile section

Use **`replace`** or **`delete`** with **`section`** set to the parent heading (e.g. `## Chapter 3`, `## Personality`) so **`target`** is matched only inside that section. Still use a generous verbatim **`target`**.

### Append at end of a chapter vs end of file

- **End of a specific chapter:** **`append`** with **`section`** = that chapter’s heading line (e.g. `## Chapter 3`). Content is inserted at the end of that chapter, before the next same-level heading.
- **New chapter at EOF:** **`append`** with **no** `section` (unchanged behavior).

### Non-outline documents

`section` is heading-agnostic. Examples:

- Character: `section: "## Personality"`
- Rules: `section: "## Vampire Physiology and Abilities"`
- Style: `section: "## Core Narrative Pillars"`
- Nonfiction outline: `section: "## Part 2: Analysis"`

## Large-document variables

`editor_adjacent_sections`, `editor_previous_section`, and related prompt variables from `llm-orchestrator/orchestrator/agents/custom_agent_runner.py` remain valid for **read** context. For **writes**, instruct the model to pass **`section`** on `patch_file` edits whenever sub-headings repeat, so resolution does not depend on full-file uniqueness of `target` alone.

## Resolver reference

Implementation: `backend/utils/editor_operations_resolver.py` (`_find_section_window`, `_resolve_replace_delete`, `_resolve_insert`).

Tool schema and mapping: `llm-orchestrator/orchestrator/tools/file_editing_tools.py` (`PatchEdit.section`, `_edit_to_op_dict`).
