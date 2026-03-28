---
title: Projects
order: 8
---

# Projects

A **project** is a folder with a predefined structure and a **project plan** file (`project_plan.md`) that agents and you can use for planning and references. You create a project from the Document Library sidebar by right-clicking a folder and choosing **Create Project**. You pick a **project type** (e.g. Electronics or General), name the project, and the system creates the folder and the plan file with the right frontmatter and content. This page describes how to create projects and how they fit into the Document Library and Agent Factory.

---

## What gets created

When you create a project:

- **Folder** — A new folder is created with the name you gave the project. It lives under the **parent folder** you selected (e.g. under My Documents or a subfolder).
- **project_plan.md** — Inside the new folder, the system creates a **project_plan.md** file. Its content depends on the **project type**:
  - **Electronics** — Template suited to electronics projects (e.g. components, schematics, build steps). Frontmatter may reference component lists, specs, or other project-specific keys that agents understand.
  - **General** — A general-purpose project plan template (e.g. goals, tasks, notes). Frontmatter can list referenced files or sections that agents use when editing.

The plan file’s frontmatter (e.g. `ref_*` fields or file lists) is used by agents and tools to know which files belong to the project and how to route content. You can edit the plan like any other Markdown file and add more files to the folder.

---

## How to create a project

- **Right-click** a folder in the Document Library sidebar (e.g. under My Documents or a team folder). From the context menu, choose **Create Project** (or equivalent). A dialog opens.
- **Project name** — Enter the name of the project. This becomes the **folder name** and typically the **title** inside the plan file.
- **Parent folder** — The folder you right-clicked is usually preselected as the parent. You can change it if the dialog lets you pick another folder. The new project folder will be created inside that parent.
- **Project type** — Choose a type from the dropdown (e.g. **Electronics**, **General**). This determines the template used for `project_plan.md`.
- **Create** — Click **Create** (or **Save**). The folder and `project_plan.md` are created. You can open the folder and the plan file from the sidebar and start adding or editing content.

If the dialog does not appear, your role or permissions may not allow project creation in that folder; try under My Documents or a folder you own.

---

## Using the project plan

- **Editing** — Open `project_plan.md` like any other document. Edit the Markdown and frontmatter to add goals, task lists, file references, or type-specific sections. Changes are saved normally; version history applies if enabled.
- **Referenced files** — The frontmatter often lists other files (e.g. `./component_list.md`, `./notes.md`). Agents and tools that support **reference file loading** use this list to know which files to open or pass to the LLM. Keep the list up to date so agents see the right context.
- **Agent Factory** — Custom agents (e.g. electronics or general project agents) can be designed to read the project plan and operate on the project folder. They may create or update files under the same folder and update the plan’s frontmatter when adding new references.

---

## Summary

- A **project** is a **folder** plus a **project_plan.md** file with a template chosen by **project type** (Electronics or General). Create one via **Create Project** from the folder context menu.
- In the dialog, set **project name**, **parent folder**, and **project type**, then create. Edit the plan and add files as needed; frontmatter and referenced files are used by agents and tools.

---

## Related

- **Document Library overview** — Folders and right-click options.
- **Markdown files** — Frontmatter and editing Markdown.
- **Agent Factory overview** — Custom agents that can use project plans.
- **Folders** — Permissions and nesting.
