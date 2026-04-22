---
title: Data Workspaces overview
order: 1
---

# Data Workspaces overview

**Data Workspaces** hold structured data—**databases**, **tables**, and **rows**—separate from your document files. You create a workspace, add at least one database, define tables (columns and types), then import or enter data. You can run SQL, link rows between tables, and share a workspace with others. Agents and playbooks access the same data through data-workspace tools.

---

## Open the Data Workspaces list

In the **Document Library**, the left sidebar includes a **Data Workspaces** section. There you can:

- **All** — Every workspace you can use (owned and shared).
- **My Workspaces** — Workspaces you created.
- **Shared with Me** — Workspaces others shared with you.

Add a workspace with the **+** next to the section title. Open a workspace from the list; it appears as a **tab** in the main area.

---

## Structure: workspace → database → table

1. **Workspace** — A container (name, optional icon and description). It can hold multiple databases.
2. **Database** — Created inside a workspace. Start here before tables; the empty state prompts you to create one.
3. **Table** — Lives in a database. It has a **schema** (column names, types, and optional **reference** columns that point to rows in another table).
4. **Rows** — Displayed in a **grid** when you select a table from the **Tables** tab (after you pick a database).

**Databases** and **Tables** each have their own top tab when a database is selected, so you move from the database list to its tables, then into a table’s data view.

---

## Create and change tables

- **Create table** — On a database’s **Tables** view, use **Create table** (or **Create your first table** when the database is empty). The wizard sets the table name, columns, and types.
- **Table schema** — In the table card menu (⋮), open **Table schema** to add, remove, or reorder columns.
- **Run SQL** — The **Run SQL** action on the same view runs SQL in the workspace (including `CREATE TABLE`, `INSERT`, and other statements, subject to your permissions). This is the in-app query editor; it complements **SQL** used by agents in playbooks (see below).

---

## Import data

From a **database** card menu, use **Import data** to open the import wizard. You can:

- Target a **new or existing** table and map file columns to table columns.
- Import **CSV**, **JSON**, and **JSON Lines (JSONL / NDJSON)** files (and related formats the wizard offers).

On errors or duplicates, messages depend on table constraints and the import result.

---

## Work with rows in the grid

- **Select a table** on the **Tables** tab, then the **row grid** opens for that table.
- **Edit** — Click cells, use row actions, or use **Quick edit** (edits the **current page** of rows; follow on-screen behavior for focus and save).
- **Formula bar** — For supported columns, select a cell and use the formula bar for values or formulas; recalculation runs when the table supports it.
- **Row links** — A column can be a **reference** to a row in another table. You can **pick a linked row**, follow **open linked table**, and clear a link from the same UI.
- **Breadcrumbs** — **Databases → (database) → (table)** at the top of the **Tables** view; use them to move up without losing context.
- **Fullscreen** — From the workspace header, **fullscreen** expands the workspace for easier grid work; exit the same way.

**Sort and filter** — Use the headers and tools in the table view; exact options are shown in the current UI.

---

## Share a workspace

If you **own** a workspace, a **Share** action on that row in the list opens sharing. You can invite other users, set read or write access, and (where enabled) use **public** or link-based access according to the dialog. Shared workspaces also appear when **Shared with Me** is selected.

---

## How agents and playbooks use the same data

- **Binding (custom agent profile)** — In **Agent Factory**, open your agent → **Data Workspace**. Select **Workspaces**, enable **Auto-inject schema** to include table definitions in the agent context when appropriate, and add optional **Context instructions**. See **Agent profile settings** and **Agent Factory tools**.
- **Agent lines** — In an **agent line**’s settings, the **Data Workspaces** section can attach workspaces with **read-only** or **read/write** SQL access for line runs (alongside each member’s profile). See **Agent Lines** in Help and the line editor’s **Data Workspaces** panel.
- **Tools and playbooks** — Data-workspace tools can list workspaces and run SQL when the playbook calls them. Step outputs can feed later steps (for example summarization or notifications). The **Run SQL** button in the grid and agent runs share the same underlying tables but are separate entry points. Additional server-side features available only to agents depend on your deployment.

---

## Summary

- Find **Data Workspaces** in the **Document Library** sidebar; add workspaces and open them in tabs.
- In each workspace: create a **database**, then **tables**; use **import** (CSV, JSON, JSONL) and the **grid** to manage rows; use **Run SQL** for ad hoc SQL; use **row links** and the **formula bar** where the schema allows.
- **Share** from the workspace list when you own a workspace. Agents use **profile-bound** data-workspace tools; see **Agent Factory** for binding and permissions.

---

## Related

- **Document Library overview** — Sidebar layout and tabs.
- **Agent Factory overview** — Custom agents, tools, and data workspace binding.
- **Agent profile settings** — Per-agent workspace and schema options.
