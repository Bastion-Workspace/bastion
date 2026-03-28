---
title: Data Workspace overview
order: 1
---

# Data Workspace overview

The **Data Workspace** is where you store and query **structured data** (tables) inside Bastion. You access it from the Document Library sidebar, create **workspaces** and **databases**, add **tables** (with column schemas), and **import** or edit data. Agents and playbooks can run **SQL** against these tables via tools. This page describes how to open the Data Workspace, create tables, import data, and how agents use it.

---

## Where the Data Workspace lives

In the **Document Library** left sidebar you see a **Data Workspace** section (or similar). Under it are **workspaces**; each workspace can contain **databases** and **tables**. Click a workspace to open it in a tab. The tab shows the **Data Workspace Manager**: a list of databases, and when you open a database, its tables. You can switch between **databases**, **tables**, and **data** views from the manager’s tabs or menu.

---

## Creating tables

- **Table creation wizard** — From the Data Workspace Manager, use **Create table** or **New table** (or open the table-creation wizard from the database or workspace menu). In the wizard you:
  - Name the table.
  - Define **columns**: name, type (e.g. text, number, date, boolean), and optional constraints.
  - Create the table. It appears in the table list and you can start adding or importing rows.

- **Column schema** — Each column has a **type** so the system can validate and display data correctly. You can edit the schema later (add or remove columns, change types) if the UI supports it; schema changes may require a migration or re-import for existing data.

---

## Importing data

- **Import wizard** — Use **Import** or **Import data** from the workspace or database view. In the wizard you typically:
  - Choose the **target table** (or create one as part of import).
  - Select a **file** (e.g. CSV) or paste data. Map columns to table columns.
  - Run the import. Rows are inserted into the table. Duplicates or errors may be reported depending on constraints.

After import, you can **edit rows** in the table view (inline or via a form), **add** rows, and **delete** rows. The **formula bar** (if present) lets you write expressions or formulas for computed columns or validation.

---

## Viewing and editing data

- **Table view** — Select a table to see its **rows** in a grid. You can sort, filter, and scroll. Click a cell to edit (if you have edit access). Column headers may offer sorting or filtering.
- **Editing rows** — Add a new row, edit cells in place, or delete rows from the row menu or selection. Changes are saved to the workspace database.
- **Sharing** — Workspaces may support **sharing** so other users or teams can view or edit the same data. Sharing controls depend on the instance; look for **Share** or **Permissions** on the workspace or database.

---

## How agents and playbooks use it

- **SQL tools** — Agent Factory playbooks and built-in agents can use **run SQL** or **query data workspace** tools. You (or the playbook) specify the workspace and optionally the table; the tool runs a read-only or read-write SQL query and returns results. So agents can summarize data, look up values, or generate reports from your tables.
- **Data in context** — Results from a SQL step can be wired into later steps (e.g. into an LLM step for summarization or into a notification). The Data Workspace is the backing store; tools are the interface.

---

## Summary

- **Data Workspace** is in the Document Library sidebar. Open a **workspace** to see **databases** and **tables**.
- Create tables with the **table creation wizard** (name and column schema). **Import** data from CSV or paste; **edit** rows in the table view. **Formula bar** and **sharing** depend on the instance.
- **Agents and playbooks** use **SQL tools** to query (and sometimes update) Data Workspace tables; results can be passed to later steps.

---

## Related

- **Document Library overview** — Sidebar layout and tabs.
- **Agent Factory overview** — Custom agents and tools.
- **Tools reference** — Tool categories including data workspace.
