---
title: Folders
order: 2
---

# Folders

Folders organize your Document Library. You create them under **My Documents**, **Teams**, or **Global Documents**, nest them as deep as you need, and use right‑click options to upload files, create new files, change metadata, or control whether contents are indexed for search. This page goes into folder behavior and options in more detail.

---

## Creating and nesting folders

- **My Documents** — Right‑click “My Documents” and choose **New User Folder**. Right‑click any existing user folder and choose **New Folder** to create a subfolder. All of these are “user” (personal) scope.
- **Global Documents** — Admins only: right‑click “Global Documents” and choose **New Global Folder**. Right‑click any global folder to create **New Folder** (global subfolder). You can’t create a user folder inside Global or vice versa.
- **Teams** — Each team has its own root. Right‑click a team folder and choose **New Folder** to create subfolders. Only team members (and admins) can create or delete; team root folders can’t be deleted (delete the team instead).

New folders inherit the scope of their parent (user, global, or team). Drag‑and‑drop to **move** folders between folders of the same scope when the UI supports it.

---

## Uploading into folders

You can’t upload files directly onto the **My Documents** or **Global Documents** root. Right‑click a **folder** (or create one first) and choose **Upload Files**. Select one or more files; optionally set category and tags. Uploaded files appear in that folder. To put files in another section (e.g. Global), navigate to a folder under that section and upload there.

---

## Folder metadata

For non-root, non-virtual folders, right‑click and choose **Edit Folder Metadata**. You can set or change:

- **Name** — Display name in the tree.
- **Description** — Optional.
- **Tags** — Optional tags for organization.

Changes apply immediately. Virtual roots (e.g. RSS, News) and the top-level “My Documents” / “Global Documents” entries don’t have this menu.

---

## Search (indexing)

By default, documents in folders are **included in search** (indexed for semantic, keyword, and entity-graph search) unless an exemption is set. You can control this per folder or per file.

**Folder right‑click → Search**

- **Exempt from search** — This folder and all its contents (including subfolders) are excluded from search indexing. Use for private or temporary content you don’t want in search.
- **Include in search** — This folder’s contents are indexed even if a parent folder is set to “Exempt from search.”
- **Use parent setting** — This folder inherits the behavior of its parent (default for new folders).

**Document right‑click → Search**

- **Exempt from search** — This file is excluded from search.
- **Include in search** — This file is included in search (e.g. after exempting the folder).

So you can turn off indexing for a whole folder, or override for specific subfolders or files. Indexing affects all search (semantic, keyword, entity graph) and tools that rely on it; it doesn’t delete the file or folder.

---

## Moving and renaming

- **Rename** — Right‑click a folder (or file) and choose **Rename**. Enter the new name. Allowed for any non-root folder and for files you have edit access to. Team root folders can’t be renamed (they follow the team name).
- **Move** — Right‑click and choose **Move**, then pick the destination folder. You can only move within the same scope (e.g. user folder to user folder, team folder to team folder). Moving a folder moves its contents; moving a file only changes its folder.

---

## File and folder sharing

For user-owned content, use **Sharing…** from the right-click menu:

- **Share folder** — Grants access to that folder and everything inside it.
- **Share document** — Grants access to a single file.
- **Permission levels** — Choose **Read** or **Write** per person.
- **Manage access** — Change permission or revoke access later from the same dialog.

People you share with see the item under **Shared with me** in their sidebar. Shared folders can be expanded so they can open files inside without moving or duplicating content.

Notes:

- Sharing applies to user-owned items. Team and global areas are already shared by their scope and permissions.
- If you have **Write** access to a shared item, you can edit according to document-type and lock/collaboration rules.

---

## Deleting folders and files

- **Delete (folder)** — Right‑click a folder and choose **Delete**. You can’t delete the “My Documents” or “Global Documents” roots, or a team’s root folder (delete the team instead). Deleting a folder can remove its contents depending on instance policy — confirm when prompted.
- **Delete (file)** — Right‑click a document and choose **Delete**. The file is removed from the library. Ensure you have permission (e.g. regular users often can’t delete from Global).

---

## Permissions (summary)

- **My Documents** — Only you create, edit, move, and delete (unless you’ve shared in some way).
- **Teams** — Team members (and admins) can create and edit according to team settings; only the folder creator or team admins can delete team folders; team root can’t be deleted.
- **Global Documents** — Typically only admins create folders and upload or edit files; others may have read-only access. Metadata and delete for global files are usually admin-only.
- **Shared with me** — Access is controlled by the owner: **Read** allows viewing/opening; **Write** allows editing where supported.

For exact behavior on your instance, check with your administrator or try the options in the UI — disabled or missing items indicate permission limits.

---

## Related

- **Document Library overview** — The three sections (My, Teams, Global), creating files, upload, and right‑click summary.
- **File types: Markdown and Org Mode** — Difference between Markdown and Org; when to use which.
- **Org Mode files** — TODOs, All TODOs view, and tools that use org data.
