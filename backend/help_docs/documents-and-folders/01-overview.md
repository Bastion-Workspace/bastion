---
title: Document Library overview
order: 1
---

# Document Library overview

The **Document Library** is where your files and folders live. You open it from the main navigation (Documents). The left sidebar shows a **folder tree** with three kinds of areas: **My Documents**, **Teams**, and **Global Documents**. You create and upload files there, organize them in folders, and use right‑click menus to manage everything. This page explains the layout and the main actions; later topics cover folder details and the difference between Markdown and Org Mode files.

---

## The three sections

### My Documents

**My Documents** is your personal space. Only you see and manage these folders and files (unless you share them). Use it for notes, drafts, and anything that doesn’t need to be team-wide or global.

- You can create **folders** and **subfolders** under My Documents.
- You can **upload** files and create new **Markdown** or **Org Mode** files here.
- You can **download** your whole library as a zip from the Documents menu (e.g. “Download my Library”).

### Teams

If you’re in one or more **teams**, each team has its own **Documents** area in the sidebar. Team folders are shared with that team: members can see and edit files according to the team’s permissions. Team roots appear as top-level entries (e.g. “Marketing”, “Engineering”); you can’t delete a team root folder — you’d remove the team instead. Inside a team folder you get the same right‑click options as in My Documents (new folder, upload, new file, rename, move, etc.).

### Global Documents

**Global Documents** is shared across the whole instance. Typically only **admins** can create folders and upload or edit files here. Regular users can usually view and open global files. Use it for templates, org-wide reference material, or shared RSS feeds. You can’t upload directly “into” the root — pick or create a folder under Global Documents first.

---

## Creating files and folders

**New folder**  
Right‑click a folder (or, for roots, right‑click “My Documents” or “Global Documents”) and choose **New Folder** (or **New User Folder** / **New Global Folder** on the roots). Name the folder and save. Subfolders inherit the same “scope” (user, team, or global).

**New Markdown**  
Right‑click a folder and choose **New Markdown**. You’ll get an empty Markdown file in that folder. Name it when prompted (e.g. `notes.md`). Good for plain notes, docs, and anything that doesn’t need Org’s structure.

**New Org-Mode**  
Right‑click a folder and choose **New Org-Mode**. You’ll get an empty Org file (e.g. `file.org`). Use Org for outlines, TODOs, agendas, contacts, and structured lists. See **Org Mode files** for how TODOs and other features work.

You can’t create a file directly on the **root** of My Documents or Global Documents — create or open a folder first, then use the folder’s right‑click menu.

---

## Uploading files

Right‑click the **folder** where you want the files and choose **Upload Files**. In the dialog you can:

- Select one or more files from your computer.
- Optionally set **category** and **tags** for the upload (if your instance supports them).

Uploaded files appear in that folder. Supported types (e.g. `.md`, `.org`, `.txt`, images, PDFs) depend on your instance; images and documents can be indexed for search (see **Folders** for search indexing options). You cannot upload directly into the “My Documents” or “Global Documents” root — choose a folder under them.

---

## Right‑click options (summary)

**On a folder**

- **New Folder** — Create a subfolder (or “New User Folder” / “New Global Folder” on roots).
- **Upload Files** — Upload files into this folder.
- **Edit Folder Metadata** — Name, description, tags (for non-root folders).
- **Describe folder images with LLM** — If enabled, generate descriptions for images in the folder.
- **New Markdown** / **New Org-Mode** — Create a new file in this folder.
- **Add RSS Feed** — (Only in RSS sections) Add a feed to the list.
- **Rename** / **Move** / **Delete** — For non-root, non-team-root folders.
- **Sharing…** — (User-owned folders) Open sharing to grant read or write access to specific users.
- **Search** — Submenu: **Exempt from search**, **Include in search**, or **Use parent setting** (whether files in this folder are indexed for search).
- On **Global Documents** root (admins only): **New Global Folder**. On **My Documents** root: **New User Folder**.

**On a file (document)**

- **Edit Image Metadata** — (Images only) Set or edit metadata for the image.
- **Re-process Document** — Re-run processing (e.g. indexing) for this file.
- **Edit Metadata** — (Non-images) Edit document metadata (admins for global files).
- **Rename** / **Move** / **Delete** — Change name, move to another folder, or remove the file.
- **Sharing…** — (User-owned files) Grant or revoke read/write access for individual users.
- **Search** — Submenu: **Exempt from search** or **Include in search** (include or exclude this file from search indexing).

Details and permissions (e.g. who can delete team folders or edit global metadata) are in **Folders**.

---

## Shared with me

When another user shares a file or folder with you, it appears in **Shared with me** in the Document Library sidebar:

- Items are grouped by the person who shared them.
- You see whether each item is **Read only** or **Can edit**.
- Shared folders can be expanded to browse and open files inside.
- Your available actions depend on the access level the owner granted.

---

## File types: Markdown vs Org Mode

The library supports **Markdown** (`.md`) and **Org Mode** (`.org`) as first-class text formats.

- **Markdown** — Simple formatting, headers, lists, and links. Good for notes, docs, and wikis. See **Markdown files**.
- **Org Mode** — Headings, TODOs, schedules, contacts, and tags. TODOs and other structures are cataloged and available in the **All TODOs** view and in tools (e.g. list todos, toggle state). See **Org Mode files** and **File types: Markdown and Org Mode**.

---

## Password-protecting files (encryption)

For sensitive **Markdown, plain text, or Org** files, you can **encrypt** them from the folder tree (right‑click → **Encrypt file…**). Encrypted files stay out of search until you unlock them in the editor, and each file has its own password. See the Help topic **Document encryption** for sessions, multiple tabs, lock behavior, and limitations (collaboration, PDF download, etc.).

---

## Where to go next

- **Folders** — Folder creation, nesting, search indexing, and permissions in more detail.
- **File types: Markdown and Org Mode** — How the two formats differ and when to use each.
- **Markdown files** — Creating and editing Markdown; behavior in search and agents.
- **Org Mode files** — TODOs, agendas, contacts, tags, All TODOs view, and how tools use org data.
