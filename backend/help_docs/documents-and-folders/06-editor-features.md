---
title: Editor features
order: 6
---

# Editor features

When you open a document in the Document Library, it opens in the **document editor** (viewer). The editor provides a toolbar, optional AI edit proposals, version history with diff and rollback, text-to-speech, export, and support for several file types (Markdown, Org Mode, PDF, DOCX, and others). This page summarizes these features and where to find them.

---

## Toolbar

At the top of the document view you get a **toolbar** with actions that depend on the file type and your permissions:

- **Edit** — Switch between view and edit mode for text documents (Markdown, Org). In edit mode you can change content; changes are saved (manually or auto-save depending on settings).
- **Version history** — Open the version history dialog to see past versions, view content, diff against current, and roll back. Available when you have edit access.
- **Read aloud** — Start or stop text-to-speech for the current document. The button toggles between “Read aloud” and “Stop reading.”
- **Download or Export** — Open a menu to export or download: e.g. **Export to PDF**, **Export as EPUB**, or download the raw file. Options depend on the document type.

Other toolbar items may include metadata, folder info, and type-specific actions (e.g. org refile, archive, tag dialogs for Org files).

---

## AI edit proposals

Agents can suggest **edits** to the open document (e.g. from the writing or fiction-editing agent). When an edit is proposed:

- A **proposal** appears in or beside the document (e.g. as a diff or inline suggestion). You see what would change.
- **Apply** — Accept the proposal. The content is updated and the proposal is removed. For proposals stored on the server, applying is persisted so they do not reappear.
- **Reject** — Dismiss the proposal. The document is unchanged and the proposal is removed (and persisted as rejected when supported).

Proposals can come from the current conversation (streaming) or from pending proposals loaded when you open the document. You can have multiple proposals; accept or reject each one. Applied edits create a new **version** in version history.

---

## Version history

For documents you can edit, **Version history** (toolbar icon) opens a dialog that lists saved **versions** of the document. Each version has a version number and change source (e.g. “user edit”, “agent edit”).

- **View** — Select a version to view its full content. Use **Back to list** to return to the version list.
- **Diff with current** — Compare an older version with the current one. You see added and removed lines (+/-) and a short summary (e.g. “+X / -Y”).
- **Rollback** — Restore the document to a selected version. You are asked to confirm; the current content is saved as a new version first, then the document is reverted to that version. You can roll back again if you need to “undo” a rollback.

Versions are created when you or an agent saves changes. If no versions exist yet, the list is empty and the message indicates that edits will create versions.

---

## Text-to-speech (Read aloud)

Use **Read aloud** in the toolbar to have the document read aloud. Click again to stop. Reading uses the browser or system TTS; language and voice depend on your device and settings. Useful for long Markdown or Org documents.

---

## Export and download

From the **Download or Export** menu:

- **Export to PDF** — Renders the current content (typically Markdown) as PDF and downloads it.
- **Export as EPUB** — Opens a dialog to set title and options, then generates an EPUB and downloads it. Useful for e-readers.
- **Download** — Download the raw file (e.g. `.md`, `.org`) as stored on the server.

Export options may vary by file type (e.g. PDF/EPUB for Markdown; raw download for all).

---

## File types in the editor

- **Markdown** (`.md`) — Edited with the Markdown editor. Full toolbar, proposals, version history, read aloud, and export.
- **Org Mode** (`.org`) — Edited with the Org editor. Same toolbar plus org-specific actions (refile, archive, tag dialogs). TODOs and structure are parsed for All TODOs and tools.
- **PDF** — Opened in a PDF viewer. You can view and, for some PDFs, use the **PDF text layer editor** (separate route) to edit the text layer. No Markdown-style version history.
- **DOCX** — Opened in a DOCX viewer. View and sometimes edit depending on instance support.
- **EML** — Email files open in an EML viewer for reading.
- **Audio** — Audio files open in an audio player for playback.
- **Images** — Images open in a viewer with optional **lightbox** (click to enlarge) and **Edit Image Metadata** from the file’s right-click menu in the tree.

---

## Summary

- The **toolbar** gives you edit mode, version history, read aloud, and export/download.
- **AI edit proposals** let you **Apply** or **Reject** agent-suggested edits; applied edits create versions.
- **Version history** lists versions; you can **View**, **Diff with current**, and **Rollback**.
- **Read aloud** and **Export** (PDF, EPUB, download) are in the toolbar. Different file types (PDF, DOCX, EML, audio, images) open in the appropriate viewer.

---

## Related

- **Document Library overview** — How to open documents and use the folder tree.
- **Markdown files** — Creating and editing Markdown.
- **Org Mode files** — TODOs, agendas, and org tools.
- **Folders** — Permissions and search indexing.
