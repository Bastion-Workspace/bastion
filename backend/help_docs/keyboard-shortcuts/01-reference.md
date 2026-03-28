---
title: Keyboard shortcuts reference
order: 1
---

# Keyboard shortcuts reference

This page lists **keyboard shortcuts** available in Bastion. Use **Ctrl** on Windows/Linux and **Cmd** on macOS where indicated. Shortcuts work when the relevant view (e.g. document editor or Quick Capture dialog) is focused.

---

## Global shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+Shift+C** (Windows/Linux) / **Cmd+Shift+C** (macOS) | **Quick Capture** — Open the Org Quick Capture dialog from any page. Capture a quick note, TODO, contact, or meeting to your inbox org file without opening the Document Library. (Journal entries use **Ctrl+Shift+J** instead.) |
| **Ctrl+Shift+J** (Windows/Linux) / **Cmd+Shift+J** (macOS) | **Journal for the day** — Open the journal-for-the-day modal. View or edit today’s journal entry (or pick another date). If the day has no entry yet, you can type and save to create it under the day heading. Uses your configured journal file and organization (Settings > Org-Mode). |

---

## Document editor (Org files)

When you have an **Org Mode** document open in the document editor (viewer), these shortcuts apply:

| Shortcut | Action |
|----------|--------|
| **Ctrl+Shift+M** / **Cmd+Shift+M** | **Refile** — Open the Refile dialog to move the current heading (and its subtree) to another org file or heading. |
| **Ctrl+Shift+A** / **Cmd+Shift+A** | **Archive** — Open the Archive dialog to archive the current heading (move it to the archive file or archive subtree). |
| **Ctrl+Shift+I** / **Cmd+Shift+I** | **Clock in** — Start time tracking on the current heading. The editor sends a clock-in request for the heading at the cursor. |
| **Ctrl+Shift+O** / **Cmd+Shift+O** | **Clock out** — Stop the active time-tracking clock. |
| **Ctrl+Shift+E** / **Cmd+Shift+E** | **Tags** — Open the tag dialog to add or edit tags on the current heading. |

Refile, Archive, Clock in/out, and Tags only take effect when the cursor is in an Org document and, where relevant, on a heading. If no heading is selected or no clock is active, the action may show a message or do nothing.

---

## Quick Capture dialog

When the **Quick Capture** dialog is open (e.g. after **Ctrl+Shift+C** / **Cmd+Shift+C**):

| Shortcut | Action |
|----------|--------|
| **Ctrl+Enter** / **Cmd+Enter** | **Submit** — Save the capture and close the dialog. The entry is sent to your inbox org file (or configured capture target). |
| **Esc** | **Cancel** — Close the dialog without saving. |

The dialog may show hints such as “Ctrl+Enter to capture, Esc to cancel” at the bottom.

---

## Summary

- **Global:** **Ctrl+Shift+C** / **Cmd+Shift+C** — Quick Capture; **Ctrl+Shift+J** / **Cmd+Shift+J** — Journal for the day.
- **Org editor:** **Ctrl+Shift+M** Refile, **Ctrl+Shift+A** Archive, **Ctrl+Shift+I** Clock in, **Ctrl+Shift+O** Clock out, **Ctrl+Shift+E** Tags.
- **Quick Capture dialog:** **Ctrl+Enter** / **Cmd+Enter** Submit, **Esc** Cancel.

---

## Related

- **Org Quick Capture** — How Quick Capture and inbox capture work.
- **Org Mode files** — TODOs, refile, archive, and time tracking.
- **Editor features** — Toolbar and org-specific actions.
