---
title: Org Quick Capture
order: 4
---

# Org Quick Capture

**Quick Capture** lets you add a note, TODO, contact, or meeting to an Org file without opening the file. For **journal entries**, use **Ctrl+Shift+J** (Journal for the day) instead; see below. You press a keyboard shortcut from anywhere in the app, choose a template and target, enter content, and capture. The item is appended to your configured inbox or the file and heading you choose. This page describes the shortcut, the dialog, and how to use each capture type.

---

## Opening Quick Capture

- **Shortcut** — Press **Ctrl+Shift+C** (Windows/Linux) or **Cmd+Shift+C** (Mac). The Quick Capture dialog opens no matter which page you are on (Documents, Chat, Agent Factory, etc.).
- **Dialog** — The dialog shows **Quick Capture** in the title and displays **(Ctrl+Shift+C)** as a reminder. Close it with **Cancel**, the **Esc** key, or the X button. **Ctrl+Enter** (or **Cmd+Enter** on Mac) submits the capture.

---

## Template types

At the top of the dialog you choose a **Template**:

- **Note** — A quick note. You enter content and optional tags; it is captured as a plain heading or note in the target Org file.
- **TODO** — A task item. You enter the task description and can add **tags**, **priority** (A, B, C), **Scheduled** date, and **Deadline**. Use **Show Advanced Options** to reveal priority and dates. The item is captured as a TODO heading.
- **Contact** — Contact info. Choosing this opens a dedicated contact form (name, phone, email, etc.). The entry is captured in the configured contacts or Org file.
- **Meeting** — Meeting notes. You enter a topic or title and optional tags; it is captured as a heading (often under a meeting structure) in the target file.

**Journal** — Journal entries are not in Quick Capture. Use **Ctrl+Shift+J** (or **Cmd+Shift+J** on Mac) to open **Journal for the day**. That modal shows today’s entry (or lets you pick another date); you can view and edit existing content or type a new entry and save. It uses your configured journal file and organization (Settings > Org-Mode). Placeholders like `%T` (date and time), `%t` (date), and `%<%I:%M %p>` (time) work in the content.

After picking a template, type in the **Content** (or **Task description** for TODO) field. For TODO you can add tags by typing in the **Tags** field and pressing **Enter**; **Ctrl+Enter** in the tags field also triggers capture.

---

## Target file and heading

Quick Capture uses an **inbox** configuration: a default Org file (and optionally a heading) where items are sent when you do not pick a different target. The inbox is set in **Settings > Org-Mode** (e.g. “Inbox file” or “Capture target”). If you have only one inbox file, all captures go there.

- **Multiple inboxes** — If your instance has multiple inbox files configured, the dialog may show a warning. Configure a single default inbox in Settings > Org-Mode to avoid ambiguity.

---

## Capturing

- **Capture (Ctrl+Enter)** — Click **Capture** or press **Ctrl+Enter** (or **Cmd+Enter** on Mac) to send the item to the target. The dialog shows success (e.g. “Captured to …”) and then closes. If capture fails (e.g. missing inbox), an error message appears in the dialog.
- **Cancel (Esc)** — Click **Cancel** or press **Esc** to close without capturing.

Captured items appear in the target Org file. TODOs are cataloged and show up in **All TODOs** and in org tools that agents and playbooks use.

---

## Summary

- **Ctrl+Shift+C** (or **Cmd+Shift+C** on Mac) opens Quick Capture from anywhere.
- Choose a template: **Note**, **TODO**, **Contact**, or **Meeting**. Fill in content and, for TODO, optional tags, priority, scheduled date, and deadline.
- For **journal**, use **Ctrl+Shift+J** (Journal for the day) instead. Set the inbox and journal location in **Settings > Org-Mode**.
- **Ctrl+Enter** captures; **Esc** cancels. Captured items appear in the target Org file and in All TODOs (for TODOs).

---

## Related

- **Org Mode files** — How Org files, TODOs, and the catalog work.
- **Chat and agents overview** — Chat and agents that can use org tools.
- **Settings overview** — Where to set Org-Mode and inbox options.
