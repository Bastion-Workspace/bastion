---
title: Agenda and calendar
order: 7
---

# Agenda and calendar

The **Agenda** view merges **org-mode** scheduled and deadline items with **external calendar events** (Microsoft O365 and CalDAV) into a single, date-grouped list. You open it as a tab from the Document Library (e.g. **Agenda** or **org-agenda**). You choose which org files and which calendars to include, switch between **Day**, **Week**, and **Month** ranges, and click items to open the source document or see event details. This page describes how the Agenda works and how to connect calendars.

---

## Opening the Agenda

- In the **Document Library**, open the **Agenda** tab (or **org-agenda** view). It may be available from the sidebar (e.g. under org-related entries) or as a tab type when viewing documents. The Agenda panel shows a header with view-mode toggles and calendar/org-file options, then a list of items grouped by date.

---

## View modes

At the top of the Agenda view you can switch the date range:

- **Day** — Items for today only.
- **Week** — Items for the next seven days (default).
- **Month** — Items for the next 30 days.

The list is grouped by date (e.g. Today, Tomorrow, or weekday and date). Within each day, items are sorted by time. Org items use their scheduled or deadline timestamp; calendar events use their start time.

---

## Calendars

- **Calendars** section — Expand or collapse the **Calendars** section to see connected calendar accounts and their calendars. The system loads **calendar connections** (e.g. Microsoft O365, CalDAV) from Settings. For each connection you see a list of **calendars** (e.g. default calendar, shared calendars). Use the checkboxes to **select which calendars** to include in the Agenda. Only selected calendars contribute events.
- **Connecting calendars** — To see O365 or CalDAV events, you must connect an account in **Settings > Connections** (e.g. Microsoft OAuth for Office 365, or CalDAV if your instance supports it). After connecting, calendars appear in the Agenda’s Calendars section. See **External connections** for setup.

---

## Org files

- **Org file selection** — The Agenda loads **scheduled** and **deadline** items from your org files. You can choose **which org files** to include (e.g. `calendar.org` and others). A list of known org files is shown; check or uncheck each file to include or exclude it. Items from selected files are merged with calendar events and grouped by date.
- **Recurring tasks** — Org items with repeater syntax (e.g. `+1w`, `.+2d`) are recognized and may show a recurring label (e.g. “Every 1 week”). They appear on the relevant dates within the selected range.

---

## Clicking items

- **Org items** — Click an **org-mode** item to open the **source document** at that heading. The document opens in a tab and scrolls to the heading (and line) so you can edit or update the TODO. If the system cannot find the document, you may see an error; ensure the org file is in your library and selected in the Agenda.
- **Calendar events** — Click a **calendar event** (O365/CalDAV) to open a **popover** with details: title, time, location, all-day flag, body preview, and optionally a link to open the event in the provider (e.g. Outlook). The popover does not open the document library; calendar events live in the external service.

---

## Summary

- The **Agenda** merges **org-mode** scheduled/deadline items with **O365 and CalDAV** events into a date-grouped list. Open it from the Document Library as the Agenda tab.
- Use **Day**, **Week**, or **Month** to set the range. **Calendars** section: select which calendars to show (connect accounts in Settings > Connections). **Org files**: select which org files to include.
- **Click** an org item to open the document at that heading; **click** a calendar event to see the event popover.

---

## Related

- **Org Mode files** — How org files, TODOs, and schedules work.
- **External connections** — Connecting Microsoft O365 and CalDAV for calendar events.
- **Document Library overview** — Tabs and sidebar.
- **Settings overview** — Where Connections lives.
