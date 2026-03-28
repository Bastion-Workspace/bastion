---
title: RSS feeds
order: 1
---

# RSS feeds

**RSS feeds** let you follow websites and blogs from inside Bastion. Feeds appear in the Document Library sidebar under **RSS** (user feeds) and **Global Documents** (global feeds). You add, refresh, and delete feeds from the sidebar or the **RSS Feed Manager**; when you open a feed, its **articles** open in a tab and can be searched like other content. This page describes how to use RSS feeds and where they live.

---

## Where feeds appear

In the **Document Library** left sidebar you see:

- **RSS** (or “RSS Feeds”) — Your **user** feeds. Only you see and manage these unless you share them.
- **Global Documents** — May contain a **Global RSS** section with **global** feeds. These are shared across the instance; typically only admins add or edit them.

Expand the RSS section to see the list of feeds. Each feed shows its **name** and may show an **unread count** (e.g. a badge). Click a feed to open its articles in a tab in the center area.

---

## Adding a feed

- **From the sidebar** — Right-click the **RSS** section (or the “RSS Feeds” / “Global RSS” node) and choose **Add RSS Feed**. In the dialog, enter the **feed URL** (e.g. `https://example.com/feed.xml`) and optionally a **name**. Choose whether it is a **user** or **global** feed if both are available. Save. The feed appears in the list and is fetched; articles show up after the first refresh.
- **RSS Feed Manager** — If your instance provides an **RSS Feed Manager** (e.g. from a menu or Documents page), you can add, edit, and delete feeds from there. Same idea: feed URL, name, and scope (user vs global).

---

## Opening and reading articles

- **Open feed** — Click a feed in the sidebar. A tab opens (e.g. “RSS” or the feed name) showing the list of **articles** (headlines, summary, date). Click an article to read it in the same tab or in an article viewer. Unread counts update when you open and read articles.
- **Article viewer** — Articles may open in an **RSS article viewer** that shows title, source, date, and full or summary content. You can scroll and, if supported, mark as read.

---

## Refreshing and deleting feeds

- **Refresh** — Right-click a feed in the sidebar and choose **Refresh** (or use the feed’s context menu). The system fetches the latest items from the feed URL and updates the article list. Unread counts update. Use refresh when you want to pull in new posts without waiting for the next automatic poll.
- **Delete** — Right-click the feed and choose **Delete**. You may be asked whether to **delete only the feed** (keep imported articles) or **delete the feed and all its articles**. Confirm. The feed is removed from the list. Deletion is permanent.

---

## Unread counts

Each feed can show an **unread count** (e.g. a number next to the feed name). Opening the feed and reading articles marks them as read; the count goes down. Counts are per user for user feeds and per user for global feeds. Refreshing the feed can add new articles and increase the count.

---

## Searching RSS articles

Imported **articles** are stored and indexed like other content. You can **search** from the main search or from the document/library search; results include matching RSS articles. So you can find past articles by keyword or phrase without opening each feed.

---

## Summary

- **RSS** and **Global RSS** in the Document Library sidebar list your **user** and **global** feeds. Click a feed to open its **articles** in a tab.
- **Add RSS Feed** from the sidebar or RSS Feed Manager; **Refresh** to fetch new items; **Delete** to remove the feed (and optionally all articles).
- **Unread counts** appear next to feeds; reading articles marks them read. **Search** includes RSS articles.

---

## Related

- **Document Library overview** — Sidebar layout and tabs.
- **News** — The News page (headlines and synthesis), which is separate from RSS.
- **Folders** — How global and user areas work.
