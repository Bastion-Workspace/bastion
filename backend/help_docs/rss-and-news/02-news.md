---
title: News
order: 2
---

# News

The **News** page shows **headlines** that are synthesized from multiple sources. You see a list of headline cards; each card has a title, summary, source count, and diversity score. Click **Open** to read the full article and any AI-generated summary. News behavior and visibility can be **capability-gated** (e.g. only admins or users with a “news” feature). This page describes the News page and how it relates to settings.

---

## Opening News

- **Navigation** — If the **News** entry appears in the main navigation, click it to go to the **News** page. If you do not see it, your instance may not have News enabled or your user may not have the required capability (e.g. `feature.news.view` or admin).
- **URL** — The News page is at `/news`. A specific article may be at `/news/:id`.

---

## Headlines list

On the News page you see a **grid or list of headline cards**. Each card typically shows:

- **Title** — The headline title.
- **Summary** — A short summary (often AI-synthesized from multiple sources).
- **Severity** — A label such as “breaking”, “urgent”, or “default”, which may be shown as a chip or badge.
- **Sources** — How many sources were used (e.g. “3 sources”).
- **Diversity** — A diversity score (e.g. “diversity 80%”) indicating source variety.
- **Open** — A button to open the full article and details.

Click **Open** (or the card) to go to the **News detail** page for that headline.

---

## News detail page

- **Full article** — The detail page shows the full headline, summary, and usually the synthesized content or links to sources.
- **Synthesis** — Headlines are often built by an **AI synthesis** step that combines multiple articles into one summary. The model and parameters used can be configured in **Settings > News**.

---

## News settings

In **Settings > News** (if available) you can configure:

- **Synthesis model** — Which model is used to generate headline summaries from multiple sources.
- **Minimum sources** — Minimum number of sources required before a headline is produced.
- **Recency** — How recent the source articles must be (e.g. last 24 hours).
- **Diversity** — Target diversity of sources so the summary is not biased toward a single outlet.

These settings affect which headlines appear and how they are summarized. Changes apply to future fetches; existing headlines are not re-synthesized.

---

## Capability gating

- **Visibility** — The News nav entry and the News page may only be visible to **admins** or to users with a capability such as `feature.news.view`. If you do not see News, ask your administrator to grant the capability or enable the feature.
- **Settings** — The News tab in Settings may also be restricted so only certain users can change synthesis and source options.

---

## Summary

- The **News** page lists **synthesized headlines** with title, summary, severity, source count, and diversity. Click **Open** to read the full article.
- **News detail** shows the full content and synthesis. **Settings > News** lets you set synthesis model, min sources, recency, and diversity.
- **News** may be **capability-gated**; if you do not see it, you may need the news feature or admin role.

---

## Related

- **RSS feeds** — RSS feeds in the Document Library (separate from News synthesis).
- **Settings overview** — Where the News tab lives.
- **Document Library overview** — Sidebar and tabs.
