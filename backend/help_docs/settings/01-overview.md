---
title: Settings overview
order: 1
---

# Settings

Settings control your workspace, model choices, external connections, appearance, and org-mode behavior. Open **Settings** from the user menu in the top bar. This page lists each settings tab and what it does; use the tabs on the left to jump to a section. For deeper detail on specific areas, see the related topics below.

---

## User Profile

Set your **display name**, **email**, **phone**, **birthday**, **timezone**, and **time format**. Optional fields like **zip** can be used for weather in the status bar. **AI context** and **user facts** let you give the system persistent information about yourself so agents can personalize responses.

---

## Appearance

- **Theme** — Switch between **light** and **dark** mode. The choice is saved and applied across the app.
- **Accent color** — Pick an accent color for buttons and highlights. Works with both themes.

You can also toggle theme from the user menu (light/dark icon) without opening Settings.

---

## Personas

Manage **personas** that influence how agents respond. You can define named personas and select which one (if any) is active so that chat and Agent Factory agents adopt the right tone and style.

---

## Models

Choose which models power chat, fast replies, image generation, and image analysis. You can pick from instance-configured models. **User LLM Providers** lets you add your own API endpoints (e.g. OpenAI-compatible or other providers) and use them as model options. See **Models and providers** for full detail.

---

## News

Configure **news synthesis** and display: which model is used to summarize headlines, minimum sources, recency, and diversity. This tab is relevant if your instance has the News feature enabled (e.g. for admins or users with the news capability).

---

## Org-Mode

Configure **TODO keywords** and sequences (e.g. TODO → NEXT → DONE), **tags**, and **agenda** display preferences. These settings affect how Org files are parsed, how the All TODOs view and agenda work, and how org tools behave in agents and playbooks.

---

## Media

Add and manage **media sources** for music, audiobooks, and podcasts. Common options include **Navidrome** (music) and **Audiobookshelf** (audiobooks and podcasts). You enter the server URL and credentials; once configured, the **Media** page and status bar can browse and play content. If no source is configured, the Media nav entry may be hidden.

---

## Connections

Configure **external connections** used by the system and by agents:

- **Email** — Microsoft OAuth or IMAP/SMTP so agents can read and send email.
- **Messaging bots** — Telegram, Discord, Slack, Twilio SMS: connect your accounts so agents can send and receive messages on those platforms.
- **System email** — SMTP for notifications (e.g. alerts, digest emails).
- **Device tokens** — Push notification tokens for devices.

See **External connections** for step-by-step setup. These are separate from **Agent Factory connectors**, which are API definitions attached to custom agents.

---

## Browser Sessions

View and manage **browser auth sessions**. You can see active sessions and revoke them (e.g. after using another device or browser) so that only current sessions remain logged in.

---

## Database (admin only)

Available only to **admins**. Options may include clearing document caches, resetting or maintaining the **Qdrant** vector store, and **Neo4j** knowledge graph. Use with care; these operations affect all data.

---

## User Management (admin only)

Available only to **admins**. Create, edit, and delete **users**, assign **roles** and **capabilities**, and manage access. Controls who can log in and what features they can use.

---

## Related

- **Models and providers** — Chat, fast, and image model selection; User LLM Providers.
- **External connections** — Email, messaging bots, and system email setup in detail.
- **Document Library overview** — Where your files and folders live.
- **Agent Factory overview** — Custom agents and playbooks.
