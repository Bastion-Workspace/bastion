---
title: Agent profile settings
order: 7
---

# Agent profile settings

An **agent profile** is the top-level definition for a custom agent: identity, model, default playbook, schedules, monitors, memory, and more. This page maps the main **Agent Editor** sections to what they control. Built-in or **shared** profiles may be read-only or offer **Make my own copy** first.

---

## Identity

From the **Identity** card:

- **Name** — Display name for lists and headers.
- **Handle** — Optional **@mention** handle for chat; leave blank for schedule-only or run-only agents.
- **Show in chat @ menu** — When off, the agent is hidden from the @ picker but can still be used from **agent lines** and other agents.
- **Model preference** — Default LLM for this profile, or **— Default** to follow the user/system default. **Model retargeted** / **Model unavailable** chips explain when the platform substituted another model; **model_source** and **model_provider_type** are stored with the choice.
- **System prompt additions** — Free text appended to the agent’s system prompt on runs.
- **Playbook** — **Default playbook** that runs when the agent is invoked.
- **Include history in prompt** — When on, recent **user + assistant** exchanges are included in LLM steps; configure **lookback** (exchanges), **summarize when over (~tokens)**, and **keep recent messages verbatim** to control cost and freshness.
- **Persona mode** — **None**, **Use default persona**, or **Select specific persona** (with **persona_id** when specific).
- **Include user context** — Injects name, email, timezone, ZIP, and AI context into LLM steps when enabled.
- **Include date/time context** — Current date and time in the user’s timezone.
- **Include user facts** — Facts from **Settings → Profile**; optional **themed memory**, **fact categories** filter, and related toggles.
- **Include agent memory** — Injects persistent **key/value agent memory** into each run (see **Agent memory** below).
- **Auto-routable** — Allows automatic selection for matching queries without an explicit @mention.

**Description** — Longer blurb on the profile may appear in sidebars or APIs even when not on the Identity card.

---

## Data Workspace

**Data Workspace** binds the agent to one or more workspaces:

- **Workspaces** — Multi-select list of workspace ids.
- **Auto-inject schema** — When on, table/schema context is added for bound workspaces.
- **Context instructions** — Extra guidance for how the model should use workspace data.

---

## External connections

**Allowed connections** caps which **email, calendar, messaging, code platform**, and similar accounts this agent may use at all. **All connected accounts** removes the profile-level cap; **Only selected accounts** is an explicit allowlist. Individual **LLM agent** / **deep agent** steps can further narrow **connection scope**.

---

## Data sources (connectors)

The **Data sources** section attaches **connector** instances to the profile so playbook **tool** steps can call your configured APIs. See **Data Connectors overview**.

---

## Schedules

**Schedule** lists **cron** or **interval** triggers (with presets such as hourly, daily, weekdays, and custom cron). Schedules run the default playbook **without** an open chat; outputs follow your run configuration and destinations.

---

## Monitors

**Monitors** are **event triggers** that run the playbook when something changes:

- **Lines** — Activity on configured agent lines / teams.
- **Email** — Mailbox-related triggers.
- **Folders** — Document folder changes.
- **Conversations** — Chat conversation triggers.

Each subsection is collapsible; chips summarize what is active.

---

## Budget

**Budget** sets optional **monthly limit (USD)**, **warning threshold** (% of limit), and whether to **enforce a hard stop** when spend exceeds the limit. Current period spend is shown when configured.

---

## Agent memory

The **Agent memory** table lists **read-only** key/value entries stored for this profile. Use **Clear memory** to wipe entries. Entries are only useful when **Include agent memory** is enabled on the Identity card.

---

## Execution history

**Execution history** lists recent runs of this profile (schedules, tests, triggers) for debugging and auditing.

---

## Active, locked, default for chat

- **Active / Paused** — Paused agents do not run on schedules or triggers until resumed.
- **Locked** — Prevents accidental edits (built-in agents behave like locked templates).
- **Set as default for chat** — New chat messages use this profile without @mention when enabled (only one default; built-in may apply when none is set).

Export (**Export YAML**), **Reset to defaults**, and **Delete** appear in the header area for eligible profiles.

---

## Test run

The **Test** card runs the agent with an optional **query** string (for steps that use `{query}`). Output streams into the panel and does not create a normal sidebar conversation when run from Agent Factory.

---

## API-only fields

Some fields exist on the profile **API** and **YAML export** even if not every control is in the UI yet, for example:

- **default_approval_policy** — `require` vs `auto_approve` for unattended runs (see **Schedules and monitors**).
- **default_run_context**, **journal_config**, **team_config**, **watch_config**, **knowledge_config** — Advanced or line-related JSON configuration.

---

## Summary

The Agent Editor is organized into **Identity** (who the agent is and how it thinks), **Data Workspace** and **connectors** (what data it can touch), **connections** (which accounts), **schedules** and **monitors** (when it runs), **budget**, **memory**, and **history**. Toggle **Active** and **default for chat** to control visibility and automatic use.

---

## Related

- **Agent Factory overview** — How profiles, playbooks, and tools fit together.
- **Playbooks overview** — Data flow and run contexts.
- **Schedules and monitors** — Triggers, `{trigger_input}`, and approval policy for background runs.
- **Tools reference** — Tools and skill discovery.
