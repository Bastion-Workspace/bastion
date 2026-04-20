---
title: Schedules and monitors
order: 8
---

# Schedules and monitors

**Schedules** run an agent’s **default playbook** on a timer. **Monitors** run it when **events** occur (lines, email, folders, or conversations). Both are configured on the **agent profile**; they behave like **background** runs: there is usually **no interactive chat thread**, and results are delivered through the channels and destinations your playbook and platform settings define.

---

## Schedules

In the **Schedule** section:

- Add one or more **schedules** per agent.
- Each schedule uses either **cron** (with presets such as every hour, daily at 9am, weekdays at 8am, or **custom cron**) or an **interval** (e.g. every 5 or 15 minutes).
- **Timezone** applies to cron interpretations where relevant.

Scheduled runs use the same playbook as **Run** / **Test**, but **without** a live user in chat—design steps accordingly (e.g. avoid relying on `{query}` unless your trigger or monitor supplies input).

---

## Monitors (event triggers)

Under **Monitors**, four areas group **event triggers**:

| Section | Typical use |
|---------|----------------|
| **Lines** | React to activity on **agent lines** / team feeds you configure. |
| **Email** | React to mail-related events (per your watch configuration). |
| **Folders** | Run when files change under **watched folders**. |
| **Conversations** | Run when **chat conversations** match watch rules. |

Expand each accordion to configure watches; summary **chips** show what is active.

---

## Trigger input (`{trigger_input}`)

When a monitor or some **invoke-agent** style steps fire, the payload is available to the playbook as the **`{trigger_input}`** runtime variable. Use it in **prompt templates** and step inputs like any other variable—for example, to summarize the incoming event or branch on fields inside the payload.

In the Workflow Composer, content wired to **input_content** for certain invocations is passed through as **`{trigger_input}`** for the child agent.

---

## Approval steps and `default_approval_policy`

**Approval** steps pause the playbook until a human approves. In **interactive chat**, that is you in the UI. In **scheduled**, **monitor**, or other **background** runs, there is no one at the keyboard—approval steps **block** unless the agent profile allows automatic progression.

Set the profile field **`default_approval_policy`** to **`auto_approve`** for trusted pipelines so approval gates are skipped on unattended runs; keep **`require`** when a human must always confirm (default). This field is available on the profile **API** and **YAML export**; pair it with playbook design so sensitive steps remain protected when policy is `require`.

---

## Summary

- **Schedules** — Cron or interval; good for briefings and periodic jobs.
- **Monitors** — Lines, email, folders, conversations; good for reactive automation.
- **`{trigger_input}`** — Carries event payload into prompts and steps.
- **`default_approval_policy`** — `require` vs `auto_approve` for background approval behavior.

---

## Related

- **Agent profile settings** — Where schedules and monitors live alongside identity and budget.
- **Playbooks overview** — Interactive vs background execution and data flow.
- **Playbook steps and flow control** — Approval steps and wiring `{trigger_input}`.
