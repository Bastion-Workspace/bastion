## Email & Ticket Triage + Calendar & PIM Adjutant — Future Agent Plan

### Overview
Two tightly related assistants: (1) Email/Ticket Triage classifies, summarizes, prioritizes, and drafts responses; (2) Calendar & PIM Adjutant schedules, finds slots, sets reminders, and manages tasks.

### Primary Use Cases
- Inbox/ticket queue triage: classification, priority, routing, suggested replies
- Summarize long threads; extract decisions and next steps
- Propose meeting times based on constraints and availability
- Create tasks/reminders tied to messages and deadlines

### Inputs & Data Sources
- Email (IMAP/Graph), ticket systems (e.g., JIRA, Zendesk), calendar APIs
- User preferences: working hours, SLAs, priority rules, routing rules

### Structured Output (example fields)
- task_status: complete|incomplete|error
- classification: { category, priority, intent }
- summary: str
- suggested_reply: str
- routing: { queue|assignee, rationale }
- tasks: [{ description, due_date, related_message_id }]
- meeting_proposals: [{ start, end, timezone, participants, location|link }]
- confidence: float
- sources: [{ system, id|url }]

### Background Tasks
- Poll inbox/tickets; batch triage with rate-limit respect
- SLA monitors: approaching deadlines and unattended items
- Reminder scheduling for follow-ups and tasks

### LangGraph & Celery Integration
- Nodes: `email_triage_agent`, `calendar_adjutant_agent`
- Background: `triage.poll`, `triage.autoreply`, `calendar.reminders`
- Permission gates for sending emails/creating events

### Env Considerations (examples)
- Mail/ticket API credentials (OAuth or app tokens)
- Timezone, work-hours window, SLA thresholds
- Default calendars/mailboxes

### Future Extensions
- Smart delegations with organization directory context
- Thread intent tracking and auto-follow-up drafts



