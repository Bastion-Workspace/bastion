## Notification Drummer — Future Agent Plan

### Overview
Unified alerting and notification coordinator: dedupes, throttles, correlates, and dispatches alerts across channels (in‑app, email, webhook, chat).

### Primary Use Cases
- Multi-source alert ingestion and correlation threads
- Throttling/deduplication to avoid alert fatigue
- Routing by severity, team, on-call schedule, and user preferences

### Inputs & Data Sources
- Events from domain agents (finance, ops, triage, incidents)
- User/team preferences, on-call schedules

### Structured Output (example fields)
- task_status: complete|incomplete|error
- notification: { id, title, body, severity, correlation_id }
- channels: [in_app|email|webhook|chat]
- recipients: [user_id|group]
- delivery_status: { channel: status }
- confidence: float
- timestamp: ISO8601

### Background Tasks
- Channel dispatchers with retries/backoff
- Quiet hours logic and digest bundling

### LangGraph & Celery Integration
- Node: `notification_drummer_agent`
- Background: `notify.dispatch`, `notify.digest`

### Env Considerations (examples)
- SMTP/Webhook/chat creds; rate limits; quiet hours

### Future Extensions
- Incident bridge with Incident Commander and DevOps Monitor



