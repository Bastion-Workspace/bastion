## Incident Commander — Future Agent Plan

### Overview
Coordinates incident response: triage, context assembly, stakeholder comms, and runbook guidance.

### Primary Use Cases
- Incident creation from alerts; severity and impact assessment
- Timeline building and root-cause hypothesis tracking
- Stakeholder updates and action assignment

### Inputs & Data Sources
- Alerts from DevOps Monitor, logs, metrics, runbooks repository

### Structured Output (example fields)
- task_status: complete|incomplete|error
- incident: { id, title, severity, status, correlation_id }
- timeline: [{ t, actor, event, evidence_ref }]
- assignments: [{ owner, task, due }]
- comms_updates: [{ channel, message, recipients }]
- confidence: float
- timestamp: ISO8601

### Background Tasks
- Auto-assemble incident rooms/threads
- Periodic status reminders and postmortem scaffolding

### LangGraph & Celery Integration
- Node: `incident_commander_agent`
- Background: `incident.create`, `incident.update_status`, `incident.postmortem`

### Env Considerations (examples)
- Channel creds, runbook repo URLs, on-call schedules

### Future Extensions
- Root-cause mapping to change events and deploys



