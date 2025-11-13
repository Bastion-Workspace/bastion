## Privacy & Secrets Warden — Future Agent Plan

### Overview
Guards against secret leakage, PII exposure, and misconfigurations across outputs, logs, and configuration. Runs inline (pre-send checks) and as background audits.

### Primary Use Cases
- Scan generated responses for PII/secrets; redact or block with rationale
- Audit `.env`, config, and logs for exposed secrets and weak settings
- Produce remediation guidance and track resolution

### Inputs & Data Sources
- Outbound responses (pre-dispatch)
- Env/config files, secret stores, application logs/diagnostics

### Structured Output (example fields)
- task_status: complete|incomplete|error
- findings: [{ type: pii|secret|policy, location, snippet_hash, severity, recommendation }]
- redactions: [{ rule_id, before_hash, after_hash }]
- policy_refs: [str]
- confidence: float
- timestamp: ISO8601

### Background Tasks
- Scheduled environment and log scans with allowlists/denylists
- Drift detection on policy rules; metrics on incident rates

### LangGraph & Celery Integration
- Node: `privacy_warden_agent` (inline check node)
- Background: `privacy.scan_env`, `privacy.scan_logs`

### Env Considerations (examples)
- Secret store endpoints, redaction modes, policy rule sets
- Allowed domains and output channels

### Future Extensions
- DLP integrations and classification tuning per jurisdiction



