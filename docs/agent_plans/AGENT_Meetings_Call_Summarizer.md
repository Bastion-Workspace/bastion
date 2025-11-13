## Meetings & Call Summarizer — Future Agent Plan

### Overview
Automates meeting/call capture: transcription, summarization, decision/action extraction, owners and deadlines. Operates interactively (on-demand summaries) and as a background agent (auto-processing scheduled meetings and uploaded recordings).

### Primary Use Cases
- Live/recorded meeting summaries with sections (Overview, Topics, Decisions, Risks)
- Action item extraction with owners and due dates
- Follow-up email draft and calendar tasks creation
- Highlights, sentiment, and Q&A on transcripts

### Inputs & Data Sources
- Audio/video recordings (file uploads, URLs)
- Calendar metadata (title, attendees, time, agenda)
- Meeting notes/chat logs (optional)

### Structured Output (example fields)
- task_status: complete|incomplete|error
- summary: str
- topics: [str]
- action_items: [{ description, owner, due_date, priority }]
- decisions: [str]
- risks: [str]
- participants: [str]
- sentiment: { overall: [-1..1], notes: str }
- sources: [{ type, url|id, timestamp }]
- confidence: float
- data_timestamp: ISO8601

### Background Tasks
- Auto-transcribe new recordings and calendar meetings
- Chunk and embed transcripts; update vector store and knowledge graph
- Generate briefs and post to conversation or notification channels

### LangGraph & Celery Integration
- Node: `meetings_summarizer_agent` (interactive)
- Background: Celery tasks `meetings.transcribe`, `meetings.summarize`, scheduled by calendar hooks
- Permission gating for external STT APIs; local-first if available

### Env Considerations (examples)
- STT provider keys (e.g., `OPENAI_API_KEY` or `WHISPER_SERVER_URL`)
- Max recording length, language preferences
- PII handling and redaction toggles

### Future Extensions
- Speaker diarization and per-speaker action capture
- Slide/OCR integration for referenced materials
- Automatic meeting minutes with approval workflow



