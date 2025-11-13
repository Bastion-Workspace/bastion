## Desktop App Prompts — Compatibility Plan for LangGraph Writing System

BULLY! By George, this document maps the desktop application's Fiction prompt logic to our LangGraph writing system. It preserves authorial control (Style/Rules supremacy), supports two operation modes, and ensures we can produce both human‑readable guidance and structured outputs for precise edits.

### Purpose
- Align desktop chains/prompts with LangGraph agents/nodes
- Enforce persona override and Style/Rules precedence
- Provide mode routing, context loading, and output contracts

### Prompt Structure (Summary)
Primary goal: “MASTER NOVELIST” producing publication‑ready prose, following the established style guide above all else, with time context.

Two modes:
- Full Story Analysis Mode (triggered by keywords)
  - Focus on overall narrative structure, consistency, and flow
  - Include only essential style principles (truncated), outline, and basic frontmatter (title, genre)
  - High‑level critique; no detailed rewrite
- Regular Editing Mode
  - Load full Style Guide (with Writing Sample usage notes and Critical Rules)
  - Load Document metadata, Rules, Character Profiles, Outline (enhanced chapter context if available)
  - Strict instructions on using Outline as goals, not as text; originality rules
  - Detailed response format for revisions and insertions

Response expectations for Regular Editing:
- Revision/Insertion format blocks for human review
- Strict scope and text‑matching rules
- New chapter formatting: `## Chapter [number]`
- Originality and plagiarism avoidance
- Use project references for consistency, never as text to copy

### Inputs Consumed (Desktop → LangGraph)
- `request`: User prompt string
- `_styleGuide`: Raw style guide text
- `_parsedStyle`: Optional parsed structure with `WritingSample`, `CriticalRules`
- `_rules`: Raw universe rules
- `_outline`: Outline text
- `_characterProfiles`: List of character profile texts
- `_frontmatter`: Map of YAML frontmatter fields (title, author, genre, series, summary, chapterX keys)
- `_fictionContent`: Current manuscript content
- `_currentCursorPosition`: Cursor index for chapter detection

### Keyword Triggers
- FULL_STORY_KEYWORDS (examples):
  - ["analyze entire story", "full story analysis", "overall story structure", "complete story review", "story-wide analysis", "analyze the whole story"]
- CHAPTER_GENERATION_KEYWORDS (examples):
  - ["generate chapter", "new chapter", "write chapter", "start chapter", "continue chapter"]

These lists should be configurable:
- `WRITING_FULL_STORY_KEYWORDS`
- `WRITING_CHAPTER_GEN_KEYWORDS`

### Mode Routing in LangGraph
- Node: `mode_router`
  - Input: `request`, `fiction_context`
  - Logic: If any FULL_STORY_KEYWORDS present → Full Story Analysis Path; else → Regular Editing Path

### Context Loader (file_context_loader)
- Resolve frontmatter references and load: Style, Rules, Outline, Characters
- For Full Story Analysis Mode:
  - Truncate Style Guide to `WRITING_STYLE_TRUNCATE_CHARS` (default 500)
  - Include Outline and basic frontmatter (title, genre)
- For Regular Editing Mode:
  - Include full Style Guide; if parsed, surface Writing Sample usage and up to N critical rules
  - Include Rules, Outline (or enhanced outline prompt), Character Profiles, and Document metadata
  - Compute current chapter via `ChapterDetectionService` equivalent

### Outline Enhancer
- Service analogous to `_outlineChapterService`:
  - `get_chapter_context(currentChapter)`
  - `build_chapter_outline_prompt(context, isChapterGeneration)`
  - Fallback: present raw outline with strict guidance to treat it as goals

### Persona Override Enforcement
- Always disable user’s preferred AI persona for writing‑critical operations:
  - System message: “Persona disabled. Adhere strictly to Style Guide and Rules. Do not invent canon beyond Rules without proposing updates.”
  - Orchestrator sets `persona=None`, injects Style/Rules context
  - If deviation detected, reject and retry

### Time Context Injection
- Add current date/time and local timezone to system or developer message for temporal awareness
- Example: `Current Date/Time: 2025-08-13 14:05`, `Local Time Zone: America/Chicago`

### Response Contracts
- Human‑readable blocks (for the editor UI):
  - Revision block: “Original text:” → “Changed to:”
  - Insertion block: “Insert after:” → “New text:”
- Structured outputs (machine‑readable):
  - Prefer emitting `ManuscriptEdit` JSON (see `WRITING_LANGGRAPH_SYSTEM_PLAN.md` schemas)
  - For Full Story Analysis, emit `AnalysisReport`
  - For Proofreading, emit `ProofreadCorrections`

Recommended dual‑channel output:
- Primary: structured JSON payload
- Secondary: human blocks for chat display

### Agent Mapping
- Full Story Analysis Mode → `story_analysis_agent`
- Regular Editing Mode → `manuscript_agent` (+ optional `proofreading_agent` post‑pass)
- Outline work (when in Outline tab) → `outline_agent`
- Characters/Rules/Style tabs → respective agents

### Safety and Originality
- Never copy from Outline/Rules/Characters; they are guidance only
- Use Writing Sample for style emulation, not for content copying
- Respect project boundaries; do not leak content from other projects

### Example System/Developer Messages

System banner (persona override):
```text
You are a MASTER NOVELIST crafting publication-ready fiction. Persona disabled. Adhere strictly to the project’s Style Guide and Rules above all else. Do not invent canon beyond Rules without proposing updates. Maintain originality; do not copy from external works.
```

Developer message (mode and context):
```text
=== CURRENT DATE AND TIME ===
Current Date/Time: {{now_iso}}
Local Time Zone: {{timezone_name}}

{{#if full_story_mode}}
=== FULL STORY ANALYSIS MODE ===
Focus on overall narrative structure, consistency, and flow.

{{#if style_core}}
=== CORE STYLE PRINCIPLES ===
{{style_core}}
{{/if}}

{{#if outline}}
=== STORY OUTLINE ===
Use this outline to understand the intended story structure:
{{outline}}
{{/if}}

=== FULL STORY ANALYSIS INSTRUCTIONS ===
- Focus on structure and pacing
- Identify cross-chapter consistency issues
- Suggest high-level improvements and note plot holes
- Consider character arcs throughout
{{/if}}

{{#if regular_mode}}
=== STYLE GUIDE ===
{{style_full}}

{{#if style_sample}}
** WRITING SAMPLE USAGE **
CRITICAL: Emulate technique (voice, pacing, syntax), never copy content.
{{/if}}

{{#if style_rules}}
** CRITICAL STYLE REQUIREMENTS **
{{style_rules}}
{{/if}}

=== DOCUMENT METADATA ===
Title: {{title}}
Author: {{author}}
Genre: {{genre}}
Series: {{series}}
Summary: {{summary}}

=== CORE RULES ===
{{rules}}

=== CHARACTER PROFILES ===
{{character_profiles}}

{{#if outline_enhanced}}
{{outline_enhanced}}
{{else}}
=== STORY OUTLINE: PLOT STRUCTURE & STORY BEATS ===
{{outline}}
{{/if}}

=== CRITICAL OUTLINE USAGE INSTRUCTIONS ===
[... full block from desktop spec ...]
{{/if}}
```

### Configuration
- `WRITING_FULL_STORY_KEYWORDS` — comma‑separated
- `WRITING_CHAPTER_GEN_KEYWORDS` — comma‑separated
- `WRITING_STYLE_TRUNCATE_CHARS` — integer (default 500)
- `WRITING_PERSONA_ENFORCEMENT=strict`

### Integration Notes
- Prefer structured outputs per schemas in `docs/WRITING_LANGGRAPH_SYSTEM_PLAN.md`
- UI should render human blocks when present and provide Apply buttons wired to `ManuscriptEdit`
- Background Consistency Checker can re‑scan after edits and propose `PropagationPlan`


