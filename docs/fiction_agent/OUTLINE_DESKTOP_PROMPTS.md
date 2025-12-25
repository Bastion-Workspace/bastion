## Outline Desktop Prompts — Compatibility Plan for LangGraph Writing System

BULLY! By George, this document maps the desktop Outline creation/refinement prompt to our LangGraph writing system. It supports structural outline generation, precise granular revisions, and strict formatting for parser compatibility.

### Purpose
- Align desktop outline chains/prompts with LangGraph nodes
- Enforce structural clarity, numbering, and bullet formatting
- Support human‑readable revision blocks and structured outputs for edits

### Prompt Structure (Summary)
- Time context: Inject current date/time and local timezone.
- Load references when available:
  - Style Guide (raw): inform tone/approach of outline suggestions
  - Rules (raw): universe constraints and facts
  - Character Profiles: via character reference service section
- Provide Outline Development Guidance with canonical sections:
  - Overall Synopsis (2–3 sentence story summary)
  - Notes (themes, tone/mood, techniques, symbols/motifs, genre considerations)
  - Characters (Protagonists/Antagonists/Supporting; concise role lines)
  - Outline (chapter‑numbered with summaries and beats)
- Document metadata (title, author, genre, series) if present in frontmatter.
- Current Outline Content: include full text to build on.
- Response format: exact “Original text:” → “Changed to:” or “Insert after:” → “New text:” blocks; strict precision and matching rules.
- Originality: avoid plot plagiarism; use references for consistency, not copying.
- Critical instructions: structural rules, chapter numbering, beats limits, parser compatibility.
- Clarifying questions: allowed sparingly when essential.

### Inputs Consumed (Desktop → LangGraph)
- `_styleGuide`: raw style guide text
- `_rules`: raw universe rules
- `_characterReferenceService`: builds character profiles section (string)
- `_frontmatter`: YAML map (title, author, genre, series)
- `_outlineContent`: current outline text
- `request`: user instruction string

### Agent Mapping
- `outline_agent` (interactive)
  - Generates new outlines or refines existing ones
  - Emits structured outputs and human blocks
- Optional background validation
  - Outline linter checks numbering, bullets, indentation; can propose fixes

### Response Contracts
- Primary structured payloads:
  - `OutlineModel` for complete outline generation (see `WRITING_LANGGRAPH_SYSTEM_PLAN.md` schemas)
  - `ManuscriptEdit` for precise insert/replace/delete ops targeting the outline file (reuse generic edit model)
- Secondary human‑readable blocks for UI:
  - Revision: “Original text:” → “Changed to:”
  - Insertion: “Insert after:” → “New text:”

Precision requirements for revision blocks:
- Exact, complete, verbatim matches from the outline
- Preserve whitespace/indentation/headings/bullets
- Provide sufficient context (10–20 words) for uniqueness

### Structural Rules (Enforced)
- Chapters must use: `## Chapter [number]` (e.g., `## Chapter 1`, `## Chapter 2`)
- Do not name chapters; only number them.
- Each chapter includes a **BRIEF, HIGH-LEVEL** 3–5 sentence summary paragraph (NOT a detailed synopsis).
  - The summary should be a quick overview of main events, not a chapter-by-chapter book
  - Think of it as a "back of the book" description for this chapter - what happens in broad strokes?
  - Details belong in beats, not in the summary paragraph
- Major beats as main bullets `-` (max 8–10 per chapter).
- Sub‑bullets use two spaces then `-` (up to 2 per main beat).
- Focus on structural events: plot actions, reveals, conflicts, consequences; avoid dialogue/prose.
- Characters section supports both header and bullet formats but must remain concise.

### Originality & Consistency
- Create original story structures; avoid copying plot sequences from published works.
- Use Character Profiles/Rules/Style Guide to ensure consistency with the project’s universe and tone.

### Clarifying Questions Protocol
- Ask only when truly necessary (ambiguity, conflicts, missing critical context).
- Limit to 1–3 focused questions; explain why they’re needed.

### Example System Banner
```text
You are an AI assistant specialized in helping users develop and refine story outlines. You will analyze and respond based on the sections provided below.
```

### Example Developer Message Skeleton
```text
=== CURRENT DATE AND TIME ===
Current Date/Time: {{now_iso}}
Local Time Zone: {{timezone_name}}

{{#if style}}
=== STYLE GUIDE ===
Use this style guide to inform tone and approach:
{{style}}
{{/if}}

{{#if rules}}
=== CORE RULES ===
Fundamental universe rules and facts:
{{rules}}
{{/if}}

{{#if characters_section}}
{{characters_section}}
{{/if}}

=== OUTLINE DEVELOPMENT GUIDANCE ===
[... canonical sections and formatting rules ...]

{{#if metadata}}
=== DOCUMENT METADATA ===
Title: {{title}}\nAuthor: {{author}}\nGenre: {{genre}}\nSeries: {{series}}
{{/if}}

{{#if current_outline}}
=== CURRENT OUTLINE CONTENT ===
{{current_outline}}
{{/if}}

=== RESPONSE FORMAT ===
Use exact revision/insert blocks as specified.
```

### Configuration
- `OUTLINE_MAX_BEATS_PER_CHAPTER=10`
- `OUTLINE_MAX_SUBBEATS_PER_BEAT=2`
- `OUTLINE_REQUIRE_NUMERIC_CHAPTERS=true`
- `OUTLINE_ALLOW_CLARIFYING_QUESTIONS=true`

### Integration Notes
- Prefer emitting `OutlineModel` for full (re)generation; use `ManuscriptEdit` for surgical updates to the outline file.
- UI renders human blocks and binds Apply actions to `ManuscriptEdit`.
- Outline linter can run post‑edit to ensure numbering and bullet compliance.



