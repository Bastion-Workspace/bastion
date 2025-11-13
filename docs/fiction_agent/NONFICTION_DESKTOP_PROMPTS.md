## Non‑Fiction Desktop Prompts — Compatibility Plan for LangGraph Writing System

BULLY! By George, this document maps the desktop application's Non‑Fiction prompt builder to our LangGraph writing system. It preserves factual rigor, structured citations, and type‑specific guidance while enabling structured outputs for precise edits and research integration.

### Purpose
- Align desktop non‑fiction chains/prompts with LangGraph agents and nodes
- Enforce factual accuracy, citation discipline, and chronology
- Surface type‑specific guidance (Biography, Memoir, History, Academic, Journalism)
- Integrate reference materials discovered via frontmatter

### Supported Non‑Fiction Types
- Biography, Autobiography, Memoir, History, Academic, Journalism

### Core Responsibilities (as prompt specifies)
1. Factual Accuracy: verifiable, well‑sourced statements
2. Research Integration: weave findings naturally into the narrative
3. Chronological Consistency: accurate timelines and historical context
4. Citation Management: proper attribution and source integration
5. Narrative Flow: balance factual content with engaging storytelling

### Type‑Specific Guidance
- Biographical Focus (Biography/Autobiography)
  - Maintain chronological accuracy of life events
  - Balance personal details with historical context
  - Respectful, accurate portrayal of relationships
  - Verify dates, locations, and significant events
  - Consider subject's privacy and dignity

- Memoir Focus
  - Emphasize personal reflection and emotional truth
  - Maintain authenticity while protecting privacy
  - Balance personal narrative with broader themes
  - Consider impact on others mentioned

- Historical Focus
  - Strict chronological accuracy
  - Verify dates, events, and historical claims
  - Consider multiple perspectives
  - Provide context for periods
  - Cross‑reference sources for accuracy

- Academic Focus
  - Rigorous academic standards and objective tone
  - Proper citation format per discipline
  - Support claims with credible sources
  - Follow discipline‑specific conventions

- Journalistic Focus
  - Verify facts via multiple sources
  - Maintain objectivity and balance; journalistic ethics
  - Timely, relevant content
  - Consider legal implications of statements

### Reference Materials (Frontmatter‑Driven)
Detect via YAML frontmatter keys (and `ref *` aliases):
- `research` → research notes
- `sources` → source bibliography
- `timeline` → chronological timeline
- `style` → style guide
- `rules` → rules/guidelines
- `outline` → content outline

When present, the agent should incorporate these in planning, consistency checks, and citations.

### Writing Standards (from prompt)
- Professional, engaging tone appropriate to subject
- Smooth transitions; readability balanced with detail
- Audience‑appropriate assumptions
- Specific, actionable feedback on content and structure
- Always prioritize accuracy, clarity, reader engagement

### Inputs Consumed (Desktop → LangGraph)
- `_nonFictionType`: enum string (biography|autobiography|memoir|history|academic|journalism)
- `_frontmatter`: YAML map (may include research/sources/timeline/style/rules/outline)
- Loaded reference documents: research notes, bibliography, timeline, style guide, rules, outline
- Current manuscript body (for context and edits)
- User request/instructions

### Agent Mapping
- Non‑Fiction Manuscript Agent (`nonfiction_manuscript_agent`)
  - Drafting/revision with type‑specific constraints
  - Emits `ManuscriptEdit` structured output for edits
- Non‑Fiction Proofreading Agent (`nonfiction_proofreading_agent`)
  - Style/citation consistency pass; emits `ProofreadCorrections`
- Research Librarian Agent (`research_agent`)
  - Sources discovery, verification, and extraction; optional web search with permission
- Timeline/Chronology Checker (background)
  - Validates event ordering; flags conflicts against timeline doc
- Citation Builder (interactive/background)
  - Suggests inline citations/footnotes; formats bibliography

### Response Contracts
- Primary structured payloads:
  - `ManuscriptEdit` for precise insert/replace/delete edits
  - `ProofreadCorrections` for style/grammar/citation tweaks
  - Optional `NonFictionAnalysisReport` (alias of `AnalysisReport` with source requirements) for high‑level critiques
- Secondary human‑readable blocks (for editor UI):
  - Revision: "Original text:" → "Changed to:"
  - Insertion: "Insert after:" → "New text:"

### Citation & Fact‑Check Policy
- Prefer verifiable, citable sources; propose inline citations and bibliography entries
- Track source reliability; encourage multi‑source corroboration
- Indicate uncertainty and suggest further research where needed
- Respect privacy/ethics (memoir/biography/journalism) and academic standards

### Example System Banner
```text
You are an expert non‑fiction writing assistant specializing in {{nonfiction_type}} writing. Your role is to help create engaging, well‑researched, and factually accurate content. Prioritize factual accuracy, proper citations, and clear narrative flow.
```

### Example Developer Message Skeleton
```text
**Non‑Fiction Type**: {{nonfiction_type}}

**Core Responsibilities**
1) Factual accuracy  2) Research integration  3) Chronological consistency  4) Citation management  5) Narrative flow

{{#if type_guidance}}
**{{nonfiction_type}} Focus**
{{type_guidance}}
{{/if}}

{{#if references}}
**Available Reference Materials**
{{references_list}}
Use these for consistency and accuracy.
{{/if}}

**Writing Standards**
- Professional tone; smooth transitions; balance detail and readability
- Audience‑appropriate; specific actionable feedback
- Prioritize accuracy, clarity, engagement
```

### Configuration
- `NONFICTION_DEFAULT_TYPE=history` (example)
- `CITATION_STYLE=chicago` (options: chicago|apa|mla|ieee)
- `NONFICTION_ENABLE_WEB_RESEARCH=true` (permission‑gated)

### Integration Notes
- Non‑Fiction agents should emit structured outputs per schemas in `docs/WRITING_LANGGRAPH_SYSTEM_PLAN.md`
- When references are present in frontmatter, load them in the `file_context_loader` node
- For research needs, route to `research_agent` with permission gating; merge results with citations



