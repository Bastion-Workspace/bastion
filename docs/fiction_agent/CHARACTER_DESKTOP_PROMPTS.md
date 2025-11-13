## Character Development Desktop Prompts — Compatibility Plan for LangGraph Writing System

BULLY! By George, this document maps the desktop Character Development prompt builder to our LangGraph writing system. It supports comprehensive character creation and surgical revisions, respects universe rules and style, and emits structured outputs for precise updates.

### Purpose
- Align desktop character chains/prompts with LangGraph agents/nodes
- Support two request modes: Development (full profile) and Granular Revision (targeted edits)
- Enforce Style/Rules precedence and universe continuity

### File Types / Focus Areas
- MajorCharacter (main character profiles)
  - Deep backstory and motivation, voice/dialogue patterns, internal monologue, growth arc, personality traits, relationships
- SupportingCast (batch or single minor characters)
  - Concise sketches, key traits/quirks, dialogue notes, role relative to majors, potential future development, consistency across appearances
- Relationships (relationship sheets)
  - Dynamics and tensions, interaction patterns, conflict sources/resolution, evolution over series, power dynamics

### Modes and Detection
- Development Request (comprehensive profile)
  - Trigger keywords (examples): ["create character", "develop character", "new character", "full character", "character profile", "character sheet", "build character"]
  - Output: Full character sheet text + structured `CharacterModel`
- Granular Revision Request (targeted changes)
  - Trigger keywords (examples): ["add", "modify", "change", "update", "adjust", "fix", "correct", "revise", "expand this", "enhance", "refine", "tweak", "small change", "specific", "particular"]
  - Output: Revision blocks for UI + structured `ManuscriptEdit` targeting the character file

Environment configuration for keyword lists:
- `CHAR_DEV_KEYWORDS_DEVELOP=create character,develop character,character profile,character sheet,build character`
- `CHAR_DEV_KEYWORDS_GRANULAR=add,modify,change,update,adjust,fix,correct,revise,enhance,refine,tweak,specific,particular`

### Inputs Consumed (Desktop → LangGraph)
- `fileType`: MajorCharacter | SupportingCast | Relationships
- `request`: user instruction string
- `_currentContext`: current character file content (frontmatter stripped for analysis)
- `_references`: list of reference objects with `Type` and `Content` (Style, Rules, Outline, Character, Relationship)
- `_rules`: raw universe rules; `_parsedRules`: optional parsed structures (Characters, Organizations, Locations, CriticalFacts)

### Agent Mapping
- `characters_agent` (interactive)
  - Development mode: generate full character content + `CharacterModel`
  - Granular mode: produce `ManuscriptEdit` edits and minimal commentary
- Background consistency passes
  - Continuity checker scans for name/timeline/rule conflicts; proposes `PropagationPlan` for dependent docs

### Response Contracts
- For Development:
  - Primary: `CharacterModel` (see schema in `WRITING_LANGGRAPH_SYSTEM_PLAN.md`)
  - Secondary: Markdown character sheet adhering to preferred formatting
- For Granular Revision:
  - Primary: `ManuscriptEdit` with targeted insert/replace/delete ops against the character file
  - Secondary: Human blocks for UI review
    - Revision block:
      - "Original text:" → exact snippet
      - "Changed to:" → revised snippet
    - Insertion block:
      - "Insert after:" → exact anchor
      - "New text:" → insertion

Precision requirements for revision blocks (must be enforced):
- Exact, complete, verbatim matches from current content (including whitespace/formatting)
- Include sufficient context (10–20 words) for unique identification
- Preserve structure (bullets, headings) and unchanged sections

### Reference Integration
- Style Guide: governs voice, dialogue style markers, formatting conventions
- Rules: universe constraints (magic/tech levels, naming, affiliations, timeline/events)
- Outline: relationship to planned beats (avoid copying; use for consistency)
- Character/Relationship refs: cross‑file harmony and continuity

Universe compliance checklist:
- Validate abilities against power systems; backgrounds against timeline/events
- Check relationships for conflicts with existing network/hierarchies
- Maintain naming conventions and cultural norms

### Preferred Formatting (summarized)
- Always include YAML frontmatter with fields like `title`, `type: character`, and references (`style`, `rules`, etc.)
- Use `##` for major sections; bullets for lists; bold labels for fields
- Major characters: include Basic Info, Personality (traits/strengths/flaws), Dialogue Patterns, Internal Monologue, Relationships, Character Arc
- Supporting cast: concise role, traits, speech notes, relationship to MC, notes
- Relationships: type, dynamics, conflict sources, interaction patterns, evolution

### Example System Banner
```text
You are a Character Development Assistant specializing in creating compelling, consistent characters for fiction writing. You excel at both creating new character content and revising existing character profiles.
```

### Example Developer Message Skeleton
```text
{{#if development_mode}}
=== CHARACTER DEVELOPMENT REQUEST ===
Focus on full profile depth: background/psychology, traits/flaws, voice/dialogue patterns, relationships, arc, proper sheet formatting.
{{/if}}

{{#if granular_mode}}
=== GRANULAR REVISION REQUEST ===
Make ONLY requested modifications; preserve existing content; surgical precision; maintain established traits/relationships.
{{/if}}

CHARACTER DEVELOPMENT GUIDELINES
- Realistic, flawed characters with clear motivations
- Distinctive voice/speech patterns; series‑level consistency
- Balance strengths with weaknesses; define growth potential

PREFERRED CHARACTER SHEET FORMATTING
- Follow YAML frontmatter + section structure per file type

=== REVISION FORMAT ===
Use exact "Original text:" → "Changed to:" or "Insert after:" → "New text:" blocks.
```

### Configuration
- `CHAR_DEV_KEYWORDS_DEVELOP`, `CHAR_DEV_KEYWORDS_GRANULAR`
- `WRITING_PERSONA_ENFORCEMENT=strict` (persona disabled; Style/Rules precedence)

### Integration Notes
- Use schemas in `docs/WRITING_LANGGRAPH_SYSTEM_PLAN.md` (`CharacterModel`, `ManuscriptEdit`)
- UI should render revision blocks with Apply actions bound to `ManuscriptEdit`
- After edits, run background continuity checks and propose `PropagationPlan` for impacted docs



