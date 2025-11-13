## LangGraph Writing System — Fiction Workflow Plan

BULLY! By George, this document outlines a disciplined, LangGraph-driven writing workflow for fiction/non‑fiction/screenplays with Markdown frontmatter, tab‑aware agents, and chainable operations. We enforce the author’s Style Guide with a big stick: persona overrides are ignored when correctness to the style canon is required.

### Overview
- Markdown editor with tabs for files: `fiction`, `outline`, `rules`, `style`, `characters`.
- Frontmatter references wire documents together: the manuscript references an `outline`, which in turn references `rules`, `style`, and `characters`.
- Each tab exposes specialized agents/chains aligned to the file `type`.
- Proofreading uses the Style Guide only; Story Analysis ignores all references and evaluates the manuscript on its own merits.
- All agents produce Pydantic‑validated structured outputs; actions that modify files go through permission/HITL gates and tracked edits.

### File Types and Frontmatter Schema
- `type: fiction`
  - `outline: path/to/outline.md`
  - Optional: `draft_stage: planning|draft|revise|polish`, `language`, `genre`, `audience`
- `type: outline`
  - `rules: path/to/rules.md`
  - `style: path/to/style.md`
  - `characters: [path/to/char_main_alex.md, path/to/supporting.md]`
  - Optional: `structure: three_act|four_act|tv_serial`, `pov: first|third-limited|omniscient`
- `type: rules`
  - Optional: `canon_version`, `continuity_refs: [urls]`
- `type: style`
  - Optional: `voice_tags: [wry, lyrical, terse]`, `tense: past|present`, `person: first|third` 
- `type: characters`
  - Main character sheets are individual files of `type: characters` (one per file); supporting cast may group in one file
  - Optional: `arcs: [redemption, tragedy]`, `relationships: [char_id]`

Frontmatter is validated on load; references are resolved relative to the manuscript’s directory. Missing references fail gracefully with actionable errors.

### Agents (Tab‑Aware)
- Outline Agent (outline tab)
  - Purpose: Generate or refine a hierarchical outline from premise, `rules`, `style`, and `characters`.
  - Output: `OutlineModel` with acts/chapters/scenes, beat summaries, POV, goals/obstacles, and dependencies.
- Rules Agent (rules tab)
  - Purpose: Create/expand canon rules; detect contradictions; suggest clarifications.
  - Output: `RulesModel` with principles, constraints, allowed/disallowed elements, and references.
- Style Guide Agent (style tab)
  - Purpose: Produce a style guide: narrative voice, tense, person, diction, pacing, dialogue conventions; includes sample text.
  - Output: `StyleGuideModel` with prescriptive rules and examples.
- Characters Agent (characters tab)
  - Purpose: Create character sheets (appearance, voice, backstory, motivations, arc beats) or supporting‑cast batches.
  - Output: `CharacterModel` (single) or `CharacterListModel` (batch) including identifiers for cross‑refs.
- Manuscript Agent (fiction tab)
  - Purpose: Draft or revise scene/manuscript text guided strictly by Outline + Style + Rules + Characters.
  - Output: `ManuscriptEdit` with targeted insertion/rewrite ops, citations to outline beats, and adherence checks.
- Proofreading Agent (fiction tab only)
  - Purpose: Corrections aligned to Style Guide only; no plot changes.
  - Output: `ProofreadCorrections` (diff, suggestions, rationale by rule id).
- Story Analysis Agent (fiction tab only)
  - Purpose: Ignore all references; assess plot coherence, pacing, theme, character arcs, and tension curve.
  - Output: `AnalysisReport` with findings, risks, recommendations; no style rewrites.
- Consistency Checker (background)
  - Purpose: Validate continuity across files (names, dates, rules); flag drift or contradictions.
  - Output: `ContinuityFindings` with severities and fix suggestions.
- Reference Propagation Agent (background)
  - Purpose: When upstream docs change (e.g., character sheet), propose updates to dependent outline/manuscript.
  - Output: `PropagationPlan` with safe edit proposals.

### Persona Override Policy
- For `outline`, `rules`, `style`, `characters`, and `fiction` drafting: the system explicitly disables any user “AI persona” and enforces the Style Guide and Rules as the sole authorities.
- Implementation: System prompt includes a high‑priority directive; orchestrator sets `persona=None` and injects style/rules context. Responses that deviate are rejected and retried.

### LangGraph Design
- StateGraph nodes
  - `file_context_loader`: Parse frontmatter; resolve referenced docs; validate schemas.
  - `intent_classifier`: Determine desired operation (generate outline, add scene, proofread, analyze, etc.).
  - `router`: Route to appropriate tab‑aware agent.
  - `agent_nodes`: `outline_agent`, `rules_agent`, `style_agent`, `characters_agent`, `manuscript_agent`, `proofreading_agent`, `story_analysis_agent`.
  - `permission_gate`: HITL approval before file edits are applied.
  - `apply_edits`: Write Markdown edits with minimal diffs and metadata footers.
  - `final_response`: Summarize actions, surface next steps.
- Persistence: PostgreSQL checkpointer; conversation state stored; file change journal entries linked to commits.
- Compilation: Graph compiled with the checkpointer; background tasks offloaded to Celery.

### Chains and Workflows
- Outline Chain
  - Inputs: high‑level premise, `rules`, `style`, `characters` → Outline Agent → `OutlineModel` → saved to outline file.
- Manuscript Drafting Chain
  - Inputs: `outline` (+ `rules`, `style`, `characters`) → Manuscript Agent → `ManuscriptEdit` → Proofreading Agent (Style only) → optional acceptance.
- Character‑First Chain
  - Characters Agent → Rules Agent (consistency pass) → Outline Agent (align plot beats to arcs) → Manuscript Agent.
- Analysis Chain (critique mode)
  - Manuscript text only → Story Analysis Agent → `AnalysisReport` with prioritized issues.

### Structured Output Models (examples)
- `OutlineModel`:
  - `task_status`, `title`, `acts: [ { name, chapters: [ { name, scenes: [ { id, title, pov, summary, goals, obstacles, outcome, word_target } ] } ] } ]`, `assumptions`, `sources`, `confidence`.
- `RulesModel`:
  - `task_status`, `principles: [str]`, `constraints: [str]`, `forbidden: [str]`, `edge_cases: [str]`, `examples`, `confidence`.
- `StyleGuideModel`:
  - `task_status`, `voice`, `tense`, `person`, `diction_tags`, `syntax_preferences`, `dialogue_conventions`, `sample_text`, `do: [str]`, `dont: [str]`.
- `CharacterModel` / `CharacterListModel`:
  - `name`, `aliases`, `appearance`, `voice_markers`, `backstory`, `desires`, `fears`, `arc_beats`, `relationships`.
- `ManuscriptEdit`:
  - `task_status`, `edits: [ { target: heading|scene_id|range, op: insert|replace|delete, text, reason, trace: { outline_ref, rule_ids, style_rules } } ]`.
- `ProofreadCorrections`:
  - `task_status`, `diff`, `suggestions: [ { rule_id, before, after, rationale } ]`, `confidence`.
- `AnalysisReport`:
  - `task_status`, `findings: { plot, pacing, theme, character, worldbuilding }`, `risks: [ { severity, description } ]`, `recommendations: [str]`.

### Tools
- Markdown File Tools
  - Read/write with minimal diff; smart insertion by heading/scene id; YAML frontmatter parser/validator; link resolver.
- Scene/Section Locator
  - Index headings and scene anchors; map edits deterministically.
- Change Journal
  - Record edit metadata: who/when/why; enable undo.

See also:
- `docs/FICTION_DESKTOP_PROMPTS.md` (or `WRITING_DESKTOP_PROMPTS.md` if present) for Fiction prompt compatibility and mode routing.
- `docs/NONFICTION_DESKTOP_PROMPTS.md` for Non‑Fiction prompt compatibility and type‑specific guidance.
 - `docs/CHARACTER_DESKTOP_PROMPTS.md` for Character Development prompt integration (development vs granular revisions).
 - `docs/OUTLINE_DESKTOP_PROMPTS.md` for Outline creation/refinement prompt and structural rules.

### Background Agents & Tasks
- Consistency Checker: scheduled passes to detect name drift, timeline contradictions, rule violations.
- Reference Propagation: watches upstream files; proposes dependent updates.
- Autosave/Versioning: periodic save points; snapshot on major operations.
- Markdown Lint & Health: validates frontmatter, broken links, missing references.

### UI & Tab Awareness
- When tab == `outline`: expose Outline Agent chain; display referenced `rules`, `style`, `characters` context.
- When tab == `rules`: expose Rules Agent; show detected contradictions and coverage.
- When tab == `style`: expose Style Agent; provide sample‑guided tuning tools.
- When tab == `characters`: expose Characters Agent; create single or batch entries.
- When tab == `fiction`: expose Manuscript, Proofreading (Style‑only), and Story Analysis (ignore references).

### Env Considerations
- `WRITING_PERSONA_ENFORCEMENT=strict` (forces persona override and Style/Rules precedence)
- `WRITING_MAX_EDIT_SIZE` (characters per edit op)
- `WRITING_AUTOSAVE_INTERVAL_SECONDS`
- `WRITING_ALLOWED_FILE_TYPES=fiction,outline,rules,style,characters`
- `WRITING_ROOT_DIR=/workspace/docs` (example)

### Best‑Practice Prompts
- System prompt always precedes with: “Persona disabled. Adhere strictly to Style Guide and Rules. Do not invent canon beyond Rules without proposing updates.”
- Agents must cite which rules/style tokens governed each decision in their structured outputs.

### Compliance & Safety
- Respect user content boundaries; never leak other projects.
- Proofreading changes are suggestions unless user approves.
- Story Analysis never rewrites; it only reports.

By thunder, this plan gives us a well‑organized cavalry charge: clear roles per agent, strict style enforcement, typed outputs, and tab‑aware workflows—all marching under the big stick of LangGraph discipline.

### Structured Output JSON Schemas

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "OutlineModel",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "title": { "type": "string" },
    "acts": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "name": { "type": "string" },
          "chapters": {
            "type": "array",
            "items": {
              "type": "object",
              "additionalProperties": false,
              "properties": {
                "name": { "type": "string" },
                "scenes": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {
                      "id": { "type": "string" },
                      "title": { "type": "string" },
                      "pov": { "type": "string", "enum": ["first", "third-limited", "omniscient"] },
                      "summary": { "type": "string" },
                      "goals": { "type": "array", "items": { "type": "string" } },
                      "obstacles": { "type": "array", "items": { "type": "string" } },
                      "outcome": { "type": "string" },
                      "word_target": { "type": "integer", "minimum": 0 }
                    },
                    "required": ["id", "title", "summary"]
                  }
                }
              },
              "required": ["name", "scenes"]
            }
          }
        },
        "required": ["name", "chapters"]
      }
    },
    "assumptions": { "type": "array", "items": { "type": "string" } },
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "type": { "type": "string" },
          "url": { "type": "string" },
          "id": { "type": "string" },
          "timestamp": { "type": "string", "format": "date-time" }
        }
      }
    },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
  },
  "required": ["task_status", "title", "acts"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "RulesModel",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "principles": { "type": "array", "items": { "type": "string" } },
    "constraints": { "type": "array", "items": { "type": "string" } },
    "forbidden": { "type": "array", "items": { "type": "string" } },
    "edge_cases": { "type": "array", "items": { "type": "string" } },
    "examples": { "type": "object", "additionalProperties": true },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
  },
  "required": ["task_status", "principles"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "StyleGuideModel",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "voice": { "type": "string" },
    "tense": { "type": "string", "enum": ["past", "present"] },
    "person": { "type": "string", "enum": ["first", "third"] },
    "diction_tags": { "type": "array", "items": { "type": "string" } },
    "syntax_preferences": { "type": "array", "items": { "type": "string" } },
    "dialogue_conventions": { "type": "array", "items": { "type": "string" } },
    "sample_text": { "type": "string" },
    "do": { "type": "array", "items": { "type": "string" } },
    "dont": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["task_status", "voice", "tense", "person", "sample_text"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "CharacterModel",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "id": { "type": "string" },
    "name": { "type": "string" },
    "aliases": { "type": "array", "items": { "type": "string" } },
    "appearance": { "type": "string" },
    "voice_markers": { "type": "array", "items": { "type": "string" } },
    "backstory": { "type": "string" },
    "desires": { "type": "array", "items": { "type": "string" } },
    "fears": { "type": "array", "items": { "type": "string" } },
    "arc_beats": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "beat": { "type": "string" },
          "description": { "type": "string" },
          "when": { "type": "string" }
        },
        "required": ["beat"]
      }
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "character_id": { "type": "string" },
          "relation": { "type": "string" }
        },
        "required": ["character_id", "relation"]
      }
    }
  },
  "required": ["name"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "ManuscriptEdit",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "edits": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "target": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "type": { "type": "string", "enum": ["heading", "scene_id", "range"] },
              "value": { "type": "string" },
              "range": {
                "type": "object",
                "additionalProperties": false,
                "properties": {
                  "start_line": { "type": "integer", "minimum": 1 },
                  "end_line": { "type": "integer", "minimum": 1 }
                },
                "required": ["start_line", "end_line"]
              }
            },
            "required": ["type"]
          },
          "op": { "type": "string", "enum": ["insert", "replace", "delete"] },
          "text": { "type": "string" },
          "reason": { "type": "string" },
          "trace": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "outline_ref": { "type": "string" },
              "rule_ids": { "type": "array", "items": { "type": "string" } },
              "style_rules": { "type": "array", "items": { "type": "string" } }
            }
          }
        },
        "required": ["target", "op"]
      }
    }
  },
  "required": ["task_status", "edits"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "ProofreadCorrections",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "diff": { "type": "string" },
    "suggestions": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "rule_id": { "type": "string" },
          "before": { "type": "string" },
          "after": { "type": "string" },
          "rationale": { "type": "string" }
        },
        "required": ["rule_id", "after"]
      }
    },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1 }
  },
  "required": ["task_status", "suggestions"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "AnalysisReport",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "findings": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "plot": { "type": "string" },
        "pacing": { "type": "string" },
        "theme": { "type": "string" },
        "character": { "type": "string" },
        "worldbuilding": { "type": "string" }
      }
    },
    "risks": {
      "type": "array",
      "items": { "type": "object", "properties": { "severity": { "type": "string", "enum": ["low", "medium", "high"] }, "description": { "type": "string" } }, "required": ["severity", "description"], "additionalProperties": false }
    },
    "recommendations": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["task_status", "findings"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "ContinuityFindings",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "type": { "type": "string", "enum": ["name", "timeline", "rule_violation"] },
          "location": { "type": "string" },
          "description": { "type": "string" },
          "severity": { "type": "string", "enum": ["low", "medium", "high"] }
        },
        "required": ["type", "description"]
      }
    },
    "suggestions": { "type": "array", "items": { "type": "string" } },
    "timestamp": { "type": "string", "format": "date-time" }
  },
  "required": ["task_status", "findings"]
}
```

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "PropagationPlan",
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "task_status": { "type": "string", "enum": ["complete", "incomplete", "error"] },
    "proposals": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "file": { "type": "string" },
          "target": { "type": "string" },
          "op": { "type": "string", "enum": ["insert", "replace", "delete"] },
          "text": { "type": "string" },
          "reason": { "type": "string" },
          "risk": { "type": "string", "enum": ["low", "medium", "high"] }
        },
        "required": ["file", "target", "op"]
      }
    }
  },
  "required": ["task_status", "proposals"]
}
```

### Frontmatter Templates (YAML)

```yaml
# fiction manuscript
---
type: fiction
title: The Iron Road
outline: ./outline.md
draft_stage: draft
language: en
genre: fantasy
audience: adult
---
```

```yaml
# outline
---
type: outline
title: The Iron Road — Outline
rules: ./rules.md
style: ./style.md
characters:
  - ./characters/alex.md
  - ./characters/supporting.md
structure: three_act
pov: third-limited
---
```

```yaml
# rules
---
type: rules
title: The Iron Road — World Rules
canon_version: v1.0
continuity_refs:
  - https://example.com/canon
---
```

```yaml
# style guide
---
type: style
title: The Iron Road — Style Guide
voice_tags: [lyrical, grounded, sardonic]
tense: past
person: third
sample_text: >-
  The rails sang under the moon, a thin silver line threading the mountains.
---
```

```yaml
# main character sheet
---
type: characters
title: Alex Kestrel
name: Alex Kestrel
aliases: [Kestrel]
voice_markers: [dry wit, concise, observant]
appearance: Tall, ash‑blond, scar on left brow
backstory: >-
  Raised by railway stewards; learned the codes of the line and the cost of silence.
desires: [belonging, truth]
fears: [abandonment, becoming like their father]
arc_beats:
  - beat: refusal of the call
    description: Avoids joining the strike, citing neutrality
relationships:
  - character_id: maya-voss
    relation: mentor
---
```

```yaml
# supporting characters (batch)
---
type: characters
title: Supporting Cast
characters:
  - name: Maya Voss
    voice_markers: [measured, empathetic]
    role: mentor
  - name: Bram Holt
    voice_markers: [gruff, impatient]
    role: foreman
---
```


