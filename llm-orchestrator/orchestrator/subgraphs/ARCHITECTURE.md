# Fiction Agent Subgraph Architecture

## Overview

The fiction editing system has been refactored into modular subgraphs to meet the 500-line workspace rule and improve reusability.

## Module Structure

```
llm-orchestrator/orchestrator/
├── agents/
│   └── fiction_editing_agent.py          (~4,200 lines) - Main orchestration
│                                          Note: Old node methods still present but unused
│                                          TODO: Remove old methods in follow-up cleanup
│                                          Proofreading: uses proofreading_subgraph (no standalone agent)
├── subgraphs/
│   ├── fiction_context_subgraph.py       (~517 lines) - Reusable context prep
│   └── fiction_validation_subgraph.py    (~636 lines) - Continuity & validation
├── services/
│   └── fiction_continuity_tracker.py     (unchanged)
└── utils/
    ├── editor_operation_resolver.py      (unchanged - already centralized)
    └── frontmatter_utils.py              (unchanged - already centralized)
```

## Subgraph Details

### Context Preparation Subgraph

**File:** `orchestrator/subgraphs/fiction_context_subgraph.py`

**Purpose:** Reusable context preparation for fiction-aware agents

**Nodes:**
- `prepare_context` - Extract active editor, validate document
- `detect_chapter_mentions` - Identify explicit chapter references in query
- `analyze_scope` - Determine working scope (single/multi-chapter)
- `load_references` - Load outline, style, rules, characters from frontmatter
- `assess_references` - Validate loaded references are usable

**State Compatibility:**
- Uses `Dict[str, Any]` for compatibility with main agent state
- Reads from and writes to main agent's `FictionEditingState`
- Sets both `manuscript` and `manuscript_content` for compatibility

**Used By:**
- `fiction_editing_agent` - Full context preparation (proofreading path uses proofreading_subgraph)

### Validation Subgraph

**File:** `orchestrator/subgraphs/fiction_validation_subgraph.py`

**Purpose:** Continuity tracking and validation for fiction agents

**Nodes:**
- `detect_outline_changes` - Compare manuscript with outline for discrepancies
- `load_continuity` - Load or create continuity state from .continuity.json
- `validate_consistency` - Check edit plan for internal consistency
- `validate_continuity` - Validate edits against continuity state
- `update_continuity` - Extract and merge continuity from new content

**State Compatibility:**
- Uses `Dict[str, Any]` for compatibility with main agent state
- Reads from and writes to main agent's `FictionEditingState`
- Supports both `chapter_number` and `current_chapter_number` for flexibility

**Used By:**
- `fiction_editing_agent` - Full validation pipeline

## Main Agent Workflow

### Fiction Editing Agent

**Entry Point:** `context_preparation` (subgraph)

**Flow:**
1. **Context Preparation** (subgraph)
   - Prepare context, detect chapters, analyze scope, load references, assess quality

2. **Fiction Type Validation**
   - Gate: Ensure document is fiction type

3. **Mode Detection**
   - Detect generation mode and creative intent

4. **Routing**
   - If no references → `generate_simple_edit` (fast path)
   - If references → `detect_request_type` → `check_multi_chapter`

5. **Generation**
   - Single chapter: `generate_edit_plan` → `validation` (subgraph)
   - Multi-chapter: Loop through `prepare_chapter_context` → `generate_edit_plan` → `validation` (subgraph)

6. **Resolution**
   - `resolve_operations` → `accumulate_chapter` (if multi) → `format_response`

### Proofreading Agent

**Entry Point:** `context_preparation` (subgraph)

**Flow:**
1. **Context Preparation** (subgraph)
   - Reuses shared context preparation logic

2. **Mode Inference**
   - Infer proofreading mode (clarity/compliance/accuracy)

3. **Style Guide Loading**
   - Load referenced style guide

4. **Scope Determination**
   - Determine proofreading scope (chapter/paragraph/full doc based on word count)

5. **Proofreading & Operations**
   - `proofread_content` → `generate_operations` → `format_response`

## State Management

### State Compatibility

Subgraphs use `Dict[str, Any]` for state to ensure compatibility with main agent's `TypedDict` state. LangGraph handles the conversion automatically.

### Key State Fields

**Context Subgraph Outputs:**
- `manuscript` / `manuscript_content` - Full manuscript text
- `frontmatter` - Document frontmatter
- `cursor_offset` - Cursor position
- `chapter_ranges` - List of chapter ranges
- `current_chapter_text` - Current chapter content
- `current_chapter_number` - Current chapter number
- `outline_body`, `style_body`, `rules_body`, `characters_bodies` - Loaded references
- `has_references` - Boolean flag for routing

**Validation Subgraph Inputs/Outputs:**
- `structured_edit` - Edit plan from generation
- `continuity_state` - Continuity tracking state
- `continuity_violations` - Detected violations
- `outline_sync_analysis` - Outline comparison results
- `updated_continuity` - Updated continuity state after chapter

## Benefits

### Code Quality
- **Modularity:** Clear separation of concerns
- **Reusability:** Context subgraph shared across agents
- **Maintainability:** Easier to understand and modify

### Performance
- **Short-circuit optimization:** Simple requests bypass unused subgraphs
- **Parallel potential:** Subgraphs could run in parallel in future

### Architecture
- **Meets workspace rules:** Subgraphs are close to 500-line limit
- **Extensible:** Easy to add new fiction-aware agents using shared subgraphs

## Future Improvements

1. **Remove old node methods** from `fiction_editing_agent.py` (currently ~4,200 lines, target ~700 lines)
2. **Split validation subgraph** if it exceeds 500 lines (currently 636 lines)
3. **Add parallel execution** for independent subgraph operations
4. **Extract more reusable components** (e.g., chapter detection utilities)

## Notes

- Old node methods in `fiction_editing_agent.py` are kept for safety but not used in workflow
- Subgraphs are designed to be stateless and reusable
- State transformation is handled automatically by LangGraph's type system

