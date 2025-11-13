# Intent Classifier Migration Analysis

## Executive Summary

Comprehensive analysis of the backend `SimpleIntentService` for migration to `llm-orchestrator` with 100% functional parity.

---

## Backend Implementation Overview

### File Structure
- **Service**: `/opt/bastion/backend/services/simple_intent_service.py` (639 lines)
- **Model**: `/opt/bastion/backend/models/simple_intent_models.py` (74 lines)

### Core Dependencies

1. **ChatService** - LLM client access (OpenRouter)
2. **SettingsService** - Classification model configuration
3. **SimpleIntentResult** - Pydantic model for structured output

### Supported Agent Types (24 total)

**Conversational**:
- `chat_agent` - General conversation, observation queries
- `research_agent` - Information gathering, general topics

**Fiction Writing**:
- `fiction_editing_agent` - Create/edit fiction prose
- `story_analysis_agent` - Critique fiction manuscripts
- `content_analysis_agent` - Document comparison, analysis
- `proofreading_agent` - Grammar/spelling corrections

**Creative Development**:
- `outline_editing_agent` - Create/refine story outlines
- `rules_editing_agent` - World-building rules
- `character_development_agent` - Character profiles

**Content Generation**:
- `podcast_script_agent` - TTS-ready podcast scripts
- `substack_agent` - Articles and tweet-sized posts

**Management**:
- `org_inbox_agent` - Manage inbox.org TODOs
- `org_project_agent` - Create projects with metadata
- `website_crawler_agent` - Recursive site ingestion

**Data & Pipelines**:
- `data_formatting_agent` - Format data as tables/CSV/JSON
- `pipeline_agent` - AWS data pipelines (S3, Glue, Lambda)

**Specialized**:
- `image_generation_agent` - DALL-E image generation
- `wargaming_agent` - Military scenario analysis
- `entertainment_agent` - Movie/TV recommendations
- `sysml_agent` - SysML/UML diagrams
- `rss_agent` - RSS feed management
- `combined_proofread_and_analyze` - Combined operations

### Action Intent Types (6 total)

1. **observation** - View/check/confirm existing content
   - Triggers: "Do you see...", "Show me...", "What's in..."
   - Routes to: Editor agents (can read and respond)

2. **generation** - Create/write/draft NEW content
   - Triggers: "Write...", "Create...", "Draft...", "Generate..."
   - Routes to: Editor agents, content agents

3. **modification** - Change/edit/revise EXISTING content
   - Triggers: "Edit...", "Revise...", "Change...", "Improve..."
   - Routes to: Editor agents, formatting agents

4. **analysis** - Critique/feedback/assessment/comparison
   - Triggers: "Analyze...", "Critique...", "Compare...", "Summarize file X..."
   - Routes to: Analysis agents (story_analysis, content_analysis)

5. **query** - Seek external information/facts (NOT documents)
   - Triggers: "Tell me about...", "What is...", "Research..."
   - Routes to: research_agent, chat_agent

6. **management** - Organize/configure/manage system
   - Triggers: "Add TODO...", "Crawl website...", "Mark as done..."
   - Routes to: Management agents (org_inbox, website_crawler)

### Context Awareness

The classifier is highly context-aware and adjusts routing based on:

#### 1. **Active Editor Context**
Detects editor frontmatter `type` field and biases routing:

- `type: fiction` → prefer `fiction_editing_agent`
- `type: rules` → prefer `rules_editing_agent`
- `type: outline` → prefer `outline_editing_agent`
- `type: character` → prefer `character_development_agent`
- `type: podcast` → prefer `podcast_script_agent`
- `type: substack/blog` → prefer `substack_agent`
- `type: sysml` → prefer `sysml_agent`
- `type: style` → prefer `chat_agent` (general edits)

**Action Intent + Editor Logic**:
- `observation` + fiction editor → `fiction_editing_agent` (reads content)
- `generation` + fiction editor → `fiction_editing_agent` (creates prose)
- `modification` + fiction editor → `fiction_editing_agent` (edits prose)
- `analysis` + fiction editor → `story_analysis_agent` (critiques)
- `query` + ANY editor → `research_agent` OR `chat_agent` (semantic choice)

#### 2. **Active Pipeline Context**
- Detects `active_pipeline_id` in shared_memory
- Makes `pipeline_agent` available ONLY when on pipelines page
- Blocks `pipeline_agent` if not on pipeline page (security isolation)

#### 3. **Conversation Intelligence**
- Detects if `org_inbox_agent` has been active recently
- Biases toward org-mode agents for task-like statements

#### 4. **Recent Messages Count**
- Ongoing conversation (>1 message) gets context hint in prompt

### Routing Logic Hierarchy

1. **LLM Classification** - Primary decision maker (semantic understanding)
2. **Action Intent Override** - Post-processing based on action + editor context
3. **Pipeline Isolation** - Security check (block if not on pipeline page)
4. **Fallback** - Keyword matching if LLM/JSON parsing fails

### Key Features

#### 1. **Dynamic Agent Lists**
Prompt shows ONLY available agents based on page context:
- Pipeline agent shown ONLY when `active_pipeline_id` exists
- All other agents always shown

#### 2. **Semantic Routing**
Uses semantic understanding, NOT brittle pattern matching:
- "Good morning" → chat_agent (conversational)
- "Research Enron scandal" → research_agent (information gathering)
- "Compare our worldcom documents" → content_analysis_agent (doc comparison)

#### 3. **Document-Specific Detection**
Queries mentioning specific files/documents:
- "Summarize our file called X" → content_analysis_agent (analysis intent)
- "What's in document Y" → content_analysis_agent (analysis intent)
- "Tell me about the Enron scandal" → research_agent (general query)

#### 4. **Robust JSON Parsing**
- Handles markdown code fences
- Validates action_intent against allowed values
- Falls back to keyword matching if parsing fails

#### 5. **Model Configuration**
Uses `settings_service.get_classification_model()` - configured per user in Settings panel:
- Fast model for classification (e.g., Claude Haiku 4.5)
- Temperature 0.1 (deterministic)
- Max tokens 500 (lean responses)

### Response Structure

```python
SimpleIntentResult(
    target_agent="chat_agent",
    action_intent="query",
    permission_required=False,
    confidence=0.85,
    reasoning="Casual greeting - conversational interaction"
)
```

### Logging Pattern

```
🎯 SIMPLE CLASSIFICATION: Processing message: Good morning...
🎯 ACTION INTENT: query
✅ SIMPLE CLASSIFICATION: → chat_agent (confidence: 0.85)
```

With context overrides:
```
📝 EDITOR CONTEXT: type=fiction, action_intent=observation
👁️ OBSERVATION INTENT: fiction → fiction_editing_agent (editor agent can read and respond)
```

---

## Migration Requirements

### Must-Have Feature Parity

✅ **All 24 agent types** must be recognized  
✅ **All 6 action intent types** must be classified  
✅ **Editor context awareness** must work identically  
✅ **Pipeline isolation** must be enforced  
✅ **Conversation intelligence** must be respected  
✅ **Semantic routing** (no brittle pattern matching)  
✅ **Robust JSON parsing** with fallback  
✅ **Model configuration** from settings  
✅ **Identical prompt structure** for consistency  
✅ **Same logging patterns** for debugging  

### llm-orchestrator Adaptations

**Service Dependencies**:
- Backend: `ChatService` → llm-orchestrator: Direct LLM client
- Backend: `SettingsService` → llm-orchestrator: Config service or environment variable
- Backend: Pydantic models → llm-orchestrator: Port models

**Context Access**:
- Backend: `conversation_context` dict → llm-orchestrator: Extract from `ChatRequest` proto
- Backend: `shared_memory` → llm-orchestrator: Parse from proto fields
- Backend: `active_editor` → llm-orchestrator: Extract from `ChatRequest.active_editor`
- Backend: `active_pipeline_id` → llm-orchestrator: Extract from `ChatRequest.pipeline_context`

**Model Access**:
- Backend: `settings_service.get_classification_model()` → llm-orchestrator: Environment variable or config

### Testing Strategy

Test these query types to ensure parity:

1. **Conversational**: "Good morning", "How are you", "Thanks"  
   Expected: `chat_agent`

2. **Research**: "Research quantum computing", "Tell me about Napoleon"  
   Expected: `research_agent`

3. **Document Analysis**: "Compare our worldcom documents", "Summarize file X"  
   Expected: `content_analysis_agent`

4. **Editor Context (Fiction)**: "Write chapter 3" (with fiction editor)  
   Expected: `fiction_editing_agent`

5. **Editor Context (Observation)**: "Do you see our outline" (with fiction editor)  
   Expected: `fiction_editing_agent` (observation intent)

6. **Pipeline Context**: "Create pipeline from S3 to Redshift" (with active_pipeline_id)  
   Expected: `pipeline_agent`

7. **Pipeline Blocked**: "Create pipeline" (without active_pipeline_id)  
   Expected: `research_agent` (redirected)

8. **Org Management**: "Add TODO: Buy milk"  
   Expected: `org_inbox_agent`

9. **Org Reading**: "What's on my TODO list?"  
   Expected: `chat_agent`

10. **Entertainment**: "Tell me about Breaking Bad"  
    Expected: `entertainment_agent`

---

## Migration Plan

### Phase 1: Model Migration ✅
- Port `SimpleIntentResult` to llm-orchestrator models
- Ensure Pydantic validation works identically

### Phase 2: Service Creation ✅
- Create `IntentClassifier` class in llm-orchestrator
- Port ALL prompt building logic (100% identical)
- Port ALL parsing logic with same fallbacks
- Port ALL context awareness logic

### Phase 3: Integration ✅
- Update llm-orchestrator gRPC service to run classifier BEFORE routing
- Extract context from `ChatRequest` proto
- Pass classification result to router

### Phase 4: Testing ✅
- Run all 10 test query types
- Compare routing decisions with backend
- Verify logging matches backend patterns

### Phase 5: Documentation ✅
- Document migration
- Update architecture diagrams
- Note any differences (if any)

---

## Success Criteria

✅ All 24 agent types recognized  
✅ All 6 action intents classified  
✅ Editor context biasing works  
✅ Pipeline isolation enforced  
✅ Conversation intelligence respected  
✅ Same model configuration  
✅ Same logging patterns  
✅ Same routing decisions as backend  
✅ Fallback logic identical  
✅ Test queries pass  

**When all criteria met: Backend intent classifier can be deprecated in favor of llm-orchestrator version.**

---

## Files to Create/Modify

### New Files:
- `/opt/bastion/llm-orchestrator/orchestrator/models/intent_models.py`
- `/opt/bastion/llm-orchestrator/orchestrator/services/intent_classifier.py`

### Modified Files:
- `/opt/bastion/llm-orchestrator/orchestrator/grpc_service.py` - Add intent classification before routing
- `/opt/bastion/llm-orchestrator/config.py` - Add classification model config

### Documentation:
- `/opt/bastion/docs/INTENT_CLASSIFIER_MIGRATION.md` - Migration guide
- `/opt/bastion/docs/LANGGRAPH_AGENT_MIGRATION_GUIDE.md` - Update with intent classifier section

---

## Notes

- The backend version is mature and battle-tested (639 lines)
- Migration must be **1:1 functional parity** - no shortcuts
- All 24 agents must work (even if not all migrated yet)
- Classification model must match backend configuration
- Logging must be identical for debugging continuity
- Fallback logic is critical for robustness

**This is infrastructure migration - precision matters!**

