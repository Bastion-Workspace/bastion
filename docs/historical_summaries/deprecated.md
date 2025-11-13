# Deprecation Candidates Audit List
## Roosevelt's "Trust-Busting" Code Cleanup Campaign

**BULLY!** This document lists all potential deprecation candidates identified during the comprehensive codebase analysis. Each item requires a decision: **Keep**, **Deprecate**, or **Refactor**.

---

## How to Use This List

1. **Review each item** systematically, category by category
2. **Check dependencies** by searching the codebase for imports/references
3. **Make a decision** by checking the appropriate box
4. **Document reasoning** in the "Notes" section for each item
5. **Execute cleanup** after review is complete

**Decision Key:**
- **Keep**: Feature is actively used and provides value
- **Deprecate**: Remove immediately, no dependencies or minimal impact
- **Refactor**: Consolidate with another feature or modernize

---

## Category 1: Infrastructure Services (Commented/External)

### 1.1 Qdrant Vector Database (Commented)
- **File**: `docker-compose.yml` (lines 3-11)
- **Status**: Commented out, using external NFS mount at `192.168.80.128:6333`
- **Why deprecated**: Breaks portability, external dependency not managed
- **Impact**: Core feature for vector search
- **Replacement**: Uncomment and containerize OR make optional with fallback
- **Decision**: [X] Keep External [ ] Containerize [ ] Make Optional

**Notes:**
**DECISION: Keep External (Kubernetes)** - ✅ COMPLETED
- Removed commented Qdrant service definition from docker-compose.yml
- Removed commented volume definitions
- Updated README.md to document external K8s infrastructure
- Updated env.example with required QDRANT_URL configuration
- External K8s deployment is the production architecture


### 1.2 Neo4j Knowledge Graph (Commented)
- **File**: `docker-compose.yml` (lines 14-27)
- **Status**: Commented out, using external instance at `192.168.80.132:7687`
- **Why deprecated**: Breaks portability, external dependency not managed
- **Impact**: Knowledge graph features
- **Replacement**: Containerize OR make optional with graceful degradation
- **Decision**: [X] Keep External [ ] Containerize [ ] Make Optional

**Notes:**
**DECISION: Keep External (Kubernetes)** - ✅ COMPLETED
- Removed commented Neo4j service definition from docker-compose.yml
- Removed commented volume definitions
- Removed commented dependency conditions in backend service
- Updated README.md to document external K8s infrastructure
- Updated env.example with required NEO4J_* configuration variables
- External K8s deployment is the production architecture


---

## Category 2: Duplicate/Overlapping Agents

### 2.1 Site Crawl Agent vs Website Crawler Agent
- **Files**: 
  - `backend/services/langgraph_agents/site_crawl_agent.py` (query-driven research)
  - `backend/services/langgraph_agents/website_crawler_agent.py` (storage-driven ingestion)
- **Why deprecated**: Potential functional overlap, unclear distinction
- **Impact**: Both integrated into orchestrator
- **Replacement**: Consolidate into single agent with dual modes OR clarify distinct use cases
- **Decision**: [ ] Keep Both [ ] Merge [ ] Deprecate One (specify): _______

**Notes:**


### 2.2 RSS Agent vs RSS Background Agent
- **Files**:
  - `backend/services/langgraph_agents/rss_agent.py` (interactive)
  - `backend/services/langgraph_agents/rss_background_agent.py` (scheduled)
- **Why deprecated**: Similar functionality, unclear separation of concerns
- **Impact**: RSS feed processing
- **Replacement**: Single agent with execution mode parameter
- **Decision**: [ ] Keep Both [ ] Merge [ ] Deprecate Background Agent

**Notes:**


---

## Category 3: Intent Classification System Duplication

**AUDIT COMPLETED:** See `intent-classification-audit.md` for complete analysis with evidence.

### 3.1 Simple Intent Service ✅ ACTIVE
- **File**: `backend/services/simple_intent_service.py`
- **Status**: ACTIVELY USED by production orchestrator
- **Impact**: Core system - used by SimpleIntentAgent in LangGraph workflow
- **Decision**: [X] Keep [ ] Deprecate [ ] Merge

**Notes:**
**✅ KEEP - ACTIVE PRODUCTION SYSTEM**
- Used by SimpleIntentAgent (line 11)
- Instantiated in official orchestrator (line 55)
- Executed via orchestrator_nodes.py::intent_classifier_node()
- Part of active LangGraph workflow
- This is THE system that's actually running


### 3.2 Intent Classification Service (Legacy) ❌ DEAD
- **File**: `backend/services/intent_classification_service.py`
- **Status**: Initialized in service_container.py but NEVER USED
- **Impact**: ZERO - no code references it except initialization
- **Replacement**: Already replaced by SimpleIntentService
- **Decision**: [ ] Keep [X] Deprecate [ ] Already Deprecated

**Notes:**
**❌ DEPRECATE - DEAD CODE**
- Only imported in service_container.py for initialization (line 15)
- Initialized but never called anywhere (waste of resources)
- No orchestrator usage, no agent usage, no API usage
- Safe to remove immediately - zero impact


### 3.3 Capability-Based Intent Service ❌ DEAD
- **File**: `backend/services/capability_based_intent_service.py`
- **Status**: Only referenced in documentation files
- **Impact**: ZERO - never used in production
- **Replacement**: N/A - was never integrated
- **Decision**: [ ] Keep [X] Deprecate [ ] Merge with Unified

**Notes:**
**❌ DEPRECATE - ONLY IN DOCUMENTATION**
- Only imported in intent_migration_guide.py (documentation)
- Only imported in legacy_intent_deprecation.py (unused wrapper)
- Never used by orchestrator, agents, or APIs
- Safe to remove immediately


### 3.4 Legacy Intent Deprecation Wrapper ❌ DEAD
- **File**: `backend/services/legacy_intent_deprecation.py`
- **Status**: Never imported anywhere in codebase
- **Impact**: ZERO - compatibility layer that was never used
- **Replacement**: N/A - never needed
- **Decision**: [ ] Keep Temporarily [X] Remove After Migration [ ] Keep Permanently

**Notes:**
**❌ DEPRECATE - NEVER IMPORTED**
- Search for imports: ZERO matches
- Compatibility wrapper that was never actually used
- Safe to remove immediately


### 3.5 Intent Migration Guide 📝 DOCUMENTATION
- **File**: `backend/services/intent_migration_guide.py`
- **Status**: Documentation file in wrong location
- **Impact**: None (pure documentation)
- **Replacement**: Convert to markdown in docs/ folder
- **Decision**: [ ] Keep [X] Move to Docs [ ] Remove

**Notes:**
**📝 MOVE TO DOCS - NOT SERVICE CODE**
- Contains migration checklists and examples
- No functional code being executed
- Should be `docs/INTENT_MIGRATION_GUIDE.md`
- Move then delete Python file


### 3.6 Unified Intent Classification Service ❌ DEAD
- **File**: `backend/services/unified_intent_classification_service.py`
- **Status**: Only used by unregistered API (chat_api_optimized.py)
- **Impact**: ZERO - the API using it isn't registered
- **Replacement**: N/A - never integrated
- **Decision**: [ ] Keep [X] Deprecate [ ] Merge

**Notes:**
**❌ DEPRECATE - USED BY DEAD API**
- Only imported in chat_api_optimized.py (5 times)
- BUT chat_api_optimized.py is NOT registered in main.py
- Dead API = dead service
- Safe to remove with API


### 3.7 Chat API Optimized ❌ DEAD API
- **File**: `backend/api/chat_api_optimized.py`
- **Status**: NOT registered in main.py
- **Impact**: ZERO - never exposed to users
- **Replacement**: async_orchestrator_api.py (active)
- **Decision**: [ ] Keep [X] Deprecate [ ] Merge

**Notes:**
**❌ DEPRECATE - UNREGISTERED API**
- Searched main.py for "chat_api_optimized": ZERO matches
- API exists but never imported/registered
- Uses UnifiedIntentClassificationService (also unused)
- Safe to remove immediately


---

## Category 4: Niche/Experimental Agents

### 4.1 Wargaming Agent
- **File**: `backend/services/langgraph_agents/wargaming_agent.py`
- **Why deprecated**: Highly specialized, niche use case
- **Impact**: Integrated into orchestrator but likely low usage
- **Replacement**: Extract to optional plugin/extension
- **Decision**: [ ] Keep Core [ ] Make Optional [ ] Deprecate

**Notes:**


### 4.2 Fiction Editing Suite (4 agents)
- **Files**:
  - `backend/services/langgraph_agents/fiction_editing_agent.py`
  - `backend/services/langgraph_agents/story_analysis_agent.py`
  - `backend/services/langgraph_agents/character_development_agent.py`
  - `backend/services/langgraph_agents/outline_editing_agent.py`
- **Why deprecated**: Specialized creative writing features, may not align with core RAG focus
- **Impact**: 4 agents integrated into orchestrator
- **Replacement**: Extract to "Creative Writing Extension" plugin
- **Decision**: [ ] Keep All Core [ ] Make Plugin [ ] Deprecate Suite

**Notes:**


### 4.3 Proofreading Agent
- **File**: `backend/services/langgraph_agents/proofreading_agent.py`
- **Why deprecated**: Specialized feature, may overlap with content analysis
- **Impact**: Part of fiction editing suite
- **Replacement**: Merge with content analysis agent OR extract to plugin
- **Decision**: [ ] Keep Core [ ] Make Plugin [ ] Deprecate

**Notes:**


### 4.4 Podcast Script Agent
- **File**: `backend/services/langgraph_agents/podcast_script_agent.py`
- **Why deprecated**: Specialized content creation feature
- **Impact**: Integrated into orchestrator
- **Replacement**: Extract to "Content Creation Extension"
- **Decision**: [ ] Keep Core [ ] Make Plugin [ ] Deprecate

**Notes:**


### 4.5 Substack Agent
- **File**: `backend/services/langgraph_agents/substack_agent.py`
- **Why deprecated**: Platform-specific integration
- **Impact**: Research integration for Substack content
- **Replacement**: Generic web content agent with Substack support
- **Decision**: [ ] Keep Core [ ] Merge with Web Tools [ ] Deprecate

**Notes:**


### 4.6 Rules Editing Agent
- **File**: `backend/services/langgraph_agents/rules_editing_agent.py`
- **Why deprecated**: Meta-feature for editing agent rules, may not be frequently used
- **Impact**: Specialized meta-programming feature
- **Replacement**: Manual rule editing OR editor agent
- **Decision**: [ ] Keep [ ] Deprecate [ ] Merge with Editor Agent

**Notes:**


---

## Category 5: API Endpoint Duplication

### 5.1 Chat API Optimized
- **File**: `backend/api/chat_api_optimized.py`
- **Why deprecated**: Multiple chat API endpoints exist
- **Impact**: 6 endpoints, may overlap with orchestrator_chat_api.py
- **Replacement**: Consolidate into unified chat API
- **Decision**: [ ] Keep [ ] Deprecate [ ] Merge

**Notes:**


### 5.2 Orchestrator Chat API
- **File**: `backend/api/orchestrator_chat_api.py`
- **Why deprecated**: Separate from main chat API
- **Impact**: 2 endpoints for orchestrator-specific chat
- **Replacement**: Merge with unified chat API
- **Decision**: [ ] Keep Separate [ ] Merge [ ] Deprecate

**Notes:**


### 5.3 Unified Chat API
- **File**: `backend/api/unified_chat_api.py`
- **Why deprecated**: Third chat API, only implements job cancellation
- **Impact**: 1 endpoint for job cancellation
- **Replacement**: Move cancellation to main chat API
- **Decision**: [ ] Keep [ ] Merge [ ] Deprecate

**Notes:**


### 5.4 Agent Chaining API
- **File**: `backend/api/agent_chaining_api.py`
- **Why deprecated**: Specialized agent chaining endpoint
- **Impact**: 8 endpoints for direct agent chaining
- **Replacement**: Orchestrator handles agent chaining internally
- **Decision**: [ ] Keep [ ] Deprecate [ ] Refactor

**Notes:**


### 5.5 Background Chat API
- **File**: `backend/api/background_chat_api.py`
- **Why deprecated**: May overlap with async orchestrator
- **Impact**: Unknown number of endpoints
- **Replacement**: Async orchestrator API
- **Decision**: [ ] Keep [ ] Deprecate [ ] Merge

**Notes:**


---

## Category 6: Frontend Hook Duplication

### 6.1 useChatManager (Original)
- **File**: `frontend/src/hooks/useChatManager.js`
- **Why deprecated**: Multiple versions of same hook exist
- **Impact**: May be used in legacy components
- **Replacement**: useChatManagerUnified
- **Decision**: [ ] Keep [ ] Deprecate [ ] Already Replaced

**Notes:**


### 6.2 useChatManagerOptimized
- **File**: `frontend/src/hooks/useChatManagerOptimized.js`
- **Why deprecated**: Intermediate optimization version
- **Impact**: May be used in some components
- **Replacement**: useChatManagerUnified
- **Decision**: [ ] Keep [ ] Deprecate [ ] Already Replaced

**Notes:**


### 6.3 useChatManagerUnified (Current)
- **File**: `frontend/src/hooks/useChatManagerUnified.js`
- **Why deprecated**: N/A - This is the current version
- **Impact**: Should be the only chat manager hook
- **Replacement**: N/A - Keep this one
- **Decision**: [X] Keep [ ] Deprecate [ ] Refactor

**Notes:**


---

## Category 7: Frontend Service Files

### 7.1 apiService.monolithic.backup.js
- **File**: `frontend/src/services/apiService.monolithic.backup.js`
- **Why deprecated**: Backup file from refactoring
- **Impact**: None (backup file)
- **Replacement**: Current modular services
- **Decision**: [ ] Keep as Backup [ ] Delete Immediately

**Notes:**


### 7.2 apiService.new.js
- **File**: `frontend/src/services/apiService.new.js`
- **Why deprecated**: Temporary "new" version during migration
- **Impact**: Unknown if actively used
- **Replacement**: Check if merged into main apiService
- **Decision**: [ ] Keep [ ] Delete [ ] Merge

**Notes:**


---

## Category 8: Backend Service Files

### 8.1 Tool Cleanup Summary
- **File**: `backend/services/tool_cleanup_summary.py`
- **Why deprecated**: Documentation/summary file, not active code
- **Impact**: None (documentation)
- **Replacement**: Move to docs/ or remove
- **Decision**: [ ] Keep [ ] Move to Docs [ ] Remove

**Notes:**


### 8.2 Priority Queue Fix
- **File**: `backend/services/priority_queue_fix.py`
- **Why deprecated**: Sounds like a one-time fix script
- **Impact**: Unknown if still needed
- **Replacement**: Remove if fix is applied
- **Decision**: [ ] Keep [ ] Verify and Remove [ ] Already Applied

**Notes:**


### 8.3 Agent Intelligence Network
- **File**: `backend/services/agent_intelligence_network.py`
- **Why deprecated**: Unclear purpose, may be experimental
- **Impact**: Unknown dependencies
- **Replacement**: Check if integrated or experimental
- **Decision**: [ ] Keep [ ] Deprecate [ ] Document Purpose

**Notes:**


### 8.4 Lazy Chat Service
- **File**: `backend/services/lazy_chat_service.py`
- **Why deprecated**: May be superseded by orchestrator
- **Impact**: Unknown usage
- **Replacement**: Orchestrator chat service
- **Decision**: [ ] Keep [ ] Deprecate [ ] Already Replaced

**Notes:**


### 8.5 Migration Service
- **File**: `backend/services/migration_service.py`
- **Why deprecated**: One-time migration utility
- **Impact**: Should be temporary
- **Replacement**: Remove after migrations complete
- **Decision**: [ ] Keep [ ] Remove After Migrations [ ] Keep for Future

**Notes:**


### 8.6 Worker Warmup
- **File**: `backend/services/worker_warmup.py`
- **Why deprecated**: May be experimental or one-time optimization
- **Impact**: Unknown if actively used
- **Replacement**: Integrate into worker startup or remove
- **Decision**: [ ] Keep [ ] Integrate [ ] Remove

**Notes:**


---

## Category 9: Twitter Integration (Potentially Incomplete)

### 9.1 Twitter Ingestion Service
- **File**: `backend/services/twitter_ingestion_service.py`
- **Why deprecated**: Platform integration may be incomplete/unused
- **Impact**: Unknown usage, API costs
- **Replacement**: Remove or document as experimental
- **Decision**: [ ] Keep [ ] Deprecate [ ] Mark Experimental

**Notes:**


### 9.2 Twitter Celery Tasks
- **File**: `backend/services/celery_tasks/twitter_tasks.py`
- **Why deprecated**: Part of Twitter integration
- **Impact**: Background tasks for Twitter
- **Replacement**: Remove with Twitter service
- **Decision**: [ ] Keep [ ] Deprecate [ ] Mark Experimental

**Notes:**


### 9.3 Twitter Settings Component
- **File**: `frontend/src/components/SettingsServicesTwitter.js`
- **Why deprecated**: Frontend for Twitter integration
- **Impact**: UI component for Twitter settings
- **Replacement**: Remove if Twitter service deprecated
- **Decision**: [ ] Keep [ ] Deprecate [ ] Mark Experimental

**Notes:**


---

## Category 10: Agent Helper Modules

### 10.1 Agent Template
- **File**: `backend/services/langgraph_agents/agent_template.py`
- **Why deprecated**: Template/scaffolding file
- **Impact**: None (template only)
- **Replacement**: Keep as reference or move to docs
- **Decision**: [ ] Keep as Template [ ] Move to Docs [ ] Remove

**Notes:**


### 10.2 Agent Chain Memory
- **File**: `backend/services/langgraph_agents/agent_chain_memory.py`
- **Why deprecated**: May be superseded by shared memory system
- **Impact**: Unknown usage
- **Replacement**: SharedMemory system in enhanced_state
- **Decision**: [ ] Keep [ ] Deprecate [ ] Already Replaced

**Notes:**


### 10.3 Agent Workflow Engine
- **File**: `backend/services/langgraph_agents/agent_workflow_engine.py`
- **Why deprecated**: May overlap with orchestrator workflow
- **Impact**: Unknown usage
- **Replacement**: Orchestrator workflow management
- **Decision**: [ ] Keep [ ] Deprecate [ ] Clarify Purpose

**Notes:**


### 10.4 Research Routing Utils
- **File**: `backend/services/langgraph_agents/research_routing_utils.py`
- **Why deprecated**: Agent-specific utility, may be better in agent file
- **Impact**: Used by research agent
- **Replacement**: Merge into research agent
- **Decision**: [ ] Keep Separate [ ] Merge [ ] Deprecate

**Notes:**


---

## Category 11: Service Container Legacy Systems

### 11.1 Capability Workflow Engine
- **File**: `backend/services/capability_workflow_engine.py`
- **Why deprecated**: Part of capability routing system
- **Impact**: May be experimental
- **Replacement**: Orchestrator workflow
- **Decision**: [ ] Keep [ ] Deprecate [ ] Merge

**Notes:**


### 11.2 Capabilities Service
- **File**: `backend/services/capabilities_service.py`
- **Why deprecated**: Related to capability-based routing
- **Impact**: Unknown usage
- **Replacement**: Check if integrated or experimental
- **Decision**: [ ] Keep [ ] Deprecate [ ] Document

**Notes:**


### 11.3 Orchestrator Routing
- **File**: `backend/services/orchestrator_routing.py`
- **Why deprecated**: May be superseded by official orchestrator
- **Impact**: Legacy routing logic
- **Replacement**: LangGraph official orchestrator
- **Decision**: [ ] Keep [ ] Deprecate [ ] Already Replaced

**Notes:**


### 11.4 Orchestrator Nodes
- **File**: `backend/services/orchestrator_nodes.py`
- **Why deprecated**: May be legacy node definitions
- **Impact**: Unknown usage
- **Replacement**: Nodes in official orchestrator
- **Decision**: [ ] Keep [ ] Deprecate [ ] Already Replaced

**Notes:**


### 11.5 Orchestrator Utils
- **File**: `backend/services/orchestrator_utils.py`
- **Why deprecated**: Utility file for potentially deprecated orchestrator
- **Impact**: Unknown usage
- **Replacement**: Official orchestrator utilities
- **Decision**: [ ] Keep [ ] Deprecate [ ] Merge

**Notes:**


### 11.6 Pending Query Manager
- **File**: `backend/services/pending_query_manager.py`
- **Why deprecated**: May be legacy HITL implementation
- **Impact**: Permission/HITL workflow
- **Replacement**: LangGraph native HITL with checkpointer
- **Decision**: [ ] Keep [ ] Deprecate [ ] Verify Replacement

**Notes:**


---

## Category 12: Intelligence/Analysis Services

### 12.1 Context Intelligence Service
- **File**: `backend/services/context_intelligence_service.py`
- **Why deprecated**: May overlap with conversation intelligence
- **Impact**: Unknown usage
- **Replacement**: Conversation intelligence service
- **Decision**: [ ] Keep Both [ ] Merge [ ] Deprecate One

**Notes:**


### 12.2 Conversation Intelligence Service
- **File**: `backend/services/conversation_intelligence_service.py`
- **Why deprecated**: May overlap with context intelligence
- **Impact**: Unknown usage
- **Replacement**: Single intelligence service
- **Decision**: [ ] Keep Both [ ] Merge [ ] Deprecate One

**Notes:**


### 12.3 Conversation Context Service
- **File**: `backend/services/conversation_context_service.py`
- **Why deprecated**: Third conversation-related service
- **Impact**: Unknown usage
- **Replacement**: Consolidate conversation services
- **Decision**: [ ] Keep [ ] Merge [ ] Deprecate

**Notes:**


---

## Category 13: Testing/Experimental Components

### 13.1 Context Aware Research Test
- **File**: `frontend/src/components/ContextAwareResearchTest.js`
- **Why deprecated**: Test component in production code
- **Impact**: None (test only)
- **Replacement**: Remove or move to tests/
- **Decision**: [ ] Keep [ ] Remove [ ] Move to Tests

**Notes:**


### 13.2 Hybrid Research Test
- **File**: `frontend/src/components/HybridResearchTest.js`
- **Why deprecated**: Test component in production code
- **Impact**: None (test only)
- **Replacement**: Remove or move to tests/
- **Decision**: [ ] Keep [ ] Remove [ ] Move to Tests

**Notes:**


### 13.3 Unified Chat Test
- **File**: `frontend/src/components/UnifiedChatTest.js`
- **Why deprecated**: Test component in production code
- **Impact**: None (test only)
- **Replacement**: Remove or move to tests/
- **Decision**: [ ] Keep [ ] Remove [ ] Move to Tests

**Notes:**


---

## Category 14: Specialized Services

### 14.1 Calibre Search Service
- **File**: `backend/services/calibre_search_service.py`
- **Why deprecated**: Platform-specific integration (ebook library)
- **Impact**: Calibre integration feature
- **Replacement**: Keep if ebook management is core feature
- **Decision**: [ ] Keep Core [ ] Make Optional [ ] Deprecate

**Notes:**


### 14.2 Substack Research Helper
- **File**: `backend/services/substack_research_helper.py`
- **Why deprecated**: Platform-specific, may overlap with Substack agent
- **Impact**: Substack content analysis
- **Replacement**: Substack agent
- **Decision**: [ ] Keep [ ] Merge with Agent [ ] Deprecate

**Notes:**


### 14.3 Collection Analysis Service
- **File**: `backend/services/collection_analysis_service.py`
- **Why deprecated**: Specialized feature, unclear usage
- **Impact**: Unknown usage
- **Replacement**: Content analysis agent
- **Decision**: [ ] Keep [ ] Deprecate [ ] Document Purpose

**Notes:**


### 14.4 Clarity Assessment Service
- **File**: `backend/services/clarity_assessment_service.py`
- **Why deprecated**: Specialized analysis feature
- **Impact**: Unknown usage
- **Replacement**: Content analysis agent or merge
- **Decision**: [ ] Keep [ ] Merge [ ] Deprecate

**Notes:**


---

## Category 15: Pipeline System Components

### 15.1 Pipeline DSL Converter
- **File**: `backend/services/pipeline_dsl_converter.py`
- **Why deprecated**: May be one-time migration utility
- **Impact**: Pipeline system
- **Replacement**: Keep if pipeline DSL is evolving
- **Decision**: [ ] Keep [ ] Remove After Migration [ ] Keep for Flexibility

**Notes:**


### 15.2 Pipeline Validator
- **File**: `backend/services/pipeline_validator.py`
- **Why deprecated**: N/A - Validation is important
- **Impact**: Pipeline system validation
- **Replacement**: N/A - Should keep
- **Decision**: [X] Keep [ ] Refactor [ ] Deprecate

**Notes:**


---

## Category 16: Org-Mode Specialized Features

### 16.1 Org Inbox Agent
- **File**: `backend/services/langgraph_agents/org_inbox_agent.py`
- **Why deprecated**: Specialized Org-Mode feature
- **Impact**: Org-Mode inbox processing
- **Replacement**: Keep if Org-Mode is core feature
- **Decision**: [ ] Keep Core [ ] Make Optional [ ] Deprecate

**Notes:**


### 16.2 Org Project Agent
- **File**: `backend/services/langgraph_agents/org_project_agent.py`
- **Why deprecated**: Specialized Org-Mode feature
- **Impact**: Org-Mode project management
- **Replacement**: Keep if Org-Mode is core feature
- **Decision**: [ ] Keep Core [ ] Make Optional [ ] Deprecate

**Notes:**


---

## Summary Statistics

**Total Items for Review**: 70

### Breakdown by Category:
- Infrastructure: 2 items (✅ 2 decided)
- Agents: 14 items
- Intent Classification: 7 items (✅ 7 decided)
- API Endpoints: 5 items
- Frontend: 6 items
- Backend Services: 20 items
- Twitter Integration: 3 items
- Specialized Services: 13 items

### Progress:
- **Completed Categories:** 2/16 (Infrastructure, Intent Classification)
- **Items Decided:** 9/70 (13%)
- **✅ CLEANUP EXECUTED:** 6 intent files + 1 API = 7 files deleted
- **See:** `INTENT_CLASSIFICATION_CLEANUP_SUMMARY.md` for full report

### Priority Levels:
- **High Priority** (Breaking changes/dependencies): Infrastructure, Intent Classification
- **Medium Priority** (Code quality): Duplicate agents, API consolidation
- **Low Priority** (Cleanup): Test components, backup files, documentation files

---

## Next Steps

1. **Review high-priority items first** (Infrastructure, Intent Classification)
2. **Check dependencies** for each item using codebase search
3. **Make decisions** and mark checkboxes
4. **Document reasoning** in Notes sections
5. **Create deprecation plan** based on decisions
6. **Execute cleanup** in phases to minimize breakage

**BULLY!** Let the systematic code cleanup campaign begin! 🎖️

s