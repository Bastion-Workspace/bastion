# Changelog

All notable changes to Bastion AI Workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.0] - 2025-01-XX
- Feature: Import generated images into the document library
- Refactor: Simplify fiction agent
- Fix: Prompt multiplication bloat in Fiction agent
- Feature: ePub export (untested)
- Feature: 'Cover' frontmatter type support in Fiction documents

## [Unreleased]
- Fix: Resolved fiction generation subgraph state preservation bug where generation_context_parts was dropped across nodes
- Fix: Added failed_operations conversion in fiction generation to display unplaceable content in chat sidebar
- Feature: Added context-aware fuzzy matching with auto-correction for fiction agent anchor texts to prevent LLM hallucination errors
- Fix: Resolved message normalization bug where editor_operations buried in metadata were not extracted to top level for display
- Feature: Implemented automatic text-splitting for large editor operations into 8K char chunks at paragraph boundaries to prevent nginx proxy truncation and maintain clean word boundaries
- Fix: Added chunk_index sorting to frontend operation application to ensure text chunks are applied in correct sequence
- Fix: Removed obsolete chunk reversal logic in backend that was causing text chunks to be sent in reverse order
- Fix: Added explicit instructions for fiction agent to use insert_after (not insert_after_heading) when continuing existing chapters to prevent duplication
- Fix: Resolved DOCX file reading errors in gRPC GetDocumentContent by using DocumentProcessor for binary formats
- Feature: Added automatic location fallback to user ZIP code for weather tools
- Feature: Added date range support for historical weather queries (e.g., "2022-10 to 2024-02")
- Fix: Resolved visualization tool TypeError in orchestrator gRPC client
- Fix: Enhanced chart generation to support static SVG versions via gRPC
- Fix: Fixed gRPC double-abort error in weather service error handling
- Feature: Added "Technical Hyperspace" Deterministic System Modeling Engine with gRPC topology service and NetworkX graph traversal
- Feature: Added Systems Engineering Agent for technical design and failure simulation analysis
- Refactor: Modularized monolithic backend/main.py into domain-specific API routers
- Refactor: Simplified PDF processing to focus on automated text extraction and vectorization
- Cleanup: Decommissioned obsolete manual OCR, coordinate-based PDF editing, and layout segmentation units
- Feature: Moved ePub cover page to the end of the spine to prevent "double cover" in readers
- Fix: Enhanced ePub cover resolution to support relative paths in frontmatter and database lookups
- Fix: Resolved XML syntax error in ePub cover page generation
- Update: Created Org-Mode enhancement roadmap in dev-notes
- Fix: Resolved issue where 'Filter by Tag' in All TODOs view was empty by correcting data property access
- Update: Centralized all weather intelligence logic into dedicated gRPC Tools Service
- Refactor: Migrated Weather Agent and Status Bar API to use gRPC weather service, eliminating redundant logic
- Feature: Added real-time moon phase and meteorological metadata to central weather tool
- Feature: Extracted dedicated gRPC Tools Service container from backend monolith
- Update: Reconfigured LLM Orchestrator to communicate directly with Tools Service via gRPC
- Refactor: Trimmed backend container size by moving tool logic to microservice
- Feature: Added image import functionality for generated images - users can now import images directly into document library folders
- Feature: Automatic cleanup of generated images when conversations are deleted - imported images are preserved, non-imported images are removed
- Fix: Implemented "Smart Validation" for editor diffs to prevent invalidation during full-document syncs
- Fix: Resolved race condition and network-sync related diff invalidation in the live editor
- Fix: Resolved race condition where rapid acceptance of multiple live diffs caused remaining diffs to disappear
- Fix: Prevented live edit diffs from disappearing when accepting multiple changes in rapid succession
- Fix: Resolved frontend build error due to missing variable definition in MessagingContext
- Fix: Resolved message duplication and unread count persistence issues in real-time chat
- Fix: Resolved issue where new rooms and initial messages failed to appear for recipients in real-time
- Fix: Resolved issue where user "offline" status failed to broadcast in real-time
- Fix: Resolved issue where users remained "online" after logging out or closing their browser
- Fix: Improved WebSocket headcount logic to correctly track multiple simultaneous connections per user
- Fix: Resolved issue where unread message counts would reappear after a page refresh
- Fix: Resolved issue with messaging presence indicators getting stuck or failing on refresh
- Fix: Resolved issue where messaging notifications failed to update until the drawer was opened
- Fix: Resolved circular RLS dependency in messaging system causing rooms to vanish on refresh
- Fix: Corrected room display name calculation to correctly identify other participants
- Feature: Added detailed logging for outline chapter extraction in fiction agent
- Fix: Improved forward-looking outline context for new manuscripts in fiction generator
- Refactor: Established standard agent handoff pattern using shared_memory for clean inter-agent data passing
- Update: Simplified intent classification to trust LLM semantic understanding over pattern matching
- Fix: Research agent now receives reference document context when delegated by reference agent
- Fix: Added 'reference' to allowed editor types in frontend for proper reference agent routing
- Fix: Backend now accepts reference documents even if not editable for reference_agent analysis
- Refactor: Removed unnecessary short-circuit routing hack, using normal intent classification
- Feature: Added in-editor suggestion capabilities for editor-interactive agents (fiction, outline, electronics)
- Feature: Media service integrations in progress (SubSonic, Audiobookshelf, Deezer)
- Fix: Fiction agent continuity file creation now uses same folder resolution as electronics agent (folder_id or folder_path from canonical_path)
- Fix: Enhanced cursor detection with complete gRPC proto support for fiction editing agent
- Fix: Chat message copy now preserves markdown formatting as rich text (HTML) while keeping raw markdown syntax for plain text editors
- Feature: Added Dictionary Agent with short-circuit routing for "/define" queries (instant lexicographic lookups)
- Update: Changed dictionary agent trigger from "define:" to "/define" command format
- Fix: Resolved conversation ID mismatch causing 403 errors when accessing conversations from checkpoint list
- Feature: Added universal DocumentEditBatch system for atomic multi-edit operations
- Feature: Added intelligent reference file creation to general and electronics agents (>1500 char threshold)
- Refactor: Migrated electronics_agent to use batch editing for atomic, efficient document updates
- Update: Enhanced both project agents with automatic file creation and frontmatter management
- Fix: Resolved FictionEditingState NameError by using forward references for type-safe state access helpers
- Fix: Improved type safety in fiction editing agent with Pydantic model conversion for structured_edit and continuity_state
- Fix: Enhanced operation resolution in fiction editing agent with better anchor text matching and error handling
- Fix: Active editor frontmatter now persists correctly during typing - removed re-parsing that was overwriting cached frontmatter with empty data

### Bug Fixes
- Fix: Resolved KeyError in electronics agent when user approves pending save operations
  - Root cause: Conditional edges from analyze_intent node missing 'save' and 'maintenance' destinations
  - Added missing routing paths for approval flow resumption

### Conversation History Persistence Fix
- **Fix**: Resolved critical conversation history issue where LLM orchestrator was treating every request as a new conversation
  - Root cause: Backend was not loading conversation state from database before sending to LLM orchestrator
  - Added `_load_conversation_state()` function to retrieve `primary_agent_selected` and conversation context
  - Intent classifier now properly maintains agent continuity between requests
  - Conversation history now persists correctly across LLM interactions
- **Fix**: LLM orchestrator agents now properly save conversations to LangGraph checkpoints matching backend behavior
  - Root cause: Agents were not adding current user query to messages before workflow invocation
  - Root cause: Agents were not adding assistant responses to messages after generation
  - Added `_prepare_messages_with_query()` helper to BaseAgent for consistent message preparation
  - Added `_add_assistant_response_to_messages()` helper to BaseAgent for checkpoint persistence
  - Updated all critical agents (chat, electronics, data_formatting, help, content_analysis) to use helpers
  - User queries and assistant responses now properly saved to PostgreSQL checkpoints
  - Created test script to verify checkpoint persistence (`test_checkpoint_persistence.py`)

### Architectural Refactoring - Vector Service Only (No Fallback)
- **Breaking**: Committed fully to Vector Service architecture (no fallback to legacy EmbeddingManager)
  - All embedding generation routes through Vector Service microservice (gRPC)
  - EmbeddingManager replaced with deprecation stub (raises NotImplementedError)
  - Removed 300+ lines of duplicate embedding generation logic
  - Single source of truth: Vector Service for all embeddings
- **Refactor**: Simplified EmbeddingServiceWrapper to Vector Service only
  - Removed fallback logic and USE_VECTOR_SERVICE flag checks
  - Always uses Vector Service for embedding generation
  - Always uses VectorStoreService for storage/search
  - Reduced from 330 lines to 350 lines (cleaner, simpler)
- **Refactor**: Updated ResilientEmbeddingManager to use EmbeddingServiceWrapper
  - No longer directly depends on EmbeddingManager
  - Uses embedding_service for generation, vector_store for storage

### Architectural Refactoring - Vector Storage Separation
- **Feature**: Created VectorStoreService for clean separation of concerns
  - New service handles all vector database operations (Qdrant)
  - Collection management, point insertion, vector search, deletion
  - Foundation for multi-backend support (Milvus, Elasticsearch)
- **Refactor**: Updated search tools to use new architecture
  - Tools generate embeddings then search via VectorStoreService
- **Breaking**: Removed unused ParallelEmbeddingManager dead code (710 lines)
- **Breaking**: Removed FreeFormNotesService and all related code including database table (1001 lines total)
- Refactor: Query expansion extracted to standalone QueryExpansionService with dedicated LangGraph tool
- Refactor: Removed automatic query expansion from EmbeddingManager (agents now explicitly control expansion)
- Refactor: Removed unused ParallelEmbeddingManager dead code
- Refactor: Removed FreeFormNotesService and all related code including database table (unused feature with no UI)

## [0.1.0] - 2025-11-13

### Initial Release
A comprehensive AI-powered personal workspace platform featuring 25+ specialized agents for knowledge management, creative writing, data workspaces, collaboration, and productivity - all controlled through natural language.

#### Core Features
- **LangGraph Multi-Agent System**: 25+ specialized agents with PostgreSQL state persistence
- **Document Management**: RAG with vector search (Qdrant) and knowledge graphs (Neo4j)
- **Data Workspaces**: Dedicated microservice with PostgreSQL for custom databases, data import (CSV/JSON/Excel), and table management
- **Real-time Collaboration**: Messaging system with presence indicators and optional encryption
- **Org-mode Integration**: WebDAV sync with dedicated inbox and project agents
- **RSS Feed Management**: Background polling and article import with dedicated agent
- **Creative Writing Suite**: Fiction editing, proofreading, character development, and outline editing agents
- **Knowledge Integration**: Entertainment knowledge base
- **Specialized Agents**: Weather, image generation, audio transcription, coding assistance, research, fact-checking, and more
- **Microservices Architecture**: Docker Compose deployment with vector-service, data-service, and llm-orchestrator

#### Technical Stack
- Backend: Python, FastAPI, LangGraph, Celery, gRPC
- Frontend: React, Material-UI, WebSockets
- Databases: PostgreSQL (main + data workspaces), Qdrant (vectors), Neo4j (knowledge graph)
- Infrastructure: Docker Compose, microservices architecture
- AI: OpenRouter/OpenAI integration with structured Pydantic outputs

