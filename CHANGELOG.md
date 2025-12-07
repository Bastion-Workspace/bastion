# Changelog

All notable changes to Bastion AI Workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
- Fix: Chat message copy now preserves markdown formatting as rich text (HTML) while keeping raw markdown syntax for plain text editors
- Feature: Added Dictionary Agent with short-circuit routing for "define:" queries (instant lexicographic lookups)
- Fix: Resolved conversation ID mismatch causing 403 errors when accessing conversations from checkpoint list
- Feature: Added universal DocumentEditBatch system for atomic multi-edit operations
- Feature: Added intelligent reference file creation to general and electronics agents (>1500 char threshold)
- Refactor: Migrated electronics_agent to use batch editing for atomic, efficient document updates
- Update: Enhanced both project agents with automatic file creation and frontmatter management

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

