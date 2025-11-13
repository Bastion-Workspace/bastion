# Changelog

All notable changes to Bastion AI Workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **Knowledge Integration**: Calibre library integration, entertainment knowledge base
- **Specialized Agents**: Weather, image generation, audio transcription, coding assistance, research, fact-checking, and more
- **Microservices Architecture**: Docker Compose deployment with vector-service, data-service, and llm-orchestrator

#### Technical Stack
- Backend: Python, FastAPI, LangGraph, Celery, gRPC
- Frontend: React, Material-UI, WebSockets
- Databases: PostgreSQL (main + data workspaces), Qdrant (vectors), Neo4j (knowledge graph)
- Infrastructure: Docker Compose, microservices architecture
- AI: OpenRouter/OpenAI integration with structured Pydantic outputs

