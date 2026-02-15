"""
Unified Dispatcher - Single entry point for skill-based dispatch.

All requests flow through this: discover skill -> pick engine -> run -> yield ChatChunk stream.
"""

import json
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from protos import orchestrator_pb2

from orchestrator.skills import get_skill_registry, load_all_skills
from orchestrator.skills.skill_schema import EngineType

logger = logging.getLogger(__name__)

_skills_loaded = False


def _ensure_skills_loaded() -> None:
    global _skills_loaded
    if not _skills_loaded:
        load_all_skills()
        _skills_loaded = True


class UnifiedDispatcher:
    """
    Dispatches requests to the appropriate engine based on discovered skill.
    Yields ChatChunk stream (status, content, complete, diagram, chart, editor_operation as needed).
    """

    def __init__(self) -> None:
        self._automation_engine = None
        self._editor_engine = None
        self._research_engine = None
        self._conversational_engine = None

    def _get_conversational_engine(self):
        if self._conversational_engine is None:
            from orchestrator.engines.conversational_engine import ConversationalEngine
            self._conversational_engine = ConversationalEngine()
        return self._conversational_engine

    def _get_research_engine(self):
        if self._research_engine is None:
            from orchestrator.engines.research_engine import ResearchEngine
            self._research_engine = ResearchEngine()
        return self._research_engine

    def _get_automation_engine(self):
        if self._automation_engine is None:
            from orchestrator.engines.automation_engine import AutomationEngine
            self._automation_engine = AutomationEngine()
        return self._automation_engine

    def _get_editor_engine(self):
        if self._editor_engine is None:
            from orchestrator.engines.editor_engine import EditorEngine
            self._editor_engine = EditorEngine()
        return self._editor_engine

    async def dispatch(
        self,
        skill_name: str,
        query: str,
        metadata: Dict[str, Any],
        messages: List[Any],
        cancellation_token: Optional[Any] = None,
    ) -> AsyncIterator[orchestrator_pb2.ChatChunk]:
        """
        Run the skill via its engine and yield ChatChunk stream.
        """
        _ensure_skills_loaded()
        registry = get_skill_registry()
        skill = registry.get(skill_name)
        if not skill:
            logger.warning("Skill not found: %s, falling back to content chunk", skill_name)
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message="Skill not found. Please try again.",
                timestamp=datetime.now().isoformat(),
                agent_name=skill_name or "system",
            )
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        agent_label = f"{skill_name}_agent" if not skill_name.endswith("_agent") else skill_name

        if skill.engine == EngineType.AUTOMATION:
            yield orchestrator_pb2.ChatChunk(
                type="status",
                message=f"Processing with {skill.description[:50]}...",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            engine = self._get_automation_engine()
            try:
                result = await engine.process(
                    query=query,
                    metadata=metadata,
                    messages=messages,
                    skill_name=skill_name,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                logger.exception("Automation engine failed: %s", e)
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=f"Error: {e}",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return
            result_status = result.get("task_status") or (result.get("response", {}) or {}).get("task_status", "")
            if result_status == "rejected":
                yield orchestrator_pb2.ChatChunk(
                    type="rejected",
                    message="Skill rejected this query",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                return
            response_text = result.get("response", "") if isinstance(result.get("response"), str) else (result.get("response", {}) or {}).get("response", "")
            if not response_text and isinstance(result.get("response"), dict):
                response_text = result.get("response", {}).get("response", "")

            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=response_text or "Done.",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        if skill.engine == EngineType.RESEARCH:
            yield orchestrator_pb2.ChatChunk(
                type="status",
                message=f"Research: {skill.description[:50]}...",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            engine = self._get_research_engine()
            try:
                result = await engine.process(
                    query=query,
                    metadata=metadata,
                    messages=messages,
                    skill_name=skill_name,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                logger.exception("Research engine failed: %s", e)
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=f"Error: {e}",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return
            # FullResearchAgent returns dict with "response", "images", citations, etc. (AgentResponse contract)
            response_text = result.get("response", "")
            if isinstance(response_text, dict):
                response_text = response_text.get("response", response_text.get("message", "")) or ""
            else:
                response_text = str(response_text) if response_text else ""
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=response_text or "Done.",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            # Include structured images and other metadata in complete chunk (frontend expects metadata.images)
            images = result.get("images") or result.get("structured_images")
            chunk_metadata = {}
            if images:
                chunk_metadata["images"] = json.dumps(images)
                logger.info("Research dispatch: including %d image(s) in complete metadata", len(images))
            for key in ("citations", "sources", "static_visualization_data", "static_format", "chart_result"):
                val = result.get(key)
                if val is not None:
                    chunk_metadata[key] = json.dumps(val) if not isinstance(val, str) else val
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
                metadata=chunk_metadata if chunk_metadata else None,
            )
            return

        if skill.engine == EngineType.EDITOR:
            # All EDITOR skills go through EditorEngine (WritingAssistantAgent).
            shared_memory = metadata.get("shared_memory", {})
            active_editor = shared_memory.get("active_editor") or {}
            if not active_editor or not active_editor.get("document_id"):
                yield orchestrator_pb2.ChatChunk(
                    type="status",
                    message="This writing skill needs an open document.",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=(
                        "This skill works with an open document. Open a document of the right type and try again, "
                        "or ask your question in chat for a general answer. For factual or how-to questions without a document, "
                        "try asking in chat or rephrase so research can look it up."
                    ),
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return

            yield orchestrator_pb2.ChatChunk(
                type="status",
                message=f"Editor: {skill.description[:50]}...",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            engine = self._get_editor_engine()
            try:
                result = await engine.process(
                    query=query,
                    metadata=metadata,
                    messages=messages,
                    skill_name=skill_name,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                logger.exception("Editor engine failed: %s", e)
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=f"Error: {e}",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return
            # Extract response text
            response = result.get("response", {})
            if isinstance(response, dict):
                response_text = response.get("response", response.get("message", ""))
            else:
                response_text = str(response) if response else "Done."
            
            # Extract editor operations and manuscript_edit for editor agents
            editor_operations = result.get("editor_operations")
            if not editor_operations and isinstance(response, dict):
                editor_operations = response.get("editor_operations")
            manuscript_edit = result.get("manuscript_edit")
            if not manuscript_edit and isinstance(response, dict):
                manuscript_edit = response.get("manuscript_edit")
            
            # Yield content chunk
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=response_text or "Done.",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            
            # Yield editor operations if present
            if editor_operations:
                # Extract document_id and filename from metadata
                shared_memory = metadata.get("shared_memory", {})
                active_editor = shared_memory.get("active_editor", {})
                document_id = metadata.get("target_document_id") or active_editor.get("document_id")
                filename = active_editor.get("filename")
                
                editor_ops_data = {
                    "operations": editor_operations,
                    "manuscript_edit": manuscript_edit,
                    "document_id": document_id,
                    "filename": filename
                }
                logger.info(f"✅ EDITOR DISPATCH: Sending {len(editor_operations)} editor operations (doc={document_id}, file={filename})")
                yield orchestrator_pb2.ChatChunk(
                    type="editor_operations",
                    message=json.dumps(editor_ops_data),
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
            
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
            )
            return

        if skill.engine == EngineType.CONVERSATIONAL:
            yield orchestrator_pb2.ChatChunk(
                type="status",
                message=f"Chat: {skill.description[:50]}...",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            engine = self._get_conversational_engine()
            try:
                result = await engine.process(
                    query=query,
                    metadata=metadata,
                    messages=messages,
                    skill_name=skill_name,
                    cancellation_token=cancellation_token,
                )
            except Exception as e:
                logger.exception("Conversational engine failed: %s", e)
                yield orchestrator_pb2.ChatChunk(
                    type="content",
                    message=f"Error: {e}",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                yield orchestrator_pb2.ChatChunk(
                    type="complete",
                    message="Complete",
                    timestamp=datetime.now().isoformat(),
                    agent_name="system",
                )
                return
            result_status = result.get("task_status") or (result.get("response", {}) or {}).get("task_status", "")
            if result_status == "rejected":
                yield orchestrator_pb2.ChatChunk(
                    type="rejected",
                    message="Skill rejected this query",
                    timestamp=datetime.now().isoformat(),
                    agent_name=agent_label,
                )
                return
            # ChatAgent returns result with "response" key (AgentResponse dict or handoff data)
            response = result.get("response", "")
            if isinstance(response, dict):
                response_text = response.get("response", response.get("message", "")) or ""
            else:
                response_text = str(response) if response else ""
            yield orchestrator_pb2.ChatChunk(
                type="content",
                message=response_text or "Done.",
                timestamp=datetime.now().isoformat(),
                agent_name=agent_label,
            )
            # Include images and other metadata when chat handed off to research (same as RESEARCH path)
            images = result.get("images") or result.get("structured_images")
            chunk_metadata = {}
            if images:
                chunk_metadata["images"] = json.dumps(images)
                logger.info("Chat dispatch: including %d image(s) in complete metadata (handoff)", len(images))
            for key in ("citations", "sources", "static_visualization_data", "static_format", "chart_result"):
                val = result.get(key)
                if val is not None:
                    chunk_metadata[key] = json.dumps(val) if not isinstance(val, str) else val
            yield orchestrator_pb2.ChatChunk(
                type="complete",
                message="Complete",
                timestamp=datetime.now().isoformat(),
                agent_name="system",
                metadata=chunk_metadata if chunk_metadata else None,
            )
            return

        logger.warning("UnifiedDispatcher: engine %s not yet implemented, skill=%s", skill.engine, skill_name)
        yield orchestrator_pb2.ChatChunk(
            type="content",
            message="This capability is not yet available via skill dispatch.",
            timestamp=datetime.now().isoformat(),
            agent_name=agent_label,
        )
        yield orchestrator_pb2.ChatChunk(
            type="complete",
            message="Complete",
            timestamp=datetime.now().isoformat(),
            agent_name="system",
        )


_dispatcher_instance: Optional[UnifiedDispatcher] = None


def get_unified_dispatcher() -> UnifiedDispatcher:
    """Return singleton dispatcher."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = UnifiedDispatcher()
    return _dispatcher_instance
