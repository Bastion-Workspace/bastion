"""
Models package for llm-orchestrator

Provides Pydantic models for structured data:
- Intent classification results
- Agent responses
- Routing decisions
- Editor operations (ManuscriptEdit) for document tools
"""

from orchestrator.models.intent_models import SimpleIntentResult
from orchestrator.models.editor_models import (
    EditorOperation,
    ManuscriptEdit
)
from orchestrator.models.agent_response_contract import (
    AgentResponse,
    ManuscriptEditMetadata,
    TaskStatus
)

__all__ = [
    'SimpleIntentResult',
    'EditorOperation',
    'ManuscriptEdit',
    'AgentResponse',
    'ManuscriptEditMetadata',
    'TaskStatus',
]
