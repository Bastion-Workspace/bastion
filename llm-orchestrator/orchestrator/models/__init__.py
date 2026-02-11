"""
Models package for llm-orchestrator

Provides Pydantic models for structured data:
- Intent classification results
- Agent responses
- Routing decisions
- Research assessments and gap analysis
- Editor operations for fiction and outline editing
"""

from orchestrator.models.intent_models import SimpleIntentResult
from orchestrator.models.research_models import (
    ResearchAssessmentResult,
    ResearchGapAnalysis,
    QuickAnswerAssessment,
    QueryTypeDetection
)
from orchestrator.models.electronics_models import (
    FileRouteItem,
    FileRoutingPlan,
    ContentStructure,
    QueryTypeAnalysis,
    ProjectPlanAnalysis,
    SearchNeedAnalysis,
    FollowUpAnalysis,
    ContentConflictAnalysis,
    ResponseQualityAnalysis,
    IncrementalUpdateAnalysis,
    ComponentCompatibilityAnalysis
)
from orchestrator.models.editor_models import (
    EditorOperation,
    ManuscriptEdit
)
from orchestrator.models.visualization_models import (
    VisualizationAnalysis,
    VisualizationResult
)
from orchestrator.models.diagramming_models import (
    DiagramAnalysis,
    DiagramResult
)
from orchestrator.models.agent_response_contract import (
    AgentResponse,
    ManuscriptEditMetadata,
    TaskStatus
)

__all__ = [
    'SimpleIntentResult',
    'ResearchAssessmentResult',
    'ResearchGapAnalysis',
    'QuickAnswerAssessment',
    'QueryTypeDetection',
    'FileRouteItem',
    'FileRoutingPlan',
    'ContentStructure',
    'QueryTypeAnalysis',
    'ProjectPlanAnalysis',
    'SearchNeedAnalysis',
    'FollowUpAnalysis',
    'ContentConflictAnalysis',
    'ResponseQualityAnalysis',
    'IncrementalUpdateAnalysis',
    'ComponentCompatibilityAnalysis',
    'EditorOperation',
    'ManuscriptEdit',
    'VisualizationAnalysis',
    'VisualizationResult',
    'DiagramAnalysis',
    'DiagramResult',
    'AgentResponse',
    'ManuscriptEditMetadata',
    'TaskStatus',
    'QueryPlan'
]

