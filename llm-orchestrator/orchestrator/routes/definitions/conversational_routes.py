"""
Conversational engine route definitions.

Routes: general chat (fallback), story analysis.
"""

from orchestrator.routes.route_schema import EngineType, Route

CONVERSATIONAL_ROUTES = [
    Route(
        name="chat",
        description="General conversation, Q&A, and fallback when no specialized route matches.",
        engine=EngineType.CONVERSATIONAL,
        domains=["general"],
        actions=["observation", "query"],
        keywords=[],
        priority=0,
        tools=[],
        system_prompt="You are a helpful assistant. Engage in natural conversation and answer questions. Defer to research when the user needs factual lookup.",
    ),
    Route(
        name="story_analysis",
        description="Discuss and critique fiction concepts and theory (not for reviewing specific chapters/content in an open file - use fiction_editing for that). Analyze story structure, discuss writing techniques, provide general writing advice. Does not edit or assess specific manuscript content.",
        engine=EngineType.CONVERSATIONAL,
        domains=["fiction", "writing", "story"],
        actions=["analysis"],
        context_boost=10,
        requires_explicit_keywords=True,
        keywords=["discuss", "concept", "theory", "writing advice", "storytelling", "technique"],
        priority=60,
        tools=[],
        system_prompt="You discuss fiction writing concepts and theory. Provide advice on storytelling techniques, structure patterns, and general writing craft. You do not review or edit specific manuscript content.",
        internal_only=True,
    ),
]
