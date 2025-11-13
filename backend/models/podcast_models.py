"""
Podcast Models - Roosevelt's Plain-Text ElevenLabs Script Structures

Structured request/response for the podcast script agent producing
plain text with inline bracket cues (e.g., [stress], [flatly], [pause:1.0s]).
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal


class PodcastScriptRequest(BaseModel):
    """User-facing request options for script generation."""
    topic: str = Field(description="Primary topic or episode theme")
    target_length_words: int = Field(default=900, ge=120, le=4000, description="Approximate target length in words")
    tone: Literal["neutral", "warm", "excited", "serious", "playful"] = Field(default="warm", description="Overall narration tone")
    pacing: Literal["brisk", "moderate", "measured"] = Field(default="moderate", description="Narration pacing guidance")
    include_music_cues: bool = Field(default=True, description="Insert [music:...] cues at intro/segues/outro")
    include_sfx_cues: bool = Field(default=False, description="Insert [sfx:...] cues where appropriate")


class PodcastScriptResponse(BaseModel):
    """Structured output for the podcast script agent."""
    task_status: Literal["complete", "incomplete", "error"] = Field(description="Overall task status")
    script_text: str = Field(description="Plain text script with inline bracket cues for ElevenLabs")
    metadata: Dict[str, int | float | Dict[str, int]] = Field(
        default_factory=dict,
        description="Auxiliary data: words, estimated_duration_sec, tag_counts"
    )


def get_podcast_structured_output() -> PodcastScriptResponse:
    """Convenience accessor for LangGraph structured output registration."""
    return PodcastScriptResponse























