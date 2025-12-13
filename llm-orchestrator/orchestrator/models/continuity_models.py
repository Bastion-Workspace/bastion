"""
Pydantic models for fiction plot continuity tracking.

Tracks character states, plot threads, timeline, world state, and unresolved tensions
across chapters to ensure narrative consistency.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class CharacterState(BaseModel):
    """Current state of a character at a point in the story."""
    character_name: str = Field(description="Character name")
    chapter_number: int = Field(description="Chapter number where this state is current")
    
    # Location tracking
    location: Optional[str] = Field(
        default=None,
        description="Current physical location (e.g., 'London', 'Aboard the yacht', 'Central Park')"
    )
    
    # Emotional/mental state
    emotional_state: Optional[str] = Field(
        default=None,
        description="Current emotional state (e.g., 'anxious', 'determined', 'confused')"
    )
    
    # Knowledge tracking
    knows_about: List[str] = Field(
        default_factory=list,
        description="Key facts this character knows (e.g., 'Fleet suspects fraud', 'The ship sails tomorrow')"
    )
    
    # Relationship states
    relationships: Dict[str, str] = Field(
        default_factory=dict,
        description="Character relationships: {character_name: relationship_status}"
    )
    
    # Physical state
    injuries_or_conditions: List[str] = Field(
        default_factory=list,
        description="Physical conditions affecting character (e.g., 'sprained ankle', 'exhausted')"
    )
    
    # Possessions
    has_items: List[str] = Field(
        default_factory=list,
        description="Important items character possesses (e.g., 'mysterious letter', 'train ticket')"
    )


class PlotThread(BaseModel):
    """Active plot thread or storyline."""
    thread_id: str = Field(description="Unique identifier for this plot thread")
    thread_name: str = Field(description="Short name for this plot thread")
    description: str = Field(description="What this plot thread is about")
    
    introduced_chapter: int = Field(description="Chapter where thread was introduced")
    last_mentioned_chapter: int = Field(description="Most recent chapter mentioning this thread")
    
    status: Literal["active", "resolved", "abandoned", "background"] = Field(
        description="Current status of this plot thread"
    )
    
    key_events: List[str] = Field(
        default_factory=list,
        description="Major events in this plot thread"
    )
    
    unresolved_questions: List[str] = Field(
        default_factory=list,
        description="Questions raised by this thread that need resolution"
    )
    
    expected_resolution_chapter: Optional[int] = Field(
        default=None,
        description="Chapter where this is expected to resolve (from outline if available)"
    )


class TimeMarker(BaseModel):
    """Timestamp or time passage marker in the story."""
    chapter_number: int = Field(description="Chapter where this time marker appears")
    
    time_type: Literal["specific_time", "time_passage", "flashback", "flashforward", "time_of_day", "relative_time"] = Field(
        description="Type of time marker"
    )
    
    description: str = Field(
        description="Description of the time (e.g., 'Morning of July 15th', '3 days later', 'Flashback to childhood')"
    )
    
    # For time passage tracking
    days_elapsed: Optional[int] = Field(
        default=None,
        description="Number of days elapsed since last marker (if calculable)"
    )
    
    # For specific times
    time_of_day: Optional[str] = Field(
        default=None,
        description="Time of day if specified (e.g., 'morning', 'noon', 'midnight')"
    )


class WorldStateChange(BaseModel):
    """Significant change to world state that persists."""
    chapter_number: int = Field(description="Chapter where change occurs")
    change_type: Literal["location", "weather", "political", "magical", "technological", "social", "location_status", "character_inventory", "character_possession", "relationship"] = Field(
        description="Type of world state change"
    )
    description: str = Field(
        description="Description of the change"
    )
    affects: List[str] = Field(
        default_factory=list,
        description="Who/what this change affects (character names, locations, etc.)"
    )
    is_permanent: bool = Field(
        default=True,
        description="Whether this change persists for the rest of the story"
    )


class UnresolvedTension(BaseModel):
    """Tension, conflict, or question that hasn't been resolved."""
    tension_id: str = Field(description="Unique identifier")
    description: str = Field(description="What the tension is")
    
    introduced_chapter: int = Field(description="Chapter where introduced")
    last_escalated_chapter: int = Field(description="Most recent escalation")
    
    tension_type: Literal["conflict", "mystery", "relationship", "internal", "external", "external_threat", "character_conflict"] = Field(
        description="Type of tension"
    )
    
    involves_characters: List[str] = Field(
        default_factory=list,
        description="Characters involved in this tension"
    )
    
    stakes: Optional[str] = Field(
        default=None,
        description="What's at stake if this tension isn't resolved"
    )


class ContinuityState(BaseModel):
    """Complete continuity state for a manuscript."""
    manuscript_filename: str = Field(description="Filename this state belongs to")
    user_id: str = Field(description="User ID owning this manuscript")
    
    last_analyzed_chapter: int = Field(
        default=0,
        description="Last chapter number that was analyzed"
    )
    
    last_updated: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO timestamp of last update"
    )
    
    # Core tracking components
    character_states: Dict[str, CharacterState] = Field(
        default_factory=dict,
        description="Current state of each character: {character_name: CharacterState}"
    )
    
    plot_threads: Dict[str, PlotThread] = Field(
        default_factory=dict,
        description="Active plot threads: {thread_id: PlotThread}"
    )
    
    timeline: List[TimeMarker] = Field(
        default_factory=list,
        description="Chronological list of time markers"
    )
    
    world_state_changes: List[WorldStateChange] = Field(
        default_factory=list,
        description="Significant changes to world state"
    )
    
    unresolved_tensions: Dict[str, UnresolvedTension] = Field(
        default_factory=dict,
        description="Active tensions: {tension_id: UnresolvedTension}"
    )
    
    # Summary for quick reference
    current_chapter_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of current story state"
    )


class ContinuityViolation(BaseModel):
    """Detected violation of continuity."""
    violation_type: Literal[
        "character_location",
        "character_knowledge", 
        "timeline",
        "character_state",
        "plot_thread",
        "world_state",
        "relationship"
    ] = Field(description="Type of continuity violation")
    
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="How serious this violation is"
    )
    
    description: str = Field(description="What the violation is")
    
    expected: str = Field(description="What continuity state says should be")
    found: str = Field(description="What the new content says")
    
    affected_character: Optional[str] = Field(
        default=None,
        description="Character involved in violation (if applicable)"
    )
    
    suggestion: str = Field(description="How to fix this violation")


class ContinuityValidationResult(BaseModel):
    """Result of validating new content against continuity."""
    is_valid: bool = Field(description="Whether content passes continuity checks")
    
    violations: List[ContinuityViolation] = Field(
        default_factory=list,
        description="Any violations detected"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-critical continuity concerns"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in validation (0.0-1.0)"
    )

