"""
Wargaming Models - Roosevelt's Strategic Simulation Structures

Defines structured outputs for the Wargaming Agent.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from enum import Enum
class Likelihood(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"


class EscalationStance(str, Enum):
    DE_ESCALATE = "de-escalate"
    MAINTAIN = "maintain"
    ESCALATE = "escalate"
    FULL_WAR = "full_war"
    SURRENDER = "surrender"


class ThirdPartyActorResponse(BaseModel):
    actor: str = Field(description="Third-party actor name, e.g., 'France', 'Germany', 'UK', 'EU' ")
    stance: EscalationStance = Field(description="Current stance of the actor with respect to this turn")
    likelihood: Likelihood = Field(description="Likelihood of this response or participation")
    trigger_conditions: List[str] = Field(default_factory=list, description="Conditions that would trigger this actor's response")
    planned_actions: List[str] = Field(default_factory=list, description="Likely actions the actor would take under triggers")


class Domain(str, Enum):
    LAND = "land"
    AIR = "air"
    SEA = "sea"
    CYBER = "cyber"
    SPACE = "space"
    INFORMATION = "information"


class LeaderStatus(str, Enum):
    ALIVE = "alive"
    WOUNDED = "wounded"
    DEAD = "dead"
    MISSING = "missing"
    CAPTURED = "captured"
    UNKNOWN = "unknown"


class LeaderStatusEntry(BaseModel):
    country: str = Field(description="Country name")
    leader_name: str = Field(description="Leader full name")
    role: str = Field(description="Role/title (e.g., Head of State, PM, Defense Minister)")
    status: LeaderStatus = Field(description="Continuity-of-government status")
    last_seen: Optional[str] = Field(default=None, description="ISO timestamp or relative marker for last confirmed status")
    location: Optional[str] = Field(default=None, description="Last known location or region")
    cog_active: Optional[bool] = Field(default=None, description="Whether continuity-of-government protocols are active for this country")
    successor_chain: List[str] = Field(default_factory=list, description="Ordered list of successors or acting officials")


class CapabilityStatus(BaseModel):
    c2: str = Field(description="Command and control status")
    leadership: str = Field(description="Leadership status")
    comms: str = Field(description="Communications status")
    sortie_rate_modifier: float = Field(ge=0.0, le=1.0, description="Sortie rate modifier (0-1)")
    mobilization_delay_minutes: int = Field(ge=0, description="Mobilization delay in minutes")


class ActorEntry(BaseModel):
    name: str = Field(description="Actor or unit name (e.g., 'US 2nd ABCT')")
    side: str = Field(description="Side allegiance (e.g., US, Belgium, France)")
    stance: EscalationStance = Field(description="Actor stance this turn")
    capability_status: CapabilityStatus = Field(description="Current capability status for constraints")
    thresholds: List[str] = Field(default_factory=list, description="Trigger thresholds (e.g., redlines)")


class CasualtyEstimates(BaseModel):
    kia_range: Optional[str] = Field(default=None, description="Killed-in-action range (e.g., '8–12')")
    wia_range: Optional[str] = Field(default=None, description="Wounded-in-action range (e.g., '15–20')")
    civilian_kia_range: Optional[str] = Field(default=None, description="Civilian killed-in-action range (e.g., '0–2')")


class DamageEntry(BaseModel):
    category: str = Field(description="Category (equipment, infrastructure, logistics, etc.)")
    count_range: Optional[str] = Field(default=None, description="Count range (e.g., '2–3')")
    area: Optional[str] = Field(default=None, description="Area or location of damage")


class COAEntry(BaseModel):
    description: str = Field(description="Course of action summary")
    probability: Likelihood = Field(description="Estimated probability")
    risks: List[str] = Field(default_factory=list, description="Key risks")
    legal: Optional[str] = Field(default=None, description="Legal considerations")
    resources: List[str] = Field(default_factory=list, description="Key resources required")
    timing: Optional[str] = Field(default=None, description="Timing notes or constraints")


class TimelineEvent(BaseModel):
    t_plus_minutes: int = Field(ge=0, description="Time relative to start of turn in minutes")
    domain: Domain = Field(description="Operating domain")
    action: str = Field(description="Event action description")
    expected_effects: Optional[str] = Field(default=None, description="Expected effects from the action")


class EscalationDoctrine(BaseModel):
    ladder: str = Field(description="Named escalation ladder (e.g., 'Kahn')")
    current_rung: int = Field(ge=0, le=10, description="Current rung on the ladder")
    previous_rung: Optional[int] = Field(default=None, description="Previous rung")
    movement_rule_applied: Optional[str] = Field(default=None, description="Rule used to move along the ladder")


class ImpactEntry(BaseModel):
    category: str = Field(description="Impact category (diplomatic_relations, alliance_credibility, transatlantic_partnership, etc.)")
    metric: str = Field(description="Quantified metric or index affected (e.g., NATO cohesion index)")
    range: Optional[str] = Field(default=None, description="Change range (e.g., '-15% to -25%' or '8–12 diplomats expelled')")
    timeframe: Optional[str] = Field(default=None, description="Timeframe for the impact (e.g., '24–72h')")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the impact assessment (0-1)")
    levers: List[str] = Field(default_factory=list, description="Concrete levers/actions causing the impact (expulsions, sanctions, votes, suspensions, filings)")


class AdviceEntry(BaseModel):
    question: Optional[str] = Field(default=None, description="The user question being answered (if detectable)")
    recommendation: str = Field(description="Primary actionable recommendation")
    rationale: Optional[str] = Field(default=None, description="Why this recommendation fits the situation")
    risks: List[str] = Field(default_factory=list, description="Key risks of the recommendation")
    legal: Optional[str] = Field(default=None, description="Legal considerations relevant to this advice")
    alternatives: List[str] = Field(default_factory=list, description="Viable alternatives if primary is rejected")
    timeframe: Optional[str] = Field(default=None, description="Timeframe for execution or effect")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the advice (0-1)")

class ClarificationEntry(BaseModel):
    question: str = Field(description="User clarification question")
    answer: str = Field(description="Direct, structured answer")
    sources: List[str] = Field(default_factory=list, description="Internal state references or citations")
    scope: Literal["troops", "leaders", "nation", "oob", "events", "impacts"] = Field(description="Clarification domain")
    gaps: List[str] = Field(default_factory=list, description="What is unknown or uncertain")
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested next questions or data to fetch")


class WargamingResponse(BaseModel):
    """Structured output for wargaming simulation turns."""
    task_status: TaskStatus = Field(description="Task status for this turn")
    turn_time: Optional[str] = Field(default=None, description="ISO timestamp for the turn's effective time or t+delta marker")
    turn_summary: str = Field(description="Narrative summary of the agent's response and effects")
    stance: EscalationStance = Field(description="Strategic stance chosen for this turn")
    actors: List[ActorEntry] = Field(default_factory=list, description="Per-actor status and thresholds")
    actions_taken: List[str] = Field(default_factory=list, description="Concrete actions taken this turn")
    actions_by_actor: Dict[str, List[str]] = Field(default_factory=dict, description="Actions grouped by actor key (e.g., 'US', 'Belgium', 'France')")
    coas: List[COAEntry] = Field(default_factory=list, description="Alternative COAs (2–3 branches minimum)")
    next_options: List[str] = Field(default_factory=list, description="Top next actions with brief rationale")
    risk_assessment: str = Field(description="Risks and potential blowback")
    international_law_considerations: Optional[str] = Field(default=None, description="Relevant IL/LOAC/diplomatic norms considerations")
    escalation_ladder_position: int = Field(ge=0, le=10, default=3, description="Position on escalation ladder (0-10)")
    confidence_level: float = Field(ge=0.0, le=1.0, default=0.8, description="Confidence in assessment (0-1)")
    third_party_responses: List[ThirdPartyActorResponse] = Field(default_factory=list, description="Per-actor structured third-party modeling with stance/likelihood/triggers/actions")
    casualty_estimates: Optional[CasualtyEstimates] = Field(default=None, description="Structured casualty estimates")
    damage: List[DamageEntry] = Field(default_factory=list, description="Structured damage entries")
    bda_by_actor: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-actor BDA with keys like 'casualties' (str) and 'material' (List[str])")
    weapons_consistency_check: Optional[str] = Field(default=None, description="Validation notes that weapons/platforms align with actors and context")
    oob_delta: Dict[str, List[str]] = Field(default_factory=dict, description="Order-of-battle changes per actor (e.g., '2nd ABCT combat ineffective', '2x M777 destroyed')")
    events: List[TimelineEvent] = Field(default_factory=list, description="Time-stepped timeline events for this turn")
    escalation_doctrine: Optional[EscalationDoctrine] = Field(default=None, description="Escalation ladder mapping and movement rule applied")
    impacts: List[ImpactEntry] = Field(default_factory=list, description="Structured strategic impacts with metrics, ranges, timeframe, confidence, and levers")
    advice: List[AdviceEntry] = Field(default_factory=list, description="On-demand advisory responses without advancing the turn")
    advance_turn: bool = Field(default=True, description="If false, do not advance state/history for this turn")
    leaders: List[LeaderStatusEntry] = Field(default_factory=list, description="Continuity-of-government leader statuses for involved countries")
    clarifications: List[ClarificationEntry] = Field(default_factory=list, description="On-demand clarifications without advancing the turn")


def get_wargaming_structured_output() -> WargamingResponse:
    return WargamingResponse


