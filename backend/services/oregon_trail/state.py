"""
Oregon Trail game state models.
All game state is serializable to/from JSON for PostgreSQL JSONB persistence.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Profession(str, Enum):
    BANKER = "banker"
    CARPENTER = "carpenter"
    FARMER = "farmer"


class Pace(str, Enum):
    STEADY = "steady"
    STRENUOUS = "strenuous"
    GRUELING = "grueling"


class Rations(str, Enum):
    FILLING = "filling"
    MEAGER = "meager"
    BARE_BONES = "bare_bones"


class GamePhase(str, Enum):
    SETUP_PARTY = "setup_party"
    SETUP_SUPPLIES = "setup_supplies"
    TRAVELING = "traveling"
    LANDMARK = "landmark"
    RIVER_CROSSING = "river_crossing"
    FORT = "fort"
    HUNTING = "hunting"
    TRADING = "trading"
    GAME_OVER = "game_over"
    VICTORY = "victory"


class HealthStatus(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"
    DEAD = "dead"


class PartyMember(BaseModel):
    name: str
    health: int = Field(default=100, ge=0, le=100)
    morale: int = Field(default=75, ge=0, le=100)
    ailments: List[str] = Field(default_factory=list)
    is_alive: bool = True
    personality: str = ""

    @property
    def status(self) -> HealthStatus:
        if not self.is_alive:
            return HealthStatus.DEAD
        if self.health >= 70:
            return HealthStatus.GOOD
        if self.health >= 40:
            return HealthStatus.FAIR
        if self.health >= 15:
            return HealthStatus.POOR
        return HealthStatus.VERY_POOR


class Resources(BaseModel):
    food: int = 0
    ammunition: int = 0
    spare_parts: int = 0
    clothing: int = 0
    money: float = 0.0
    oxen: int = 4


class JournalEntry(BaseModel):
    day: int
    game_date: str
    text: str
    location: str


class GameEvent(BaseModel):
    day: int
    category: str
    description: str
    outcome: str


class NPCMemory(BaseModel):
    npc_id: str
    name: str
    archetype: str
    location: str
    exchanges: List[str] = Field(default_factory=list)


class ActionChoice(BaseModel):
    key: str
    label: str
    description: str = ""


class GameState(BaseModel):
    game_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    model_id: str = ""

    leader_name: str = ""
    profession: Profession = Profession.BANKER
    party_members: List[PartyMember] = Field(default_factory=list)

    resources: Resources = Field(default_factory=Resources)

    miles_traveled: int = 0
    total_miles: int = 2170
    current_landmark_idx: int = 0
    day_number: int = 1
    game_date: str = "1848-04-01"
    weather: str = "clear"
    temperature: str = "mild"
    terrain: str = "prairie"
    pace: Pace = Pace.STEADY
    rations: Rations = Rations.FILLING

    journal: List[JournalEntry] = Field(default_factory=list)
    event_history: List[GameEvent] = Field(default_factory=list)
    npc_memory: List[NPCMemory] = Field(default_factory=list)

    phase: GamePhase = GamePhase.SETUP_PARTY
    pending_narrative: str = ""
    available_actions: List[ActionChoice] = Field(default_factory=list)
    is_finished: bool = False
    final_score: Optional[int] = None

    def advance_date(self, days: int = 1) -> str:
        d = date.fromisoformat(self.game_date) + timedelta(days=days)
        self.game_date = d.isoformat()
        self.day_number += days
        return self.game_date

    def alive_members(self) -> List[PartyMember]:
        return [m for m in self.party_members if m.is_alive]

    def party_health_summary(self) -> str:
        parts = []
        for m in self.party_members:
            tag = m.status.value.replace("_", " ").title() if m.is_alive else "Dead"
            parts.append(f"{m.name}({tag})")
        return "  ".join(parts)


STARTING_MONEY: Dict[Profession, float] = {
    Profession.BANKER: 1600.0,
    Profession.CARPENTER: 800.0,
    Profession.FARMER: 400.0,
}

SCORE_MULTIPLIER: Dict[Profession, float] = {
    Profession.BANKER: 1.0,
    Profession.CARPENTER: 2.0,
    Profession.FARMER: 3.0,
}
