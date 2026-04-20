"""
Oregon Trail event system.
Probability tables determine WHAT happens; the LLM narrator describes HOW it happens.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from services.oregon_trail.state import GameState, Pace, Rations
from services.oregon_trail.trail_data import AILMENTS, TRAIL_LANDMARKS


EVENT_CATEGORIES = [
    "breakdown", "illness", "weather_hazard", "animal_encounter",
    "traveler_encounter", "discovery", "theft", "oxen_trouble",
    "good_fortune", "trail_obstacle",
]

EVENT_WEIGHTS_BASE = {
    "breakdown": 8,
    "illness": 10,
    "weather_hazard": 7,
    "animal_encounter": 8,
    "traveler_encounter": 12,
    "discovery": 6,
    "theft": 4,
    "oxen_trouble": 6,
    "good_fortune": 8,
    "trail_obstacle": 10,
}


def should_event_trigger(state: GameState) -> bool:
    base_chance = 0.35
    if state.pace == Pace.GRUELING:
        base_chance += 0.10
    if state.weather in ("heavy_rain", "blizzard", "snow"):
        base_chance += 0.08
    if state.resources.food < 50:
        base_chance += 0.05
    return random.random() < base_chance


def pick_event_category(state: GameState) -> str:
    weights = dict(EVENT_WEIGHTS_BASE)
    _apply_context_weights(state, weights)
    categories = list(weights.keys())
    probs = list(weights.values())
    return random.choices(categories, weights=probs, k=1)[0]


def generate_event_effects(state: GameState, category: str) -> Dict:
    """Generate mechanical effects for an event. Narrator adds the story."""
    handler = _EVENT_HANDLERS.get(category, _handle_generic)
    return handler(state)


def _apply_context_weights(state: GameState, weights: Dict[str, int]) -> None:
    if state.resources.food < 30:
        weights["illness"] += 6
        weights["good_fortune"] += 3
    if state.resources.spare_parts == 0:
        weights["breakdown"] += 8
    if state.resources.oxen <= 2:
        weights["oxen_trouble"] += 5
    if state.terrain == "mountains":
        weights["trail_obstacle"] += 6
        weights["weather_hazard"] += 4
    if state.terrain == "desert":
        weights["illness"] += 3
        weights["animal_encounter"] -= 2
    near_fort = any(
        lm.has_fort and abs(state.miles_traveled - lm.mile_marker) < 60
        for lm in TRAIL_LANDMARKS
    )
    if near_fort:
        weights["traveler_encounter"] += 8
        weights["theft"] += 3
    for k in weights:
        weights[k] = max(1, weights[k])


def _handle_breakdown(state: GameState) -> Dict:
    if state.resources.spare_parts > 0:
        state.resources.spare_parts -= 1
        return {"type": "breakdown", "severity": "minor", "parts_used": 1,
                "time_lost_days": 0, "hint": "fixed_with_parts"}

    time_lost = random.randint(1, 3)
    for _ in range(time_lost):
        state.advance_date()
    return {"type": "breakdown", "severity": "major", "parts_used": 0,
            "time_lost_days": time_lost, "hint": "no_parts_delay"}


def _handle_illness(state: GameState) -> Dict:
    alive = state.alive_members()
    if not alive:
        return {"type": "illness", "hint": "no_alive_members"}
    victim = random.choice(alive)
    ailment = random.choice(AILMENTS[:5])
    if ailment not in victim.ailments:
        victim.ailments.append(ailment)
    return {"type": "illness", "member": victim.name, "ailment": ailment,
            "hint": "member_fell_ill"}


def _handle_weather_hazard(state: GameState) -> Dict:
    hazards = {
        "heavy_rain": ("flash_flood", random.randint(10, 40)),
        "blizzard": ("whiteout", random.randint(20, 60)),
        "snow": ("snowdrift", random.randint(5, 20)),
        "hot": ("heatstroke", random.randint(5, 15)),
    }
    hazard_type, food_lost = hazards.get(state.weather, ("windstorm", random.randint(5, 15)))
    state.resources.food = max(0, state.resources.food - food_lost)
    return {"type": "weather_hazard", "hazard": hazard_type, "food_lost": food_lost,
            "weather": state.weather, "hint": f"weather_{hazard_type}"}


def _handle_animal_encounter(state: GameState) -> Dict:
    animals = ["buffalo_herd", "wolf_pack", "rattlesnake", "bear", "deer_herd", "eagle"]
    animal = random.choice(animals)
    effects: Dict = {"type": "animal_encounter", "animal": animal}

    if animal in ("wolf_pack", "bear"):
        if state.resources.ammunition >= 3:
            state.resources.ammunition -= 3
            effects["hint"] = "scared_off"
            effects["ammo_used"] = 3
        else:
            victim = random.choice(state.alive_members()) if state.alive_members() else None
            if victim:
                damage = random.randint(10, 25)
                victim.health = max(0, victim.health - damage)
                effects["hint"] = "member_injured"
                effects["member"] = victim.name
                effects["damage"] = damage
    elif animal == "rattlesnake":
        if state.alive_members():
            victim = random.choice(state.alive_members())
            if "snakebite" not in victim.ailments:
                victim.ailments.append("snakebite")
            effects["hint"] = "snakebite"
            effects["member"] = victim.name
    elif animal in ("buffalo_herd", "deer_herd"):
        effects["hint"] = "food_opportunity"
    else:
        effects["hint"] = "scenic"

    return effects


def _handle_traveler_encounter(state: GameState) -> Dict:
    archetypes = ["fellow_emigrant", "mountain_man", "missionary", "soldier",
                  "native_guide", "trader", "lost_child", "broken_wagon_family"]
    archetype = random.choice(archetypes)
    return {"type": "traveler_encounter", "archetype": archetype,
            "hint": f"meet_{archetype}"}


def _handle_discovery(state: GameState) -> Dict:
    discoveries = [
        ("abandoned_wagon", {"food": random.randint(10, 30), "spare_parts": 1}),
        ("wild_berries", {"food": random.randint(15, 40)}),
        ("freshwater_spring", {"health_boost": random.randint(5, 15)}),
        ("carved_message", {}),
        ("gold_nugget", {"money": random.uniform(5.0, 25.0)}),
    ]
    discovery, gains = random.choice(discoveries)
    for key, val in gains.items():
        if key == "health_boost":
            for m in state.alive_members():
                m.health = min(100, m.health + val)
        elif key == "money":
            state.resources.money += val
        elif hasattr(state.resources, key):
            current = getattr(state.resources, key)
            setattr(state.resources, key, current + val)

    return {"type": "discovery", "discovery": discovery, "gains": gains,
            "hint": f"found_{discovery}"}


def _handle_theft(state: GameState) -> Dict:
    stolen_item = random.choice(["food", "ammunition", "clothing"])
    amount = random.randint(5, 25)
    current = getattr(state.resources, stolen_item, 0)
    actual = min(amount, current)
    setattr(state.resources, stolen_item, current - actual)
    return {"type": "theft", "item": stolen_item, "amount": actual,
            "hint": "theft_in_night"}


def _handle_oxen_trouble(state: GameState) -> Dict:
    if state.resources.oxen <= 0:
        return {"type": "oxen_trouble", "hint": "no_oxen"}
    issues = ["ox_lame", "ox_wandered", "ox_sick"]
    issue = random.choice(issues)
    if issue == "ox_wandered" and random.random() < 0.4:
        state.resources.oxen -= 1
        return {"type": "oxen_trouble", "issue": issue, "oxen_lost": 1,
                "hint": "ox_lost"}
    return {"type": "oxen_trouble", "issue": issue, "oxen_lost": 0,
            "hint": "ox_recovered"}


def _handle_good_fortune(state: GameState) -> Dict:
    fortunes = ["helpful_stranger", "beautiful_sunset", "party_morale_boost",
                "found_shortcut", "good_grazing"]
    fortune = random.choice(fortunes)
    effects: Dict = {"type": "good_fortune", "fortune": fortune}

    if fortune == "helpful_stranger":
        item = random.choice(["food", "ammunition", "spare_parts"])
        amount = random.randint(5, 20)
        current = getattr(state.resources, item, 0)
        setattr(state.resources, item, current + amount)
        effects["gift_item"] = item
        effects["gift_amount"] = amount
    elif fortune == "party_morale_boost":
        for m in state.alive_members():
            m.morale = min(100, m.morale + random.randint(5, 15))
    elif fortune == "found_shortcut":
        bonus_miles = random.randint(5, 15)
        state.miles_traveled += bonus_miles
        effects["bonus_miles"] = bonus_miles
    elif fortune == "good_grazing":
        for m in state.alive_members():
            m.health = min(100, m.health + random.randint(2, 6))

    effects["hint"] = f"fortune_{fortune}"
    return effects


def _handle_trail_obstacle(state: GameState) -> Dict:
    obstacles = ["fallen_tree", "mudslide", "rockslide", "washed_out_trail", "steep_hill"]
    obstacle = random.choice(obstacles)
    time_cost = random.randint(0, 2)
    for _ in range(time_cost):
        state.advance_date()
    return {"type": "trail_obstacle", "obstacle": obstacle,
            "time_lost_days": time_cost, "hint": f"obstacle_{obstacle}"}


def _handle_generic(state: GameState) -> Dict:
    return {"type": "generic", "hint": "nothing_notable"}


_EVENT_HANDLERS = {
    "breakdown": _handle_breakdown,
    "illness": _handle_illness,
    "weather_hazard": _handle_weather_hazard,
    "animal_encounter": _handle_animal_encounter,
    "traveler_encounter": _handle_traveler_encounter,
    "discovery": _handle_discovery,
    "theft": _handle_theft,
    "oxen_trouble": _handle_oxen_trouble,
    "good_fortune": _handle_good_fortune,
    "trail_obstacle": _handle_trail_obstacle,
}
