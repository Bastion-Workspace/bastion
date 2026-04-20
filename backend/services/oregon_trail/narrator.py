"""
LLM Narrator for Oregon Trail.
Generates dynamic narrative, NPC dialogue, journal entries, and adjudicates creative actions.
Uses the player's chosen model via resolve_model_context.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from services.oregon_trail.state import GameState, NPCMemory
from services.oregon_trail.trail_data import TRAIL_LANDMARKS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the narrator and Game Master for an 1848 Oregon Trail journey.
Write in vivid, period-appropriate prose. Keep responses concise (2-4 sentences for events,
3-6 sentences for scenes). Use sensory details. Reference real historical context where possible.
Never break character. The year is 1848 and the party is traveling by covered wagon.
Refer to party members by name. Be dramatic but fair."""

NPC_SYSTEM_PROMPT = """You are an NPC in an 1848 Oregon Trail game. Stay in character.
Speak in period-appropriate dialect. Be helpful but not omniscient.
Keep responses to 2-4 sentences. You may offer to trade, share advice, or tell stories."""

CREATIVE_ACTION_SYSTEM = """You are the Game Master adjudicating a player's creative action
in an 1848 Oregon Trail game. Evaluate feasibility given the historical period, current
resources, terrain, and weather. Respond ONLY with valid JSON matching this schema:
{
  "feasible": true or false,
  "difficulty": "easy" or "moderate" or "hard" or "impossible",
  "narrative": "2-3 sentence description of what happens",
  "outcome": "success" or "partial" or "failure",
  "state_changes": {
    "food_delta": 0,
    "ammo_delta": 0,
    "money_delta": 0.0,
    "health_effects": [{"member": "Name", "delta": -5}],
    "time_cost_days": 0,
    "morale_delta": 0,
    "items_gained": [],
    "items_lost": []
  }
}"""


async def _get_client_and_model(state: GameState):
    from services.model_source_resolver import resolve_model_context
    from utils.openrouter_client import get_openrouter_client

    ctx = await resolve_model_context(state.user_id, state.model_id)
    if ctx:
        client = get_openrouter_client(api_key=ctx["api_key"], base_url=ctx["base_url"])
        model = ctx.get("real_model_id", state.model_id)
    else:
        client = get_openrouter_client()
        model = state.model_id

    return client, model


def _state_summary(state: GameState) -> str:
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    alive = state.alive_members()
    members_str = ", ".join(
        f"{m.name} (health:{m.health}, morale:{m.morale}"
        + (f", ailments:{','.join(m.ailments)}" if m.ailments else "")
        + ")" for m in alive
    )
    dead = [m.name for m in state.party_members if not m.is_alive]
    dead_str = f" Dead: {', '.join(dead)}." if dead else ""
    return (
        f"Day {state.day_number}, {state.game_date}. "
        f"Location: near {lm.name} ({state.miles_traveled}/{state.total_miles} mi). "
        f"Terrain: {state.terrain}. Weather: {state.weather}. "
        f"Party: {members_str}.{dead_str} "
        f"Food: {state.resources.food} lbs, Ammo: {state.resources.ammunition}, "
        f"Parts: {state.resources.spare_parts}, Clothing: {state.resources.clothing}, "
        f"Money: ${state.resources.money:.2f}, Oxen: {state.resources.oxen}. "
        f"Pace: {state.pace.value}, Rations: {state.rations.value}."
    )


async def _llm_call(client, model: str, system: str, user_msg: str, max_tokens: int = 300) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM narrator call failed: {e}")
        return ""


async def narrate_event(state: GameState, event_effects: Dict) -> str:
    client, model = await _get_client_and_model(state)
    context = _state_summary(state)
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"An event just occurred. Category: {event_effects.get('type', 'unknown')}. "
        f"Details: {json.dumps(event_effects, default=str)}\n\n"
        "Narrate this event in 2-4 vivid sentences. Focus on what the party experiences."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt)
    return text or _fallback_event_text(event_effects)


async def narrate_scene(state: GameState, scene_type: str = "arrival") -> str:
    client, model = await _get_client_and_model(state)
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    context = _state_summary(state)
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"The party has arrived at {lm.name}. {lm.description_hint}.\n"
        f"Scene type: {scene_type}.\n\n"
        "Describe the scene in 3-5 vivid sentences. Include sensory details."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=400)
    return text or f"The party arrives at {lm.name}. {lm.description_hint}."


async def narrate_day_summary(state: GameState, day_report: Dict) -> str:
    client, model = await _get_client_and_model(state)
    context = _state_summary(state)
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"Today's travel report: {json.dumps(day_report, default=str)}\n\n"
        "Write a brief 1-2 sentence summary of the day's travel, like a journal entry."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=150)
    return text or f"Day {day_report.get('day', '?')} — traveled {day_report.get('miles_today', 0)} miles."


async def narrate_hunt(state: GameState, hunt_result: Dict) -> str:
    client, model = await _get_client_and_model(state)
    context = _state_summary(state)
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"Hunting result: {json.dumps(hunt_result, default=str)}\n\n"
        "Narrate this hunt in 2-3 sentences."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=200)
    if text:
        return text
    if hunt_result.get("success"):
        return f"The hunt was successful! Gained {hunt_result.get('food_gained', 0)} pounds of food."
    return "The hunt came up empty. The party returns to camp with nothing."


async def narrate_river_crossing(state: GameState, crossing_result: Dict) -> str:
    client, model = await _get_client_and_model(state)
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    context = _state_summary(state)
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"River crossing at {lm.name}. Method: {crossing_result.get('method')}. "
        f"Result: {json.dumps(crossing_result, default=str)}\n\n"
        "Narrate this river crossing in 2-4 sentences."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=250)
    return text or f"The party attempts to cross the river by {crossing_result.get('method', 'unknown means')}."


async def narrate_death(state: GameState, member_name: str) -> str:
    client, model = await _get_client_and_model(state)
    member = next((m for m in state.party_members if m.name == member_name), None)
    context = _state_summary(state)
    cause = ", ".join(member.ailments) if member and member.ailments else "unknown causes"
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"{member_name} has died from {cause}.\n\n"
        "Write a somber 2-3 sentence epitaph. Include the party's reaction."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=200)
    return text or f"{member_name} has died. The party mourns their loss."


async def generate_npc_dialogue(
    state: GameState, archetype: str, player_message: str, npc_memory: Optional[NPCMemory] = None
) -> Dict[str, str]:
    client, model = await _get_client_and_model(state)
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    context = _state_summary(state)
    prior = ""
    if npc_memory and npc_memory.exchanges:
        prior = "\nPrior conversation:\n" + "\n".join(npc_memory.exchanges[-4:])
    npc_system = (
        f"{NPC_SYSTEM_PROMPT}\n\nYou are a {archetype} encountered near {lm.name}. "
        f"The year is 1848.{prior}"
    )
    prompt = f"Traveler situation:\n{context}\n\nThe traveler says: \"{player_message}\""
    text = await _llm_call(client, model, npc_system, prompt, max_tokens=250)
    npc_name = npc_memory.name if npc_memory else archetype.replace("_", " ").title()
    return {"npc_name": npc_name, "dialogue": text or "The stranger nods but says nothing."}


async def adjudicate_creative_action(state: GameState, player_action: str) -> Dict[str, Any]:
    client, model = await _get_client_and_model(state)
    context = _state_summary(state)
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"The player typed: \"{player_action}\"\n\n"
        "Evaluate this action. Respond ONLY with the JSON object described in your instructions."
    )
    raw = await _llm_call(client, model, CREATIVE_ACTION_SYSTEM, prompt, max_tokens=500)
    parsed = _parse_json(raw)
    if parsed:
        _apply_creative_effects(state, parsed.get("state_changes", {}))
        return parsed
    return {
        "feasible": True, "difficulty": "moderate",
        "narrative": raw or "The party attempts the action with mixed results.",
        "outcome": "partial", "state_changes": {},
    }


async def generate_journal_entry(state: GameState, day_events: List[str]) -> str:
    client, model = await _get_client_and_model(state)
    context = _state_summary(state)
    events_text = "\n".join(f"- {e}" for e in day_events) if day_events else "An uneventful day."
    prompt = (
        f"Current situation:\n{context}\n\n"
        f"Today's events:\n{events_text}\n\n"
        "Write a brief journal entry (2-3 sentences) in first person, as the wagon train leader. "
        "Use period-appropriate language."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=200)
    return text or f"Day {state.day_number} — We pressed on."


async def generate_party_personalities(state: GameState) -> None:
    client, model = await _get_client_and_model(state)
    names = [m.name for m in state.party_members]
    prompt = (
        f"Generate brief personality descriptions (1 sentence each) for these 1848 Oregon Trail travelers: "
        f"{', '.join(names)}. The first is the leader. Make each distinct and interesting."
    )
    text = await _llm_call(client, model, SYSTEM_PROMPT, prompt, max_tokens=300)
    if text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for i, member in enumerate(state.party_members):
            if i < len(lines):
                clean = re.sub(r"^[\d.\-*]+\s*", "", lines[i])
                clean = re.sub(r"^\w+:\s*", "", clean)
                member.personality = clean.strip()


def _apply_creative_effects(state: GameState, changes: Dict) -> None:
    if not changes:
        return
    state.resources.food = max(0, state.resources.food + changes.get("food_delta", 0))
    state.resources.ammunition = max(0, state.resources.ammunition + changes.get("ammo_delta", 0))
    state.resources.money = max(0.0, state.resources.money + changes.get("money_delta", 0.0))
    for effect in changes.get("health_effects", []):
        name = effect.get("member", "")
        delta = effect.get("delta", 0)
        for m in state.alive_members():
            if m.name.lower() == name.lower():
                m.health = max(0, min(100, m.health + delta))
    morale_delta = changes.get("morale_delta", 0)
    if morale_delta:
        for m in state.alive_members():
            m.morale = max(0, min(100, m.morale + morale_delta))
    time_cost = changes.get("time_cost_days", 0)
    for _ in range(time_cost):
        state.advance_date()


def _parse_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    text = raw.strip()
    if "```json" in text:
        match = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fallback_event_text(effects: Dict) -> str:
    etype = effects.get("type", "event")
    hint = effects.get("hint", "")
    return f"A {etype.replace('_', ' ')} occurs on the trail. ({hint})"
