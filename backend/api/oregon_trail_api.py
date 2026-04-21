"""
Oregon Trail REST API.
Game lifecycle: create, load, save, action dispatch, journal, NPC talk.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from models.api_models import AuthenticatedUserResponse
from services.model_source_resolver import get_enabled_models
from services.oregon_trail import engine, events, narrator
from services.oregon_trail.state import (
    ActionChoice,
    GameEvent,
    GamePhase,
    GameState,
    JournalEntry,
    NPCMemory,
)
from services.oregon_trail.trail_data import TRAIL_LANDMARKS
from services.service_container import service_container
from utils.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/games/oregon-trail", tags=["oregon-trail"])

_NOT_FOUND_DETAIL = (
    "No saved game for this id and account. If you just started, confirm migration "
    "129 (oregon_trail_saves) ran and the backend database pool is initialized."
)


def _uid(user_id: Any) -> str:
    """Normalize user id from JWT / DB (str or UUID) for asyncpg."""
    if user_id is None:
        return ""
    if isinstance(user_id, uuid.UUID):
        return str(user_id)
    return str(user_id).strip()


def _parse_game_id(game_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(game_id).strip())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid game_id (expected UUID)")


def _pool_usable(pool) -> bool:
    if pool is None:
        return False
    is_closing = getattr(pool, "is_closing", None)
    if callable(is_closing):
        try:
            return not bool(is_closing())
        except Exception:
            return True
    return True


async def _require_db_pool():
    """
    Resolve the asyncpg pool used elsewhere in the backend.
    service_container.db_pool is normally set during lifespan init; if it is
    still None (race, partial init, or stale import), fall back to
    document_repository.pool, lazy container init, or the global shared pool.
    """
    sc = service_container

    if _pool_usable(sc.db_pool):
        return sc.db_pool

    dr = sc.document_repository
    if dr is not None and _pool_usable(getattr(dr, "pool", None)):
        sc.db_pool = dr.pool
        logger.info("Oregon Trail: using document_repository.pool (service_container.db_pool was unset)")
        return sc.db_pool

    try:
        from services.service_container import get_service_container

        c = await get_service_container()
        if _pool_usable(c.db_pool):
            if sc.db_pool is None:
                sc.db_pool = c.db_pool
            return c.db_pool
        dr2 = c.document_repository
        if dr2 is not None and _pool_usable(getattr(dr2, "pool", None)):
            c.db_pool = dr2.pool
            if sc.db_pool is None:
                sc.db_pool = c.db_pool
            logger.info("Oregon Trail: pool bound after get_service_container()")
            return c.db_pool
    except Exception as e:
        logger.warning("Oregon Trail: get_service_container() failed: %s", e)

    try:
        from utils.shared_db_pool import get_shared_db_pool

        pool = await get_shared_db_pool()
        if _pool_usable(pool):
            logger.info("Oregon Trail: using utils.shared_db_pool fallback")
            return pool
    except Exception as e:
        logger.error("Oregon Trail: get_shared_db_pool() failed: %s", e)

    logger.error("Oregon Trail: no database pool available after all fallbacks")
    raise HTTPException(
        status_code=503,
        detail="Database is not available; Oregon Trail cannot load or save games.",
    )


# ---------- request / response models ----------

class NewGameRequest(BaseModel):
    leader_name: str
    party_names: List[str] = Field(default_factory=list, max_length=3)
    profession: str = "banker"
    model_id: str = ""


class ActionRequest(BaseModel):
    action: str
    detail: Optional[str] = None
    quantity: Optional[int] = None


class TalkRequest(BaseModel):
    message: str


class GameSummary(BaseModel):
    game_id: str
    leader_name: str
    miles_traveled: int
    day_number: int
    phase: str
    is_finished: bool
    final_score: Optional[int] = None
    updated_at: Optional[str] = None


# ---------- persistence helpers ----------

async def _save_game(state: GameState) -> None:
    pool = await _require_db_pool()
    data = state.model_dump(mode="json")
    uid = _uid(state.user_id)
    await pool.execute(
        """
        INSERT INTO oregon_trail_saves (id, user_id, game_state, is_active, final_score, updated_at)
        VALUES ($1::uuid, $2::varchar, $3::jsonb, $4, $5, NOW())
        ON CONFLICT (id) DO UPDATE SET
            game_state = EXCLUDED.game_state,
            is_active = EXCLUDED.is_active,
            final_score = EXCLUDED.final_score,
            updated_at = NOW()
        WHERE oregon_trail_saves.user_id = EXCLUDED.user_id
        """,
        uuid.UUID(state.game_id),
        uid,
        json.dumps(data),
        not state.is_finished,
        state.final_score,
    )


async def _load_game(game_id: str, user_id: str) -> Optional[GameState]:
    pool = await _require_db_pool()
    gid = _parse_game_id(game_id)
    uid = _uid(user_id)
    row = await pool.fetchrow(
        "SELECT game_state FROM oregon_trail_saves WHERE id = $1::uuid AND user_id = $2::varchar",
        gid,
        uid,
    )
    if not row:
        return None
    return GameState(**json.loads(row["game_state"]))


async def _list_saves(user_id: str) -> List[Dict[str, Any]]:
    pool = await _require_db_pool()
    rows = await pool.fetch(
        """
        SELECT id, game_state, is_active, final_score, updated_at
        FROM oregon_trail_saves WHERE user_id = $1::varchar
        ORDER BY updated_at DESC LIMIT 20
        """,
        _uid(user_id),
    )
    results = []
    for r in rows:
        gs = json.loads(r["game_state"])
        results.append({
            "game_id": str(r["id"]),
            "leader_name": gs.get("leader_name", ""),
            "miles_traveled": gs.get("miles_traveled", 0),
            "day_number": gs.get("day_number", 1),
            "phase": gs.get("phase", ""),
            "is_finished": gs.get("is_finished", False),
            "final_score": r["final_score"],
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
        })
    return results


# ---------- endpoints ----------

@router.get("/models")
async def get_available_game_models(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    models = await get_enabled_models(_uid(current_user.user_id))
    return {"models": models}


@router.post("/new")
async def new_game(
    req: NewGameRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    if not req.leader_name.strip():
        raise HTTPException(400, "Leader name is required")
    uid = _uid(current_user.user_id)
    if not req.model_id:
        models = await get_enabled_models(uid)
        if not models:
            raise HTTPException(400, "No LLM models available. Configure models in Settings first.")
        req.model_id = models[0]

    state = engine.create_new_game(
        user_id=uid,
        model_id=req.model_id,
        leader_name=req.leader_name.strip(),
        party_names=[n.strip() for n in req.party_names if n.strip()],
        profession=req.profession,
    )
    await narrator.generate_party_personalities(state)
    welcome = "Welcome to the Oregon Trail! Stock up on supplies before you depart."
    state.pending_narrative = welcome
    await _save_game(state)
    return _game_response(state, narrative=welcome)


@router.get("/saves")
async def list_saves(
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    saves = await _list_saves(_uid(current_user.user_id))
    return {"saves": saves}


# Sub-routes under /{game_id}/... MUST be declared before GET /{game_id} so the
# catch-all path parameter does not take precedence (Starlette matches in order).


@router.post("/{game_id}/action")
async def perform_action(
    game_id: str,
    req: ActionRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    state = await _load_game(game_id, _uid(current_user.user_id))
    if not state:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)
    if state.is_finished:
        raise HTTPException(400, "Game is already finished")

    narrative_parts: List[str] = []
    action = req.action.lower().strip()

    if state.phase == GamePhase.SETUP_SUPPLIES:
        narrative_parts.append(await _handle_setup_action(state, action, req))
    elif state.phase == GamePhase.TRAVELING:
        narrative_parts.append(await _handle_travel_action(state, action, req))
    elif state.phase == GamePhase.RIVER_CROSSING:
        narrative_parts.append(await _handle_river_action(state, action))
    elif state.phase in (GamePhase.FORT, GamePhase.LANDMARK):
        narrative_parts.append(await _handle_location_action(state, action, req))
    else:
        if req.detail:
            result = await narrator.adjudicate_creative_action(state, req.detail)
            narrative_parts.append(result.get("narrative", "Nothing happens."))
        else:
            narrative_parts.append("What would you like to do?")

    combined = "\n\n".join(p for p in narrative_parts if p)
    if combined:
        state.pending_narrative = combined
    await _save_game(state)
    return _game_response(state, narrative=combined or state.pending_narrative)


@router.post("/{game_id}/talk")
async def talk_to_npc(
    game_id: str,
    req: TalkRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    state = await _load_game(game_id, _uid(current_user.user_id))
    if not state:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)

    archetype = "fellow_emigrant"
    for evt in reversed(state.event_history):
        if evt.category == "traveler_encounter":
            archetype = evt.outcome
            break

    existing = next((n for n in state.npc_memory if n.archetype == archetype and
                     n.location == TRAIL_LANDMARKS[state.current_landmark_idx].name), None)
    if not existing:
        npc_id = str(uuid.uuid4())[:8]
        existing = NPCMemory(
            npc_id=npc_id, name=archetype.replace("_", " ").title(),
            archetype=archetype,
            location=TRAIL_LANDMARKS[state.current_landmark_idx].name,
        )
        state.npc_memory.append(existing)

    result = await narrator.generate_npc_dialogue(state, archetype, req.message, existing)
    existing.exchanges.append(f"You: {req.message}")
    existing.exchanges.append(f"{result['npc_name']}: {result['dialogue']}")

    talk_line = f"{result['npc_name']}: {result['dialogue']}"
    if state.pending_narrative:
        state.pending_narrative = f"{state.pending_narrative}\n\n{talk_line}"
    else:
        state.pending_narrative = talk_line

    await _save_game(state)
    return {"npc_name": result["npc_name"], "dialogue": result["dialogue"],
            **_game_response(state)}


@router.get("/{game_id}/journal")
async def get_journal(
    game_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    state = await _load_game(game_id, _uid(current_user.user_id))
    if not state:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)
    return {"journal": [e.model_dump() for e in state.journal]}


@router.delete("/{game_id}")
async def delete_game(
    game_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    pool = await _require_db_pool()
    gid = _parse_game_id(game_id)
    await pool.execute(
        "DELETE FROM oregon_trail_saves WHERE id = $1::uuid AND user_id = $2::varchar",
        gid,
        _uid(current_user.user_id),
    )
    return {"deleted": True}


@router.get("/{game_id}")
async def get_game_state(
    game_id: str,
    current_user: AuthenticatedUserResponse = Depends(get_current_user),
):
    state = await _load_game(game_id, _uid(current_user.user_id))
    if not state:
        raise HTTPException(status_code=404, detail=_NOT_FOUND_DETAIL)
    return _game_response(state)


# ---------- action handlers ----------

async def _handle_setup_action(state: GameState, action: str, req: ActionRequest) -> str:
    item_map = {"buy_food": "food", "buy_ammo": "ammunition", "buy_clothing": "clothing",
                "buy_parts": "spare_parts", "buy_oxen": "oxen"}
    if action in item_map:
        qty = req.quantity or {"food": 100, "ammunition": 20, "clothing": 2,
                               "spare_parts": 2, "oxen": 1}.get(item_map[action], 1)
        ok, msg = engine.buy_supplies(state, item_map[action], qty)
        engine._set_supply_shop_actions(state)
        return msg
    if action == "done":
        engine.finish_shopping(state)
        scene = await narrator.narrate_scene(state, scene_type="departure")
        return scene or "The wagon lurches forward. The Oregon Trail awaits."
    return "Choose an item to buy or hit the trail."


async def _handle_travel_action(state: GameState, action: str, req: ActionRequest) -> str:
    if action == "travel":
        return await _do_travel_day(state)
    if action == "rest":
        msg = engine.rest_at_location(state, days=1)
        engine._set_travel_actions(state)
        return msg
    if action == "hunt":
        result = engine.attempt_hunt(state)
        text = await narrator.narrate_hunt(state, result)
        engine._set_travel_actions(state)
        return text
    if action == "pace":
        pace_val = (req.detail or "steady").lower()
        if pace_val in ("steady", "strenuous", "grueling"):
            msg = engine.set_pace(state, pace_val)
            engine._set_travel_actions(state)
            return msg
        return "Choose: steady, strenuous, or grueling."
    if action == "rations":
        ration_val = (req.detail or "filling").lower()
        if ration_val in ("filling", "meager", "bare_bones"):
            msg = engine.set_rations(state, ration_val)
            engine._set_travel_actions(state)
            return msg
        return "Choose: filling, meager, or bare_bones."
    if action == "journal":
        return "journal_view"
    if action == "status":
        return _status_text(state)
    if action == "custom" and req.detail:
        result = await narrator.adjudicate_creative_action(state, req.detail)
        engine._set_travel_actions(state)
        return result.get("narrative", "Nothing happens.")
    return "What would you like to do?"


async def _handle_river_action(state: GameState, action: str) -> str:
    if action in ("ford", "caulk", "ferry", "wait"):
        result = engine.attempt_river_crossing(state, action)
        text = await narrator.narrate_river_crossing(state, result)
        if result.get("success"):
            engine.continue_from_landmark(state)
        return text
    return "Choose: ford, caulk, ferry, or wait."


async def _handle_location_action(state: GameState, action: str, req: ActionRequest) -> str:
    if action == "continue":
        engine.continue_from_landmark(state)
        return "The party presses onward."
    if action == "rest":
        days = min(req.quantity or 1, 5)
        msg = engine.rest_at_location(state, days=days)
        return msg
    if action == "hunt":
        result = engine.attempt_hunt(state)
        return await narrator.narrate_hunt(state, result)
    if action == "trade" and state.phase == GamePhase.FORT:
        if req.detail and req.quantity:
            ok, msg = engine.trade_at_fort(state, req.detail, req.quantity)
            return msg
        return "Specify item and quantity to trade. e.g. action=trade, detail=food, quantity=50"
    if action == "talk":
        return "Use the /talk endpoint to have a conversation."
    if action == "custom" and req.detail:
        result = await narrator.adjudicate_creative_action(state, req.detail)
        return result.get("narrative", "Nothing happens.")
    return "What would you like to do?"


async def _do_travel_day(state: GameState) -> str:
    day_report = engine.advance_day(state)
    parts: List[str] = []
    day_events: List[str] = []

    if events.should_event_trigger(state):
        category = events.pick_event_category(state)
        effects = events.generate_event_effects(state, category)
        event_text = await narrator.narrate_event(state, effects)
        parts.append(event_text)
        day_events.append(event_text)
        state.event_history.append(GameEvent(
            day=state.day_number, category=category,
            description=event_text[:200], outcome=effects.get("hint", ""),
        ))

    for death_name in day_report.get("deaths", []):
        death_text = await narrator.narrate_death(state, death_name)
        parts.append(death_text)
        day_events.append(f"{death_name} died")

    if day_report.get("landmark"):
        scene = await narrator.narrate_scene(state, scene_type="arrival")
        parts.append(scene)
        day_events.append(f"Arrived at {day_report['landmark']}")

    if day_report.get("victory"):
        parts.append(f"You've reached Oregon! Final score: {day_report.get('score', 0)}")
    elif day_report.get("game_over"):
        parts.append("Your entire party has perished. The trail claims another wagon.")

    summary = await narrator.narrate_day_summary(state, day_report)
    if not parts:
        parts.append(summary)
    day_events.append(summary)

    journal_text = await narrator.generate_journal_entry(state, day_events)
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    state.journal.append(JournalEntry(
        day=state.day_number, game_date=state.game_date,
        text=journal_text, location=lm.name,
    ))

    if not state.is_finished:
        engine._set_travel_actions(state)

    return "\n\n".join(parts)


# ---------- response helpers ----------

def _game_response(state: GameState, narrative: str = "") -> Dict[str, Any]:
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    return {
        "game_id": state.game_id,
        "phase": state.phase.value,
        "day_number": state.day_number,
        "game_date": state.game_date,
        "location": lm.name,
        "miles_traveled": state.miles_traveled,
        "total_miles": state.total_miles,
        "weather": state.weather,
        "terrain": state.terrain,
        "pace": state.pace.value,
        "rations": state.rations.value,
        "party": [
            {"name": m.name, "health": m.health, "morale": m.morale,
             "status": m.status.value, "ailments": m.ailments,
             "is_alive": m.is_alive, "personality": m.personality}
            for m in state.party_members
        ],
        "resources": state.resources.model_dump(),
        "available_actions": [a.model_dump() for a in state.available_actions],
        "narrative": narrative or state.pending_narrative,
        "is_finished": state.is_finished,
        "final_score": state.final_score,
        "model_id": state.model_id,
    }


def _status_text(state: GameState) -> str:
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    r = state.resources
    lines = [
        f"Day {state.day_number} — {state.game_date}",
        f"Location: {lm.name} ({state.miles_traveled}/{state.total_miles} miles)",
        f"Weather: {state.weather} | Terrain: {state.terrain}",
        f"Pace: {state.pace.value} | Rations: {state.rations.value.replace('_', ' ')}",
        "",
        "Party:",
    ]
    for m in state.party_members:
        tag = m.status.value.replace("_", " ").title() if m.is_alive else "Dead"
        ail = f" [{', '.join(m.ailments)}]" if m.ailments else ""
        lines.append(f"  {m.name}: {tag}{ail}")
    lines.append("")
    lines.append(f"Food: {r.food} lbs | Ammo: {r.ammunition} | Parts: {r.spare_parts}")
    lines.append(f"Clothing: {r.clothing} | Oxen: {r.oxen} | Money: ${r.money:.2f}")
    return "\n".join(lines)
