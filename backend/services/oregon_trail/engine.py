"""
Oregon Trail deterministic game engine.
Handles travel, health, food, weather, trading, hunting, and river crossings.
"""

from __future__ import annotations

import logging
import random
from datetime import date
from typing import Dict, List, Optional, Tuple

from services.oregon_trail.state import (
    ActionChoice,
    GamePhase,
    GameState,
    Pace,
    PartyMember,
    Profession,
    Rations,
    Resources,
    SCORE_MULTIPLIER,
    STARTING_MONEY,
)
from services.oregon_trail.trail_data import (
    AILMENT_DAILY_DAMAGE,
    AILMENT_RECOVERY_CHANCE,
    AILMENTS,
    FORT_PRICE_MULTIPLIERS,
    MONTH_WEATHER_WEIGHTS,
    PACE_BASE_MILES,
    PACE_HEALTH_PENALTY,
    RATIONS_FOOD_PER_PERSON,
    RATIONS_HEALTH_BONUS,
    SHOP_PRICES,
    TERRAIN_SPEED_MODIFIER,
    TRAIL_LANDMARKS,
    WEATHER_SPEED_MODIFIER,
    Landmark,
)

logger = logging.getLogger(__name__)


def create_new_game(
    user_id: str,
    model_id: str,
    leader_name: str,
    party_names: List[str],
    profession: str,
) -> GameState:
    prof = Profession(profession)
    members = [PartyMember(name=leader_name)]
    for n in party_names[:3]:
        members.append(PartyMember(name=n))

    state = GameState(
        user_id=user_id,
        model_id=model_id,
        leader_name=leader_name,
        profession=prof,
        party_members=members,
        resources=Resources(money=STARTING_MONEY[prof]),
        phase=GamePhase.SETUP_SUPPLIES,
    )
    _set_supply_shop_actions(state)
    return state


def buy_supplies(state: GameState, item: str, quantity: int) -> Tuple[bool, str]:
    price_per = SHOP_PRICES.get(item, 0)
    if price_per == 0:
        return False, f"Unknown item: {item}"

    total_cost = price_per * quantity
    if total_cost > state.resources.money:
        return False, f"Not enough money. Need ${total_cost:.2f}, have ${state.resources.money:.2f}"

    state.resources.money -= total_cost
    if item == "food":
        state.resources.food += quantity
    elif item == "ammunition":
        state.resources.ammunition += quantity
    elif item == "clothing":
        state.resources.clothing += quantity
    elif item == "spare_parts":
        state.resources.spare_parts += quantity
    elif item == "oxen":
        state.resources.oxen += quantity

    return True, f"Bought {quantity} {item} for ${total_cost:.2f}. Remaining: ${state.resources.money:.2f}"


def finish_shopping(state: GameState) -> None:
    state.phase = GamePhase.TRAVELING
    _set_travel_actions(state)


def set_pace(state: GameState, pace: str) -> str:
    state.pace = Pace(pace)
    return f"Pace set to {pace}."


def set_rations(state: GameState, rations: str) -> str:
    state.rations = Rations(rations)
    return f"Rations set to {rations.replace('_', ' ')}."


def advance_day(state: GameState) -> Dict:
    """Advance one day: travel, consume food, update weather/health. Returns day report dict."""
    report: Dict = {"day": state.day_number, "events": []}

    _update_weather(state)
    report["weather"] = state.weather

    miles = _calculate_miles(state)
    state.miles_traveled += miles
    report["miles_today"] = miles
    report["miles_total"] = state.miles_traveled

    food_consumed = _consume_food(state)
    report["food_consumed"] = food_consumed
    report["food_remaining"] = state.resources.food

    health_changes = _update_health(state)
    report["health_changes"] = health_changes

    _check_deaths(state)
    report["deaths"] = [m.name for m in state.party_members if not m.is_alive and
                        not any(e.get("name") == m.name for e in report.get("prior_deaths", []))]

    state.advance_date()
    report["game_date"] = state.game_date

    landmark_reached = _check_landmark(state)
    if landmark_reached:
        report["landmark"] = landmark_reached.name
        _arrive_at_landmark(state, landmark_reached)

    if not state.alive_members():
        state.phase = GamePhase.GAME_OVER
        state.is_finished = True
        report["game_over"] = True

    if state.miles_traveled >= state.total_miles:
        state.phase = GamePhase.VICTORY
        state.is_finished = True
        state.final_score = _calculate_score(state)
        report["victory"] = True
        report["score"] = state.final_score

    return report


def attempt_hunt(state: GameState) -> Dict:
    if state.resources.ammunition < 5:
        return {"success": False, "narrative_hint": "not_enough_ammo",
                "message": "Not enough ammunition to hunt (need 5)."}

    ammo_used = random.randint(5, 15)
    ammo_used = min(ammo_used, state.resources.ammunition)
    state.resources.ammunition -= ammo_used

    terrain = state.terrain
    base_chance = {"prairie": 0.7, "plains": 0.65, "hills": 0.5,
                   "mountains": 0.4, "desert": 0.2, "valley": 0.6}.get(terrain, 0.5)

    if random.random() < base_chance:
        food_gained = random.randint(30, 120)
        state.resources.food += food_gained
        return {"success": True, "food_gained": food_gained, "ammo_used": ammo_used,
                "narrative_hint": "hunt_success", "terrain": terrain}
    return {"success": False, "food_gained": 0, "ammo_used": ammo_used,
            "narrative_hint": "hunt_fail", "terrain": terrain}


def attempt_river_crossing(state: GameState, method: str) -> Dict:
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    if not lm.has_river:
        return {"success": True, "method": method, "message": "No river here."}

    depth = (lm.river_base_depth_ft or 3.0) * random.uniform(0.7, 1.5)
    width = lm.river_width_ft or 200

    result: Dict = {"method": method, "depth_ft": round(depth, 1), "width_ft": width}

    if method == "ford":
        if depth > 4.0:
            loss_chance = min(0.8, (depth - 4.0) * 0.2)
            if random.random() < loss_chance:
                food_lost = random.randint(20, 80)
                state.resources.food = max(0, state.resources.food - food_lost)
                result.update({"success": False, "food_lost": food_lost,
                               "narrative_hint": "ford_fail"})
                return result
        result.update({"success": True, "narrative_hint": "ford_success"})

    elif method == "caulk":
        fail_chance = 0.15 + (depth / 20.0)
        if random.random() < fail_chance:
            items_lost = random.choice(["food", "ammunition", "spare_parts"])
            amount = random.randint(10, 40)
            current = getattr(state.resources, items_lost, 0)
            setattr(state.resources, items_lost, max(0, current - amount))
            result.update({"success": False, "items_lost": items_lost, "amount_lost": amount,
                           "narrative_hint": "caulk_fail"})
            return result
        result.update({"success": True, "narrative_hint": "caulk_success"})

    elif method == "ferry":
        cost = 5.0 + (width / 100)
        if state.resources.money < cost:
            result.update({"success": False, "cost": cost,
                           "narrative_hint": "ferry_no_money"})
            return result
        state.resources.money -= cost
        result.update({"success": True, "cost": cost, "narrative_hint": "ferry_success"})

    elif method == "wait":
        state.advance_date(days=random.randint(1, 3))
        _consume_food(state)
        result.update({"success": True, "narrative_hint": "wait_success"})

    else:
        result.update({"success": False, "narrative_hint": "unknown_method"})

    return result


def trade_at_fort(state: GameState, item: str, quantity: int) -> Tuple[bool, str]:
    lm = TRAIL_LANDMARKS[state.current_landmark_idx]
    mult = FORT_PRICE_MULTIPLIERS.get(lm.name, 1.5)
    price_per = SHOP_PRICES.get(item, 0) * mult
    total = price_per * quantity
    if total > state.resources.money:
        return False, f"Not enough money. Need ${total:.2f}, have ${state.resources.money:.2f}"
    return buy_supplies(state, item, quantity)


def rest_at_location(state: GameState, days: int = 1) -> str:
    days = max(1, min(days, 5))
    for _ in range(days):
        state.advance_date()
        _consume_food(state)
        for m in state.alive_members():
            m.health = min(100, m.health + random.randint(3, 8))
            m.morale = min(100, m.morale + random.randint(1, 4))
            recovered = [a for a in m.ailments if random.random() < AILMENT_RECOVERY_CHANCE.get(a, 0.1) * 1.5]
            for a in recovered:
                m.ailments.remove(a)
    _check_deaths(state)
    return f"Rested for {days} day{'s' if days > 1 else ''}."


def continue_from_landmark(state: GameState) -> None:
    state.phase = GamePhase.TRAVELING
    _set_travel_actions(state)


# ---------- internal helpers ----------

def _calculate_miles(state: GameState) -> int:
    base = PACE_BASE_MILES.get(state.pace.value, 14)
    terrain_mod = TERRAIN_SPEED_MODIFIER.get(state.terrain, 1.0)
    weather_mod = WEATHER_SPEED_MODIFIER.get(state.weather, 1.0)
    oxen_factor = min(1.0, state.resources.oxen / 4.0) if state.resources.oxen > 0 else 0.0
    miles = int(base * terrain_mod * weather_mod * oxen_factor * random.uniform(0.85, 1.15))
    return max(0, miles)


def _consume_food(state: GameState) -> int:
    alive = len(state.alive_members())
    per_person = RATIONS_FOOD_PER_PERSON.get(state.rations.value, 3)
    total = alive * per_person
    state.resources.food = max(0, state.resources.food - total)
    return total


def _update_weather(state: GameState) -> None:
    month = date.fromisoformat(state.game_date).month
    weights = MONTH_WEATHER_WEIGHTS.get(month, MONTH_WEATHER_WEIGHTS[6])
    options = list(weights.keys())
    probs = list(weights.values())
    state.weather = random.choices(options, weights=probs, k=1)[0]


def _update_health(state: GameState) -> List[Dict]:
    changes: List[Dict] = []
    pace_penalty = PACE_HEALTH_PENALTY.get(state.pace.value, 0)
    ration_bonus = RATIONS_HEALTH_BONUS.get(state.rations.value, 0)
    starving = state.resources.food == 0

    for m in state.alive_members():
        delta = ration_bonus - pace_penalty
        if starving:
            delta -= random.randint(5, 12)

        for ailment in list(m.ailments):
            dmg = AILMENT_DAILY_DAMAGE.get(ailment, 3)
            delta -= dmg
            if random.random() < AILMENT_RECOVERY_CHANCE.get(ailment, 0.1):
                m.ailments.remove(ailment)
                changes.append({"name": m.name, "recovered": ailment})

        if random.random() < _ailment_chance(state):
            new_ailment = random.choice(AILMENTS)
            if new_ailment not in m.ailments:
                m.ailments.append(new_ailment)
                changes.append({"name": m.name, "new_ailment": new_ailment})

        rest_heal = random.randint(1, 3) if not starving else 0
        delta += rest_heal

        m.health = max(0, min(100, m.health + delta))
        m.morale = max(0, min(100, m.morale + random.randint(-3, 2)))

        if m.health <= 0:
            m.is_alive = False
            m.ailments.clear()
            changes.append({"name": m.name, "died": True})

    return changes


def _ailment_chance(state: GameState) -> float:
    base = 0.02
    if state.resources.food == 0:
        base += 0.06
    if state.rations == Rations.BARE_BONES:
        base += 0.02
    if state.pace == Pace.GRUELING:
        base += 0.03
    if state.weather in ("heavy_rain", "blizzard", "snow"):
        base += 0.03
    if state.resources.clothing < len(state.alive_members()):
        base += 0.02
    return base


def _check_deaths(state: GameState) -> None:
    for m in state.party_members:
        if m.is_alive and m.health <= 0:
            m.is_alive = False
            m.ailments.clear()


def _check_landmark(state: GameState) -> Optional[Landmark]:
    next_idx = state.current_landmark_idx + 1
    if next_idx >= len(TRAIL_LANDMARKS):
        return None
    lm = TRAIL_LANDMARKS[next_idx]
    if state.miles_traveled >= lm.mile_marker:
        state.current_landmark_idx = next_idx
        state.terrain = lm.terrain
        return lm
    return None


def _arrive_at_landmark(state: GameState, lm: Landmark) -> None:
    if lm.has_river:
        state.phase = GamePhase.RIVER_CROSSING
        state.available_actions = [
            ActionChoice(key="ford", label="Ford the river", description="Wade across — risky if deep"),
            ActionChoice(key="caulk", label="Caulk and float", description="Seal wagon and float across"),
            ActionChoice(key="ferry", label="Take the ferry", description="Pay for safe passage"),
            ActionChoice(key="wait", label="Wait for conditions", description="Rest and hope water drops"),
        ]
    elif lm.has_fort:
        state.phase = GamePhase.FORT
        state.available_actions = [
            ActionChoice(key="trade", label="Visit the trading post", description="Buy supplies at fort prices"),
            ActionChoice(key="rest", label="Rest here", description="Recover health for 1-3 days"),
            ActionChoice(key="continue", label="Continue on the trail"),
            ActionChoice(key="talk", label="Talk to people", description="See who's around"),
        ]
    else:
        state.phase = GamePhase.LANDMARK
        state.available_actions = [
            ActionChoice(key="rest", label="Rest here"),
            ActionChoice(key="continue", label="Continue on the trail"),
            ActionChoice(key="hunt", label="Hunt for food"),
            ActionChoice(key="talk", label="Look around", description="See what's here"),
        ]


def _set_travel_actions(state: GameState) -> None:
    state.available_actions = [
        ActionChoice(key="travel", label="Continue traveling"),
        ActionChoice(key="rest", label="Rest for a day"),
        ActionChoice(key="hunt", label="Hunt for food"),
        ActionChoice(key="pace", label="Change pace",
                     description=f"Current: {state.pace.value}"),
        ActionChoice(key="rations", label="Change rations",
                     description=f"Current: {state.rations.value.replace('_', ' ')}"),
        ActionChoice(key="journal", label="Read journal"),
        ActionChoice(key="status", label="Check status"),
    ]


def _set_supply_shop_actions(state: GameState) -> None:
    state.available_actions = [
        ActionChoice(key="buy_food", label="Buy food",
                     description=f"${SHOP_PRICES['food']:.2f}/lb"),
        ActionChoice(key="buy_ammo", label="Buy ammunition",
                     description=f"${SHOP_PRICES['ammunition']:.2f}/box"),
        ActionChoice(key="buy_clothing", label="Buy clothing",
                     description=f"${SHOP_PRICES['clothing']:.2f}/set"),
        ActionChoice(key="buy_parts", label="Buy spare parts",
                     description=f"${SHOP_PRICES['spare_parts']:.2f}/set"),
        ActionChoice(key="buy_oxen", label="Buy extra oxen",
                     description=f"${SHOP_PRICES['oxen']:.2f}/head"),
        ActionChoice(key="done", label="Hit the trail!",
                     description="Finish shopping and depart"),
    ]


def _calculate_score(state: GameState) -> int:
    alive = len(state.alive_members())
    health_sum = sum(m.health for m in state.alive_members())
    resource_val = (state.resources.food * 0.04 + state.resources.ammunition * 0.5 +
                    state.resources.spare_parts * 2 + state.resources.clothing * 2 +
                    state.resources.money * 0.2)
    day_bonus = max(0, 300 - state.day_number) * 2
    multiplier = SCORE_MULTIPLIER.get(state.profession, 1.0)
    raw = (alive * 200 + health_sum + resource_val + day_bonus)
    return int(raw * multiplier)
