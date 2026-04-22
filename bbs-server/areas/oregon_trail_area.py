"""
Oregon Trail BBS door game.
Retro telnet interface to the backend Oregon Trail game engine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from rendering.paginator import paginate_text
from rendering.text import draw_box, format_header_context, markdown_to_ansi, section_header, word_wrap

if TYPE_CHECKING:
    from session import BBSSession

logger = logging.getLogger(__name__)

TITLE_ART = r"""
    ___  ____  ____  ___  ____  _  _    ____  ____   __   __  __
   / __)(  _ \( ___)/ __)( ___)( \( )  (_  _)(  _ \ / _\ (  )(  )
  ( (__  )   / )__)( (__  )__)  )  (     )(   )   //    \ )( / (_/\
   \___)(_)\_)(____)\___)(____)(__)\_)  (__) (_)\_)\_/\_/(__)\____/
            =======================================
                   THE OREGON TRAIL - 1848
            =======================================
"""


async def _write(session: "BBSSession", text: str) -> None:
    await session._write(text)


async def _write_narrative(session: "BBSSession", text: str) -> None:
    t = session.theme
    ansi_text = markdown_to_ansi(text, t)
    lines = word_wrap(ansi_text, session.term_width - 4)
    for line in lines:
        await session._write(f"  {line}\r\n")
        await asyncio.sleep(0.03)


async def _write_wrapped(session: "BBSSession", text: str) -> None:
    t = markdown_to_ansi(text, session.theme)
    lines = word_wrap(t, session.term_width - 2)
    page_h = max(5, session.term_height - 3)

    async def wl(s: str) -> None:
        await session._write(s)

    await paginate_text(
        lines,
        page_h,
        session.theme,
        wl,
        session.read_pager_key,
        after_line_input_drain=session.drain_stray_line_terminators,
    )


async def _spinner(session: "BBSSession", task: asyncio.Task) -> None:
    t = session.theme
    frames = ("|", "/", "-", "\\")
    fi = 0
    while not task.done():
        status = f"  {t.dim}{frames[fi]} Working...{t.reset}"
        await session._write_bytes(b"\r" + status.encode("utf-8", errors="replace"))
        fi = (fi + 1) % len(frames)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.25)
        except asyncio.TimeoutError:
            pass
        except Exception:
            break
    await session._write_bytes(b"\r" + b" " * 30 + b"\r")


def _render_status_bar(gs: Dict, theme) -> List[str]:
    t = theme
    lines = []
    location = gs.get("location", "Unknown")
    miles = gs.get("miles_traveled", 0)
    total = gs.get("total_miles", 2170)
    day = gs.get("day_number", 1)
    game_date = gs.get("game_date", "")
    weather = gs.get("weather", "clear").replace("_", " ")
    pace = gs.get("pace", "steady")
    rations = gs.get("rations", "filling").replace("_", " ")

    progress_pct = min(100, int(miles / total * 100))
    bar_width = 20
    filled = int(bar_width * progress_pct / 100)
    bar = "#" * filled + "." * (bar_width - filled)

    lines.append(f"{t.bold}Day {day}{t.reset} - {game_date}  "
                 f"{t.fg_bright_cyan}{location}{t.reset}")
    lines.append(f"  [{bar}] {miles}/{total} mi ({progress_pct}%)")
    lines.append(f"  Weather: {weather} | Pace: {pace} | Rations: {rations}")

    party = gs.get("party", [])
    party_parts = []
    for m in party:
        name = m["name"]
        status = m.get("status", "good").replace("_", " ").title()
        if not m.get("is_alive", True):
            party_parts.append(f"{t.dim}{name}(Dead){t.reset}")
        elif status == "Good":
            party_parts.append(f"{t.fg_bright_green}{name}({status}){t.reset}")
        elif status in ("Fair",):
            party_parts.append(f"{t.fg_yellow}{name}({status}){t.reset}")
        else:
            party_parts.append(f"{t.fg_bright_yellow}{name}({status}){t.reset}")
    lines.append("  Party: " + "  ".join(party_parts))

    res = gs.get("resources", {})
    lines.append(
        f"  Food: {res.get('food', 0)} lbs | Ammo: {res.get('ammunition', 0)} | "
        f"Parts: {res.get('spare_parts', 0)} | Clothing: {res.get('clothing', 0)} | "
        f"Oxen: {res.get('oxen', 0)} | ${res.get('money', 0):.2f}"
    )
    return lines


def _render_actions(gs: Dict, theme) -> List[str]:
    t = theme
    actions = gs.get("available_actions", [])
    lines = [f"{t.fg_bright_cyan}{'-' * 40}{t.reset}"]
    for a in actions:
        key = a.get("key", "?")
        label = a.get("label", "")
        desc = a.get("description", "")
        line = f"  [{t.bold}{key}{t.reset}] {label}"
        if desc:
            line += f"  {t.dim}({desc}){t.reset}"
        lines.append(line)
    lines.append(f"  [{t.bold}>{t.reset}] Type a custom action")
    lines.append(f"  [{t.bold}quit{t.reset}] Return to menu")
    return lines


async def _pick_model(session: "BBSSession") -> str:
    client = session.client
    result = await client.ot_get_models(session.jwt_token)
    if result.get("error"):
        await _write(session, f"  Could not load models: {result['error'][:200]}\r\n")
        return ""
    models = result.get("models", [])
    if not models:
        await _write(session, "  No LLM models available.\r\n")
        return ""
    if len(models) == 1:
        return models[0]

    t = session.theme
    await _write(session, f"\r\n{t.bold}Choose your narrator model:{t.reset}\r\n")
    for i, m in enumerate(models, 1):
        short = m.split("/")[-1] if "/" in m else m
        await _write(session, f"  {i}) {short}\r\n")
    await _write(session, f"\r\nEnter number [1]: ")
    choice = (await session.read_menu_choice(allow_digit_suffix=True)).strip()
    if not choice:
        return models[0]
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    return models[0]


async def _new_game_wizard(session: "BBSSession") -> Optional[str]:
    t = session.theme
    await session.clear_screen()
    await _write(session, TITLE_ART + "\r\n")

    model_id = await _pick_model(session)
    if not model_id:
        return None

    await _write(session, f"\r\n{t.bold}What is your name, traveler?{t.reset}\r\n> ")
    leader = (await session.read_line()).strip()
    if not leader:
        leader = "Pioneer"

    await _write(session, f"\r\nName your travel companions (up to 3, comma-separated):\r\n> ")
    raw = (await session.read_line()).strip()
    party_names = [n.strip() for n in raw.split(",") if n.strip()][:3]
    if not party_names:
        party_names = ["Mary", "Tom", "Sara"]

    await _write(session, f"\r\n{t.bold}Choose your profession:{t.reset}\r\n")
    await _write(session, "  1) Banker    - $1600 starting money\r\n")
    await _write(session, "  2) Carpenter - $800, 2x score multiplier\r\n")
    await _write(session, "  3) Farmer    - $400, 3x score multiplier\r\n")
    await _write(session, "Choose [1]: ")
    prof_choice = (await session.read_menu_choice()).strip()
    prof = {"2": "carpenter", "3": "farmer"}.get(prof_choice, "banker")

    await _write(session, f"\r\n{t.dim}Creating your journey...{t.reset}\r\n")
    task = asyncio.create_task(
        session.client.ot_new_game(session.jwt_token, leader, party_names, prof, model_id)
    )
    await _spinner(session, task)
    result = task.result()

    if result.get("error"):
        await _write(session, f"\r\nError: {result['error'][:300]}\r\n")
        return None

    game_id = result.get("game_id")
    narrative = result.get("narrative", "")
    if narrative:
        await _write(session, "\r\n")
        await _write_narrative(session, narrative)
    return game_id


async def _game_loop(session: "BBSSession", game_id: str) -> None:
    client = session.client
    jwt = session.jwt_token
    t = session.theme

    gs = await client.ot_get_state(jwt, game_id)
    if gs.get("error"):
        await _write(session, f"Error loading game: {gs['error'][:200]}\r\n")
        return

    while True:
        await _write(session, "\r\n")
        for line in _render_status_bar(gs, t):
            await _write(session, line + "\r\n")
        await _write(session, "\r\n")

        if gs.get("narrative"):
            await _write_narrative(session, gs["narrative"])
            await _write(session, "\r\n")

        if gs.get("is_finished"):
            score = gs.get("final_score")
            phase = gs.get("phase", "")
            if phase == "victory":
                await _write(session, f"{t.fg_bright_green}{t.bold}YOU HAVE REACHED OREGON!{t.reset}\r\n")
                await _write(session, f"Final score: {score}\r\n")
            else:
                await _write(session, f"{t.fg_bright_yellow}{t.bold}GAME OVER{t.reset}\r\n")
            await _write(session, "\r\nPress Enter to return to menu.\r\n")
            await session.read_line()
            return

        for line in _render_actions(gs, t):
            await _write(session, line + "\r\n")

        await _write(session, f"\r\n{t.bold}>{t.reset} ")
        raw = (await session.read_line()).strip()
        if not raw:
            continue

        lower = raw.lower()
        if lower in ("quit", "q", "/quit", "/menu"):
            await _write(session, "Game saved. Returning to menu.\r\n")
            return

        if lower == "journal":
            await _show_journal(session, game_id)
            gs = await client.ot_get_state(jwt, game_id)
            continue

        if lower == "status":
            gs = await client.ot_get_state(jwt, game_id)
            continue

        action = lower
        detail = None
        quantity = None

        if lower.startswith(">") or lower.startswith("custom "):
            action = "custom"
            detail = raw.lstrip(">").strip()
            if detail.startswith("custom "):
                detail = detail[7:].strip()

        elif lower.startswith("talk ") or lower == "talk":
            if lower == "talk":
                await _write(session, "What do you say? > ")
                detail = (await session.read_line()).strip()
            else:
                detail = raw[5:].strip()
            if detail:
                task = asyncio.create_task(client.ot_talk(jwt, game_id, detail))
                await _spinner(session, task)
                result = task.result()
                if result.get("error"):
                    await _write(session, f"Error: {result['error'][:200]}\r\n")
                else:
                    npc = result.get("npc_name", "Stranger")
                    dialogue = result.get("dialogue", "...")
                    await _write(session, f"\r\n{t.bold}{npc}:{t.reset}\r\n")
                    await _write_narrative(session, dialogue)
                    gs = result
                continue

        elif lower.startswith("buy_") or lower.startswith("trade"):
            if lower.startswith("trade "):
                parts = lower[6:].strip().split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    detail = parts[0]
                    quantity = int(parts[-1])
                    action = "trade"

        elif lower.startswith("pace "):
            detail = lower[5:].strip()
            action = "pace"
        elif lower.startswith("rations "):
            detail = lower[8:].strip()
            action = "rations"
        elif lower.startswith("rest ") and lower[5:].strip().isdigit():
            quantity = int(lower[5:].strip())
            action = "rest"

        task = asyncio.create_task(client.ot_action(jwt, game_id, action, detail, quantity))
        await _spinner(session, task)
        result = task.result()

        if result.get("error"):
            await _write(session, f"Error: {result['error'][:200]}\r\n")
            gs = await client.ot_get_state(jwt, game_id)
        else:
            gs = result


async def _show_journal(session: "BBSSession", game_id: str) -> None:
    result = await session.client.ot_journal(session.jwt_token, game_id)
    if result.get("error"):
        await _write(session, f"Error: {result['error'][:200]}\r\n")
        return
    entries = result.get("journal", [])
    if not entries:
        await _write(session, "  The journal is empty.\r\n")
        return
    t = session.theme
    lines = [
        section_header(
            "Trail Journal",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
    ]
    for e in entries[-20:]:
        lines.append(f"  {t.bold}Day {e.get('day', '?')} - {e.get('game_date', '')}{t.reset}  ({e.get('location', '')})")
        for wl in word_wrap(e.get("text", ""), session.term_width - 6):
            lines.append(f"    {wl}")
        lines.append("")
    await _write_wrapped(session, "\n".join(lines))


async def oregon_trail_menu(session: "BBSSession", *, back_label: str = "Back to main menu") -> None:
    t = session.theme
    client = session.client
    jwt = session.jwt_token

    while True:
        await session.clear_screen()
        await _write(session, TITLE_ART + "\r\n")
        hdr = section_header(
            "Oregon Trail",
            session.term_width - 2,
            t,
            context=format_header_context(session.display_name or session.username),
        )
        await _write(session, hdr + "\r\n\r\n")

        saves = await client.ot_list_saves(jwt)
        save_list = saves.get("saves", []) if not saves.get("error") else []
        active_saves = [s for s in save_list if not s.get("is_finished")]

        await _write(session, f"  [{t.bold}N{t.reset}] New Game\r\n")
        if active_saves:
            for i, s in enumerate(active_saves[:5], 1):
                name = s.get("leader_name", "?")
                miles = s.get("miles_traveled", 0)
                day = s.get("day_number", 1)
                await _write(session, f"  [{t.bold}{i}{t.reset}] Resume: {name} - Day {day}, {miles} mi\r\n")
        await _write(session, f"  [{t.bold}Q{t.reset}] {back_label}\r\n")
        await _write(session, f"\r\nChoose> ")

        choice = (await session.read_menu_choice()).strip().lower()
        if choice in ("q", "quit", "/quit", "/menu"):
            return
        if choice == "n":
            game_id = await _new_game_wizard(session)
            if game_id:
                await _game_loop(session, game_id)
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(active_saves):
                gid = active_saves[idx].get("game_id")
                if gid:
                    await _game_loop(session, gid)
