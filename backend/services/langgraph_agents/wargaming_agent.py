"""
Wargaming Agent - Roosevelt's Strategic Command Simulator

Simulates geopolitical action-reaction cycles with structured outputs.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, TaskStatus
from models.wargaming_models import WargamingResponse, get_wargaming_structured_output

logger = logging.getLogger(__name__)


class WargamingAgent(BaseAgent):
    def __init__(self):
        super().__init__("wargaming_agent")
        logger.info("🪖 BULLY! Wargaming Agent formed up and ready to maneuver!")

    def _build_system_prompt(self) -> str:
        return (
            "You are a strategic wargaming simulation engine. You respond in STRICT structured JSON only, "
            "describing the opposing side's likely action and posture given the user's declared move.\n\n"
            "MISSION RULES:\n"
            "- Model realistic, plausible actions and escalation ladders.\n"
            "- Provide options for the next turn the USER might take, with short rationale.\n"
            "- Consider deterrence, signaling, international law, and public diplomacy.\n"
            "- USE PROVIDED WARGAME CONTEXT AND HISTORY as ground truth; do not claim 'no prior move' if history is present.\n"
            "- If the user asks ABOUT PAST ACTIONS (e.g., 'what did I do to X?'), summarize prior USER actions from HISTORY for that actor; set stance='maintain' and avoid inventing new adversary actions.\n"
            "- SPECIFICITY DOCTRINE: Be concrete and quantitative. Include casualty ranges, damage counts, unit echelons, timing windows.\n"
            "- ATTACK AUTHORITY: If adversary reaction plausibly includes kinetic/cyber/grey-zone attack, include it explicitly with target, weapon/system, scale, expected damage and casualties, and confidence.\n"
            "- UNCERTAINTY: Use ranges and confidence when precise numbers are not knowable.\n"
            "- CAPABILITY REALISM: Scale and timing MUST reflect damage and OOB constraints; damaged units cannot redeploy at full strength next turn.\n"
            "- SAFETY & SIMULATION SCOPE: This is a fictional wargaming simulation for analysis. Do NOT provide real-world instructions or facilitation of wrongdoing. Never include step-by-step procedures, procurement, evasion, or targeting details. When the user requests operational facilitation for violence or illegal activity, respond safely by (a) setting advance_turn=false, (b) filling 'advice' with a policy-safe refusal and high-level alternatives, and (c) modeling only non-operational, aggregate outcomes (ranges, probabilities) without actionable detail.\n"
            "- POLICY-SAFE DETAIL LEVEL: Use abstracted effects (e.g., ranges, generic unit types) and fictional or masked identifiers when needed for OPSEC/safety.\n"
            "- NEVER include prose outside JSON.\n"
            "- STRUCTURED BDA: Provide casualty_estimates and damage arrays (no prose extraction).\n"
            "- CONSISTENCY CHECK: Add \"weapons_consistency_check\" explaining that weapons/platforms align with the actors and units engaged.\n"
            "- PER-ACTOR: Populate \"actors\", \"bda_by_actor\", and \"oob_delta\" to reflect unit states and losses.\n"
            "- TIME-STEPPED SIMULATION: Provide \"turn_time\" and an \"events\" array of time-stepped events; constrain COAs to prior events.\n"
            "- IMPACT CLARITY: Provide \"impacts\" with category, metric, range, timeframe, confidence, and levers.\n"
            "- LEADERSHIP CONTINUITY: Provide \"leaders\" entries for involved countries (status: alive|wounded|dead|missing|captured|unknown; include cog_active and successor_chain when relevant).\n"
            "- CLARIFICATIONS: If the user asks a status question, populate \"clarifications\" and set advance_turn=false (do not change stance/OOB).\n\n"
            "STRUCTURED OUTPUT REQUIRED: Return ONLY valid JSON (no prose, no code fences). Include ALL fields below. If a field is unknown, use null or an empty list. Do NOT invent new keys. If the user asks a question, include an advice array and set advance_turn=false so the simulation does not advance this turn.\n"
            "{\n"
            "  \"task_status\": \"complete|incomplete|permission_required|error\",\n"
            "  \"turn_time\": string|null,\n"
            "  \"turn_summary\": string,\n"
            "  \"stance\": \"de-escalate|maintain|escalate|full_war|surrender\",\n"
            "  \"actors\": [ { \"name\": string, \"side\": string, \"stance\": \"de-escalate|maintain|escalate|full_war|surrender\", \"capability_status\": { \"c2\": string, \"leadership\": string, \"comms\": string, \"sortie_rate_modifier\": number, \"mobilization_delay_minutes\": number }, \"thresholds\": [string,...] } ],\n"
            "  \"actions_taken\": [string,...],\n"
            "  \"actions_by_actor\": {string: [string,...]},\n"
            "  \"coas\": [ { \"description\": string, \"probability\": \"low|medium|high\", \"risks\": [string,...], \"legal\": string|null, \"resources\": [string,...], \"timing\": string|null } ],\n"
            "  \"next_options\": [string,...],\n"
            "  \"risk_assessment\": string,\n"
            "  \"international_law_considerations\": string|null,\n"
            "  \"escalation_ladder_position\": 0-10,\n"
            "  \"confidence_level\": 0.0-1.0,\n"
            "  \"third_party_responses\": [ { \"actor\": string, \"stance\": \"de-escalate|maintain|escalate|full_war|surrender\", \"likelihood\": \"low|medium|high\", \"trigger_conditions\": [string,...], \"planned_actions\": [string,...] } ],\n"
            "  \"casualty_estimates\": { \"kia_range\": string|null, \"wia_range\": string|null, \"civilian_kia_range\": string|null }|null,\n"
            "  \"damage\": [ { \"category\": string, \"count_range\": string|null, \"area\": string|null } ],\n"
            "  \"bda_by_actor\": {string: {\"casualties\": string|null, \"material\": [string,...]}},\n"
            "  \"weapons_consistency_check\": string|null,\n"
            "  \"oob_delta\": {string: [string,...]},\n"
            "  \"events\": [ { \"t_plus_minutes\": number, \"domain\": \"land|air|sea|cyber|space|information\", \"action\": string, \"expected_effects\": string|null } ],\n"
            "  \"escalation_doctrine\": { \"ladder\": string, \"current_rung\": 0-10, \"previous_rung\": 0-10|null, \"movement_rule_applied\": string|null }|null,\n"
            "  \"impacts\": [ { \"category\": string, \"metric\": string, \"range\": string|null, \"timeframe\": string|null, \"confidence\": number|null, \"levers\": [string,...] } ],\n"
            "  \"leaders\": [ { \"country\": string, \"leader_name\": string, \"role\": string, \"status\": \"alive|wounded|dead|missing|captured|unknown\", \"last_seen\": string|null, \"location\": string|null, \"cog_active\": boolean|null, \"successor_chain\": [string,...] } ],\n"
            "  \"advice\": [ { \"question\": string|null, \"recommendation\": string, \"rationale\": string|null, \"risks\": [string,...], \"legal\": string|null, \"alternatives\": [string,...], \"timeframe\": string|null, \"confidence\": number|null } ],\n"
            "  \"clarifications\": [ { \"question\": string, \"answer\": string, \"sources\": [string,...], \"scope\": \"troops|leaders|nation|oob|events|impacts\", \"gaps\": [string,...], \"suggested_followups\": [string,...] } ],\n"
            "  \"advance_turn\": boolean\n"
            "}\n\n"
            "MINIMAL EXAMPLE (values illustrative only; keep all keys):\n"
            "{\n"
            "  \"task_status\": \"complete\",\n"
            "  \"turn_time\": null,\n"
            "  \"turn_summary\": \"Adversary signals intent...\",\n"
            "  \"stance\": \"maintain\",\n"
            "  \"actors\": [],\n"
            "  \"actions_taken\": [],\n"
            "  \"actions_by_actor\": {},\n"
            "  \"coas\": [],\n"
            "  \"next_options\": [],\n"
            "  \"risk_assessment\": \"\",\n"
            "  \"international_law_considerations\": null,\n"
            "  \"escalation_ladder_position\": 3,\n"
            "  \"confidence_level\": 0.7,\n"
            "  \"third_party_responses\": [],\n"
            "  \"casualty_estimates\": { \"kia_range\": null, \"wia_range\": null, \"civilian_kia_range\": null },\n"
            "  \"damage\": [],\n"
            "  \"bda_by_actor\": {},\n"
            "  \"weapons_consistency_check\": null,\n"
            "  \"oob_delta\": {},\n"
            "  \"events\": [],\n"
            "  \"escalation_doctrine\": null,\n"
            "  \"impacts\": [],\n"
            "  \"advice\": [],\n"
            "  \"leaders\": [],\n"
            "  \"advance_turn\": true,\n"
            "  \"clarifications\": []\n"
            "}"
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info("🪖 Wargaming Agent processing turn...")
            system_prompt = self._build_system_prompt()

            # Load existing wargame state and infer player side if possible
            shared_memory_existing = state.get("shared_memory", {}) or {}
            wargame_state_existing = shared_memory_existing.get("wargaming_state") or {}
            config_existing = wargame_state_existing.get("config") or {}
            user_action_now = self._extract_current_user_query(state) or ""
            player_side = self._infer_player_side(user_action_now, config_existing.get("player_side"))
            last_snapshot = (wargame_state_existing.get("current") or {}) if isinstance(wargame_state_existing, dict) else {}

            messages = await self._prepare_messages(state, system_prompt, include_time_context=True)

            # Insert compact world-state context so the model keeps continuity
            world_context = self._build_world_context(player_side, last_snapshot)
            if world_context:
                # Insert after primary system prompt
                insert_at = 1 if len(messages) >= 1 else 0
                messages.insert(insert_at, {"role": "system", "content": world_context})

            # Insert recent history summary for continuity-aware answers
            history_context = self._build_history_context((wargame_state_existing.get("history") or []) if isinstance(wargame_state_existing, dict) else [])
            if history_context:
                insert_at2 = 2 if len(messages) >= 2 else len(messages)
                messages.insert(insert_at2, {"role": "system", "content": history_context})

            # Validate capability consistency from recent history (no regex extraction)
            try:
                cap_ctx = self._validate_capability_consistency((wargame_state_existing.get("history") or []) if isinstance(wargame_state_existing, dict) else [])
                if cap_ctx:
                    insert_at3 = 3 if len(messages) >= 3 else len(messages)
                    messages.insert(insert_at3, {"role": "system", "content": cap_ctx})
            except Exception:
                pass

            # Execute LLM call with structured output schema
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            # Deliver without tools; pure simulation
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.6,
            )

            content = response.choices[0].message.content or "{}"
            structured = self._parse_structured_wargaming_response(content)

            # Build final answer for user display
            final_answer = self._format_user_message(structured)

            # Persist running world state in shared_memory
            shared_memory = state.get("shared_memory", {}) or {}
            wargame_state = shared_memory.get("wargaming_state") or {
                "history": [],
                "current": {
                    "stance": None,
                    "escalation_ladder_position": 0,
                    "last_turn_summary": None,
                }
            }
            # Ensure config exists and keep player side
            wargame_state.setdefault("config", {})
            if player_side:
                wargame_state["config"]["player_side"] = player_side

            user_action = user_action_now
            # If the LLM is providing advice without advancing the turn, skip history append
            advance_turn = bool(getattr(structured, "advance_turn", True))
            if advance_turn:
                wargame_state["history"].append({
                "timestamp": datetime.now().isoformat(),
                "user_action": user_action,
                "adversary_stance": str(structured.stance),
                "adversary_actions": structured.actions_taken,
                "escalation_ladder_position": structured.escalation_ladder_position,
                "turn_summary": structured.turn_summary,
                "turn_time": getattr(structured, "turn_time", None),
                "events": [
                    {
                        "t_plus_minutes": (ev.get("t_plus_minutes") if isinstance(ev, dict) else getattr(ev, "t_plus_minutes", None)),
                        "domain": (ev.get("domain") if isinstance(ev, dict) else getattr(ev, "domain", None)),
                        "action": (ev.get("action") if isinstance(ev, dict) else getattr(ev, "action", None)),
                        "expected_effects": (ev.get("expected_effects") if isinstance(ev, dict) else getattr(ev, "expected_effects", None))
                    } for ev in (getattr(structured, "events", []) or [])
                ],
                "next_options": structured.next_options,
                "risk_assessment": structured.risk_assessment,
                "bda": {
                    "casualties": (getattr(structured, "casualty_estimates", None).dict() if getattr(structured, "casualty_estimates", None) else None),
                    "material": [
                        (
                            (
                                (getattr(d, "category", None) or "") +
                                (f" {getattr(d, 'count_range', '')}" if getattr(d, 'count_range', None) else "") +
                                (f" @ {getattr(d, 'area', '')}" if getattr(d, 'area', None) else "")
                            ).strip()
                        ) for d in (getattr(structured, "damage", []) or []) if d is not None
                    ],
                    "by_actor": getattr(structured, "bda_by_actor", {}) or {}
                },
                "oob_delta": getattr(structured, "oob_delta", {}) or {},
                "impacts": [
                    {
                        "category": (imp.get("category") if isinstance(imp, dict) else getattr(imp, "category", None)),
                        "metric": (imp.get("metric") if isinstance(imp, dict) else getattr(imp, "metric", None)),
                        "range": (imp.get("range") if isinstance(imp, dict) else getattr(imp, "range", None)),
                        "timeframe": (imp.get("timeframe") if isinstance(imp, dict) else getattr(imp, "timeframe", None)),
                        "confidence": (imp.get("confidence") if isinstance(imp, dict) else getattr(imp, "confidence", None)),
                        "levers": (imp.get("levers") if isinstance(imp, dict) else getattr(imp, "levers", [])),
                    } for imp in (getattr(structured, "impacts", []) or [])
                ],
                "leaders": [
                    {
                        "country": (ld.get("country") if isinstance(ld, dict) else getattr(ld, "country", None)),
                        "leader_name": (ld.get("leader_name") if isinstance(ld, dict) else getattr(ld, "leader_name", None)),
                        "role": (ld.get("role") if isinstance(ld, dict) else getattr(ld, "role", None)),
                        "status": (ld.get("status") if isinstance(ld, dict) else getattr(ld, "status", None)),
                        "last_seen": (ld.get("last_seen") if isinstance(ld, dict) else getattr(ld, "last_seen", None)),
                        "location": (ld.get("location") if isinstance(ld, dict) else getattr(ld, "location", None)),
                        "cog_active": (ld.get("cog_active") if isinstance(ld, dict) else getattr(ld, "cog_active", None)),
                        "successor_chain": (ld.get("successor_chain") if isinstance(ld, dict) else getattr(ld, "successor_chain", [])),
                    } for ld in (getattr(structured, "leaders", []) or [])
                ],
                })
            # Update current snapshot; optionally maintain a simple OOB view as counts only (future: detailed OOB)
            if advance_turn:
                wargame_state["current"] = {
                    "stance": str(structured.stance),
                    "escalation_ladder_position": structured.escalation_ladder_position,
                    "last_turn_summary": structured.turn_summary,
                }
            else:
                # Store latest advice for display/reference, without altering state
                shared_memory["wargaming_latest_advice"] = [
                    {
                        "question": (a.get("question") if isinstance(a, dict) else getattr(a, "question", None)),
                        "recommendation": (a.get("recommendation") if isinstance(a, dict) else getattr(a, "recommendation", None)),
                        "rationale": (a.get("rationale") if isinstance(a, dict) else getattr(a, "rationale", None)),
                        "risks": (a.get("risks") if isinstance(a, dict) else getattr(a, "risks", [])),
                        "legal": (a.get("legal") if isinstance(a, dict) else getattr(a, "legal", None)),
                        "alternatives": (a.get("alternatives") if isinstance(a, dict) else getattr(a, "alternatives", [])),
                        "timeframe": (a.get("timeframe") if isinstance(a, dict) else getattr(a, "timeframe", None)),
                        "confidence": (a.get("confidence") if isinstance(a, dict) else getattr(a, "confidence", None)),
                    } for a in (getattr(structured, "advice", []) or [])
                ]
                # Store latest clarifications similarly
                shared_memory["wargaming_latest_clarifications"] = [
                    {
                        "question": (c.get("question") if isinstance(c, dict) else getattr(c, "question", "")),
                        "answer": (c.get("answer") if isinstance(c, dict) else getattr(c, "answer", "")),
                        "sources": (c.get("sources") if isinstance(c, dict) else getattr(c, "sources", [])),
                        "scope": (c.get("scope") if isinstance(c, dict) else getattr(c, "scope", "")),
                        "gaps": (c.get("gaps") if isinstance(c, dict) else getattr(c, "gaps", [])),
                        "suggested_followups": (c.get("suggested_followups") if isinstance(c, dict) else getattr(c, "suggested_followups", [])),
                    } for c in (getattr(structured, "clarifications", []) or [])
                ]
            # Persist latest capability constraints from validation context is no longer used
            shared_memory["wargaming_state"] = wargame_state

            # Determine if this is the first activation (no prior wargame state)
            is_first_activation = not bool(wargame_state_existing)

            result = {
                "agent_results": {
                    "agent_type": "wargaming_agent",
                    "response": final_answer,
                    "structured_response": structured.dict(),
                    "tools_used": [],
                    "processing_time": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    # On first activation, request a conversation lock to wargaming
                    **({"locked_agent": "wargaming"} if is_first_activation else {}),
                },
                "latest_response": final_answer,
                "is_complete": structured.task_status == TaskStatus.COMPLETE,
                "shared_memory": shared_memory,
            }

            return result

        except Exception as e:
            logger.error(f"❌ Wargaming Agent error: {e}")
            return self._create_error_result(self._make_error("wargaming_error", str(e)))

    def _parse_structured_wargaming_response(self, content: str) -> WargamingResponse:
        try:
            # Extract JSON even if wrapped in markdown
            text = content.strip()
            if "```" in text:
                import re
                m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            data = json.loads(text)
            # Try direct parse
            try:
                return WargamingResponse(**data)
            except Exception as e:
                # Normalize legacy/partial payloads to new schema
                logger.info(f"🔧 WG NORMALIZE: Attempting schema normalization after parse error: {e}")
                normalized = self._normalize_wargaming_payload(data)
                return WargamingResponse(**normalized)
        except Exception:
            # Defensive minimal fallback to keep flow
            return WargamingResponse(
                task_status=TaskStatus.COMPLETE,
                turn_summary="Adversary issues a stern diplomatic protest and begins limited naval patrols.",
                stance="maintain",  # type: ignore
                actions_taken=["Diplomatic demarche", "Increased EEZ patrols"],
                next_options=["Signal de-escalation via backchannel", "Shadow patrols with ROE clarification"],
                risk_assessment="Low immediate risk; possible miscalculation at sea.",
                international_law_considerations="UNCLOS navigation rights; Vienna Convention after expulsions.",
                escalation_ladder_position=3,
                confidence_level=0.7,
                advice=[],
                advance_turn=True,
            )

    def _normalize_wargaming_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Start with a clean scaffold of all required keys
            norm: Dict[str, Any] = {
                "task_status": data.get("task_status", "complete"),
                "turn_time": data.get("turn_time"),
                "turn_summary": data.get("turn_summary", ""),
                "stance": data.get("stance", "maintain"),
                "actors": data.get("actors") or [],
                "actions_taken": data.get("actions_taken") or [],
                "actions_by_actor": data.get("actions_by_actor") or {},
                "coas": data.get("coas") or [],
                "next_options": data.get("next_options") or [],
                "risk_assessment": data.get("risk_assessment", ""),
                "international_law_considerations": data.get("international_law_considerations"),
                "escalation_ladder_position": int(data.get("escalation_ladder_position", 3) or 3),
                "confidence_level": float(data.get("confidence_level", 0.7) or 0.7),
                "third_party_responses": data.get("third_party_responses") or [],
                "casualty_estimates": data.get("casualty_estimates") or {
                    "kia_range": None,
                    "wia_range": None,
                    "civilian_kia_range": None
                },
                "damage": data.get("damage") or [],
                "bda_by_actor": data.get("bda_by_actor") or {},
                "weapons_consistency_check": data.get("weapons_consistency_check"),
                "oob_delta": data.get("oob_delta") or {},
                "events": data.get("events") or [],
                "escalation_doctrine": data.get("escalation_doctrine"),
                "impacts": data.get("impacts") or [],
            }

            # === SANITIZE / COERCE RANGES ===
            # confidence_level 0..1
            try:
                cl = float(norm.get("confidence_level", 0.7) or 0.7)
                if cl > 1:
                    # If likely given as percentage, convert
                    cl = cl / 100.0 if cl <= 100 else 1.0
                if cl < 0:
                    cl = 0.0
                norm["confidence_level"] = cl
            except Exception:
                norm["confidence_level"] = 0.7

            # escalation_ladder_position 0..10
            try:
                elp = int(norm.get("escalation_ladder_position", 3) or 3)
                norm["escalation_ladder_position"] = max(0, min(10, elp))
            except Exception:
                norm["escalation_ladder_position"] = 3

            # events: enforce domain + t_plus_minutes
            if isinstance(norm.get("events"), list):
                allowed_domains = {"land", "air", "sea", "cyber", "space", "information"}
                sanitized_events = []
                for ev in norm.get("events", []):
                    try:
                        tpm = ev.get("t_plus_minutes", 0)
                        try:
                            tpm = int(tpm)
                        except Exception:
                            tpm = 0
                        if tpm < 0:
                            tpm = 0
                        domain = str(ev.get("domain", "land")).lower()
                        if domain not in allowed_domains:
                            domain = "land"
                        sanitized_events.append({
                            "t_plus_minutes": tpm,
                            "domain": domain,
                            "action": ev.get("action", ""),
                            "expected_effects": ev.get("expected_effects")
                        })
                    except Exception:
                        continue
                norm["events"] = sanitized_events

            # actors.capability_status: clamp sortie_rate_modifier 0..1, mobilization_delay_minutes >=0
            if isinstance(norm.get("actors"), list):
                sanitized_actors = []
                for ac in norm.get("actors", []):
                    try:
                        cs = ac.get("capability_status") or {}
                        srm = cs.get("sortie_rate_modifier", 1.0)
                        try:
                            srm = float(srm)
                        except Exception:
                            srm = 1.0
                        if srm > 1:
                            srm = srm / 100.0 if srm <= 100 else 1.0
                        if srm < 0:
                            srm = 0.0
                        md = cs.get("mobilization_delay_minutes", 0)
                        try:
                            md = int(md)
                        except Exception:
                            md = 0
                        if md < 0:
                            md = 0
                        cs["sortie_rate_modifier"] = srm
                        cs["mobilization_delay_minutes"] = md
                        ac["capability_status"] = cs
                        sanitized_actors.append(ac)
                    except Exception:
                        sanitized_actors.append(ac)
                norm["actors"] = sanitized_actors

            # third_party_responses: coerce enums
            if isinstance(norm.get("third_party_responses"), list):
                tr_sanitized = []
                for tp in norm.get("third_party_responses", []):
                    try:
                        lh = str(tp.get("likelihood", "medium")).lower()
                        # normalize synonyms and ranges
                        lh_map = {
                            "low-to-medium": "medium",
                            "medium-to-high": "high",
                            "med": "medium",
                            "hi": "high",
                            "lo": "low"
                        }
                        lh = lh_map.get(lh, lh.replace(" ", "-").strip())
                        if lh not in {"low", "medium", "high"}:
                            lh = "medium"
                        tp["likelihood"] = lh
                        st = str(tp.get("stance", "maintain")).lower()
                        if st not in {"de-escalate", "maintain", "escalate", "full_war", "surrender"}:
                            st = "maintain"
                        tp["stance"] = st
                        tr_sanitized.append(tp)
                    except Exception:
                        tr_sanitized.append(tp)
                norm["third_party_responses"] = tr_sanitized

            # impacts: clamp confidence 0..1
            if isinstance(norm.get("impacts"), list):
                imp_sanitized = []
                for imp in norm.get("impacts", []):
                    try:
                        conf = imp.get("confidence")
                        if conf is not None:
                            try:
                                conf = float(conf)
                                if conf > 1:
                                    conf = conf / 100.0 if conf <= 100 else 1.0
                                if conf < 0:
                                    conf = 0.0
                            except Exception:
                                conf = None
                            imp["confidence"] = conf
                        imp_sanitized.append(imp)
                    except Exception:
                        imp_sanitized.append(imp)
                norm["impacts"] = imp_sanitized

            # next_options: ensure list of strings
            if isinstance(norm.get("next_options"), list):
                no_sanitized = []
                for opt in norm.get("next_options", []):
                    try:
                        if isinstance(opt, str):
                            no_sanitized.append(opt)
                        elif isinstance(opt, dict):
                            val = opt.get("option") or opt.get("text") or opt.get("value")
                            if isinstance(val, str):
                                no_sanitized.append(val)
                            else:
                                no_sanitized.append(str(opt))
                        else:
                            no_sanitized.append(str(opt))
                    except Exception:
                        continue
                norm["next_options"] = no_sanitized

            # escalation_doctrine: clamp rung values 0..10 and coerce from strings
            ed = norm.get("escalation_doctrine")
            if isinstance(ed, dict):
                try:
                    cr = ed.get("current_rung")
                    pr = ed.get("previous_rung")
                    if cr is not None:
                        try:
                            cr = int(cr)
                        except Exception:
                            cr = 0
                        cr = max(0, min(10, cr))
                        ed["current_rung"] = cr
                    if pr is not None:
                        try:
                            pr = int(pr)
                        except Exception:
                            pr = 0
                        pr = max(0, min(10, pr))
                        ed["previous_rung"] = pr
                    norm["escalation_doctrine"] = ed
                except Exception:
                    pass

            # Legacy mappings
            if not norm["events"] and isinstance(data.get("timeline"), list):
                tl = []
                for ev in data.get("timeline", []):
                    try:
                        tl.append({
                            "t_plus_minutes": ev.get("t_plus_minutes") or 0,
                            "domain": ev.get("domain") or "land",
                            "action": ev.get("action") or ev.get("event") or "",
                            "expected_effects": ev.get("expected_effects") or ev.get("effects")
                        })
                    except Exception:
                        continue
                norm["events"] = tl

            if not data.get("casualty_estimates"):
                bc = data.get("bda_casualties")
                if bc is not None:
                    norm["casualty_estimates"] = {
                        "kia_range": bc,
                        "wia_range": None,
                        "civilian_kia_range": None
                    }

            if not norm["damage"] and isinstance(data.get("bda_material"), list):
                dm = []
                for item in data.get("bda_material", []):
                    dm.append({"category": "unspecified", "count_range": str(item), "area": None})
                norm["damage"] = dm

            return norm
        except Exception as e:
            logger.warning(f"⚠️ WG NORMALIZE: Failed to normalize payload: {e}")
            return data

    def _format_user_message(self, structured: WargamingResponse) -> str:
        lines: List[str] = []
        stance_value = getattr(structured.stance, "value", str(structured.stance)).lower()
        status_value = getattr(structured.task_status, "value", str(structured.task_status)).lower()
        lines.append(f"Stance: {stance_value} | Escalation: {structured.escalation_ladder_position}/10 | Confidence: {structured.confidence_level:.2f}")
        lines.append(structured.turn_summary)

        # BDA block (bold heading). Prefer structured fields only (no regex)
        bda_lines: List[str] = []
        if getattr(structured, "casualty_estimates", None):
            ce = structured.casualty_estimates
            try:
                if getattr(ce, "kia_range", None):
                    bda_lines.append(f"KIA: {ce.kia_range}")
                if getattr(ce, "wia_range", None):
                    bda_lines.append(f"WIA: {ce.wia_range}")
                if getattr(ce, "civilian_kia_range", None):
                    bda_lines.append(f"CIV KIA: {ce.civilian_kia_range}")
            except Exception:
                pass
        if getattr(structured, "damage", None):
            for d in structured.damage:
                try:
                    cat = getattr(d, "category", None)
                    cnt = getattr(d, "count_range", None)
                    area = getattr(d, "area", None)
                    parts = [str(cat)] if cat else []
                    if cnt:
                        parts.append(str(cnt))
                    if area:
                        parts.append(str(area))
                    if parts:
                        bda_lines.append(" | ".join(parts))
                except Exception:
                    continue
        # Add per-actor BDA if present
        bda_by_actor = getattr(structured, "bda_by_actor", {}) or {}
        if isinstance(bda_by_actor, dict) and bda_by_actor:
            for actor, actor_bda in bda_by_actor.items():
                if not isinstance(actor, str) or not isinstance(actor_bda, dict):
                    continue
                cas = actor_bda.get("casualties")
                mat = actor_bda.get("material") or []
                header_added = False
                if cas and isinstance(cas, str) and cas.strip():
                    if not header_added:
                        bda_lines.append(f"{actor}:")
                        header_added = True
                    bda_lines.append(f"  - Casualties: {cas.strip()}")
                if isinstance(mat, list):
                    for mm in mat:
                        if isinstance(mm, str) and mm.strip():
                            if not header_added:
                                bda_lines.append(f"{actor}:")
                                header_added = True
                            bda_lines.append(f"  - {mm.strip()}")
        # No heuristic fallback; rely on structured fields only
        if bda_lines:
            lines.append("**BDA:**")
            for entry in bda_lines:
                lines.append(f"- {entry}")

        # Actions by actor (preferred), otherwise adversary actions
        actions_by_actor = getattr(structured, "actions_by_actor", {}) or {}
        if isinstance(actions_by_actor, dict) and actions_by_actor:
            for actor, acts in actions_by_actor.items():
                if isinstance(actor, str) and isinstance(acts, list) and acts:
                    lines.append(f"**Actions Taken ({actor}):**")
                    for a in self._prioritize_actions([str(x) for x in acts]):
                        prefix = self._action_prefix(a)
                        lines.append(f"- {prefix}{a}")
        elif structured.actions_taken:
            lines.append("**Actions Taken (Adversary):**")
            for a in self._prioritize_actions(structured.actions_taken):
                prefix = self._action_prefix(a)
                lines.append(f"- {prefix}{a}")
        if structured.next_options:
            lines.append("**Next Options:**")
            for o in structured.next_options:
                lines.append(f"- {o}")
        if getattr(structured, "third_party_responses", None):
            lines.append("**Third-Party Responses:**")
            for tp in structured.third_party_responses:
                try:
                    actor = getattr(tp, "actor", None) or (tp.get("actor") if isinstance(tp, dict) else None)
                    stance = getattr(tp, "stance", None) or (tp.get("stance") if isinstance(tp, dict) else None)
                    likelihood = getattr(tp, "likelihood", None) or (tp.get("likelihood") if isinstance(tp, dict) else None)
                    triggers = getattr(tp, "trigger_conditions", None) or (tp.get("trigger_conditions") if isinstance(tp, dict) else [])
                    actions = getattr(tp, "planned_actions", None) or (tp.get("planned_actions") if isinstance(tp, dict) else [])
                    stance_val = getattr(stance, "value", stance)
                    likelihood_val = getattr(likelihood, "value", likelihood)
                    header = f"- {actor}: stance={stance_val}, likelihood={likelihood_val}"
                    lines.append(header)
                    if isinstance(triggers, list) and triggers:
                        lines.append("  triggers:")
                        for tr in triggers:
                            if isinstance(tr, str) and tr.strip():
                                lines.append(f"  - {tr.strip()}")
                    if isinstance(actions, list) and actions:
                        lines.append("  actions:")
                        for ac in actions:
                            if isinstance(ac, str) and ac.strip():
                                lines.append(f"  - {ac.strip()}")
                except Exception:
                    # Fallback to simple string rendering
                    lines.append(f"- {tp}")
        # Optional weapons consistency note for analyst confidence
        if getattr(structured, "weapons_consistency_check", None):
            lines.append(f"Consistency: {structured.weapons_consistency_check}")

        # Advice (bold heading) - only display if present; does not imply state advancement
        advice = getattr(structured, "advice", []) or []
        if isinstance(advice, list) and advice:
            lines.append("**Advice:**")
            for a in advice:
                try:
                    q = a.get("question") if isinstance(a, dict) else getattr(a, "question", None)
                    rec = a.get("recommendation") if isinstance(a, dict) else getattr(a, "recommendation", None)
                    rat = a.get("rationale") if isinstance(a, dict) else getattr(a, "rationale", None)
                    risks = a.get("risks") if isinstance(a, dict) else getattr(a, "risks", [])
                    legal = a.get("legal") if isinstance(a, dict) else getattr(a, "legal", None)
                    alts = a.get("alternatives") if isinstance(a, dict) else getattr(a, "alternatives", [])
                    tf = a.get("timeframe") if isinstance(a, dict) else getattr(a, "timeframe", None)
                    conf = a.get("confidence") if isinstance(a, dict) else getattr(a, "confidence", None)
                    if q:
                        lines.append(f"- question: {q}")
                    head = f"- recommendation: {rec}" if rec else "- recommendation:"
                    if tf:
                        head += f" | {tf}"
                    if conf is not None:
                        head += f" | confidence {conf:.2f}"
                    lines.append(head)
                    if rat:
                        lines.append(f"  - rationale: {rat}")
                    if isinstance(risks, list) and risks:
                        lines.append("  - risks:")
                        for r in risks:
                            if isinstance(r, str) and r.strip():
                                lines.append(f"    - {r.strip()}")
                    if legal:
                        lines.append(f"  - legal: {legal}")
                    if isinstance(alts, list) and alts:
                        lines.append("  - alternatives:")
                        for alt in alts:
                            if isinstance(alt, str) and alt.strip():
                                lines.append(f"    - {alt.strip()}")
                except Exception:
                    continue

        # Include OOB changes for transparency (succinct)
        oob_delta = getattr(structured, "oob_delta", {}) or {}
        if isinstance(oob_delta, dict) and oob_delta:
            lines.append("OOB changes:")
            for actor, changes in oob_delta.items():
                if isinstance(actor, str) and isinstance(changes, list) and changes:
                    lines.append(f"- {actor}:")
                    for ch in changes:
                        if isinstance(ch, str) and ch.strip():
                            lines.append(f"  - {ch.strip()}")

        # Timeline (bold heading)
        events = getattr(structured, "events", []) or []
        if isinstance(events, list) and events:
            lines.append("**Timeline:**")
            for ev in events:
                try:
                    tpm = ev.get("t_plus_minutes") if isinstance(ev, dict) else getattr(ev, "t_plus_minutes", None)
                    domain = ev.get("domain") if isinstance(ev, dict) else getattr(ev, "domain", None)
                    action = ev.get("action") if isinstance(ev, dict) else getattr(ev, "action", None)
                    effects_s = ev.get("expected_effects") if isinstance(ev, dict) else getattr(ev, "expected_effects", None)
                    core = f"- [t+{tpm}m][{domain}] {action}" if tpm is not None else f"- [{domain}] {action}"
                    lines.append(core)
                    if effects_s:
                        lines.append(f"  - expected: {effects_s}")
                except Exception:
                    continue

        # Impacts (bold heading)
        impacts = getattr(structured, "impacts", []) or []
        if isinstance(impacts, list) and impacts:
            lines.append("**Impacts:**")
            for imp in impacts:
                try:
                    category = imp.get("category") if isinstance(imp, dict) else getattr(imp, "category", None)
                    metric = imp.get("metric") if isinstance(imp, dict) else getattr(imp, "metric", None)
                    rng = imp.get("range") if isinstance(imp, dict) else getattr(imp, "range", None)
                    timeframe = imp.get("timeframe") if isinstance(imp, dict) else getattr(imp, "timeframe", None)
                    conf = imp.get("confidence") if isinstance(imp, dict) else getattr(imp, "confidence", None)
                    levers = imp.get("levers") if isinstance(imp, dict) else getattr(imp, "levers", [])
                    head = f"- {category}: {metric}"
                    if rng:
                        head += f" ({rng})"
                    if timeframe:
                        head += f" | {timeframe}"
                    if conf is not None:
                        head += f" | confidence {conf:.2f}"
                    lines.append(head)
                    if isinstance(levers, list) and levers:
                        lines.append("  - levers:")
                        for lv in levers:
                            if isinstance(lv, str) and lv.strip():
                                lines.append(f"    - {lv.strip()}")
                except Exception:
                    continue
        if structured.international_law_considerations:
            lines.append(f"Legal considerations: {structured.international_law_considerations}")
        # Leaders (bold heading)
        leaders = getattr(structured, "leaders", []) or []
        if isinstance(leaders, list) and leaders:
            lines.append("**Leaders (Continuity):**")
            for ld in leaders:
                try:
                    country = ld.get("country") if isinstance(ld, dict) else getattr(ld, "country", None)
                    name = ld.get("leader_name") if isinstance(ld, dict) else getattr(ld, "leader_name", None)
                    role = ld.get("role") if isinstance(ld, dict) else getattr(ld, "role", None)
                    status = ld.get("status") if isinstance(ld, dict) else getattr(ld, "status", None)
                    cog = ld.get("cog_active") if isinstance(ld, dict) else getattr(ld, "cog_active", None)
                    head = f"- {country}: {name} ({role}) — status={status}"
                    if cog is not None:
                        head += f", CoG={'active' if cog else 'inactive'}"
                    lines.append(head)
                except Exception:
                    continue
        lines.append(f"Status: {status_value}")
        return "\n".join(lines)

    def _prioritize_actions(self, actions: List[str]) -> List[str]:
        try:
            if not actions:
                return actions
            attack_keywords = [
                "attack", "strike", "retaliat", "bombard", "missile", "kinetic",
                "airstrike", "counterstrike", "offensive cyber", "ddos", "sabotage",
                "destroy", "neutralize"
            ]
            priority = []
            rest = []
            for a in actions:
                al = a.lower()
                if any(k in al for k in attack_keywords):
                    priority.append(a)
                else:
                    rest.append(a)
            return priority + rest
        except Exception:
            return actions

    def _is_kinetic(self, text: str) -> bool:
        try:
            t = (text or "").lower()
            keywords = [
                "attack", "strike", "retaliat", "bombard", "missile", "kinetic",
                "airstrike", "counterstrike", "artillery", "mlrs", "rocket", "shell",
                "drone strike", "uav strike", "saboteur", "sabotage", "destroy", "neutralize"
            ]
            return any(k in t for k in keywords)
        except Exception:
            return False

    def _action_prefix(self, text: str) -> str:
        try:
            t = (text or "").lower()
            if self._is_kinetic(t):
                return "🟥 **KINETIC**: "
            cyber = ["cyber", "ddos", "malware", "ransomware", "phishing", "intrusion", "exfiltration"]
            info = ["information ops", "information warfare", "propaganda", "influence", "media", "psyop", "psychological"]
            if any(k in t for k in cyber):
                return "🟦 **CYBER**: "
            if any(k in t for k in info):
                return "🟨 **INFORMATION**: "
            return ""
        except Exception:
            return ""

    def _validate_capability_consistency(self, history: List[Dict[str, Any]]) -> str:
        try:
            if not history:
                return ""
            # Very light validation context: summarize last OOB deltas to remind model of constraints
            last = history[-1]
            oob = last.get("oob_delta") if isinstance(last, dict) else None
            if isinstance(oob, dict) and oob:
                lines = ["CAPABILITY CONSTRAINTS:"]
                for actor, changes in oob.items():
                    if isinstance(actor, str) and isinstance(changes, list) and changes:
                        lines.append(f"- {actor}:")
                        for ch in changes[:5]:
                            if isinstance(ch, str) and ch.strip():
                                lines.append(f"  - {ch.strip()}")
                return "\n".join(lines)
            return ""
        except Exception:
            return ""

    def _make_error(self, error_type: str, message: str):
        from .base_agent import AgentError
        return AgentError(error_type=error_type, message=message, recovery_actions=["Retry turn", "Simplify prompt"]) 

    def _infer_player_side(self, user_text: str, existing_side: Optional[str]) -> Optional[str]:
        try:
            if existing_side:
                return existing_side
            t = (user_text or "").lower()
            mapping = {
                "us": ["i am the us", "i'm the us", "united states", "usa", "u.s."],
                "russia": ["i am russia", "russian federation"],
                "china": ["i am china", "people's republic of china", "prc"],
                "iran": ["i am iran", "islamic republic of iran"],
                "uk": ["i am the uk", "i am britain", "united kingdom"],
                "france": ["i am france"],
                "germany": ["i am germany"],
                "india": ["i am india"],
                "japan": ["i am japan"],
            }
            for side, cues in mapping.items():
                if any(cue in t for cue in cues):
                    return side.upper()
            return None
        except Exception:
            return existing_side

    def _build_world_context(self, player_side: Optional[str], last_snapshot: Dict[str, Any]) -> str:
        try:
            stance = last_snapshot.get("stance")
            esc = last_snapshot.get("escalation_ladder_position")
            last = last_snapshot.get("last_turn_summary")
            parts: List[str] = []
            parts.append("WARGAME CONTEXT:")
            parts.append(f"- Player side: {player_side or 'unspecified'}")
            if esc is not None:
                parts.append(f"- Escalation ladder: {esc}/10")
            if stance:
                parts.append(f"- Adversary last stance: {stance}")
            if last:
                parts.append(f"- Last turn summary: {last}")
            parts.append("Always maintain continuity with this context unless the user explicitly changes sides.")
            return "\n".join(parts)
        except Exception:
            return ""

    def _build_history_context(self, history: List[Dict[str, Any]]) -> str:
        try:
            if not history:
                return ""
            recent = history[-30:]
            lines: List[str] = []
            lines.append("WARGAME HISTORY (most recent first, up to 30 turns):")
            for item in reversed(recent):
                ts = item.get("timestamp", "")
                ua = item.get("user_action", "")
                stance = item.get("adversary_stance", "")
                esc = item.get("escalation_ladder_position", "")
                lines.append(f"- {ts}: USER → {ua}")
                lines.append(f"  Adversary: stance={stance}, escalation={esc}/10")
            return "\n".join(lines)
        except Exception:
            return ""

    # Legacy capability derivation removed; replaced by _validate_capability_consistency
    pass


