"""
gRPC Context Gatherer - Comprehensive Context Assembly for llm-orchestrator

Gathers all necessary context from backend state and assembles it into
gRPC proto messages for sending to llm-orchestrator.

This is the SINGLE SOURCE OF TRUTH for what context gets sent to llm-orchestrator.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from protos import orchestrator_pb2
from services.prompt_service import prompt_service

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 if either has zero norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _build_user_memory_for_prompt(user_id: str, grpc_request) -> str:
    """
    Single USER MEMORY block: facts + episodic activity, one query embedding when pruning.
    Respects facts_inject_enabled and episodes_inject_enabled independently.
    """
    from services.episode_service import episode_service
    from services.settings_service import settings_service

    facts_inject = await settings_service.get_facts_inject_enabled(user_id)
    episodes_inject = await settings_service.get_episodes_inject_enabled(user_id)

    facts: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    try:
        if facts_inject:
            facts = await settings_service.get_user_facts(user_id) or []
        if episodes_inject:
            episodes = await episode_service.get_user_episodes(user_id, limit=50, days=30) or []
    except Exception as e:
        logger.warning("USER MEMORY: load failed: %s", e)
        return ""

    total = len(facts) + len(episodes)
    if total == 0:
        return ""

    query = (getattr(grpc_request, "query", None) or "").strip()
    facts_sel = list(facts)
    episodes_sel = list(episodes)

    if total > 15 and query:
        try:
            from services.embedding_service_wrapper import get_embedding_service

            emb_svc = await get_embedding_service()
            query_embeddings = await emb_svc.generate_embeddings([query])
            qvec = query_embeddings[0] if query_embeddings else None
            if qvec:
                ranked: List[tuple] = []
                for f in facts:
                    emb = f.get("embedding")
                    if emb:
                        s = _cosine_similarity(emb, qvec)
                        if s >= 0.32:
                            ranked.append(("fact", f, s))
                for e in episodes:
                    emb = e.get("embedding")
                    if emb:
                        s = _cosine_similarity(emb, qvec)
                        if e.get("is_aged"):
                            s *= 0.88
                        if s >= 0.28:
                            ranked.append(("episode", e, s))
                ranked.sort(key=lambda x: -x[2])
                facts_sel = []
                episodes_sel = []
                max_f, max_e = 12, 6
                for kind, obj, _s in ranked:
                    if kind == "fact" and len(facts_sel) < max_f and obj not in facts_sel:
                        facts_sel.append(obj)
                    elif kind == "episode" and len(episodes_sel) < max_e and obj not in episodes_sel:
                        episodes_sel.append(obj)
                for f in facts:
                    if not f.get("embedding") and f not in facts_sel:
                        facts_sel.append(f)
                for e in episodes:
                    if not e.get("embedding") and e not in episodes_sel:
                        episodes_sel.append(e)
        except Exception as emb_err:
            logger.warning("USER MEMORY: relevance filter failed, using full lists: %s", emb_err)

    parts: List[str] = ["USER MEMORY:\n"]
    if facts_inject and facts_sel:
        try:
            ft = settings_service.format_user_facts_for_prompt(facts_sel)
            if ft.strip():
                inner = ft.strip()
                if inner.startswith("USER FACTS:"):
                    inner = inner.replace("USER FACTS:", "Known facts:", 1).strip()
                else:
                    inner = "Known facts:\n" + inner
                parts.append(inner)
        except Exception as fe:
            logger.warning("USER MEMORY: format facts failed: %s", fe)
    if episodes_inject and episodes_sel:
        try:
            et = episode_service.format_episodes_for_prompt(episodes_sel)
            if et.strip():
                block = et.strip()
                if block.startswith("RECENT ACTIVITY:"):
                    block = "Recent activity:\n" + block[len("RECENT ACTIVITY:") :].lstrip()
                else:
                    block = "Recent activity:\n" + block
                if len(parts) > 1:
                    parts.append("")
                parts.append(block)
        except Exception as ee:
            logger.warning("USER MEMORY: format episodes failed: %s", ee)

    if len(parts) <= 1:
        return ""
    return "\n\n".join(parts)


async def _resolve_agent_handle(handle: str, user_id: str) -> Optional[str]:
    """Resolve an @handle to an agent_profile_id. Returns None if not found."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_profiles WHERE user_id = $1 AND handle = $2 AND is_active = true",
        user_id,
        handle,
    )
    return str(row["id"]) if row else None


async def _resolve_team_handle(handle: str, user_id: str) -> Optional[str]:
    """Resolve an @handle to an agent_lines id. Returns None if not found."""
    from services.database_manager.database_helpers import fetch_one
    row = await fetch_one(
        "SELECT id FROM agent_lines WHERE user_id = $1 AND handle = $2",
        user_id,
        handle,
    )
    return str(row["id"]) if row else None


async def _inject_line_capability_metadata(
    grpc_request: orchestrator_pb2.ChatRequest,
    user_id: str,
    line_id: str,
    member_agent_profile_id: str,
) -> None:
    """Inject team tool packs, reference config, and membership additional_tools for a line member."""
    import json as _json

    try:
        from services.database_manager.database_helpers import fetch_one
        from services.agent_line_service import _normalize_pack_entries

        team_row = await fetch_one(
            "SELECT team_tool_packs, team_skill_ids, reference_config, data_workspace_config FROM agent_lines WHERE id = $1",
            str(line_id),
        )
        if team_row:
            raw = team_row.get("team_tool_packs") or []
            grpc_request.metadata["team_tool_packs"] = _json.dumps(_normalize_pack_entries(raw))
            grpc_request.metadata["team_skill_ids"] = _json.dumps(team_row.get("team_skill_ids") or [])
            ref_cfg = team_row.get("reference_config") or {}
            if isinstance(ref_cfg, str):
                try:
                    ref_cfg = _json.loads(ref_cfg)
                except (ValueError, TypeError):
                    ref_cfg = {}
            if not isinstance(ref_cfg, dict):
                ref_cfg = {}
            grpc_request.metadata["line_reference_config"] = _json.dumps(ref_cfg)
            dw_cfg = team_row.get("data_workspace_config") or {}
            if isinstance(dw_cfg, str):
                try:
                    dw_cfg = _json.loads(dw_cfg)
                except (ValueError, TypeError):
                    dw_cfg = {}
            if not isinstance(dw_cfg, dict):
                dw_cfg = {}
            grpc_request.metadata["line_data_workspace_config"] = _json.dumps(dw_cfg)
        membership_row = await fetch_one(
            "SELECT additional_tools FROM agent_line_memberships "
            "WHERE line_id = $1 AND agent_profile_id = $2",
            str(line_id),
            str(member_agent_profile_id),
        )
        if membership_row:
            raw_tools = membership_row.get("additional_tools") or []
            if isinstance(raw_tools, str):
                try:
                    raw_tools = _json.loads(raw_tools)
                except (ValueError, TypeError):
                    raw_tools = []
            if not isinstance(raw_tools, list):
                raw_tools = []
            grpc_request.metadata["member_additional_tools"] = _json.dumps(raw_tools)
    except Exception as e:
        if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
            logger.warning("Failed to inject line capability metadata: %s", e)


async def _apply_profile_model_preference(
    grpc_request: orchestrator_pb2.ChatRequest,
    user_id: str,
    profile_id: str,
    requested: str,
    profile_chat_model_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Resolve profile model_preference with soft retarget, notify, persist, and set metadata."""
    from services.model_configuration_notifier import (
        maybe_notify_model_configuration_issue,
        persist_agent_profile_model_preference,
    )
    from services.model_source_resolver import pick_fallback_model_id, try_soft_retarget

    if profile_chat_model_state is not None:
        profile_chat_model_state["applied"] = True

    pre_sidebar_model = grpc_request.metadata.get("user_chat_model")
    retarget = await try_soft_retarget(user_id, requested)
    available = bool(retarget.get("available"))
    effective = retarget.get("model_id") or requested
    retargeted = bool(retarget.get("retargeted"))

    if available:
        grpc_request.metadata["user_chat_model"] = effective
        if retargeted or requested != effective:
            await persist_agent_profile_model_preference(user_id, profile_id, effective)
            await maybe_notify_model_configuration_issue(
                user_id,
                title="Agent model updated",
                preview=(
                    f"Saved model «{requested}» is not usable with current settings. "
                    f"Preference updated to «{effective}»."
                )[:500],
                dedupe_key=f"profile:{profile_id}:{requested}->{effective}",
                agent_profile_id=profile_id,
                requested_model=requested,
                effective_model=effective,
            )
        logger.info("CONTEXT: Using Agent Factory profile model (soft retarget): %s", effective)
        return

    await persist_agent_profile_model_preference(user_id, profile_id, None)
    emergency = await pick_fallback_model_id(user_id)
    if emergency:
        preview = (
            f"Agent saved model «{requested}» is unavailable. Preference cleared. Using «{emergency}» for this request."
        )
    else:
        preview = (
            f"Agent saved model «{requested}» is unavailable. Preference cleared. "
            "Configure enabled models in Settings > AI Models."
        )
    await maybe_notify_model_configuration_issue(
        user_id,
        title="Agent model unavailable",
        preview=preview[:500],
        dedupe_key=f"profile:{profile_id}:{requested}:cleared",
        agent_profile_id=profile_id,
        requested_model=requested,
        effective_model=emergency,
    )
    if pre_sidebar_model:
        grpc_request.metadata["user_chat_model"] = pre_sidebar_model
    elif emergency:
        grpc_request.metadata["user_chat_model"] = emergency
    else:
        grpc_request.metadata.pop("user_chat_model", None)
    logger.info(
        "CONTEXT: Profile model unavailable; fallback metadata user_chat_model=%s",
        grpc_request.metadata.get("user_chat_model"),
    )


class GRPCContextGatherer:
    """
    Assembles comprehensive context for llm-orchestrator requests
    
    Responsibilities:
    - Extract conversation history from LangGraph state
    - Build user persona from settings
    - Conditionally add editor context (when on editor page)
    - Conditionally add pipeline context (when on pipeline page)
    - Add permission grants from shared memory
    - Add pending operations from state
    - Add routing locks and other context
    """
    
    def __init__(self):
        self.prompt_service = prompt_service

    def _parse_message_dicts_to_langchain(self, message_dicts: List[Dict[str, Any]]) -> List[Any]:
        """Convert conversation message dicts from DB/API into LangChain messages with sequence_number and created_at."""
        from langchain_core.messages import HumanMessage, AIMessage

        messages: List[Any] = []
        for msg_dict in message_dicts:
            role = msg_dict.get("message_type", "user")
            content = msg_dict.get("content", "")
            seq = msg_dict.get("sequence_number")
            created_at = msg_dict.get("created_at", "") or ""
            kwargs: Dict[str, Any] = {}
            if seq is not None:
                try:
                    kwargs["sequence_number"] = int(seq)
                except (TypeError, ValueError):
                    kwargs["sequence_number"] = 0
            if created_at:
                kwargs["created_at"] = created_at
            if role == "assistant":
                msg_metadata = msg_dict.get("metadata_json") or msg_dict.get("metadata") or {}
                tool_summary = msg_metadata.get("tool_call_summary", "") if isinstance(msg_metadata, dict) else ""
                if tool_summary:
                    kwargs["tool_call_summary"] = tool_summary
            if role == "user":
                messages.append(HumanMessage(content=content, additional_kwargs=kwargs))
            elif role == "assistant":
                messages.append(AIMessage(content=content, additional_kwargs=kwargs))
        return messages

    def _append_langchain_messages_to_proto(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        recent_messages: List[Any],
    ) -> None:
        """Append LangChain messages to conversation_history with sequence_number metadata and real timestamps."""
        for msg in recent_messages:
            if not (hasattr(msg, "content") and hasattr(msg, "type")):
                continue
            role = "user" if msg.type == "human" else "assistant"
            ak = getattr(msg, "additional_kwargs", None) or {}
            seq = ak.get("sequence_number")
            meta: Dict[str, str] = {}
            if seq is not None:
                meta["sequence_number"] = str(int(seq))
            if role == "assistant":
                tcs = ak.get("tool_call_summary")
                if isinstance(tcs, str) and tcs.strip():
                    meta["tool_call_summary"] = tcs
            created = ak.get("created_at") or ""
            ts = created if isinstance(created, str) and created.strip() else datetime.now().isoformat()
            grpc_request.conversation_history.append(
                orchestrator_pb2.ConversationMessage(
                    role=role,
                    content=msg.content,
                    timestamp=ts,
                    metadata=meta,
                )
            )

    async def build_chat_request(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
        session_id: str = "default",
        request_context: Optional[Dict[str, Any]] = None,
        state: Optional[Dict[str, Any]] = None,
        agent_type: Optional[str] = None,
        routing_reason: Optional[str] = None
    ) -> orchestrator_pb2.ChatRequest:
        """
        Build comprehensive ChatRequest for llm-orchestrator
        
        Args:
            query: Current user query
            user_id: User UUID
            conversation_id: Conversation UUID
            session_id: Session identifier
            request_context: Frontend request context (active_editor, pipeline, etc.)
            state: LangGraph conversation state (if available)
            agent_type: Optional explicit agent routing
            routing_reason: Optional reason for routing decision
            
        Returns:
            Fully populated ChatRequest proto message
        """
        try:
            logger.info(f"🔧 CONTEXT GATHERER: Building gRPC request for user {user_id}")
            
            # Create base request with core fields
            grpc_request = orchestrator_pb2.ChatRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                query=query,
                session_id=session_id
            )
            profile_chat_model_state: Dict[str, Any] = {"applied": False}

            # Add routing control if specified
            if agent_type:
                grpc_request.agent_type = agent_type
                logger.info(f"🎯 CONTEXT GATHERER: Explicit routing to {agent_type}")

            if routing_reason:
                grpc_request.routing_reason = routing_reason

            # Initialize request context if not provided
            request_context = request_context or {}
            # Canonical key for agent line UUID is line_id; accept legacy team_id from older callers.
            _line = request_context.get("line_id") or request_context.get("team_id")
            if _line:
                _line = str(_line).strip()
                request_context["line_id"] = _line
                request_context["team_id"] = _line  # mirror for legacy metadata consumers

            # Check if models are properly configured
            models_configured = request_context.get("models_configured", True)  # Default to True for backward compatibility
            if not models_configured:
                grpc_request.metadata["models_not_configured_warning"] = "AI models are using system fallback defaults. Consider configuring specific models in Settings > AI Models for better performance and consistency."
                logger.warning("⚠️ CONTEXT GATHERER: Models not explicitly configured - using fallbacks")
            # Before context_limit calculation, resolve sticky agent_profile_id so lookback uses it
            sticky_profile_id = None
            if not request_context.get("agent_profile_id"):
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    agent_metadata = await conversation_service.get_agent_metadata(
                        conversation_id=conversation_id, user_id=user_id
                    )
                    if agent_metadata:
                        sticky_profile_id = agent_metadata.get("agent_profile_id")
                except Exception:
                    pass
            if not request_context.get("agent_profile_id") and not sticky_profile_id:
                try:
                    from services.database_manager.database_helpers import fetch_one
                    from services.user_settings_kv_service import get_user_setting

                    default_pid = None
                    pref = await get_user_setting(user_id, "default_chat_agent_profile_id")
                    if pref:
                        custom_row = await fetch_one(
                            """
                            SELECT id FROM agent_profiles
                            WHERE id = $1::uuid AND user_id = $2
                              AND is_active = true
                              AND COALESCE(is_builtin, false) = false
                            LIMIT 1
                            """,
                            pref,
                            user_id,
                        )
                        if custom_row and custom_row.get("id"):
                            default_pid = str(custom_row["id"])
                    if not default_pid:
                        builtin_row = await fetch_one(
                            "SELECT id FROM agent_profiles WHERE user_id = $1 AND is_builtin = true AND is_active = true LIMIT 1",
                            user_id,
                        )
                        if builtin_row and builtin_row.get("id"):
                            default_pid = str(builtin_row["id"])
                    if default_pid:
                        grpc_request.metadata["default_agent_profile_id"] = default_pid
                except Exception as e:
                    if "is_builtin" not in str(e) and "column" not in str(e).lower():
                        logger.warning("Failed to load default agent profile: %s", e)
            context_limit = request_context.get("context_window_size")
            if context_limit is None and (request_context.get("agent_profile_id") or sticky_profile_id):
                profile_id_for_lookback = request_context.get("agent_profile_id") or sticky_profile_id
                try:
                    from services.database_manager.database_helpers import fetch_one
                    profile_row = await fetch_one(
                        "SELECT chat_history_enabled, chat_history_lookback FROM agent_profiles WHERE id = $1",
                        str(profile_id_for_lookback),
                    )
                    if profile_row:
                        if profile_row.get("chat_history_enabled") and profile_row.get("chat_history_lookback") is not None:
                            context_limit = int(profile_row["chat_history_lookback"])
                            logger.info("CONTEXT: Using custom agent chat_history_lookback: %s", context_limit)
                        elif profile_row.get("chat_history_enabled") is False:
                            context_limit = 0
                            logger.info("CONTEXT: Agent profile has chat_history_enabled=False, sending no history")
                except Exception as e:
                    if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                        logger.warning("Failed to load agent profile chat_history_lookback: %s", e)
            if context_limit is None:
                context_limit = 20
            context_limit = int(context_limit)
            grpc_request.metadata["context_window_size"] = str(context_limit)
            if request_context.get("user_chat_model"):
                grpc_request.metadata["user_chat_model"] = request_context["user_chat_model"]
            if request_context.get("agent_profile_id"):
                grpc_request.metadata["agent_profile_id"] = str(request_context["agent_profile_id"])
                if request_context.get("trigger_input") is not None:
                    grpc_request.metadata["trigger_input"] = str(request_context["trigger_input"])
                if request_context.get("execution_id") is not None:
                    grpc_request.metadata["execution_id"] = str(request_context["execution_id"])
                if request_context.get("trigger_type") is not None:
                    grpc_request.metadata["trigger_type"] = str(request_context["trigger_type"])
                if request_context.get("resume_approval_id") is not None:
                    grpc_request.metadata["resume_approval_id"] = str(request_context["resume_approval_id"])
                if request_context.get("resume_playbook_config_json") is not None:
                    grpc_request.metadata["resume_playbook_config_json"] = str(request_context["resume_playbook_config_json"])
                for team_key in ("line_id", "team_id", "post_id", "post_author_id", "respond_as"):
                    if request_context.get(team_key) is not None:
                        grpc_request.metadata[team_key] = str(request_context[team_key])
                # Inject team-level capability configuration when running in agent line context
                line_id = request_context.get("line_id") or request_context.get("team_id")
                if line_id and request_context.get("agent_profile_id"):
                    import json as _json
                    try:
                        from services.database_manager.database_helpers import fetch_one
                        team_row = await fetch_one(
                            "SELECT team_tool_packs, team_skill_ids, reference_config, data_workspace_config FROM agent_lines WHERE id = $1",
                            str(line_id),
                        )
                        if team_row:
                            from services.agent_line_service import _normalize_pack_entries
                            raw = team_row.get("team_tool_packs") or []
                            grpc_request.metadata["team_tool_packs"] = _json.dumps(_normalize_pack_entries(raw))
                            grpc_request.metadata["team_skill_ids"] = _json.dumps(team_row.get("team_skill_ids") or [])
                            ref_cfg = team_row.get("reference_config") or {}
                            if isinstance(ref_cfg, str):
                                try:
                                    ref_cfg = _json.loads(ref_cfg)
                                except (ValueError, TypeError):
                                    ref_cfg = {}
                            if not isinstance(ref_cfg, dict):
                                ref_cfg = {}
                            grpc_request.metadata["line_reference_config"] = _json.dumps(ref_cfg)
                            dw_cfg = team_row.get("data_workspace_config") or {}
                            if isinstance(dw_cfg, str):
                                try:
                                    dw_cfg = _json.loads(dw_cfg)
                                except (ValueError, TypeError):
                                    dw_cfg = {}
                            if not isinstance(dw_cfg, dict):
                                dw_cfg = {}
                            grpc_request.metadata["line_data_workspace_config"] = _json.dumps(dw_cfg)
                        membership_row = await fetch_one(
                            "SELECT additional_tools FROM agent_line_memberships "
                            "WHERE line_id = $1 AND agent_profile_id = $2",
                            str(line_id),
                            str(request_context["agent_profile_id"]),
                        )
                        if membership_row:
                            raw_tools = membership_row.get("additional_tools") or []
                            if isinstance(raw_tools, str):
                                try:
                                    raw_tools = _json.loads(raw_tools)
                                except (ValueError, TypeError):
                                    raw_tools = []
                            if not isinstance(raw_tools, list):
                                raw_tools = []
                            grpc_request.metadata["member_additional_tools"] = _json.dumps(raw_tools)
                    except Exception as e:
                        if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                            logger.warning("Failed to load team capability config: %s", e)
                try:
                    from services.database_manager.database_helpers import fetch_all
                    import json as _json
                    rows = await fetch_all(
                        "SELECT plugin_name, credentials_encrypted FROM agent_plugin_configs "
                        "WHERE agent_profile_id = $1 AND is_enabled = true",
                        str(request_context["agent_profile_id"]),
                    )
                    if rows:
                        plugin_creds = {
                            r["plugin_name"]: (r.get("credentials_encrypted") or {})
                            for r in rows
                        }
                        grpc_request.metadata["plugin_credentials"] = _json.dumps(plugin_creds)
                except Exception as e:
                    if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                        logger.warning("Failed to load plugin configs for profile: %s", e)
                # Agent Factory: use profile's model_preference as user_chat_model when set (with soft retarget)
                try:
                    from services.database_manager.database_helpers import fetch_one

                    profile_row = await fetch_one(
                        "SELECT model_preference FROM agent_profiles WHERE id = $1",
                        str(request_context["agent_profile_id"]),
                    )
                    if profile_row and profile_row.get("model_preference"):
                        await _apply_profile_model_preference(
                            grpc_request,
                            user_id,
                            str(request_context["agent_profile_id"]),
                            profile_row["model_preference"],
                            profile_chat_model_state,
                        )
                except Exception as e:
                    if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                        logger.warning("Failed to load agent profile model_preference: %s", e)
            else:
                query_stripped = (query or "").strip()
                auto_match = re.match(r"^@auto\s+", query_stripped, re.IGNORECASE)
                if auto_match:
                    grpc_request.query = query_stripped[auto_match.end() :].strip()
                    try:
                        from services.conversation_service import ConversationService
                        conversation_service = ConversationService()
                        await conversation_service.update_agent_metadata(
                            conversation_id=grpc_request.conversation_id,
                            user_id=grpc_request.user_id,
                            primary_agent_selected="chat_agent",
                            last_agent=None,
                            clear_agent_profile_id=True,
                            clear_active_line=True,
                        )
                        logger.info("CONTEXT GATHERER: @auto cleared sticky agent and line routing for conversation %s", grpc_request.conversation_id)
                    except Exception as e:
                        logger.warning("Failed to clear agent routing for @auto: %s", e)
                else:
                    mention_match = re.match(r"^@([\w-]+)\s+", query_stripped)
                    if mention_match:
                        handle = mention_match.group(1)
                        profile_id = await _resolve_agent_handle(handle, user_id)
                        if profile_id:
                            grpc_request.metadata["agent_profile_id"] = profile_id
                            grpc_request.metadata["resolved_agent_handle"] = handle
                            grpc_request.query = query_stripped[mention_match.end() :].strip()
                            logger.info("CONTEXT GATHERER: Resolved @%s to agent_profile_id=%s", handle, profile_id)
                            try:
                                from services.database_manager.database_helpers import fetch_all
                                import json as _json
                                rows = await fetch_all(
                                    "SELECT plugin_name, credentials_encrypted FROM agent_plugin_configs "
                                    "WHERE agent_profile_id = $1 AND is_enabled = true",
                                    profile_id,
                                )
                                if rows:
                                    plugin_creds = {
                                        r["plugin_name"]: (r.get("credentials_encrypted") or {})
                                        for r in rows
                                    }
                                    grpc_request.metadata["plugin_credentials"] = _json.dumps(plugin_creds)
                            except Exception as e:
                                if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                                    logger.warning("Failed to load plugin configs for profile: %s", e)
                            try:
                                from services.database_manager.database_helpers import fetch_one

                                profile_row = await fetch_one(
                                    "SELECT model_preference FROM agent_profiles WHERE id = $1",
                                    profile_id,
                                )
                                if profile_row and profile_row.get("model_preference"):
                                    await _apply_profile_model_preference(
                                        grpc_request,
                                        user_id,
                                        profile_id,
                                        profile_row["model_preference"],
                                        profile_chat_model_state,
                                    )
                            except Exception as e:
                                if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                                    logger.warning("Failed to load agent profile model_preference: %s", e)
                            try:
                                from services.database_manager.database_helpers import fetch_one
                                profile_row = await fetch_one(
                                    "SELECT chat_history_enabled, chat_history_lookback FROM agent_profiles WHERE id = $1",
                                    profile_id,
                                )
                                if profile_row:
                                    if profile_row.get("chat_history_enabled") and profile_row.get("chat_history_lookback") is not None:
                                        context_limit = int(profile_row["chat_history_lookback"])
                                        grpc_request.metadata["context_window_size"] = str(context_limit)
                                        logger.info("CONTEXT: Using custom agent chat_history_lookback (from @mention): %s", context_limit)
                                    elif profile_row.get("chat_history_enabled") is False:
                                        context_limit = 0
                                        grpc_request.metadata["context_window_size"] = "0"
                                        logger.info("CONTEXT: @mention profile has chat_history_enabled=False, sending no history")
                            except Exception as e:
                                if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                                    logger.warning("Failed to load agent profile chat_history_lookback: %s", e)
                        else:
                            team_id = await _resolve_team_handle(handle, user_id)
                            if team_id:
                                from services import agent_line_service
                                grpc_request.metadata["team_context_id"] = team_id
                                grpc_request.metadata["resolved_team_handle"] = handle
                                grpc_request.query = query_stripped[mention_match.end() :].strip()
                                try:
                                    team_chat_context = await agent_line_service.get_line_chat_context(team_id, user_id)
                                    grpc_request.metadata["team_chat_context"] = team_chat_context
                                except Exception as e:
                                    logger.warning("Failed to load team chat context for @%s: %s", handle, e)
                                    grpc_request.metadata["team_chat_context"] = "Team context unavailable."
                                logger.info("CONTEXT GATHERER: Resolved @%s to team_context_id=%s", handle, team_id)
                                ceo = await agent_line_service.get_ceo_agent_for_heartbeat(team_id)
                                if ceo and ceo.get("agent_profile_id"):
                                    grpc_request.metadata["line_dispatch_mode"] = "true"
                                    grpc_request.metadata["ceo_profile_id"] = str(ceo["agent_profile_id"])
                                    grpc_request.metadata["line_id"] = str(team_id)
                                    grpc_request.metadata["team_id"] = str(team_id)
                                    await _inject_line_capability_metadata(
                                        grpc_request, user_id, str(team_id), str(ceo["agent_profile_id"])
                                    )
                                    try:
                                        from services.database_manager.database_helpers import fetch_one

                                        profile_row = await fetch_one(
                                            "SELECT model_preference FROM agent_profiles WHERE id = $1",
                                            str(ceo["agent_profile_id"]),
                                        )
                                        if profile_row and profile_row.get("model_preference"):
                                            await _apply_profile_model_preference(
                                                grpc_request,
                                                user_id,
                                                str(ceo["agent_profile_id"]),
                                                profile_row["model_preference"],
                                                profile_chat_model_state,
                                            )
                                    except Exception as e:
                                        if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                                            logger.warning("Failed to load CEO model_preference for line dispatch: %s", e)
                                    try:
                                        from services.database_manager.database_helpers import fetch_all
                                        import json as _json

                                        rows = await fetch_all(
                                            "SELECT plugin_name, credentials_encrypted FROM agent_plugin_configs "
                                            "WHERE agent_profile_id = $1 AND is_enabled = true",
                                            str(ceo["agent_profile_id"]),
                                        )
                                        if rows:
                                            plugin_creds = {
                                                r["plugin_name"]: (r.get("credentials_encrypted") or {})
                                                for r in rows
                                            }
                                            grpc_request.metadata["plugin_credentials"] = _json.dumps(plugin_creds)
                                    except Exception as e:
                                        if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                                            logger.warning("Failed to load plugin configs for CEO profile: %s", e)
                                    try:
                                        from services.celery_tasks.team_heartbeat_context import (
                                            TEAM_TOOL_IDS_RULE,
                                            _build_heartbeat_context,
                                        )

                                        hb = await _build_heartbeat_context(
                                            str(team_id), user_id, str(ceo["agent_profile_id"])
                                        )
                                        grpc_request.metadata["line_dispatch_briefing"] = hb
                                        grpc_request.metadata["line_dispatch_tool_rules"] = TEAM_TOOL_IDS_RULE
                                    except Exception as e:
                                        logger.warning("Failed to build line dispatch briefing: %s", e)
                                        grpc_request.metadata["line_dispatch_briefing"] = (
                                            grpc_request.metadata.get("team_chat_context") or ""
                                        )
                                        grpc_request.metadata["line_dispatch_tool_rules"] = ""
                                    if grpc_request.conversation_id:
                                        try:
                                            from services.conversation_service import ConversationService

                                            line_row = await agent_line_service.get_line(team_id, user_id)
                                            line_name = (line_row or {}).get("name") or ""
                                            conv_svc = ConversationService()
                                            await conv_svc.update_agent_metadata(
                                                conversation_id=grpc_request.conversation_id,
                                                user_id=user_id,
                                                primary_agent_selected="line_dispatch",
                                                last_agent="line_dispatch",
                                                clear_agent_profile_id=True,
                                                active_line_id=str(team_id),
                                                active_line_name=line_name or None,
                                            )
                                        except Exception as e:
                                            logger.warning("Failed to persist active line on conversation: %s", e)
                                else:
                                    logger.warning(
                                        "CONTEXT GATHERER: Line %s has no CEO (root member); line chat dispatch disabled",
                                        team_id,
                                    )

            # === CONVERSATION HISTORY + ATTACHMENTS (load messages once when not in state) ===
            messages_data: Optional[Dict[str, Any]] = None
            if not (state and "messages" in state):
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)
                    messages_data = await conversation_service.get_conversation_messages(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        skip=0,
                        limit=context_limit,
                        most_recent=True,
                    )
                    if messages_data and messages_data.get("messages"):
                        logger.info(
                            "CONTEXT: Loaded %s messages from database for conversation %s",
                            len(messages_data["messages"]),
                            conversation_id,
                        )
                except Exception as db_error:
                    logger.warning("CONTEXT: Failed to load conversation messages: %s", db_error)

            await self._add_conversation_history(
                grpc_request, state, conversation_id, user_id, messages_data=messages_data, limit=context_limit
            )
            await self._add_attachments(
                grpc_request, conversation_id, user_id, messages_data=messages_data
            )
            
            # === USER PERSONA ===
            await self._add_user_persona(
                grpc_request, user_id, request_context, profile_chat_model_state
            )
            
            # === EDITOR CONTEXT (Conditional) ===
            await self._add_editor_context(grpc_request, request_context)
            
            # === DATA WORKSPACE CONTEXT (Conditional) ===
            self._add_data_workspace_context(grpc_request, request_context)
            
            # === PIPELINE CONTEXT (Conditional) ===
            await self._add_pipeline_context(grpc_request, request_context)
            
            # === PERMISSION GRANTS (Conditional) ===
            await self._add_permission_grants(grpc_request, state)
            
            # === PENDING OPERATIONS (Conditional) ===
            await self._add_pending_operations(grpc_request, state)
            
            # === ROUTING LOCKS (Conditional) ===
            await self._add_routing_locks(grpc_request, request_context, state)
            
            # === CHECKPOINTING (Conditional) ===
            await self._add_checkpoint_info(grpc_request, request_context)
            
            # === IMAGE DESCRIPTION CONTEXT (when agent_type is image_description) ===
            if agent_type == "image_description":
                self._add_image_description_context(grpc_request, request_context)
            # === DOCUMENT DESCRIPTION CONTEXT (when agent_type is document_description) ===
            if agent_type == "document_description":
                self._add_document_description_context(grpc_request, request_context)
            
            # === PRIMARY AGENT SELECTED (for conversation continuity) ===
            await self._add_primary_agent_selected(grpc_request, state)
            
            # Log summary
            self._log_context_summary(grpc_request)
            
            return grpc_request
            
        except Exception as e:
            logger.error(f"❌ CONTEXT GATHERER: Failed to build request: {e}")
            # Return minimal valid request
            return orchestrator_pb2.ChatRequest(
                user_id=user_id,
                conversation_id=conversation_id,
                query=query,
                session_id=session_id
            )
    
    async def _add_conversation_history(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]],
        conversation_id: str,
        user_id: str,
        messages_data: Optional[Dict[str, Any]] = None,
        limit: int = 20,
    ) -> None:
        """Add conversation history to request. Uses state['messages'] if present, else messages_data if provided, else loads from DB."""
        try:
            messages = []

            if state and "messages" in state:
                messages = state["messages"]
            elif messages_data and "messages" in messages_data:
                messages = self._parse_message_dicts_to_langchain(messages_data["messages"])
                logger.info("CONTEXT: Added %s messages to history (from shared messages_data)", len(messages))
            else:
                try:
                    from services.conversation_service import ConversationService
                    conversation_service = ConversationService()
                    conversation_service.set_current_user(user_id)
                    loaded = await conversation_service.get_conversation_messages(
                        conversation_id=conversation_id,
                        user_id=user_id,
                        skip=0,
                        limit=limit,
                        most_recent=True,
                    )
                    if loaded and "messages" in loaded:
                        messages = self._parse_message_dicts_to_langchain(loaded["messages"])
                        logger.info("CONTEXT: Loaded %s messages from database for conversation %s", len(messages), conversation_id)
                except Exception as db_error:
                    logger.warning("CONTEXT: Failed to load conversation history from database: %s", db_error)

            if limit <= 0:
                recent_messages = []
            elif len(messages) > limit:
                recent_messages = messages[-limit:]
            else:
                recent_messages = messages

            self._append_langchain_messages_to_proto(grpc_request, recent_messages)
            
            if len(grpc_request.conversation_history) > 0:
                logger.info(f"✅ CONTEXT: Added {len(grpc_request.conversation_history)} messages to history")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add conversation history: {e}")
    
    async def _add_attachments(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        conversation_id: str,
        user_id: str,
        messages_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract attachments from latest user message and add to request metadata"""
        try:
            attachments = await self._extract_latest_message_attachments(
                conversation_id=conversation_id,
                user_id=user_id,
                messages_data=messages_data,
            )
            
            if attachments:
                import json
                grpc_request.metadata["attachments"] = json.dumps(attachments)
                attached_images = [{"data": a.get("base64_data"), "base64": a.get("base64_data")} for a in attachments if a.get("base64_data")]
                if attached_images:
                    grpc_request.metadata["attached_images"] = json.dumps(attached_images)
                logger.info(f"✅ CONTEXT: Added {len(attachments)} attachment(s) from latest message")
            else:
                logger.debug("📎 CONTEXT: No attachments found in latest message")
                
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to extract attachments: {e}")
    
    async def _extract_latest_message_attachments(
        self,
        conversation_id: str,
        user_id: str,
        messages_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract attachments from the latest user message in the conversation.
        Uses messages_data when provided (avoids duplicate load); otherwise loads from DB.
        Queries conversation_message_attachments when available; falls back to metadata_json.
        For images, reads file and adds base64_data for orchestrator vision/face analysis.
        """
        try:
            import base64
            import json
            from pathlib import Path
            from services.conversation_service import ConversationService
            from services.database_manager.database_helpers import fetch_all

            if not messages_data or "messages" not in messages_data:
                conversation_service = ConversationService()
                conversation_service.set_current_user(user_id)
                messages_data = await conversation_service.get_conversation_messages(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    skip=0,
                    limit=20,
                    most_recent=True,
                )

            if not messages_data or "messages" not in messages_data:
                return []

            messages = messages_data["messages"]
            latest_message_id = None

            for msg in reversed(messages):
                if msg.get("message_type") == "user" or msg.get("role") == "user":
                    latest_message_id = msg.get("message_id")
                    break

            if not latest_message_id:
                return []

            attachments = []

            try:
                rows = await fetch_all(
                    """
                    SELECT attachment_id, filename, content_type, file_size, file_path,
                           uploaded_at, is_image, metadata_json
                    FROM conversation_message_attachments
                    WHERE message_id = $1
                    ORDER BY uploaded_at DESC
                    """,
                    latest_message_id,
                )
                if rows:
                    for row in rows:
                        file_path = row.get("file_path")
                        content_type = row.get("content_type", "")
                        is_image = row.get("is_image", content_type.startswith("image/"))
                        base64_data = None
                        if is_image and file_path and Path(file_path).exists():
                            try:
                                with open(file_path, "rb") as f:
                                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                            except Exception as e:
                                logger.warning(f"Failed to read image for attachment: {e}")
                        uploaded_at = row.get("uploaded_at")
                        attachments.append({
                            "attachment_id": str(row.get("attachment_id", "")),
                            "filename": row.get("filename", ""),
                            "content_type": content_type,
                            "file_size": row.get("file_size", 0),
                            "file_path": file_path,
                            "base64_data": base64_data,
                            "uploaded_at": uploaded_at.isoformat() if hasattr(uploaded_at, "isoformat") else str(uploaded_at),
                        })
            except Exception as e:
                logger.debug(f"message_attachments table not available or query failed: {e}")

            if attachments:
                logger.info(f"Found {len(attachments)} attachment(s) from conversation_message_attachments")
                return attachments

            metadata_json = None
            for msg in reversed(messages):
                if msg.get("message_type") == "user" or msg.get("role") == "user":
                    metadata_json = msg.get("metadata_json", {})
                    if isinstance(metadata_json, str):
                        metadata_json = json.loads(metadata_json)
                    break

            if not metadata_json:
                return []

            attachments = metadata_json.get("attachments", [])
            if not attachments:
                return []

            result = []
            for att in attachments:
                storage_path = att.get("storage_path") or att.get("file_path")
                content_type = att.get("content_type", "")
                is_image = content_type.startswith("image/") if content_type else False
                base64_data = att.get("base64_data") or att.get("data")
                if is_image and storage_path and not base64_data:
                    try:
                        if Path(storage_path).exists():
                            with open(storage_path, "rb") as f:
                                base64_data = base64.b64encode(f.read()).decode("utf-8")
                    except Exception as e:
                        logger.warning(f"Failed to read image at {storage_path}: {e}")
                result.append({
                    "attachment_id": att.get("attachment_id", ""),
                    "filename": att.get("filename", ""),
                    "content_type": content_type or "application/octet-stream",
                    "file_size": att.get("file_size") or att.get("size_bytes", 0),
                    "file_path": storage_path,
                    "base64_data": base64_data,
                    "uploaded_at": att.get("uploaded_at", ""),
                })
            logger.info(f"Found {len(result)} attachment(s) in latest user message metadata")
            return result

        except Exception as e:
            logger.warning(f"Failed to extract attachments from latest message: {e}")
            return []
    
    async def _add_user_persona(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        user_id: str,
        request_context: Optional[Dict[str, Any]] = None,
        profile_chat_model_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add user persona and preferences (from default persona in DB)."""
        try:
            from services.settings_service import settings_service
            timezone = await settings_service.get_user_timezone(user_id)
            grpc_request.metadata["user_timezone"] = timezone or "UTC"
            persona = await settings_service.get_default_persona(user_id)
            if persona:
                grpc_request.persona.ai_name = persona.get("ai_name") or "Alex"
                grpc_request.persona.persona_style = persona.get("name", "professional").replace(" ", "_").lower()[:50]
                grpc_request.persona.political_bias = persona.get("political_bias") or "neutral"
                grpc_request.persona.timezone = timezone or "UTC"
                style_instruction = persona.get("style_instruction")
                if style_instruction:
                    grpc_request.persona.custom_preferences["style_instruction"] = style_instruction
                logger.info(f"CONTEXT: Added persona (ai_name={persona.get('ai_name')})")
            
            preferred_name = await settings_service.get_user_preferred_name(user_id, fallback_to_display=True)
            birthday = await settings_service.get_user_birthday(user_id)
            ai_context = await settings_service.get_user_ai_context(user_id)
            if preferred_name:
                grpc_request.metadata["user_preferred_name"] = preferred_name
                logger.info(f"CONTEXT: Added preferred name: {preferred_name}")
            if birthday:
                grpc_request.metadata["user_birthday"] = birthday
                logger.info("CONTEXT: Added user birthday")
            if ai_context:
                grpc_request.metadata["user_ai_context"] = ai_context
                logger.info(f"CONTEXT: Added AI context ({len(ai_context)} chars)")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add user context: {e}")

        try:
            mem = await _build_user_memory_for_prompt(user_id, grpc_request)
            if mem.strip():
                grpc_request.metadata["user_memory"] = mem
                logger.info("CONTEXT: Added unified user memory (%s chars)", len(mem))
        except Exception as e:
            logger.warning("CONTEXT: Failed to add user memory: %s", e)
        
        # Add model preferences from settings service
        # Priority: request_context (e.g. stream body or bot selection) > conversation metadata (sidebar selection) > user default > org default
        try:
            from services.model_configuration_notifier import handle_user_model_retarget_flow
            from services.model_source_resolver import resolve_model_context
            from services.user_settings_kv_service import get_user_setting

            request_context = request_context or {}
            profile_chat_model_state = profile_chat_model_state or {}
            applied_profile = bool(profile_chat_model_state.get("applied"))

            conversation_chat_model = None
            conversation_id = grpc_request.conversation_id or ""
            if conversation_id:
                try:
                    from services.conversation_service import ConversationService

                    conv_svc = ConversationService()
                    conv_svc.set_current_user(user_id)
                    conv = await conv_svc.get_conversation(conversation_id, user_id)
                    conv_metadata = (conv or {}).get("metadata_json") or {}
                    conversation_chat_model = conv_metadata.get("user_chat_model") or None
                    if conversation_chat_model:
                        logger.debug(
                            "CONTEXT: Using chat model from conversation metadata: %s",
                            conversation_chat_model,
                        )
                except Exception as conv_e:
                    logger.debug("CONTEXT: Could not load conversation metadata for chat model: %s", conv_e)

            meta_chat = grpc_request.metadata.get("user_chat_model")
            user_kv_chat = await get_user_setting(user_id, "user_chat_model")
            org_chat = await settings_service.get_llm_model()

            if meta_chat:
                if applied_profile:
                    chat_source = "profile"
                else:
                    chat_source = "request"
                chat_model = meta_chat
            elif conversation_chat_model:
                chat_model = conversation_chat_model
                chat_source = "conversation"
            elif user_kv_chat:
                chat_model = user_kv_chat
                chat_source = "user_kv"
            else:
                chat_model = org_chat
                chat_source = "org"

            if chat_model and chat_source != "profile":
                chat_model = await handle_user_model_retarget_flow(
                    user_id,
                    chat_model,
                    role="chat",
                    source=chat_source,
                    conversation_id=conversation_id,
                )

            kv_fast = await get_user_setting(user_id, "user_fast_model")
            if kv_fast:
                fast_model = await handle_user_model_retarget_flow(
                    user_id,
                    kv_fast,
                    role="fast",
                    source="user_kv",
                    conversation_id=conversation_id,
                )
            else:
                fast_model = await handle_user_model_retarget_flow(
                    user_id,
                    await settings_service.get_classification_model(),
                    role="fast",
                    source="org",
                    conversation_id=conversation_id,
                )

            kv_image = await get_user_setting(user_id, "user_image_gen_model")
            if kv_image:
                image_model = await handle_user_model_retarget_flow(
                    user_id,
                    kv_image,
                    role="image",
                    source="user_kv",
                    conversation_id=conversation_id,
                )
            else:
                image_model = await handle_user_model_retarget_flow(
                    user_id,
                    await settings_service.get_image_generation_model(),
                    role="image",
                    source="org",
                    conversation_id=conversation_id,
                )

            if chat_model:
                grpc_request.metadata["user_chat_model"] = chat_model
                logger.info("SENDING TO ORCHESTRATOR: user_chat_model = %s", chat_model)
            if fast_model:
                grpc_request.metadata["user_fast_model"] = fast_model
            if image_model:
                grpc_request.metadata["user_image_model"] = image_model

            logger.info(
                "CONTEXT: Added model preferences (chat=%s, fast=%s, image=%s)",
                chat_model,
                fast_model,
                image_model,
            )

            if chat_model:
                llm_ctx = await resolve_model_context(user_id, chat_model)
                if llm_ctx:
                    grpc_request.metadata["user_chat_model"] = llm_ctx.get("real_model_id", chat_model)
                    grpc_request.metadata["user_llm_api_key"] = llm_ctx["api_key"]
                    grpc_request.metadata["user_llm_base_url"] = llm_ctx["base_url"]
                    if llm_ctx.get("provider_type"):
                        grpc_request.metadata["user_llm_provider_type"] = llm_ctx["provider_type"]
                    logger.info(
                        "CONTEXT: Injected credentials for chat model=%s",
                        llm_ctx.get("real_model_id", chat_model),
                    )
            if fast_model:
                fast_ctx = await resolve_model_context(user_id, fast_model)
                if fast_ctx:
                    grpc_request.metadata["user_fast_model"] = fast_ctx.get("real_model_id", fast_model)
                    grpc_request.metadata["user_fast_llm_api_key"] = fast_ctx["api_key"]
                    grpc_request.metadata["user_fast_llm_base_url"] = fast_ctx["base_url"]
                    if fast_ctx.get("provider_type"):
                        grpc_request.metadata["user_fast_llm_provider_type"] = fast_ctx["provider_type"]
            if image_model:
                image_ctx = await resolve_model_context(user_id, image_model)
                if image_ctx:
                    grpc_request.metadata["user_image_model"] = image_ctx.get("real_model_id", image_model)
                    grpc_request.metadata["user_image_llm_api_key"] = image_ctx["api_key"]
                    grpc_request.metadata["user_image_llm_base_url"] = image_ctx["base_url"]
                    if image_ctx.get("provider_type"):
                        grpc_request.metadata["user_image_llm_provider_type"] = image_ctx["provider_type"]
        except Exception as e:
            logger.warning("CONTEXT: Failed to add model preferences: %s", e)

        try:
            user_timezone = await settings_service.get_user_timezone(user_id)
            if user_timezone:
                grpc_request.metadata["user_timezone"] = user_timezone
                logger.info(f"CONTEXT: Added user timezone to metadata: {user_timezone}")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add user timezone: {e}")

        try:
            user_zip = await settings_service.get_user_zip_code(user_id)
            if user_zip:
                grpc_request.metadata["user_weather_location"] = user_zip
                logger.info(f"CONTEXT: Added user_weather_location to metadata: {user_zip}")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add user_weather_location: {e}")

    async def _add_editor_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add active editor context (for fiction editing, proofreading, etc.)"""
        try:
            active_editor = request_context.get("active_editor")
            editor_preference = request_context.get("editor_preference", "prefer")
            
            logger.info(f"🔍 EDITOR CONTEXT CHECK: request_context keys={list(request_context.keys())}, active_editor={active_editor is not None}, editor_preference={editor_preference}")
            
            # CRITICAL: Always send editor_preference in metadata, even when active_editor is not sent
            # This ensures the orchestrator can block editor-gated agents when preference is 'ignore'
            if editor_preference:
                grpc_request.metadata["editor_preference"] = editor_preference
                logger.info(f"📝 EDITOR PREFERENCE: Added to metadata = '{editor_preference}'")
            
            # Skip if user said to ignore editor (don't send active_editor, but editor_preference is already in metadata)
            if editor_preference == "ignore":
                logger.info(f"⚠️ EDITOR CONTEXT: Skipping active_editor - editor_preference is 'ignore' (but editor_preference sent in metadata)")
                return
            
            # Skip if no editor context
            if not active_editor:
                logger.info(f"⚠️ EDITOR CONTEXT: Skipping - no active_editor in request_context")
                return
            
            # Validate editor context
            if not isinstance(active_editor, dict):
                logger.warning(f"⚠️ EDITOR CONTEXT: Skipping - active_editor is not a dict (type={type(active_editor)})")
                return
            
            # CRITICAL: Reject if is_editable is False or missing - this ensures stale editor data is cleared
            # EXCEPTION: Reference documents (type: reference) should ALWAYS be sent even if not editable
            # because reference_agent needs to READ them (journals, logs, etc.)
            is_editable = active_editor.get("is_editable")
            frontmatter_type = active_editor.get("frontmatter", {}).get("type", "").strip().lower()
            
            if (not is_editable or is_editable is False) and frontmatter_type != "reference":
                logger.info(f"⚠️ EDITOR CONTEXT: Skipping - active_editor.is_editable is False or missing (editor tab likely closed)")
                return
            elif frontmatter_type == "reference" and not is_editable:
                logger.info(f"📚 EDITOR CONTEXT: Including reference document even though not editable (reference_agent needs to read it)")
            
            filename = active_editor.get("filename", "")
            if not (filename.endswith(".md") or filename.endswith(".org")):
                logger.warning(f"⚠️ EDITOR CONTEXT: Skipping - filename '{filename}' does not end with .md or .org")
                return
            
            # Parse frontmatter
            frontmatter_data = active_editor.get("frontmatter", {})
            frontmatter = orchestrator_pb2.EditorFrontmatter(
                type=frontmatter_data.get("type", ""),
                title=frontmatter_data.get("title", ""),
                author=frontmatter_data.get("author", ""),
                status=frontmatter_data.get("status", "")
            )
            
            # Add tags
            tags = frontmatter_data.get("tags", [])
            if tags:
                frontmatter.tags.extend(tags)
            
            # Add custom fields
            custom_fields_added = []
            for key, value in frontmatter_data.items():
                if key not in ["type", "title", "author", "tags", "status"]:
                    frontmatter.custom_fields[key] = str(value)
                    custom_fields_added.append(key)
                    if key in ["files", "components", "protocols", "schematics", "specifications"]:
                        logger.info(f"🔍 ADDING CUSTOM FIELD: {key} = {str(value)[:200]} (original type: {type(value).__name__})")
            if custom_fields_added:
                logger.info(f"✅ CONTEXT: Added {len(custom_fields_added)} custom frontmatter field(s): {custom_fields_added}")
            else:
                logger.info(f"⚠️ CONTEXT: No custom frontmatter fields found (frontmatter keys: {list(frontmatter_data.keys())})")
            
            # Build editor message
            canonical_path = active_editor.get("canonical_path") or active_editor.get("canonicalPath") or ""
            if canonical_path:
                logger.info(f"✅ CONTEXT: Active editor canonical_path: {canonical_path}")
            else:
                logger.warning(f"⚠️ CONTEXT: Active editor has no canonical_path - relative references may fail!")
            
            # Extract cursor and selection state
            cursor_offset = active_editor.get("cursor_offset", -1)
            selection_start = active_editor.get("selection_start", -1)
            selection_end = active_editor.get("selection_end", -1)
            
            # Extract document metadata
            document_id = active_editor.get("document_id") or ""
            folder_id = active_editor.get("folder_id") or ""
            file_path = active_editor.get("file_path") or ""
            
            # Log cursor state for debugging
            if cursor_offset >= 0:
                logger.info(f"✅ CONTEXT: Cursor detected at offset {cursor_offset}")
            if selection_start >= 0 and selection_end > selection_start:
                logger.info(f"✅ CONTEXT: Selection detected from {selection_start} to {selection_end}")
            
            grpc_request.active_editor.CopyFrom(
                orchestrator_pb2.ActiveEditor(
                    is_editable=is_editable if is_editable is not None else True,
                    filename=filename,
                    language=active_editor.get("language", "markdown"),
                    content=active_editor.get("content", ""),
                    content_length=len(active_editor.get("content", "")),
                    frontmatter=frontmatter,
                    editor_preference=editor_preference,
                    canonical_path=canonical_path,
                    cursor_offset=cursor_offset,
                    selection_start=selection_start,
                    selection_end=selection_end,
                    document_id=document_id,
                    folder_id=folder_id,
                    file_path=file_path
                )
            )
            
            logger.info(f"✅ CONTEXT: Added editor context (file={filename}, type={frontmatter.type}, {len(active_editor.get('content', ''))} chars)")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add editor context: {e}")

    def _add_data_workspace_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add active data workspace context when user has a data workspace table open."""
        try:
            active_data_workspace = request_context.get("active_data_workspace")
            if not active_data_workspace or not isinstance(active_data_workspace, dict):
                return
            table_id = active_data_workspace.get("table_id")
            if not table_id:
                return
            schema = active_data_workspace.get("schema")
            if not schema or not isinstance(schema, list) or len(schema) == 0:
                return
            columns = []
            for col in schema:
                if isinstance(col, dict) and col.get("name"):
                    columns.append(
                        orchestrator_pb2.DataWorkspaceColumn(
                            name=str(col.get("name", "")),
                            type=str(col.get("type", "TEXT")),
                            description=str(col.get("description", ""))
                        )
                    )
            visible_rows = active_data_workspace.get("visible_rows", [])
            visible_rows_json = ""
            if visible_rows:
                try:
                    visible_rows_json = json.dumps(visible_rows)
                except (TypeError, ValueError):
                    visible_rows_json = "[]"
            grpc_request.active_data_workspace.CopyFrom(
                orchestrator_pb2.ActiveDataWorkspace(
                    workspace_id=active_data_workspace.get("workspace_id", ""),
                    workspace_name=active_data_workspace.get("workspace_name", ""),
                    database_id=active_data_workspace.get("database_id", ""),
                    database_name=active_data_workspace.get("database_name", ""),
                    table_id=table_id,
                    table_name=active_data_workspace.get("table_name", ""),
                    row_count=int(active_data_workspace.get("row_count", 0)),
                    columns=columns,
                    visible_rows_json=visible_rows_json,
                    visible_row_count=int(active_data_workspace.get("visible_row_count", 0))
                )
            )
            logger.info(f"CONTEXT: Added data workspace context (table={active_data_workspace.get('table_name', '')}, {len(columns)} columns, {len(visible_rows)} visible rows)")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add data workspace context: {e}")
    
    async def _add_pipeline_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add pipeline execution context"""
        try:
            pipeline_preference = request_context.get("pipeline_preference")
            active_pipeline_id = request_context.get("active_pipeline_id")
            
            # Skip if user said to ignore pipelines
            if pipeline_preference == "ignore":
                return
            
            # Skip if no pipeline context
            if not active_pipeline_id:
                return
            
            # Build pipeline message
            grpc_request.pipeline_context.CopyFrom(
                orchestrator_pb2.PipelineContext(
                    pipeline_preference=pipeline_preference or "prefer",
                    active_pipeline_id=active_pipeline_id,
                    pipeline_name=""  # Could fetch from DB if needed
                )
            )
            
            logger.info(f"✅ CONTEXT: Added pipeline context (id={active_pipeline_id})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add pipeline context: {e}")
    
    async def _add_permission_grants(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]]
    ) -> None:
        """Add HITL permission grants"""
        try:
            if not state:
                return
            
            shared_memory = state.get("shared_memory", {})
            
            # Check if any permissions exist
            has_permissions = any([
                shared_memory.get("web_search_permission"),
                shared_memory.get("web_crawl_permission"),
                shared_memory.get("file_write_permission"),
                shared_memory.get("external_api_permission")
            ])
            
            if not has_permissions:
                return
            
            # Build permission grants
            grpc_request.permission_grants.CopyFrom(
                orchestrator_pb2.PermissionGrants(
                    web_search_permission=shared_memory.get("web_search_permission", False),
                    web_crawl_permission=shared_memory.get("web_crawl_permission", False),
                    file_write_permission=shared_memory.get("file_write_permission", False),
                    external_api_permission=shared_memory.get("external_api_permission", False)
                )
            )
            
            granted = [k.replace("_permission", "") for k, v in shared_memory.items() if k.endswith("_permission") and v]
            logger.info(f"✅ CONTEXT: Added permission grants ({', '.join(granted)})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add permission grants: {e}")
    
    async def _add_pending_operations(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]]
    ) -> None:
        """Add pending operations awaiting user approval"""
        try:
            if not state:
                return
            
            pending_ops = state.get("pending_operations", [])
            
            if not pending_ops:
                return
            
            for op in pending_ops:
                if not isinstance(op, dict):
                    continue
                
                grpc_request.pending_operations.append(
                    orchestrator_pb2.PendingOperationInfo(
                        id=op.get("id", ""),
                        type=op.get("type", ""),
                        summary=op.get("summary", ""),
                        permission_required=op.get("permission_required", False),
                        status=op.get("status", "pending"),
                        created_at=op.get("created_at", datetime.now().isoformat())
                    )
                )
            
            logger.info(f"✅ CONTEXT: Added {len(grpc_request.pending_operations)} pending operations")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add pending operations: {e}")
    
    async def _add_routing_locks(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any],
        state: Optional[Dict[str, Any]]
    ) -> None:
        """Add routing locks for dedicated agent sessions"""
        try:
            # Check request context first
            locked_agent = request_context.get("locked_agent")
            
            # Check shared memory as fallback
            if not locked_agent and state:
                shared_memory = state.get("shared_memory", {})
                locked_agent = shared_memory.get("locked_agent")
            
            if locked_agent:
                grpc_request.locked_agent = locked_agent
                logger.info(f"✅ CONTEXT: Added routing lock (agent={locked_agent})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add routing lock: {e}")
    
    async def _add_checkpoint_info(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add checkpoint info for conversation branching"""
        try:
            base_checkpoint_id = request_context.get("base_checkpoint_id")
            
            if base_checkpoint_id:
                grpc_request.base_checkpoint_id = base_checkpoint_id
                logger.info(f"✅ CONTEXT: Added checkpoint branching (checkpoint={base_checkpoint_id})")
            
        except Exception as e:
            logger.warning(f"⚠️ CONTEXT: Failed to add checkpoint info: {e}")

    def _add_image_description_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add image description context (image_base64, image_analysis_model) for image_description agent."""
        try:
            image_base64 = request_context.get("image_base64")
            image_analysis_model = request_context.get("image_analysis_model")
            if image_base64:
                grpc_request.metadata["image_base64"] = image_base64
                logger.info("CONTEXT: Added image_base64 for image description")
            if image_analysis_model:
                grpc_request.metadata["image_analysis_model"] = image_analysis_model
                logger.info(f"CONTEXT: Added image_analysis_model = {image_analysis_model}")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add image description context: {e}")

    def _add_document_description_context(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        request_context: Dict[str, Any]
    ) -> None:
        """Add document description context (document_content, document_analysis_model) for document_description agent."""
        try:
            document_content = request_context.get("document_content")
            document_analysis_model = request_context.get("document_analysis_model") or request_context.get("image_analysis_model")
            if document_content is not None:
                grpc_request.metadata["document_content"] = document_content
                logger.info("CONTEXT: Added document_content for document description")
            if document_analysis_model:
                grpc_request.metadata["document_analysis_model"] = document_analysis_model
                logger.info(f"CONTEXT: Added document_analysis_model = {document_analysis_model}")
        except Exception as e:
            logger.warning(f"CONTEXT: Failed to add document description context: {e}")
    
    async def _add_primary_agent_selected(
        self,
        grpc_request: orchestrator_pb2.ChatRequest,
        state: Optional[Dict[str, Any]]
    ) -> None:
        """
        Load agent routing metadata from backend conversation database
        
        This provides agent continuity information (primary_agent_selected, last_agent)
        to the orchestrator for proper routing decisions. Backend database is the
        source of truth for conversation-level agent metadata.
        """
        try:
            # Load agent metadata from backend database
            from services.conversation_service import ConversationService
            conversation_service = ConversationService()
            
            agent_metadata = await conversation_service.get_agent_metadata(
                conversation_id=grpc_request.conversation_id,
                user_id=grpc_request.user_id
            )
            
            if agent_metadata:
                # Add to metadata for orchestrator
                primary_agent = agent_metadata.get("primary_agent_selected")
                last_agent = agent_metadata.get("last_agent")
                saved_profile_id = agent_metadata.get("agent_profile_id")
                sticky_line_id = agent_metadata.get("active_line_id")

                if primary_agent:
                    grpc_request.metadata["primary_agent_selected"] = primary_agent
                    logger.info(f"📋 CONTEXT: Loaded primary_agent_selected from backend DB: {primary_agent}")

                if last_agent:
                    grpc_request.metadata["last_agent"] = last_agent
                    logger.debug(f"📋 CONTEXT: Loaded last_agent from backend DB: {last_agent}")

                # Sticky agent line: same conversation continues routing to CEO when no new @mention
                if (
                    sticky_line_id
                    and grpc_request.metadata.get("line_dispatch_mode") != "true"
                    and "agent_profile_id" not in grpc_request.metadata
                ):
                    from services import agent_line_service

                    ceo = await agent_line_service.get_ceo_agent_for_heartbeat(sticky_line_id)
                    if ceo and ceo.get("agent_profile_id"):
                        grpc_request.metadata["line_dispatch_mode"] = "true"
                        grpc_request.metadata["ceo_profile_id"] = str(ceo["agent_profile_id"])
                        grpc_request.metadata["line_id"] = str(sticky_line_id)
                        grpc_request.metadata["team_id"] = str(sticky_line_id)
                        grpc_request.metadata["team_context_id"] = str(sticky_line_id)
                        try:
                            team_chat_context = await agent_line_service.get_line_chat_context(
                                sticky_line_id, grpc_request.user_id
                            )
                            grpc_request.metadata["team_chat_context"] = team_chat_context
                        except Exception as e:
                            logger.warning("Sticky line chat context failed: %s", e)
                            grpc_request.metadata["team_chat_context"] = "Team context unavailable."
                        await _inject_line_capability_metadata(
                            grpc_request,
                            grpc_request.user_id,
                            str(sticky_line_id),
                            str(ceo["agent_profile_id"]),
                        )
                        try:
                            from services.database_manager.database_helpers import fetch_one

                            profile_row = await fetch_one(
                                "SELECT model_preference FROM agent_profiles WHERE id = $1",
                                str(ceo["agent_profile_id"]),
                            )
                            if profile_row and profile_row.get("model_preference"):
                                await _apply_profile_model_preference(
                                    grpc_request,
                                    grpc_request.user_id,
                                    str(ceo["agent_profile_id"]),
                                    profile_row["model_preference"],
                                    None,
                                )
                        except Exception as e:
                            if "does not exist" not in str(e).lower() and "relation" not in str(e).lower():
                                logger.warning("Failed CEO model for sticky line: %s", e)
                        try:
                            from services.celery_tasks.team_heartbeat_context import (
                                TEAM_TOOL_IDS_RULE,
                                _build_heartbeat_context,
                            )

                            hb = await _build_heartbeat_context(
                                str(sticky_line_id),
                                grpc_request.user_id,
                                str(ceo["agent_profile_id"]),
                            )
                            grpc_request.metadata["line_dispatch_briefing"] = hb
                            grpc_request.metadata["line_dispatch_tool_rules"] = TEAM_TOOL_IDS_RULE
                        except Exception as e:
                            logger.warning("Failed sticky line briefing: %s", e)
                            grpc_request.metadata["line_dispatch_briefing"] = (
                                grpc_request.metadata.get("team_chat_context") or ""
                            )
                            grpc_request.metadata["line_dispatch_tool_rules"] = ""
                        logger.info(
                            "📋 CONTEXT: Sticky line dispatch line_id=%s ceo=%s",
                            sticky_line_id,
                            ceo["agent_profile_id"],
                        )

                # Sticky @handle: use saved agent_profile_id only when none was set and not in line dispatch mode
                elif saved_profile_id and "agent_profile_id" not in grpc_request.metadata:
                    if grpc_request.metadata.get("line_dispatch_mode") != "true":
                        grpc_request.metadata["agent_profile_id"] = saved_profile_id
                        logger.info("📋 CONTEXT: Loaded agent_profile_id from backend DB (sticky routing): %s", saved_profile_id)
            else:
                logger.debug(f"📋 CONTEXT: No agent metadata in backend DB (new conversation or first request)")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load agent metadata from backend: {e}")
    
    def _log_context_summary(self, grpc_request: orchestrator_pb2.ChatRequest) -> None:
        """Log summary of what context was included"""
        context_items = []
        
        if len(grpc_request.conversation_history) > 0:
            context_items.append(f"history({len(grpc_request.conversation_history)})")
        
        if grpc_request.HasField("persona"):
            context_items.append("persona")
        
        if grpc_request.HasField("active_editor"):
            context_items.append(f"editor({grpc_request.active_editor.filename})")
        
        if grpc_request.HasField("pipeline_context"):
            context_items.append("pipeline")
        
        if grpc_request.HasField("permission_grants"):
            context_items.append("permissions")
        
        if len(grpc_request.pending_operations) > 0:
            context_items.append(f"pending_ops({len(grpc_request.pending_operations)})")
        
        if grpc_request.locked_agent:
            context_items.append(f"locked({grpc_request.locked_agent})")
        
        if grpc_request.agent_type:
            context_items.append(f"route({grpc_request.agent_type})")
        
        logger.info(f"📦 CONTEXT SUMMARY: {', '.join(context_items) if context_items else 'minimal'}")


# Singleton instance
_context_gatherer: Optional[GRPCContextGatherer] = None


def get_context_gatherer() -> GRPCContextGatherer:
    """Get singleton context gatherer instance"""
    global _context_gatherer
    if _context_gatherer is None:
        _context_gatherer = GRPCContextGatherer()
    return _context_gatherer

