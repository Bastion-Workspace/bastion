"""gRPC handlers for Agent Profile operations."""

import json
import logging
import uuid

import grpc
from protos import tool_service_pb2
from services.grpc_handlers._utils import jsonb_list, jsonb_dict, json_default
from utils.grpc_rls import grpc_user_rls as _grpc_rls

logger = logging.getLogger(__name__)


class AgentProfileHandlersMixin:
    """Mixin providing Agent Profile gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for agent profiles,
    playbook reads, team posts, and auto-routable profile listing.
    """

    async def GetAgentProfile(
        self,
        request: tool_service_pb2.GetAgentProfileRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetAgentProfileResponse:
        """Return agent profile by ID for custom agent execution."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            profile_id = request.profile_id
            if not profile_id:
                return tool_service_pb2.GetAgentProfileResponse(
                    success=False, profile_json="", error="profile_id required"
                )
            row = await fetch_one(
                """SELECT * FROM agent_profiles
                   WHERE id = $1
                     AND (user_id = $2
                          OR EXISTS (
                              SELECT 1 FROM agent_artifact_shares _sh
                              WHERE _sh.artifact_type = 'agent_profile'
                                AND _sh.artifact_id = agent_profiles.id
                                AND _sh.shared_with_user_id = $2
                          ))""",
                uuid.UUID(profile_id),
                user_id,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.GetAgentProfileResponse(
                    success=False, profile_json="", error="Profile not found"
                )
            persona_mode = row.get("persona_mode") or "none"
            persona_id = str(row["persona_id"]) if row.get("persona_id") else None
            profile = {
                "id": str(row["id"]),
                "user_id": row["user_id"],
                "name": row["name"],
                "handle": row["handle"],
                "description": row.get("description"),
                "is_active": row.get("is_active", True),
                "model_preference": row.get("model_preference"),
                "max_research_rounds": row.get("max_research_rounds", 3),
                "system_prompt_additions": row.get("system_prompt_additions"),
                "knowledge_config": jsonb_dict(row.get("knowledge_config")),
                "default_playbook_id": str(row["default_playbook_id"]) if row.get("default_playbook_id") else None,
                "default_run_context": row.get("default_run_context") or "interactive",
                "default_approval_policy": row.get("default_approval_policy") or "require",
                "journal_config": jsonb_dict(row.get("journal_config")),
                "team_config": jsonb_dict(row.get("team_config")),
                "prompt_history_enabled": row.get("chat_history_enabled", False),
                "chat_history_lookback": row.get("chat_history_lookback", 10),
                "summary_threshold_tokens": row.get("summary_threshold_tokens", 5000),
                "summary_keep_messages": row.get("summary_keep_messages", 10),
                "persona_mode": persona_mode,
                "persona_id": persona_id,
                "include_user_context": row.get("include_user_context", False),
                "include_datetime_context": row.get("include_datetime_context", True),
                "include_user_facts": row.get("include_user_facts", False),
                "include_facts_categories": jsonb_list(row.get("include_facts_categories")),
                "use_themed_memory": row.get("use_themed_memory", True),
                "include_agent_memory": row.get("include_agent_memory", False),
                "auto_routable": row.get("auto_routable", False),
                "data_workspace_config": jsonb_dict(row.get("data_workspace_config")),
                "allowed_connections": jsonb_list(row.get("allowed_connections")),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
            if persona_mode == "specific":
                if persona_id:
                    logger.info(
                        "GetAgentProfile: resolving specific persona profile_id=%s persona_id=%s user_id=%s",
                        profile_id,
                        persona_id,
                        user_id,
                    )
                    from services.settings_service import settings_service
                    persona = await settings_service.get_persona_by_id(persona_id, user_id)
                    if persona:
                        profile["persona"] = persona
                        logger.info(
                            "GetAgentProfile: embedded persona name=%s ai_name=%s",
                            persona.get("name"),
                            persona.get("ai_name"),
                        )
                    else:
                        logger.warning(
                            "GetAgentProfile: persona_mode=specific but get_persona_by_id returned "
                            "nothing (profile_id=%s persona_id=%s user_id=%s)",
                            profile_id,
                            persona_id,
                            user_id,
                        )
                else:
                    logger.warning(
                        "GetAgentProfile: persona_mode=specific but persona_id is null "
                        "(profile_id=%s user_id=%s)",
                        profile_id,
                        user_id,
                    )
            return tool_service_pb2.GetAgentProfileResponse(
                success=True,
                profile_json=json.dumps(profile, default=json_default),
            )
        except Exception as e:
            logger.exception("GetAgentProfile failed")
            return tool_service_pb2.GetAgentProfileResponse(
                success=False, profile_json="", error=str(e)
            )

    async def ListAutoRoutableProfiles(
        self,
        request: tool_service_pb2.ListAutoRoutableProfilesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListAutoRoutableProfilesResponse:
        """Return agent profiles for the user where auto_routable=true and is_active=true."""
        try:
            from services.database_manager.database_helpers import fetch_all
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            rows = await fetch_all(
                "SELECT id, handle, name, description FROM agent_profiles "
                "WHERE user_id = $1 AND is_active = true AND auto_routable = true ORDER BY name",
                user_id,
                rls_context=ctx,
            )
            profiles = [
                {
                    "id": str(r["id"]),
                    "handle": r["handle"],
                    "name": r["name"],
                    "description": r.get("description") or "",
                }
                for r in rows
            ]
            return tool_service_pb2.ListAutoRoutableProfilesResponse(
                success=True,
                profiles_json=json.dumps(profiles),
            )
        except Exception as e:
            logger.exception("ListAutoRoutableProfiles failed")
            return tool_service_pb2.ListAutoRoutableProfilesResponse(
                success=False, profiles_json="[]", error=str(e)
            )

    async def ResolveAgentHandle(
        self,
        request: tool_service_pb2.ResolveAgentHandleRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ResolveAgentHandleResponse:
        """Resolve agent target string to agent_profile_id: handle, @handle, UUID, or display name (unique)."""
        try:
            from services.database_manager.database_helpers import fetch_all, fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            raw = (request.handle or "").strip()
            if raw.startswith("@"):
                raw = raw[1:].strip()
            if not raw:
                return tool_service_pb2.ResolveAgentHandleResponse(found=False)

            # 1) Exact handle match
            row = await fetch_one(
                "SELECT id, name FROM agent_profiles WHERE user_id = $1 AND handle = $2 AND is_active = true",
                user_id,
                raw,
                rls_context=ctx,
            )
            if row:
                return tool_service_pb2.ResolveAgentHandleResponse(
                    agent_profile_id=str(row["id"]),
                    agent_name=row.get("name") or raw,
                    found=True,
                )

            # 2) Agent profile UUID (workers sometimes paste ids from briefings)
            try:
                uid = uuid.UUID(raw)
                row = await fetch_one(
                    "SELECT id, name FROM agent_profiles WHERE user_id = $1 AND id = $2 AND is_active = true",
                    user_id,
                    uid,
                    rls_context=ctx,
                )
                if row:
                    return tool_service_pb2.ResolveAgentHandleResponse(
                        agent_profile_id=str(row["id"]),
                        agent_name=row.get("name") or raw,
                        found=True,
                    )
            except (ValueError, TypeError):
                pass

            # 3) Unique display name (case-insensitive) — matches "send to your manager (Name)" style prompts
            rows = await fetch_all(
                """
                SELECT id, name FROM agent_profiles
                WHERE user_id = $1 AND is_active = true
                  AND LOWER(TRIM(COALESCE(name, ''))) = LOWER(TRIM($2))
                """,
                user_id,
                raw,
                rls_context=ctx,
            )
            if len(rows) == 1:
                row = rows[0]
                return tool_service_pb2.ResolveAgentHandleResponse(
                    agent_profile_id=str(row["id"]),
                    agent_name=row.get("name") or raw,
                    found=True,
                )
            if len(rows) > 1:
                logger.warning(
                    "ResolveAgentHandle: ambiguous display name %r matches %s profiles for user_id=%s",
                    raw[:80],
                    len(rows),
                    user_id,
                )
            return tool_service_pb2.ResolveAgentHandleResponse(found=False)
        except Exception as e:
            logger.exception("ResolveAgentHandle failed")
            return tool_service_pb2.ResolveAgentHandleResponse(found=False)

    async def EnqueueAgentInvocation(
        self,
        request: tool_service_pb2.EnqueueAgentInvocationRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.EnqueueAgentInvocationResponse:
        """Enqueue async agent-to-agent invocation via Celery. Used by output_router."""
        try:
            from services.celery_tasks.agent_tasks import dispatch_agent_invocation
            agent_profile_id = request.agent_profile_id or ""
            input_content = request.input_content or ""
            user_id = request.user_id or "system"
            source_agent_name = request.source_agent_name or ""
            chain_depth = request.chain_depth or 0
            chain_path_json = request.chain_path_json or "[]"
            if not agent_profile_id or not input_content:
                return tool_service_pb2.EnqueueAgentInvocationResponse(
                    success=False,
                    error="agent_profile_id and input_content required",
                )
            task = dispatch_agent_invocation.apply_async(
                args=[
                    agent_profile_id,
                    input_content,
                    user_id,
                    source_agent_name,
                    chain_depth,
                    chain_path_json,
                ],
            )
            return tool_service_pb2.EnqueueAgentInvocationResponse(
                success=True,
                task_id=task.id or "",
            )
        except Exception as e:
            logger.exception("EnqueueAgentInvocation failed")
            return tool_service_pb2.EnqueueAgentInvocationResponse(
                success=False,
                error=str(e),
            )

    async def ReadTeamPosts(
        self,
        request: tool_service_pb2.ReadTeamPostsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ReadTeamPostsResponse:
        """Return team posts (optionally since last_read_at for the user). Used by read_team_posts tool."""
        try:
            from services.database_manager.database_helpers import fetch_one, fetch_all
            from services.team_post_service import TeamPostService
            from services.team_service import TeamService
            team_id = (request.team_id or "").strip()
            user_id = request.user_id or "system"
            since_last_read = request.since_last_read
            limit = max(1, min(100, request.limit or 20))
            mark_as_read = request.mark_as_read
            if not team_id:
                return tool_service_pb2.ReadTeamPostsResponse(
                    success=False,
                    error="team_id required",
                )
            team_service = TeamService()
            await team_service.initialize()
            team_post_service = TeamPostService()
            await team_post_service.initialize(team_service=team_service)
            since_ts = None
            if since_last_read:
                row = await fetch_one(
                    "SELECT last_read_at FROM team_members WHERE team_id = $1 AND user_id = $2",
                    uuid.UUID(team_id),
                    user_id,
                )
                if row and row.get("last_read_at"):
                    since_ts = row["last_read_at"]
            posts_result = await team_post_service.get_team_posts_since(
                team_id=team_id,
                user_id=user_id,
                since_ts=since_ts,
                limit=limit,
            )
            team_row = await fetch_one(
                "SELECT team_name FROM teams WHERE team_id = $1",
                uuid.UUID(team_id),
            )
            team_name = (team_row.get("team_name") or "") if team_row else ""
            if mark_as_read and posts_result:
                await team_service.mark_team_posts_as_read(team_id, user_id)
            out_posts = []
            for p in posts_result:
                out_posts.append(
                    tool_service_pb2.TeamPost(
                        post_id=p.get("post_id", ""),
                        author_id=p.get("author_id", ""),
                        author_name=p.get("author_name") or p.get("author_display_name") or p.get("author_username") or "",
                        content=p.get("content", ""),
                        post_type=p.get("post_type", "text"),
                        created_at=p.get("created_at").isoformat() if p.get("created_at") else "",
                    )
                )
            return tool_service_pb2.ReadTeamPostsResponse(
                posts=out_posts,
                count=len(out_posts),
                team_name=team_name,
                success=True,
            )
        except PermissionError as e:
            return tool_service_pb2.ReadTeamPostsResponse(
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("ReadTeamPosts failed")
            return tool_service_pb2.ReadTeamPostsResponse(
                success=False,
                error=str(e),
            )

    async def CreateTeamPost(
        self,
        request: tool_service_pb2.CreateTeamPostRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateTeamPostResponse:
        """Create a team post or comment. Used by post_to_team tool and output_router team_post destination."""
        try:
            from services.team_post_service import TeamPostService
            from services.team_service import TeamService
            team_id = (request.team_id or "").strip()
            user_id = request.user_id or "system"
            content = (request.content or "").strip()
            post_type = (request.post_type or "text").strip() or "text"
            reply_to_post_id = (request.reply_to_post_id or "").strip()
            if not team_id or not content:
                return tool_service_pb2.CreateTeamPostResponse(
                    success=False,
                    error="team_id and content required",
                )
            team_service = TeamService()
            await team_service.initialize()
            team_post_service = TeamPostService()
            await team_post_service.initialize(team_service=team_service)
            if reply_to_post_id:
                comment = await team_post_service.create_comment(
                    post_id=reply_to_post_id,
                    author_id=user_id,
                    content=content,
                )
                post_id = comment.get("comment_id", "")
            else:
                post = await team_post_service.create_post(
                    team_id=team_id,
                    author_id=user_id,
                    content=content,
                    post_type=post_type,
                    attachments=None,
                )
                post_id = post.get("post_id", "")
            return tool_service_pb2.CreateTeamPostResponse(
                post_id=post_id,
                success=True,
            )
        except PermissionError as e:
            return tool_service_pb2.CreateTeamPostResponse(
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.exception("CreateTeamPost failed")
            return tool_service_pb2.CreateTeamPostResponse(
                success=False,
                error=str(e),
            )

    async def GetPlaybook(
        self,
        request: tool_service_pb2.GetPlaybookRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetPlaybookResponse:
        """Return playbook by ID for custom agent execution."""
        try:
            from services.database_manager.database_helpers import fetch_one
            user_id = request.user_id or "system"
            ctx = _grpc_rls(user_id)
            playbook_id = (request.playbook_id or "").strip()
            if not playbook_id:
                return tool_service_pb2.GetPlaybookResponse(
                    success=False, playbook_json="", error="playbook_id required"
                )
            try:
                uuid.UUID(playbook_id)
            except (ValueError, TypeError):
                slug_normalized = playbook_id.replace("_", "-").lower()
                resolved = await fetch_one(
                    """SELECT id FROM custom_playbooks
                       WHERE user_id = $1 AND (
                         name = $2
                         OR LOWER(REGEXP_REPLACE(TRIM(name), '\\s+', '-')) = $3
                       )
                       LIMIT 1""",
                    user_id,
                    playbook_id,
                    slug_normalized,
                    rls_context=ctx,
                )
                if not resolved:
                    return tool_service_pb2.GetPlaybookResponse(
                        success=False, playbook_json="",
                        error="Playbook not found (use id from list_playbooks, or exact name)",
                    )
                playbook_id = str(resolved["id"])
            row = await fetch_one(
                """SELECT * FROM custom_playbooks
                   WHERE id = $1
                     AND (user_id = $2
                          OR (user_id IS NULL AND is_builtin = true)
                          OR EXISTS (
                              SELECT 1 FROM agent_artifact_shares _sh
                              WHERE _sh.artifact_type = 'playbook'
                                AND _sh.artifact_id = custom_playbooks.id
                                AND _sh.shared_with_user_id = $2
                          ))""",
                uuid.UUID(playbook_id),
                user_id,
                rls_context=ctx,
            )
            if not row:
                return tool_service_pb2.GetPlaybookResponse(
                    success=False, playbook_json="", error="Playbook not found"
                )
            raw_def = row.get("definition") or {}
            if isinstance(raw_def, str):
                try:
                    raw_def = json.loads(raw_def) if raw_def else {}
                except (json.JSONDecodeError, TypeError):
                    raw_def = {}
            if not isinstance(raw_def, dict):
                raw_def = {}
            raw_triggers = row.get("triggers") or []
            if isinstance(raw_triggers, str):
                try:
                    raw_triggers = json.loads(raw_triggers) if raw_triggers else []
                except (json.JSONDecodeError, TypeError):
                    raw_triggers = []
            if not isinstance(raw_triggers, list):
                raw_triggers = []
            playbook = {
                "id": str(row["id"]),
                "user_id": row["user_id"],
                "name": row["name"],
                "description": row.get("description"),
                "version": row.get("version", "1.0"),
                "definition": raw_def,
                "triggers": raw_triggers,
                "is_template": row.get("is_template", False),
                "is_locked": row.get("is_locked", False),
                "category": row.get("category"),
                "tags": jsonb_list(row.get("tags")),
                "required_connectors": list(row.get("required_connectors") or []),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
                "updated_at": row.get("updated_at").isoformat() if row.get("updated_at") else None,
            }
            return tool_service_pb2.GetPlaybookResponse(
                success=True,
                playbook_json=json.dumps(playbook),
            )
        except Exception as e:
            logger.exception("GetPlaybook failed")
            return tool_service_pb2.GetPlaybookResponse(
                success=False, playbook_json="", error=str(e)
            )

