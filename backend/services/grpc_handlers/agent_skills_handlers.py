"""gRPC handlers for Agent Skills CRUD operations."""

import json
import logging

import grpc
from protos import tool_service_pb2
from services.grpc_handlers._utils import grpc_metadata_json_string_list

logger = logging.getLogger(__name__)


class AgentSkillsHandlersMixin:
    """Mixin providing Agent Skills gRPC handlers.

    Mixed into ToolServiceImplementation; provides handlers for skills CRUD
    (search, list, create, update, get by ID/slug).
    """

    async def GetSkillsByIds(
        self,
        request: tool_service_pb2.GetSkillsByIdsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetSkillsByIdsResponse:
        """Batch fetch skills by IDs for pipeline skill injection."""
        try:
            from tools_service.services import agent_skills_ops

            skill_ids = list(request.skill_ids) if request.skill_ids else []
            out = await agent_skills_ops.op_get_skills_by_ids(skill_ids)
            return tool_service_pb2.GetSkillsByIdsResponse(
                success=True,
                skills_json=out["skills_json"],
            )
        except Exception as e:
            logger.exception("GetSkillsByIds failed")
            return tool_service_pb2.GetSkillsByIdsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def SearchSkills(
        self,
        request: tool_service_pb2.SearchSkillsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.SearchSkillsResponse:
        """Semantic search over skills for auto-discovery at step invocation."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            query = (request.query or "").strip()
            limit = request.limit or 3
            score_threshold = request.score_threshold if request.score_threshold > 0 else 0.5
            active_types = await grpc_metadata_json_string_list(context, "active-connection-types")
            out = await agent_skills_ops.op_search_skills(
                user_id=user_id,
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                active_connection_types=active_types,
            )
            return tool_service_pb2.SearchSkillsResponse(
                success=True,
                skills_json=out["skills_json"],
            )
        except Exception as e:
            logger.exception("SearchSkills failed")
            return tool_service_pb2.SearchSkillsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def ListSkills(
        self,
        request: tool_service_pb2.ListSkillsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListSkillsResponse:
        """List user and optionally built-in skills for agent self-awareness."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            category = request.category or None
            include_builtin = getattr(request, "include_builtin", True)
            out = await agent_skills_ops.op_list_skills(
                user_id=user_id, category=category, include_builtin=include_builtin
            )
            return tool_service_pb2.ListSkillsResponse(
                success=True,
                skills_json=out["skills_json"],
            )
        except Exception as e:
            logger.exception("ListSkills failed")
            return tool_service_pb2.ListSkillsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def ListSkillSummaries(
        self,
        request: tool_service_pb2.ListSkillSummariesRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.ListSkillSummariesResponse:
        """Return lightweight skill summaries for manifest/catalog injection."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            include_builtin = getattr(request, "include_builtin", True)
            out = await agent_skills_ops.op_list_skill_summaries(
                user_id=user_id, include_builtin=include_builtin
            )
            return tool_service_pb2.ListSkillSummariesResponse(
                success=True,
                summaries_json=out["summaries_json"],
            )
        except Exception as e:
            logger.exception("ListSkillSummaries failed")
            return tool_service_pb2.ListSkillSummariesResponse(
                success=False, summaries_json="[]", error=str(e)
            )

    async def GetSkillBySlug(
        self,
        request: tool_service_pb2.GetSkillBySlugRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetSkillBySlugResponse:
        """Fetch a single skill by slug for direct acquisition."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            slug = (request.slug or "").strip()
            out = await agent_skills_ops.op_get_skill_by_slug(user_id=user_id, slug=slug)
            return tool_service_pb2.GetSkillBySlugResponse(
                success=out["success"],
                skill_json=out["skill_json"],
                error=out.get("error") or "",
            )
        except Exception as e:
            logger.exception("GetSkillBySlug failed")
            return tool_service_pb2.GetSkillBySlugResponse(
                success=False, skill_json="", error=str(e)
            )

    async def GetCandidateForSlug(
        self,
        request: tool_service_pb2.GetCandidateForSlugRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetCandidateForSlugResponse:
        """Return the candidate version of a skill if one exists."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            slug = (request.slug or "").strip()
            out = await agent_skills_ops.op_get_candidate_for_slug(user_id=user_id, slug=slug)
            return tool_service_pb2.GetCandidateForSlugResponse(
                success=out["success"],
                has_candidate=out["has_candidate"],
                skill_json=out["skill_json"],
                error=out.get("error") or "",
            )
        except Exception as e:
            logger.exception("GetCandidateForSlug failed")
            return tool_service_pb2.GetCandidateForSlugResponse(
                success=False, has_candidate=False, skill_json="", error=str(e)
            )

    async def GetSkillsBySlugs(
        self,
        request: tool_service_pb2.GetSkillsBySlugsRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.GetSkillsBySlugsResponse:
        """Batch fetch skills by slugs for dependency resolution."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            slugs = list(request.slugs) if request.slugs else []
            out = await agent_skills_ops.op_get_skills_by_slugs(user_id=user_id, slugs=slugs)
            return tool_service_pb2.GetSkillsBySlugsResponse(
                success=True,
                skills_json=out["skills_json"],
            )
        except Exception as e:
            logger.exception("GetSkillsBySlugs failed")
            return tool_service_pb2.GetSkillsBySlugsResponse(
                success=False, skills_json="[]", error=str(e)
            )

    async def CreateSkill(
        self,
        request: tool_service_pb2.CreateSkillRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.CreateSkillResponse:
        """Create a user-authored skill."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            name = (request.name or "").strip() or "Unnamed skill"
            slug = (request.slug or "").strip().lower().replace(" ", "-")[:100] or "unnamed-skill"
            procedure = request.procedure or ""
            required_tools = list(request.required_tools) if request.required_tools else []
            optional_tools = list(request.optional_tools) if request.optional_tools else []
            description = (request.description or "").strip() or None
            category = (request.category or "").strip() or None
            tags = list(request.tags) if request.tags else []
            out = await agent_skills_ops.op_create_skill(
                user_id=user_id,
                name=name,
                slug=slug,
                procedure=procedure,
                required_tools=required_tools,
                optional_tools=optional_tools,
                description=description,
                category=category,
                tags=tags,
            )
            return tool_service_pb2.CreateSkillResponse(
                success=True,
                skill_id=out["skill_id"],
                skill_json=out["skill_json"],
            )
        except ValueError as e:
            return tool_service_pb2.CreateSkillResponse(
                success=False, skill_id="", skill_json="", error=str(e)
            )
        except Exception as e:
            logger.exception("CreateSkill failed")
            return tool_service_pb2.CreateSkillResponse(
                success=False, skill_id="", skill_json="", error=str(e)
            )

    async def UpdateSkill(
        self,
        request: tool_service_pb2.UpdateSkillRequest,
        context: grpc.aio.ServicerContext,
    ) -> tool_service_pb2.UpdateSkillResponse:
        """Update a user skill (creates new version). Used for propose_skill_update flow."""
        try:
            from tools_service.services import agent_skills_ops

            user_id = request.user_id or "system"
            skill_id = request.skill_id or ""
            procedure = (request.procedure or "").strip() or None
            improvement_rationale = (request.improvement_rationale or "").strip() or None
            evidence_metadata = None
            if request.evidence_metadata_json:
                try:
                    evidence_metadata = json.loads(request.evidence_metadata_json)
                except json.JSONDecodeError:
                    pass
            name = (request.name or "").strip() or None
            description = (request.description or "").strip() or None
            category = (request.category or "").strip() or None
            required_tools = list(request.required_tools) if request.required_tools else None
            optional_tools = list(request.optional_tools) if request.optional_tools else None
            out = await agent_skills_ops.op_update_skill(
                skill_id=skill_id,
                user_id=user_id,
                procedure=procedure,
                improvement_rationale=improvement_rationale,
                evidence_metadata=evidence_metadata,
                name=name,
                description=description,
                category=category,
                required_tools=required_tools,
                optional_tools=optional_tools,
            )
            return tool_service_pb2.UpdateSkillResponse(
                success=True,
                skill_id=out["skill_id"],
                version=out["version"],
                skill_json=out["skill_json"],
            )
        except ValueError as e:
            return tool_service_pb2.UpdateSkillResponse(
                success=False, skill_id="", version=0, skill_json="", error=str(e)
            )
        except Exception as e:
            logger.exception("UpdateSkill failed")
            return tool_service_pb2.UpdateSkillResponse(
                success=False, skill_id="", version=0, skill_json="", error=str(e)
            )
