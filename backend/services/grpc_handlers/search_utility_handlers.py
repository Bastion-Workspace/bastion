"""gRPC handlers for Search Utility operations (entities, query expansion, cache, help docs)."""

import logging

import grpc
from protos import tool_service_pb2

logger = logging.getLogger(__name__)


class SearchUtilityHandlersMixin:
    """Mixin providing Search Utility gRPC handlers.

    Mixed into ToolServiceImplementation; accesses self._get_search_service(),
    self._get_embedding_manager(), self._get_document_repo(), etc. via standard Python MRO.
    Non-placeholder search logic lives in `tools_service.services.help_search` and
    `tools_service.services.search_utility_ops` (help docs, KG co-occurrence, query expansion,
    conversation cache normalization).
    """

    # ===== Entity Operations =====
    
    async def SearchEntities(
        self,
        request: tool_service_pb2.EntitySearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.EntitySearchResponse:
        """Search entities"""
        try:
            logger.info(f"SearchEntities: query={request.query}")
            
            # Placeholder implementation
            response = tool_service_pb2.EntitySearchResponse()
            return response
            
        except Exception as e:
            logger.error(f"SearchEntities error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Entity search failed: {str(e)}")
    
    async def GetEntity(
        self,
        request: tool_service_pb2.EntityRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.EntityResponse:
        """Get entity details"""
        try:
            logger.info(f"GetEntity: entity_id={request.entity_id}")
            
            # Placeholder implementation
            entity = tool_service_pb2.Entity(
                entity_id=request.entity_id,
                entity_type="unknown",
                name="Placeholder"
            )
            response = tool_service_pb2.EntityResponse(entity=entity)
            return response
            
        except Exception as e:
            logger.error(f"GetEntity error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Get entity failed: {str(e)}")

    async def FindCoOccurringEntities(
        self,
        request: tool_service_pb2.FindCoOccurringEntitiesRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.FindCoOccurringEntitiesResponse:
        """Find entities that co-occur with given entities"""
        try:
            logger.info(f"FindCoOccurringEntities: entities={list(request.entity_names)}")

            from tools_service.services.search_utility_ops import find_co_occurring_entities

            co_occurring = await find_co_occurring_entities(
                list(request.entity_names),
                request.min_co_occurrences or 2,
            )
            entities = [
                tool_service_pb2.EntityInfo(
                    name=e["name"],
                    type=e["type"],
                    co_occurrence_count=e["co_occurrence_count"],
                )
                for e in co_occurring
            ]
            return tool_service_pb2.FindCoOccurringEntitiesResponse(entities=entities)

        except Exception as e:
            logger.error(f"FindCoOccurringEntities failed: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Co-occurrence search failed: {str(e)}")
    

    # ===== Query Enhancement =====
    
    async def ExpandQuery(
        self,
        request: tool_service_pb2.QueryExpansionRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.QueryExpansionResponse:
        """Expand query with variations"""
        try:
            logger.info(f"ExpandQuery: query={request.query}")

            conversation_context = None
            if hasattr(request, "conversation_context") and request.conversation_context:
                conversation_context = request.conversation_context
                logger.info(
                    "ExpandQuery: Using conversation context (%d chars)",
                    len(conversation_context),
                )

            from tools_service.services.search_utility_ops import expand_query_for_rpc

            result = await expand_query_for_rpc(
                original_query=request.query,
                num_variations=request.num_variations or 3,
                conversation_context=conversation_context,
            )

            response = tool_service_pb2.QueryExpansionResponse(
                original_query=result.get("original_query", request.query),
                expansion_count=0,
            )
            response.expanded_queries.extend(result.get("expanded_queries") or [])
            response.key_entities.extend(result.get("key_entities") or [])
            response.expansion_count = len(response.expanded_queries)

            logger.info(f"ExpandQuery: Generated {response.expansion_count} variations")
            return response

        except Exception as e:
            logger.error(f"ExpandQuery error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Query expansion failed: {str(e)}")
    
    # ===== Conversation Cache =====
    
    async def SearchConversationCache(
        self,
        request: tool_service_pb2.CacheSearchRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.CacheSearchResponse:
        """Search conversation cache for previous research"""
        try:
            logger.info(f"SearchConversationCache: query={request.query}")

            from tools_service.services.search_utility_ops import search_conversation_cache_for_rpc

            normalized = await search_conversation_cache_for_rpc(
                query=request.query,
                conversation_id=request.conversation_id if request.conversation_id else None,
                freshness_hours=request.freshness_hours or 24,
            )
            response = tool_service_pb2.CacheSearchResponse(
                cache_hit=bool(normalized.get("cache_hit", False)),
            )
            for entry in normalized.get("entries") or []:
                response.entries.append(
                    tool_service_pb2.CacheEntry(
                        content=entry.get("content", ""),
                        timestamp=str(entry.get("timestamp", "")),
                        agent_name=str(entry.get("agent_name", "")),
                        relevance_score=float(entry.get("relevance_score", 0.0)),
                    )
                )

            logger.info(
                "SearchConversationCache: Cache hit=%s, %d entries",
                response.cache_hit,
                len(response.entries),
            )
            return response

        except Exception as e:
            logger.error(f"SearchConversationCache error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Cache search failed: {str(e)}")
    
    async def SearchHelpDocs(
        self,
        request: tool_service_pb2.SearchHelpDocsRequest,
        context: grpc.aio.ServicerContext
    ) -> tool_service_pb2.SearchHelpDocsResponse:
        """Search app help documentation in the help_docs vector collection."""
        try:
            query = (request.query or "").strip()
            limit = request.limit if request.limit > 0 else 5
            logger.info("SearchHelpDocs: query=%s, limit=%d", query[:80] if query else "", limit)
            if not query:
                return tool_service_pb2.SearchHelpDocsResponse(results=[], total_count=0)
            embedding_manager = await self._get_embedding_manager()
            if not embedding_manager:
                return tool_service_pb2.SearchHelpDocsResponse(results=[], total_count=0)

            from tools_service.services.help_search import search_help_docs

            rows = await search_help_docs(
                query=query,
                limit=limit,
                embedding_manager=embedding_manager,
            )
            out = [
                tool_service_pb2.HelpSearchResult(
                    topic_id=r.get("topic_id", "") or "",
                    title=r.get("title", "") or "",
                    content=r.get("content", "") or "",
                    score=float(r.get("score") or 0.0),
                )
                for r in rows
            ]
            return tool_service_pb2.SearchHelpDocsResponse(results=out, total_count=len(out))
        except Exception as e:
            logger.error("SearchHelpDocs error: %s", e)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Help docs search failed: {e}",
            )
    
