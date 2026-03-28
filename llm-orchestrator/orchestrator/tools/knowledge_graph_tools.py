"""
Knowledge Graph Tools - Neo4j entity cross-reference for Agent Factory

Tools to find documents by entity names, traverse entity relationships,
and discover co-occurring entities. All operations use RLS-filtered Neo4j
via the backend gRPC Tool Service.
"""

import logging
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from orchestrator.backend_tool_client import get_backend_tool_client
from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for find_documents_by_entities_tool ─────────────────────────

class FindDocumentsByEntitiesInputs(BaseModel):
    """Required inputs for find_documents_by_entities."""
    entity_names: List[str] = Field(description="List of entity names to search for (e.g. person, org, location)")


class FindDocumentsByEntitiesOutputs(BaseModel):
    """Typed outputs for find_documents_by_entities."""
    document_ids: List[str] = Field(description="Document IDs that mention the given entities (RLS filtered)")
    count: int = Field(description="Number of documents found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── I/O models for find_related_documents_by_entities_tool ─────────────────

class FindRelatedDocumentsByEntitiesInputs(BaseModel):
    """Required inputs for find_related_documents_by_entities."""
    entity_names: List[str] = Field(description="Starting entity names for graph traversal")


class FindRelatedDocumentsByEntitiesParams(BaseModel):
    """Optional parameters."""
    max_hops: int = Field(default=1, description="Maximum graph traversal depth (1 or 2)")


class FindRelatedDocumentsByEntitiesOutputs(BaseModel):
    """Typed outputs for find_related_documents_by_entities."""
    document_ids: List[str] = Field(description="Document IDs reachable via entity relationships (RLS filtered)")
    count: int = Field(description="Number of related documents found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── I/O models for find_co_occurring_entities_tool ─────────────────────────

class FindCoOccurringEntitiesInputs(BaseModel):
    """Required inputs for find_co_occurring_entities."""
    entity_names: List[str] = Field(description="Target entity names to find co-occurring entities for")


class FindCoOccurringEntitiesParams(BaseModel):
    """Optional parameters."""
    min_co_occurrences: int = Field(default=2, description="Minimum co-occurrence threshold")


class CoOccurringEntityRef(BaseModel):
    """Reference to a co-occurring entity."""
    name: str
    type: str
    co_occurrence_count: int


class FindCoOccurringEntitiesOutputs(BaseModel):
    """Typed outputs for find_co_occurring_entities."""
    entities: List[Dict[str, Any]] = Field(description="Co-occurring entities with name, type, co_occurrence_count")
    count: int = Field(description="Number of co-occurring entities found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── I/O models for search_entities_tool ────────────────────────────────────

class SearchEntitiesInputs(BaseModel):
    """Required inputs for search_entities."""
    query: str = Field(
        description="Name or keyword to search entities for (person, org, location, etc.)"
    )


class SearchEntitiesParams(BaseModel):
    """Optional parameters for search_entities."""
    entity_types: List[str] = Field(
        default_factory=list,
        description="Filter by type: PERSON, ORG, LOCATION, PRODUCT, EVENT, etc.",
    )
    limit: int = Field(default=10, description="Max entities to return")


class SearchEntitiesOutputs(BaseModel):
    """Outputs for search_entities."""
    entities: List[Dict[str, Any]] = Field(
        description="Entities with entity_id, name, entity_type, properties"
    )
    count: int = Field(description="Number of entities found")
    query_used: str = Field(description="Query that was executed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── I/O models for get_entity_tool ──────────────────────────────────────────

class GetEntityInputs(BaseModel):
    """Required inputs for get_entity."""
    entity_id: str = Field(description="Entity ID (from search_entities_tool)")


class GetEntityOutputs(BaseModel):
    """Outputs for get_entity."""
    entity: Dict[str, Any] = Field(
        description="Entity details: entity_id, name, entity_type, properties"
    )
    related_document_ids: List[str] = Field(
        description="Document IDs that mention this entity"
    )
    related_document_count: int = Field(description="Number of related documents")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


# ── Tool functions ─────────────────────────────────────────────────────────

async def find_documents_by_entities_tool(
    entity_names: List[str],
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Find documents that mention specific entities (person, org, location, etc.).

    Uses the knowledge graph to return document IDs where the given entities
    are mentioned. Results are RLS filtered for the current user.
    """
    if not entity_names:
        formatted = "No entity names provided; no documents found."
        return {
            "document_ids": [],
            "count": 0,
            "formatted": formatted,
        }
    client = await get_backend_tool_client()
    doc_ids = await client.find_documents_by_entities(entity_names=entity_names, user_id=user_id)
    formatted = f"Found {len(doc_ids)} document(s) mentioning: {', '.join(entity_names[:5])}{'...' if len(entity_names) > 5 else ''}."
    if doc_ids:
        formatted += f"\nDocument IDs: {', '.join(doc_ids[:10])}{'...' if len(doc_ids) > 10 else ''}"
    return {
        "document_ids": doc_ids,
        "count": len(doc_ids),
        "formatted": formatted,
    }


async def find_related_documents_by_entities_tool(
    entity_names: List[str],
    max_hops: int = 1,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Find documents by traversing entity relationships in the knowledge graph.

    Starts from the given entities and follows relationships (e.g. co-occurrence)
    to discover related documents. Results are RLS filtered for the current user.
    """
    if not entity_names:
        formatted = "No entity names provided; no related documents found."
        return {
            "document_ids": [],
            "count": 0,
            "formatted": formatted,
        }
    client = await get_backend_tool_client()
    doc_ids = await client.find_related_documents_by_entities(
        entity_names=entity_names,
        max_hops=max_hops,
        user_id=user_id,
    )
    formatted = f"Found {len(doc_ids)} related document(s) via entity relationships (max_hops={max_hops})."
    if doc_ids:
        formatted += f"\nDocument IDs: {', '.join(doc_ids[:10])}{'...' if len(doc_ids) > 10 else ''}"
    return {
        "document_ids": doc_ids,
        "count": len(doc_ids),
        "formatted": formatted,
    }


async def find_co_occurring_entities_tool(
    entity_names: List[str],
    min_co_occurrences: int = 2,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Find entities that co-occur with the given entities in the same documents.

    Returns entity name, type, and co-occurrence count. Useful for expanding
    context or discovering related people, orgs, or locations.
    """
    if not entity_names:
        formatted = "No entity names provided; no co-occurring entities found."
        return {
            "entities": [],
            "count": 0,
            "formatted": formatted,
        }
    client = await get_backend_tool_client()
    entities = await client.find_co_occurring_entities(
        entity_names=entity_names,
        min_co_occurrences=min_co_occurrences,
        user_id=user_id,
    )
    formatted = f"Found {len(entities)} co-occurring entit{'y' if len(entities) == 1 else 'ies'} (min_co_occurrences={min_co_occurrences})."
    for e in entities[:5]:
        formatted += f"\n- {e.get('name', '')} ({e.get('type', '')}): {e.get('co_occurrence_count', 0)} co-occurrences"
    if len(entities) > 5:
        formatted += f"\n... and {len(entities) - 5} more"
    return {
        "entities": entities,
        "count": len(entities),
        "formatted": formatted,
    }


async def search_entities_tool(
    query: str,
    user_id: str = "system",
    entity_types: List[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search the knowledge graph for entities by name or keyword.

    Returns entities (person, org, location, etc.) matching the query.
    Use get_entity_tool to fetch details and related documents for a specific entity_id.
    """
    if not query or not query.strip():
        return {
            "entities": [],
            "count": 0,
            "query_used": "",
            "formatted": "No search query provided.",
        }
    client = await get_backend_tool_client()
    entities = await client.search_entities(
        query=query.strip(),
        user_id=user_id,
        entity_types=entity_types or [],
        limit=limit,
    )
    formatted = f"Found {len(entities)} entit{'y' if len(entities) == 1 else 'ies'} for '{query}'."
    for e in entities[:5]:
        name = e.get("name", "")
        etype = e.get("entity_type", "")
        eid = e.get("entity_id", "")
        formatted += f"\n- {name} ({etype}) [id: {eid}]"
    if len(entities) > 5:
        formatted += f"\n... and {len(entities) - 5} more"
    return {
        "entities": entities,
        "count": len(entities),
        "query_used": query.strip(),
        "formatted": formatted,
    }


async def get_entity_tool(
    entity_id: str,
    user_id: str = "system",
) -> Dict[str, Any]:
    """
    Get details for a single entity and the documents that mention it.

    Use after search_entities_tool when you need full entity details and related document IDs.
    """
    if not entity_id or not entity_id.strip():
        return {
            "entity": {},
            "related_document_ids": [],
            "related_document_count": 0,
            "formatted": "No entity_id provided.",
        }
    client = await get_backend_tool_client()
    result = await client.get_entity(entity_id=entity_id.strip(), user_id=user_id)
    if result is None:
        return {
            "entity": {},
            "related_document_ids": [],
            "related_document_count": 0,
            "formatted": f"Entity not found: {entity_id}",
        }
    entity = result.get("entity", {})
    related_docs = result.get("related_documents", [])
    name = entity.get("name", "")
    etype = entity.get("entity_type", "")
    formatted = f"Entity: **{name}** ({etype}). Related documents: {len(related_docs)}."
    if related_docs:
        formatted += f"\nDocument IDs: {', '.join(related_docs[:10])}{'...' if len(related_docs) > 10 else ''}"
    return {
        "entity": entity,
        "related_document_ids": list(related_docs),
        "related_document_count": len(related_docs),
        "formatted": formatted,
    }


# ── Registry entries ───────────────────────────────────────────────────────

register_action(
    name="find_documents_by_entities",
    category="search",
    description="Find documents that mention specific entities (person, org, location) in the knowledge graph",
    inputs_model=FindDocumentsByEntitiesInputs,
    params_model=None,
    outputs_model=FindDocumentsByEntitiesOutputs,
    tool_function=find_documents_by_entities_tool,
)

register_action(
    name="find_related_documents_by_entities",
    category="search",
    description="Find documents by traversing entity relationships (co-occurrence, etc.) in the knowledge graph",
    inputs_model=FindRelatedDocumentsByEntitiesInputs,
    params_model=FindRelatedDocumentsByEntitiesParams,
    outputs_model=FindRelatedDocumentsByEntitiesOutputs,
    tool_function=find_related_documents_by_entities_tool,
)

register_action(
    name="find_co_occurring_entities",
    category="knowledge",
    description="Find entities that co-occur with the given entities in the same documents",
    inputs_model=FindCoOccurringEntitiesInputs,
    params_model=FindCoOccurringEntitiesParams,
    outputs_model=FindCoOccurringEntitiesOutputs,
    tool_function=find_co_occurring_entities_tool,
)

register_action(
    name="search_entities",
    category="knowledge_graph",
    description="Search the knowledge graph for entities by name or keyword (person, org, location, etc.)",
    inputs_model=SearchEntitiesInputs,
    params_model=SearchEntitiesParams,
    outputs_model=SearchEntitiesOutputs,
    tool_function=search_entities_tool,
)

register_action(
    name="get_entity",
    category="knowledge_graph",
    description="Get details for an entity and the document IDs that mention it (use after search_entities)",
    inputs_model=GetEntityInputs,
    params_model=None,
    outputs_model=GetEntityOutputs,
    tool_function=get_entity_tool,
)
