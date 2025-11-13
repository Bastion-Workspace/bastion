"""
Entity Search Tool - MCP Tool for Knowledge Graph Entity Queries
Allows LLM to perform sophisticated entity-based searches using Neo4j knowledge graph
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntitySearchInput(BaseModel):
    """Input for entity-based search"""
    search_type: str = Field(..., description="REQUIRED: Type of entity search. Must be one of: 'by_entity' (find specific entities), 'related_entities' (find entities related to given ones), 'co_occurring' (find entities that appear together), 'entity_relationships' (get relationships), 'document_entities' (entities in a document), 'entity_importance' (importance scores)")
    entity_names: Optional[List[str]] = Field(None, description="List of entity names to search for")
    entity_type: Optional[str] = Field(None, description="Entity type filter (e.g., 'PERSON', 'ORGANIZATION', 'LOCATION')")
    document_id: Optional[str] = Field(None, description="Document ID for document-specific entity searches")
    max_hops: int = Field(2, ge=1, le=5, description="Maximum relationship hops for graph traversal")
    min_co_occurrences: int = Field(2, ge=1, description="Minimum co-occurrence count for co-occurring entities")
    min_shared_entities: int = Field(2, ge=1, description="Minimum shared entities for document similarity")
    limit: int = Field(100, ge=1, le=100, description="Maximum number of results")


class EntityResult(BaseModel):
    """Result from entity search"""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type")
    confidence: Optional[float] = Field(None, description="Entity confidence score")
    importance_score: Optional[float] = Field(None, description="Entity importance score")
    distance: Optional[int] = Field(None, description="Graph distance from source entity")
    co_occurrence_count: Optional[int] = Field(None, description="Number of co-occurrences")
    relationship_type: Optional[str] = Field(None, description="Type of relationship")
    target_name: Optional[str] = Field(None, description="Target entity name")


class DocumentEntityResult(BaseModel):
    """Result for document-entity searches"""
    document_id: str = Field(..., description="Document ID")
    shared_entities: Optional[int] = Field(None, description="Number of shared entities")
    entities: Optional[List[EntityResult]] = Field(None, description="Entities in document")


class EntitySearchOutput(BaseModel):
    """Output from entity search"""
    search_type: str = Field(..., description="Type of search performed")
    entities: List[EntityResult] = Field(..., description="Found entities")
    documents: List[DocumentEntityResult] = Field(..., description="Related documents")
    total_entities: int = Field(..., description="Total number of entities found")
    total_documents: int = Field(..., description="Total number of documents found")
    search_summary: str = Field(..., description="Summary of search results")
    search_time: float = Field(..., description="Search execution time")


class EntitySearchTool:
    """MCP tool for knowledge graph entity searches"""
    
    def __init__(self, knowledge_graph_service=None):
        """Initialize with required services"""
        self.knowledge_graph_service = knowledge_graph_service
        self.name = "search_by_entities"
        self.description = "Search knowledge graph using entities and relationships. Required: search_type must be one of: 'by_entity', 'related_entities', 'co_occurring', 'entity_relationships', 'document_entities', 'entity_importance'"
        
    async def initialize(self):
        """Initialize the entity search tool"""
        if not self.knowledge_graph_service:
            raise ValueError("KnowledgeGraphService is required")
        
        logger.info("🔍 EntitySearchTool initialized")
    
    async def execute(self, input_data: EntitySearchInput) -> ToolResponse:
        """Execute entity-based search"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Executing entity search: {input_data.search_type}")
            
            entities = []
            documents = []
            search_summary = ""
            
            if input_data.search_type == "by_entity":
                # Search for specific entities
                if input_data.entity_names:
                    entities = await self._search_by_entity_names(input_data.entity_names, input_data.entity_type, input_data.limit)
                    search_summary = f"Found {len(entities)} entities matching: {', '.join(input_data.entity_names)}"
                
            elif input_data.search_type == "related_entities":
                # Find entities related to given entities
                if input_data.entity_names:
                    entities = await self._find_related_entities(input_data.entity_names, input_data.max_hops, input_data.limit)
                    search_summary = f"Found {len(entities)} entities related to: {', '.join(input_data.entity_names)}"
                
            elif input_data.search_type == "co_occurring":
                # Find co-occurring entities
                if input_data.entity_names:
                    entities = await self._find_co_occurring_entities(input_data.entity_names, input_data.min_co_occurrences, input_data.limit)
                    search_summary = f"Found {len(entities)} entities co-occurring with: {', '.join(input_data.entity_names)}"
                
            elif input_data.search_type == "entity_relationships":
                # Get entity relationships
                if input_data.entity_names:
                    entities = await self._get_entity_relationships(input_data.entity_names[0])
                    search_summary = f"Found {len(entities)} relationships for entity: {input_data.entity_names[0]}"
                
            elif input_data.search_type == "document_entities":
                # Get entities in a specific document
                if input_data.document_id:
                    entities = await self._get_document_entities(input_data.document_id)
                    search_summary = f"Found {len(entities)} entities in document: {input_data.document_id}"
                
            elif input_data.search_type == "entity_importance":
                # Get entity importance scores
                if input_data.entity_names:
                    entities = await self._get_entity_importance(input_data.entity_names)
                    search_summary = f"Calculated importance scores for {len(entities)} entities"
                
            elif input_data.search_type == "documents_by_entities":
                # Find documents mentioning entities
                if input_data.entity_names:
                    doc_ids = await self.knowledge_graph_service.find_documents_by_entities(input_data.entity_names)
                    documents = [DocumentEntityResult(document_id=doc_id) for doc_id in doc_ids[:input_data.limit]]
                    search_summary = f"Found {len(documents)} documents mentioning: {', '.join(input_data.entity_names)}"
                
            elif input_data.search_type == "similar_documents":
                # Find documents similar by shared entities
                if input_data.document_id:
                    similar_docs = await self.knowledge_graph_service.get_document_similarity_by_entities(
                        input_data.document_id, input_data.min_shared_entities
                    )
                    documents = [
                        DocumentEntityResult(
                            document_id=doc["document_id"],
                            shared_entities=doc["shared_entities"]
                        ) for doc in similar_docs[:input_data.limit]
                    ]
                    search_summary = f"Found {len(documents)} documents similar to: {input_data.document_id}"
                
            else:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="INVALID_SEARCH_TYPE",
                        error_message=f"Unknown search type: {input_data.search_type}",
                        details={"valid_types": ["by_entity", "related_entities", "co_occurring", "entity_relationships", "document_entities", "entity_importance", "documents_by_entities", "similar_documents"]}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Create output
            output = EntitySearchOutput(
                search_type=input_data.search_type,
                entities=entities,
                documents=documents,
                total_entities=len(entities),
                total_documents=len(documents),
                search_summary=search_summary,
                search_time=time.time() - start_time
            )
            
            logger.info(f"✅ Entity search completed: {len(entities)} entities, {len(documents)} documents in {output.search_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Entity search failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="ENTITY_SEARCH_FAILED",
                    error_message=str(e),
                    details={"search_type": input_data.search_type}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _search_by_entity_names(self, entity_names: List[str], entity_type: Optional[str], limit: int) -> List[EntityResult]:
        """Search for specific entities by name"""
        entities = await self.knowledge_graph_service.get_entities(entity_type, limit)
        
        # Filter by provided names if specified
        if entity_names:
            entities = [e for e in entities if e["name"] in entity_names]
        
        return [EntityResult(**entity) for entity in entities]
    
    async def _find_related_entities(self, entity_names: List[str], max_hops: int, limit: int) -> List[EntityResult]:
        """Find entities related to given entities"""
        all_related = []
        for entity_name in entity_names:
            related = await self.knowledge_graph_service.find_related_entities(entity_name, max_hops)
            all_related.extend(related)
        
        # Remove duplicates and limit results
        seen = set()
        unique_related = []
        for entity in all_related:
            if entity["name"] not in seen:
                seen.add(entity["name"])
                unique_related.append(EntityResult(**entity))
                if len(unique_related) >= limit:
                    break
        
        return unique_related
    
    async def _find_co_occurring_entities(self, entity_names: List[str], min_co_occurrences: int, limit: int) -> List[EntityResult]:
        """Find co-occurring entities"""
        co_occurring = await self.knowledge_graph_service.find_co_occurring_entities(entity_names, min_co_occurrences)
        
        return [EntityResult(
            name=entity["name"],
            type=entity["type"],
            co_occurrence_count=entity["co_occurrence_count"]
        ) for entity in co_occurring[:limit]]
    
    async def _get_entity_relationships(self, entity_name: str) -> List[EntityResult]:
        """Get relationships for a specific entity"""
        relationships = await self.knowledge_graph_service.get_entity_relationships(entity_name)
        
        return [EntityResult(
            name=entity_name,
            type="RELATIONSHIP",
            relationship_type=rel["relationship_type"],
            target_name=rel["target_name"]
        ) for rel in relationships]
    
    async def _get_document_entities(self, document_id: str) -> List[EntityResult]:
        """Get entities in a specific document"""
        entities = await self.knowledge_graph_service.get_document_entities(document_id)
        
        return [EntityResult(**entity) for entity in entities]
    
    async def _get_entity_importance(self, entity_names: List[str]) -> List[EntityResult]:
        """Get importance scores for entities"""
        scores = await self.knowledge_graph_service.get_entity_importance_scores(entity_names)
        
        return [EntityResult(
            name=entity_name,
            type="IMPORTANCE",
            importance_score=score
        ) for entity_name, score in scores.items()]
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": EntitySearchInput.schema(),
            "outputSchema": EntitySearchOutput.schema()
        } 