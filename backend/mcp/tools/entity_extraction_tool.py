"""
Entity Extraction Tool - MCP Tool for Extracting and Storing Entities
Allows LLM to extract entities from text and populate the knowledge graph
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from mcp.schemas.tool_schemas import ToolResponse, ToolError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EntityExtractionInput(BaseModel):
    """Input for entity extraction"""
    text: str = Field(..., description="Text to extract entities from")
    document_id: Optional[str] = Field(None, description="Document ID to associate entities with")
    store_in_graph: bool = Field(True, description="Whether to store extracted entities in knowledge graph")
    extraction_method: str = Field("auto", description="Extraction method: 'auto', 'keywords', 'ner_model'")


class ExtractedEntity(BaseModel):
    """Extracted entity information"""
    name: str = Field(..., description="Entity name")
    type: str = Field(..., description="Entity type (PERSON, ORGANIZATION, LOCATION, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence score")
    start_position: Optional[int] = Field(None, description="Start position in text")
    end_position: Optional[int] = Field(None, description="End position in text")
    context: Optional[str] = Field(None, description="Surrounding context")


class EntityExtractionOutput(BaseModel):
    """Output from entity extraction"""
    entities: List[ExtractedEntity] = Field(..., description="Extracted entities")
    total_extracted: int = Field(..., description="Total number of entities extracted")
    entity_types: Dict[str, int] = Field(..., description="Count by entity type")
    extraction_summary: str = Field(..., description="Summary of extraction results")
    stored_in_graph: bool = Field(..., description="Whether entities were stored in knowledge graph")
    extraction_time: float = Field(..., description="Time taken for extraction")


class EntityExtractionTool:
    """MCP tool for entity extraction and knowledge graph population"""
    
    def __init__(self, knowledge_graph_service=None, chat_service=None):
        """Initialize with required services"""
        self.knowledge_graph_service = knowledge_graph_service
        self.chat_service = chat_service
        self.name = "extract_entities"
        self.description = "Extract entities from text and optionally store in knowledge graph"
        
    async def initialize(self):
        """Initialize the entity extraction tool"""
        if not self.knowledge_graph_service:
            raise ValueError("KnowledgeGraphService is required")
        
        logger.info("🔍 EntityExtractionTool initialized")
    
    async def execute(self, input_data: EntityExtractionInput) -> ToolResponse:
        """Execute entity extraction"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Extracting entities from text ({len(input_data.text)} characters)")
            
            # Extract entities based on method
            if input_data.extraction_method == "auto":
                entities = await self._extract_entities_auto(input_data.text)
            elif input_data.extraction_method == "keywords":
                entities = await self._extract_entities_keywords(input_data.text)
            elif input_data.extraction_method == "ner_model":
                entities = await self._extract_entities_ner(input_data.text)
            else:
                return ToolResponse(
                    success=False,
                    error=ToolError(
                        error_code="INVALID_EXTRACTION_METHOD",
                        error_message=f"Unknown extraction method: {input_data.extraction_method}",
                        details={"valid_methods": ["auto", "keywords", "ner_model"]}
                    ),
                    execution_time=time.time() - start_time
                )
            
            # Store in knowledge graph if requested
            stored_in_graph = False
            if input_data.store_in_graph and input_data.document_id and entities:
                await self._store_entities_in_graph(entities, input_data.document_id)
                stored_in_graph = True
            
            # Count by entity type
            entity_types = {}
            for entity in entities:
                entity_type = entity.type
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            # Create summary
            extraction_summary = f"Extracted {len(entities)} entities from {len(input_data.text)} characters"
            if entity_types:
                type_list = [f"{count} {entity_type}" for entity_type, count in entity_types.items()]
                extraction_summary += f" ({', '.join(type_list)})"
            
            # Create output
            output = EntityExtractionOutput(
                entities=entities,
                total_extracted=len(entities),
                entity_types=entity_types,
                extraction_summary=extraction_summary,
                stored_in_graph=stored_in_graph,
                extraction_time=time.time() - start_time
            )
            
            logger.info(f"✅ Entity extraction completed: {len(entities)} entities in {output.extraction_time:.2f}s")
            
            return ToolResponse(
                success=True,
                data=output,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return ToolResponse(
                success=False,
                error=ToolError(
                    error_code="EXTRACTION_FAILED",
                    error_message=str(e),
                    details={"text_length": len(input_data.text)}
                ),
                execution_time=time.time() - start_time
            )
    
    async def _extract_entities_auto(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using automatic method (combines multiple approaches)"""
        entities = []
        
        # Try NER model first
        ner_entities = await self._extract_entities_ner(text)
        entities.extend(ner_entities)
        
        # If NER didn't find much, try keyword extraction
        if len(entities) < 5:
            keyword_entities = await self._extract_entities_keywords(text)
            # Add keyword entities that aren't already found
            existing_names = {e.name.lower() for e in entities}
            for entity in keyword_entities:
                if entity.name.lower() not in existing_names:
                    entities.append(entity)
        
        return entities
    
    async def _extract_entities_keywords(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using keyword-based approach"""
        entities = []
        words = text.split()
        
        # Look for capitalized words (potential names)
        for i, word in enumerate(words):
            # Clean the word
            clean_word = word.strip('.,!?;:()[]{}"\'').strip()
            
            if (len(clean_word) > 2 and 
                clean_word[0].isupper() and 
                not clean_word.isupper() and  # Not all caps
                not clean_word.isdigit()):  # Not a number
                
                # Determine entity type based on context
                entity_type = self._classify_entity_type(clean_word, words, i)
                
                # Get context
                start_pos = max(0, i - 2)
                end_pos = min(len(words), i + 3)
                context = " ".join(words[start_pos:end_pos])
                
                entities.append(ExtractedEntity(
                    name=clean_word,
                    type=entity_type,
                    confidence=0.7,  # Medium confidence for keyword extraction
                    start_position=text.find(clean_word),
                    end_position=text.find(clean_word) + len(clean_word),
                    context=context
                ))
        
        # Remove duplicates and limit results
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.name.lower() not in seen:
                seen.add(entity.name.lower())
                unique_entities.append(entity)
        
        return unique_entities[:20]  # Limit to 20 entities
    
    async def _extract_entities_ner(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using NER model (placeholder for now)"""
        # This would use a proper NER model like spaCy, transformers, etc.
        # For now, using the knowledge graph service's basic extraction
        raw_entities = await self.knowledge_graph_service.extract_entities_from_text(text)
        
        entities = []
        for raw_entity in raw_entities:
            entities.append(ExtractedEntity(
                name=raw_entity['name'],
                type=raw_entity['type'],
                confidence=raw_entity['confidence'],
                start_position=text.find(raw_entity['name']),
                end_position=text.find(raw_entity['name']) + len(raw_entity['name'])
            ))
        
        return entities
    
    def _classify_entity_type(self, word: str, all_words: List[str], position: int) -> str:
        """Classify entity type based on word characteristics and context"""
        # Simple classification logic
        if word.endswith(('Corp', 'Inc', 'Ltd', 'LLC', 'Company', 'Organization')):
            return "ORGANIZATION"
        elif word.endswith(('City', 'State', 'Country', 'Republic', 'Kingdom')):
            return "LOCATION"
        elif any(title in word for title in ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'President']):
            return "PERSON"
        elif word.isdigit() or word.replace('.', '').isdigit():
            return "NUMBER"
        else:
            # Default to PERSON for capitalized words
            return "PERSON"
    
    async def _store_entities_in_graph(self, entities: List[ExtractedEntity], document_id: str):
        """Store extracted entities in the knowledge graph"""
        try:
            # Convert to Entity model format
            from models.api_models import Entity
            
            entity_models = []
            for entity in entities:
                entity_models.append(Entity(
                    name=entity.name,
                    entity_type=entity.type,
                    confidence=entity.confidence
                ))
            
            # Store in knowledge graph
            await self.knowledge_graph_service.store_entities(entity_models, document_id)
            logger.info(f"🔗 Stored {len(entities)} entities in knowledge graph for document {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store entities in graph: {e}")
            raise
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get MCP tool definition for registration"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": EntityExtractionInput.schema(),
            "outputSchema": EntityExtractionOutput.schema()
        } 