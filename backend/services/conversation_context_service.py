"""
Conversation Context Service - Manages conversation context for research planning
Provides entity extraction, conversation summarization, and reference resolution
"""

import logging
import json
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of entities that can be extracted from conversations"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    TIME_PERIOD = "time_period"


@dataclass
class Entity:
    """Represents an entity extracted from conversation"""
    name: str
    entity_type: EntityType
    aliases: List[str] = None
    context: str = ""
    first_mentioned: datetime = None
    last_mentioned: datetime = None
    mention_count: int = 0
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class ResearchContext:
    """Comprehensive conversation context for research planning"""
    entities: List[Entity]
    summary: str
    topics: List[str]
    recent_subjects: List[str]
    entity_relationships: Dict[str, List[str]]
    conversation_id: str
    last_updated: datetime
    context_score: float  # 0-1 confidence in context quality


class ConversationContextService:
    """Service for extracting and managing conversation context"""
    
    def __init__(self, mcp_chat_service=None, db_pool=None):
        self.mcp_chat_service = mcp_chat_service
        self.db_pool = db_pool
        self.entity_cache = {}  # Cache for entity extraction results
        
    async def extract_conversation_context(self, conversation_id: str, 
                                         limit: int = 20) -> ResearchContext:
        """Extract comprehensive context from conversation"""
        try:
            logger.info(f"🔍 Extracting conversation context for: {conversation_id}")
            
            # Get recent messages
            messages = await self._get_recent_messages(conversation_id, limit)
            if not messages:
                return self._create_empty_context(conversation_id)
            
            # Extract entities and relationships
            entities = await self._extract_entities(messages)
            
            # Generate conversation summary
            summary = await self._generate_conversation_summary(messages)
            
            # Identify topics and themes
            topics = await self._identify_topics(messages)
            
            # Get recent subjects
            recent_subjects = self._get_recent_subjects(messages, entities)
            
            # Map entity relationships
            entity_relationships = self._map_entity_relationships(entities, messages)
            
            # Calculate context confidence score
            context_score = self._calculate_context_score(entities, summary, topics)
            
            context = ResearchContext(
                entities=entities,
                summary=summary,
                topics=topics,
                recent_subjects=recent_subjects,
                entity_relationships=entity_relationships,
                conversation_id=conversation_id,
                last_updated=datetime.utcnow(),
                context_score=context_score
            )
            
            logger.info(f"✅ Extracted context with {len(entities)} entities, {len(topics)} topics")
            return context
            
        except Exception as e:
            logger.error(f"❌ Failed to extract conversation context: {e}")
            return self._create_empty_context(conversation_id)
    
    async def _get_recent_messages(self, conversation_id: str, limit: int) -> List[Dict]:
        """Get recent messages from conversation"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    messages = await conn.fetch(
                        """
                        SELECT content, message_type, created_at, metadata_json
                        FROM conversation_messages 
                        WHERE conversation_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                        """, conversation_id, limit
                    )
                    
                    # Convert to list and reverse for chronological order
                    message_list = []
                    for row in messages:
                        message_list.append({
                            "content": row["content"],
                            "role": row["message_type"],
                            "timestamp": row["created_at"].isoformat(),
                            "metadata": json.loads(row["metadata_json"] or "{}")
                        })
                    
                    return list(reversed(message_list))  # Chronological order
            
            # Fallback to MCP chat service if no database
            if self.mcp_chat_service:
                return await self.mcp_chat_service._get_recent_conversation(
                    session_id=conversation_id, conversation_id=conversation_id
                )
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Failed to get recent messages: {e}")
            return []
    
    async def _extract_entities(self, messages: List[Dict]) -> List[Entity]:
        """Extract entities from conversation messages"""
        try:
            # Combine all message content
            full_text = " ".join([msg["content"] for msg in messages])
            
            # Use LLM to extract entities if available
            if self.mcp_chat_service:
                entities = await self._extract_entities_with_llm(full_text, messages)
            else:
                entities = self._extract_entities_simple(full_text)
            
            # Update entity metadata
            for entity in entities:
                self._update_entity_metadata(entity, messages)
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Failed to extract entities: {e}")
            return []
    
    async def _extract_entities_with_llm(self, text: str, messages: List[Dict]) -> List[Entity]:
        """Extract entities using LLM for better accuracy"""
        try:
            extraction_prompt = f"""
            Extract entities from this conversation text. Focus on:
            - People (names, titles, pronouns that can be resolved)
            - Organizations (companies, institutions, groups)
            - Locations (places, countries, cities)
            - Concepts (ideas, policies, theories)
            - Events (historical events, meetings, occurrences)
            - Time periods (dates, eras, timeframes)
            
            Conversation text:
            {text}
            
            Return JSON array of entities:
            [
                {{
                    "name": "entity name",
                    "type": "person|organization|location|concept|event|time_period",
                    "aliases": ["alternative names", "pronouns"],
                    "context": "brief description of role/meaning"
                }}
            ]
            
            Focus on entities that are important for research planning.
            """
            
            conversation_log = [
                {"role": "system", "content": "You are an entity extraction specialist. Extract only the JSON array."},
                {"role": "user", "content": extraction_prompt}
            ]
            
            response = await self.mcp_chat_service._get_llm_response(conversation_log)
            
            # Parse JSON response
            try:
                entities_data = json.loads(response)
                entities = []
                
                for entity_data in entities_data:
                    entity = Entity(
                        name=entity_data["name"],
                        entity_type=EntityType(entity_data["type"]),
                        aliases=entity_data.get("aliases", []),
                        context=entity_data.get("context", "")
                    )
                    entities.append(entity)
                
                return entities
                
            except json.JSONDecodeError:
                logger.warning("⚠️ Failed to parse LLM entity extraction response")
                return self._extract_entities_simple(text)
                
        except Exception as e:
            logger.error(f"❌ LLM entity extraction failed: {e}")
            return self._extract_entities_simple(text)
    
    def _extract_entities_simple(self, text: str) -> List[Entity]:
        """Simple entity extraction using basic patterns"""
        entities = []
        
        # Simple patterns for common entity types
        # This is a fallback when LLM is not available
        import re
        
        # Extract potential names (capitalized words)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        names = re.findall(name_pattern, text)
        
        for name in set(names):
            if len(name.split()) >= 2:  # Likely a person name
                entities.append(Entity(
                    name=name,
                    entity_type=EntityType.PERSON,
                    aliases=[name.split()[0], name.split()[-1]]  # First and last name
                ))
        
        return entities
    
    def _update_entity_metadata(self, entity: Entity, messages: List[Dict]):
        """Update entity metadata with mention tracking"""
        entity.mention_count = 0
        first_mention = None
        last_mention = None
        
        for msg in messages:
            content = msg["content"].lower()
            entity_name = entity.name.lower()
            
            # Check for entity mentions
            if entity_name in content or any(alias.lower() in content for alias in entity.aliases):
                entity.mention_count += 1
                
                timestamp = datetime.fromisoformat(msg["timestamp"])
                
                if first_mention is None:
                    first_mention = timestamp
                
                last_mention = timestamp
        
        entity.first_mentioned = first_mention
        entity.last_mentioned = last_mention
    
    async def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """Generate a summary of the conversation"""
        try:
            if not messages:
                return "No conversation history available."
            
            # Combine recent messages (last 10 for summary)
            recent_messages = messages[-10:]
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            if self.mcp_chat_service:
                summary_prompt = f"""
                Summarize this conversation in 2-3 sentences, focusing on:
                - Main topics discussed
                - Key entities mentioned
                - Current focus or ongoing discussion
                
                Conversation:
                {conversation_text}
                
                Provide a concise summary that would be useful for research planning.
                """
                
                conversation_log = [
                    {"role": "system", "content": "You are a conversation summarizer. Provide concise, research-relevant summaries."},
                    {"role": "user", "content": summary_prompt}
                ]
                
                summary = await self.mcp_chat_service._get_llm_response(conversation_log)
                return summary.strip()
            
            else:
                # Simple fallback summary
                return f"Conversation with {len(messages)} messages covering various topics."
                
        except Exception as e:
            logger.error(f"❌ Failed to generate conversation summary: {e}")
            return "Unable to generate conversation summary."
    
    async def _identify_topics(self, messages: List[Dict]) -> List[str]:
        """Identify main topics from conversation"""
        try:
            if not messages:
                return []
            
            # Combine recent messages
            recent_messages = messages[-10:]
            conversation_text = " ".join([msg["content"] for msg in recent_messages])
            
            if self.mcp_chat_service:
                topic_prompt = f"""
                Identify 3-5 main topics from this conversation that would be relevant for research planning.
                Focus on subjects, themes, or areas of interest.
                
                Conversation:
                {conversation_text}
                
                Return a JSON array of topic strings:
                ["topic1", "topic2", "topic3"]
                """
                
                conversation_log = [
                    {"role": "system", "content": "You are a topic identification specialist. Return only the JSON array."},
                    {"role": "user", "content": topic_prompt}
                ]
                
                response = await self.mcp_chat_service._get_llm_response(conversation_log)
                
                try:
                    topics = json.loads(response)
                    return topics if isinstance(topics, list) else []
                except json.JSONDecodeError:
                    logger.warning("⚠️ Failed to parse LLM topic identification response")
                    return self._identify_topics_simple(conversation_text)
            
            else:
                return self._identify_topics_simple(conversation_text)
                
        except Exception as e:
            logger.error(f"❌ Failed to identify topics: {e}")
            return []
    
    def _identify_topics_simple(self, text: str) -> List[str]:
        """Simple topic identification using keyword patterns"""
        # Basic keyword extraction as fallback
        import re
        
        # Common research-related keywords
        research_keywords = [
            "policy", "history", "technology", "science", "politics", 
            "business", "economics", "culture", "society", "education",
            "health", "environment", "military", "diplomacy", "art"
        ]
        
        topics = []
        text_lower = text.lower()
        
        for keyword in research_keywords:
            if keyword in text_lower:
                topics.append(keyword.title())
        
        return topics[:5]  # Limit to 5 topics
    
    def _get_recent_subjects(self, messages: List[Dict], entities: List[Entity]) -> List[str]:
        """Get recent subjects from conversation and entities"""
        subjects = []
        
        # Add entity names as subjects
        for entity in entities:
            if entity.mention_count > 0:
                subjects.append(entity.name)
        
        # Add topics from recent messages
        recent_content = " ".join([msg["content"] for msg in messages[-5:]])
        
        # Extract potential subjects (capitalized phrases)
        import re
        subject_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_subjects = re.findall(subject_pattern, recent_content)
        
        for subject in potential_subjects:
            if subject not in subjects and len(subject.split()) <= 3:
                subjects.append(subject)
        
        return subjects[:10]  # Limit to 10 subjects
    
    def _map_entity_relationships(self, entities: List[Entity], messages: List[Dict]) -> Dict[str, List[str]]:
        """Map relationships between entities"""
        relationships = {}
        
        for entity in entities:
            relationships[entity.name] = []
        
        # Simple relationship detection based on proximity
        for msg in messages:
            content = msg["content"]
            
            for entity1 in entities:
                for entity2 in entities:
                    if entity1.name != entity2.name:
                        # Check if entities appear in same message
                        if entity1.name in content and entity2.name in content:
                            if entity2.name not in relationships[entity1.name]:
                                relationships[entity1.name].append(entity2.name)
        
        return relationships
    
    def _calculate_context_score(self, entities: List[Entity], summary: str, topics: List[str]) -> float:
        """Calculate confidence score for context quality"""
        score = 0.0
        
        # Entity score (0-0.4)
        if entities:
            score += min(0.4, len(entities) * 0.1)
        
        # Summary score (0-0.3)
        if summary and len(summary) > 50:
            score += 0.3
        
        # Topic score (0-0.3)
        if topics:
            score += min(0.3, len(topics) * 0.1)
        
        return min(1.0, score)
    
    def _create_empty_context(self, conversation_id: str) -> ResearchContext:
        """Create empty context when no conversation data is available"""
        return ResearchContext(
            entities=[],
            summary="No conversation history available.",
            topics=[],
            recent_subjects=[],
            entity_relationships={},
            conversation_id=conversation_id,
            last_updated=datetime.utcnow(),
            context_score=0.0
        )
    
    async def resolve_references(self, query: str, context: ResearchContext) -> str:
        """Resolve pronouns and vague references using conversation context"""
        try:
            if not context.entities or context.context_score < 0.3:
                return query  # Return original if insufficient context
            
            if self.mcp_chat_service:
                resolution_prompt = f"""
                Resolve pronouns and references in this query using conversation context.
                Replace vague references with specific entities when clear from context.
                
                Query: "{query}"
                Recent Entities: {[e.name for e in context.entities]}
                Entity Relationships: {context.entity_relationships}
                Conversation Summary: {context.summary}
                
                Instructions:
                - Replace pronouns (he/she/it/they) with specific names when clear
                - Replace vague references (the company/the policy) with specific entities
                - Keep original query if context is unclear
                - Only resolve when you're confident about the reference
                
                Return the resolved query, or the original if unclear.
                """
                
                conversation_log = [
                    {"role": "system", "content": "You are a reference resolution specialist. Only resolve when confident."},
                    {"role": "user", "content": resolution_prompt}
                ]
                
                resolved_query = await self.mcp_chat_service._get_llm_response(conversation_log)
                return resolved_query.strip()
            
            else:
                # Simple fallback resolution
                return self._resolve_references_simple(query, context)
                
        except Exception as e:
            logger.error(f"❌ Failed to resolve references: {e}")
            return query
    
    def _resolve_references_simple(self, query: str, context: ResearchContext) -> str:
        """Simple reference resolution using basic patterns"""
        resolved_query = query
        
        # Simple pronoun resolution
        if context.entities:
            # Find most recently mentioned person
            recent_person = None
            for entity in context.entities:
                if entity.entity_type == EntityType.PERSON and entity.last_mentioned:
                    if recent_person is None or entity.last_mentioned > recent_person.last_mentioned:
                        recent_person = entity
            
            if recent_person:
                # Replace common pronouns
                resolved_query = resolved_query.replace(" his ", f" {recent_person.name}'s ")
                resolved_query = resolved_query.replace(" her ", f" {recent_person.name}'s ")
                resolved_query = resolved_query.replace(" their ", f" {recent_person.name}'s ")
        
        return resolved_query
