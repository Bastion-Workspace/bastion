"""
Entertainment Agent for LLM Orchestrator
Movie and TV show recommendations, information, and comparison agent
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .base_agent import BaseAgent, TaskStatus
from orchestrator.tools.document_tools import (
    get_document_content_tool,
    search_documents_structured
)

logger = logging.getLogger(__name__)


class EntertainmentState(TypedDict):
    """State for entertainment agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    shared_memory: Dict[str, Any]
    query_type: str
    documents: List[Dict[str, Any]]
    response: Dict[str, Any]
    task_status: str
    error: str


class EntertainmentAgent(BaseAgent):
    """
    Entertainment Intelligence Agent
    Provides movie and TV recommendations, information lookup, and comparisons
    """
    
    def __init__(self):
        super().__init__("entertainment_agent")
        logger.info("🎬 Entertainment Agent ready for recommendations!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for entertainment agent"""
        workflow = StateGraph(EntertainmentState)
        
        # Add nodes
        workflow.add_node("detect_query_type", self._detect_query_type_node)
        workflow.add_node("search_content", self._search_content_node)
        workflow.add_node("generate_response", self._generate_response_node)
        
        # Entry point
        workflow.set_entry_point("detect_query_type")
        
        # Linear flow: detect_query_type -> search_content -> generate_response -> END
        workflow.add_edge("detect_query_type", "search_content")
        workflow.add_edge("search_content", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    def _build_entertainment_prompt(self) -> str:
        """Build system prompt for entertainment queries"""
        return """You are an Entertainment Intelligence Agent - an expert in movies and TV shows!

**MISSION**: Provide recommendations, information, and comparisons for movies and TV shows based on user's entertainment library.

**HOW YOU SEARCH**: You search the user's entertainment library using precise metadata tags.
- Documents are tagged with 'entertainment', 'movie', 'tv', or 'tv_show' tags
- You find and synthesize information from the user's personal entertainment dossiers
- Focus on what's actually in their knowledge base

**CAPABILITIES**:
1. **Information Lookup**: Detailed information about specific movies/TV shows in their library
2. **Recommendations**: Suggest similar content from what they have
3. **Search**: Find movies/shows in their collection
4. **Comparison**: Compare cast, ratings, genres, themes

**STRUCTURED OUTPUT REQUIRED**:
You MUST respond with valid JSON matching this exact schema:

{
  "task_status": "complete",
  "response": "Your formatted natural language response with entertainment information",
  "content_type": "movie|tv_show|tv_episode|mixed|recommendation",
  "items_found": [
    {
      "title": "Movie/Show Title",
      "type": "movie|tv_show|tv_episode",
      "year": "Year or range",
      "genre": "Genre(s)",
      "rating": "Rating if available",
      "summary": "Brief summary"
    }
  ],
  "recommendations": ["Title 1", "Title 2", "Title 3"],
  "comparison_summary": "Summary if comparing multiple items",
  "confidence": 0.85
}

**RESPONSE GUIDELINES**:
- Use a conversational, enthusiastic tone like a knowledgeable friend
- Include relevant details: cast, director, genre, ratings, plot summaries
- For recommendations, explain WHY items are similar
- For comparisons, highlight key differences and similarities
- Use markdown formatting for better readability

**IMPORTANT**:
- Base ALL responses on the documents provided from the user's entertainment library
- Each document represents a movie or TV show profile/dossier from their collection
- Synthesize information across multiple documents for comprehensive answers
- If a movie/TV show isn't in their library, acknowledge this limitation
- Focus on intelligent synthesis of their existing entertainment knowledge base
- Don't make up information - only use what's actually in their documents
"""
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of entertainment query"""
        query_lower = query.lower()
        
        # Recommendation queries
        if any(keyword in query_lower for keyword in [
            "recommend", "suggestion", "similar to", "like", "what should i watch"
        ]):
            return "recommendation"
        
        # Comparison queries
        if any(keyword in query_lower for keyword in [
            "compare", "versus", "vs", "difference", "similar"
        ]):
            return "comparison"
        
        # Search queries
        if any(keyword in query_lower for keyword in [
            "find", "search", "show me", "list"
        ]):
            return "search"
        
        # Default to information lookup
        return "information"
    
    def _extract_search_terms(self, query: str, query_type: str) -> List[str]:
        """Extract search terms for entertainment content"""
        search_terms = []
        
        # Add query type keywords
        if query_type == "recommendation":
            search_terms.append("movie recommendation tv show")
        elif query_type == "comparison":
            search_terms.append("movie tv show comparison")
        
        # Extract quoted titles
        quoted = re.findall(r'"([^"]+)"', query)
        quoted += re.findall(r"'([^']+)'", query)
        search_terms.extend(quoted)
        
        # Add the full query
        search_terms.append(query)
        
        # Add entertainment-specific terms
        search_terms.append("movie tv show film series")
        
        return search_terms
    
    async def _detect_query_type_node(self, state: EntertainmentState) -> Dict[str, Any]:
        """Detect the type of entertainment query"""
        try:
            query = state.get("query", "")
            query_type = self._detect_query_type(query)
            logger.info(f"🎯 Detected query type: {query_type}")
            
            return {
                "query_type": query_type
            }
        except Exception as e:
            logger.error(f"❌ Query type detection failed: {e}")
            return {
                "query_type": "information",  # Default fallback
                "error": str(e)
            }
    
    async def _search_content_node(self, state: EntertainmentState) -> Dict[str, Any]:
        """Search for entertainment content in local documents using structured tools"""
        try:
            query = state.get("query", "")
            query_type = state.get("query_type", "information")
            user_id = state.get("user_id", "system")
            
            documents = []

            # Use semantic search with entertainment-focused queries
            logger.info(f"🎬 Searching for entertainment content with query: {query[:100]}")

            # Build search queries that will find entertainment content
            search_queries = [
                f"{query} movie film entertainment",
                f"{query} tv show television entertainment",
                f"entertainment {query}",
                query  # Original query as fallback
            ]

            found_docs = []
            for search_query in search_queries:
                try:
                    # Use structured tool for document search (LangGraph best practice)
                    search_result = await search_documents_structured(
                        query=search_query,
                        user_id=user_id,
                        limit=10
                    )

                    if search_result.get('total_count', 0) > 0:
                        logger.info(f"✅ Found {search_result['total_count']} results for query: {search_query[:50]}")
                        found_docs.extend(search_result.get('results', []))
                        break  # Use first successful search

                except Exception as e:
                    logger.warning(f"Failed search with query '{search_query[:50]}': {e}")
                    continue

            if not found_docs:
                logger.info("🎬 No entertainment documents found")
                return {
                    "documents": []
                }

            # Remove duplicates based on document_id
            unique_docs = {}
            for doc in found_docs:
                doc_id = doc.get('document_id')
                if doc_id and doc_id not in unique_docs:
                    unique_docs[doc_id] = doc

            # Process documents - use content_preview if available, otherwise try full content
            logger.info(f"🎬 Processing {len(unique_docs)} documents")

            for doc_id, doc_info in list(unique_docs.items())[:5]:  # Limit to 5 documents
                try:
                    doc_title = doc_info.get('title') or doc_info.get('filename', f"Document {doc_id[:8]}")
                    content_preview = doc_info.get('content_preview', '')

                    # Use content_preview if it's substantial, otherwise get full content
                    if not content_preview or len(content_preview) < 200:
                        logger.info(f"🎬 Getting full content for {doc_id[:8]} (preview too short: {len(content_preview) if content_preview else 0} chars)")
                        try:
                            doc_content = await get_document_content_tool(doc_id, user_id)
                            
                            if doc_content and isinstance(doc_content, str) and "Document not found" not in doc_content and len(doc_content) > 0:
                                content_preview = doc_content
                                logger.info(f"✅ Got full content: {len(doc_content)} chars")
                            elif content_preview:
                                logger.info(f"Using existing preview: {len(content_preview)} chars")
                            else:
                                logger.warning(f"No content available for document {doc_id[:8]}")
                                continue
                        except Exception as e:
                            logger.warning(f"Failed to get full content for {doc_id[:8]}: {e}")
                            if not content_preview:
                                continue
                    else:
                        logger.info(f"Using content preview: {len(content_preview)} chars")

                    # Extract relevant content from available text
                    relevant_content = self._extract_relevant_content(content_preview, query)

                    if relevant_content and len(relevant_content.strip()) > 50:
                        documents.append({
                            "document_id": doc_id,
                            "title": doc_title,
                            "filename": doc_info.get('filename', ''),
                            "content": relevant_content,
                            "tags": doc_info.get('metadata', {}).get('tags', []),
                            "category": doc_info.get('metadata', {}).get('category', ''),
                            "source": "local"
                        })
                        logger.info(f"✅ Processed document {doc_title} ({len(relevant_content)} chars)")

                except Exception as e:
                    logger.warning(f"Failed to process document {doc_id}: {e}")
                    continue

            logger.info(f"✅ Retrieved content from {len(documents)} entertainment documents")
            return {
                "documents": documents
            }

        except Exception as e:
            logger.error(f"❌ Entertainment search failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "documents": [],
                "error": str(e)
            }

    def _extract_relevant_content(self, full_content: str, query: str, max_length: int = 3000) -> str:
        """Extract relevant content sections from entertainment document based on query"""
        try:
            content_lower = full_content.lower()
            query_lower = query.lower()

            # For entertainment queries, look for specific patterns
            relevant_patterns = []

            # Extract movie/TV show titles mentioned in query
            title_matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
            titles = [match[0] or match[1] for match in title_matches]

            # Add common entertainment keywords from query
            entertainment_keywords = [
                'director', 'directed', 'actor', 'actress', 'cast', 'starring',
                'genre', 'rating', 'year', 'release', 'plot', 'summary',
                'imdb', 'rotten tomatoes', 'review', 'critic', 'award',
                'sequel', 'prequel', 'franchise', 'series', 'episode'
            ]

            query_words = query_lower.split()
            found_keywords = [word for word in query_words if word in entertainment_keywords]

            # Build relevance patterns
            patterns = titles + found_keywords + ['movie', 'tv', 'show', 'film']

            # Split content into sections (markdown headers, paragraphs, etc.)
            sections = re.split(r'\n(#{1,6}\s+|\n\n+)', full_content)

            # Find sections with highest relevance
            relevant_sections = []
            section_scores = []

            for section in sections:
                if not section.strip():
                    continue

                section_lower = section.lower()
                score = 0

                # Title matches get highest score
                for title in titles:
                    if title.lower() in section_lower:
                        score += 10

                # Keyword matches
                for keyword in found_keywords:
                    if keyword in section_lower:
                        score += 5

                # General entertainment terms
                if any(term in section_lower for term in ['director', 'cast', 'genre', 'plot', 'rating']):
                    score += 2

                if score > 0:
                    section_scores.append((score, section))

            # Sort by relevance score
            section_scores.sort(reverse=True, key=lambda x: x[0])

            # Combine top sections, prioritizing highly relevant content
            combined_content = ""
            for score, section in section_scores[:4]:  # Take top 4 sections
                if len(combined_content) + len(section) > max_length:
                    # If adding this section would exceed limit, truncate it
                    remaining_space = max_length - len(combined_content)
                    if remaining_space > 100:  # Only add if we have meaningful space
                        combined_content += section[:remaining_space] + "...\n\n"
                    break
                else:
                    combined_content += section + "\n\n"

            # If we found some relevant content, return it
            if combined_content.strip():
                return combined_content.strip()

            # Fallback: if no specific sections found, try to find paragraphs containing entertainment info
            paragraphs = full_content.split('\n\n')
            relevant_paragraphs = []

            for para in paragraphs:
                para_lower = para.lower()
                if any(term in para_lower for term in patterns + ['movie', 'tv', 'show', 'film']):
                    relevant_paragraphs.append(para)

            if relevant_paragraphs:
                combined_content = '\n\n'.join(relevant_paragraphs[:3])
                return combined_content[:max_length]

            # Final fallback: return beginning of document
            return full_content[:max_length]

        except Exception as e:
            logger.warning(f"Failed to extract relevant content: {e}")
            # Fallback to returning first part of content
            return full_content[:max_length]
    
    async def _generate_response_node(self, state: EntertainmentState) -> Dict[str, Any]:
        """Generate LLM response based on search results"""
        try:
            query = state.get("query", "")
            documents = state.get("documents", [])
            query_type = state.get("query_type", "information")
            
            # Prepare context from documents
            if not documents:
                # Update shared_memory for conversation continuity
                shared_memory = state.get("shared_memory", {}) or {}
                shared_memory["primary_agent_selected"] = "entertainment_agent"
                shared_memory["last_agent"] = "entertainment_agent"
                
                no_content_response = {
                    "task_status": "complete",
                    "response": "🎬 **No Entertainment Content Found**\n\nI couldn't find any movies or TV shows in your library that match your query.\n\n**To add entertainment content**:\n1. Create markdown files with movie/TV show information\n2. Tag them with `movie`, `tv_show`, or `tv_episode`\n3. Include relevant details like genres, actors, directors\n4. Upload them to your document library\n\nOnce you've added some entertainment content, I'll be able to provide recommendations and information!",
                    "content_type": "recommendation",
                    "items_found": [],
                    "confidence": 1.0
                }
                return {
                    "response": no_content_response,
                    "task_status": "complete",
                    "shared_memory": shared_memory
                }
            
            context_parts = []
            for i, doc in enumerate(documents[:5], 1):
                content = doc.get("content", "")[:2000]
                context_parts.append(f"### Document {i}:\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Build task instruction based on query type
            if query_type == "recommendation":
                task_instruction = "Provide movie/TV recommendations based on the user's request and the entertainment content below."
            elif query_type == "comparison":
                task_instruction = "Compare the movies/TV shows mentioned, highlighting similarities and differences."
            elif query_type == "search":
                task_instruction = "List and describe the movies/TV shows that match the user's search criteria."
            else:  # information
                task_instruction = "Provide detailed information about the requested movie or TV show."
            
            prompt = f"""{task_instruction}

**USER QUERY**: {query}

**AVAILABLE ENTERTAINMENT CONTENT**:
{context}

**INSTRUCTIONS**:
- Base your response on the documents provided above
- Use a conversational, enthusiastic tone
- Include specific details like cast, directors, genres, ratings
- For recommendations, explain WHY items are similar
- Format response in markdown for readability
- Structure your JSON response according to the schema provided in the system prompt
"""
            
            # Build system prompt
            system_prompt = self._build_entertainment_prompt()
            
            # Extract conversation history for proper message structure
            # Build messages with conversation history using standardized helper
            messages_list = state.get("messages", [])
            messages = self._build_conversational_agent_messages(
                system_prompt=system_prompt,
                user_prompt=prompt,
                messages_list=messages_list,
                look_back_limit=6
            )
            
            logger.info(f"🤖 Calling LLM for entertainment response")
            
            # Get LLM response - pass state to access user's model selection
            llm = self._get_llm(temperature=0.7, state=state)
            response = await llm.ainvoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"✅ Got LLM response: {len(content)} chars")
            
            # Parse JSON response
            parsed_response = self._parse_json_response(content)
            
            # Ensure task_status is set
            if "task_status" not in parsed_response:
                parsed_response["task_status"] = "complete"
            
            # Update shared_memory for conversation continuity
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory["primary_agent_selected"] = "entertainment_agent"
            shared_memory["last_agent"] = "entertainment_agent"
            
            return {
                "response": parsed_response,
                "task_status": parsed_response.get("task_status", "complete"),
                "shared_memory": shared_memory
            }
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            # Update shared_memory even on error
            shared_memory = state.get("shared_memory", {}) or {}
            shared_memory["primary_agent_selected"] = "entertainment_agent"
            shared_memory["last_agent"] = "entertainment_agent"
            error_response = self._create_error_response(str(e))
            return {
                "response": error_response,
                "task_status": "error",
                "error": str(e),
                "shared_memory": shared_memory
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process entertainment query using LangGraph workflow"""
        try:
            logger.info(f"🎬 Entertainment agent processing: {query[:100]}...")
            
            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Get workflow (lazy initialization with checkpointer)
            workflow = await self._get_workflow()
            
            # Get checkpoint config (handles thread_id from conversation_id/user_id)
            config = self._get_checkpoint_config(metadata)
            
            # Prepare new messages (current query)
            new_messages = self._prepare_messages_with_query(messages, query)
            
            # Load and merge checkpointed messages to preserve conversation history
            conversation_messages = await self._load_and_merge_checkpoint_messages(
                workflow, config, new_messages
            )
            
            # Load shared_memory from checkpoint if available
            checkpoint_state = await workflow.aget_state(config)
            existing_shared_memory = {}
            if checkpoint_state and checkpoint_state.values:
                existing_shared_memory = checkpoint_state.values.get("shared_memory", {})
            
            # Merge with any shared_memory from metadata
            shared_memory = metadata.get("shared_memory", {}) or {}
            shared_memory.update(existing_shared_memory)
            metadata["shared_memory"] = shared_memory
            
            # Initialize state for LangGraph workflow
            initial_state: EntertainmentState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "shared_memory": shared_memory,
                "query_type": "",
                "documents": [],
                "response": {},
                "task_status": "",
                "error": ""
            }
            
            # Run LangGraph workflow with checkpointing
            result_state = await workflow.ainvoke(initial_state, config=config)
            
            # Extract final response
            response = result_state.get("response", {})
            task_status = result_state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = result_state.get("error", "Unknown error")
                logger.error(f"❌ Entertainment agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            logger.info(f"✅ Entertainment agent completed: {task_status}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Entertainment agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))


# Factory function for lazy loading
_entertainment_agent_instance = None

def get_entertainment_agent() -> EntertainmentAgent:
    """Get or create entertainment agent instance"""
    global _entertainment_agent_instance
    if _entertainment_agent_instance is None:
        _entertainment_agent_instance = EntertainmentAgent()
    return _entertainment_agent_instance

