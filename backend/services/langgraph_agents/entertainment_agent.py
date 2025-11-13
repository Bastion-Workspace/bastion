"""
Entertainment Agent - Roosevelt's "Entertainment Intelligence" Service
Movie and TV show recommendations, information, and comparison agent
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from services.langgraph_agents.base_agent import BaseAgent
from models.agent_response_models import EntertainmentResponse, TaskStatus
from config import settings

logger = logging.getLogger(__name__)


class EntertainmentAgent(BaseAgent):
    """
    Roosevelt's Entertainment Intelligence Agent
    Provides movie and TV recommendations, information lookup, and domain-specific comparisons
    """
    
    def __init__(self):
        super().__init__("entertainment_agent")
        logger.info("🎬 BULLY! Entertainment Agent ready to recommend movies and TV shows!")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for entertainment queries"""
        return """You are Roosevelt's Entertainment Intelligence Agent - an expert in movies and TV shows!

**MISSION**: Provide recommendations, information, and comparisons for movies and TV shows based on user's entertainment library.

**CAPABILITIES**:
1. **Information Lookup**: Detailed information about specific movies/TV shows
2. **Recommendations**: Suggest similar content based on user preferences
3. **Search**: Find movies/shows by genre, actor, director, or themes
4. **Comparison**: Domain-specific comparison (cast, ratings, genres, themes)

**STRUCTURED OUTPUT REQUIRED**:
You MUST respond with valid JSON matching this exact schema:

{
  "task_status": "complete|incomplete|error",
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
- Reference specific episodes for TV shows when relevant
- Use markdown formatting for better readability

**IMPORTANT**: Base all responses on the documents provided in the search results. If no relevant documents are found, acknowledge this and suggest the user add entertainment content to their library.
"""
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process entertainment requests with structured outputs"""
        try:
            logger.info("🎬 ENTERTAINMENT AGENT: Starting entertainment intelligence operation...")
            
            # Extract user message and context
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            user_id = state.get("user_id")
            
            # Detect query type
            query_type = self._detect_query_type(user_message)
            logger.info(f"🎯 Detected query type: {query_type}")
            
            # Search for entertainment content
            search_results = await self._search_entertainment_content(
                user_message, user_id, query_type
            )
            
            if not search_results:
                return await self._create_no_content_response(query_type)
            
            # **BULLY! GRAPH-POWERED RECOMMENDATIONS!** 🎬
            # For recommendation queries, enhance with Neo4j graph relationships
            graph_recommendations = []
            if query_type == "recommendation":
                graph_recommendations = await self._get_graph_recommendations(user_message)
            
            # Generate LLM response based on search results + graph recommendations
            llm_response = await self._generate_entertainment_response(
                user_message, search_results, query_type, graph_recommendations
            )
            
            # Create structured response
            structured_result = await self._create_structured_response(llm_response, state)
            
            logger.info(f"✅ ENTERTAINMENT AGENT: Successfully processed {query_type} query")
            return structured_result
            
        except Exception as e:
            logger.error(f"❌ ENTERTAINMENT AGENT: Processing failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return await self._create_error_response(str(e))
    
    def _detect_query_type(self, user_message: str) -> str:
        """Detect the type of entertainment query"""
        message_lower = user_message.lower()
        
        # Recommendation queries
        if any(keyword in message_lower for keyword in [
            "recommend", "suggestion", "similar to", "like", "what should i watch"
        ]):
            return "recommendation"
        
        # Comparison queries
        if any(keyword in message_lower for keyword in [
            "compare", "versus", "vs", "difference", "similar"
        ]):
            return "comparison"
        
        # Search queries
        if any(keyword in message_lower for keyword in [
            "find", "search", "show me", "list"
        ]):
            return "search"
        
        # Default to information lookup
        return "information"
    
    async def _search_entertainment_content(
        self, user_message: str, user_id: str, query_type: str
    ) -> List[Dict[str, Any]]:
        """
        Search for entertainment content using tags and semantic search
        
        Returns list of matching documents with content and metadata
        """
        try:
            from services.langgraph_tools.unified_search_tools import UnifiedSearchTools
            from repositories.document_repository import DocumentRepository
            from models.api_models import DocumentFilterRequest
            
            search_tools = UnifiedSearchTools()
            doc_repo = DocumentRepository()
            await doc_repo.initialize()
            
            # Extract entertainment-specific tags from query
            tags = await self._extract_entertainment_tags(user_message)
            logger.info(f"🎬 Extracted tags: {tags}")
            
            # Always include entertainment type tags
            entertainment_tags = ["movie", "tv_show", "tv_episode"]
            
            documents = []
            
            # If specific tags found, use tag-based retrieval
            if tags:
                filter_request = DocumentFilterRequest(
                    tags=tags,
                    limit=10,
                    skip=0,
                    sort_by="upload_date",
                    sort_order="desc"
                )
                
                db_documents, total = await doc_repo.filter_documents(filter_request)
                logger.info(f"📚 Found {len(db_documents)} documents with tags {tags}")
                
                # Retrieve full content
                for doc in db_documents:
                    full_doc = await search_tools.get_document(
                        document_id=str(doc.document_id), 
                        user_id=user_id
                    )
                    logger.info(f"📄 Document retrieval result for {doc.document_id}: success={full_doc.get('success') if full_doc else False}")
                    if full_doc:
                        if not full_doc.get("success"):
                            logger.warning(f"⚠️ Failed to get document content: {full_doc.get('error', 'Unknown error')}")
                        else:
                            content = full_doc.get("content", "")
                            logger.info(f"✅ Got document content, length: {len(content)}")
                            if content:
                                documents.append({
                                    "document_id": str(doc.document_id),
                                    "content": content,
                                    "metadata": full_doc.get("metadata", {}),
                                    "title": full_doc.get("metadata", {}).get("title") or doc.filename
                                })
                            else:
                                logger.warning(f"⚠️ Document {doc.document_id} has empty content")
            
            # Also try semantic search for entertainment content
            semantic_results = await search_tools.search_local(
                query=user_message,
                user_id=user_id,
                limit=5,
                search_types=["documents"],
                filter_tags=entertainment_tags if not tags else None
            )
            
            if semantic_results and semantic_results.get("success"):
                logger.info(f"🔍 Semantic search found {len(semantic_results.get('results', []))} results")
                for result in semantic_results.get("results", [])[:5]:
                    # Avoid duplicates
                    doc_id = result.get("document_id")
                    if not any(d["document_id"] == doc_id for d in documents):
                        full_doc = await search_tools.get_document(
                            document_id=doc_id,
                            user_id=user_id
                        )
                        if full_doc and full_doc.get("success"):
                            content = full_doc.get("content", "")
                            if content:
                                documents.append({
                                    "document_id": doc_id,
                                    "content": content,
                                    "metadata": full_doc.get("metadata", {}),
                                    "title": full_doc.get("metadata", {}).get("title") or result.get("filename")
                                })
            else:
                logger.info(f"🔍 Semantic search returned no results")
            
            logger.info(f"✅ Retrieved {len(documents)} entertainment documents")
            return documents[:10]  # Limit to 10 documents
            
        except Exception as e:
            logger.error(f"❌ Entertainment search failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def _extract_entertainment_tags(self, user_message: str) -> List[str]:
        """Extract entertainment-specific tags from query"""
        import re
        
        tags = []
        message_lower = user_message.lower()
        
        # Genre detection
        genres = [
            "action", "comedy", "drama", "horror", "thriller", "sci-fi", "science fiction",
            "fantasy", "romance", "mystery", "crime", "documentary", "animation",
            "adventure", "western", "war", "musical", "biographical", "historical"
        ]
        for genre in genres:
            if genre in message_lower:
                tags.append(genre.replace(" ", "_"))
        
        # Type detection
        if any(word in message_lower for word in ["movie", "film", "cinema"]):
            tags.append("movie")
        if any(word in message_lower for word in ["tv", "television", "series", "show"]):
            tags.append("tv_show")
        if any(word in message_lower for word in ["episode", "season"]):
            tags.append("tv_episode")
        
        # Extract decade references
        decades = re.findall(r'\b(19\d0s|20\d0s)\b', message_lower)
        tags.extend(decades)
        
        # Extract quoted titles or names
        quoted = re.findall(r'"([^"]+)"', user_message)
        quoted += re.findall(r"'([^']+)'", user_message)
        
        # Clean and normalize tags
        tags = [tag.strip().lower().replace(" ", "_") for tag in tags]
        
        return list(set(tags))  # Remove duplicates
    
    async def _get_graph_recommendations(self, user_message: str) -> List[Dict[str, Any]]:
        """
        Get graph-based recommendations using Neo4j relationship traversal
        
        **BULLY!** Relationship-based recommendations from the knowledge graph!
        """
        try:
            from services.service_container import service_container
            kg_service = service_container.kg_service
            
            if not kg_service:
                logger.warning("⚠️  KG service not available for graph recommendations")
                return []
            
            # Extract movie/show title from user message
            # Look for patterns like "like X", "similar to Y", "recommend movies like Z"
            patterns = [
                r'like\s+["\']?([^"\']+?)["\']?(?:\s|$)',
                r'similar to\s+["\']?([^"\']+?)["\']?(?:\s|$)',
                r'recommend.*?like\s+["\']?([^"\']+?)["\']?(?:\s|$)',
            ]
            
            work_title = None
            for pattern in patterns:
                match = re.search(pattern, user_message, re.IGNORECASE)
                if match:
                    work_title = match.group(1).strip()
                    break
            
            if not work_title:
                return []
            
            logger.info(f"🎬 Getting graph recommendations for: {work_title}")
            recommendations = await kg_service.get_entertainment_recommendations(work_title, limit=10)
            
            if recommendations:
                logger.info(f"✅ Found {len(recommendations)} graph-based recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get graph recommendations: {e}")
            return []
    
    async def _generate_entertainment_response(
        self, user_message: str, documents: List[Dict[str, Any]], query_type: str,
        graph_recommendations: List[Dict[str, Any]] = None
    ) -> str:
        """Generate LLM response based on search results"""
        try:
            # Prepare context from documents
            context_parts = []
            for i, doc in enumerate(documents[:5], 1):  # Limit to 5 for context
                title = doc.get("title", "Unknown")
                content = doc.get("content", "")[:2000]  # Limit content length
                context_parts.append(f"### Document {i}: {title}\n{content}\n")
            
            context = "\n".join(context_parts)
            
            # Add graph recommendations to context if available
            graph_context = ""
            if graph_recommendations:
                graph_context = "\n\n### Graph-Based Recommendations (from knowledge graph relationships):\n"
                for i, rec in enumerate(graph_recommendations[:5], 1):
                    graph_context += f"{i}. **{rec['title']}** ({rec.get('type', 'Movie')})\n"
                    if rec.get('year'):
                        graph_context += f"   - Year: {rec['year']}\n"
                    if rec.get('rating'):
                        graph_context += f"   - Rating: {rec['rating']}/10\n"
                    graph_context += f"   - Recommendation Score: {rec.get('score', 0)}\n"
                    graph_context += f"   - Shared Actors: {rec.get('shared_actors', 0)}, "
                    graph_context += f"Directors: {rec.get('shared_directors', 0)}, "
                    graph_context += f"Genres: {rec.get('shared_genres', 0)}\n"
            
            # Build prompt based on query type
            if query_type == "recommendation":
                task_instruction = "Provide movie/TV recommendations based on the user's request, the entertainment content below, and the graph-based recommendations from relationship analysis."
            elif query_type == "comparison":
                task_instruction = "Compare the movies/TV shows mentioned, highlighting similarities and differences in cast, themes, ratings, and overall quality."
            elif query_type == "search":
                task_instruction = "List and describe the movies/TV shows that match the user's search criteria."
            else:  # information
                task_instruction = "Provide detailed information about the requested movie or TV show."
            
            prompt = f"""{task_instruction}

**USER QUERY**: {user_message}

**AVAILABLE ENTERTAINMENT CONTENT**:
{context}{graph_context}

**INSTRUCTIONS**:
- Base your response on the documents provided above
- Use a conversational, enthusiastic tone
- Include specific details like cast, directors, genres, ratings
- For recommendations, explain WHY items are similar
- Format response in markdown for readability
- Structure your JSON response according to the schema provided in the system prompt
"""
            
            # Get chat service
            chat_service = await self._get_chat_service()
            
            # Generate response with structured output
            system_prompt = self._build_system_prompt()
            
            # Use OpenRouter client directly
            model_name = chat_service.current_model or settings.DEFAULT_MODEL
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"🤖 Calling LLM with model: {model_name}")
            logger.info(f"📝 Prompt length: {len(prompt)} chars, System prompt length: {len(system_prompt)} chars")
            logger.info(f"📄 Number of documents in context: {len(documents)}")
            
            try:
                response = await chat_service.openai_client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                logger.info(f"✅ LLM call completed")
                logger.info(f"📊 Response object type: {type(response)}")
                logger.info(f"📊 Response has choices: {hasattr(response, 'choices')}")
                
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    logger.info(f"📊 Number of choices: {len(response.choices)}")
                    logger.info(f"📊 First choice type: {type(response.choices[0])}")
                    logger.info(f"📊 First choice has message: {hasattr(response.choices[0], 'message')}")
                    
                    if hasattr(response.choices[0], 'message'):
                        message_content = response.choices[0].message.content
                        logger.info(f"📊 Message content type: {type(message_content)}")
                        logger.info(f"📊 Message content length: {len(message_content) if message_content else 0}")
                        
                        if message_content:
                            logger.info(f"✅ Got LLM response: {len(message_content)} chars")
                            logger.info(f"🔍 Response preview: {message_content[:200]}...")
                            return message_content
                        else:
                            logger.error(f"❌ LLM returned empty content!")
                            return ""
                    else:
                        logger.error(f"❌ Response choice has no message attribute!")
                        return ""
                else:
                    logger.error(f"❌ Response has no choices!")
                    return ""
                    
            except Exception as llm_error:
                logger.error(f"❌ LLM call failed with exception: {llm_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            
        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {e}")
            raise
    
    async def _create_structured_response(
        self, llm_response: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse LLM response and create structured state update"""
        try:
            # **ROOSEVELT FIX**: Strip markdown code fences if present
            # LLMs often wrap JSON in ```json ... ```
            cleaned_response = llm_response.strip()
            
            if cleaned_response.startswith("```json"):
                # Remove opening ```json
                cleaned_response = cleaned_response[len("```json"):].strip()
            elif cleaned_response.startswith("```"):
                # Remove opening ```
                cleaned_response = cleaned_response[3:].strip()
            
            if cleaned_response.endswith("```"):
                # Remove closing ```
                cleaned_response = cleaned_response[:-3].strip()
            
            logger.info(f"🔍 Cleaned response length: {len(cleaned_response)} chars")
            logger.info(f"🔍 Cleaned response preview: {cleaned_response[:200]}...")
            
            # Try to parse as JSON
            response_data = json.loads(cleaned_response)
            
            # Validate with Pydantic
            entertainment_response = EntertainmentResponse(**response_data)
            
            logger.info(f"✅ Successfully parsed structured response: {entertainment_response.task_status}")
            
            # Return state update
            return {
                "messages": state.get("messages", []),
                "agent_results": {
                    "entertainment_agent": entertainment_response.dict()
                },
                "is_complete": entertainment_response.task_status == TaskStatus.COMPLETE,
                "latest_response": entertainment_response.response
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse JSON response: {e}")
            logger.warning(f"⚠️ Attempting fallback to plain text response")
            # Fallback: treat as plain text response
            return await self._create_fallback_response(llm_response, state)
        except ValidationError as e:
            logger.error(f"❌ Pydantic validation failed: {e}")
            return await self._create_fallback_response(llm_response, state)
    
    async def _create_fallback_response(
        self, llm_response: str, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create fallback response when structured parsing fails"""
        entertainment_response = EntertainmentResponse(
            task_status=TaskStatus.COMPLETE,
            response=llm_response,
            content_type="mixed",
            items_found=[],
            confidence=0.7
        )
        
        return {
            "messages": state.get("messages", []),
            "agent_results": {
                "entertainment_agent": entertainment_response.dict()
            },
            "is_complete": True,
            "latest_response": llm_response
        }
    
    async def _create_no_content_response(self, query_type: str) -> Dict[str, Any]:
        """Create response when no entertainment content is found"""
        response_text = """🎬 **No Entertainment Content Found**

I couldn't find any movies or TV shows in your library that match your query. 

**To add entertainment content**:
1. Create markdown files with movie/TV show information
2. Tag them with `movie`, `tv_show`, or `tv_episode`
3. Include relevant tags like genres, actors, directors
4. Upload them to your document library

Once you've added some entertainment content, I'll be able to provide recommendations and information!"""
        
        entertainment_response = EntertainmentResponse(
            task_status=TaskStatus.INCOMPLETE,
            response=response_text,
            content_type="recommendation",
            items_found=[],
            confidence=1.0
        )
        
        return {
            "agent_results": {
                "entertainment_agent": entertainment_response.dict()
            },
            "is_complete": True,
            "latest_response": response_text
        }
    
    async def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        response_text = f"""❌ **Entertainment Agent Error**

I encountered an error while processing your entertainment query:

{error_message}

Please try rephrasing your query or contact support if the issue persists."""
        
        entertainment_response = EntertainmentResponse(
            task_status=TaskStatus.ERROR,
            response=response_text,
            content_type="mixed",
            items_found=[],
            confidence=0.0
        )
        
        return {
            "agent_results": {
                "entertainment_agent": entertainment_response.dict()
            },
            "is_complete": True,
            "latest_response": response_text
        }

