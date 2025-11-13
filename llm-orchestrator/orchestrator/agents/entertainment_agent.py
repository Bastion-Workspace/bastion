"""
Entertainment Agent for LLM Orchestrator
Movie and TV show recommendations, information, and comparison agent
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import BaseAgent, TaskStatus
from orchestrator.tools import search_documents_tool, get_document_content_tool

logger = logging.getLogger(__name__)


class EntertainmentAgent(BaseAgent):
    """
    Entertainment Intelligence Agent
    Provides movie and TV recommendations, information lookup, and comparisons
    """
    
    def __init__(self):
        super().__init__("entertainment_agent")
        logger.info("🎬 Entertainment Agent ready for recommendations!")
    
    def _build_entertainment_prompt(self) -> str:
        """Build system prompt for entertainment queries"""
        return """You are an Entertainment Intelligence Agent - an expert in movies and TV shows!

**MISSION**: Provide recommendations, information, and comparisons for movies and TV shows based on user's entertainment library.

**CAPABILITIES**:
1. **Information Lookup**: Detailed information about specific movies/TV shows
2. **Recommendations**: Suggest similar content based on user preferences
3. **Search**: Find movies/shows by genre, actor, director, or themes
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

**IMPORTANT**: Base all responses on the documents provided in the search results. If no relevant documents are found, acknowledge this and suggest the user add entertainment content to their library.
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
    
    async def _search_entertainment_content(
        self, query: str, query_type: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Search for entertainment content using document search"""
        try:
            documents = []
            
            # Generate search terms
            search_terms = self._extract_search_terms(query, query_type)
            logger.info(f"🎬 Search terms: {search_terms}")
            
            # Search for each term
            seen_ids = set()
            for term in search_terms[:3]:  # Limit to 3 searches
                search_result = await search_documents_tool(
                    query=term,
                    limit=5,
                    user_id=user_id
                )
                
                # Parse search results to extract document IDs
                doc_ids = re.findall(r'ID:\s*([a-f0-9\-]+)', search_result)
                
                for doc_id in doc_ids:
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        
                        # Get full document content
                        content = await get_document_content_tool(
                            document_id=doc_id,
                            user_id=user_id
                        )
                        
                        if content and not content.startswith("Document not found"):
                            documents.append({
                                "document_id": doc_id,
                                "content": content[:3000],  # Limit content length
                            })
                
                if len(documents) >= 5:
                    break
            
            logger.info(f"✅ Retrieved {len(documents)} entertainment documents")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Entertainment search failed: {e}")
            return []
    
    async def _generate_response(
        self, query: str, documents: List[Dict[str, Any]], query_type: str
    ) -> Dict[str, Any]:
        """Generate LLM response based on search results"""
        try:
            # Prepare context from documents
            if not documents:
                return {
                    "task_status": "complete",
                    "response": "🎬 **No Entertainment Content Found**\n\nI couldn't find any movies or TV shows in your library that match your query.\n\n**To add entertainment content**:\n1. Create markdown files with movie/TV show information\n2. Tag them with `movie`, `tv_show`, or `tv_episode`\n3. Include relevant details like genres, actors, directors\n4. Upload them to your document library\n\nOnce you've added some entertainment content, I'll be able to provide recommendations and information!",
                    "content_type": "recommendation",
                    "items_found": [],
                    "confidence": 1.0
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
            
            # Build messages
            messages = self._build_messages(system_prompt, prompt)
            
            logger.info(f"🤖 Calling LLM for entertainment response")
            
            # Get LLM response
            llm = self._get_llm(temperature=0.7)
            response = await llm.ainvoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            logger.info(f"✅ Got LLM response: {len(content)} chars")
            
            # Parse JSON response
            parsed_response = self._parse_json_response(content)
            
            # Ensure task_status is set
            if "task_status" not in parsed_response:
                parsed_response["task_status"] = "complete"
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return self._create_error_response(str(e))
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process entertainment query"""
        try:
            logger.info(f"🎬 Entertainment agent processing: {query[:100]}...")
            
            # Extract user_id from metadata
            metadata = metadata or {}
            user_id = metadata.get("user_id", "system")
            
            # Detect query type
            query_type = self._detect_query_type(query)
            logger.info(f"🎯 Detected query type: {query_type}")
            
            # Search for entertainment content
            documents = await self._search_entertainment_content(query, query_type, user_id)
            
            # Generate response
            result = await self._generate_response(query, documents, query_type)
            
            logger.info(f"✅ Entertainment agent completed: {result.get('task_status')}")
            return result
            
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

