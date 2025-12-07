"""
Content Analysis Agent - LangGraph Implementation
Provides higher-level analysis for both fiction and non-fiction content beyond grammar/style.
Handles both story analysis and article/opinion piece critique in one unified agent.
Supports multi-document comparative analysis.
"""

import logging
import json
import re
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import ValidationError

from .base_agent import BaseAgent, TaskStatus
from orchestrator.tools.document_tools import (
    get_document_content_tool,
    find_documents_by_tags_tool,
    search_documents_structured
)
from orchestrator.clients.backend_tool_client import get_backend_tool_client
from orchestrator.models.content_analysis_models import (
    ArticleAnalysisResult,
    ComparisonAnalysisResult,
    DocumentSummary
)

logger = logging.getLogger(__name__)


class ContentAnalysisState(TypedDict):
    """State for content analysis agent LangGraph workflow"""
    query: str
    user_id: str
    metadata: Dict[str, Any]
    messages: List[Any]
    persona: Optional[Dict[str, Any]]
    active_editor: Optional[Dict[str, Any]]
    request_type: str  # "single_analysis" | "comparison" | "unknown"
    documents: List[Dict[str, Any]]
    document_content: str
    analysis_result: Dict[str, Any]
    response: Dict[str, Any]
    task_status: str
    error: str


class ContentAnalysisAgent(BaseAgent):
    """
    Content Analysis Agent for document critique and comparison
    
    Handles:
    - Single document analysis (non-fiction articles, op-eds)
    - Multi-document comparative analysis
    - Document retrieval by title, author, or tags
    Uses LangGraph workflow for explicit state management
    """
    
    def __init__(self):
        super().__init__("content_analysis_agent")
        self._grpc_client = None
        logger.info("Content Analysis Agent ready!")
    
    def _build_workflow(self, checkpointer) -> StateGraph:
        """Build LangGraph workflow for content analysis agent"""
        workflow = StateGraph(ContentAnalysisState)
        
        # Add nodes
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("detect_request_type", self._detect_request_type_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("analyze_content", self._analyze_content_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Entry point
        workflow.set_entry_point("prepare_context")
        
        # Flow: prepare_context -> detect_request_type -> retrieve_documents -> analyze_content -> format_response -> END
        workflow.add_edge("prepare_context", "detect_request_type")
        workflow.add_edge("detect_request_type", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "analyze_content")
        workflow.add_edge("analyze_content", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile(checkpointer=checkpointer)
    
    async def _get_grpc_client(self):
        """Get or create gRPC client for backend tools"""
        if self._grpc_client is None:
            self._grpc_client = await get_backend_tool_client()
        return self._grpc_client
    
    async def _get_model_context_window(self, model_name: str) -> int:
        """
        Get model context window size in tokens
        
        Uses conservative fallback since we don't have direct access to chat_service.
        In production, this could be exposed via gRPC.
        
        Returns context window size in tokens
        """
        try:
            # Try to get from metadata if available
            # For now, use conservative fallback based on common model patterns
            model_lower = model_name.lower()
            
            # Common model context windows
            if "claude-3-5-sonnet" in model_lower or "claude-sonnet-4" in model_lower:
                return 200000  # 200K tokens
            elif "claude-3-opus" in model_lower or "claude-opus" in model_lower:
                return 200000
            elif "claude-3-sonnet" in model_lower:
                return 200000
            elif "gpt-4" in model_lower and "turbo" in model_lower:
                return 128000  # 128K tokens
            elif "gpt-4" in model_lower:
                return 8192  # 8K tokens (older GPT-4)
            elif "gpt-3.5" in model_lower:
                return 16385  # 16K tokens
            else:
                # Conservative fallback for unknown models
                logger.warning(f"Unknown model {model_name}, using conservative 8000 token fallback")
                return 8000
                
        except Exception as e:
            logger.warning(f"Failed to get model context window: {e}, using conservative fallback")
            return 8000
    
    async def _calculate_document_char_limit(self, model_name: str, num_documents: int = 1) -> int:
        """
        Calculate dynamic character limit per document based on model context window
        
        Args:
            model_name: Name of the LLM model
            num_documents: Number of documents being processed
        
        Returns:
            Character limit per document
        """
        context_window = await self._get_model_context_window(model_name)
        
        # Reserve tokens for:
        # - System prompt: ~500 tokens
        # - User instructions: ~500 tokens
        # - Response generation: ~2000 tokens
        # - Safety buffer: 20%
        reserved_tokens = 3000
        safety_multiplier = 0.8
        
        available_tokens = int((context_window - reserved_tokens) * safety_multiplier)
        
        # Divide available tokens by number of documents
        tokens_per_doc = available_tokens // max(num_documents, 1)
        
        # Convert tokens to characters (rough estimate: 1 token ≈ 4 chars)
        chars_per_doc = tokens_per_doc * 4
        
        logger.info(
            f"Model capacity: {model_name} has {context_window:,} token context window\n"
            f"   Available for content: {available_tokens:,} tokens\n"
            f"   Per document ({num_documents} docs): {tokens_per_doc:,} tokens (~{chars_per_doc:,} chars)"
        )
        
        return chars_per_doc
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for content analysis"""
        return (
            "You are a content analysis expert evaluating both fiction and non-fiction works. "
            "For fiction: Focus on plot, characters, themes, pacing, and story elements.\n"
            "For non-fiction: Focus on thesis clarity, argument structure, evidentiary support, counterarguments, tone/audience fit, and actionable improvements.\n\n"
            "STRUCTURED OUTPUT REQUIRED: Return ONLY valid JSON matching this exact schema:\n"
            "{\n"
            '  "task_status": "complete",\n'
            '  "summary": "Overall assessment in 2-4 sentences",\n'
            '  "thesis_clarity": "Assessment of thesis/stance clarity",\n'
            '  "structure_coherence": "Assessment of organization and flow",\n'
            '  "evidence_quality": "Assessment of evidence, citations, examples",\n'
            '  "counterarguments": "Coverage of counterarguments and fairness",\n'
            '  "tone_audience_fit": "Tone and audience alignment",\n'
            '  "strengths": ["Key strength 1", "Key strength 2"],\n'
            '  "weaknesses": ["Key weakness 1", "Key weakness 2"],\n'
            '  "recommendations": ["Concrete improvement 1", "Concrete improvement 2"],\n'
            '  "missing_elements": ["Missing element 1", "Missing element 2"],\n'
            '  "outline_suggestions": ["Outline suggestion 1", "Outline suggestion 2"],\n'
            '  "verdict": "solid|needs_more",\n'
            '  "confidence": 0.8\n'
            "}\n\n"
            "CRITICAL: Use these exact field names. Do not use 'overallStrength' or other variations."
        )
    
    def _detect_comparison_request(self, user_message: str) -> bool:
        """Detect if user is asking for multi-document comparison"""
        comparison_keywords = [
            "compare", "contrast", "difference", "differences", "similarity", "similarities",
            "conflict", "conflicts", "contradiction", "contradictions", "discrepancy", "discrepancies",
            "versus", "vs", "between", "across"
        ]
        
        user_message_lower = user_message.lower()
        for keyword in comparison_keywords:
            if keyword in user_message_lower:
                logger.info(f"Comparison detected: Found keyword '{keyword}' in query")
                return True
        
        return False
    
    async def _extract_titles_from_query(self, user_message: str) -> List[str]:
        """Extract document titles or filenames from user query"""
        patterns = [
            r"'([^']+\.(?:pdf|epub|docx|txt|md))'",
            r'"([^"]+\.(?:pdf|epub|docx|txt|md))"',
            r"file\s+(?:named|called)\s+['\"]?([^'\"]+)['\"]?",
            r"document\s+['\"]?([^'\"]+)['\"]?",
            r"['\"]([^'\"]+)['\"]",
        ]
        
        titles = []
        for pattern in patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            titles.extend(matches)
        
        bare_files = re.findall(r'\b(\w+\.(?:pdf|epub|docx|txt|md))\b', user_message, re.IGNORECASE)
        titles.extend(bare_files)
        
        titles = list(set(titles))
        
        if titles:
            logger.info(f"Title detection: Found {len(titles)} potential titles: {titles}")
        
        return titles
    
    async def _extract_author_from_query(self, user_message: str) -> str:
        """Extract author name from user query"""
        patterns = [
            r"by\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",
            r"author\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",
            r"written\s+by\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_message)
            if match:
                author = match.group(1).strip()
                logger.info(f"Author detection: Found author: {author}")
                return author
        
        return ""
    
    async def _extract_tags_from_query(self, user_message: str, user_id: str) -> List[str]:
        """Extract document tags from user query"""
        try:
            # Use search to find documents, then extract tags from results
            search_result = await search_documents_structured(user_message, limit=20, user_id=user_id)
            
            if search_result.get("total_count", 0) > 0:
                # Extract unique tags from search results
                all_tags = set()
                for doc in search_result.get("results", []):
                    metadata = doc.get("metadata", {})
                    doc_tags = metadata.get("tags", [])
                    if isinstance(doc_tags, list):
                        all_tags.update(doc_tags)
                
                tags = list(all_tags)
                if tags:
                    logger.info(f"Tag extraction: Found tags: {tags}")
                    return tags[:5]  # Limit to top 5 tags
            
            return []
            
        except Exception as e:
            logger.error(f"Tag extraction failed: {e}")
            return []
    
    async def _prepare_context_node(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Prepare context: extract metadata, active editor, persona"""
        try:
            logger.info(f"Preparing context for content analysis: {state['query'][:100]}...")
            
            metadata = state.get("metadata", {})
            persona = metadata.get("persona")
            active_editor = metadata.get("active_editor") or {}
            
            return {
                "persona": persona,
                "active_editor": active_editor
            }
            
        except Exception as e:
            logger.error(f"Context preparation failed: {e}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _detect_request_type_node(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Detect if this is a comparison request or single document analysis"""
        try:
            query = state["query"]
            is_comparison = self._detect_comparison_request(query)
            
            request_type = "comparison" if is_comparison else "single_analysis"
            
            logger.info(f"Request type detected: {request_type}")
            
            return {
                "request_type": request_type
            }
            
        except Exception as e:
            logger.error(f"Request type detection failed: {e}")
            return {
                "request_type": "unknown",
                "error": str(e)
            }
    
    async def _retrieve_documents_node(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Retrieve documents based on query (by title, author, tags, or active editor)"""
        try:
            query = state["query"]
            user_id = state["user_id"]
            request_type = state.get("request_type", "single_analysis")
            active_editor = state.get("active_editor", {})
            
            documents = []
            document_content = ""
            
            if request_type == "comparison":
                # Multi-document comparison: try titles, author, then tags
                titles = await self._extract_titles_from_query(query)
                if titles:
                    # Search for documents by title
                    for title in titles:
                        search_result = await search_documents_structured(title, limit=5, user_id=user_id)
                        if search_result.get("total_count", 0) > 0:
                            documents.extend(search_result.get("results", []))
                
                if not documents:
                    author = await self._extract_author_from_query(query)
                    if author:
                        search_result = await search_documents_structured(author, limit=8, user_id=user_id)
                        if search_result.get("total_count", 0) > 0:
                            documents.extend(search_result.get("results", []))
                
                if not documents:
                    tags = await self._extract_tags_from_query(query, user_id)
                    if tags:
                        docs_by_tags = await find_documents_by_tags_tool(tags, user_id=user_id, limit=8)
                        documents.extend(docs_by_tags)
                
                # Limit to 8 documents for comparison
                documents = documents[:8]
                
                # Check if we hit the document limit and need to warn the user
                if len(documents) >= 8:
                    logger.warning("Document limit reached (8 documents). User should be more specific.")
                
                logger.info(f"Retrieved {len(documents)} documents for comparison")
                
            else:
                # Single document analysis: try title first, then active editor
                titles = await self._extract_titles_from_query(query)
                if titles and user_id:
                    search_result = await search_documents_structured(titles[0], limit=1, user_id=user_id)
                    if search_result.get("total_count", 0) > 0:
                        doc_result = search_result.get("results", [])[0]
                        doc_id = doc_result.get("document_id")
                        if doc_id:
                            content = await get_document_content_tool(doc_id, user_id)
                            if content and not content.startswith("Error"):
                                document_content = content
                                documents = [doc_result]
                                logger.info(f"Retrieved document by title: {doc_result.get('title', 'Unknown')}")
                
                # Fall back to active editor
                if not document_content and active_editor:
                    document_content = active_editor.get("content", "") or ""
                    filename = active_editor.get("filename", "document.md")
                    frontmatter = active_editor.get("frontmatter", {}) or {}
                    doc_type = str((frontmatter.get('type') or '')).strip().lower()
                    
                    if document_content:
                        documents = [{
                            "document_id": "active_editor",
                            "title": filename,
                            "filename": filename,
                            "metadata": {"doc_type": doc_type}
                        }]
                        logger.info(f"Using active editor content: {filename}")
                
                # Check if fiction document (should route to story_analysis_agent)
                if documents:
                    doc_metadata = documents[0].get("metadata", {})
                    doc_type = doc_metadata.get("doc_type", "")
                    if doc_type == "fiction":
                        logger.warning("Fiction document detected - should route to story_analysis_agent")
                        return {
                            "error": "Fiction documents should be analyzed by the Story Analysis Agent",
                            "task_status": "error"
                        }
            
            if not documents and not document_content:
                logger.warning("No documents found for analysis")
                return {
                    "error": "No documents found. Please specify a document title, open a document in the editor, or use a comparison query.",
                    "task_status": "error"
                }
            
            return {
                "documents": documents,
                "document_content": document_content
            }
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _analyze_content_node(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Analyze content: single document or multi-document comparison"""
        try:
            query = state["query"]
            request_type = state.get("request_type", "single_analysis")
            documents = state.get("documents", [])
            document_content = state.get("document_content", "")
            user_id = state["user_id"]
            
            if request_type == "comparison":
                # Multi-document comparison
                if len(documents) < 2:
                    return {
                        "error": "Comparison requires at least 2 documents. Found fewer.",
                        "task_status": "error"
                    }
                
                # Retrieve full content for all documents
                full_documents = []
                for doc in documents:
                    doc_id = doc.get("document_id")
                    if doc_id and doc_id != "active_editor":
                        content = await get_document_content_tool(doc_id, user_id)
                        if content and not content.startswith("Error"):
                            full_documents.append({
                                "document_id": doc_id,
                                "title": doc.get("title", "Unknown"),
                                "content": content,
                                "metadata": doc.get("metadata", {})
                            })
                
                if len(full_documents) < 2:
                    return {
                        "error": "Could not retrieve content for enough documents to compare.",
                        "task_status": "error"
                    }
                
                # Get model name for intelligent comparison - pass state to access user's model selection
                llm = self._get_llm(temperature=0.3, state=state)
                model_name = llm.model_name if hasattr(llm, 'model_name') else "claude-3-5-sonnet-20241022"
                
                # Perform intelligent comparison analysis (chooses direct vs summarization)
                comparison_result: ComparisonAnalysisResult = await self._compare_multiple_documents(
                    full_documents, query, model_name
                )
                
                # Convert Pydantic model to dict for state
                return {
                    "analysis_result": comparison_result.dict(),
                    "task_status": "complete"
                }
                
            else:
                # Single document analysis
                if not document_content:
                    # Retrieve content if we have document_id
                    if documents and documents[0].get("document_id") != "active_editor":
                        doc_id = documents[0].get("document_id")
                        document_content = await get_document_content_tool(doc_id, user_id)
                
                if not document_content:
                    return {
                        "error": "No document content available for analysis.",
                        "task_status": "error"
                    }
                
                # Perform single document analysis
                analysis_result = await self._analyze_single_document(document_content, query, documents[0] if documents else {})
                
                return {
                    "analysis_result": analysis_result,
                    "task_status": "complete"
                }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "task_status": "error"
            }
    
    async def _analyze_single_document(self, content: str, query: str, doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single document"""
        try:
            filename = doc_metadata.get("title", doc_metadata.get("filename", "document"))
            doc_type = doc_metadata.get("metadata", {}).get("doc_type", "non-fiction")
            
            system_prompt = self._build_system_prompt()
            user_prompt = (
                "=== ARTICLE CONTEXT ===\n"
                f"Filename: {filename}\nType: {doc_type}\n\n"
                f"{content}\n\n"
                "TASKS:\n"
                "1) Assess overall strength as an opinion piece.\n"
                "2) Identify missing arguments, weak sections, and needed additions.\n"
                "3) Provide concrete recommendations and outline bullets to strengthen it.\n"
                "4) Set verdict: 'solid' or 'needs_more'.\n"
            )
            
            llm = self._get_llm(temperature=0.2, state=state)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            content_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            json_text = self._unwrap_json_response(content_text)
            if '```json' in json_text:
                match = re.search(r'```json\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            
            try:
                data = json.loads(json_text)
                # Validate with Pydantic model
                try:
                    result = ArticleAnalysisResult(**data)
                    return result.dict()
                except ValidationError as e:
                    logger.warning(f"ArticleAnalysisResult validation failed: {e}, using raw data")
                    return data
            except json.JSONDecodeError:
                # Fallback structure
                fallback_data = {
                    "task_status": "complete",
                    "summary": content_text[:800],
                    "thesis_clarity": "fair",
                    "structure_coherence": "fair",
                    "evidence_quality": "unknown",
                    "counterarguments": "unknown",
                    "tone_audience_fit": "general",
                    "strengths": [],
                    "weaknesses": [],
                    "recommendations": ["Expand arguments and add concrete evidence."],
                    "missing_elements": [],
                    "outline_suggestions": [],
                    "verdict": "needs_more",
                    "confidence": 0.6,
                    "metadata": {}
                }
                try:
                    result = ArticleAnalysisResult(**fallback_data)
                    return result.dict()
                except ValidationError:
                    return fallback_data
            
            return data
            
        except Exception as e:
            logger.error(f"Single document analysis failed: {e}")
            return {
                "task_status": "error",
                "error": str(e)
            }
    
    async def _compare_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        user_message: str,
        model_name: str = None
    ) -> ComparisonAnalysisResult:
        """
        Perform comparative analysis across multiple documents using intelligent summarization strategy
        
        Strategy:
        1. First summarize each document individually (keeps each under token limits)
        2. Then compare the summaries (much smaller total token count)
        3. If needed, dive deeper into specific sections
        
        Returns ComparisonAnalysisResult with structured data
        """
        try:
            total_chars = sum(len(doc.get("content", "")) for doc in documents)
            total_estimated_tokens = total_chars // 4
            
            logger.info(f"Total content: {total_chars:,} characters (~{total_estimated_tokens:,} tokens)")
            
            # Get model name from state or use default - pass state to access user's model selection
            if model_name is None:
                llm = self._get_llm(temperature=0.3, state=state)
                model_name = llm.model_name if hasattr(llm, 'model_name') else "claude-3-5-sonnet-20241022"
            
            # Token-aware strategy: If over 400K tokens, use summarization approach
            # (Conservative threshold to account for OpenRouter limits vs base model limits)
            if total_estimated_tokens > 400000:
                logger.info("Large comparison: Using intelligent summarization strategy")
                return await self._compare_with_summaries(documents, user_message, model_name)
            else:
                logger.info("Small comparison: Direct comparison feasible")
                return await self._compare_direct(documents, user_message, model_name)
                
        except Exception as e:
            logger.error(f"Document comparison failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return structured error result
            return ComparisonAnalysisResult(
                task_status="error",
                key_similarities=[],
                key_differences=[],
                conflicts_contradictions=[],
                unique_contributions={},
                overall_assessment=f"Document comparison failed: {str(e)}",
                dominant_themes=[],
                gaps_missing_perspectives=[],
                reading_recommendations=[],
                follow_up_questions=[],
                documents_compared=len(documents),
                document_titles=[doc.get("title", f"Document {i+1}") for i, doc in enumerate(documents)],
                comparison_strategy="direct",
                confidence_level=0.0
            )
    
    async def _summarize_single_document(
        self,
        doc: Dict[str, Any],
        doc_index: int,
        model_name: str,
        char_limit: int
    ) -> DocumentSummary:
        """
        Summarize a single document with structured output
        
        Returns DocumentSummary with structured data
        """
        title = doc.get("title", f"Document {doc_index}")
        content = doc.get("content", "")
        char_count = len(content)
        
        logger.info(f"Summarizing Document {doc_index}: '{title}' ({char_count:,} chars, limit: {char_limit:,})")
        
        # Use dynamic char limit based on model capacity
        content_to_summarize = content[:char_limit]
        if len(content) > char_limit:
            logger.info(f"   Document truncated from {char_count:,} to {char_limit:,} chars")
        
        # Build structured output schema for DocumentSummary
        summary_schema = json.dumps({
            "title": "string - document title",
            "summary": "string - comprehensive 300-500 word summary",
            "main_topics": ["list", "of", "primary topics"],
            "key_arguments": ["list", "of", "main arguments or findings"],
            "key_data_points": ["list", "of", "important data/dates/facts"],
            "author_perspective": "string - author's stance or conclusions (optional)",
            "unique_insights": ["list", "of", "unique aspects"],
            "document_id": "string - optional document ID",
            "original_length": "integer - original document length",
            "confidence": "float 0.0-1.0 - confidence in summary quality"
        }, indent=2)
        
        summary_prompt = f"""
**DOCUMENT TO SUMMARIZE**: {title}

**CONTENT** ({len(content_to_summarize):,} characters):
{content_to_summarize}

**TASK**: Create a comprehensive structured summary that captures:
1. Main topic/subject matter (2-3 sentences)
2. Key arguments or findings (3-5 bullet points)
3. Important data, dates, or facts (specific details)
4. Author's perspective or conclusions
5. Any unique insights or notable aspects

**STRUCTURED OUTPUT REQUIRED**: Return ONLY valid JSON matching this exact schema:
{summary_schema}

Provide detailed, specific information in each field. The summary field should be 300-500 words.
"""
        
        try:
            llm = self._get_llm(temperature=0.2, model=model_name)
            messages = [
                SystemMessage(content="You are an expert document summarizer. Create structured, comprehensive summaries with specific details. Return ONLY valid JSON."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            raw_content = response.content if hasattr(response, 'content') else str(response)
            
            # Truncation detection
            finish_reason = getattr(response, 'response_metadata', {}).get('finish_reason', '')
            if finish_reason == "length":
                logger.warning(f"Truncation: Summary for {title} was cut off (hit max_tokens limit)")
            
            # Parse JSON response
            json_text = raw_content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            
            # Parse with Pydantic
            try:
                summary_data = json.loads(json_text)
                summary_obj = DocumentSummary(**summary_data)
                logger.info(f"Summarized {title} - {len(summary_obj.summary)} chars, {len(summary_obj.key_arguments)} arguments")
                return summary_obj
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Failed to parse structured summary for {title}: {e}")
                logger.warning(f"   Raw response preview: {raw_content[:200]}...")
                
                # Fallback to basic summary
                fallback_summary = f"Document contains {char_count:,} characters."
                
                # Try to extract summary field from malformed JSON
                try:
                    if '"summary"' in raw_content:
                        summary_match = re.search(r'"summary":\s*"([^"]*(?:"[^"]*)*)"', raw_content, re.DOTALL)
                        if summary_match:
                            extracted = summary_match.group(1)
                            extracted = extracted.replace('\\"', '"').replace('\\n', ' ')
                            if len(extracted) > 100:
                                fallback_summary = extracted[:1000]
                                logger.info(f"Extracted summary from malformed JSON: {len(extracted)} chars")
                except Exception:
                    pass
                
                # Fallback to basic summary
                return DocumentSummary(
                    title=title,
                    summary=fallback_summary,
                    main_topics=[],
                    key_arguments=[],
                    key_data_points=[],
                    unique_insights=[],
                    original_length=char_count,
                    confidence=0.5
                )
                
        except Exception as e:
            logger.error(f"Failed to summarize {title}: {e}")
            return DocumentSummary(
                title=title,
                summary=f"Summary unavailable. Document contains {char_count:,} characters.",
                main_topics=[],
                key_arguments=[],
                key_data_points=[],
                unique_insights=[],
                original_length=char_count,
                confidence=0.3
            )
    
    async def _compare_with_summaries(
        self,
        documents: List[Dict[str, Any]],
        user_message: str,
        model_name: str
    ) -> ComparisonAnalysisResult:
        """
        Compare documents using a two-phase summarization strategy with PARALLEL processing
        
        Phase 1: Summarize ALL documents in PARALLEL using asyncio.gather()
        Phase 2: Compare the summaries with structured output
        
        Returns ComparisonAnalysisResult with structured data
        """
        try:
            logger.info("Phase 1: Summarizing documents IN PARALLEL...")
            
            # Calculate dynamic char limit per document
            char_limit = await self._calculate_document_char_limit(model_name, len(documents))
            
            # Parallel execution: Summarize all documents simultaneously
            summarization_tasks = [
                self._summarize_single_document(doc, i+1, model_name, char_limit)
                for i, doc in enumerate(documents)
            ]
            
            # Execute all summarizations in parallel
            document_summaries: List[DocumentSummary] = await asyncio.gather(*summarization_tasks)
            
            logger.info(f"Phase 1 complete: Summarized {len(document_summaries)} documents in parallel")
            
            # Phase 2: Compare the summaries with structured output
            logger.info("Phase 2: Comparing document summaries with structured output...")
            
            # Build comparison prompt with summaries
            summary_text_parts = []
            for summary_obj in document_summaries:
                summary_text_parts.append(f"### {summary_obj.title}")
                summary_text_parts.append(f"**Size**: {summary_obj.original_length:,} characters")
                summary_text_parts.append(f"**Summary**: {summary_obj.summary}")
                if summary_obj.main_topics:
                    summary_text_parts.append(f"**Topics**: {', '.join(summary_obj.main_topics)}")
                if summary_obj.key_arguments:
                    summary_text_parts.append(f"**Key Arguments**:")
                    for arg in summary_obj.key_arguments:
                        summary_text_parts.append(f"  - {arg}")
                summary_text_parts.append("")
            
            # Build structured comparison schema
            comparison_schema = json.dumps({
                "task_status": "complete",
                "key_similarities": ["list of themes/ideas/patterns across documents"],
                "key_differences": ["list of how documents differ in approach/focus/conclusions"],
                "conflicts_contradictions": ["list of contradictory claims or conflicting information"],
                "unique_contributions": {
                    "Document Title 1": ["unique insights from this document"],
                    "Document Title 2": ["unique insights from this document"]
                },
                "overall_assessment": "synthesized collective picture from all documents",
                "dominant_themes": ["most prominent themes across collection"],
                "gaps_missing_perspectives": ["important gaps or missing perspectives"],
                "reading_recommendations": [
                    {"title": "Document Title", "reason": "why read this first"}
                ],
                "follow_up_questions": ["suggested questions for deeper analysis"],
                "documents_compared": len(documents),
                "document_titles": [doc.get("title") for doc in documents],
                "comparison_strategy": "summarize_then_compare",
                "confidence_level": 0.85
            }, indent=2)
            
            comparison_prompt = f"""
# Multi-Document Comparative Analysis

**Analysis based on {len(document_summaries)} documents**
**User Request**: {user_message}

## Document Summaries

{chr(10).join(summary_text_parts)}

## Your Task

Provide a comprehensive comparative analysis based on the document summaries above.

**STRUCTURED OUTPUT REQUIRED**: Return ONLY valid JSON matching this exact schema:
{comparison_schema}

**Analysis Requirements**:
1. **key_similarities**: Identify themes, ideas, or patterns that appear across documents
2. **key_differences**: How documents differ in approach, focus, conclusions, or perspectives
3. **conflicts_contradictions**: Any contradictory claims, conflicting information, or competing viewpoints
4. **unique_contributions**: For EACH document by title, what unique perspectives or insights it offers
5. **overall_assessment**: The collective picture that emerges from all documents together (2-3 paragraphs)
6. **dominant_themes**: The 3-5 most prominent themes across the collection
7. **gaps_missing_perspectives**: What's missing or underexplored across all documents
8. **reading_recommendations**: Which documents to read first and why (prioritize 2-3)
9. **follow_up_questions**: 3-5 questions for deeper analysis

**CRITICAL**: Use actual document titles in unique_contributions and reading_recommendations, not generic labels!
"""
            
            llm = self._get_llm(temperature=0.3, model=model_name)
            messages = [
                SystemMessage(content="You are an expert comparative analyst. Provide structured, detailed comparative analysis. Return ONLY valid JSON."),
                HumanMessage(content=comparison_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            raw_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            json_text = raw_content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            
            # Parse with Pydantic
            try:
                comparison_data = json.loads(json_text)
                comparison_result = ComparisonAnalysisResult(**comparison_data)
                
                # Add document summaries to result
                comparison_result.document_summaries = document_summaries
                
                logger.info(f"Phase 2 complete: Structured comparison analysis completed")
                logger.info(f"   - {len(comparison_result.key_similarities)} similarities")
                logger.info(f"   - {len(comparison_result.key_differences)} differences")
                logger.info(f"   - {len(comparison_result.conflicts_contradictions)} conflicts")
                logger.info(f"   - Confidence: {comparison_result.confidence_level:.2f}")
                
                return comparison_result
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse structured comparison result: {e}")
                logger.error(f"   Raw response: {raw_content[:500]}...")
                
                # Fallback to basic comparison result
                return ComparisonAnalysisResult(
                    task_status="complete",
                    key_similarities=[],
                    key_differences=[],
                    conflicts_contradictions=[],
                    unique_contributions={},
                    overall_assessment=raw_content[:2000] if raw_content else "Comparison analysis completed with parsing issues.",
                    dominant_themes=[],
                    gaps_missing_perspectives=[],
                    reading_recommendations=[],
                    follow_up_questions=[],
                    documents_compared=len(documents),
                    document_titles=[doc.get("title", f"Document {i+1}") for i, doc in enumerate(documents)],
                    comparison_strategy="summarize_then_compare",
                    confidence_level=0.5,
                    document_summaries=document_summaries
                )
                
        except Exception as e:
            logger.error(f"Summary-based comparison failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ComparisonAnalysisResult(
                task_status="error",
                key_similarities=[],
                key_differences=[],
                conflicts_contradictions=[],
                unique_contributions={},
                overall_assessment=f"Document comparison failed during summarization phase: {str(e)}",
                dominant_themes=[],
                gaps_missing_perspectives=[],
                reading_recommendations=[],
                follow_up_questions=[],
                documents_compared=len(documents),
                document_titles=[doc.get("title", f"Document {i+1}") for i, doc in enumerate(documents)],
                comparison_strategy="summarize_then_compare",
                confidence_level=0.0
            )
    
    async def _compare_direct(
        self,
        documents: List[Dict[str, Any]],
        user_message: str,
        model_name: str
    ) -> ComparisonAnalysisResult:
        """
        Perform direct comparison when documents are small enough to fit in context
        
        Returns ComparisonAnalysisResult with structured data
        """
        try:
            # Calculate dynamic char limit per document
            char_limit = await self._calculate_document_char_limit(model_name, len(documents))
            
            # Build comparison schema
            comparison_schema = json.dumps({
                "task_status": "complete",
                "key_similarities": ["list of themes/ideas/patterns across documents"],
                "key_differences": ["list of how documents differ in approach/focus/conclusions"],
                "conflicts_contradictions": ["list of contradictory claims or conflicting information"],
                "unique_contributions": {
                    "Document Title 1": ["unique insights from this document"],
                    "Document Title 2": ["unique insights from this document"]
                },
                "overall_assessment": "synthesized collective picture from all documents",
                "dominant_themes": ["most prominent themes across collection"],
                "gaps_missing_perspectives": ["important gaps or missing perspectives"],
                "reading_recommendations": [
                    {"title": "Document Title", "reason": "why read this first"}
                ],
                "follow_up_questions": ["suggested questions for deeper analysis"],
                "documents_compared": len(documents),
                "document_titles": [doc.get("title") for doc in documents],
                "comparison_strategy": "direct",
                "confidence_level": 0.85
            }, indent=2)
            
            # Build comparison prompt with full documents
            comparison_prompt_parts = [
                f"# Multi-Document Comparative Analysis",
                "",
                f"**Analysis based on {len(documents)} documents**",
                f"**User Request**: {user_message}",
                "",
                "**STRUCTURED OUTPUT REQUIRED**: Return ONLY valid JSON matching this exact schema:",
                comparison_schema,
                "",
                "**Analysis Requirements**:",
                "1. **key_similarities**: Themes, ideas, or patterns that appear across documents",
                "2. **key_differences**: How documents differ in approach, focus, conclusions, or perspectives",
                "3. **conflicts_contradictions**: Contradictory claims, conflicting information, or competing viewpoints",
                "4. **unique_contributions**: For EACH document by title, what unique perspectives or insights it offers",
                "5. **overall_assessment**: Collective picture that emerges from all documents (2-3 paragraphs)",
                "6. **dominant_themes**: 3-5 most prominent themes across the collection",
                "7. **gaps_missing_perspectives**: What's missing or underexplored",
                "8. **reading_recommendations**: Which documents to read first and why (prioritize 2-3)",
                "9. **follow_up_questions**: 3-5 questions for deeper analysis",
                "",
                "**CRITICAL**: Use actual document titles, not generic labels like 'Document 1'!",
                "",
                "## Documents to Compare:",
                ""
            ]
            
            # Add each document with clear separation
            for i, doc in enumerate(documents, 1):
                title = doc.get("title", f"Document {i}")
                content = doc.get("content", "")
                char_count = len(content)
                estimated_tokens = char_count // 4
                
                # Apply char limit
                content_to_use = content[:char_limit]
                if len(content) > char_limit:
                    logger.info(f"Document {i}: '{title}' ({char_count:,} chars, truncated to {char_limit:,})")
                else:
                    logger.info(f"Document {i}: '{title}' ({char_count:,} chars, ~{estimated_tokens:,} tokens)")
                
                comparison_prompt_parts.append(f"### {title}")
                comparison_prompt_parts.append(f"**Size**: {char_count:,} characters")
                comparison_prompt_parts.append("")
                comparison_prompt_parts.append(content_to_use)
                comparison_prompt_parts.append("")
            
            comparison_prompt = "\n".join(comparison_prompt_parts)
            
            # Call LLM for comparison
            llm = self._get_llm(temperature=0.3, model=model_name)
            messages = [
                SystemMessage(content="You are a document comparison expert. Provide structured, detailed comparative analysis. Return ONLY valid JSON."),
                HumanMessage(content=comparison_prompt)
            ]
            
            response = await llm.ainvoke(messages)
            raw_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            json_text = raw_content.strip()
            if '```json' in json_text:
                match = re.search(r'```json\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            elif '```' in json_text:
                match = re.search(r'```\s*\n([\s\S]*?)\n```', json_text)
                if match:
                    json_text = match.group(1).strip()
            
            # Parse with Pydantic
            try:
                comparison_data = json.loads(json_text)
                comparison_result = ComparisonAnalysisResult(**comparison_data)
                
                logger.info(f"Direct comparison complete: Structured comparison analysis")
                logger.info(f"   - {len(comparison_result.key_similarities)} similarities")
                logger.info(f"   - {len(comparison_result.key_differences)} differences")
                logger.info(f"   - {len(comparison_result.conflicts_contradictions)} conflicts")
                logger.info(f"   - Confidence: {comparison_result.confidence_level:.2f}")
                
                return comparison_result
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse structured comparison result: {e}")
                logger.error(f"   Raw response: {raw_content[:500]}...")
                
                # Fallback to basic comparison result
                return ComparisonAnalysisResult(
                    task_status="complete",
                    key_similarities=[],
                    key_differences=[],
                    conflicts_contradictions=[],
                    unique_contributions={},
                    overall_assessment=raw_content[:2000] if raw_content else "Comparison analysis completed with parsing issues.",
                    dominant_themes=[],
                    gaps_missing_perspectives=[],
                    reading_recommendations=[],
                    follow_up_questions=[],
                    documents_compared=len(documents),
                    document_titles=[doc.get("title", f"Document {i+1}") for i, doc in enumerate(documents)],
                    comparison_strategy="direct",
                    confidence_level=0.5
                )
                
        except Exception as e:
            logger.error(f"Direct document comparison failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ComparisonAnalysisResult(
                task_status="error",
                key_similarities=[],
                key_differences=[],
                conflicts_contradictions=[],
                unique_contributions={},
                overall_assessment=f"Document comparison failed: {str(e)}",
                dominant_themes=[],
                gaps_missing_perspectives=[],
                reading_recommendations=[],
                follow_up_questions=[],
                documents_compared=len(documents),
                document_titles=[doc.get("title", f"Document {i+1}") for i, doc in enumerate(documents)],
                comparison_strategy="direct",
                confidence_level=0.0
            )
    
    def _unwrap_json_response(self, content: str) -> str:
        """Unwrap accidental JSON envelope like {"message": "..."} or code fences"""
        try:
            txt = content.strip()
            if '```json' in txt:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            elif '```' in txt:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            
            if txt.startswith('{') and txt.endswith('}'):
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj.get('message') or obj.get('text') or content
            
            return content
        except Exception:
            return content
    
    def _format_comparison_result(self, result: Dict[str, Any], limit_warning: str = "") -> str:
        """
        Format structured comparison result into markdown for user display
        
        Args:
            result: ComparisonAnalysisResult structured output (as dict)
            limit_warning: Optional warning about document limits
        
        Returns:
            Formatted markdown string
        """
        lines = []
        
        # Add limit warning if present
        if limit_warning:
            lines.append(limit_warning)
        
        # Header
        lines.append("# 📊 Multi-Document Comparative Analysis")
        lines.append("")
        lines.append(f"**Documents Analyzed**: {result.get('documents_compared', 0)}")
        lines.append(f"**Strategy**: {result.get('comparison_strategy', 'direct').replace('_', ' ').title()}")
        lines.append(f"**Confidence**: {int(result.get('confidence_level', 0) * 100)}%")
        lines.append("")
        
        # Overall Assessment
        if result.get("overall_assessment"):
            lines.append("## 🎯 Overall Assessment")
            lines.append("")
            lines.append(result["overall_assessment"])
            lines.append("")
        
        # Dominant Themes
        if result.get("dominant_themes"):
            lines.append("## 🔥 Dominant Themes")
            lines.append("")
            for theme in result["dominant_themes"]:
                lines.append(f"- **{theme}**")
            lines.append("")
        
        # Key Similarities
        if result.get("key_similarities"):
            lines.append("## 🤝 Key Similarities")
            lines.append("")
            for similarity in result["key_similarities"]:
                lines.append(f"- {similarity}")
            lines.append("")
        
        # Key Differences
        if result.get("key_differences"):
            lines.append("## ⚡ Key Differences")
            lines.append("")
            for difference in result["key_differences"]:
                lines.append(f"- {difference}")
            lines.append("")
        
        # Conflicts & Contradictions
        if result.get("conflicts_contradictions"):
            lines.append("## ⚠️ Conflicts & Contradictions")
            lines.append("")
            for conflict in result["conflicts_contradictions"]:
                lines.append(f"- {conflict}")
            lines.append("")
        
        # Unique Contributions by Document
        if result.get("unique_contributions"):
            lines.append("## 💎 Unique Contributions by Document")
            lines.append("")
            for doc_title, insights in result["unique_contributions"].items():
                lines.append(f"### {doc_title}")
                for insight in insights:
                    lines.append(f"- {insight}")
                lines.append("")
        
        # Gaps & Missing Perspectives
        if result.get("gaps_missing_perspectives"):
            lines.append("## 🔍 Gaps & Missing Perspectives")
            lines.append("")
            for gap in result["gaps_missing_perspectives"]:
                lines.append(f"- {gap}")
            lines.append("")
        
        # Reading Recommendations
        if result.get("reading_recommendations"):
            lines.append("## 📚 Reading Recommendations")
            lines.append("")
            for i, rec in enumerate(result["reading_recommendations"], 1):
                title = rec.get("title", "Unknown")
                reason = rec.get("reason", "No reason provided")
                lines.append(f"**{i}. {title}**")
                lines.append(f"   {reason}")
                lines.append("")
        
        # Follow-up Questions
        if result.get("follow_up_questions"):
            lines.append("## 🤔 Suggested Follow-up Questions")
            lines.append("")
            for question in result["follow_up_questions"]:
                lines.append(f"- {question}")
            lines.append("")
        
        # Document Summaries (if available from summarization strategy)
        document_summaries = result.get("document_summaries")
        if document_summaries:
            lines.append("---")
            lines.append("")
            lines.append("## 📄 Individual Document Summaries")
            lines.append("")
            for summary in document_summaries:
                # Handle both dict and DocumentSummary object
                if isinstance(summary, dict):
                    title = summary.get("title", "Unknown")
                    original_length = summary.get("original_length", 0)
                    confidence = summary.get("confidence", 0.0)
                    summary_text = summary.get("summary", "")
                    main_topics = summary.get("main_topics", [])
                    key_arguments = summary.get("key_arguments", [])
                else:
                    title = summary.title
                    original_length = summary.original_length
                    confidence = summary.confidence
                    summary_text = summary.summary
                    main_topics = summary.main_topics
                    key_arguments = summary.key_arguments
                
                lines.append(f"### {title}")
                lines.append(f"**Size**: {original_length:,} characters | **Confidence**: {int(confidence * 100)}%")
                lines.append("")
                lines.append(summary_text)
                
                if main_topics:
                    lines.append("")
                    lines.append(f"**Topics**: {', '.join(main_topics)}")
                
                if key_arguments:
                    lines.append("")
                    lines.append("**Key Arguments**:")
                    for arg in key_arguments[:3]:  # Show top 3
                        lines.append(f"- {arg}")
                
                lines.append("")
        
        return "\n".join(lines)
    
    async def _format_response_node(self, state: ContentAnalysisState) -> Dict[str, Any]:
        """Format final response for user"""
        try:
            request_type = state.get("request_type", "single_analysis")
            analysis_result = state.get("analysis_result", {})
            task_status = state.get("task_status", "complete")
            
            if task_status == "error":
                error_msg = state.get("error", "Unknown error")
                return {
                    "response": {
                        "response": f"Content analysis failed: {error_msg}",
                        "task_status": "error",
                        "agent_type": "content_analysis_agent"
                    },
                    "task_status": "error"
                }
            
            if request_type == "comparison":
                # Check if we need to show limit warning
                limit_warning = ""
                documents_compared = analysis_result.get("documents_compared", 0)
                if documents_compared >= 8:
                    limit_warning = (
                        "\n\n---\n"
                        "**Note**: This analysis is limited to the first 8 documents. "
                        "If you need to compare more, please refine your query to be more specific.\n"
                        "---\n\n"
                    )
                
                formatted_text = self._format_comparison_result(analysis_result, limit_warning)
            else:
                # Single document analysis formatting
                summary_lines = [
                    "## Opinion Piece Analysis",
                    analysis_result.get("summary", "Analysis completed"),
                    f"Verdict: {analysis_result.get('verdict', 'needs_more')} (confidence {int((analysis_result.get('confidence', 0.8)) * 100)}%)",
                    "\n### Top Recommendations",
                ]
                for rec in (analysis_result.get("recommendations", []))[:5]:
                    summary_lines.append(f"- {rec}")
                if analysis_result.get("outline_suggestions"):
                    summary_lines.append("\n### Outline Suggestions")
                    for b in (analysis_result.get("outline_suggestions", []))[:6]:
                        summary_lines.append(f"- {b}")
                
                formatted_text = "\n".join(summary_lines)
            
            # Add assistant response to messages for checkpoint persistence
            state = self._add_assistant_response_to_messages(state, formatted_text)
            
            return {
                "response": {
                    "response": formatted_text,
                    "task_status": task_status,
                    "agent_type": "content_analysis_agent",
                    "structured_response": analysis_result,
                    "timestamp": datetime.now().isoformat()
                },
                "task_status": task_status,
                "messages": state.get("messages", [])
            }
            
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return {
                "response": self._create_error_response(str(e)),
                "task_status": "error"
            }
    
    async def process(self, query: str, metadata: Dict[str, Any] = None, messages: List[Any] = None) -> Dict[str, Any]:
        """Process content analysis query using LangGraph workflow"""
        try:
            logger.info(f"Content analysis agent processing: {query[:100]}...")
            
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
            initial_state: ContentAnalysisState = {
                "query": query,
                "user_id": user_id,
                "metadata": metadata,
                "messages": conversation_messages,
                "persona": None,
                "active_editor": None,
                "request_type": "unknown",
                "documents": [],
                "document_content": "",
                "analysis_result": {},
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
                logger.error(f"Content analysis agent failed: {error_msg}")
                return self._create_error_response(error_msg)
            
            logger.info(f"Content analysis agent completed: {task_status}")
            return response
            
        except Exception as e:
            logger.error(f"Content analysis agent failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._create_error_response(str(e))


def get_content_analysis_agent() -> ContentAnalysisAgent:
    """Get singleton content analysis agent instance"""
    global _content_analysis_agent
    if _content_analysis_agent is None:
        _content_analysis_agent = ContentAnalysisAgent()
    return _content_analysis_agent


_content_analysis_agent: Optional[ContentAnalysisAgent] = None

