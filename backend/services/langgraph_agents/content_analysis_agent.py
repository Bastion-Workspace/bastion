"""
Content Analysis Agent - Roosevelt's Unified Critique Cavalry

Provides higher-level analysis for both fiction and non-fiction content beyond grammar/style.
Handles both story analysis and article/opinion piece critique in one unified agent.

**BULLY!** Now supports multi-document comparative analysis!
"""

import logging
import json
import re
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from pydantic import ValidationError

from config import settings
from .base_agent import BaseAgent
from models.article_analysis_models import ArticleAnalysisResult, get_article_analysis_structured_output
from models.agent_response_models import (
    DocumentSummary,
    ComparisonAnalysisResult,
    TaskStatus
)

logger = logging.getLogger(__name__)


class ContentAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("content_analysis_agent")  # Use content_analysis toolset with document retrieval
        logger.info("📚 BULLY! Content Analysis Agent ready to critique and compare documents with a big stick!")

    def _build_system_prompt(self) -> str:
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

    def _unwrap_json_response(self, content: str) -> str:
        """Unwrap accidental JSON envelope like {"message": "..."} or code fences"""
        try:
            txt = content.strip()
            # Handle code fences first
            if '```json' in txt:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            elif '```' in txt:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', txt)
                if m:
                    txt = m.group(1).strip()
            
            # Handle JSON envelope
            if txt.startswith('{') and txt.endswith('}'):
                obj = json.loads(txt)
                if isinstance(obj, dict):
                    return obj.get('message') or obj.get('text') or content
            
            return content
        except Exception:
            return content

    def _detect_comparison_request(self, user_message: str) -> bool:
        """
        Detect if user is asking for multi-document comparison
        
        Returns True if comparison keywords are detected
        """
        comparison_keywords = [
            "compare", "contrast", "difference", "differences", "similarity", "similarities",
            "conflict", "conflicts", "contradiction", "contradictions", "discrepancy", "discrepancies",
            "versus", "vs", "between", "across"
        ]
        
        user_message_lower = user_message.lower()
        
        for keyword in comparison_keywords:
            if keyword in user_message_lower:
                logger.info(f"🔍 COMPARISON DETECTED: Found keyword '{keyword}' in query")
                return True
        
        return False

    async def _extract_titles_from_query(self, user_message: str) -> List[str]:
        """
        Extract document titles or filenames from user query
        
        **ROOSEVELT'S TITLE DETECTION**: Find specific document references!
        
        Returns list of potential titles/filenames
        """
        import re
        
        # Common patterns for title/filename references
        patterns = [
            r"'([^']+\.(?:pdf|epub|docx|txt|md))'",  # 'filename.pdf'
            r'"([^"]+\.(?:pdf|epub|docx|txt|md))"',  # "filename.pdf"
            r"file\s+(?:named|called)\s+['\"]?([^'\"]+)['\"]?",  # file named X
            r"document\s+['\"]?([^'\"]+)['\"]?",  # document "X"
            r"['\"]([^'\"]+)['\"]",  # Generic quoted strings
        ]
        
        titles = []
        for pattern in patterns:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            titles.extend(matches)
        
        # Also check for bare filenames (with extensions)
        bare_files = re.findall(r'\b(\w+\.(?:pdf|epub|docx|txt|md))\b', user_message, re.IGNORECASE)
        titles.extend(bare_files)
        
        # Remove duplicates
        titles = list(set(titles))
        
        if titles:
            logger.info(f"🔍 TITLE DETECTION: Found {len(titles)} potential titles: {titles}")
        
        return titles

    async def _extract_author_from_query(self, user_message: str) -> str:
        """
        Extract author name from user query
        
        **ROOSEVELT'S AUTHOR DETECTION**: Find author references!
        
        Returns author name if found, empty string otherwise
        """
        import re
        
        # Patterns for author references
        patterns = [
            r"by\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",  # by Tom Clancy
            r"author\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",  # author Tom Clancy
            r"written\s+by\s+['\"]?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)['\"]?",  # written by Tom Clancy
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_message)
            if match:
                author = match.group(1).strip()
                logger.info(f"🔍 AUTHOR DETECTION: Found author: {author}")
                return author
        
        return ""

    async def _extract_tags_from_query(self, user_message: str, user_id: str) -> List[str]:
        """
        Extract document tags from user query using tag detection service
        
        **ROOSEVELT'S GLOBAL + USER SCOPE**: Gets tags from BOTH global and user-specific documents!
        
        Returns list of matched tags
        """
        try:
            from services.langgraph_tools.tag_detection_service import get_tag_detection_service
            from repositories.document_repository import DocumentRepository
            
            # Get available tags from database (includes both global and user-specific)
            doc_repo = DocumentRepository()
            await doc_repo.initialize()  # ← CRITICAL: Must initialize first!
            available_tags = await doc_repo.get_all_tags()  # ← No user_id parameter!
            available_categories = await doc_repo.get_all_categories()  # ← No user_id parameter!
            
            logger.info(f"🔍 Available tags (global + user): {available_tags}")
            logger.info(f"🔍 Available categories (global + user): {available_categories}")
            
            # Use tag detection service
            tag_service = get_tag_detection_service()
            detection_result = await tag_service.detect_and_match_filters(
                user_message,
                available_tags,
                available_categories
            )
            
            logger.info(f"🔍 TAG DETECTION: {detection_result}")
            
            matched_tags = detection_result.get("filter_tags", [])
            if matched_tags:
                logger.info(f"✅ Extracted tags: {matched_tags}")
            
            return matched_tags
            
        except Exception as e:
            logger.error(f"❌ Tag extraction failed: {e}")
            return []

    async def _retrieve_documents_by_tags(self, tags: List[str], user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieve documents by tags using direct PostgreSQL query
        
        **ROOSEVELT'S TAG-BASED RETRIEVAL**: Query PostgreSQL directly to get ALL documents
        with the specified tags, not just semantically similar ones!
        
        Returns list of document dictionaries with content and metadata
        """
        try:
            logger.info(f"🔍 DOCUMENT RETRIEVAL: Querying PostgreSQL for documents with tags={tags}, user_id={user_id}")
            
            # **ROOSEVELT FIX**: Query PostgreSQL directly for ALL documents with tags
            from repositories.document_repository import DocumentRepository
            from models.api_models import DocumentFilterRequest
            
            doc_repo = DocumentRepository()
            await doc_repo.initialize()
            
            # **ROOSEVELT'S 8-DOCUMENT LIMIT**: Keep comparisons manageable!
            max_docs = 8
            
            # Create filter request to get documents by tags
            filter_request = DocumentFilterRequest(
                tags=tags,
                limit=max_docs,  # Hard limit at 8 documents
                skip=0,
                sort_by="upload_date",
                sort_order="desc"
            )
            
            # Get documents from PostgreSQL
            db_documents, total_count = await doc_repo.filter_documents(filter_request)
            logger.info(f"📚 Found {len(db_documents)} documents in PostgreSQL with tags {tags} (total: {total_count})")
            
            # Warn if there are more documents than our limit
            if total_count > max_docs:
                logger.warning(f"⚠️ Found {total_count} documents with tags {tags}, showing first {max_docs}. User should be more specific.")

            # Extract document IDs
            document_ids = [str(doc.document_id) for doc in db_documents]
            logger.info(f"📚 Document IDs to retrieve: {document_ids[:5]}{'...' if len(document_ids) > 5 else ''}")
            
            # Retrieve full document content for each document_id
            documents = []
            
            # Initialize UnifiedSearchTools to access get_document method
            from services.langgraph_tools.unified_search_tools import UnifiedSearchTools
            search_tools = UnifiedSearchTools()
            
            for doc_id in document_ids:
                try:
                    # Use get_document method to retrieve full content
                    full_doc = await search_tools.get_document(document_id=doc_id, user_id=user_id)
                    
                    if full_doc and isinstance(full_doc, dict) and full_doc.get("success"):
                        # Extract metadata from the document
                        metadata = full_doc.get("metadata", {})
                        title = metadata.get("title") or full_doc.get("title") or full_doc.get("filename") or doc_id
                        
                        documents.append({
                            "document_id": doc_id,
                            "content": full_doc.get("content", ""),
                            "metadata": metadata,
                            "title": title
                        })
                        
                        logger.info(f"  📄 Retrieved full document: {title} ({len(full_doc.get('content', ''))} chars)")
                    else:
                        logger.warning(f"⚠️ Failed to retrieve document {doc_id}: {full_doc.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to retrieve full document {doc_id}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
            
            logger.info(f"✅ Successfully retrieved {len(documents)} full documents for comparison")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return []

    async def _retrieve_documents_by_titles(self, titles: List[str], user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents by title or filename
        
        **ROOSEVELT'S TITLE SEARCH**: Fuzzy matching to find documents by name!
        
        Returns list of document dictionaries with content
        """
        try:
            logger.info(f"🔍 DOCUMENT RETRIEVAL: Searching for documents with titles={titles}, user_id={user_id}")
            
            from repositories.document_repository import DocumentRepository
            from services.database_manager.database_helpers import fetch_all
            
            doc_repo = DocumentRepository()
            await doc_repo.initialize()
            
            # Search for documents matching any of the titles (fuzzy match)
            documents = []
            
            for title in titles:
                # Try exact match first, then fuzzy match
                # Search in filename, title fields
                query = """
                    SELECT id, filename, title FROM documents 
                    WHERE (LOWER(filename) LIKE $1 OR LOWER(title) LIKE $1)
                    AND processing_status = 'completed'
                    LIMIT 10
                """
                
                # Try with and without extension
                title_pattern = f"%{title.lower()}%"
                
                results = await fetch_all(query, title_pattern)
                
                if results:
                    logger.info(f"📚 Found {len(results)} documents matching title: {title}")
                    
                    # Retrieve full content for each
                    from services.langgraph_tools.unified_search_tools import UnifiedSearchTools
                    search_tools = UnifiedSearchTools()
                    
                    for row in results:
                        doc_id = str(row['id'])
                        try:
                            full_doc = await search_tools.get_document(document_id=doc_id, user_id=user_id)
                            
                            if full_doc and isinstance(full_doc, dict) and full_doc.get("success"):
                                metadata = full_doc.get("metadata", {})
                                doc_title = metadata.get("title") or full_doc.get("title") or full_doc.get("filename") or doc_id
                                
                                documents.append({
                                    "document_id": doc_id,
                                    "content": full_doc.get("content", ""),
                                    "metadata": metadata,
                                    "title": doc_title
                                })
                                
                                logger.info(f"  📄 Retrieved document: {doc_title} ({len(full_doc.get('content', ''))} chars)")
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to retrieve document {doc_id}: {e}")
                            continue
                else:
                    logger.warning(f"⚠️ No documents found matching title: {title}")
            
            logger.info(f"✅ Successfully retrieved {len(documents)} documents by title")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Document retrieval by title failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def _retrieve_documents_by_author(self, author: str, user_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieve documents by author name
        
        **ROOSEVELT'S AUTHOR SEARCH**: Find all books/documents by an author!
        
        Returns list of document dictionaries with content
        """
        try:
            logger.info(f"🔍 DOCUMENT RETRIEVAL: Searching for documents by author={author}, user_id={user_id}")
            
            from repositories.document_repository import DocumentRepository
            from models.api_models import DocumentFilterRequest
            
            doc_repo = DocumentRepository()
            await doc_repo.initialize()
            
            # **ROOSEVELT'S 8-DOCUMENT LIMIT**: Keep comparisons manageable!
            max_docs = 8
            
            # Create filter request to get documents by author
            filter_request = DocumentFilterRequest(
                author=author,
                limit=max_docs,  # Hard limit at 8 documents
                skip=0,
                sort_by="upload_date",
                sort_order="desc"
            )
            
            # Get documents from PostgreSQL
            db_documents, total_count = await doc_repo.filter_documents(filter_request)
            logger.info(f"📚 Found {len(db_documents)} documents by author '{author}' (total: {total_count})")
            
            # Warn if there are more documents than our limit
            if total_count > max_docs:
                logger.warning(f"⚠️ Found {total_count} documents by author '{author}', showing first {max_docs}. User should be more specific.")

            # Extract document IDs
            document_ids = [str(doc.document_id) for doc in db_documents]
            
            # Retrieve full document content
            documents = []
            
            from services.langgraph_tools.unified_search_tools import UnifiedSearchTools
            search_tools = UnifiedSearchTools()
            
            for doc_id in document_ids:
                try:
                    full_doc = await search_tools.get_document(document_id=doc_id, user_id=user_id)
                    
                    if full_doc and isinstance(full_doc, dict) and full_doc.get("success"):
                        metadata = full_doc.get("metadata", {})
                        title = metadata.get("title") or full_doc.get("title") or full_doc.get("filename") or doc_id
                        
                        documents.append({
                            "document_id": doc_id,
                            "content": full_doc.get("content", ""),
                            "metadata": metadata,
                            "title": title
                        })
                        
                        logger.info(f"  📄 Retrieved document by {author}: {title} ({len(full_doc.get('content', ''))} chars)")
                    else:
                        logger.warning(f"⚠️ Failed to retrieve document {doc_id}: {full_doc.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to retrieve document {doc_id}: {e}")
                    continue
            
            logger.info(f"✅ Successfully retrieved {len(documents)} documents by author '{author}'")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Document retrieval by author failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    async def _get_model_context_window(self, model_name: str) -> int:
        """
        Get model context window size in tokens
        
        **ROOSEVELT'S CENTRALIZED INTELLIGENCE**: Use chat_service for authoritative model data!
        
        Returns context window size in tokens
        """
        try:
            # Use centralized chat_service method for authoritative model data
            chat_service = await self._get_chat_service()
            context_size = await chat_service.get_model_context_window(model_name)
            return context_size
        except Exception as e:
            logger.warning(f"⚠️ Failed to get model context from chat_service: {e}, using conservative fallback")
            # Conservative fallback
            return 8000
    
    async def _calculate_document_char_limit(self, model_name: str, num_documents: int = 1) -> int:
        """
        Calculate dynamic character limit per document based on model context window
        
        **ROOSEVELT'S DYNAMIC LIMITS**: Use the full capacity of your cavalry!
        
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
            f"📐 MODEL CAPACITY: {model_name} has {context_window:,} token context window\n"
            f"   Available for content: {available_tokens:,} tokens\n"
            f"   Per document ({num_documents} docs): {tokens_per_doc:,} tokens (~{chars_per_doc:,} chars)"
        )
        
        return chars_per_doc

    async def _compare_multiple_documents(
        self,
        documents: List[Dict[str, Any]],
        user_message: str,
        chat_service: Any,
        model_name: str
    ) -> ComparisonAnalysisResult:
        """
        Perform comparative analysis across multiple documents using intelligent summarization strategy

        **ROOSEVELT'S SMART COMPARISON STRATEGY**:
        1. First summarize each document individually (keeps each under token limits)
        2. Then compare the summaries (much smaller total token count)
        3. If needed, dive deeper into specific sections

        Returns ComparisonAnalysisResult with structured data
        """
        try:
            total_chars = sum(len(doc.get("content", "")) for doc in documents)
            total_estimated_tokens = total_chars // 4

            logger.info(f"📊 TOTAL CONTENT: {total_chars:,} characters (~{total_estimated_tokens:,} tokens)")

            # **ROOSEVELT'S TOKEN-AWARE STRATEGY**: If over 400K tokens, use summarization approach
            # (Conservative threshold to account for OpenRouter limits vs base model limits)
            if total_estimated_tokens > 400000:
                logger.info("🤖 LARGE COMPARISON: Using intelligent summarization strategy")
                return await self._compare_with_summaries(documents, user_message, chat_service, model_name)
            else:
                logger.info("📝 SMALL COMPARISON: Direct comparison feasible")
                return await self._compare_direct(documents, user_message, chat_service, model_name)

        except Exception as e:
            logger.error(f"❌ Document comparison failed: {e}")
            # Return structured error result
            return ComparisonAnalysisResult(
                task_status=TaskStatus.ERROR,
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
        chat_service: Any,
        model_name: str,
        char_limit: int
    ) -> DocumentSummary:
        """
        Summarize a single document with structured output
        
        **ROOSEVELT'S PARALLEL PROCESSING**: Each cavalry unit operates independently!
        
        Returns DocumentSummary with structured data
        """
        title = doc.get("title", f"Document {doc_index}")
        content = doc.get("content", "")
        char_count = len(content)
        
        logger.info(f"📝 Summarizing Document {doc_index}: '{title}' ({char_count:,} chars, limit: {char_limit:,})")
        
        # Use dynamic char limit based on model capacity
        content_to_summarize = content[:char_limit]
        if len(content) > char_limit:
            logger.info(f"   ⚠️ Document truncated from {char_count:,} to {char_limit:,} chars")
        
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
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert document summarizer. Create structured, comprehensive summaries with specific details. Return ONLY valid JSON."},
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.2,
                max_tokens=settings.DEFAULT_MAX_TOKENS  # **CENTRALIZED**: Use global max_tokens setting
            )
            
            raw_content = response.choices[0].message.content or "{}"
            
            # **ROOSEVELT'S TRUNCATION DETECTION**: Check if response was cut off
            finish_reason = response.choices[0].finish_reason
            if finish_reason == "length":
                logger.warning(f"⚠️ TRUNCATION: Summary for {title} was cut off (hit max_tokens limit)")
                logger.warning(f"   Consider increasing max_tokens or reducing input size")
            
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
                logger.info(f"✅ Summarized {title} - {len(summary_obj.summary)} chars, {len(summary_obj.key_arguments)} arguments")
                return summary_obj
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"⚠️ Failed to parse structured summary for {title}: {e}")
                logger.warning(f"   Raw response preview: {raw_content[:200]}...")
                
                # **ROOSEVELT'S CLEAN FALLBACK**: Extract summary text from malformed JSON
                fallback_summary = f"Document contains {char_count:,} characters."
                
                # Try to extract summary field from malformed JSON
                try:
                    if '"summary"' in raw_content:
                        # Find summary field value
                        summary_match = re.search(r'"summary":\s*"([^"]*(?:"[^"]*)*)"', raw_content, re.DOTALL)
                        if summary_match:
                            extracted = summary_match.group(1)
                            # Clean up escaped characters
                            extracted = extracted.replace('\\"', '"').replace('\\n', ' ')
                            if len(extracted) > 100:  # Only use if substantial
                                fallback_summary = extracted[:1000]  # Limit to 1000 chars
                                logger.info(f"✅ Extracted summary from malformed JSON: {len(extracted)} chars")
                    
                    # If no summary field found, try to use the whole response (cleaned)
                    if fallback_summary == f"Document contains {char_count:,} characters." and raw_content:
                        # Remove JSON syntax artifacts
                        cleaned = raw_content.replace('{', '').replace('}', '').replace('"', '')
                        cleaned = re.sub(r'\b(title|summary|main_topics|key_arguments|key_data_points|author_perspective|unique_insights|document_id|original_length|confidence)\b:\s*', '', cleaned)
                        cleaned = cleaned.strip()
                        if len(cleaned) > 100:
                            fallback_summary = cleaned[:1000]
                            logger.info(f"✅ Cleaned malformed JSON to text: {len(cleaned)} chars")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Fallback cleanup failed: {cleanup_error}")
                
                # Fallback to basic summary with cleaned text
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
            logger.error(f"❌ Failed to summarize {title}: {e}")
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
        chat_service: Any,
        model_name: str
    ) -> ComparisonAnalysisResult:
        """
        Compare documents using a two-phase summarization strategy with PARALLEL processing

        **ROOSEVELT'S PARALLEL CAVALRY CHARGE**:
        Phase 1: Summarize ALL documents in PARALLEL using asyncio.gather()
        Phase 2: Compare the summaries with structured output

        Returns ComparisonAnalysisResult with structured data
        """
        try:
            logger.info("📋 PHASE 1: Summarizing documents IN PARALLEL...")
            
            # Calculate dynamic char limit per document
            char_limit = await self._calculate_document_char_limit(model_name, len(documents))

            # **ROOSEVELT'S PARALLEL EXECUTION**: Summarize all documents simultaneously!
            summarization_tasks = [
                self._summarize_single_document(doc, i+1, chat_service, model_name, char_limit)
                for i, doc in enumerate(documents)
            ]
            
            # Execute all summarizations in parallel
            document_summaries: List[DocumentSummary] = await asyncio.gather(*summarization_tasks)
            
            logger.info(f"✅ PHASE 1 COMPLETE: Summarized {len(document_summaries)} documents in parallel")

            # Phase 2: Compare the summaries with structured output
            logger.info("📋 PHASE 2: Comparing document summaries with structured output...")

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

            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert comparative analyst. Provide structured, detailed comparative analysis. Return ONLY valid JSON."},
                    {"role": "user", "content": comparison_prompt},
                ],
                temperature=0.3,
                max_tokens=settings.DEFAULT_MAX_TOKENS  # **CENTRALIZED**: Use global max_tokens setting
            )

            raw_content = response.choices[0].message.content or "{}"
            
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
                
                logger.info(f"✅ PHASE 2 COMPLETE: Structured comparison analysis completed")
                logger.info(f"   - {len(comparison_result.key_similarities)} similarities")
                logger.info(f"   - {len(comparison_result.key_differences)} differences")
                logger.info(f"   - {len(comparison_result.conflicts_contradictions)} conflicts")
                logger.info(f"   - Confidence: {comparison_result.confidence_level:.2f}")
                
                return comparison_result
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"❌ Failed to parse structured comparison result: {e}")
                logger.error(f"   Raw response: {raw_content[:500]}...")
                
                # Fallback to basic comparison result
                return ComparisonAnalysisResult(
                    task_status=TaskStatus.COMPLETE,
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
            logger.error(f"❌ Summary-based comparison failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ComparisonAnalysisResult(
                task_status=TaskStatus.ERROR,
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
        chat_service: Any,
        model_name: str
    ) -> ComparisonAnalysisResult:
        """
        Perform direct comparison when documents are small enough to fit in context
        
        **ROOSEVELT'S DIRECT COMPARISON**: Full documents with structured output!
        
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
                    logger.info(f"📝 Document {i}: '{title}' ({char_count:,} chars, truncated to {char_limit:,})")
                else:
                    logger.info(f"📝 Document {i}: '{title}' ({char_count:,} chars, ~{estimated_tokens:,} tokens)")

                comparison_prompt_parts.append(f"### {title}")
                comparison_prompt_parts.append(f"**Size**: {char_count:,} characters")
                comparison_prompt_parts.append("")
                comparison_prompt_parts.append(content_to_use)
                comparison_prompt_parts.append("")

            comparison_prompt = "\n".join(comparison_prompt_parts)

            # Call LLM for comparison
            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a document comparison expert. Provide structured, detailed comparative analysis. Return ONLY valid JSON."},
                    {"role": "user", "content": comparison_prompt},
                ],
                temperature=0.3,
                max_tokens=settings.DEFAULT_MAX_TOKENS  # **CENTRALIZED**: Use global max_tokens setting
            )

            raw_content = response.choices[0].message.content or "{}"
            
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
                
                logger.info(f"✅ DIRECT COMPARISON COMPLETE: Structured comparison analysis")
                logger.info(f"   - {len(comparison_result.key_similarities)} similarities")
                logger.info(f"   - {len(comparison_result.key_differences)} differences")
                logger.info(f"   - {len(comparison_result.conflicts_contradictions)} conflicts")
                logger.info(f"   - Confidence: {comparison_result.confidence_level:.2f}")
                
                return comparison_result
                
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"❌ Failed to parse structured comparison result: {e}")
                logger.error(f"   Raw response: {raw_content[:500]}...")
                
                # Fallback to basic comparison result
                return ComparisonAnalysisResult(
                    task_status=TaskStatus.COMPLETE,
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
            logger.error(f"❌ Direct document comparison failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ComparisonAnalysisResult(
                task_status=TaskStatus.ERROR,
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

    def _format_comparison_result(self, result: ComparisonAnalysisResult, limit_warning: str = "") -> str:
        """
        Format structured comparison result into beautiful markdown for user display
        
        **ROOSEVELT'S PRESENTATION DOCTRINE**: Present structured data with style!
        
        Args:
            result: ComparisonAnalysisResult structured output
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
        lines.append(f"**Documents Analyzed**: {result.documents_compared}")
        lines.append(f"**Strategy**: {result.comparison_strategy.replace('_', ' ').title()}")
        lines.append(f"**Confidence**: {int(result.confidence_level * 100)}%")
        lines.append("")
        
        # Overall Assessment
        lines.append("## 🎯 Overall Assessment")
        lines.append("")
        lines.append(result.overall_assessment)
        lines.append("")
        
        # Dominant Themes
        if result.dominant_themes:
            lines.append("## 🔥 Dominant Themes")
            lines.append("")
            for theme in result.dominant_themes:
                lines.append(f"- **{theme}**")
            lines.append("")
        
        # Key Similarities
        if result.key_similarities:
            lines.append("## 🤝 Key Similarities")
            lines.append("")
            for similarity in result.key_similarities:
                lines.append(f"- {similarity}")
            lines.append("")
        
        # Key Differences
        if result.key_differences:
            lines.append("## ⚡ Key Differences")
            lines.append("")
            for difference in result.key_differences:
                lines.append(f"- {difference}")
            lines.append("")
        
        # Conflicts & Contradictions
        if result.conflicts_contradictions:
            lines.append("## ⚠️ Conflicts & Contradictions")
            lines.append("")
            for conflict in result.conflicts_contradictions:
                lines.append(f"- {conflict}")
            lines.append("")
        
        # Unique Contributions by Document
        if result.unique_contributions:
            lines.append("## 💎 Unique Contributions by Document")
            lines.append("")
            for doc_title, insights in result.unique_contributions.items():
                lines.append(f"### {doc_title}")
                for insight in insights:
                    lines.append(f"- {insight}")
                lines.append("")
        
        # Gaps & Missing Perspectives
        if result.gaps_missing_perspectives:
            lines.append("## 🔍 Gaps & Missing Perspectives")
            lines.append("")
            for gap in result.gaps_missing_perspectives:
                lines.append(f"- {gap}")
            lines.append("")
        
        # Reading Recommendations
        if result.reading_recommendations:
            lines.append("## 📚 Reading Recommendations")
            lines.append("")
            for i, rec in enumerate(result.reading_recommendations, 1):
                title = rec.get("title", "Unknown")
                reason = rec.get("reason", "No reason provided")
                lines.append(f"**{i}. {title}**")
                lines.append(f"   {reason}")
                lines.append("")
        
        # Follow-up Questions
        if result.follow_up_questions:
            lines.append("## 🤔 Suggested Follow-up Questions")
            lines.append("")
            for question in result.follow_up_questions:
                lines.append(f"- {question}")
            lines.append("")
        
        # Document Summaries (if available from summarization strategy)
        if result.document_summaries:
            lines.append("---")
            lines.append("")
            lines.append("## 📄 Individual Document Summaries")
            lines.append("")
            for summary in result.document_summaries:
                lines.append(f"### {summary.title}")
                lines.append(f"**Size**: {summary.original_length:,} characters | **Confidence**: {int(summary.confidence * 100)}%")
                lines.append("")
                lines.append(summary.summary)
                
                if summary.main_topics:
                    lines.append("")
                    lines.append(f"**Topics**: {', '.join(summary.main_topics)}")
                
                if summary.key_arguments:
                    lines.append("")
                    lines.append("**Key Arguments**:")
                    for arg in summary.key_arguments[:3]:  # Show top 3
                        lines.append(f"- {arg}")
                
                lines.append("")
        
        return "\n".join(lines)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Extract user message
            user_message = ""
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if hasattr(msg, "type") and msg.type == "human":
                    user_message = str(msg.content)
                    break
            
            logger.info(f"📚 CONTENT ANALYSIS: Processing request: {user_message[:100]}...")
            
            # Check if this is a multi-document comparison request
            is_comparison = self._detect_comparison_request(user_message)
            
            if is_comparison:
                logger.info("🔍 MULTI-DOCUMENT COMPARISON MODE DETECTED")
                
                # Extract user ID
                shared_memory = state.get("shared_memory", {}) or {}
                user_id = shared_memory.get("user_id") or state.get("user_id")
                
                if not user_id:
                    logger.error("❌ No user_id found in state for document retrieval")
                    state["latest_response"] = "Unable to retrieve documents: user authentication required."
                    state["is_complete"] = True
                    return state
                
                # **ROOSEVELT'S SMART RETRIEVAL**: Try titles first, then author, then tags!
                documents = []
                retrieval_method = ""
                
                # 1. Check for specific titles/filenames
                titles = await self._extract_titles_from_query(user_message)
                if titles:
                    logger.info(f"🔍 Attempting retrieval by titles: {titles}")
                    documents = await self._retrieve_documents_by_titles(titles, user_id)
                    retrieval_method = f"titles: {', '.join(titles)}"
                
                # 2. Check for author
                if not documents:
                    author = await self._extract_author_from_query(user_message)
                    if author:
                        logger.info(f"🔍 Attempting retrieval by author: {author}")
                        documents = await self._retrieve_documents_by_author(author, user_id)
                        retrieval_method = f"author: {author}"
                
                # 3. Fall back to tags
                if not documents:
                    tags = await self._extract_tags_from_query(user_message, user_id)
                    if tags:
                        logger.info(f"🔍 Attempting retrieval by tags: {tags}")
                        documents = await self._retrieve_documents_by_tags(tags, user_id)
                        retrieval_method = f"tags: {', '.join(tags)}"
                
                # If still no documents found
                if not documents:
                    logger.warning("⚠️ No documents found by any method")
                    state["latest_response"] = (
                        "I couldn't identify which documents to analyze. Please specify:\n"
                        "- Specific titles (e.g., 'summarize ebberss504.pdf')\n"
                        "- Author names (e.g., 'compare books by Tom Clancy')\n"
                        "- Document tags (e.g., 'compare our worldcom documents')"
                    )
                    state["is_complete"] = True
                    return state
                
                logger.info(f"✅ Found {len(documents)} documents to compare")
                
                # Check if we hit the document limit and need to warn the user
                limit_warning = ""
                if len(documents) >= 8:
                    # Check if there might be more documents (this is approximate)
                    limit_warning = (
                        "\n\n---\n"
                        "**Note**: This analysis is limited to the first 8 documents. "
                        "If you need to compare more, please refine your query to be more specific.\n"
                        "---\n\n"
                    )
                
                # Perform comparison
                chat_service = await self._get_chat_service()
                model_name = await self._get_model_name()
                
                comparison_result: ComparisonAnalysisResult = await self._compare_multiple_documents(
                    documents, user_message, chat_service, model_name
                )
                
                # **ROOSEVELT'S STRUCTURED FORMATTING**: Present structured data in a beautiful way!
                formatted_response = self._format_comparison_result(comparison_result, limit_warning)
                
                # Build response with structured data
                state["agent_results"] = {
                    "structured_response": comparison_result.dict(),
                    "timestamp": datetime.now().isoformat(),
                    "mode": "multi_document_comparison",
                }
                state["latest_response"] = formatted_response
                
                try:
                    from langchain_core.messages import AIMessage
                    if formatted_response and formatted_response.strip():
                        state.setdefault("messages", []).append(AIMessage(content=formatted_response))
                except Exception:
                    pass
                
                state["is_complete"] = True
                return state
            
            # SINGLE DOCUMENT ANALYSIS MODE (NON-FICTION ONLY)
            # **ROOSEVELT'S STRICT ROUTING**: Fiction documents should NEVER reach this agent!
            # Fiction → fiction_editing_agent, story_analysis_agent, or proofreading_agent
            
            shared_memory = state.get("shared_memory", {}) or {}
            user_id = shared_memory.get("user_id") or state.get("user_id")
            active_editor = shared_memory.get("active_editor", {}) or {}

            # **ROOSEVELT'S SMART SINGLE-DOC ANALYSIS**: Check for specific titles first, then active editor
            manuscript = ""
            filename = ""
            doc_type = ""
            
            # 1. Check if user specified a document by title
            titles = await self._extract_titles_from_query(user_message)
            if titles and user_id:
                logger.info(f"🔍 Single document analysis by title: {titles}")
                documents = await self._retrieve_documents_by_titles(titles, user_id)
                if documents:
                    # Analyze the first document found
                    doc = documents[0]
                    manuscript = doc.get("content", "")
                    filename = doc.get("title", "document")
                    doc_type = doc.get("metadata", {}).get("doc_type", "non-fiction")
                    logger.info(f"✅ Retrieved document for analysis: {filename} ({len(manuscript)} chars)")
            
            # 2. Fall back to active editor if no title specified
            if not manuscript:
                manuscript = active_editor.get("content", "") or ""
                filename = active_editor.get("filename") or "document.md"
                frontmatter = active_editor.get("frontmatter", {}) or {}
                doc_type = str((frontmatter.get('type') or '')).strip().lower()

            # **CRITICAL CHECK**: Fiction documents should never reach content_analysis_agent
            if doc_type == "fiction":
                logger.error(f"❌ ROUTING ERROR: Fiction document reached content_analysis_agent! This should go to story_analysis_agent!")
                state["latest_response"] = "This appears to be a fiction document. Please use the Story Analysis Agent for fiction manuscript analysis."
                state["is_complete"] = True
                return state
            
            # If no document found at all
            if not manuscript:
                logger.warning("⚠️ No document found for analysis")
                state["latest_response"] = (
                    "I couldn't find a document to analyze. Please either:\n"
                    "- Specify a document title (e.g., 'summarize ebberss504.pdf')\n"
                    "- Open a document in the editor\n"
                    "- Use a comparison query (e.g., 'compare our worldcom documents')"
                )
                state["is_complete"] = True
                return state

            # NON-FICTION ANALYSIS (article/op-ed/etc)
            chat_service = await self._get_chat_service()
            model_name = await self._get_model_name()
            
            system_prompt = self._build_system_prompt()
            user_prompt = (
                "=== ARTICLE CONTEXT (frontmatter stripped) ===\n"
                f"Filename: {filename}\nType: {doc_type}\n\n"
                f"{manuscript}\n\n"
                "TASKS:\n"
                "1) Assess overall strength as an opinion piece.\n"
                "2) Identify missing arguments, weak sections, and needed additions.\n"
                "3) Provide concrete recommendations and outline bullets to strengthen it.\n"
                "4) Set verdict: 'solid' or 'needs_more'.\n"
            )

            response = await chat_service.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content or "{}"
            # Unwrap any JSON formatting issues
            content = self._unwrap_json_response(content)
            
            text = content.strip()
            if '```json' in text:
                m = re.search(r'```json\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            elif '```' in text:
                m = re.search(r'```\s*\n([\s\S]*?)\n```', text)
                if m:
                    text = m.group(1).strip()
            try:
                data = json.loads(text)
            except Exception:
                data = {
                    "task_status": "complete",
                    "summary": content[:800],
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
                    "metadata": {},
                }

            try:
                result = ArticleAnalysisResult(**data)
            except Exception as e:
                logger.error(f"❌ Content analysis failed: {e}")
                # Create fallback result with proper structure
                result = ArticleAnalysisResult(
                    task_status="complete",
                    summary=f"Analysis completed with parsing issues. Raw response: {content[:500]}...",
                    thesis_clarity="Unable to assess due to parsing error",
                    structure_coherence="Unable to assess due to parsing error", 
                    evidence_quality="Unable to assess due to parsing error",
                    counterarguments="Unable to assess due to parsing error",
                    tone_audience_fit="Unable to assess due to parsing error",
                    strengths=[],
                    weaknesses=[],
                    recommendations=["Please re-run the analysis for a proper assessment"],
                    missing_elements=[],
                    outline_suggestions=[],
                    verdict="needs_more",
                    confidence=0.3,
                    metadata={"parse_error": str(e), "raw_data": data}
                )
        
            # Present concise analysis summary in chat; store full structured
            summary_lines = [
                "## Opinion Piece Analysis",
                result.summary,
                f"Verdict: {result.verdict} (confidence {int((result.confidence or 0.8)*100)}%)",
                "\n### Top Recommendations",
            ]
            for rec in (result.recommendations or [])[:5]:
                summary_lines.append(f"- {rec}")
            if result.outline_suggestions:
                summary_lines.append("\n### Outline Suggestions")
                for b in (result.outline_suggestions or [])[:6]:
                    summary_lines.append(f"- {b}")

            state["agent_results"] = {
                "structured_response": result.dict(),
                "timestamp": datetime.now().isoformat(),
                "mode": "analysis",
            }
            state["latest_response"] = "\n".join(summary_lines)
            try:
                from langchain_core.messages import AIMessage
                if state["latest_response"] and state["latest_response"].strip():
                    state.setdefault("messages", []).append(AIMessage(content=state["latest_response"]))
            except Exception:
                pass
            state["is_complete"] = True
            return state
        except Exception as e:
            logger.error(f"❌ Content analysis failed: {e}")
            state["latest_response"] = "Content analysis failed."
            state["is_complete"] = True
            return state
