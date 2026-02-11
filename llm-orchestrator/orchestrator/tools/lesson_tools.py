"""
Lesson Tools - Functions for finding and managing learning lessons
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def resolve_lesson_images(
    lesson_data: Dict[str, Any],
    lesson_document_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    Resolve relative image paths in lesson to absolute URLs
    
    Uses the load_file_by_path utility to resolve relative image references
    from lesson files. Converts relative paths like "./images/puffin.jpg" 
    to absolute document URLs.
    
    Args:
        lesson_data: Lesson JSON data with questions containing image_path fields
        lesson_document_id: Document ID of the lesson file (for base path)
        user_id: User ID for access control
        
    Returns:
        Updated lesson_data with image_url fields added to questions
    """
    try:
        from orchestrator.tools.reference_file_loader import load_file_by_path
        from orchestrator.tools.document_tools import get_document_content_tool
        
        # Get lesson document to extract canonical_path
        lesson_content = await get_document_content_tool(lesson_document_id, user_id)
        if not lesson_content or lesson_content.startswith("Error") or lesson_content.startswith("Document not found"):
            logger.warning(f"⚠️ Could not load lesson document {lesson_document_id} for image resolution")
            return lesson_data
        
        # Build active_editor context for path resolution
        active_editor = {
            "document_id": lesson_document_id,
            "content": lesson_content
        }
        
        # Try to get canonical_path from backend
        try:
            from orchestrator.backend_tool_client import get_backend_tool_client
            client = await get_backend_tool_client()
            doc_info = await client.get_document_metadata(lesson_document_id, user_id)
            if doc_info and "canonical_path" in doc_info:
                active_editor["canonical_path"] = doc_info["canonical_path"]
                logger.info(f"📄 Using canonical_path for lesson: {doc_info['canonical_path']}")
        except Exception as e:
            logger.debug(f"Could not get canonical_path from backend: {e}")
        
        # Resolve images for each question
        questions = lesson_data.get("questions", [])
        for question in questions:
            if question.get("type") == "image" and question.get("image_path"):
                try:
                    image_doc = await load_file_by_path(
                        ref_path=question["image_path"],
                        user_id=user_id,
                        active_editor=active_editor
                    )
                    
                    if image_doc:
                        question["image_url"] = f"/api/documents/{image_doc['document_id']}/content"
                        question["image_metadata"] = {
                            "document_id": image_doc["document_id"],
                            "filename": image_doc.get("filename"),
                            "title": image_doc.get("title")
                        }
                        logger.info(f"✅ Resolved image: {question['image_path']} -> {image_doc['document_id']}")
                    else:
                        # Fallback to text question
                        logger.warning(f"⚠️ Image not found: {question['image_path']}, converting to text question")
                        question["type"] = "text"
                        question["image_path_error"] = f"Image not found: {question['image_path']}"
                        
                except Exception as e:
                    logger.error(f"❌ Error resolving image {question.get('image_path')}: {e}")
                    question["type"] = "text"
                    question["image_path_error"] = str(e)
        
        return lesson_data
        
    except Exception as e:
        logger.error(f"❌ Error in resolve_lesson_images: {e}")
        return lesson_data


async def search_lessons(
    query: str,
    user_id: str,
    limit: int = 5
) -> Dict[str, Any]:
    """
    Search for lesson files using vector search
    
    Searches for documents tagged with "lesson" and returns relevant matches.
    Resolves image paths in the lesson content.
    
    Args:
        query: Search query (e.g., "ocean life", "mathematics", "history")
        user_id: User ID for access control
        limit: Maximum number of results to return
        
    Returns:
        Dict with:
            - lessons: List of lesson objects with resolved images
            - count: Number of lessons found
    """
    try:
        from orchestrator.tools.document_tools import search_documents_structured
        
        logger.info(f"🔍 Searching for lessons: {query}")
        
        # Search for documents with "lesson" tag
        search_result = await search_documents_structured(
            query=query,
            user_id=user_id,
            limit=limit,
            tags=["lesson"],
            search_type="semantic"
        )
        
        if not search_result.get("results"):
            logger.info("📚 No lessons found")
            return {"lessons": [], "count": 0}
        
        lessons = []
        for result in search_result["results"]:
            try:
                import json
                
                # Parse lesson content as JSON
                content = result.get("content", "")
                lesson_data = json.loads(content)
                
                # Resolve image paths
                lesson_data = await resolve_lesson_images(
                    lesson_data=lesson_data,
                    lesson_document_id=result.get("document_id"),
                    user_id=user_id
                )
                
                # Add document metadata
                lesson_data["document_id"] = result.get("document_id")
                lesson_data["filename"] = result.get("filename", "Unknown Lesson")
                
                lessons.append(lesson_data)
                logger.info(f"✅ Loaded lesson: {lesson_data.get('title', 'Untitled')}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Skipping non-JSON lesson file: {result.get('filename')}")
            except Exception as e:
                logger.error(f"❌ Error processing lesson {result.get('filename')}: {e}")
        
        return {
            "lessons": lessons,
            "count": len(lessons)
        }
        
    except Exception as e:
        logger.error(f"❌ Error searching lessons: {e}")
        return {"lessons": [], "count": 0, "error": str(e)}


async def generate_lesson(
    topic: str,
    user_id: str,
    difficulty: str = "medium",
    num_questions: int = 5,
    llm: Any = None
) -> Dict[str, Any]:
    """
    Generate a new lesson using LLM
    
    Creates a lesson structure with questions on the specified topic.
    Optionally searches for relevant images to include.
    
    Args:
        topic: Topic for the lesson (e.g., "Ocean Life", "Ancient Rome")
        user_id: User ID for access control
        difficulty: Difficulty level (easy, medium, hard)
        num_questions: Number of questions to generate
        llm: LLM instance to use for generation
        
    Returns:
        Dict with generated lesson data
    """
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        import json
        
        if not llm:
            logger.error("❌ No LLM provided for lesson generation")
            return {"error": "No LLM available"}
        
        logger.info(f"🎓 Generating lesson: {topic} ({difficulty}, {num_questions} questions)")
        
        # Build prompt for lesson generation
        system_prompt = f"""You are an expert educator creating engaging learning content.

Generate a {difficulty} difficulty lesson about {topic} with {num_questions} multiple-choice questions.

LESSON STRUCTURE:
{{
  "lesson_id": "unique_id",
  "title": "Lesson Title",
  "description": "Brief description",
  "difficulty": "{difficulty}",
  "category": "lesson",
  "tags": ["lesson", "education", ...],
  "introduction": "2-3 paragraph introduction to the topic",
  "time_limit_default": 30,
  "questions": [
    {{
      "id": 1,
      "type": "text",
      "question": "Clear, specific question",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Exact match to one option",
      "explanation": "Why this is correct and educational context",
      "time_limit": 30
    }}
  ]
}}

GUIDELINES:
- Make questions educational and interesting
- Include explanations that teach something new
- Vary difficulty within the lesson
- Use clear, unambiguous language
- Ensure exactly one correct answer per question
- For now, generate only text-based questions (type: "text")

Return ONLY the JSON structure, no other text."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a lesson about: {topic}")
        ]
        
        # Generate lesson with LLM
        response = await llm.ainvoke(messages)
        lesson_json = response.content
        
        # Parse JSON response
        import re
        # Remove markdown code blocks if present
        if '```json' in lesson_json:
            match = re.search(r'```json\s*\n(.*?)\n```', lesson_json, re.DOTALL)
            if match:
                lesson_json = match.group(1).strip()
        elif '```' in lesson_json:
            match = re.search(r'```\s*\n(.*?)\n```', lesson_json, re.DOTALL)
            if match:
                lesson_json = match.group(1).strip()
        
        lesson_data = json.loads(lesson_json)
        
        logger.info(f"✅ Generated lesson with {len(lesson_data.get('questions', []))} questions")
        
        return {
            "lesson_data": lesson_data,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating lesson: {e}")
        return {"error": str(e), "success": False}
