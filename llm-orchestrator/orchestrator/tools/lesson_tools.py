"""
Lesson Tools - Functions for finding and managing learning lessons
"""

import logging
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from orchestrator.utils.action_io_registry import register_action

logger = logging.getLogger(__name__)


# ── I/O models for lesson tools ──────────────────────────────────────────────

class ResolveLessonImagesInputs(BaseModel):
    """Required inputs for resolve_lesson_images."""
    lesson_data: Dict[str, Any] = Field(description="Lesson JSON with questions containing image_path")
    lesson_document_id: str = Field(description="Document ID of the lesson file")
    user_id: str = Field(description="User ID for access control")


class ResolveLessonImagesOutputs(BaseModel):
    """Typed outputs for resolve_lesson_images."""
    lesson_data: Dict[str, Any] = Field(description="Updated lesson_data with image_url on questions")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class SearchLessonsInputs(BaseModel):
    """Required inputs for search_lessons."""
    query: str = Field(description="Search query e.g. ocean life, mathematics")
    user_id: str = Field(description="User ID for access control")


class SearchLessonsParams(BaseModel):
    """Optional parameters."""
    limit: int = Field(default=5, description="Max number of results")


class SearchLessonsOutputs(BaseModel):
    """Typed outputs for search_lessons."""
    lessons: List[Dict[str, Any]] = Field(description="List of lesson objects with resolved images")
    count: int = Field(description="Number of lessons found")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


class GenerateLessonInputs(BaseModel):
    """Required inputs for generate_lesson."""
    topic: str = Field(description="Topic for the lesson e.g. Ocean Life, Ancient Rome")
    user_id: str = Field(description="User ID for access control")


class GenerateLessonParams(BaseModel):
    """Optional parameters."""
    difficulty: str = Field(default="medium", description="easy, medium, hard")
    num_questions: int = Field(default=5, description="Number of questions to generate")


class GenerateLessonOutputs(BaseModel):
    """Typed outputs for generate_lesson."""
    lesson_data: Optional[Dict[str, Any]] = Field(default=None, description="Generated lesson structure")
    success: bool = Field(description="Whether generation succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    formatted: str = Field(description="Human-readable summary for LLM/chat")


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
        lesson_content_result = await get_document_content_tool(lesson_document_id, user_id)
        lesson_content = lesson_content_result.get("content", lesson_content_result) if isinstance(lesson_content_result, dict) else lesson_content_result
        if not lesson_content or lesson_content.startswith("Error") or lesson_content.startswith("Document not found"):
            logger.warning(f"⚠️ Could not load lesson document {lesson_document_id} for image resolution")
            return {"lesson_data": lesson_data, "formatted": "Could not load lesson document for image resolution."}
        
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
        
        resolved_count = sum(1 for q in lesson_data.get("questions", []) if q.get("image_url"))
        formatted = f"Resolved images for lesson: {resolved_count} image(s) resolved."
        return {"lesson_data": lesson_data, "formatted": formatted}
        
    except Exception as e:
        logger.error(f"❌ Error in resolve_lesson_images: {e}")
        return {"lesson_data": lesson_data, "formatted": f"Error resolving lesson images: {str(e)}"}


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
        from orchestrator.tools.document_tools import search_documents_tool
        
        logger.info(f"🔍 Searching for lessons: {query}")
        
        # Search for documents with "lesson" tag
        search_result = await search_documents_tool(
            query=query,
            user_id=user_id,
            limit=limit,
            tags=["lesson"],
            search_type="semantic"
        )
        
        if not search_result.get("documents"):
            logger.info("📚 No lessons found")
            return {"lessons": [], "count": 0, "formatted": "No lessons found for this query."}
        
        lessons = []
        for result in search_result["documents"]:
            try:
                import json
                
                # Parse lesson content as JSON
                content = result.get("content", "")
                lesson_data = json.loads(content)
                
                # Resolve image paths
                resolved = await resolve_lesson_images(
                    lesson_data=lesson_data,
                    lesson_document_id=result.get("document_id"),
                    user_id=user_id
                )
                lesson_data = resolved.get("lesson_data", resolved) if isinstance(resolved, dict) else resolved
                # Add document metadata
                lesson_data["document_id"] = result.get("document_id")
                lesson_data["filename"] = result.get("filename", "Unknown Lesson")
                
                lessons.append(lesson_data)
                logger.info(f"✅ Loaded lesson: {lesson_data.get('title', 'Untitled')}")
                
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Skipping non-JSON lesson file: {result.get('filename')}")
            except Exception as e:
                logger.error(f"❌ Error processing lesson {result.get('filename')}: {e}")
        
        formatted_parts = [f"Found {len(lessons)} lesson(s) for query."]
        for i, lesson in enumerate(lessons[:10], 1):
            title = lesson.get("title") or "Untitled"
            topic = lesson.get("topic") or lesson.get("category") or ", ".join(lesson.get("tags", [])[:3]) or "—"
            difficulty = lesson.get("difficulty") or "—"
            questions = lesson.get("questions") or []
            q_count = len(questions)
            line = f"  {i}. **{title}** — topic: {topic}, difficulty: {difficulty}, questions: {q_count}"
            formatted_parts.append(line)
        if len(lessons) > 10:
            formatted_parts.append(f"  ... and {len(lessons) - 10} more.")
        formatted = "\n".join(formatted_parts)
        return {
            "lessons": lessons,
            "count": len(lessons),
            "formatted": formatted
        }
        
    except Exception as e:
        logger.error(f"❌ Error searching lessons: {e}")
        return {"lessons": [], "count": 0, "error": str(e), "formatted": f"Error searching lessons: {str(e)}"}


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
            return {"error": "No LLM available", "success": False, "formatted": "No LLM available for lesson generation."}
        
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
        
        formatted = f"Generated lesson '{lesson_data.get('title', topic)}' with {len(lesson_data.get('questions', []))} questions."
        return {
            "lesson_data": lesson_data,
            "success": True,
            "formatted": formatted
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating lesson: {e}")
        return {"error": str(e), "success": False, "formatted": f"Error generating lesson: {str(e)}"}


register_action(
    name="resolve_lesson_images",
    category="lesson",
    description="Resolve relative image paths in lesson to absolute URLs",
    inputs_model=ResolveLessonImagesInputs,
    outputs_model=ResolveLessonImagesOutputs,
    tool_function=resolve_lesson_images,
)
register_action(
    name="search_lessons",
    category="lesson",
    description="Search for lesson files using vector search",
    inputs_model=SearchLessonsInputs,
    params_model=SearchLessonsParams,
    outputs_model=SearchLessonsOutputs,
    tool_function=search_lessons,
)
register_action(
    name="generate_lesson",
    category="lesson",
    description="Generate a new lesson using LLM",
    inputs_model=GenerateLessonInputs,
    params_model=GenerateLessonParams,
    outputs_model=GenerateLessonOutputs,
    tool_function=generate_lesson,
)
