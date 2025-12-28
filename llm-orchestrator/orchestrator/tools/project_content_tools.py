"""
Project Content Management Tools

Tools for intelligently managing project content across multiple files.
These tools coordinate multiple backend operations to:
- Determine where content should be saved
- Enrich documents with metadata
- Check if new files are needed
- Create new project files
- Update or append content to existing files

Used by agents that manage structured projects (electronics, fiction, etc.)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from orchestrator.tools.document_tools import get_document_content_tool
from orchestrator.tools.document_editing_tools import (
    update_document_content_tool,
    propose_document_edit_tool
)
from orchestrator.tools.file_creation_tools import create_user_file_tool
from orchestrator.utils.project_utils import (
    sanitize_filename,
    generate_filename_from_content,
    generate_title_from_content,
    extract_description_from_content
)

logger = logging.getLogger(__name__)


def extract_structured_content(
    response_text: str,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract and structure different content types from agent response.
    
    Separates conversational response into structured documentation:
    - Current state information (what exists now)
    - New plans/recommendations (what to do/build)
    - Component specifications
    - Code snippets
    - Calculations
    - General information
    
    Args:
        response_text: Raw conversational response text
        result: Agent response dict with structured fields (components, code_snippets, etc.)
        
    Returns:
        Dict with structured content by type
    """
    structured = {
        "current_state": "",
        "new_plans": "",
        "components": "",
        "code": "",
        "calculations": "",
        "general": ""
    }
    
    # Extract structured data from result if available
    components = result.get("components", [])
    code_snippets = result.get("code_snippets", [])
    calculations = result.get("calculations", [])
    recommendations = result.get("recommendations", [])
    
    # Format components as structured documentation
    if components:
        structured["components"] = "## Component Specifications\n\n"
        for comp in components:
            structured["components"] += f"### {comp.get('name', 'Component')}\n"
            structured["components"] += f"- **Type**: {comp.get('type', 'N/A')}\n"
            structured["components"] += f"- **Value/Specification**: {comp.get('value', 'N/A')}\n"
            structured["components"] += f"- **Purpose**: {comp.get('purpose', 'N/A')}\n"
            if comp.get('alternatives'):
                structured["components"] += f"- **Alternatives**: {', '.join(comp.get('alternatives', []))}\n"
            structured["components"] += "\n"
    
    # Format code snippets as structured documentation
    if code_snippets:
        structured["code"] = "## Code Implementation\n\n"
        for code in code_snippets:
            structured["code"] += f"### {code.get('purpose', 'Code Snippet')}\n"
            structured["code"] += f"- **Platform**: {code.get('platform', 'N/A')}\n"
            structured["code"] += f"- **Language**: {code.get('language', 'N/A')}\n"
            structured["code"] += f"\n```{code.get('language', 'cpp')}\n"
            structured["code"] += f"{code.get('code', '')}\n"
            structured["code"] += "```\n\n"
    
    # Format calculations as structured documentation
    if calculations:
        structured["calculations"] = "## Calculations\n\n"
        for calc in calculations:
            structured["calculations"] += f"### {calc.get('type', 'Calculation').replace('_', ' ').title()}\n"
            structured["calculations"] += f"- **Formula**: {calc.get('formula', 'N/A')}\n"
            structured["calculations"] += f"- **Result**: {calc.get('result', 'N/A')}\n"
            structured["calculations"] += f"- **Explanation**: {calc.get('explanation', 'N/A')}\n"
            structured["calculations"] += "\n"
    
    # Analyze response text to identify current state vs new plans
    response_lower = response_text.lower()
    
    # Keywords indicating current state (what exists now)
    current_state_indicators = [
        "currently", "now", "existing", "already", "have", "has", "is using",
        "current setup", "current system", "present", "at present", "right now"
    ]
    
    # Keywords indicating new plans/recommendations (what to do)
    new_plans_indicators = [
        "should", "recommend", "suggest", "plan", "propose", "consider",
        "next step", "would be", "could", "might want", "option",
        "alternative", "better", "improve", "upgrade", "replace"
    ]
    
    # Split response into sentences
    sentences = re.split(r'[.!?]\s+', response_text)
    
    current_state_sentences = []
    new_plans_sentences = []
    general_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence_lower = sentence.lower()
        
        # Check if sentence describes current state
        if any(indicator in sentence_lower for indicator in current_state_indicators):
            current_state_sentences.append(sentence.strip())
        # Check if sentence describes new plans
        elif any(indicator in sentence_lower for indicator in new_plans_indicators):
            new_plans_sentences.append(sentence.strip())
        else:
            general_sentences.append(sentence.strip())
    
    # Format current state
    if current_state_sentences:
        structured["current_state"] = "## Current State\n\n" + " ".join(current_state_sentences)
    
    # Format new plans/recommendations
    if new_plans_sentences:
        structured["new_plans"] = "## Recommendations and Plans\n\n" + " ".join(new_plans_sentences)
    
    # Format recommendations from structured data
    if recommendations:
        if structured["new_plans"]:
            structured["new_plans"] += "\n\n### Additional Recommendations\n\n"
        else:
            structured["new_plans"] = "## Recommendations\n\n"
        for i, rec in enumerate(recommendations, 1):
            structured["new_plans"] += f"{i}. {rec}\n"
    
    # Format general information (everything else)
    if general_sentences:
        structured["general"] = " ".join(general_sentences)
    
    return structured


def format_as_reference_document(
    content: str,
    content_type: str = "general"
) -> str:
    """
    Convert conversational text to structured reference documentation.
    
    Removes conversational elements and formats as technical documentation:
    - Removes "I", "you", "we" conversational language
    - Converts questions to statements
    - Structures information clearly
    - Removes conversational fillers
    
    Args:
        content: Conversational text to format
        content_type: Type of content (component, code, calculation, etc.)
        
    Returns:
        Formatted reference documentation text
    """
    if not content or len(content.strip()) < 50:
        return content
    
    # Remove conversational markers
    conversational_patterns = [
        (r'\b(I|you|we|your|my|our)\s+(think|believe|feel|know|see|understand)\b', ''),
        (r'\b(Let me|Let\'s|I\'ll|I\'m|I\'ve|You\'ll|You\'re|We\'ll|We\'re)\b', ''),
        (r'\b(please|thank you|thanks|great|excellent|perfect|awesome)\b', '', re.IGNORECASE),
        (r'\?\s*(Yes|No|Sure|Okay|OK)\s*[.!]', '.'),
        (r'Would you like|Do you want|Can I help', ''),
        (r'\*I\'ve (saved|updated|created|added).*?\*', ''),  # Remove save notifications
        (r'\*\*File Organization Suggestion\*\*:.*', ''),  # Remove file suggestions
    ]
    
    formatted = content
    for pattern in conversational_patterns:
        if len(pattern) == 3:
            formatted = re.sub(pattern[0], pattern[1], formatted, flags=pattern[2])
        else:
            formatted = re.sub(pattern[0], pattern[1], formatted)
    
    # Convert questions to statements where appropriate
    question_patterns = [
        (r'Would you like to (.+)\?', r'\1 is recommended.'),
        (r'Do you want to (.+)\?', r'\1 is recommended.'),
        (r'Should we (.+)\?', r'\1 is recommended.'),
        (r'Could we (.+)\?', r'\1 is possible.'),
    ]
    
    for pattern, replacement in question_patterns:
        formatted = re.sub(pattern, replacement, formatted, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and newlines
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    formatted = re.sub(r' {2,}', ' ', formatted)
    
    # Ensure proper sentence structure
    formatted = formatted.strip()
    if formatted and not formatted.endswith(('.', '!', '?')):
        formatted += '.'
    
    return formatted


async def save_or_update_project_content(
    result: Dict[str, Any],
    project_plan_document_id: str,
    referenced_context: Dict[str, Any],
    documents: List[Dict[str, Any]],
    user_id: str,
    metadata: Dict[str, Any]
) -> None:
    """
    Intelligently save or update project content in appropriate files.
    
    **ENHANCED**: Now supports saving to MULTIPLE files when content has different types.
    For example, current state information can go to one file while new plans go to another.
    
    Also formats content as structured reference documentation instead of conversational text.
    
    Args:
        result: Agent response dict (will be modified to add save/update notifications)
        project_plan_document_id: Document ID of the main project plan
        referenced_context: Dict of referenced files by category
        documents: List of document dicts from search results
        user_id: User ID
        metadata: Metadata dict containing shared_memory, active_editor, etc.
    """
    try:
        logger.info(f"🔌 save_or_update_project_content: Starting multi-file content routing (project_plan={project_plan_document_id}, has_referenced={bool(referenced_context)})")
        response_text = result.get("response", "")
        if isinstance(response_text, dict):
            response_text = response_text.get("response", str(response_text))
        
        if not response_text or len(response_text) < 100:
            return  # Not substantial enough
        
        # Extract structured content from response
        structured_content = extract_structured_content(response_text, result)
        
        # Get project plan frontmatter to find referenced files
        shared_memory = metadata.get("shared_memory", {})
        active_editor = shared_memory.get("active_editor", {})
        frontmatter = active_editor.get("frontmatter", {})
        
        # Load ALL project files with their titles/descriptions for intelligent routing
        enriched_documents = await enrich_documents_with_metadata(
            documents, referenced_context, user_id
        )
        
        # Track which files we've updated
        updated_files = []
        
        # **MULTI-FILE SAVING**: Save different content types to appropriate files
        
        # 1. Save current state information (if present)
        if structured_content.get("current_state"):
            current_state_text = format_as_reference_document(
                structured_content["current_state"], "current_state"
            )
            # Determine target for current state (often goes to project plan or specifications)
            content_type, target_file_info = determine_content_target(
                current_state_text, frontmatter, referenced_context, enriched_documents
            )
            
            if target_file_info:
                await _save_content_to_file(
                    target_file_info, current_state_text, user_id, "Current State"
                )
                updated_files.append(target_file_info.get("filename", "project file"))
            elif project_plan_document_id:
                # Fallback to project plan
                await _save_content_to_file(
                    {
                        "document_id": project_plan_document_id,
                        "section": "Current State",
                        "filename": "project_plan.md"
                    },
                    current_state_text, user_id, "Current State"
                )
                updated_files.append("project plan")
        
        # 2. Save new plans/recommendations (if present)
        if structured_content.get("new_plans"):
            new_plans_text = format_as_reference_document(
                structured_content["new_plans"], "new_plans"
            )
            # New plans often go to project plan (system-level) or specific planning files
            content_type, target_file_info = determine_content_target(
                new_plans_text, frontmatter, referenced_context, enriched_documents
            )
            
            # Prefer project plan for recommendations/plans (system-level)
            if project_plan_document_id:
                await _save_content_to_file(
                    {
                        "document_id": project_plan_document_id,
                        "section": "Recommendations and Plans",
                        "filename": "project_plan.md"
                    },
                    new_plans_text, user_id, "Recommendations and Plans"
                )
                if "project plan" not in updated_files:
                    updated_files.append("project plan")
            elif target_file_info:
                await _save_content_to_file(
                    target_file_info, new_plans_text, user_id, "Recommendations and Plans"
                )
                updated_files.append(target_file_info.get("filename", "project file"))
        
        # 3. Save component specifications (if present)
        if structured_content.get("components"):
            components_text = format_as_reference_document(
                structured_content["components"], "component"
            )
            content_type, target_file_info = determine_content_target(
                components_text, frontmatter, referenced_context, enriched_documents
            )
            
            if target_file_info:
                await _save_content_to_file(
                    target_file_info, components_text, user_id, "Component Specifications"
                )
                updated_files.append(target_file_info.get("filename", "component file"))
            elif project_plan_document_id:
                # Fallback to project plan
                await _save_content_to_file(
                    {
                        "document_id": project_plan_document_id,
                        "section": "Component Specifications",
                        "filename": "project_plan.md"
                    },
                    components_text, user_id, "Component Specifications"
                )
                if "project plan" not in updated_files:
                    updated_files.append("project plan")
        
        # 4. Save code snippets (if present)
        if structured_content.get("code"):
            code_text = format_as_reference_document(
                structured_content["code"], "code"
            )
            content_type, target_file_info = determine_content_target(
                code_text, frontmatter, referenced_context, enriched_documents
            )
            
            if target_file_info:
                await _save_content_to_file(
                    target_file_info, code_text, user_id, "Code Implementation"
                )
                updated_files.append(target_file_info.get("filename", "code file"))
            elif project_plan_document_id:
                # Fallback to project plan
                await _save_content_to_file(
                    {
                        "document_id": project_plan_document_id,
                        "section": "Code Implementation",
                        "filename": "project_plan.md"
                    },
                    code_text, user_id, "Code Implementation"
                )
                if "project plan" not in updated_files:
                    updated_files.append("project plan")
        
        # 5. Save calculations (if present)
        if structured_content.get("calculations"):
            calc_text = format_as_reference_document(
                structured_content["calculations"], "calculation"
            )
            # Calculations often go to specifications or project plan
            content_type, target_file_info = determine_content_target(
                calc_text, frontmatter, referenced_context, enriched_documents
            )
            
            if target_file_info:
                await _save_content_to_file(
                    target_file_info, calc_text, user_id, "Calculations"
                )
                updated_files.append(target_file_info.get("filename", "specification file"))
            elif project_plan_document_id:
                await _save_content_to_file(
                    {
                        "document_id": project_plan_document_id,
                        "section": "Calculations",
                        "filename": "project_plan.md"
                    },
                    calc_text, user_id, "Calculations"
                )
                if "project plan" not in updated_files:
                    updated_files.append("project plan")
        
        # 6. Save general information (if no structured content was extracted, use original)
        if not any([
            structured_content.get("current_state"),
            structured_content.get("new_plans"),
            structured_content.get("components"),
            structured_content.get("code"),
            structured_content.get("calculations")
        ]):
            # No structured content extracted - use original response but format it
            general_text = format_as_reference_document(response_text, "general")
            content_type, target_file_info = determine_content_target(
                general_text, frontmatter, referenced_context, enriched_documents
            )
            
            if target_file_info:
                await _save_content_to_file(
                    target_file_info, general_text, user_id, "Information"
                )
                updated_files.append(target_file_info.get("filename", "project file"))
            elif project_plan_document_id:
                await _save_content_to_file(
                    {
                        "document_id": project_plan_document_id,
                        "section": "Information",
                        "filename": "project_plan.md"
                    },
                    general_text, user_id, "Information"
                )
                if "project plan" not in updated_files:
                    updated_files.append("project plan")
        
        # Update response to mention files updated
        if updated_files:
            files_list = ", ".join(set(updated_files))
            action_msg = f"*I've saved this information to {files_list}.*"
            if isinstance(result.get("response"), dict):
                result["response"]["response"] = result["response"]["response"] + f"\n\n{action_msg}"
            else:
                result["response"] = str(result.get("response", "")) + f"\n\n{action_msg}"
        
        # Check if we should suggest creating a new file (for additional content that doesn't fit)
        new_file_suggestion = await check_if_new_file_needed(
            response_text, content_type, frontmatter, enriched_documents, referenced_context
        )
        
        if new_file_suggestion:
            logger.info(f"🔌 Also suggesting new file creation: {new_file_suggestion.get('suggested_filename')}")
            if "shared_memory" not in metadata:
                metadata["shared_memory"] = {}
            metadata["shared_memory"]["new_file_suggestion"] = new_file_suggestion
            
            if isinstance(result.get("response"), dict):
                result["response"]["response"] = result["response"]["response"] + f"\n\n**File Organization Suggestion**: {new_file_suggestion.get('suggestion_message')}"
                result["response"]["new_file_suggestion"] = new_file_suggestion
            else:
                result["response"] = str(result.get("response", "")) + f"\n\n**File Organization Suggestion**: {new_file_suggestion.get('suggestion_message')}"
                result["new_file_suggestion"] = new_file_suggestion
            
    except Exception as e:
        logger.error(f"❌ Error saving/updating project content: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def _save_content_to_file(
    target_file_info: Dict[str, Any],
    content: str,
    user_id: str,
    section_name: str
) -> None:
    """
    Helper function to save content to a specific file.
    
    Args:
        target_file_info: Dict with document_id, section, filename
        content: Content to save
        user_id: User ID
        section_name: Name of section to save to
    """
    try:
        document_id = target_file_info.get("document_id")
        if not document_id:
            logger.warning(f"⚠️ No document_id in target_file_info")
            return
        
        # Get existing content to check for updates
        existing_content = await get_document_content_tool(document_id, user_id)
        
        if existing_content.startswith("Error"):
            logger.warning(f"⚠️ Could not read existing content: {existing_content}")
            existing_content = ""
        
        # Check if we should update existing section or append
        update_existing = should_update_existing_section(
            existing_content, section_name, content
        )
        
        if update_existing:
            logger.info(f"🔌 Updating existing section '{section_name}' in {target_file_info.get('filename', 'file')}")
            await propose_section_update(
                document_id, existing_content, section_name, content, user_id
            )
        else:
            logger.info(f"🔌 Appending new section '{section_name}' to {target_file_info.get('filename', 'file')}")
            await append_project_content(
                document_id, section_name, content, user_id
            )
    except Exception as e:
        logger.error(f"❌ Error saving content to file: {e}")


def determine_content_target(
    response_text: str,
    frontmatter: Dict[str, Any],
    referenced_context: Dict[str, Any],
    documents: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Determine which file and section to update based on content type, file titles, and descriptions.
    
    Uses intelligent scoring to match content to the most appropriate referenced file.
    
    Args:
        response_text: Content text to analyze
        frontmatter: Frontmatter from active editor (contains file references)
        referenced_context: Dict of referenced files by category
        documents: List of enriched document dicts with titles/descriptions
        
    Returns:
        Tuple of (content_type, target_file_info) or (None, None) if no match
    """
    response_lower = response_text.lower()
    
    # Content type detection keywords
    content_types = {
        "component": {
            "keywords": ["component", "resistor", "capacitor", "microcontroller", "sensor", "ic", "chip", "transistor", "mosfet", "diode", "led"],
            "section": "Component Specifications",
            "frontmatter_key": "components",
            "file_type_keywords": ["component", "specification", "spec", "part", "hardware"]
        },
        "protocol": {
            "keywords": ["protocol", "communication", "i2c", "spi", "uart", "serial", "can", "ethernet", "network", "data format"],
            "section": "Protocol Documentation",
            "frontmatter_key": "protocols",
            "file_type_keywords": ["protocol", "communication", "interface", "data"]
        },
        "schematic": {
            "keywords": ["schematic", "circuit diagram", "wiring", "connection", "pinout", "layout", "pcb"],
            "section": "Schematic Documentation",
            "frontmatter_key": "schematics",
            "file_type_keywords": ["schematic", "circuit", "diagram", "wiring", "layout"]
        },
        "specification": {
            "keywords": ["specification", "spec", "requirement", "standard", "voltage", "current", "power", "rating"],
            "section": "Technical Specifications",
            "frontmatter_key": "specifications",
            "file_type_keywords": ["specification", "spec", "requirement", "standard", "technical"]
        },
        "architecture": {
            "keywords": [
                "system architecture", "high-level system", "block diagram", "system design", "overview",
                "system requirement", "system requirements", "overarching", "system process", "system processes",
                "integration", "system integration", "source of truth", "project goal", "project goals",
                "project scope", "system constraint", "system constraints", "high-level", "system-level"
            ],
            "section": "System Architecture",
            "frontmatter_key": None,  # Goes to project plan (source of truth for system-level content)
            "file_type_keywords": ["architecture", "system", "design", "overview", "requirement", "process"]
        },
        "code": {
            "keywords": ["code", "programming", "firmware", "arduino", "esp32", "embedded", "function", "void", "int", "python", "cpp"],
            "section": "Code",
            "frontmatter_key": "code",
            "file_type_keywords": ["code", "programming", "firmware", "software", "implementation"]
        }
    }
    
    # Find best matching content type
    best_match = None
    best_score = 0
    
    for content_type, config in content_types.items():
        score = sum(1 for keyword in config["keywords"] if keyword in response_lower)
        if score > best_score:
            best_score = score
            best_match = (content_type, config)
    
    if not best_match or best_score == 0:
        return None, None
    
    content_type, config = best_match
    
    # Find target file from frontmatter references, considering titles and descriptions
    target_file = None
    if config["frontmatter_key"]:
        # Get referenced files for this content type
        referenced_files = frontmatter.get(config["frontmatter_key"], [])
        if isinstance(referenced_files, str):
            referenced_files = [referenced_files]
        
        if referenced_files:
            logger.info(f"🔌 Found {len(referenced_files)} referenced file(s) for content type '{content_type}': {referenced_files}")
            # Score each potential file based on title, description, and filename match
            best_file_score = 0
            best_file_match = None
            
            # **PRIORITIZE**: Check referenced_context files FIRST (these are the project files we want to update)
            # These should already be in enriched_documents with source="referenced_context"
            for target_filename in referenced_files:
                # Remove ./ prefix if present
                clean_filename = target_filename.lstrip('./')
                
                # FIRST: Check referenced_context directly (most reliable)
                ref_docs = referenced_context.get(config["frontmatter_key"], [])
                if ref_docs and isinstance(ref_docs, list):
                    for ref_doc in ref_docs:
                        if isinstance(ref_doc, dict):
                            ref_filename = ref_doc.get("filename", "")
                            ref_title = ref_doc.get("title", "")
                            ref_description = ref_doc.get("description", "") or ref_doc.get("content", "")[:200]
                            
                            # Check filename match
                            if (ref_filename.endswith(clean_filename) or 
                                clean_filename in ref_filename or
                                clean_filename in ref_filename.replace("_", "-")):
                                
                                # Score based on title and description relevance
                                score = 3.0  # High base score for referenced_context files
                                title_lower = ref_title.lower()
                                desc_lower = ref_description.lower()
                                
                                # Boost score if title/description matches content type keywords
                                for keyword in config["file_type_keywords"]:
                                    if keyword in title_lower:
                                        score += 0.5
                                    if keyword in desc_lower:
                                        score += 0.3
                                
                                # Boost score if title/description matches response keywords
                                response_words = set(response_lower.split())
                                title_words = set(title_lower.split())
                                desc_words = set(desc_lower.split())
                                
                                title_overlap = len(response_words & title_words) / max(len(response_words), 1)
                                desc_overlap = len(response_words & desc_words) / max(len(response_words), 1)
                                
                                score += title_overlap * 0.5
                                score += desc_overlap * 0.3
                                
                                if score > best_file_score:
                                    best_file_score = score
                                    best_file_match = {
                                        "document_id": ref_doc.get("document_id"),
                                        "file_type": content_type,
                                        "section": config["section"],
                                        "filename": ref_filename,
                                        "title": ref_title,
                                        "description": ref_description,
                                        "match_score": score
                                    }
                                    logger.info(f"🔌 Found match in referenced_context: {ref_filename} (score: {score:.2f})")
                
                # SECOND: Check enriched_documents (which includes referenced_context files with source="referenced_context")
                for doc in documents:
                    doc_filename = doc.get("filename", "")
                    doc_title = doc.get("title", "")
                    doc_description = doc.get("description", "") or doc.get("metadata", {}).get("description", "")
                    
                    # Check filename match
                    filename_match = (doc_filename.endswith(clean_filename) or 
                                   clean_filename in doc_filename or
                                   clean_filename in doc_filename.replace("_", "-"))
                    
                    if filename_match:
                        # Score based on title and description relevance
                        score = 1.0  # Base score for filename match
                        
                        # BOOST: Prioritize referenced_context files (project files we want to update)
                        if doc.get("source") == "referenced_context":
                            score += 2.0  # Strong boost for referenced files
                        
                        # Boost score if title/description matches content type keywords
                        title_lower = doc_title.lower()
                        desc_lower = doc_description.lower()
                        
                        for keyword in config["file_type_keywords"]:
                            if keyword in title_lower:
                                score += 0.5
                            if keyword in desc_lower:
                                score += 0.3
                        
                        # Boost score if title/description matches response keywords
                        response_words = set(response_lower.split())
                        title_words = set(title_lower.split())
                        desc_words = set(desc_lower.split())
                        
                        title_overlap = len(response_words & title_words) / max(len(response_words), 1)
                        desc_overlap = len(response_words & desc_words) / max(len(response_words), 1)
                        
                        score += title_overlap * 0.5
                        score += desc_overlap * 0.3
                        
                        if score > best_file_score:
                            best_file_score = score
                            best_file_match = {
                                "document_id": doc.get("document_id"),
                                "file_type": content_type,
                                "section": config["section"],
                                "filename": doc_filename,
                                "title": doc_title,
                                "description": doc_description,
                                "match_score": score
                            }
            
            
            target_file = best_file_match
    
    # If no specific file found, return None to use project plan
    if not target_file:
        return content_type, None
    
    return content_type, target_file


async def enrich_documents_with_metadata(
    documents: List[Dict[str, Any]],
    referenced_context: Dict[str, Any],
    user_id: str
) -> List[Dict[str, Any]]:
    """
    Enrich documents list with titles, descriptions, and metadata from actual files.
    
    Loads full document content to extract frontmatter metadata (title, description).
    This ensures agents are aware of the full project structure.
    
    PRIORITIZES referenced_context files over search results for content routing.
    
    Args:
        documents: List of document dicts from search results
        referenced_context: Dict of referenced files by category
        user_id: User ID
    
    Returns:
        List of enriched document dicts with title and description fields
    """
    enriched = []
    
    # FIRST: Process referenced_context files (these are the project files we want to update)
    # These take priority over search results
    if referenced_context:
        for category, ref_docs in referenced_context.items():
            if isinstance(ref_docs, list):
                for ref_doc in ref_docs:
                    if isinstance(ref_doc, dict):
                        doc_id = ref_doc.get("document_id")
                        if doc_id:
                            try:
                                # Get full document content to extract title and description
                                content = await get_document_content_tool(doc_id, user_id)
                                if not content.startswith("Error"):
                                    # Parse frontmatter for title and description
                                    frontmatter_match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", content)
                                    if frontmatter_match:
                                        frontmatter_text = frontmatter_match.group(1)
                                        # Extract title
                                        title_match = re.search(r'^title:\s*(.+)$', frontmatter_text, re.MULTILINE)
                                        if title_match:
                                            ref_doc["title"] = title_match.group(1).strip().strip('"\'')
                                        # Extract description (if in frontmatter)
                                        desc_match = re.search(r'^description:\s*(.+)$', frontmatter_text, re.MULTILINE)
                                        if desc_match:
                                            ref_doc["description"] = desc_match.group(1).strip().strip('"\'')
                                    
                                    # Mark as referenced file for priority
                                    ref_doc["source"] = "referenced_context"
                                    ref_doc["category"] = category
                                    
                                    # Add to enriched list (these are prioritized)
                                    enriched.append(ref_doc)
                            except Exception as e:
                                logger.warning(f"⚠️ Could not enrich referenced file {doc_id}: {e}")
    
    # SECOND: Process documents from search results (these are secondary)
    for doc in documents:
        doc_id = doc.get("document_id")
        if doc_id and doc_id != "active_editor":
            try:
                # Get full document content to extract title and description
                content = await get_document_content_tool(doc_id, user_id)
                if not content.startswith("Error"):
                    # Parse frontmatter for title and description
                    frontmatter_match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", content)
                    if frontmatter_match:
                        frontmatter_text = frontmatter_match.group(1)
                        # Extract title
                        title_match = re.search(r'^title:\s*(.+)$', frontmatter_text, re.MULTILINE)
                        if title_match:
                            doc["title"] = title_match.group(1).strip().strip('"\'')
                        # Extract description (if in frontmatter)
                        desc_match = re.search(r'^description:\s*(.+)$', frontmatter_text, re.MULTILINE)
                        if desc_match:
                            doc["description"] = desc_match.group(1).strip().strip('"\'')
                    
                    # Also check document metadata
                    if not doc.get("title"):
                        doc["title"] = doc.get("metadata", {}).get("title", doc.get("filename", ""))
                    if not doc.get("description"):
                        doc["description"] = doc.get("metadata", {}).get("description", "")
            except Exception as e:
                logger.warning(f"⚠️ Could not enrich document {doc_id}: {e}")
        
        enriched.append(doc)
    
    # Also enrich referenced context documents
    for category, ref_docs in referenced_context.items():
        if isinstance(ref_docs, list):
            for ref_doc in ref_docs:
                if isinstance(ref_doc, dict):
                    doc_id = ref_doc.get("document_id")
                    if doc_id and doc_id not in [d.get("document_id") for d in enriched]:
                        try:
                            # Get content to extract metadata
                            content = await get_document_content_tool(doc_id, user_id)
                            if not content.startswith("Error"):
                                # Parse frontmatter
                                frontmatter_match = re.match(r"^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n", content)
                                if frontmatter_match:
                                    frontmatter_text = frontmatter_match.group(1)
                                    title_match = re.search(r'^title:\s*(.+)$', frontmatter_text, re.MULTILINE)
                                    if title_match:
                                        ref_doc["title"] = title_match.group(1).strip().strip('"\'')
                                    desc_match = re.search(r'^description:\s*(.+)$', frontmatter_text, re.MULTILINE)
                                    if desc_match:
                                        ref_doc["description"] = desc_match.group(1).strip().strip('"\'')
                                
                                # Add to enriched list if not already there
                                if doc_id not in [d.get("document_id") for d in enriched]:
                                    enriched.append(ref_doc)
                        except Exception as e:
                            logger.warning(f"⚠️ Could not enrich referenced document {doc_id}: {e}")
    
    return enriched


async def check_if_new_file_needed(
    response_text: str,
    content_type: Optional[str],
    frontmatter: Dict[str, Any],
    documents: List[Dict[str, Any]],
    referenced_context: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Check if a new file should be created for this content.
    
    Analyzes content length, specificity, and existing file matches to determine
    if content warrants its own dedicated file.
    
    Args:
        response_text: Content text to analyze
        content_type: Detected content type (component, protocol, etc.)
        frontmatter: Frontmatter from active editor
        documents: List of enriched document dicts
        referenced_context: Dict of referenced files by category
        
    Returns:
        Dict with file suggestion details, or None if no new file needed
    """
    try:
        response_lower = response_text.lower()
        
        # Don't suggest new files for architecture (goes to project plan)
        if not content_type or content_type == "architecture":
            return None
        
        # Check if content is substantial and specific enough to warrant its own file
        if len(response_text) < 500:
            return None  # Too short for a dedicated file
        
        # Check if there's already a good file for this content type
        config = {
            "component": {"frontmatter_key": "components", "file_type": "component_spec", "section": "Component Specifications"},
            "protocol": {"frontmatter_key": "protocols", "file_type": "protocol", "section": "Protocol Documentation"},
            "schematic": {"frontmatter_key": "schematics", "file_type": "schematic", "section": "Schematic Documentation"},
            "specification": {"frontmatter_key": "specifications", "file_type": "specification", "section": "Technical Specifications"},
            "code": {"frontmatter_key": "code", "file_type": "code", "section": "Code"}
        }.get(content_type)
        
        if not config:
            return None
        
        # Check if we have files of this type
        existing_files = frontmatter.get(config["frontmatter_key"], [])
        if isinstance(existing_files, str):
            existing_files = [existing_files]
        
        # **CONSERVATIVE APPROACH**: Only suggest new files if existing files are truly insufficient
        # If we have ANY files of this type, prefer updating them over creating new ones
        if existing_files:
            # Check if any existing file would be a good match
            best_match_score = 0
            best_match_doc = None
            
            # First check referenced_context files (these are the project files we want to update)
            for category, ref_docs in referenced_context.items():
                if isinstance(ref_docs, list):
                    for doc in ref_docs:
                        doc_type = doc.get("metadata", {}).get("type", "").lower() if isinstance(doc.get("metadata"), dict) else ""
                        doc_filename = doc.get("filename", "").lower()
                        
                        # Check if this is the right type of file
                        if (config["file_type"] in doc_type or 
                            content_type in doc_type or
                            config["file_type"] in doc_filename or
                            content_type in doc_filename):
                            
                            # Check title/description relevance
                            title = doc.get("title", "").lower()
                            description = doc.get("description", "").lower()
                            
                            # Simple relevance check
                            response_words = set(response_lower.split())
                            title_words = set(title.split())
                            desc_words = set(description.split())
                            
                            title_overlap = len(response_words & title_words) / max(len(response_words), 1)
                            desc_overlap = len(response_words & desc_words) / max(len(response_words), 1)
                            
                            score = title_overlap * 0.6 + desc_overlap * 0.4
                            if score > best_match_score:
                                best_match_score = score
                                best_match_doc = doc
            
            # Also check documents from search results
            for doc in documents:
                doc_type = doc.get("metadata", {}).get("type", "").lower() if isinstance(doc.get("metadata"), dict) else ""
                if config["file_type"] in doc_type or content_type in doc_type:
                    # Check title/description relevance
                    title = doc.get("title", "").lower()
                    description = doc.get("description", "").lower()
                    
                    # Simple relevance check
                    response_words = set(response_lower.split())
                    title_words = set(title.split())
                    desc_words = set(description.split())
                    
                    title_overlap = len(response_words & title_words) / max(len(response_words), 1)
                    desc_overlap = len(response_words & desc_words) / max(len(response_words), 1)
                    
                    score = title_overlap * 0.6 + desc_overlap * 0.4
                    if score > best_match_score:
                        best_match_score = score
                        best_match_doc = doc
            
            # **CONSERVATIVE THRESHOLD**: If we have ANY existing file of this type, use it
            # Only suggest new file if match is VERY poor (<20% overlap) AND content is very substantial
            # This ensures we prefer updating existing files over creating new ones
            if best_match_score > 0.2 or len(response_text) < 1500:
                logger.info(f"🔌 Existing file found with {best_match_score:.2%} match - will update instead of creating new file")
                return None
        
        # Check if content is about a specific, distinct topic that warrants its own file
        topic_indicators = [
            "specific", "dedicated", "separate", "standalone", "individual",
            "focused", "specialized", "particular"
        ]
        
        has_topic_indicator = any(indicator in response_lower for indicator in topic_indicators)
        
        # Check for specific component/system names
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response_text)
        has_specific_name = len(capitalized_words) >= 2
        
        # Check content length
        is_substantial = len(response_text) > 1500  # Higher threshold for new file suggestions
        
        # **VERY CONSERVATIVE**: Only suggest new file if:
        # 1. Content is VERY substantial (>1500 chars) AND
        # 2. Has specific topic indicators AND specific component names AND
        # 3. NO existing files of this type OR match is VERY poor (<20%)
        # This ensures we almost always prefer updating existing files
        should_suggest = (is_substantial and 
                        has_topic_indicator and 
                        has_specific_name and
                        (not existing_files or best_match_score < 0.2))
        
        if not should_suggest:
            return None
        
        # Generate suggestion
        suggested_filename = generate_filename_from_content(response_text, content_type)
        suggested_title = generate_title_from_content(response_text, content_type)
        suggested_description = extract_description_from_content(response_text)
        
        suggestion_message = (
            f"I notice this content is substantial and focused on a specific topic. "
            f"Would you like me to create a new file '{suggested_filename}' for this {content_type} content? "
            f"This would help keep your project organized. Just say 'yes' or 'create {suggested_filename}' to proceed."
        )
        
        return {
            "suggested": True,
            "suggested_filename": suggested_filename,
            "suggested_title": suggested_title,
            "suggested_description": suggested_description,
            "content_type": content_type,
            "file_type": config["file_type"],
            "frontmatter_key": config["frontmatter_key"],
            "section": config["section"],
            "suggestion_message": suggestion_message
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking if new file needed: {e}")
        return None


async def create_new_project_file(
    file_suggestion: Dict[str, Any],
    project_plan_document_id: str,
    initial_content: str,
    project_plan_frontmatter: Dict[str, Any],
    user_id: str,
    metadata: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Create a new project file based on user approval.
    
    Creates the file in the same folder as the project plan and updates
    the project plan frontmatter to include the new file reference.
    
    Args:
        file_suggestion: Dict with suggested_filename, suggested_title, etc.
        project_plan_document_id: Document ID of the main project plan
        initial_content: Content to add to the new file
        project_plan_frontmatter: Frontmatter from project plan
        user_id: User ID
        metadata: Metadata dict containing shared_memory, active_editor
        
    Returns:
        Dict with document_id, filename, title, file_type, or None if failed
    """
    try:
        filename = file_suggestion.get("suggested_filename")
        title = file_suggestion.get("suggested_title")
        description = file_suggestion.get("suggested_description", "")
        file_type = file_suggestion.get("file_type")
        frontmatter_key = file_suggestion.get("frontmatter_key")
        
        if not filename or not title:
            logger.error("❌ Missing filename or title in file suggestion")
            return None
        
        # Get project plan folder_id from active_editor or document metadata
        shared_memory = metadata.get("shared_memory", {})
        active_editor = shared_memory.get("active_editor", {})
        
        folder_id = None
        folder_path = None
        
        # **PRIORITY 1**: Try to get folder_id from active_editor
        if active_editor:
            folder_id = active_editor.get("folder_id")
            if folder_id:
                logger.info(f"📁 Using folder_id from active_editor: {folder_id}")
        
        # **PRIORITY 2**: Get folder_id from document metadata (MOST RELIABLE)
        if not folder_id and project_plan_document_id:
            try:
                from orchestrator.backend_tool_client import get_backend_tool_client
                client = await get_backend_tool_client()
                doc_info = await client.get_document(project_plan_document_id, user_id)
                if doc_info and doc_info.get("metadata"):
                    metadata_dict = doc_info["metadata"]
                    folder_id = metadata_dict.get("folder_id")
                    if folder_id:
                        logger.info(f"📁 Got folder_id from document metadata: {folder_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get folder_id from document: {e}")
        
        # **PRIORITY 3**: Extract folder_path from canonical_path (fallback)
        if not folder_id:
            canonical_path = active_editor.get("canonical_path", "")
            if not canonical_path and project_plan_document_id:
                # Try to get canonical_path from document metadata
                try:
                    from orchestrator.backend_tool_client import get_backend_tool_client
                    client = await get_backend_tool_client()
                    doc_info = await client.get_document(project_plan_document_id, user_id)
                    if doc_info and doc_info.get("metadata"):
                        canonical_path = doc_info["metadata"].get("canonical_path", "")
                except Exception:
                    pass
            
            if canonical_path:
                from pathlib import Path
                try:
                    # Parse canonical_path to get folder hierarchy
                    # Format: /app/uploads/Users/{username}/Projects/Allen Organ Controls/project_plan.md
                    path_parts = Path(canonical_path).parts
                    
                    # Find "Users" to start folder path
                    if "Users" in path_parts:
                        users_idx = path_parts.index("Users")
                        if users_idx + 2 < len(path_parts) - 1:  # username + at least one folder + filename
                            # Get folder parts (skip username and filename)
                            folder_parts = path_parts[users_idx + 2:-1]
                            if folder_parts:
                                folder_path = "/".join(folder_parts)
                                logger.info(f"📁 Extracted folder_path from canonical_path: {folder_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to extract folder_path from canonical_path: {e}")
        
        if not folder_id and not folder_path:
            logger.error(f"❌ Could not determine folder_id or folder_path for project file creation")
            return None
        
        # Create file content with frontmatter
        content = f"""---
type: electronics
title: {title}
description: {description}
tags: [electronics, {file_type}]
category: electronics
---

# {title}

{description if description else "<!-- Content will be added here -->"}

<!-- Content will be added here -->
"""
        
        # Create the file - use folder_id if available, otherwise folder_path
        result = await create_user_file_tool(
            filename=filename,
            content=content,
            folder_id=folder_id,  # Prefer folder_id over folder_path
            folder_path=folder_path if not folder_id else None,
            title=title,
            tags=["electronics", file_type],
            category="electronics",
            user_id=user_id
        )
        
        if not result.get("success"):
            logger.error(f"❌ Failed to create new file: {result.get('error')}")
            return None
        
        new_file_document_id = result.get("document_id")
        logger.info(f"✅ Created new project file: {filename} ({new_file_document_id})")
        
        # Update project plan frontmatter to include this new file
        try:
            # Get current project plan content
            project_plan_content = await get_document_content_tool(project_plan_document_id, user_id)
            if not project_plan_content.startswith("Error"):
                # Use reusable frontmatter utility to add file reference
                from orchestrator.utils.frontmatter_utils import add_to_frontmatter_list
                
                new_entry = f"./{filename}"
                updated_content, success = await add_to_frontmatter_list(
                    content=project_plan_content,
                    list_key=frontmatter_key,
                    new_items=[new_entry],
                    also_update_files=True  # Also add to 'files' list for active editor context
                )
                
                if success:
                    # Update project plan
                    update_result = await update_document_content_tool(
                        document_id=project_plan_document_id,
                        content=updated_content,
                        user_id=user_id,
                        append=False
                    )
                    
                    if update_result.get("success"):
                        logger.info(f"✅ Updated project plan frontmatter with new file reference: {filename} (preserved all existing fields)")
                    else:
                        logger.warning(f"⚠️ Failed to update project plan frontmatter: {update_result.get('error')}")
                else:
                    logger.warning(f"⚠️ Failed to parse/update frontmatter for project plan")
        except Exception as e:
            logger.warning(f"⚠️ Could not update project plan frontmatter: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Continue - file was created successfully
        
        return {
            "document_id": new_file_document_id,
            "filename": filename,
            "title": title,
            "file_type": file_type
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating new project file: {e}")
        return None


def _is_placeholder_content(content: str) -> bool:
    """
    Check if content is a placeholder that should be replaced.
    
    Detects common placeholder patterns like:
    - "Content will be added here"
    - "<!-- Content will be added here -->"
    - "TODO: Add content"
    - Empty sections with just headers
    - Very short placeholder text
    
    Args:
        content: Content to check
        
    Returns:
        True if content appears to be a placeholder
    """
    if not content or len(content.strip()) < 50:
        return True
    
    content_lower = content.lower().strip()
    
    # Common placeholder patterns
    placeholder_patterns = [
        r"content will be added",
        r"<!--\s*content will be added",
        r"todo:\s*add",
        r"placeholder",
        r"to be added",
        r"will be added here",
        r"coming soon",
        r"tbd\s*$",
        r"^<!--\s*$",  # Empty HTML comment
        r"^#+\s*$",  # Just headers, no content
    ]
    
    for pattern in placeholder_patterns:
        if re.search(pattern, content_lower, re.IGNORECASE | re.MULTILINE):
            return True
    
    # Check if content is mostly whitespace, comments, or very short
    # Remove markdown headers, HTML comments, and whitespace
    cleaned = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if len(cleaned) < 30:
        return True
    
    return False


def should_update_existing_section(
    existing_content: str,
    section_name: str,
    new_content: str
) -> bool:
    """
    Check if an existing section should be updated (vs appending new section).
    
    Determines if a section with the given name already exists and if the new
    content overlaps significantly with existing content (indicating an update),
    OR if the existing section contains placeholder content that should be replaced.
    
    Args:
        existing_content: Current document content
        section_name: Name of the section to check
        new_content: New content to potentially update with
        
    Returns:
        True if section exists and should be updated, False if should append
    """
    # **IMPROVED SECTION MATCHING**: Look for existing sections with similar names
    # Check exact matches first, then fuzzy matches
    section_headers = [
        f"## {section_name}",
        f"### {section_name}",
        f"# {section_name}"
    ]

    # Also check for fuzzy matches (similar section names)
    section_words = set(section_name.lower().split())
    content_lower = existing_content.lower()

    # Look for sections that share significant keywords
    potential_matches = []

    # First check exact header matches
    for header in section_headers:
        if header.lower() in content_lower:
            pattern = rf"{re.escape(header)}.*?(?=\n##|\n###|\n#|$)"
            match = re.search(pattern, existing_content, re.IGNORECASE | re.DOTALL)
            if match:
                potential_matches.append((header, match.group(0)))

    # If no exact matches, look for fuzzy matches (sections with similar names)
    if not potential_matches:
        # Find all section headers in the document
        all_headers = re.findall(r'^(#{1,3})\s+(.+)$', existing_content, re.MULTILINE)
        for level, header_text in all_headers:
            header_words = set(header_text.lower().split())
            # Check if section names share significant keywords
            overlap = len(section_words & header_words) / max(len(section_words), len(header_words))
            if overlap > 0.5:  # 50% word overlap
                full_header = f"{level} {header_text}"
                pattern = rf"{re.escape(full_header)}.*?(?=\n##|\n###|\n#|$)"
                match = re.search(pattern, existing_content, re.IGNORECASE | re.DOTALL)
                if match:
                    potential_matches.append((full_header, match.group(0)))
                    logger.info(f"🔍 Found fuzzy section match: '{full_header}' ≈ '{section_name}' (overlap: {overlap:.1%})")

    # Evaluate each potential match
    for matched_header, existing_section in potential_matches:
        # **NEW**: Check if existing section is a placeholder - if so, always update
        if _is_placeholder_content(existing_section):
            logger.info(f"🔌 Detected placeholder content in section '{matched_header}' - will replace")
            return True

        # Calculate keyword overlap between existing section and new content
        existing_words = set(existing_section.lower().split())
        new_words = set(new_content.lower().split())

        overlap = len(existing_words & new_words) / max(len(new_words), 1)

        # **ENHANCED UPDATE LOGIC**: Update existing section if:
        # 1. >15% keyword overlap (further reduced for better updating)
        # 2. OR if the new content is significantly longer (likely more detailed)
        # 3. OR if existing section is very short (< 200 chars, likely incomplete)
        # 4. OR if new content mentions updating/replacing existing content
        # 5. OR if this is a fuzzy match (similar section name)
        # 6. **NEW**: OR if this is a component section and component names differ (replacement)
        # 7. **NEW**: OR if section name matches exactly (always update exact matches)

        should_update = False

        # **NEW**: Always update if section name matches exactly (exact match)
        if section_name.lower() in [h.lower().replace('#', '').strip() for h in section_headers]:
            should_update = True
            logger.info(f"🔄 Exact section name match - updating '{matched_header}'")
        elif overlap > 0.15:  # Further reduced threshold
            should_update = True
            logger.info(f"🔄 Content overlap ({overlap:.1%}) - updating existing section '{matched_header}'")
        elif len(existing_section.strip()) < 200:  # Short existing content
            should_update = True
            logger.info(f"🔄 Existing section '{matched_header}' is short ({len(existing_section)} chars) - updating")
        elif len(new_content) > len(existing_section) * 1.2:  # New content longer
            should_update = True
            logger.info(f"🔄 New content longer than existing - updating section '{matched_header}'")
        elif any(phrase in new_content.lower() for phrase in ["update", "replace", "revise", "modify", "improve", "expand", "enhance", "changed", "switching", "instead of"]):
            should_update = True
            logger.info(f"🔄 New content mentions updating - replacing section '{matched_header}'")
        elif matched_header not in [f"## {section_name}", f"### {section_name}", f"# {section_name}"]:  # Fuzzy match
            should_update = True
            logger.info(f"🔄 Fuzzy section match - updating '{matched_header}' with content for '{section_name}'")
        else:
            # **NEW**: Check for component replacements (e.g., "Teensy 4.1" -> "ESP32")
            # Extract component names from both sections
            # Look for component names (capitalized words, model numbers, etc.)
            existing_components = set(re.findall(r'\b([A-Z][a-zA-Z0-9]+(?:\s+\d+\.?\d*)?)\b', existing_section))
            new_components = set(re.findall(r'\b([A-Z][a-zA-Z0-9]+(?:\s+\d+\.?\d*)?)\b', new_content))
            
            # If new content has different component names, it's likely a replacement
            if new_components and existing_components and new_components != existing_components:
                # Check if there's significant overlap in component names (partial match)
                component_overlap = len(new_components & existing_components) / max(len(new_components | existing_components), 1)
                if component_overlap < 0.5:  # Less than 50% overlap = likely replacement
                    should_update = True
                    logger.info(f"🔄 Component names differ (existing: {existing_components}, new: {new_components}) - replacing section '{matched_header}'")

        if should_update:
            return True
    
    return False


def _is_document_open_in_editor(document_id: str, active_editor: Dict[str, Any] = None) -> bool:
    """
    Check if a document is currently open in the editor.
    
    Args:
        document_id: Document ID to check
        active_editor: Active editor context from metadata (optional)
    
    Returns:
        True if document is open in editor, False otherwise
    """
    if not active_editor:
        return False
    
    # Check if active_editor has document_id field
    editor_doc_id = active_editor.get("document_id")
    if editor_doc_id and editor_doc_id == document_id:
        return True
    
    # Check if active_editor filename matches (less reliable but fallback)
    editor_filename = active_editor.get("filename")
    if editor_filename:
        # This is a heuristic - we'd need document_id in active_editor for certainty
        # For now, assume if active_editor exists, the file might be open
        # We'll use a more conservative approach: only auto-apply if we're certain it's NOT open
        pass
    
    return False


async def propose_section_update(
    document_id: str,
    existing_content: str,
    section_name: str,
    new_content: str,
    user_id: str,
    active_editor: Optional[Dict[str, Any]] = None,
    auto_apply_if_closed: bool = True,
    add_timestamp: bool = True,
    agent_name: str = "project_content_manager"
) -> None:
    """
    Propose an update to an existing section using edit proposal system.

    Creates an operations-based edit proposal to replace the existing section
    with updated content, allowing user to review changes before applying.

    Args:
        document_id: Document ID to update
        existing_content: Current document content
        section_name: Name of the section to update
        new_content: New content for the section
        user_id: User ID for permissions
        active_editor: Active editor context (optional)
        auto_apply_if_closed: Whether to auto-apply if document is not open
        add_timestamp: Whether to add an update timestamp (default: True)
    """
    try:
        # Find the section in existing content
        section_headers = [
            f"## {section_name}",
            f"### {section_name}",
            f"# {section_name}"
        ]
        
        section_start = -1
        section_end = -1
        used_header = None
        
        for header in section_headers:
            pattern = rf"{re.escape(header)}.*?(?=\n##|\n###|\n#|$)"
            match = re.search(pattern, existing_content, re.IGNORECASE | re.DOTALL)
            if match:
                section_start = match.start()
                section_end = match.end()
                used_header = header
                break
        
        if section_start == -1:
            # Section not found, append instead
            await append_project_content(document_id, section_name, new_content, user_id)
            return
        
        # Extract existing section content
        existing_section = existing_content[section_start:section_end]
        
        # Check if existing section is a placeholder
        is_placeholder = _is_placeholder_content(existing_section)
        
        # Create updated section content
        # Add timestamp if requested (agent decides)
        if add_timestamp and not is_placeholder:
            # Add timestamp for non-placeholder updates
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            updated_section = f"{used_header}\n\n*Updated on {timestamp}*\n\n{new_content}\n"
            logger.info(f"🔌 Updating section '{section_name}' with timestamp")
        else:
            # No timestamp - clean replacement
            updated_section = f"{used_header}\n\n{new_content}\n"
            if is_placeholder:
                logger.info(f"🔌 Replacing placeholder content in section '{section_name}'")
            else:
                logger.info(f"🔌 Updating section '{section_name}' without timestamp")
        
        # Check if document is open in editor
        is_open = _is_document_open_in_editor(document_id, active_editor)
        
        if is_open or not auto_apply_if_closed:
            # Document is open - propose edit for inline suggestions
            edit_result = await propose_document_edit_tool(
                document_id=document_id,
                edit_type="operations",
                operations=[{
                    "op_type": "replace_range",
                    "start": section_start,
                    "end": section_end,
                    "text": updated_section,
                    "pre_hash": "",  # Will be calculated by backend
                    "original_text": existing_section,
                    "note": f"Update {section_name} section with new information"
                }],
                agent_name="project_content_manager",
                summary=f"Update {section_name} section with revised information",
                requires_preview=True,
                user_id=user_id
            )
            
            if edit_result.get("success"):
                logger.info(f"✅ Proposed update to {section_name} section (file is open - showing inline suggestions)")
                # WebSocket notification is handled by backend's propose_document_edit_tool
            else:
                logger.warning(f"⚠️ Failed to propose section update: {edit_result.get('error')}")
        else:
            # Document is not open - auto-apply using granular operations
            # This allows precise edits even when file is closed
            logger.info(f"🔌 Auto-applying update to {section_name} section using operations (file is not open)")
            
            # Use granular operation instead of full content replacement
            from orchestrator.tools.document_editing_tools import apply_operations_directly_tool
            
            # Create operation to replace the section
            operation = {
                "op_type": "replace_range",
                "start": section_start,
                "end": section_end,
                "text": updated_section,
                "pre_hash": "",  # Will be calculated by backend if needed
                "original_text": existing_section,
                "note": f"Update {section_name} section with new information"
            }
            
            # Apply operation directly (only works for authorized agents like electronics_agent)
            apply_result = await apply_operations_directly_tool(
                document_id=document_id,
                operations=[operation],
                user_id=user_id,
                agent_name=agent_name  # Pass through agent name for security check
            )
            
            if apply_result.get("success"):
                logger.info(f"✅ Auto-applied update to {section_name} section using granular operation")
            else:
                # Fallback to full content replacement if operation application fails
                logger.warning(f"⚠️ Granular operation failed, falling back to content replacement: {apply_result.get('error')}")
                from orchestrator.tools.document_editing_tools import update_document_content_tool
                
                # Build new content with updated section
                new_doc_content = existing_content[:section_start] + updated_section + existing_content[section_end:]
                
                update_result = await update_document_content_tool(
                    document_id=document_id,
                    content=new_doc_content,
                    user_id=user_id,
                    append=False
                )
                
                if update_result.get("success"):
                    logger.info(f"✅ Auto-applied update to {section_name} section (fallback method)")
                else:
                    logger.warning(f"⚠️ Failed to auto-apply section update: {update_result.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Error proposing section update: {e}")
        # Fallback to append
        await append_project_content(document_id, section_name, new_content, user_id)


async def append_project_content(
    document_id: str,
    section_name: str,
    content: str,
    user_id: str,
    active_editor: Optional[Dict[str, Any]] = None,
    auto_apply_if_closed: bool = True
) -> None:
    """
    Append new content to a project file.
    
    Adds a new section with timestamp to the end of the document.
    If file is open in editor, proposes edit for inline suggestions.
    If file is closed, auto-applies the edit.
    
    Args:
        document_id: Document ID to append to
        section_name: Name of the section to add
        content: Content to append
        user_id: User ID
        active_editor: Active editor context (optional)
        auto_apply_if_closed: If True, auto-apply when file is not open
    """
    try:
        # Format the content to append
        content_to_add = f"\n\n## {section_name}\n\n*Added on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n{content}\n"
        
        # Check if document is open in editor
        is_open = _is_document_open_in_editor(document_id, active_editor)
        
        if is_open or not auto_apply_if_closed:
            # Document is open - propose edit for inline suggestions
            edit_result = await propose_document_edit_tool(
                document_id=document_id,
                edit_type="content",
                content_edit={
                    "edit_mode": "append",
                    "content": content_to_add,
                    "note": f"Add new {section_name} section"
                },
                agent_name="project_content_manager",
                summary=f"Add new {section_name} section to document",
                requires_preview=False,  # Append is safe to auto-apply if small
                user_id=user_id
            )
            
            if edit_result.get("success"):
                logger.info(f"✅ Proposed append of {section_name} section (file is open - showing inline suggestions)")
                # WebSocket notification is handled by backend's propose_document_edit_tool
            else:
                logger.warning(f"⚠️ Failed to propose append: {edit_result.get('error')}")
        else:
            # Document is not open - auto-apply the edit
            logger.info(f"🔌 Auto-appending {section_name} section (file is not open)")
            from orchestrator.tools.document_editing_tools import update_document_content_tool
            from orchestrator.tools.document_tools import get_document_content_tool
            from orchestrator.utils.frontmatter_utils import parse_frontmatter
            
            # **FRONTMATTER PRESERVATION**: Read existing content and frontmatter BEFORE append
            existing_content = await get_document_content_tool(document_id, user_id)
            existing_frontmatter = {}
            existing_body = ""
            if not existing_content.startswith("Error"):
                try:
                    existing_frontmatter, existing_body = await parse_frontmatter(existing_content)
                    logger.debug(f"📋 Preserving frontmatter with {len(existing_frontmatter)} fields before append")
                except Exception as parse_error:
                    logger.warning(f"⚠️ Could not parse frontmatter before append: {parse_error}")
            
            # Append new section
            update_result = await update_document_content_tool(
                document_id=document_id,
                content=content_to_add,
                user_id=user_id,
                append=True
            )
            
            if update_result.get("success"):
                # **FRONTMATTER VALIDATION**: Verify frontmatter is still intact after append
                if existing_frontmatter:
                    try:
                        updated_content = await get_document_content_tool(document_id, user_id)
                        if not updated_content.startswith("Error"):
                            updated_frontmatter, _ = await parse_frontmatter(updated_content)
                            updated_fields = set(updated_frontmatter.keys())
                            existing_fields = set(existing_frontmatter.keys())
                            
                            # Check if we lost any fields
                            lost_fields = existing_fields - updated_fields
                            if lost_fields:
                                logger.error(f"❌ CRITICAL: Lost frontmatter fields after append: {lost_fields}")
                                # Restore frontmatter by updating with preserved fields
                                from orchestrator.utils.frontmatter_utils import update_frontmatter_field
                                restored_content, restore_success = await update_frontmatter_field(
                                    content=updated_content,
                                    field_updates=existing_frontmatter,  # Restore all original fields
                                    list_updates={}  # Don't modify lists, just restore scalar fields
                                )
                                if restore_success:
                                    # Also restore list fields
                                    for key, value in existing_frontmatter.items():
                                        if isinstance(value, list) and key not in updated_frontmatter:
                                            restored_content, restore_success = await update_frontmatter_field(
                                                content=restored_content,
                                                field_updates={},
                                                list_updates={key: value}
                                            )
                                    
                                    if restore_success:
                                        # Write back restored content
                                        restore_result = await update_document_content_tool(
                                            document_id=document_id,
                                            content=restored_content,
                                            user_id=user_id,
                                            append=False
                                        )
                                        if restore_result.get("success"):
                                            logger.info(f"✅ Restored frontmatter fields: {lost_fields}")
                                        else:
                                            logger.error(f"❌ Failed to restore frontmatter: {restore_result.get('error')}")
                                    else:
                                        logger.error(f"❌ Failed to restore frontmatter fields")
                                else:
                                    logger.error(f"❌ Failed to restore frontmatter")
                            else:
                                logger.debug(f"✅ Frontmatter intact after append ({len(updated_fields)} fields preserved)")
                    except Exception as validate_error:
                        logger.warning(f"⚠️ Could not validate frontmatter after append: {validate_error}")
                
                logger.info(f"✅ Auto-appended {section_name} section")
            else:
                logger.warning(f"⚠️ Failed to auto-append section: {update_result.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Error appending project content: {e}")


