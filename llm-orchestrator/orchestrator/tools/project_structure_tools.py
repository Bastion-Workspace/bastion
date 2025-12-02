"""
Project Structure Management Tools

Tools for planning and creating project file structures.
These tools handle:
- LLM-based project structure planning
- File and folder creation from plans
- Loading referenced files from frontmatter

Used by agents that create structured projects (electronics, software, etc.)
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from orchestrator.tools.document_tools import (
    get_document_content_tool,
    search_documents_structured
)
from orchestrator.tools.file_creation_tools import create_user_file_tool
from orchestrator.tools.document_editing_tools import update_document_metadata_tool
from orchestrator.utils.project_utils import sanitize_filename

logger = logging.getLogger(__name__)


async def plan_project_structure(
    query: str,
    user_id: str,
    llm: ChatOpenAI,
    project_type: str = "electronics",
    default_folder: str = "Projects"
) -> Dict[str, Any]:
    """
    Use LLM to intelligently plan project structure, files, and organization.
    
    Analyzes the user's project request and creates a structured plan for organizing
    project files, extracting project name, folder path, and file list.
    
    Args:
        query: User's project request/description
        user_id: User ID
        llm: Configured LLM instance for planning
        project_type: Type of project (e.g., "electronics", "software")
        default_folder: Default folder path if not specified (e.g., "Projects")
        
    Returns:
        Dict with:
            - success: bool
            - plan: Dict with project_name, folder_path, files (if success)
            - error: str (if failed)
    """
    try:
        logger.info(f"🔌 Planning project structure with LLM for query: {query[:100]}...")
        
        # Build prompt for project structure planning
        # Note: Prompt is generic but can be customized per project_type
        planning_prompt = f"""You are a {project_type} project planning expert. Analyze the user's project request and create a comprehensive project structure plan.

**CRITICAL**: The user has explicitly requested project creation. You MUST create a complete project plan - do NOT ask for clarification unless the request is completely unintelligible.

**USER REQUEST**: {query}

**YOUR TASK**: Create a project structure plan that includes:
1. A clear, concise project name (max 50 characters, suitable for filenames) - extract from the description or explicit naming
2. A folder path where the project should be created - extract from explicit location or default to "{default_folder}" or "{default_folder}/{project_type.title()}"
3. A list of files that should be created to organize this project properly
4. Appropriate filenames for each file (max 80 characters each, no special characters except - and _)
5. File types/purposes (e.g., "project_plan", "component_spec", "schematic", "protocol", "code")
6. **PROJECT PLAN SECTIONS**: A list of 4-8 main sections that should appear in the project plan document, tailored specifically to this project's goals and requirements. Each section should have:
   - **name**: A concise section title (max 40 characters)
   - **description**: A brief explanation of what belongs in this section (max 200 characters)

   **Section Generation Guidelines**:
   - Analyze the project type and requirements to determine the most relevant sections
   - For electronics projects, consider: System Architecture, Power Systems, Control Logic, User Interface, Testing & Validation, etc.
   - For audio projects, consider: Signal Processing, Amplification, Filtering, Acoustics, etc.
   - For robotics projects, consider: Mechanical Design, Motor Control, Sensors, Navigation, etc.
   - Always include sections for: requirements gathering, design decisions, implementation details, and testing/validation
   - Make sections specific to the project's unique aspects, not generic boilerplate

**EXTRACTION GUIDELINES**:
- **PROJECT NAME EXTRACTION**: Look for explicit naming patterns first:
  * "Let's call it 'X'" → use "X" as project name
  * "call it 'X'" → use "X" as project name
  * "name it 'X'" → use "X" as project name
  * "named 'X'" → use "X" as project name
  * If no explicit name, extract from key nouns/phrases (e.g., "Allen organ digital control" → "Allen Organ Digital Control")
- **FOLDER PATH EXTRACTION**: Look for explicit location patterns:
  * "put it under 'X'" → use "X" as ROOT folder (will create subfolder "X/ProjectName/")
  * "put it in 'X'" → if X is a root folder (like "Projects"), create subfolder "X/ProjectName/"; if X is a specific path (like "Projects/Electronics"), use that path directly
  * "create it in 'X'" → same logic as "put it in"
  * "under 'X'" → use "X" as ROOT folder (will create subfolder "X/ProjectName/")
  * **CRITICAL DISTINCTION**:
    - Root folder (e.g., "Projects", "My Documents") → create subfolder "RootFolder/ProjectName/" for files
    - Specific path (e.g., "Projects/Electronics", "My Documents/ExistingFolder") → use that path directly
  * If user mentions "Projects" folder anywhere, use "Projects" as root (will create "Projects/ProjectName/")
  * Otherwise default to "{default_folder}" as root (will create "{default_folder}/ProjectName/")
- Create a project_plan.md file as the main document
- Add additional files based on project complexity
- Be intelligent about file segmentation - break complex projects into logical files

**CONSTRAINTS**:
- Project name should preserve spaces (e.g., "Allen Control System Redesign") - filesystem supports spaces
- Folder path should preserve spaces in project names (e.g., "Projects/Allen Control System Redesign")
- **Filenames should preserve spaces** (e.g., "system architecture.md", "component list.md") - filesystem supports spaces in filenames
- Only remove truly problematic filesystem characters: / \\ : * ? " < > |
- Keep filenames under 80 characters (excluding extension)
- Organize files logically based on the project type

**RESPOND WITH VALID JSON ONLY - NO MARKDOWN, NO CODE BLOCKS, JUST RAW JSON**:
{{
  "project_name": "Extracted or derived project name",
  "folder_path": "{default_folder} or {default_folder}/{project_type.title()}",
  "files": [
    {{
      "filename": "project_name_project_plan.md",
      "type": "project_plan",
      "title": "Project Name - Project Plan",
      "description": "Main project plan document"
    }},
    {{
      "filename": "component_specs.md",
      "type": "component_spec",
      "title": "Component Specifications",
      "description": "Component selection and specifications"
    }}
  ],
  "project_plan_sections": [
    {{
      "name": "System Architecture",
      "description": "High-level system design, component relationships, and data flow for the organ control system"
    }},
    {{
      "name": "Power Distribution",
      "description": "Power supply design, voltage regulation, and current requirements for organ components"
    }},
    {{
      "name": "Control Logic",
      "description": "Microcontroller programming, state machines, and control algorithms for organ functions"
    }},
    {{
      "name": "Testing & Validation",
      "description": "Test procedures, performance metrics, and validation criteria for the control system"
    }}
  ]
}}

**CRITICAL**: Your response must be ONLY valid JSON, no markdown code blocks, no explanations, just the raw JSON object starting with {{ and ending with }}.

**IMPORTANT**: 
- ONLY set "needs_clarification": true if the request is genuinely ambiguous or missing critical information
- If the user describes a project (even vaguely), extract a reasonable project name from the description
- Default to creating at least a project_plan.md file
- Be proactive - create a useful project structure even if some details are unclear"""
        
        # Call LLM for planning
        messages = [HumanMessage(content=planning_prompt)]
        response = await llm.ainvoke(messages)
        
        # Parse JSON response
        try:
            # Extract JSON from response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            plan = json.loads(content)
            
            # Validate plan structure
            if plan.get("needs_clarification"):
                logger.warning("⚠️ LLM determined clarification needed - but user requested creation, so creating anyway with best guess")
                # Don't fail - create a default plan instead
                plan["needs_clarification"] = False
                if not plan.get("project_name"):
                    # Extract a reasonable project name from query
                    words = query.split()[:5]  # First few words
                    plan["project_name"] = " ".join(words).title()[:50]
                if not plan.get("folder_path"):
                    plan["folder_path"] = default_folder
                if not plan.get("files"):
                    # Preserve spaces in filename - only remove problematic filesystem characters
                    safe_name = sanitize_filename(plan["project_name"])
                    plan["files"] = [{
                        "filename": f"{safe_name} - Project Plan.md",
                        "type": "project_plan",
                        "title": plan["project_name"],
                        "description": "Main project plan document"
                    }]
            
            # Validate required fields - be lenient, create defaults if needed
            if not plan.get("project_name"):
                # Extract from query as fallback
                words = query.split()[:5]
                plan["project_name"] = " ".join(words).title()[:50]
                logger.info(f"🔌 Generated project name from query: {plan['project_name']}")
            
            if len(plan.get("project_name", "")) > 50:
                plan["project_name"] = plan["project_name"][:50]
                logger.warning(f"⚠️ Truncated project name to 50 chars: {plan['project_name']}")
            
            if not plan.get("folder_path"):
                plan["folder_path"] = default_folder
                logger.info(f"🔌 Using default folder path: {default_folder}")
            
            if not plan.get("files") or len(plan.get("files", [])) == 0:
                # Create at least a project plan file
                # Preserve spaces in filename - only remove problematic filesystem characters
                safe_name = sanitize_filename(plan["project_name"])
                plan["files"] = [{
                    "filename": f"{safe_name} - Project Plan.md",
                    "type": "project_plan",
                    "title": plan["project_name"],  # Preserve original name with spaces
                    "description": "Main project plan document"
                }]
                logger.info("🔌 Created default project plan file entry")
            
            logger.info(f"✅ Project structure planned: {plan.get('project_name')} with {len(plan.get('files', []))} files")
            
            return {
                "success": True,
                "plan": plan
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse project structure plan JSON: {e}")
            logger.error(f"Raw response: {response.content}")
            return {
                "success": False,
                "plan": None,
                "error": f"Failed to parse JSON: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"❌ Project structure planning failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "plan": None,
            "error": str(e)
        }


async def execute_project_structure_plan(
    plan: Dict[str, Any],
    query: str,
    user_id: str,
    project_type: str = "electronics",
    project_category: str = "electronics",
    project_plan_sections: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Execute the project structure plan by creating files and folders.
    
    Creates all files specified in the plan, handling folder path logic
    (root folder vs specific path), and builds frontmatter with file references.
    
    Args:
        plan: Project structure plan from plan_project_structure()
        query: Original user query (for project plan content)
        user_id: User ID
        project_type: Type of project (e.g., "electronics", "software", "hvac")
        project_category: Category for file metadata (e.g., "electronics", "hvac")
        project_plan_sections: Optional list of dicts with "name" and "description" keys
            to customize project plan template sections. If None, uses default sections
            based on project_type. Example:
            [
                {"name": "Components", "description": "Component specifications will be added here"},
                {"name": "Schematics", "description": "Circuit schematics and diagrams will be added here"}
            ]
        
    Returns:
        Dict with:
            - success: bool
            - created_files: List of created file info dicts
            - project_plan_document_id: Document ID of main project plan
            - project_plan_filename: Filename of main project plan
            - actual_folder_path: Final folder path used
            - error: str (if failed)
    """
    try:
        project_name = plan.get("project_name")
        folder_path = plan.get("folder_path", "Projects")
        files = plan.get("files", [])
        
        # Determine if folder_path is a root folder (needs subfolder) or specific path
        # Root folders: "Projects", "My Documents", "Documents"
        # Specific paths: "Projects/Electronics", "My Documents/ExistingFolder", etc.
        root_folders = ["projects", "my documents", "documents"]
        folder_path_lower = folder_path.lower()
        is_root_folder = folder_path_lower in root_folders or folder_path_lower.count("/") == 0
        
        if is_root_folder:
            # Create project-specific subfolder under root
            # IMPORTANT: Use the original project_name from the plan, NOT from folder_path
            # The project_name preserves spaces, folder_path might have underscores from LLM
            # Preserve original project name with spaces (filesystem supports spaces)
            # Only sanitize truly problematic characters
            safe_project_name = sanitize_filename(project_name)
            actual_folder_path = f"{folder_path}/{safe_project_name}"
            logger.info(f"🔌 Root folder detected: '{folder_path}' → creating subfolder '{actual_folder_path}' for project '{project_name}'")
        else:
            # Use the specific path as-is, but check if it contains the project name with underscores
            # If folder_path contains a project name with underscores, replace with spaces
            if project_name and project_name.replace(" ", "_") in folder_path:
                # Replace underscored version with spaced version
                underscored_name = project_name.replace(" ", "_")
                actual_folder_path = folder_path.replace(underscored_name, project_name)
                logger.info(f"🔌 Replaced underscored project name in path: '{folder_path}' → '{actual_folder_path}'")
            else:
                actual_folder_path = folder_path
            logger.info(f"🔌 Specific path provided: using '{actual_folder_path}' for project '{project_name}'")
        
        logger.info(f"🔌 Creating project '{project_name}' with {len(files)} files in '{actual_folder_path}'")
        
        created_files = []
        
        # Separate project plan from other files (create project plan last so we can reference other files)
        project_plan_spec = None
        other_files = []
        for file_spec in files:
            if file_spec.get("type") == "project_plan":
                project_plan_spec = file_spec
            else:
                other_files.append(file_spec)
        
        # Create all non-project-plan files first
        for file_spec in other_files:
            filename = file_spec.get("filename")
            # Sanitize filename - preserve spaces, only remove problematic characters
            if filename:
                filename = sanitize_filename(filename)
            file_type = file_spec.get("type", "document")
            title = file_spec.get("title", project_name)
            description = file_spec.get("description", "")
            
            # Generate content based on file type
            # Include detailed description in frontmatter for lazy loading relevance scoring
            description_line = f"description: {description}" if description else ""
            guidance_text = description if description else f"This file contains detailed {file_type} information for the project."
            content = f"""---
type: {project_type}
title: {title}
{description_line}
---

# {title}

{guidance_text}

<!-- Detailed content will be added here as the project develops -->
"""
            
            try:
                result = await create_user_file_tool(
                    filename=filename,
                    content=content,
                    folder_path=actual_folder_path,  # Use the computed folder path (with subfolder if needed)
                    title=title,
                    tags=[project_category, file_type],
                    category=project_category,
                    user_id=user_id
                )
                
                if result.get("success"):
                    document_id = result.get("document_id")
                    logger.info(f"✅ Created file: {filename} ({document_id})")
                    
                    # Update frontmatter type
                    try:
                        await update_document_metadata_tool(
                            document_id=document_id,
                            frontmatter_type=project_type,
                            user_id=user_id
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to update frontmatter for {filename}: {e}")
                    
                    created_files.append({
                        "filename": filename,
                        "document_id": document_id,
                        "type": file_type
                    })
                else:
                    logger.warning(f"⚠️ Failed to create file {filename}: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"❌ Error creating file {filename}: {e}")
                continue
        
        # Now create the project plan file with references to all other files
        if project_plan_spec:
            filename = project_plan_spec.get("filename")
            # Sanitize filename - preserve spaces, only remove problematic characters
            if filename:
                filename = sanitize_filename(filename)
            title = project_plan_spec.get("title", project_name)
            
            # Categorize created files by type for frontmatter references
            components = []
            protocols = []
            schematics = []
            specifications = []
            code_files = []
            all_files = []  # Always include all files for active editor context
            
            for created_file in created_files:
                file_type = created_file.get("type", "")
                file_filename = created_file.get("filename", "")
                
                # Always add to all_files list
                all_files.append(file_filename)
                
                # Map file types to reference categories
                file_type_lower = file_type.lower()
                if file_type_lower in ["component_spec", "component", "components"]:
                    components.append(file_filename)
                elif file_type_lower in ["protocol", "protocol_doc", "protocols"]:
                    protocols.append(file_filename)
                elif file_type_lower in ["schematic", "schematic_doc", "schematics", "diagram"]:
                    schematics.append(file_filename)
                elif file_type_lower in ["specification", "spec", "specs", "specifications"]:
                    specifications.append(file_filename)
                elif file_type_lower in ["code", "firmware", "software", "programming"]:
                    code_files.append(file_filename)
            
            # Build frontmatter with references
            frontmatter_lines = [
                "---",
                f"type: {project_type}",
                f"title: {title}",
                f"description: High-level project overview and plan for {project_name}, including component relationships, design goals, and implementation roadmap"
            ]
            
            # Always add all files reference (for active editor context)
            # Use relative path format (./filename.md) for consistency with fiction agent
            if all_files:
                files_yaml = "\n".join([f"  - ./{f}" for f in all_files])
                frontmatter_lines.append(f"files:\n{files_yaml}")
            
            # Add categorized reference fields if we have files (YAML list format)
            # Use relative path format (./filename.md) for consistency
            if components:
                components_yaml = "\n".join([f"  - ./{c}" for c in components])
                frontmatter_lines.append(f"components:\n{components_yaml}")
            if protocols:
                protocols_yaml = "\n".join([f"  - ./{p}" for p in protocols])
                frontmatter_lines.append(f"protocols:\n{protocols_yaml}")
            if schematics:
                schematics_yaml = "\n".join([f"  - ./{s}" for s in schematics])
                frontmatter_lines.append(f"schematics:\n{schematics_yaml}")
            if specifications:
                specs_yaml = "\n".join([f"  - ./{s}" for s in specifications])
                frontmatter_lines.append(f"specifications:\n{specs_yaml}")
            if code_files:
                code_yaml = "\n".join([f"  - ./{c}" for c in code_files])
                frontmatter_lines.append(f"code:\n{code_yaml}")
            
            frontmatter_lines.append("---")
            frontmatter_block = "\n".join(frontmatter_lines)
            
            # Build project plan content with references in frontmatter
            # Use sections from the plan if provided, otherwise use parameter, otherwise use defaults
            if plan.get("project_plan_sections"):
                # Use LLM-generated sections from the plan
                project_plan_sections = plan["project_plan_sections"]
                logger.info(f"🔌 Using {len(project_plan_sections)} LLM-generated project plan sections")
            elif project_plan_sections is not None:
                # Use provided parameter sections
                logger.info(f"🔌 Using {len(project_plan_sections)} provided project plan sections")
            else:
                # Use defaults based on project type
                if project_type.lower() == "electronics":
                    project_plan_sections = [
                        {"name": "Components", "description": "Component specifications and selection criteria"},
                        {"name": "Schematics", "description": "Circuit schematics and wiring diagrams"},
                        {"name": "Protocols", "description": "Communication protocols and data formats"},
                        {"name": "Code", "description": "Embedded code and firmware implementation"},
                        {"name": "Notes", "description": "Project notes, testing results, and updates"}
                    ]
                else:
                    # Generic default for other project types
                    project_plan_sections = [
                        {"name": "Specifications", "description": "Project specifications and requirements"},
                        {"name": "Design", "description": "Design documents and technical approach"},
                        {"name": "Implementation", "description": "Implementation details and development"},
                        {"name": "Notes", "description": "Project notes and documentation"}
                    ]
                logger.info(f"🔌 Using {len(project_plan_sections)} default project plan sections for {project_type}")
            
            # Build sections content
            sections_content = []
            for section in project_plan_sections:
                section_name = section.get("name", "")
                section_desc = section.get("description", "Content will be added here")
                sections_content.append(f"## {section_name}\n\n{section_desc}")

            sections_block = "\n\n".join(sections_content)
            
            # Build project plan content with references in frontmatter
            content = f"""{frontmatter_block}

# {title}

## Project Overview

{query if len(query) < 500 else query[:500] + "..."}

{sections_block}
"""
            
            try:
                result = await create_user_file_tool(
                    filename=filename,
                    content=content,
                    folder_path=actual_folder_path,
                    title=title,
                    tags=[project_category, "project"],
                    category=project_category,
                    user_id=user_id
                )
                
                if result.get("success"):
                    document_id = result.get("document_id")
                    logger.info(f"✅ Created project plan file: {filename} ({document_id}) with references to {len(created_files)} other files")
                    
                    # DON'T call update_document_metadata_tool here - it will overwrite the frontmatter
                    # The frontmatter is already correctly set in the content with all file references
                    # Only update if we need to change something specific
                    
                    created_files.append({
                        "filename": filename,
                        "document_id": document_id,
                        "type": "project_plan"
                    })
                else:
                    logger.warning(f"⚠️ Failed to create project plan file {filename}: {result.get('error')}")
                    
            except Exception as e:
                logger.error(f"❌ Error creating project plan file: {e}")
        
        if created_files:
            # Find the project plan file
            project_plan_file = next((f for f in created_files if f.get("type") == "project_plan"), created_files[0])
            
            logger.info(f"✅ Successfully created {len(created_files)} project file(s)")
            
            return {
                "success": True,
                "created_files": created_files,
                "project_plan_document_id": project_plan_file.get("document_id"),
                "project_plan_filename": project_plan_file.get("filename"),
                "actual_folder_path": actual_folder_path
            }
        else:
            logger.warning("⚠️ No files were created from plan")
            return {
                "success": False,
                "created_files": [],
                "error": "No files were created"
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to execute project structure plan: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "created_files": [],
            "error": str(e)
        }


async def load_referenced_context(
    active_editor: Optional[Dict[str, Any]],
    user_id: str,
    doc_type_filter: Optional[str] = "electronics"
) -> Dict[str, Any]:
    """
    Load referenced files from active editor frontmatter (electronics/project style).
    
    Uses the unified reference file loader with electronics-specific configuration.
    
    Args:
        active_editor: Active editor dict with frontmatter (from shared_memory)
        user_id: User ID
        doc_type_filter: Only load references if document type matches (None = load any type)
        
    Returns:
        Dict with:
            - referenced_context: Dict of loaded files by category (components, protocols, etc.)
            - error: str (if failed)
    """
    from orchestrator.tools.reference_file_loader import load_referenced_files
    
    # Electronics/project reference configuration
    reference_config = {
        "components": ["components", "component", "component_docs"],
        "protocols": ["protocols", "protocol", "protocol_docs"],
        "schematics": ["schematics", "schematic", "schematic_docs"],
        "specifications": ["specifications", "spec", "specs", "specification"],
        "other": ["references", "reference", "docs", "documents", "related", "files"]
    }
    
    # Use unified loader
    result = await load_referenced_files(
        active_editor=active_editor,
        user_id=user_id,
        reference_config=reference_config,
        doc_type_filter=doc_type_filter,
        cascade_config=None  # No cascading for electronics
    )
    
    # Convert to expected format (referenced_context instead of loaded_files)
    loaded_files = result.get("loaded_files", {})
    referenced_context = {
        "components": loaded_files.get("components", []),
        "protocols": loaded_files.get("protocols", []),
        "schematics": loaded_files.get("schematics", []),
        "specifications": loaded_files.get("specifications", []),
        "other": loaded_files.get("other", [])
    }
    
    return {
        "referenced_context": referenced_context,
        "error": result.get("error")
    }

