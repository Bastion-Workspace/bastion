# Complete Tools Inventory

## Universal vs Project-Specific Tools

**Universal Tools**: Work for ANY MD file (active editor, referenced files, or any document)
**Project-Specific Tools**: Designed for managing multi-file projects with frontmatter references

---

## 🔍 FINDING FILES

### Universal Tools (Work for ANY MD file)

#### 1. **`find_document_by_path`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/backend_tool_client.py`

**Purpose**: Find document by true filesystem path (deterministic, not semantic search)

**Parameters**:
- `file_path`: Relative or absolute path (e.g., `"./component_list.md"`, `"../file.md"`)
- `user_id`: User ID for access control
- `base_path`: Base directory for resolving relative paths (optional)

**Returns**: `Dict` with `document_id`, `filename`, `resolved_path` or `None`

**Usage**:
```python
from orchestrator.backend_tool_client import get_backend_tool_client

client = await get_backend_tool_client()
doc_info = await client.find_document_by_path(
    file_path="./component_list.md",
    base_path="/app/uploads/Users/admin/Projects/MyProject",
    user_id=user_id
)
```

**Universal**: ✅ Works for ANY file path, not just referenced files

---

#### 2. **`get_document_content_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_tools.py`

**Purpose**: Get full content of any document by document_id

**Parameters**:
- `document_id`: Document ID
- `user_id`: User ID for access control

**Returns**: Full document content as string or error message

**Usage**:
```python
from orchestrator.tools.document_tools import get_document_content_tool

content = await get_document_content_tool(document_id, user_id)
```

**Universal**: ✅ Works for ANY document by ID

---

#### 3. **`search_documents_structured`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_tools.py`

**Purpose**: Semantic vector search across all documents

**Parameters**:
- `query`: Search query string
- `limit`: Maximum results (default: 10)
- `user_id`: User ID for access control

**Returns**: `Dict` with `results` (list of documents) and `total_count`

**Usage**:
```python
from orchestrator.tools.document_tools import search_documents_structured

results = await search_documents_structured(
    query="circuit design",
    user_id=user_id,
    limit=20
)
```

**Universal**: ✅ Searches ALL documents, not just project files

---

#### 4. **`search_by_tags_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_tools.py`

**Purpose**: Search documents by tags and/or categories (metadata search)

**Parameters**:
- `tags`: List of tags to filter by
- `categories`: List of categories to filter by (optional)
- `query`: Optional search query for additional filtering
- `limit`: Maximum results (default: 20)
- `user_id`: User ID for access control

**Returns**: Formatted search results as string

**Usage**:
```python
from orchestrator.tools.document_tools import search_by_tags_tool

results = await search_by_tags_tool(
    tags=["electronics", "component"],
    categories=["electronics"],
    user_id=user_id
)
```

**Universal**: ✅ Searches ALL documents by metadata

---

#### 5. **`find_documents_by_tags_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_tools.py`

**Purpose**: Find documents that contain ALL specified tags (database query)

**Parameters**:
- `required_tags`: List of tags that ALL must be present
- `user_id`: User ID for access control
- `collection_type`: Filter by collection ("user", "global", or empty for both)
- `limit`: Maximum results

**Returns**: List of document dictionaries with metadata

**Usage**:
```python
from orchestrator.tools.document_tools import find_documents_by_tags_tool

docs = await find_documents_by_tags_tool(
    required_tags=["electronics", "component"],
    user_id=user_id,
    limit=20
)
```

**Universal**: ✅ Finds ANY documents matching tags

---

### Project-Specific Tools (For Referenced Files)

#### 6. **`load_referenced_files`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/reference_file_loader.py`

**Purpose**: Load referenced files from active editor's frontmatter using true path resolution

**Parameters**:
- `active_editor`: Active editor dict with frontmatter
- `user_id`: User ID
- `reference_config`: Dict mapping categories to frontmatter keys
- `doc_type_filter`: Optional document type filter
- `cascade_config`: Optional cascade config for nested references

**Returns**: `Dict` with `loaded_files` organized by category

**Usage**:
```python
from orchestrator.tools.reference_file_loader import load_referenced_files

result = await load_referenced_files(
    active_editor=active_editor,
    user_id=user_id,
    reference_config={
        "components": ["components", "component"],
        "protocols": ["protocols", "protocol"]
    }
)
```

**Universal**: ❌ Only works for files referenced in active editor's frontmatter

---

## 📝 CREATING FILES

### Universal Tools (Work for ANY file creation)

#### 1. **`create_user_file_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/file_creation_tools.py`

**Purpose**: Create a new file in user's My Documents section

**Parameters**:
- `filename`: Name of file to create (e.g., `"sensor_spec.md"`)
- `content`: File content as string
- `folder_id`: Optional folder ID (must be user's folder)
- `folder_path`: Optional folder path (e.g., `"Projects/Electronics"`) - will create if needed
- `title`: Optional document title (defaults to filename)
- `tags`: Optional list of tags
- `category`: Optional category
- `user_id`: User ID (required)

**Returns**: `Dict` with `success`, `document_id`, `filename`, `folder_id`, `message`

**Usage**:
```python
from orchestrator.tools.file_creation_tools import create_user_file_tool

result = await create_user_file_tool(
    filename="component_spec.md",
    content="# Component Spec\n\n...",
    folder_path="Projects/Electronics",
    title="Component Specification",
    tags=["electronics", "component"],
    category="electronics",
    user_id=user_id
)
```

**Universal**: ✅ Creates ANY file, anywhere in user's documents

---

#### 2. **`create_user_folder_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/file_creation_tools.py`

**Purpose**: Create a folder in user's My Documents section

**Parameters**:
- `folder_name`: Name of folder to create
- `parent_folder_id`: Optional parent folder ID
- `parent_folder_path`: Optional parent folder path (e.g., `"Projects"`)
- `user_id`: User ID (required)

**Returns**: `Dict` with `success`, `folder_id`, `folder_name`, `parent_folder_id`, `message`

**Usage**:
```python
from orchestrator.tools.file_creation_tools import create_user_folder_tool

result = await create_user_folder_tool(
    folder_name="Electronics Projects",
    parent_folder_path="Projects",
    user_id=user_id
)
```

**Universal**: ✅ Creates ANY folder in user's documents

---

### Project-Specific Tools (For Project Structure)

#### 3. **`plan_project_structure`** (LLM-Powered)
**Location**: `llm-orchestrator/orchestrator/tools/project_structure_tools.py`

**Purpose**: Use LLM to intelligently plan project structure, files, and organization

**Parameters**:
- `query`: User query describing project
- `user_id`: User ID
- `llm`: Configured ChatOpenAI instance
- `project_type`: Project type (e.g., `"electronics"`, `"fiction"`)
- `default_folder`: Default folder for project

**Returns**: `Dict` with `success`, `plan` (contains `project_name`, `folder_path`, `files[]`)

**Universal**: ❌ Designed for multi-file project planning

---

#### 4. **`execute_project_structure_plan`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/project_structure_tools.py`

**Purpose**: Execute project structure plan by creating files and folders with proper frontmatter

**Parameters**:
- `plan`: Project structure plan from `plan_project_structure`
- `query`: Original user query
- `user_id`: User ID
- `project_type`: Project type
- `project_category`: Project category
- `project_plan_sections`: Optional sections for project plan file

**Returns**: `Dict` with `success`, `project_plan_document_id`, created files info

**Universal**: ❌ Designed for executing multi-file project plans

---

## ✏️ EDITING/UPDATING FILES

### Universal Tools (Work for ANY MD file)

#### 1. **`update_document_content_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_editing_tools.py`

**Purpose**: Update document content (append or replace entire content)

**Parameters**:
- `document_id`: Document ID to update
- `content`: New content (if `append=True`, adds to existing; if `append=False`, replaces entire content)
- `user_id`: User ID (required - must match document owner)
- `append`: If `True`, append content; if `False`, replace entire content

**Returns**: `Dict` with `success`, `document_id`, `content_length`, `message`

**Usage**:
```python
from orchestrator.tools.document_editing_tools import update_document_content_tool

# Append content
result = await update_document_content_tool(
    document_id=doc_id,
    content="\n\n## New Section\n\nNew content...",
    user_id=user_id,
    append=True
)

# Replace entire content
result = await update_document_content_tool(
    document_id=doc_id,
    content="# New Document\n\nComplete new content...",
    user_id=user_id,
    append=False
)
```

**Universal**: ✅ Works for ANY document by ID (active editor, referenced files, or any document)

**File Types**: Works for `.md`, `.txt`, `.org` files (text-editable types)

---

#### 2. **`propose_document_edit_tool`** (Backend gRPC - HITL)
**Location**: `llm-orchestrator/orchestrator/tools/document_editing_tools.py`

**Purpose**: Propose document edits for user review (Human-in-the-Loop)

**Parameters**:
- `document_id`: Document ID to edit
- `edit_type`: `"operations"` or `"content"`
- `operations`: List of EditorOperation dicts (for operation-based edits)
- `content_edit`: ContentEdit dict (for content-based edits)
- `agent_name`: Name of proposing agent
- `summary`: Human-readable summary of proposed changes
- `requires_preview`: If `False` and edit is small, frontend may auto-apply
- `user_id`: User ID (required - must match document owner)

**Returns**: `Dict` with `success`, `proposal_id`, `document_id`, `message`

**Usage**:
```python
from orchestrator.tools.document_editing_tools import propose_document_edit_tool

# Operation-based edit (precise, structured)
result = await propose_document_edit_tool(
    document_id=doc_id,
    edit_type="operations",
    operations=[
        {
            "op_type": "insert_after_heading",
            "heading": "## Components",
            "content": "\n\n### New Component\n\nDescription..."
        }
    ],
    agent_name="electronics_agent",
    summary="Add new component specification",
    user_id=user_id
)

# Content-based edit (full content replacement)
result = await propose_document_edit_tool(
    document_id=doc_id,
    edit_type="content",
    content_edit={
        "old_content": "Old content...",
        "new_content": "New content..."
    },
    agent_name="electronics_agent",
    summary="Update component specifications",
    user_id=user_id
)
```

**Universal**: ✅ Works for ANY document by ID (HITL pattern for safety)

**File Types**: Works for `.md`, `.txt`, `.org` files

---

#### 3. **`apply_document_edit_proposal_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_editing_tools.py`

**Purpose**: Apply an approved document edit proposal

**Parameters**:
- `proposal_id`: ID of proposal to apply
- `selected_operation_indices`: Which operations to apply (None = all, only for operation-based edits)
- `user_id`: User ID (required - must match proposal owner)

**Returns**: `Dict` with `success`, `document_id`, `applied_count`, `message`

**Usage**:
```python
from orchestrator.tools.document_editing_tools import apply_document_edit_proposal_tool

result = await apply_document_edit_proposal_tool(
    proposal_id=proposal_id,
    selected_operation_indices=None,  # Apply all operations
    user_id=user_id
)
```

**Universal**: ✅ Applies proposals to ANY document

---

#### 4. **`update_document_metadata_tool`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/document_editing_tools.py`

**Purpose**: Update document title and/or frontmatter type

**Parameters**:
- `document_id`: Document ID to update
- `title`: Optional new title (updates both database metadata and frontmatter)
- `frontmatter_type`: Optional frontmatter type (e.g., `"electronics"`, `"fiction"`, `"rules"`)
- `user_id`: User ID (required - must match document owner)

**Returns**: `Dict` with `success`, `document_id`, `updated_fields`, `message`

**Usage**:
```python
from orchestrator.tools.document_editing_tools import update_document_metadata_tool

result = await update_document_metadata_tool(
    document_id=doc_id,
    title="New Title",
    frontmatter_type="electronics",
    user_id=user_id
)
```

**Universal**: ✅ Works for ANY document by ID

---

### Project-Specific Tools (For Multi-File Project Management)

#### 5. **`save_or_update_project_content`** (Backend gRPC - Intelligent Routing)
**Location**: `llm-orchestrator/orchestrator/tools/project_content_tools.py`

**Purpose**: Intelligently save/update project content in appropriate files based on content type and frontmatter references

**Parameters**:
- `result`: Agent response dict (will be modified to add save/update notifications)
- `project_plan_document_id`: Document ID of main project plan
- `referenced_context`: Dict of referenced files by category
- `documents`: List of document dicts from search results
- `user_id`: User ID
- `metadata`: Metadata dict containing shared_memory, active_editor, etc.

**Returns**: `None` (modifies `result` dict in place)

**Usage**:
```python
from orchestrator.tools.project_content_tools import save_or_update_project_content

await save_or_update_project_content(
    result=agent_response,
    project_plan_document_id=project_plan_id,
    referenced_context=referenced_files,
    documents=search_results,
    user_id=user_id,
    metadata={"shared_memory": shared_memory, "active_editor": active_editor}
)
```

**Universal**: ❌ Designed for routing content to project files based on frontmatter references

**What it does**:
1. Analyzes content type (component, protocol, schematic, etc.)
2. Determines target file from frontmatter references
3. Checks if new file needed or existing file should be updated
4. Uses `propose_section_update` or `append_project_content` internally

---

#### 6. **`determine_content_target`** (Pure Function)
**Location**: `llm-orchestrator/orchestrator/tools/project_content_tools.py`

**Purpose**: Determine which file and section to update based on content type, file titles, and descriptions

**Parameters**:
- `response_text`: Content text to analyze
- `frontmatter`: Frontmatter from active editor (contains file references)
- `referenced_context`: Dict of referenced files by category
- `documents`: List of enriched document dicts with titles/descriptions

**Returns**: `Tuple` of `(content_type, target_file_info)` or `(None, None)`

**Universal**: ❌ Designed for routing content to project files

---

#### 7. **`propose_section_update`** (Backend gRPC - HITL)
**Location**: `llm-orchestrator/orchestrator/tools/project_content_tools.py`

**Purpose**: Propose an update to an existing section in a project file

**Parameters**:
- `document_id`: Document ID
- `existing_content`: Current document content
- `section_name`: Section name to update
- `new_content`: New content for the section
- `user_id`: User ID

**Returns**: `Dict` with proposal info

**Universal**: ✅ Works for ANY document, but designed for section-based updates

---

#### 8. **`append_project_content`** (Backend gRPC)
**Location**: `llm-orchestrator/orchestrator/tools/project_content_tools.py`

**Purpose**: Append new content to a project file under a specific section

**Parameters**:
- `document_id`: Document ID
- `section_name`: Section name (e.g., `"Component Specifications"`)
- `content`: Content to append
- `user_id`: User ID

**Returns**: `Dict` with success info

**Universal**: ✅ Works for ANY document, but designed for section-based appends

---

## Summary: Universal vs Project-Specific

### ✅ Universal Tools (Work for ANY MD file)

**Finding**:
- `find_document_by_path` - Find by filesystem path
- `get_document_content_tool` - Get content by document_id
- `search_documents_structured` - Semantic search
- `search_by_tags_tool` - Search by tags/categories
- `find_documents_by_tags_tool` - Find by required tags

**Creating**:
- `create_user_file_tool` - Create any file
- `create_user_folder_tool` - Create any folder

**Editing/Updating**:
- `update_document_content_tool` - Update content (append/replace)
- `propose_document_edit_tool` - Propose edits (HITL)
- `apply_document_edit_proposal_tool` - Apply approved proposals
- `update_document_metadata_tool` - Update title/type
- `propose_section_update` - Propose section updates (HITL)
- `append_project_content` - Append to sections

### ❌ Project-Specific Tools (For Multi-File Projects)

**Finding**:
- `load_referenced_files` - Load files from frontmatter references

**Creating**:
- `plan_project_structure` - Plan project structure
- `execute_project_structure_plan` - Execute project plan

**Editing/Updating**:
- `save_or_update_project_content` - Intelligent content routing
- `determine_content_target` - Determine target file for content
- `enrich_documents_with_metadata` - Enrich with metadata
- `check_if_new_file_needed` - Check if new file needed
- `create_new_project_file` - Create new project file

---

## Recommendation

**For universal file operations (ANY MD file, active editor or referenced)**:
- Use `update_document_content_tool` for direct updates
- Use `propose_document_edit_tool` for HITL-safe updates
- Use `find_document_by_path` to find files by path
- Use `get_document_content_tool` to read files by ID

**For project-specific operations (multi-file projects with frontmatter)**:
- Use `save_or_update_project_content` for intelligent routing
- Use `load_referenced_files` to load referenced files
- Use project structure tools for planning/execution

**The universal tools are already sufficient for updating ANY MD file!** The project-specific tools add intelligent routing and multi-file coordination, but the universal tools work for any document.

