# Document Editor Capabilities

## Overview

The Plato Knowledge Base includes a comprehensive document management system with hierarchical folder organization, document processing, and integrated editing capabilities. This document outlines the current and planned features for document management and editing.

## Current Features

### File Tree Sidebar

The **File Tree Sidebar** provides a hierarchical view of documents and folders with the following capabilities:

#### Folder Management
- **Root Organization**: 
  - **"My Documents"** - Virtual root node containing user's personal folders
  - **"Global Documents"** - Virtual root node containing admin/global folders (admin users only)
- **Create Folders**: Right-click any folder to create new subfolders
- **Default Folders**: Automatic creation of "My Documents" and "Notes" folders for new users
- **Folder Operations**: 
  - Rename folders
  - Move folders between locations
  - Delete folders (with recursive option)
  - Expand/collapse folder structure

#### Document Organization
- **Root Documents**: Documents not assigned to any folder are displayed in a "Root Documents" section
- **Document Counts**: Visual indicators showing the number of documents and subfolders in each folder
- **Drag & Drop**: Upload files by dragging them onto folders

#### Context Menu Actions
- **For Folders**:
  - New Folder
  - Upload Files
  - Rename
  - Move
  - Delete

- **For Documents**:
  - **Re-process Document** ⭐ (New Feature)
  - **Edit Metadata** ⭐ (New Feature) - Edit tags, title, description, author, category, and publication date
  - Rename
  - Move
  - Delete

### Document Processing

#### Supported File Types
- **Text Documents**: `.txt`, `.md`
- **Org Mode Files**: `.org` (stored for structured access, not vectorized)
- **Office Documents**: `.docx`, `.doc`
- **PDF Documents**: `.pdf`
- **E-books**: `.epub`
- **Web Content**: `.html`
- **Archives**: `.zip`
- **Email**: `.eml`
- **Subtitles**: `.srt`

#### Processing Pipeline

**Note on Org Mode Files:**
Org Mode files (`.org`) are handled differently from other documents:
- ✅ **Stored in the document system** for file management
- ❌ **NOT vectorized** - no semantic search or embeddings generated
- 🏇 **Accessed via structured queries** through OrgInboxAgent and OrgProjectAgent
- 📋 **Optimized for task management** - TODO states, tags, schedules, properties

This design choice recognizes that Org files contain structured task data, not prose, and benefit from direct structural queries rather than semantic similarity search.

#### Processing Pipeline (Non-Org Documents)
1. **File Upload**: Files are uploaded and stored securely
2. **Content Extraction**: Text content is extracted from various formats
3. **Segmentation**: Content is split into meaningful chunks
4. **Vectorization**: Chunks are converted to embeddings for semantic search
5. **Metadata Extraction**: Document metadata is extracted and stored
6. **Indexing**: Documents are indexed for fast retrieval

### Re-processing Capability

#### What is Re-processing?
Re-processing allows you to run a document through the entire processing pipeline again, updating:
- **Embeddings**: Re-generate vector representations
- **Metadata**: Update document metadata and properties
- **Segmentation**: Re-segment content with current algorithms
- **Indexing**: Re-index the document in search systems

#### When to Use Re-processing
- **Content Updates**: After editing a document's content
- **Processing Errors**: If initial processing failed or was incomplete
- **Algorithm Updates**: When processing algorithms have been improved
- **Metadata Issues**: To fix or update document metadata
- **Search Problems**: If the document isn't appearing in search results

#### How to Re-process
1. **Right-click** any document in the file tree
2. **Select "Re-process Document"** from the context menu
3. **Confirm** the action when prompted
4. **Wait** for processing to complete (progress will be shown)

### Metadata Editing Capability

#### What is Metadata Editing?
The metadata editing feature allows you to view and edit comprehensive document metadata including:
- **Title**: Document title for better identification
- **Description**: Detailed description of the document content
- **Author**: Document author or creator
- **Category**: Predefined categories for organization
- **Tags**: Custom tags for flexible categorization
- **Publication Date**: Original publication date of the document

#### How to Edit Metadata
1. **Right-click** any document in the file tree
2. **Select "Edit Metadata"** from the context menu
3. **A floating pane** will appear with all metadata fields
4. **Edit** any fields as needed
5. **Add/remove tags** using the tag management interface
6. **Click "Save Changes"** to update the document metadata

#### Metadata Features
- **Autocomplete**: Available categories and tags are suggested
- **Tag Management**: Add new tags or remove existing ones
- **Date Picker**: Easy date selection for publication dates
- **Real-time Updates**: Changes are immediately reflected in the system
- **Search Integration**: Updated metadata improves search results

### Document Editor Integration

#### CodeMirror Editor Implementation
The document editor uses **CodeMirror 6** for a full-featured editing experience with the following capabilities:

- **Rich Text Editing**: Full-featured text editor with syntax highlighting
- **Markdown Support**: Live editing for `.md` files with syntax highlighting
- **Org Mode Support**: Specialized editing for `.org` files with org-mode specific features
- **Auto-save**: Changes are automatically saved as you type
- **Frontmatter Support**: Automatic parsing and editing of YAML frontmatter in markdown files

#### Editor Features

**Syntax Highlighting:**
- Markdown syntax highlighting (headers, links, code blocks, lists, etc.)
- Org Mode syntax highlighting (headings, TODO states, checkboxes, links, properties)
- Code block support with language detection
- Dark mode and light mode themes

**Search & Replace:**
- Advanced search and replace functionality (Ctrl+F / Cmd+F)
- Persistent search - remembers last search term
- Case-sensitive and regex search options
- Search across entire document

**Code Folding (Org Mode):**
- Fold/unfold headings with `Ctrl+Shift+H` (or `Cmd+Shift+H` on Mac)
- Fold all headings with `Ctrl+Alt+H`
- Unfold all headings with `Ctrl+Alt+Shift+H`
- Fold state persists between sessions
- Supports Emacs-compatible `#+STARTUP` keywords

**Org Mode Specific Features:**

**Keyboard Shortcuts:**
- **M-RET (Alt+Enter)**: Create new list item or heading at same level
- **Ctrl+Shift+H**: Toggle fold for current heading
- **Ctrl+Shift+T**: Toggle checkbox at cursor
- **Ctrl+Alt+L**: Insert file link at cursor
- **Ctrl+Shift+M**: Refile entry at cursor (opens refile dialog)
- **Ctrl+Shift+A**: Archive entry at cursor
- **Ctrl+Shift+E**: Tag entry at cursor
- **Ctrl+Alt+H**: Fold all headings
- **Ctrl+Alt+Shift+H**: Unfold all headings

**Note:** For complete keyboard shortcut documentation, see [ORG_MODE_HOTKEYS.md](org_mode/ORG_MODE_HOTKEYS.md)

**Org Mode Capabilities:**
- **Checkbox Management**: Toggle checkboxes with `Ctrl+Shift+T`
  - Parent checkboxes auto-update when children change
  - Progress indicators auto-update when checkboxes toggle
  - Supports `- [ ]`, `- [x]`, and `- [-]` (partially done) states
- **File Links**: Insert file links with `Ctrl+Alt+L`
  - Automatically calculates relative paths
  - Opens document picker dialog
  - Inserts in format: `[[file:./path/to/file.md][Description]]`
- **Refile**: Move entries between org files with `Ctrl+Shift+M`
  - AI suggests best location
  - Works from editor or from All TODOs page
- **Archive**: Archive DONE entries with `Ctrl+Shift+A`
  - Moves to `{filename}_archive.org`
  - Preserves entry with ARCHIVE_TIME and ARCHIVE_FILE properties
- **Tagging**: Add tags to entries with `Ctrl+Shift+E`
  - Merge with existing tags or replace them
  - Supports org-mode tag syntax
- **Content Indentation**: Optional visual indentation of content to heading level
  - Enable in Org Mode Settings
  - Improves readability of nested structures

**Markdown Editor Features:**
- Frontmatter editing with visual editor
- Syntax highlighting for all markdown elements
- Code block support
- Link and image support
- List formatting (ordered and unordered)

**Chat Integration:**
- **"Prefer Editor" Toggle**: Control whether editor context is sent to chat agents
  - Located in chat sidebar header (toggle button)
  - When enabled (`prefer`): Active editor content and context is sent to agents
  - When disabled (`ignore`): Editor context is excluded from chat
  - Context-aware: Only active on documents page, automatically disabled elsewhere
  - Persists user preference across sessions
  - Enables domain-specific agent routing (e.g., electronics_agent when project_plan.md is open)

**File Operations:**
- **Create New Files**: Right-click folders to create new `.md` or `.org` files
- **Auto-save**: Changes saved automatically as you type
- **Scroll Position**: Remembers scroll position between sessions
- **Line Navigation**: Jump to specific lines or headings

#### Auto-Re-processing
When editing documents, you can manually trigger re-processing:
- **Right-click** document in file tree
- **Select "Re-process Document"** to update embeddings and metadata
- Future: Automatic re-processing on save (planned)

### Advanced Features

#### Collaborative Editing
- **Real-time Collaboration**: Multiple users can edit simultaneously
- **Change Tracking**: Track who made what changes
- **Comments**: Add comments and annotations
- **Conflict Resolution**: Handle editing conflicts gracefully

#### Editor-Chat Integration

**"Prefer Editor" Feature:**
The "Prefer Editor" toggle in the chat sidebar controls whether the active editor context is included in chat requests:

- **When Enabled (`prefer`)**:
  - Active editor content is sent to chat agents
  - Editor metadata (filename, frontmatter, document type) is included
  - Enables domain-specific agent routing:
    - `electronics_agent` when `project_plan.md` with `type: electronics` is open
    - `reference_agent` when reference documents are open
    - `org_content_agent` when org files are open
  - Agents can make context-aware edits to the open document
  - Editor-gated agents (agents that require editor context) can be routed

- **When Disabled (`ignore`)**:
  - Editor context is excluded from chat requests
  - Agents operate without document context
  - Editor-gated agents are blocked from routing

**How It Works:**
1. Open a document in the editor
2. Open or focus the chat sidebar
3. Toggle "Prefer Editor" button in chat header (on/off)
4. Your preference is saved and persists across sessions
5. Preference is context-aware: automatically disabled when not on documents page

**Technical Details:**
- Editor state is tracked via `EditorContext` React context
- Editor preference stored in `localStorage` as `userEditorPreference`
- Editor context sent via `active_editor` in gRPC requests
- Orchestrator uses editor context for intelligent agent routing

#### Integration Features
- **Chat Integration**: "Prefer Editor" toggle for context-aware chat interactions
- **Search Integration**: Direct search from within documents
- **Knowledge Graph**: Link documents to knowledge graph entities
- **Version Control**: Git-like version control for documents (planned)

#### Advanced Processing
- **Custom Processing**: User-defined processing pipelines
- **Batch Operations**: Process multiple documents at once
- **Processing Rules**: Automatic processing based on file types or content
- **Quality Assessment**: Automatic quality scoring of processed documents

## Technical Implementation

### Backend Architecture

#### Folder Service
- **Hierarchical Storage**: PostgreSQL with recursive queries
- **User Isolation**: Separate folders for each user
- **Admin Access**: Global document access for administrators
- **Performance**: Optimized queries with proper indexing

#### Document Processing
- **Parallel Processing**: Multi-threaded document processing
- **Error Handling**: Robust error handling and recovery
- **Progress Tracking**: Real-time progress updates via WebSocket
- **Resource Management**: Efficient memory and CPU usage

#### API Endpoints
```
GET    /api/folders/tree              # Get folder hierarchy
GET    /api/folders/{id}/contents     # Get folder contents
POST   /api/folders                   # Create new folder
PUT    /api/folders/{id}              # Update folder
DELETE /api/folders/{id}              # Delete folder
POST   /api/folders/default           # Create default folders
POST   /api/documents/{id}/reprocess  # Re-process document
PUT    /api/documents/{id}/metadata   # Update document metadata
GET    /api/documents/categories      # Get available categories and tags
```

### Frontend Architecture

#### React Components
- **FileTreeSidebar**: Main file tree component
- **DocumentViewer**: Main document viewing and editing component
- **OrgCMEditor**: CodeMirror-based org mode editor with org-specific features
- **MarkdownCMEditor**: CodeMirror-based markdown editor with frontmatter support
- **OrgEditorPlugins**: Org mode specific plugins (folding, checkboxes, list insertion, etc.)
- **ContextMenu**: Right-click context menus
- **UploadDialog**: File upload interface
- **ChatSidebar**: Chat interface with "Prefer Editor" toggle

#### State Management
- **React Query**: Server state management
- **Local State**: UI state management
- **WebSocket**: Real-time updates
- **Caching**: Intelligent caching for performance

#### User Experience
- **Responsive Design**: Works on all screen sizes
- **Keyboard Shortcuts**: Power user shortcuts
- **Drag & Drop**: Intuitive file operations
- **Loading States**: Clear feedback during operations

## Usage Guidelines

### Best Practices

#### Folder Organization
- **Use Descriptive Names**: Clear, meaningful folder names
- **Limit Depth**: Avoid deeply nested folder structures
- **Consistent Naming**: Use consistent naming conventions
- **Regular Cleanup**: Periodically organize and clean up folders

#### Document Management
- **Re-process After Changes**: Always re-process documents after editing
- **Use Appropriate Formats**: Choose the right format for your content
- **Add Metadata**: Include relevant tags and descriptions
- **Regular Backups**: Keep backups of important documents

#### Performance Optimization
- **Batch Operations**: Process multiple documents together
- **Monitor Resources**: Watch system resource usage
- **Clean Up**: Remove unused or duplicate documents
- **Update Regularly**: Keep the system updated

### Troubleshooting

#### Common Issues
- **Processing Failures**: Check file format and content
- **Missing Documents**: Verify folder permissions and ownership
- **Search Issues**: Re-process documents to update embeddings
- **Performance Problems**: Check system resources and database

#### Support
- **Logs**: Check application logs for detailed error information
- **Documentation**: Refer to this documentation for guidance
- **Community**: Join the community for help and support

## Future Roadmap

### Phase 2: Enhanced Editor Features (In Progress)
- **Auto-Re-processing on Save**: Automatic re-processing when documents are saved
- **Version History**: Track changes and revert to previous versions
- **Live Preview**: Real-time preview for Markdown documents
- **Auto-completion**: Intelligent suggestions and completions
- **Error Detection**: Real-time error checking and validation

### Phase 3: Advanced Editor Features
- **Rich Media Support**: Images, videos, and interactive content
- **Advanced Formatting**: Tables, charts, and diagrams
- **Plugin System**: Extensible editor with plugins
- **Mobile Support**: Full mobile editing experience
- **Export**: Export documents in various formats
- **Templates**: Pre-defined templates for common document types

### Phase 4: AI Integration
- **AI-Assisted Editing**: AI-powered writing assistance (partially implemented via chat integration)
- **Content Generation**: AI-generated content suggestions
- **Smart Organization**: AI-powered document organization
- **Intelligent Search**: AI-enhanced search capabilities

### Phase 5: Enterprise Features
- **Advanced Permissions**: Granular access control
- **Audit Logging**: Comprehensive activity logging
- **Integration APIs**: Third-party system integration
- **Scalability**: Enterprise-grade performance and reliability
- **Collaborative Editing**: Real-time collaboration with multiple users
- **Change Tracking**: Track who made what changes
- **Comments**: Add comments and annotations

---

*This documentation is maintained as part of the Plato Knowledge Base project. For questions or contributions, please refer to the project repository.* 