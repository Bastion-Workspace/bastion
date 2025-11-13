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

#### Auto-Re-processing (Future Feature)
When the document editor is implemented, saving a file will automatically trigger re-processing:
- **Auto-save**: Changes are automatically saved and re-processed
- **Manual Save**: Ctrl+S will save and re-process the document
- **Background Processing**: Re-processing happens in the background
- **Progress Indication**: Visual feedback during re-processing

## Planned Features (Phase 2)

### Document Editor Integration

#### Monaco Editor Integration
- **Rich Text Editing**: Full-featured text editor with syntax highlighting
- **Markdown Support**: Live preview and editing for `.md` files
- **Org Mode Support**: Specialized editing for `.org` files
- **Auto-save**: Automatic saving of changes
- **Version History**: Track changes and revert to previous versions

#### Editor Features
- **Syntax Highlighting**: 
  - Markdown syntax highlighting
  - Org Mode syntax highlighting
  - Code block support
- **Live Preview**: Real-time preview for Markdown documents
- **Auto-completion**: Intelligent suggestions and completions
- **Error Detection**: Real-time error checking and validation
- **Search & Replace**: Advanced search and replace functionality

#### File Operations
- **Create New Files**: Right-click folders to create new `.md` or `.org` files
- **Save As**: Save documents with different names or formats
- **Export**: Export documents in various formats
- **Templates**: Pre-defined templates for common document types

### Advanced Features

#### Collaborative Editing
- **Real-time Collaboration**: Multiple users can edit simultaneously
- **Change Tracking**: Track who made what changes
- **Comments**: Add comments and annotations
- **Conflict Resolution**: Handle editing conflicts gracefully

#### Integration Features
- **Chat Integration**: "Save as Note" functionality from chat
- **Search Integration**: Direct search from within documents
- **Knowledge Graph**: Link documents to knowledge graph entities
- **Version Control**: Git-like version control for documents

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
- **DocumentEditor**: Monaco editor integration (planned)
- **ContextMenu**: Right-click context menus
- **UploadDialog**: File upload interface

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

### Phase 3: Advanced Editor Features
- **Rich Media Support**: Images, videos, and interactive content
- **Advanced Formatting**: Tables, charts, and diagrams
- **Plugin System**: Extensible editor with plugins
- **Mobile Support**: Full mobile editing experience

### Phase 4: AI Integration
- **AI-Assisted Editing**: AI-powered writing assistance
- **Content Generation**: AI-generated content suggestions
- **Smart Organization**: AI-powered document organization
- **Intelligent Search**: AI-enhanced search capabilities

### Phase 5: Enterprise Features
- **Advanced Permissions**: Granular access control
- **Audit Logging**: Comprehensive activity logging
- **Integration APIs**: Third-party system integration
- **Scalability**: Enterprise-grade performance and reliability

---

*This documentation is maintained as part of the Plato Knowledge Base project. For questions or contributions, please refer to the project repository.* 