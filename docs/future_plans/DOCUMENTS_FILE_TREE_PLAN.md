# Documents File Tree & Editor Implementation Plan

## 🎯 Vision: Hierarchical Document Management with Integrated Editor

Transform the current flat document listing into a powerful file tree sidebar with integrated text editing capabilities, supporting both document organization and content creation.

## 🏗️ Current State Analysis

### **Existing Infrastructure:**
- ✅ Document upload and processing (PDF, TXT, DOCX, EPUB, HTML, ZIP, EML, SRT)
- ✅ User/Global document separation (admin vs user collections)
- ✅ "Save as Note" functionality from chat
- ✅ Document metadata management (categories, tags, descriptions)
- ✅ Vectorization and search capabilities
- ✅ File deduplication and processing status tracking

### **Current Documents Page:**
- 📄 Flat table-based document listing
- 🔄 Upload via drag-and-drop or file picker
- 📊 Search, filter, and sort functionality
- 👁️ Document preview and metadata editing
- 📁 ZIP hierarchy view for nested content

## 🚀 Target Architecture

### **New Layout Structure:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Top Navigation                                 │
├─────────────────┬───────────────────────────────────┬─────────────────────┤
│   File Tree     │          Document Editor/Viewer   │   Context Panel     │
│   Sidebar       │                                   │   (Future)          │
│                 │                                   │                     │
│  📁 My Documents│  ╔══════════════════════════════╗ │                     │
│  ├── 📁 Notes   │  ║                              ║ │  💬 Context Chat   │
│  │   ├── 📄 Note│  ║        Markdown Editor       ║ │                     │
│  │   └── 📄 Note│  ║        or Org Editor         ║ │                     │
│  ├── 📁 Projects│  ║                              ║ │  📋 Document Info   │
│  │   └── 📄 Doc │  ║        Live Preview          ║ │                     │
│  └── 📁 Archive │  ║                              ║ │  🏷️ Tags & Meta     │
│                 │  ╚══════════════════════════════╝ │                     │
│  📁 Global Docs │                                   │                     │
│  (Admin Only)   │                                   │                     │
└─────────────────┴───────────────────────────────────┴─────────────────────┘
```

## 📋 Implementation Phases

### **Phase 1: File Tree Sidebar Foundation**
**Duration: 2-3 days**

#### **1.1 New Document Types Support**
- Add `MD` and `ORG` to `DocumentType` enum in `backend/models/api_models.py`
- Update document processing pipeline to handle `.md` and `.org` files
- Add text extraction for these formats (direct content reading)

#### **1.2 Folder Structure Backend**
- Create new database table: `document_folders`
  ```sql
  CREATE TABLE document_folders (
    folder_id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    parent_folder_id UUID REFERENCES document_folders(folder_id),
    user_id UUID REFERENCES users(user_id),
    collection_type VARCHAR(20) DEFAULT 'user', -- 'user' or 'global'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
  );
  ```
- Add `folder_id` column to `documents` table
- Create `FolderService` for CRUD operations
- Add folder endpoints to `main.py`

#### **1.3 File Tree Component**
- Create `FileTreeSidebar.js` component
- Implement hierarchical folder display with expand/collapse
- Add right-click context menu for folders
- Support drag-and-drop for file organization

### **Phase 2: Document Editor Integration**
**Duration: 3-4 days**

#### **2.1 Editor Component**
- Create `DocumentEditor.js` component
- Integrate Monaco Editor or CodeMirror for rich text editing
- Support Markdown and Org Mode syntax highlighting
- Implement live preview for Markdown
- Add auto-save functionality

#### **2.2 Document Creation & Management**
- Right-click "New File" functionality (.md/.org)
- File creation with default templates
- Save/rename/delete operations
- Version history tracking

#### **2.3 Upload Integration**
- Right-click "Upload" on folders
- Maintain current upload processing pipeline
- Support all existing file types
- Process new .md files through vectorization
- Process new .org files for structured access (no vectorization)

### **Phase 3: Enhanced User Experience**
**Duration: 2-3 days**

#### **3.1 Default Folder Structure**
- Auto-create "Notes" folder in "My Documents" for new users
- Migrate existing "Save as Note" functionality to use Notes folder
- Create default templates for new files

#### **3.2 Search & Navigation**
- Tree-based search functionality
- Breadcrumb navigation
- Recent files list
- Quick access to frequently used folders

#### **3.3 Context Menus & Actions**
- Right-click context menus for files and folders
- Bulk operations (move, delete, tag)
- Export functionality
- Share/collaboration features (future)

## 🛠️ Technical Implementation Details

### **Backend Changes**

#### **New Models (`backend/models/api_models.py`)**
```python
class DocumentFolder(BaseModel):
    folder_id: str = Field(..., description="Unique folder identifier")
    name: str = Field(..., description="Folder name")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")
    user_id: Optional[str] = Field(None, description="Owner user ID")
    collection_type: str = Field(default="user", description="Collection type")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

class FolderCreateRequest(BaseModel):
    name: str = Field(..., description="Folder name")
    parent_folder_id: Optional[str] = Field(None, description="Parent folder ID")

class DocumentCreateRequest(BaseModel):
    filename: str = Field(..., description="Document filename")
    content: str = Field(..., description="Initial content")
    folder_id: Optional[str] = Field(None, description="Target folder ID")
    doc_type: str = Field(..., description="Document type (md/org)")
```

#### **New Service (`backend/services/folder_service.py`)**
```python
class FolderService:
    async def create_folder(self, name: str, parent_folder_id: str = None, user_id: str = None) -> DocumentFolder
    async def get_folder_tree(self, user_id: str = None, collection_type: str = "user") -> List[DocumentFolder]
    async def get_folder_contents(self, folder_id: str) -> Dict[str, Any]
    async def move_folder(self, folder_id: str, new_parent_id: str = None) -> bool
    async def delete_folder(self, folder_id: str, recursive: bool = False) -> bool
```

#### **Updated Document Service**
- Add folder support to existing document operations
- Update upload processing to handle .md/.org files
- Add text-based document creation endpoints

### **Frontend Changes**

#### **New Components**
1. **`FileTreeSidebar.js`** - Main tree component
2. **`DocumentEditor.js`** - Text editor with preview
3. **`FolderContextMenu.js`** - Right-click menu for folders
4. **`FileContextMenu.js`** - Right-click menu for files
5. **`BreadcrumbNavigation.js`** - Path navigation

#### **Updated Components**
1. **`DocumentsPage.js`** - Convert to new layout
2. **`apiService.js`** - Add folder and editor endpoints

#### **New Routes**
- `/documents/editor/:documentId` - Document editor
- `/documents/folder/:folderId` - Folder view

### **Database Schema Updates**

#### **New Tables**
```sql
-- Document folders table
CREATE TABLE document_folders (
    folder_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    parent_folder_id UUID REFERENCES document_folders(folder_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    collection_type VARCHAR(20) DEFAULT 'user' CHECK (collection_type IN ('user', 'global')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_document_folders_user_id ON document_folders(user_id);
CREATE INDEX idx_document_folders_parent_id ON document_folders(parent_folder_id);
CREATE INDEX idx_document_folders_collection ON document_folders(collection_type);
```

#### **Updated Tables**
```sql
-- Add folder_id to documents table
ALTER TABLE documents ADD COLUMN folder_id UUID REFERENCES document_folders(folder_id) ON DELETE SET NULL;
CREATE INDEX idx_documents_folder_id ON documents(folder_id);
```

## 🎨 User Experience Design

### **File Tree Sidebar Features**
- **Hierarchical Display**: Nested folder structure with expand/collapse
- **Visual Indicators**: Icons for different file types and folders
- **Context Menus**: Right-click for create, upload, rename, delete
- **Drag & Drop**: Move files between folders
- **Search**: Filter tree by name
- **Breadcrumbs**: Show current path

### **Editor Features**
- **Syntax Highlighting**: Markdown and Org Mode support
- **Live Preview**: Real-time Markdown rendering
- **Auto-save**: Automatic content saving
- **Version History**: Track document changes
- **Full-screen Mode**: Distraction-free editing
- **Split View**: Editor and preview side-by-side

### **Integration Points**
- **Chat Integration**: "Save as Note" creates files in Notes folder
- **Search**: Include file tree in global search
- **Navigation**: Seamless switching between tree and editor
- **Responsive**: Mobile-friendly tree collapse

## 🔄 Migration Strategy

### **Data Migration**
1. **Create Default Structure**: Auto-create "My Documents" and "Notes" folders for existing users
2. **Migrate Existing Documents**: Move existing documents to appropriate folders
3. **Preserve Metadata**: Maintain all existing document metadata and relationships
4. **Backup Strategy**: Create backup before migration

### **User Experience Migration**
1. **Gradual Rollout**: Deploy tree view alongside existing table view
2. **User Preference**: Allow users to switch between views
3. **Tutorial**: Provide onboarding for new tree interface
4. **Feedback Loop**: Collect user feedback and iterate

## 🧪 Testing Strategy

### **Backend Testing**
- Folder CRUD operations
- Document creation and editing
- File upload to folders
- Permission validation
- Data integrity checks

### **Frontend Testing**
- Tree navigation and interaction
- Editor functionality
- Context menu operations
- Drag and drop behavior
- Responsive design

### **Integration Testing**
- End-to-end document workflows
- Chat integration ("Save as Note")
- Search functionality
- Performance under load

## 📊 Success Metrics

### **User Adoption**
- Percentage of users using tree view vs table view
- Time spent in editor vs viewer
- Number of new documents created
- Folder organization patterns

### **Performance Metrics**
- Tree load time
- Editor responsiveness
- Search performance
- Upload processing time

### **Quality Metrics**
- User satisfaction scores
- Support ticket reduction
- Feature usage analytics
- Error rates

## 🚀 Future Enhancements

### **Phase 4: Advanced Features**
- **Collaboration**: Real-time collaborative editing
- **Templates**: Pre-built document templates
- **Plugins**: Editor extensions and plugins
- **AI Integration**: Smart content suggestions
- **Version Control**: Git-like versioning

### **Phase 5: Enterprise Features**
- **Permissions**: Granular folder and file permissions
- **Workflows**: Document approval workflows
- **Integration**: Third-party tool integrations
- **Analytics**: Advanced usage analytics
- **Backup**: Automated backup and recovery

## 📝 Implementation Checklist

### **Phase 1 Checklist**
- [ ] Add MD/ORG to DocumentType enum
- [ ] Create document_folders table
- [ ] Implement FolderService
- [ ] Add folder endpoints to main.py
- [ ] Create FileTreeSidebar component
- [ ] Add folder context menus

### **Phase 2 Checklist**
- [ ] Create DocumentEditor component
- [ ] Integrate Monaco/CodeMirror editor
- [ ] Implement Markdown preview
- [ ] Add file creation functionality
- [ ] Update upload processing for .md/.org
- [ ] Add auto-save functionality

### **Phase 3 Checklist**
- [ ] Create default folder structure
- [ ] Migrate "Save as Note" to Notes folder
- [ ] Implement tree search
- [ ] Add breadcrumb navigation
- [ ] Create file context menus
- [ ] Add bulk operations

### **Testing Checklist**
- [ ] Backend unit tests
- [ ] Frontend component tests
- [ ] Integration tests
- [ ] Performance tests
- [ ] User acceptance testing

## 🎯 Success Criteria

1. **Functional**: All existing document features work with new tree structure
2. **Performance**: Tree navigation is responsive (< 100ms interactions)
3. **Usability**: Users can easily organize and edit documents
4. **Integration**: "Save as Note" seamlessly creates files in Notes folder
5. **Scalability**: System handles large numbers of folders and files
6. **Accessibility**: Tree and editor are keyboard and screen reader accessible

This plan provides a comprehensive roadmap for transforming the documents page into a powerful file tree sidebar with integrated editing capabilities, while maintaining all existing functionality and ensuring a smooth user experience transition. 