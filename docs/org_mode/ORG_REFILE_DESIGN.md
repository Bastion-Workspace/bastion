# Org-Mode Refile Feature Design

**BULLY!** Roosevelt's "Organize the Cavalry" - Move tasks from inbox to proper project locations!

## Overview

**Refile** is the org-mode feature for moving entries (TODOs, headings, notes) from one location to another, typically from `inbox.org` to organized project files.

## Core Workflow

```
inbox.org
├─ * TODO Review quarterly reports :@work:
├─ * TODO Buy chocolate :@errands:
└─ * TODO Research machine learning :@personal:

[User reviews inbox and refiles]

projects.org
└─ * Work Projects
    └─ * TODO Review quarterly reports :@work:

errands.org  
└─ * TODO Buy chocolate :@errands:

learning.org
└─ * Personal Learning
    └─ * TODO Research machine learning :@personal:
```

## Required Components

### 1. List Refile Targets
**Endpoint:** `GET /api/org/refile-targets`

Returns available refile destinations:
```json
{
  "targets": [
    {
      "type": "file",
      "path": "Users/admin/OrgMode/projects.org",
      "display": "projects.org (root)",
      "description": "Top-level in projects.org"
    },
    {
      "type": "heading",
      "path": "Users/admin/OrgMode/projects.org",
      "heading": "Work Projects",
      "level": 1,
      "display": "projects.org > Work Projects",
      "description": "Under Work Projects heading"
    },
    {
      "type": "heading",
      "path": "Users/admin/OrgMode/learning.org",
      "heading": "Personal Learning",
      "level": 1,
      "display": "learning.org > Personal Learning",
      "description": "Under Personal Learning heading"
    }
  ]
}
```

### 2. Refile Entry
**Endpoint:** `POST /api/org/refile`

```json
{
  "source_file": "inbox.org",
  "source_line": 15,
  "target_file": "projects.org",
  "target_heading": "Work Projects",  // Optional
  "target_position": "last"  // "first" or "last" under heading
}
```

**Response:**
```json
{
  "success": true,
  "message": "Refiled TODO to projects.org > Work Projects",
  "source_updated": true,
  "target_updated": true,
  "entry_preview": "* TODO Review quarterly reports :@work:"
}
```

### 3. Refile Service
**File:** `backend/services/org_refile_service.py`

**Key methods:**
- `get_refile_targets(user_id)` - Find all org files and headings
- `refile_entry(source_file, line_number, target_file, target_heading)` - Move entry
- `parse_org_structure(file_path)` - Extract headings for targeting

### 4. Refile Tools for Agents
**File:** `backend/services/langgraph_tools/org_refile_tools.py`

**Tools:**
- `list_refile_targets()` - Get available destinations
- `refile_entry()` - Move an entry
- `suggest_refile_target()` - AI suggests best location based on content/tags

### 5. UI Components

#### Quick Refile Modal
**Trigger:** Click refile icon next to TODO in inbox view

**Interface:**
```
┌─────────────────────────────────────────┐
│ Refile: TODO Review quarterly reports  │
├─────────────────────────────────────────┤
│ Select destination:                     │
│                                         │
│ ○ projects.org (root)                   │
│ ○ projects.org > Work Projects          │
│ ○ projects.org > Personal Projects      │
│ ○ learning.org (root)                   │
│ ○ learning.org > Personal Learning      │
│                                         │
│ [🤖 AI Suggest] [Cancel] [Refile]      │
└─────────────────────────────────────────┘
```

#### Inbox Review Mode
**Interface:** Dedicated inbox review screen with:
- List of inbox items
- Refile button for each
- Keyboard shortcuts: `r` to refile current item
- Bulk refile with checkboxes

## Implementation Details

### Backend: Refile Service

```python
class OrgRefileService:
    """Service for refiling org-mode entries"""
    
    async def get_refile_targets(self, user_id: str) -> List[Dict]:
        """
        Get all possible refile targets
        Returns files + headings within files
        """
        org_files = await self._find_user_org_files(user_id)
        targets = []
        
        for file in org_files:
            # Add file root as target
            targets.append({
                "type": "file",
                "path": str(file),
                "display": f"{file.name} (root)"
            })
            
            # Parse headings as targets
            headings = await self._parse_org_headings(file)
            for heading in headings:
                targets.append({
                    "type": "heading",
                    "path": str(file),
                    "heading": heading["title"],
                    "level": heading["level"],
                    "display": f"{file.name} > {heading['title']}"
                })
        
        return targets
    
    async def refile_entry(
        self,
        user_id: str,
        source_file: str,
        line_number: int,
        target_file: str,
        target_heading: Optional[str] = None,
        position: str = "last"
    ) -> Dict:
        """
        Move an org entry from source to target
        
        Steps:
        1. Extract entry from source file (with properties, content)
        2. Remove from source file
        3. Insert into target file at appropriate location
        4. Return success status
        """
        # Extract entry (heading + all sub-content until next heading)
        entry = await self._extract_entry(source_file, line_number)
        
        # Remove from source
        await self._remove_entry(source_file, line_number)
        
        # Insert into target
        if target_heading:
            await self._insert_under_heading(
                target_file, target_heading, entry, position
            )
        else:
            await self._append_to_file(target_file, entry)
        
        return {
            "success": True,
            "entry_preview": entry.split("\n")[0]
        }
```

### Agent Integration

**Org Inbox Agent gets new capabilities:**
```python
# In org_inbox_agent.py

async def process(self, state):
    operation = state.get("org_inbox_operation")
    
    if operation == "refile":
        # User: "Refile task 3 to projects.org"
        task_id = state.get("task_id")
        target = state.get("refile_target")
        
        # Get refile targets
        targets = await org_refile_list_targets(user_id)
        
        # Match target
        matched_target = self._find_best_target(target, targets)
        
        # Refile
        result = await org_refile_entry(
            source_file="inbox.org",
            line_number=task_id,
            target_file=matched_target["path"],
            target_heading=matched_target.get("heading")
        )
        
        return result
```

**Research Agent gets smart suggest:**
```python
# Research agent can suggest refile locations
"Based on the task content and tags, I recommend refiling to:
projects.org > Work Projects (matches @work tag)"
```

## UI/UX Features

### 1. Inbox View
- Dedicated "Inbox" section in sidebar
- Shows all inbox.org TODOs
- Refile button on each item
- Keyboard shortcut: `r` to refile focused item

### 2. Refile Dialog
- Search/filter destinations
- AI-powered suggestions based on:
  - Task tags
  - Task content
  - Historical refile patterns
- Preview destination file structure
- Keyboard navigation

### 3. Bulk Operations
- Select multiple inbox items
- Refile all to same destination
- "Process Inbox" mode for review workflow

### 4. Drag-and-Drop (Future)
- Drag TODO from inbox to project file in sidebar
- Visual feedback during drag
- Drop on file or specific heading

## Agent Commands

```
User: "Refile task 3 to work projects"
Agent: Moves task from inbox to projects.org > Work Projects

User: "Organize my inbox"
Agent: Suggests refile targets for each item based on tags/content

User: "Refile all @work tasks to projects.org"
Agent: Batch refiles filtered items

User: "Where should I refile this machine learning task?"
Agent: Suggests learning.org > Personal Learning based on content
```

## Advanced Features

### 1. Smart Suggestions
```python
def suggest_refile_target(entry: Dict) -> str:
    """
    AI suggests best refile location based on:
    - Tags (e.g., @work → work projects)
    - Keywords in content
    - Historical patterns
    - Related existing tasks
    """
    # Use LLM to analyze entry and suggest target
```

### 2. Refile Rules
**User-configurable patterns:**
```json
{
  "refile_rules": [
    {
      "condition": {"tag": "@work"},
      "target": "projects.org > Work Projects"
    },
    {
      "condition": {"tag": "@errands"},
      "target": "errands.org"
    },
    {
      "condition": {"keyword": "learning|study|research"},
      "target": "learning.org > Personal Learning"
    }
  ]
}
```

### 3. Refile History
- Track where items are refiled
- Suggest based on previous patterns
- "Undo refile" functionality

## Database Schema

```sql
-- Track refile operations for history/patterns
CREATE TABLE org_refile_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    source_file VARCHAR(500) NOT NULL,
    source_line INTEGER NOT NULL,
    target_file VARCHAR(500) NOT NULL,
    target_heading VARCHAR(500),
    entry_text TEXT,
    tags TEXT[],
    refiled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_refile_user ON org_refile_history(user_id);
CREATE INDEX idx_refile_tags ON org_refile_history USING GIN(tags);
```

## Implementation Priority

### Phase 1: Core Refile (MVP)
1. ✅ `OrgRefileService` - backend service
2. ✅ `GET /api/org/refile-targets` - list destinations
3. ✅ `POST /api/org/refile` - move entries
4. ✅ Basic UI refile dialog
5. ✅ Org inbox agent refile command support

### Phase 2: Enhanced Experience
1. ✅ Smart AI suggestions
2. ✅ Refile rules configuration
3. ✅ Bulk refile operations
4. ✅ Keyboard shortcuts
5. ✅ Refile history tracking

### Phase 3: Advanced Features
1. ✅ Drag-and-drop refile
2. ✅ Process inbox workflow mode
3. ✅ Undo refile
4. ✅ Cross-file search during refile target selection

## Other Org-Mode Features to Consider

After refile, these are the next most valuable:

### 1. **Archive** (High Priority)
- Move DONE items to archive files
- Keep active org files clean
- Searchable archive

### 2. **Agenda View** (High Priority)
- Calendar view of scheduled/deadline items
- Daily/weekly/monthly views
- Across all org files

### 3. **Clocking** (Medium Priority)
- Time tracking on tasks
- Clock in/out
- Reports

### 4. **Repeating Tasks** (Medium Priority - Partial)
- Daily/weekly/monthly repeats
- We have basic repeater support via `org_inbox_set_schedule_and_repeater`

### 5. **Custom TODO Keywords** (Medium Priority - Partial)
- We have TODO state sequences in settings
- Need UI for managing them

### 6. **Properties Drawer** (Low Priority)
- Currently capture uses simple timestamps
- Could add full properties support

## Why Refile is Critical

**Without refile:**
- Inbox.org becomes a dumping ground
- Hard to organize captured items
- No clear workflow for processing

**With refile:**
- ✅ Clean, organized project files
- ✅ GTD workflow support
- ✅ Easy review and processing
- ✅ Context-based organization

**By George!** Refile completes the capture → organize → process workflow! 🏇



