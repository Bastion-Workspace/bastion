# Org-Mode Agent Routing Strategy

**BULLY!** Clear separation of responsibilities for org-mode queries!

## The Three-Agent Approach

### 1. **Org Inbox Agent** - The Task Manager
**Responsibility:** MODIFY inbox.org (write operations only)

**Routes to this agent when user wants to:**
- ✅ Add new TODO items
- ✅ Mark tasks as done/toggle completion
- ✅ Update task text
- ✅ Change TODO states (TODO → DONE, etc.)
- ✅ Apply tags to tasks
- ✅ Set schedules and deadlines

**Example queries:**
```
"Add TODO: Buy chocolate"
"Mark task 3 as done"
"Toggle the status of my meeting task"
"Update my quarterly report task"
"Change task 5 to WAITING"
```

**Tools available:**
- `org_inbox_add_item`
- `org_inbox_toggle_done`
- `org_inbox_update_line`
- `org_inbox_set_state`
- `org_inbox_apply_tags`
- `org_inbox_set_schedule_and_repeater`

**Key characteristic:** **Write operations** - modifies inbox.org

---

### 2. **Chat Agent** - The Quick Lookup
**Responsibility:** READ org data for casual queries (simple/quick operations)

**Routes to this agent when user wants to:**
- ✅ List current TODOs
- ✅ Quick tag-based lookups
- ✅ "What's on my list?" type questions
- ✅ Simple status checks

**Example queries:**
```
"What's on my TODO list?"
"What tasks are tagged @work?"
"Show me my tasks"
"Do I have any errands?"
```

**Tools available:**
- `list_org_todos` - Lists all TODO items with optional filtering
- `search_org_by_tag` - Quick search by a single tag

**Key characteristic:** **Simple read operations** - quick answers without deep search

---

### 3. **Research Agent** - The Deep Searcher
**Responsibility:** SEARCH all org files for reference material and research

**Routes to this agent when user wants to:**
- ✅ Find notes/content in org files
- ✅ Search for specific topics across all .org files
- ✅ Complex filtered searches (e.g., "WAITING tasks related to budget")
- ✅ Research historical org content
- ✅ Deep dives into project notes

**Example queries:**
```
"Find my notes about the quarterly report"
"Search my org files for anything related to project Alpha"
"What did I write about machine learning in my org files?"
"Show me all WAITING tasks that mention 'budget'"
"What org entries are tagged both @work and urgent?"
```

**Tools available:**
- `search_org_files` - Full-text search with tag/state filtering across ALL .org files
- `list_org_todos` - Same as chat agent (for comprehensive TODO searches)
- `search_org_by_tag` - Same as chat agent (for research context)

**Key characteristic:** **Complex read operations** - deep search across reference files

---

## Routing Decision Tree

```
User query about org-mode
│
├─ Does it MODIFY inbox.org?
│  ├─ YES: Add, toggle, update, change state
│  │       → ORG_INBOX_AGENT
│  │
│  └─ NO: It's a read operation
│         │
│         ├─ Is it a SIMPLE list/lookup?
│         │  ├─ YES: "What's on my list?", "What's tagged X?"
│         │  │       → CHAT_AGENT (quick lookup)
│         │  │
│         │  └─ NO: It's a search/research query
│         │          │
│         │          └─ "Find notes about X", "Search for Y"
│         │              → RESEARCH_AGENT (deep search)
```

## Key Distinctions

### Write vs Read
- **Write (modify)** → `org_inbox_agent`
- **Read (query)** → `chat_agent` or `research_agent`

### Simple vs Complex Read
- **Simple read** (list, quick tag lookup) → `chat_agent`
- **Complex read** (search, filters, research) → `research_agent`

### Inbox vs All Org Files
- **Inbox.org only** (management) → `org_inbox_agent`
- **All .org files** (search/reference) → `research_agent` or `chat_agent`

## Why This Separation?

1. **Performance**: Chat agent handles quick queries without spinning up research tools
2. **Specialization**: Each agent optimized for its specific use case
3. **Tool access**: Agents only get tools they need (principle of least privilege)
4. **Clear intent**: Reduces ambiguity in routing decisions
5. **User experience**: Right agent = right response speed and depth

## Edge Cases

### "What's on my TODO list?"
**Route to:** `chat_agent`
**Reason:** Simple list operation, not research

### "Find all TODO items related to quarterly report"
**Route to:** `research_agent`
**Reason:** Search across content, not just listing

### "Add a TODO to research quarterly report"
**Route to:** `org_inbox_agent`
**Reason:** Modification operation (adding TODO)

### "What did I write about quarterly report in my org files?"
**Route to:** `research_agent`
**Reason:** Deep search for reference material

## Intent Classifier Prompt Guidance

The intent classifier (`backend/services/simple_intent_service.py`) now includes:

1. **Agent capability descriptions** with org-mode tool mentions
2. **Explicit org-mode routing rules** with examples for each agent type
3. **Decision keywords** to help LLM classify intent correctly

**Key prompt sections:**
- Agent capabilities list (lines 151-158)
- Org-mode routing rules (lines 160-174)
- Example mappings for each scenario

## Testing Routing

To verify correct routing, test these queries:

```python
# Should route to org_inbox_agent
"Add TODO: Buy milk"
"Mark task 5 as done"
"Change my meeting task to WAITING"

# Should route to chat_agent  
"What's on my TODO list?"
"What tasks are tagged @work?"
"Show me my errands"

# Should route to research_agent
"Find my notes about the quarterly report"
"Search org files for machine learning topics"
"Show me all WAITING tasks mentioning budget"
```

## Future Enhancements

- **Org Project Agent**: For project-level org-mode operations (separate from inbox)
- **Org Agenda Agent**: For scheduling and calendar operations across org files
- **Hybrid queries**: "Add TODO based on my notes about X" (research → then add)

**By George!** With this routing strategy, each agent knows its role and executes it perfectly! 🏇



