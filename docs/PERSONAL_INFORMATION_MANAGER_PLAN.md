# Personal Information Manager (PIM) Implementation Plan

## 🎯 Vision: Intelligent Calendar & Contacts Integration

Transform the existing knowledge base into a comprehensive Personal Information Manager that intelligently connects calendar events, contacts, documents, and notes through natural language queries.

## 🏗️ Architecture Overview

### Current Infrastructure Strengths
- ✅ **User Isolation System** - Perfect for per-user PIM data
- ✅ **PostgreSQL Database** - Ready for structured calendar/contact data
- ✅ **Knowledge Graph (Neo4j)** - Ideal for relationship mapping
- ✅ **MCP Tools System** - Extensible for new data types
- ✅ **Natural Language Processing** - Ready for intelligent queries
- ✅ **LLM Integration** - Can synthesize cross-referenced information

### New Components Needed
- 📅 **Calendar Database Schema** - Events, recurrence, reminders
- 👥 **Contacts Database Schema** - People, organizations, relationships
- 🔗 **Knowledge Graph Extensions** - Calendar/contact entity types
- 🛠️ **New MCP Tools** - Calendar and contact query tools
- 🎨 **UI Components** - Calendar and contact management interfaces

## 📊 Database Schema Design

### Calendar Tables

```sql
-- Calendar Events
CREATE TABLE calendar_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    location VARCHAR(500),
    start_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    end_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
    all_day BOOLEAN DEFAULT FALSE,
    timezone VARCHAR(100) DEFAULT 'UTC',
    
    -- Recurrence
    recurrence_rule TEXT, -- RRULE format
    recurrence_parent_id UUID REFERENCES calendar_events(event_id),
    
    -- Categories and Status
    category VARCHAR(100),
    status VARCHAR(50) DEFAULT 'confirmed', -- confirmed, tentative, cancelled
    priority INTEGER DEFAULT 0, -- 0=none, 1=low, 5=normal, 9=high
    
    -- Attendees and Organizer
    organizer_email VARCHAR(255),
    organizer_name VARCHAR(255),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE, -- Soft delete
    
    -- Search and Integration
    search_vector tsvector,
    metadata_json JSONB DEFAULT '{}',
    
    CONSTRAINT valid_datetime CHECK (end_datetime >= start_datetime)
);

-- Event Attendees
CREATE TABLE calendar_event_attendees (
    id SERIAL PRIMARY KEY,
    event_id UUID NOT NULL REFERENCES calendar_events(event_id) ON DELETE CASCADE,
    contact_id UUID REFERENCES contacts(contact_id) ON DELETE SET NULL,
    email VARCHAR(255),
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'attendee', -- organizer, attendee, optional
    status VARCHAR(50) DEFAULT 'needs-action', -- accepted, declined, tentative, needs-action
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Event Reminders
CREATE TABLE calendar_reminders (
    reminder_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id UUID NOT NULL REFERENCES calendar_events(event_id) ON DELETE CASCADE,
    reminder_minutes INTEGER NOT NULL, -- Minutes before event
    reminder_method VARCHAR(50) DEFAULT 'popup', -- popup, email, sms
    triggered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Contacts Tables

```sql
-- Main Contacts
CREATE TABLE contacts (
    contact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Basic Information
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    display_name VARCHAR(500), -- Computed or custom display name
    nickname VARCHAR(255),
    
    -- Contact Methods
    primary_email VARCHAR(255),
    primary_phone VARCHAR(50),
    
    -- Organization
    company VARCHAR(255),
    job_title VARCHAR(255),
    department VARCHAR(255),
    
    -- Personal
    birthday DATE,
    anniversary DATE,
    spouse_name VARCHAR(255),
    
    -- Social/Web
    website VARCHAR(500),
    linkedin_url VARCHAR(500),
    twitter_handle VARCHAR(100),
    
    -- Physical Address (primary)
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    
    -- Metadata
    notes TEXT,
    tags TEXT[], -- Array of tags
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE, -- Soft delete
    
    -- Search
    search_vector tsvector,
    metadata_json JSONB DEFAULT '{}',
    
    -- Constraints
    CONSTRAINT contact_has_name CHECK (
        first_name IS NOT NULL OR last_name IS NOT NULL OR display_name IS NOT NULL
    )
);

-- Additional Contact Methods
CREATE TABLE contact_methods (
    method_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID NOT NULL REFERENCES contacts(contact_id) ON DELETE CASCADE,
    method_type VARCHAR(50) NOT NULL, -- email, phone, address, social
    method_label VARCHAR(100), -- home, work, mobile, etc.
    method_value VARCHAR(500) NOT NULL,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Contact Groups/Categories
CREATE TABLE contact_groups (
    group_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_name VARCHAR(255) NOT NULL,
    group_description TEXT,
    color VARCHAR(7), -- Hex color code
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Contact Group Memberships
CREATE TABLE contact_group_members (
    contact_id UUID NOT NULL REFERENCES contacts(contact_id) ON DELETE CASCADE,
    group_id UUID NOT NULL REFERENCES contact_groups(group_id) ON DELETE CASCADE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (contact_id, group_id)
);
```

### Search Indexes

```sql
-- Calendar search indexes
CREATE INDEX idx_calendar_events_user_datetime ON calendar_events(user_id, start_datetime);
CREATE INDEX idx_calendar_events_search ON calendar_events USING gin(search_vector);
CREATE INDEX idx_calendar_events_deleted ON calendar_events(user_id, deleted_at) WHERE deleted_at IS NULL;

-- Contacts search indexes  
CREATE INDEX idx_contacts_user ON contacts(user_id) WHERE deleted_at IS NULL;
CREATE INDEX idx_contacts_search ON contacts USING gin(search_vector);
CREATE INDEX idx_contacts_name ON contacts(user_id, first_name, last_name) WHERE deleted_at IS NULL;
CREATE INDEX idx_contacts_email ON contact_methods(method_value) WHERE method_type = 'email';
```

## 🔗 Knowledge Graph Integration

### New Entity Types

```cypher
// Calendar Event Entities
(:CalendarEvent {
    id: "event_uuid",
    user_id: "user_123", 
    title: "Meeting with Jerome Butler",
    start_date: "2024-01-15",
    location: "Conference Room A",
    description: "Quarterly review discussion"
})

// Contact Entities  
(:Contact {
    id: "contact_uuid",
    user_id: "user_123",
    name: "Jerome Butler", 
    email: "jerome@company.com",
    company: "Tech Corp",
    title: "Senior Developer"
})

// Document/Note Entities (existing, enhanced)
(:Document {
    id: "doc_uuid",
    user_id: "user_123",
    title: "Jerome Butler Project Notes",
    content_snippet: "Discussion with Jerome about new features..."
})
```

### Relationship Types

```cypher
// Calendar-Contact Relationships
(:CalendarEvent)-[:INVOLVES]->(:Contact)
(:CalendarEvent)-[:ORGANIZED_BY]->(:Contact)
(:CalendarEvent)-[:ATTENDED_BY]->(:Contact)

// Contact-Document Relationships  
(:Contact)-[:MENTIONED_IN]->(:Document)
(:Contact)-[:AUTHORED]->(:Document)
(:Document)-[:REFERENCES]->(:Contact)

// Calendar-Document Relationships
(:CalendarEvent)-[:DOCUMENTED_IN]->(:Document)
(:Document)-[:RELATED_TO_EVENT]->(:CalendarEvent)

// Contact-Contact Relationships
(:Contact)-[:WORKS_WITH]->(:Contact)
(:Contact)-[:REPORTS_TO]->(:Contact)
(:Contact)-[:COLLABORATES_WITH]->(:Contact)
```

### Cross-Reference Queries

```cypher
// Find all information about a person
MATCH (c:Contact {name: "Jerome Butler", user_id: $user_id})
OPTIONAL MATCH (c)<-[:INVOLVES|ATTENDED_BY|ORGANIZED_BY]-(events:CalendarEvent)
OPTIONAL MATCH (c)<-[:MENTIONED_IN|AUTHORED|REFERENCES]-(docs:Document)
OPTIONAL MATCH (c)-[rel:WORKS_WITH|REPORTS_TO|COLLABORATES_WITH]-(related:Contact)
RETURN c, events, docs, related, rel

// Find context for a time period
MATCH (events:CalendarEvent {user_id: $user_id})
WHERE events.start_date >= $start_date AND events.start_date <= $end_date
OPTIONAL MATCH (events)-[:INVOLVES]->(contacts:Contact)
OPTIONAL MATCH (events)-[:DOCUMENTED_IN]->(docs:Document)
RETURN events, contacts, docs
```

## 🛠️ New MCP Tools

### Calendar Tools

```python
# calendar_query_tool.py
class CalendarQueryTool:
    """Query calendar events with natural language"""
    
    async def search_events(
        self, 
        query: str,
        date_range: Optional[tuple] = None,
        user_id: str = None
    ) -> CalendarSearchResult:
        """
        Examples:
        - "meetings with Jerome next week"
        - "what do I have scheduled for Friday?"
        - "find all events at Tech Corp offices"
        """
        
    async def get_event_context(
        self,
        event_id: str,
        user_id: str
    ) -> EventContextResult:
        """Get full context including attendees, related docs, follow-ups"""

# calendar_management_tool.py  
class CalendarManagementTool:
    """Create, update, delete calendar events"""
    
    async def create_event(self, event_data: dict, user_id: str) -> str
    async def update_event(self, event_id: str, updates: dict, user_id: str) -> bool
    async def delete_event(self, event_id: str, user_id: str) -> bool
```

### Contact Tools

```python
# contact_query_tool.py
class ContactQueryTool:
    """Query contacts with natural language"""
    
    async def search_contacts(
        self,
        query: str, 
        user_id: str
    ) -> ContactSearchResult:
        """
        Examples:
        - "find Jerome Butler's contact info"
        - "who works at Tech Corp?"
        - "contacts I haven't spoken to in 6 months"
        """
        
    async def get_contact_relationship_history(
        self,
        contact_id: str,
        user_id: str
    ) -> RelationshipHistoryResult:
        """Get full history: meetings, emails, documents, projects"""

# contact_management_tool.py
class ContactManagementTool:
    """Create, update, delete contacts"""
    
    async def create_contact(self, contact_data: dict, user_id: str) -> str
    async def update_contact(self, contact_id: str, updates: dict, user_id: str) -> bool
    async def merge_contacts(self, contact_ids: list, user_id: str) -> str
```

### Cross-Reference Tools

```python
# pim_intelligence_tool.py
class PIMIntelligenceTool:
    """Intelligent cross-referencing across calendar, contacts, and documents"""
    
    async def analyze_relationship(
        self,
        person_name: str,
        user_id: str
    ) -> RelationshipAnalysis:
        """
        Example: "my relationship with Jerome Butler"
        Returns:
        - Contact details
        - Meeting history
        - Email exchanges  
        - Shared documents/projects
        - Collaboration patterns
        - Suggested follow-ups
        """
        
    async def find_meeting_patterns(
        self,
        criteria: dict,
        user_id: str  
    ) -> MeetingPatternAnalysis:
        """
        Examples:
        - "who do I meet with most often?"
        - "what types of meetings take most of my time?"
        - "recurring meetings that might be obsolete"
        """
        
    async def suggest_connections(
        self,
        user_id: str
    ) -> ConnectionSuggestions:
        """
        Suggest introductions, follow-ups, and networking opportunities
        based on calendar patterns and contact relationships
        """
```

## 🎨 User Interface Design

### Calendar Views

```
📅 Calendar Page
├── Month/Week/Day Views
├── Event Creation Modal
├── Quick Add via Natural Language
│   └── "Meeting with Jerome tomorrow 2pm about project X"
├── Search Bar with NL Queries
│   └── "show me all meetings with Tech Corp people"
└── Integration Indicators
    └── 📄 Related documents
    └── 👥 Contact details
    └── 🔗 Linked events
```

### Contact Views

```
👥 Contacts Page  
├── Contact List with Smart Grouping
│   ├── Recent interactions
│   ├── Frequent contacts
│   └── By company/group
├── Contact Detail with Context
│   ├── Basic info
│   ├── Recent meetings
│   ├── Shared documents
│   └── Interaction timeline
├── Search with NL Queries
│   └── "people I worked with on the Q4 project"
└── Relationship Visualization
    └── Connection maps and interaction heat maps
```

### Unified Search

```
🔍 Global Search
├── Cross-Entity Results
│   ├── 📄 Documents mentioning "Jerome"
│   ├── 📅 Calendar events with Jerome
│   ├── 👥 Jerome's contact card
│   └── 📊 Relationship summary
├── Natural Language Queries
│   ├── "my relationship with Jerome Butler"
│   ├── "what projects am I working on with Tech Corp?"
│   └── "who should I follow up with this week?"
└── Smart Suggestions
    └── Related people, events, and documents
```

## 🤖 Natural Language Query Examples

### Relationship Analysis
```
User: "Tell me about my relationship with Jerome Butler"

LLM Response: "Based on your data, here's your relationship with Jerome Butler:

📞 **Contact Info**: Senior Developer at Tech Corp (jerome@techcorp.com)

📅 **Recent Meetings**:
- Q4 Planning Session (Dec 15, 2024) - You discussed new feature roadmap
- Weekly Standup (Dec 12, 2024) - Jerome presented authentication updates  
- Project Kickoff (Nov 20, 2024) - Initial planning for mobile app

📄 **Shared Documents**:
- "Mobile App Architecture Notes" - Collaborative planning document
- "Q4 Feature Specifications" - Jerome provided technical requirements
- "Code Review Feedback" - Your notes on Jerome's authentication module

🤝 **Collaboration Pattern**: 
You meet with Jerome weekly, primarily on technical architecture topics. He's been your go-to person for authentication and security features. Your interactions suggest a strong working relationship with mutual respect for technical expertise.

💡 **Suggested Follow-up**: 
Jerome mentioned wanting feedback on the new security framework during your last meeting. Consider scheduling a technical review session."
```

### Calendar Intelligence
```
User: "What patterns do you see in my meeting schedule?"

LLM Response: "Here are your meeting patterns from the last 3 months:

📊 **Meeting Distribution**:
- 40% Technical Reviews (mostly with Jerome, Sarah, Mike)
- 25% Client Calls (primarily Tech Corp and StartupXYZ)  
- 20% Internal Planning (team standups, sprint planning)
- 15% One-on-ones (direct reports and manager check-ins)

⏰ **Time Patterns**:
- Most productive meeting slots: Tuesday-Thursday 10-11am
- Least effective: Friday afternoons (3 cancellations in 4 weeks)
- Average meeting length: 45 minutes (15 min over calendar allocation)

🎯 **Optimization Suggestions**:
- Consider consolidating client calls to specific days
- Block Friday afternoons for deep work
- Suggest 30-min default for internal meetings
- Jerome and you have 6 recurring meetings - consider combining some"
```

### Proactive Insights
```
User: "Who should I reach out to this week?"

LLM Response: "Based on your interaction patterns and calendar:

🔴 **Urgent Follow-ups**:
- **Maria Santos** (Client): Last meeting was 3 weeks ago, project deadline approaching
- **Jerome Butler**: Promised security framework feedback 5 days ago

🟡 **Relationship Maintenance**:
- **David Kim**: Haven't connected in 6 weeks, but worked closely on Q3 project
- **Lisa Chen**: LinkedIn shows she started new role - congratulations note opportunity

🟢 **Strategic Networking**:
- **Tech Corp team**: Multiple contacts there, could introduce Jerome to their new CTO
- **Conference attendees**: 3 people from AI Summit saved contacts but no follow-up yet

💡 **Calendar Opportunities**:
- Thursday 2-3pm slot open - perfect for Jerome's security review
- Next Tuesday lunch break - coffee with David to catch up"
```

## 🚀 Implementation Phases

### Phase 1: Foundation (2-3 weeks)
- ✅ Database schema creation and migration
- ✅ Basic CRUD operations for calendar and contacts
- ✅ User isolation enforcement
- ✅ Search indexing setup

### Phase 2: Knowledge Graph Integration (2-3 weeks)  
- ✅ Entity extraction and relationship mapping
- ✅ Cross-reference queries
- ✅ Basic MCP tools for calendar/contact queries
- ✅ Knowledge graph population from existing data

### Phase 3: MCP Tools & Intelligence (2-3 weeks)
- ✅ Advanced query tools with natural language
- ✅ Relationship analysis capabilities
- ✅ Pattern recognition and insights
- ✅ Integration with existing document search

### Phase 4: User Interface (2-3 weeks)
- ✅ Calendar management interface
- ✅ Contact management interface  
- ✅ Unified search with cross-entity results
- ✅ Relationship visualization

### Phase 5: Advanced Features (ongoing)
- ✅ Smart scheduling suggestions
- ✅ Meeting preparation automation
- ✅ Contact relationship insights
- ✅ Proactive networking suggestions
- ✅ Integration with external calendar/contact systems

## 🔧 Technical Considerations

### Data Privacy & Security
- All PIM data isolated per user (existing user isolation system)
- Encrypted sensitive fields (emails, phone numbers)
- Audit logging for all PIM operations
- GDPR compliance for contact data export/deletion

### Performance Optimization
- Efficient indexing for date-range calendar queries
- Contact search optimization with fuzzy matching
- Knowledge graph query caching
- Pagination for large datasets

### Integration Points
- Calendar sync with external systems (Google Calendar, Outlook)
- Contact import/export (vCard, CSV)
- Email integration for automatic event/contact creation
- Mobile app synchronization

### Scalability
- Partitioned tables by user_id for large datasets
- Knowledge graph sharding strategies
- Caching layers for frequent relationship queries
- Background processing for heavy analytics

## 💡 Advanced Use Cases

### Smart Meeting Preparation
```
"Prepare me for my 2pm meeting with Jerome"
→ Shows Jerome's contact info, recent email threads, shared documents, 
  previous meeting notes, and suggested talking points
```

### Project Team Discovery
```
"Who worked on the mobile app project?"
→ Cross-references calendar events, document collaborators, 
  and email threads to identify all team members
```

### Networking Intelligence
```
"Find connections between me and TechCorp's new CTO"
→ Analyzes contact relationships to find mutual connections
  and suggests introduction pathways
```

### Time Management Insights
```
"How much time do I spend in meetings vs. deep work?"
→ Analyzes calendar patterns and provides productivity insights
  with suggestions for optimization
```

This Personal Information Manager would transform the existing knowledge base into a comprehensive productivity and relationship management system, leveraging all the infrastructure we've built while adding powerful new capabilities for understanding and managing professional relationships.

## 🎯 Success Metrics

- **Query Accuracy**: Natural language calendar/contact queries return relevant results >90% of the time
- **Relationship Discovery**: Cross-referencing finds connections that users didn't remember >70% of the time  
- **Time Savings**: Users spend 30% less time searching for contact info and meeting context
- **Proactive Value**: System suggestions (follow-ups, introductions) are acted upon >40% of the time
- **User Adoption**: Calendar and contact features used daily by >80% of active users

The combination of structured PIM data with the existing document knowledge base creates a uniquely powerful system for understanding professional relationships and optimizing productivity. 