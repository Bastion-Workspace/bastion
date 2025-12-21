# Dynamic Tool Loading Mechanism

## Overview

Dynamic tool loading is a Kiro-style mechanism that loads only relevant tools into the LLM context based on query analysis. This dramatically reduces token usage, improves context quality, and enhances scalability by providing agents with a focused, minimal tool set rather than their entire tool catalog.

## Benefits

1. **Token Optimization**: Reduces LLM context window by 60-80% by loading only necessary tools
2. **Improved Context Quality**: LLM receives focused, relevant tool descriptions instead of overwhelming catalog
3. **Better Performance**: Faster LLM processing with smaller context windows
4. **Scalability**: System can support hundreds of tools without context bloat
5. **Progressive Loading**: Agents can request additional tools mid-workflow if needed

## Architecture

### Core Components

1. **Tool Categories** (`tool_categories.py`)
   - Defines tool categories (SEARCH_LOCAL, SEARCH_WEB, DOCUMENT_OPS, etc.)
   - Maps keywords to categories for detection
   - Defines core tools per agent (always loaded)

2. **Dynamic Tool Analyzer** (`dynamic_tool_analyzer.py`)
   - Analyzes queries to detect needed tool categories
   - Considers conversation context
   - Returns confidence scores and reasoning

3. **Centralized Tool Registry** (`centralized_tool_registry.py`)
   - Manages tool definitions and permissions
   - Implements `get_tools_for_agent_dynamic()` method
   - Handles progressive tool loading

4. **Base Agent** (`base_agent.py`)
   - Provides `_get_agent_tools_dynamic()` method
   - Integrates dynamic loading into agent workflows

## Implementation Details

### Tool Categories

Tools are organized into logical categories:

```python
class ToolCategory(str, Enum):
    SEARCH_LOCAL = "search_local"      # Document/entity search
    SEARCH_WEB = "search_web"          # Web search/crawl
    DOCUMENT_OPS = "document_ops"      # Get/update documents
    ANALYSIS = "analysis"              # Content analysis
    MATH = "math"                      # Calculations
    WEATHER = "weather"                # Weather queries
    AWS_PRICING = "aws_pricing"        # AWS cost tools
    ORG_FILES = "org_files"            # Org-mode search
    MESSAGING = "messaging"            # Room messaging
    FILE_CREATION = "file_creation"    # File/folder ops
    EXPANSION = "expansion"            # Query expansion
    IMAGE_GENERATION = "image_generation"  # Image creation
    WEBSITE_CRAWL = "website_crawl"    # Website crawling
```

### Core Tools Per Agent

Each agent has a set of core tools that are always loaded:

```python
AGENT_CORE_TOOLS = {
    "research_agent": [
        "search_local",
        "expand_query",
        "search_conversation_cache"
    ],
    "chat_agent": [
        "search_conversation_cache"
    ],
    "content_analysis_agent": [
        "search_local",
        "get_document"
    ],
    # ... more agents
}
```

### Query Analysis

The `DynamicToolAnalyzer` analyzes queries using:

1. **Keyword Detection**: Matches query text against category keywords
2. **Conversation Context**: Considers previously used tools
3. **Confidence Scoring**: Calculates confidence based on keyword matches

Example analysis:

```python
analysis = await analyzer.analyze_tool_needs(
    query="What is the weather in New York?",
    agent_type=AgentType.RESEARCH_AGENT
)

# Returns:
{
    "core_categories": [ToolCategory.SEARCH_LOCAL],
    "conditional_categories": [ToolCategory.WEATHER],
    "confidence": 0.95,
    "reasoning": "Query triggers categories: weather",
    "core_tools": ["search_local", "expand_query", "search_conversation_cache"]
}
```

### Dynamic Tool Loading Flow

```
User Query
    ↓
DynamicToolAnalyzer.analyze_tool_needs()
    ↓
Detect Categories (keyword matching + context)
    ↓
Get Core Tools (always loaded)
    ↓
Get Conditional Tools (based on detected categories)
    ↓
Merge & Deduplicate
    ↓
Return Focused Tool Set
```

## Applying to New Agents

### Step 1: Define Core Tools

Add core tools for your agent in `tool_categories.py`:

```python
AGENT_CORE_TOOLS = {
    "your_agent": [
        "essential_tool_1",
        "essential_tool_2"
    ]
}
```

### Step 2: Categorize Your Tools

When registering tools in `centralized_tool_registry.py`, assign categories:

```python
self._tools["your_tool"] = ToolDefinition(
    name="your_tool",
    function=your_tool_function,
    description="Tool description",
    access_level=ToolAccessLevel.READ_ONLY,
    parameters={...},
    categories=[ToolCategory.YOUR_CATEGORY]  # Add category
)
```

### Step 3: Add Category Keywords (if new category)

If creating a new category, add keywords in `tool_categories.py`:

```python
CATEGORY_KEYWORDS = {
    ToolCategory.YOUR_CATEGORY: [
        "keyword1", "keyword2", "keyword3"
    ]
}
```

### Step 4: Update Agent to Use Dynamic Loading

In your agent's `BaseAgent` subclass, use dynamic tool loading:

```python
async def _process_request_node(self, state: YourAgentState) -> Dict[str, Any]:
    """Process request with dynamic tool loading"""
    query = state.get("query", "")
    metadata = state.get("metadata", {})
    
    # Get tools dynamically based on query
    tools = await self._get_agent_tools_dynamic(query, metadata)
    
    # Use tools with LLM
    llm = self._get_llm(temperature=0.7, state=state)
    llm_with_tools = llm.bind_tools(tools)
    
    # ... rest of processing
```

### Step 5: Map Tools to Categories

Update `_get_tools_by_category()` in `centralized_tool_registry.py`:

```python
category_to_tools = {
    # ... existing mappings
    "your_category": ["your_tool_1", "your_tool_2"],
}
```

## Code Examples

### Example 1: Research Agent with Dynamic Loading

```python
class ResearchAgent(BaseAgent):
    async def _process_request_node(self, state: ResearchState) -> Dict[str, Any]:
        query = state.get("query", "")
        
        # Dynamic tool loading - only loads relevant tools
        tools = await self._get_agent_tools_dynamic(
            query=query,
            metadata=state.get("metadata", {})
        )
        
        # LLM gets focused tool set
        llm = self._get_llm(temperature=0.7, state=state)
        llm_with_tools = llm.bind_tools(tools)
        
        # Process with minimal context
        response = await llm_with_tools.ainvoke(messages)
        return {"response": response.content}
```

### Example 2: Progressive Tool Loading

If an agent realizes mid-workflow it needs more tools:

```python
# Agent detects it needs web search tools
from services.langgraph_tools.centralized_tool_registry import get_tool_registry

registry = await get_tool_registry()
additional_tools = await registry.load_additional_tools(
    agent_type=AgentType.RESEARCH_AGENT,
    categories=[ToolCategory.SEARCH_WEB],
    existing_tool_names=current_tool_names
)

# Add to existing tools
all_tools = current_tools + additional_tools
```

### Example 3: Custom Category Detection

For complex detection logic, extend `DynamicToolAnalyzer`:

```python
class CustomToolAnalyzer(DynamicToolAnalyzer):
    async def analyze_tool_needs(self, query, agent_type, context):
        # Call parent analysis
        analysis = await super().analyze_tool_needs(query, agent_type, context)
        
        # Add custom logic
        if self._needs_special_tool(query):
            analysis["conditional_categories"].append(ToolCategory.SPECIAL)
        
        return analysis
```

## Configuration

### Environment Variables

```bash
# Enable/disable dynamic tool loading (default: true)
ENABLE_DYNAMIC_TOOL_LOADING=true

# Model for tool analysis (default: fast model)
DYNAMIC_TOOL_ANALYSIS_MODEL=anthropic/claude-haiku-4.5

# Tool loading strategy
TOOL_LOADING_STRATEGY=dynamic  # or "static" for fallback
```

### Configuration in `config.py`

```python
ENABLE_DYNAMIC_TOOL_LOADING = os.getenv("ENABLE_DYNAMIC_TOOL_LOADING", "true").lower() == "true"
DYNAMIC_TOOL_ANALYSIS_MODEL = os.getenv("DYNAMIC_TOOL_ANALYSIS_MODEL", settings.FAST_MODEL)
TOOL_LOADING_STRATEGY = os.getenv("TOOL_LOADING_STRATEGY", "dynamic")
```

## Best Practices

### 1. Core Tools Should Be Minimal

Only include tools that are **always** needed:

```python
# ✅ GOOD: Minimal core tools
"research_agent": ["search_local", "expand_query"]

# ❌ BAD: Too many core tools
"research_agent": ["search_local", "search_web", "get_document", "analyze", ...]
```

### 2. Use Descriptive Category Names

Categories should clearly indicate their purpose:

```python
# ✅ GOOD
ToolCategory.SEARCH_WEB = "search_web"

# ❌ BAD
ToolCategory.WEB = "web"  # Too vague
```

### 3. Keyword Coverage

Ensure keywords cover common query patterns:

```python
CATEGORY_KEYWORDS = {
    ToolCategory.WEATHER: [
        "weather", "temperature", "forecast", "climate",
        "rain", "snow", "sunny", "cloudy"  # Comprehensive coverage
    ]
}
```

### 4. Logging for Debugging

Always log dynamic tool selection:

```python
logger.info(
    f"🎯 Dynamic loading: {len(tools)} tools for {agent_type.value} "
    f"(core: {len(core_tools)}, conditional: {len(conditional_tools)})"
)
logger.debug(f"🎯 Categories detected: {[c.value for c in categories]}")
```

### 5. Graceful Fallback

Always provide fallback to static loading:

```python
try:
    tools = await self._get_agent_tools_dynamic(query, metadata)
except Exception as e:
    logger.warning(f"Dynamic loading failed: {e}, falling back to static")
    tools = await self._get_agent_tools_async()  # Static fallback
```

## Testing Dynamic Tool Loading

### Test Query Analysis

```python
async def test_tool_analysis():
    analyzer = DynamicToolAnalyzer()
    
    # Test weather query
    analysis = await analyzer.analyze_tool_needs(
        query="What's the weather in Seattle?",
        agent_type=AgentType.RESEARCH_AGENT
    )
    
    assert ToolCategory.WEATHER in analysis["conditional_categories"]
    assert analysis["confidence"] > 0.8
```

### Test Tool Loading

```python
async def test_dynamic_loading():
    registry = await get_tool_registry()
    
    # Test research agent with web query
    tools = await registry.get_tools_for_agent_dynamic(
        agent_type=AgentType.RESEARCH_AGENT,
        query="Search the web for latest AI news"
    )
    
    # Should include web search tools
    tool_names = [t["function"]["name"] for t in tools]
    assert "search_and_crawl" in tool_names or "search_web" in tool_names
```

## Troubleshooting

### Tools Not Loading

1. **Check category assignment**: Ensure tools have categories assigned
2. **Verify keywords**: Check if query keywords match category keywords
3. **Review core tools**: Verify agent has core tools defined
4. **Check permissions**: Ensure agent has permission for the tools

### Low Confidence Scores

1. **Add more keywords**: Expand keyword coverage for categories
2. **Improve query analysis**: Enhance detection logic in analyzer
3. **Consider context**: Use conversation context for better detection

### Performance Issues

1. **Reduce core tools**: Minimize always-loaded tools
2. **Optimize analysis**: Cache analysis results when possible
3. **Lazy loading**: Load tools only when needed in workflow

## Migration Guide

### Migrating Existing Agents

1. **Identify core tools**: Determine which tools are always needed
2. **Categorize tools**: Assign categories to all tools
3. **Update agent code**: Replace `_get_agent_tools_async()` with `_get_agent_tools_dynamic()`
4. **Test thoroughly**: Verify tool loading works correctly
5. **Monitor performance**: Check token usage improvements

### Example Migration

**Before (Static Loading):**
```python
async def _process_request_node(self, state):
    tools = await self._get_agent_tools_async()  # All tools
    # ... use tools
```

**After (Dynamic Loading):**
```python
async def _process_request_node(self, state):
    query = state.get("query", "")
    tools = await self._get_agent_tools_dynamic(query, state.get("metadata"))  # Focused tools
    # ... use tools
```

## Performance Metrics

### Expected Improvements

- **Token Reduction**: 60-80% reduction in tool description tokens
- **Context Quality**: Focused, relevant tool descriptions
- **Response Time**: 10-20% faster LLM processing
- **Scalability**: Support for 100+ tools without context bloat

### Monitoring

Track these metrics:
- Average tools loaded per query
- Token usage per request
- Tool loading time
- Confidence scores

## Related Documentation

- [LangGraph Best Practices](./langgraph-best-practices.md)
- [Tool Registry Architecture](./tool-registry-architecture.md)
- [Agent Development Guide](./agent-development-guide.md)

## References

- AWS Kiro Powers: Dynamic tool loading concept
- LangGraph ToolNode: Automatic tool execution
- OpenAI Function Calling: Tool binding and execution


