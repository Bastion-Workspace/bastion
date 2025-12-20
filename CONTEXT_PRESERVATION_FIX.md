# Context Preservation in Query Expansion - Issue & Fix

## The Problem

When a user has a conversation like:
1. "Has the All-In Podcast released any new episodes lately?" 
2. "What other podcasts should I listen to?"

The query expansion for the second query generates generic queries like:
- "podcast recommendations"
- "similar podcasts suggestions"  
- "other podcasts to listen to"

**Instead of context-aware queries like:**
- "podcasts similar to All-In Podcast"
- "tech business politics podcasts like All-In"
- "podcast recommendations for All-In listeners"

## Root Cause

The query expansion service **HAS** a conversation context feature (lines 125-156 in query_expansion_service.py), and the research workflow subgraph **DOES** extract conversation messages (lines 122-150 in research_workflow_subgraph.py).

However, the conversation messages may not be flowing properly from the Full Research Agent to the research subgraph, or they may not be in the expected format.

## What We Know

### The Flow
1. **Full Research Agent** (`full_research_agent.py`):
   - Receives messages from checkpoint
   - Passes them to research subgraph in `_call_research_subgraph_node` (line 187)
   
2. **Research Workflow Subgraph** (`research_workflow_subgraph.py`):
   - Receives messages in `query_expansion_node` (line 123)
   - Extracts last 2 messages for context (line 125)
   - Passes context to `expand_query_tool` (line 146-149)

3. **Query Expansion Service** (`query_expansion_service.py`):
   - Receives conversation_context parameter (line 75)
   - Includes it in the prompt (lines 126-135)
   - Uses it to resolve vague references

### The Evidence from Logs

```
🔍 Including 3 conversation messages for context  (quick answer check)
📚 Loaded 1 messages from checkpoint  (research agent)
```

**Missing log:**
```
"Including conversation context for query expansion: {len(context_parts)} messages"
```

This log should appear if the context extraction works, but it doesn't show up.

## Hypothesis

The messages might be:
1. **Not persisting properly through checkpoints** - Only 1 message loaded instead of 3
2. **In an unexpected format** - The parsing logic expects HumanMessage/AIMessage or dicts with specific keys
3. **Being filtered out** - The logic only takes last 2 messages, but maybe they're not in the right format

## Diagnostic Changes Made

### 1. Added logging in `research_workflow_subgraph.py` 

**Lines 123-125:**
```python
conversation_messages = state.get("messages", [])
logger.info(f"🔍 DEBUG: query_expansion_node received {len(conversation_messages)} messages from state")
if conversation_messages:
    for idx, msg in enumerate(conversation_messages[:3]):  # Log first 3 for debugging
        msg_type = type(msg).__name__
        if isinstance(msg, dict):
            logger.info(f"   Message {idx}: dict with keys={list(msg.keys())}, role={msg.get('role', 'N/A')}")
        else:
            logger.info(f"   Message {idx}: {msg_type}")
```

**Line 145 (added warning):**
```python
else:
    logger.warning(f"⚠️ Had {len(conversation_messages)} messages but extracted 0 context_parts - message format issue?")
```

### 2. Added logging in `full_research_agent.py`

**Lines 191-193:**
```python
logger.info(f"🔬 Passing query to research subgraph: '{query[:100]}'")
logger.info(f"🔍 DEBUG: Passing {len(messages)} messages to research subgraph")
```

## Testing Plan

Run the same conversation sequence and look for these logs:

### Expected Diagnostic Output

```
🔬 Passing query to research subgraph: 'What other podcasts might I be interested in?'
🔍 DEBUG: Passing X messages to research subgraph
🔍 DEBUG: query_expansion_node received X messages from state
   Message 0: [type and keys]
   Message 1: [type and keys]
   Message 2: [type and keys]
```

### Scenarios

**If X = 0:** Messages not being loaded from checkpoint
**If X = 1:** Only current message, not history
**If X >= 2 but no context extracted:** Message format issue
**If X >= 2 and context extracted:** Should work!

## Next Steps

Based on diagnostic output:

### If messages not flowing (X < 2):
- Check checkpoint loading in base_agent.py
- Verify conversation history persistence
- Check if messages are being saved to state correctly

### If message format issue (X >= 2 but no context):
- Look at message structure from diagnostics
- Update parsing logic to handle actual format
- May need to convert message types

### If context flows but expansion still generic:
- Check if query expansion service is actually using the context
- Verify the prompt in query_expansion_service.py
- Check for caching issues (context not in cache key)

## The Fix (Once Diagnosed)

### Option A: Message Format Conversion
If messages are in wrong format, convert them in `_call_research_subgraph_node`:

```python
# Convert messages to format expected by subgraph
formatted_messages = []
for msg in messages:
    if hasattr(msg, 'type') and hasattr(msg, 'content'):
        # LangChain message object
        formatted_messages.append({
            'role': 'human' if msg.type == 'human' else 'ai',
            'content': msg.content
        })
    elif isinstance(msg, dict):
        formatted_messages.append(msg)
```

### Option B: Load More Messages from Checkpoint
If only 1 message being loaded, increase the limit in base_agent.py checkpoint loading.

### Option C: Use Conversation Cache
If messages aren't persisting, use the conversation cache tool to get recent context:
- Call `search_conversation_cache_tool` in query expansion
- Use cached responses as conversation context

## Expected Outcome

After fix, query expansion for "What other podcasts should I listen to?" should generate:
- "podcasts similar to All-In Podcast tech business"
- "All-In Podcast alternatives recommendations"
- "tech business politics podcasts like All-In"

This will lead to much better web search results and more relevant recommendations.







