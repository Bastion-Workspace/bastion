# Roosevelt's "Best Practice" HITL Integration Guide

**BULLY!** This guide shows how to integrate the new **LangGraph Human-in-the-Loop Orchestrator** with your frontend and test the system.

## 🎯 API Endpoints

### 1. Start/Continue Conversation
```
POST /api/hitl/chat
{
    "message": "Was Hans Wilsdorf a nazi?",
    "conversation_id": "optional-uuid",
    "persona": {}
}

Response:
{
    "status": "interrupted|complete|error",
    "response": "I found limited local information...",
    "conversation_id": "uuid",
    "next_step": "web_search_permission",
    "permission_request": {
        "operation_type": "web_search",
        "query": "Was Hans Wilsdorf a nazi?",
        "reasoning": "Local search insufficient...",
        "estimated_cost": "~$0.05",
        "safety_level": "low"
    }
}
```

### 2. Handle Permission Response
```
POST /api/hitl/permission
{
    "conversation_id": "uuid",
    "response": "yes"  // or "no", "cancel"
}

Response:
{
    "status": "complete",
    "response": "Based on my web search, Hans Wilsdorf...",
    "conversation_id": "uuid",
    "final_agent": "research"
}
```

### 3. Stream with Real-time Updates
```
POST /api/hitl/stream
Content-Type: text/event-stream

data: {"type": "start", "message": "Processing..."}
data: {"type": "conversation_id", "conversation_id": "uuid"}
data: {"type": "final", "status": "interrupted", "response": "...", "permission_request": {...}}
```

### 4. Check Conversation Status
```
GET /api/hitl/status/{conversation_id}

Response:
{
    "status": "interrupted|complete|processing",
    "next_step": "web_search_permission",
    "permission_request": {...}
}
```

### 5. Cancel Conversation
```
DELETE /api/hitl/conversation/{conversation_id}

Response:
{
    "status": "cancelled",
    "message": "Conversation cancelled successfully"
}
```

## 🔄 Frontend Integration Flow

### Research Query with Permission Request

```javascript
// 1. Start conversation
const response = await fetch('/api/hitl/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message: "Was Hans Wilsdorf a nazi?"
    })
});

const result = await response.json();

if (result.status === 'interrupted') {
    // Show permission request UI
    showPermissionRequest(result.permission_request);
    
    // Handle user response
    const permissionResponse = await getUserPermissionResponse();
    
    // Send permission response
    const finalResponse = await fetch('/api/hitl/permission', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            conversation_id: result.conversation_id,
            response: permissionResponse  // "yes", "no", "cancel"
        })
    });
    
    const finalResult = await finalResponse.json();
    displayFinalAnswer(finalResult.response);
}
```

### React Component Example

```jsx
function ChatWithPermissions() {
    const [conversation, setConversation] = useState(null);
    const [permissionRequest, setPermissionRequest] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const sendMessage = async (message) => {
        setLoading(true);
        
        try {
            const response = await fetch('/api/hitl/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message })
            });
            
            const result = await response.json();
            
            if (result.status === 'interrupted') {
                setPermissionRequest(result.permission_request);
                setConversation(result);
            } else if (result.status === 'complete') {
                setConversation(result);
                setPermissionRequest(null);
            }
        } finally {
            setLoading(false);
        }
    };
    
    const handlePermissionResponse = async (response) => {
        setLoading(true);
        
        try {
            const result = await fetch('/api/hitl/permission', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    conversation_id: conversation.conversation_id,
                    response
                })
            });
            
            const finalResult = await result.json();
            setConversation(finalResult);
            setPermissionRequest(null);
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div>
            {permissionRequest && (
                <PermissionRequestDialog 
                    request={permissionRequest}
                    onResponse={handlePermissionResponse}
                />
            )}
            
            {conversation && (
                <div>{conversation.response}</div>
            )}
        </div>
    );
}
```

## 🧪 Testing the System

### Manual Testing

1. **Basic Research Query**:
   ```bash
   curl -X POST http://localhost:8000/api/hitl/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Was Hans Wilsdorf a nazi?"}'
   ```

2. **Expected Response** (Status: interrupted):
   ```json
   {
     "status": "interrupted",
     "response": "I found limited local information...",
     "next_step": "web_search_permission",
     "permission_request": {
       "operation_type": "web_search",
       "query": "Was Hans Wilsdorf a nazi?",
       "reasoning": "Local search insufficient..."
     }
   }
   ```

3. **Grant Permission**:
   ```bash
   curl -X POST http://localhost:8000/api/hitl/permission \
     -H "Content-Type: application/json" \
     -d '{"conversation_id": "uuid-from-step-1", "response": "yes"}'
   ```

4. **Expected Final Response** (Status: complete):
   ```json
   {
     "status": "complete",
     "response": "Based on my web search, Hans Wilsdorf...",
     "final_agent": "research"
   }
   ```

### Health Check

```bash
curl http://localhost:8000/api/hitl/health
```

Expected response:
```json
{
  "status": "healthy",
  "orchestrator_initialized": true,
  "checkpointer_available": true,
  "graph_compiled": true
}
```

## 🔧 Key Differences from Old System

### ✅ Official LangGraph Patterns
- Uses `interrupt_before` for clean breakpoints
- Built-in state persistence with `MemorySaver`
- Official `.update_state()` for user input injection

### ✅ Cleaner Architecture
- Dedicated permission checkpoint nodes
- No custom pending operations
- Clear separation of concerns

### ✅ Better Error Handling
- Graceful fallbacks for failed classifications
- Proper exception handling at each node
- Clear error messages in responses

### ✅ Type Safety
- Full TypedDict support for state
- Pydantic models for API requests/responses
- Structured permission requests

## 🚀 Deployment Notes

1. **No Breaking Changes**: The old orchestrator API still works
2. **Gradual Migration**: Can test HITL endpoints alongside existing ones
3. **Docker Compatible**: Works with existing `docker compose up --build`

## 📊 Benefits Achieved

1. **No More Infinite Loops**: Official HITL patterns prevent routing issues
2. **Better UX**: Clear permission requests with context
3. **Maintainable**: Standard LangGraph patterns instead of custom logic
4. **Scalable**: Built-in state persistence for multiple conversations
5. **Testable**: Clean API endpoints for automated testing

**By George!** The new HITL system follows **best-of-breed practices** and should eliminate the pain points we were experiencing!
