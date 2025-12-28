# Signal Corps: Spontaneous Notification System

The Signal Corps notification system enables agents and services to emit out-of-band alerts, status updates, and spontaneous intelligence into the active chat sidebar without interrupting the primary conversation flow.

## Overview

Notifications are ephemeral messages that appear in the chat sidebar as centered, borderless "pills" rather than full chat bubbles. They're designed for:

- **Tactical Alerts**: System status updates, quota warnings, completion notifications
- **Progress Updates**: Real-time status during long-running operations
- **Spontaneous Intelligence**: Proactive insights or pattern detection notifications

## Architecture

Notifications flow through the existing SSE (Server-Sent Events) streaming infrastructure:

```
Agent/Service → gRPC ChatChunk (type="notification") → Backend Proxy → SSE → Frontend → UI
```

## SSE Payload Structure

When emitting a notification, the SSE payload should have the following structure:

```json
{
  "type": "notification",
  "message": "Your notification text here",
  "severity": "info",  // "info" | "success" | "warning" | "error"
  "temporary": false,   // true = auto-removes after 10 seconds, false = persists
  "timestamp": "2024-01-15T10:30:00Z",
  "agent": "agent_name",  // Optional: which agent/service emitted this
  "browser_notify": false  // Optional: true = show browser notification (auto-enabled for warning/error)
}
```

### Severity Levels

- **info**: General informational messages (blue)
- **success**: Success confirmations (green)
- **warning**: Warnings that need attention (orange)
- **error**: Error notifications (red)

## Usage in Agents

### Method 1: Direct gRPC Chunk Emission (Orchestrator)

In the orchestrator's `grpc_service.py`, agents can yield notification chunks directly:

```python
# In your agent processing code
yield orchestrator_pb2.ChatChunk(
    type="notification",
    message="Research complete! 500 documents indexed.",
    timestamp=datetime.now().isoformat(),
    agent_name="research_agent",
    metadata={
        "severity": "success",
        "temporary": "false"
    }
)
```

### Method 2: State-Based Signals (Future Pattern)

For a more integrated approach, agents can add notifications to their LangGraph state, and the orchestrator will automatically emit them:

```python
# In your agent node
state["signals"] = state.get("signals", [])
state["signals"].append({
    "type": "notification",
    "message": "Web search quota at 80%",
    "severity": "warning",
    "temporary": True
})

# Orchestrator automatically converts signals to gRPC chunks
```

**Note**: The state-based pattern is a future enhancement. Currently, use Method 1 for direct emission.

## Example Use Cases

### 1. Progress Updates During Long Operations

```python
# Emit progress notification
yield orchestrator_pb2.ChatChunk(
    type="notification",
    message="Indexing documents... 250/500 complete",
    timestamp=datetime.now().isoformat(),
    agent_name="indexing_service",
    metadata={
        "severity": "info",
        "temporary": "true"  # Auto-removes after 10 seconds
    }
)
```

### 2. Quota Warnings

```python
# Emit quota warning
yield orchestrator_pb2.ChatChunk(
    type="notification",
    message="⚠️ Web Search quota at 80%",
    timestamp=datetime.now().isoformat(),
    agent_name="system",
    metadata={
        "severity": "warning",
        "temporary": "false"  # Persists until user dismisses
    }
)
```

### 3. Completion Alerts

```python
# Emit completion notification
yield orchestrator_pb2.ChatChunk(
    type="notification",
    message="✅ Research complete! 500 documents indexed.",
    timestamp=datetime.now().isoformat(),
    agent_name="research_agent",
    metadata={
        "severity": "success",
        "temporary": "true"
    }
)
```

### 4. Spontaneous Intelligence

```python
# Emit pattern detection notification
yield orchestrator_pb2.ChatChunk(
    type="notification",
    message="💡 Pattern detected: You've been researching quantum computing frequently. Would you like a summary?",
    timestamp=datetime.now().isoformat(),
    agent_name="intelligence_service",
    metadata={
        "severity": "info",
        "temporary": "false"
    }
)
```

## Frontend Behavior

### Rendering

Notifications are rendered as centered Material-UI `Chip` components with:
- Severity-based color coding
- Italic, smaller font
- Outlined variant (borderless appearance)
- Semi-transparent background in dark mode

### Browser Notifications

Notifications can also appear as browser-level notifications (using the Web Notifications API):

- **Automatic browser notifications** are shown for:
  - `severity: "error"` - Critical errors that need immediate attention
  - `severity: "warning"` - Warnings that should be brought to user's attention
  - Any notification with `browser_notify: true` in metadata

- **Browser notification behavior**:
  - Permission is requested automatically on first use
  - Notifications include agent name and severity in the title
  - Clicking a notification focuses the browser window
  - Error notifications require user interaction (don't auto-dismiss)
  - Duplicate notifications are prevented using tag-based deduplication

- **To explicitly request browser notification** (even for info/success):
  ```python
  yield orchestrator_pb2.ChatChunk(
      type="notification",
      message="Important update that needs browser notification",
      metadata={
          "severity": "info",
          "browser_notify": "true"  # Explicitly request browser notification
      }
  )
  ```

### Auto-Removal

If `temporary: true`, notifications automatically fade out and are removed from the message list after 10 seconds.

### Persistence

Notifications marked as `ephemeral: true` are not saved to the conversation database. They exist only in the current session's message state.

## Best Practices

1. **Use Appropriate Severity**: Match the severity level to the importance of the notification
   - `info`: General updates, progress indicators
   - `success`: Completion confirmations, positive outcomes
   - `warning`: Quota limits, potential issues
   - `error`: Failures, critical problems

2. **Temporary vs. Persistent**: 
   - Use `temporary: true` for progress updates that will be superseded
   - Use `temporary: false` for alerts that need user attention

3. **Keep Messages Concise**: Notifications are small pills - keep text brief and actionable

4. **Don't Overwhelm**: Limit notification frequency to avoid UI clutter

5. **Context Matters**: Include enough context in the message that users understand what the notification refers to

## Integration Points

### From Backend Services

Backend services can emit notifications by sending gRPC chunks through the orchestrator service. Ensure the chunk type is set to `"notification"` and includes appropriate metadata.

### From LangGraph Agents

Agents running in the `llm-orchestrator` microservice can emit notifications during their workflow execution by yielding `ChatChunk` messages with `type="notification"`.

### From Background Jobs

Background jobs can emit progress notifications by sending updates through the orchestrator's streaming interface.

## Future Enhancements

1. **State-Based Signals**: Automatic emission of notifications from LangGraph state `signals` array
2. **User Dismissal**: Allow users to manually dismiss persistent notifications
3. **Notification History**: Optional persistence of important notifications to conversation history
4. **Action Buttons**: Add clickable actions to notifications (e.g., "View Details", "Dismiss")
5. **Sound/Visual Effects**: Optional audio or animation cues for important notifications
6. **User Preferences**: Allow users to configure which notification severities trigger browser notifications

## Troubleshooting

### Notifications Not Appearing

1. Check that the gRPC chunk type is exactly `"notification"` (case-sensitive)
2. Verify the backend proxy is forwarding the chunk (check logs for "📢 Forwarded notification chunk")
3. Ensure the frontend SSE handler is receiving the notification type (check browser console)
4. Verify the message has valid content and severity

### Notifications Persisting Too Long

- Ensure `temporary: "true"` is set in metadata for auto-removal
- Check that the frontend timeout is working (10 seconds default)

### Styling Issues

- Verify Material-UI theme is properly configured
- Check that severity colors are valid: `info`, `success`, `warning`, `error`
- Ensure the notification rendering branch is being hit (add console.log for debugging)

## See Also

- [GRPC_MICROSERVICES_ARCHITECTURE.md](GRPC_MICROSERVICES_ARCHITECTURE.md) - Overall gRPC architecture
- [PROTO_CONTEXT_USAGE_GUIDE.md](PROTO_CONTEXT_USAGE_GUIDE.md) - gRPC protocol details
- [backend/api/grpc_orchestrator_proxy.py](../backend/api/grpc_orchestrator_proxy.py) - Proxy implementation
- [frontend/src/contexts/ChatSidebarContext.js](../frontend/src/contexts/ChatSidebarContext.js) - Frontend SSE handling
- [frontend/src/components/chat/ChatMessagesArea.js](../frontend/src/components/chat/ChatMessagesArea.js) - Notification rendering

