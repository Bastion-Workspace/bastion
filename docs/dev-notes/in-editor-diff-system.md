# In-Editor Diff System Documentation

## Overview

The in-editor diff system provides a persistent, user-friendly interface for reviewing and applying AI-generated edits to documents. It displays proposed changes as inline visual diffs with accept/reject controls, maintains state across tab switches, and integrates seamlessly with CodeMirror 6's native history system.

## Architecture

### Components

1. **Document Diff Store** (`frontend/src/services/documentDiffStore.js`)
   - Centralized storage for diff operations
   - User-specific localStorage persistence
   - Multi-point content hashing for drift detection
   - Cross-tab synchronization via event system

2. **Live Edit Diff Extension** (`frontend/src/components/editor/extensions/liveEditDiffExtension.js`)
   - CodeMirror 6 plugin for rendering diffs
   - Singleton registry (one plugin per document)
   - Decoration-based visualization (marks + widgets)
   - Position tracking and adjustment

3. **Markdown Editor Integration** (`frontend/src/components/MarkdownCMEditor.js`)
   - CodeMirror transaction-based edit application
   - Batch operation handlers
   - Navigation controls
   - Scroll position preservation

4. **Backend Resolver** (`llm-orchestrator/orchestrator/utils/editor_operation_resolver.py`)
   - Progressive text matching strategies
   - Fuzzy position recovery
   - Frontmatter protection
   - Confidence scoring

## Features

### Visual Diff Display

- **Replace Range**: Red strikethrough for original text, green background for proposed text
- **Delete Range**: Red strikethrough for text to be removed
- **Insert Operations**: Green background for new text to be inserted
- **Accept/Reject Buttons**: Inline controls for each diff
- **Info Tooltips**: Hover over info icon (ℹ) to see agent reasoning/notes

### Navigation

- **Previous Edit** / **Next Edit** buttons: Jump between diffs in long documents
- Wraps around (last → first, first → last)
- Automatically scrolls to diff position with margin
- Sets cursor to diff location

### Batch Operations

- **Accept All**: Apply all pending diffs in a single transaction
- **Reject All**: Remove all pending diffs
- Operations processed in reverse position order (end → start) for stability

### Persistence

- Diffs persist across tab switches via localStorage
- User-specific storage keys
- Automatic cleanup of diffs older than 24 hours
- Content hash validation detects document drift

### Position Recovery

- **Exact Match**: Primary strategy using `original_text` or `anchor_text`
- **Normalized Whitespace**: Fallback for formatting-only changes
- **Fuzzy Window Search**: Searches ±1000 characters around expected position when exact match fails
- **Confidence Scoring**: 1.0 (exact) → 0.7 (fuzzy) → 0.0 (failed)

### Drift Detection

- **Multi-Point Sampling**: Content hash samples at 0%, 25%, 50%, 75%, 100% positions
- **Change Detection**: Compares length and sample hashes
- **Stale Marking**: Flags diffs as stale when content drifts >5% or <3/5 samples match
- **Automatic Invalidation**: Removes diffs that overlap with manual user edits

## Operation Types

### replace_range
- **Use Case**: Modify existing text
- **Required Fields**: `original_text` (exact text to replace), `text` (new content)
- **Visual**: Red strikethrough + green addition widget

### delete_range
- **Use Case**: Remove text
- **Required Fields**: `original_text` (exact text to delete)
- **Visual**: Red strikethrough only

### insert_after_heading
- **Use Case**: Add content after a specific heading
- **Required Fields**: `anchor_text` (exact heading line), `text` (content to insert)
- **Visual**: Green addition widget at insertion point

### insert_after
- **Use Case**: Add content after specific text (e.g., continue paragraph)
- **Required Fields**: `anchor_text` (exact text to insert after), `text` (content to insert)
- **Visual**: Green addition widget at insertion point

## Data Flow

### 1. Agent Generation
```
Editing Agent → ManuscriptEdit (with EditorOperations)
  ↓
Editor Operation Resolver (resolves positions, adds pre_hash)
  ↓
API Response → Frontend
```

### 2. Frontend Display
```
editorOperationsLive Event
  ↓
LiveEditDiffExtension.addOperations()
  ↓
documentDiffStore.setDiffs() (persist to localStorage)
  ↓
applyDecorationUpdate() (render visual diffs)
```

### 3. User Interaction
```
User clicks Accept/Reject
  ↓
acceptOperation() / rejectOperation()
  ↓
liveEditAccepted / liveEditRejected Event
  ↓
MarkdownCMEditor.applyOperations() (CodeMirror transaction)
  ↓
Document updated + undo history preserved
```

## CodeMirror Integration

### Transactions
- All edits applied via `view.dispatch()` with `ChangeSet`
- Integrated into native undo/redo history (Cmd+Z / Ctrl+Z)
- Atomic operations preserve cursor positions and markers
- No separate undo stack needed

### Decorations
- **Marks**: For highlighting original text (strikethrough)
- **Widgets**: For proposed text and control buttons
- Sorted by position for proper rendering order
- Automatically adjusted on document changes

## State Management

### Plugin Registry
- Singleton pattern: one plugin instance per documentId
- Registry: `documentPluginRegistry` (Map<documentId, plugin>)
- Reuses existing plugin on tab switch/editor remount
- Automatic cleanup on destroy

### Operation Storage
- In-memory: `plugin.operations` (Map<operationId, operation>)
- Persistent: `documentDiffStore.diffs` (localStorage)
- Synchronized via event system
- Operations include: `start`, `end`, `original`, `proposed`, `opType`, `note`, `messageId`

## Error Handling

### Position Validation
- Clamps positions to valid document range
- Removes operations with negative or reversed positions
- Adjusts positions when document changes (after accepts)

### Hash Mismatch
- Pre-hash validation prevents applying edits to stale content
- Skips operation with warning if hash doesn't match
- Fuzzy recovery attempts to find text in nearby window

### Missing Operations
- Graceful degradation if operation not found
- Logs warnings for debugging
- Continues processing remaining operations

## Performance Considerations

### Decoration Updates
- Debounced via `requestAnimationFrame`
- Hash-based change detection (skips if no changes)
- Batch updates for multiple operations

### Storage
- localStorage writes throttled (500ms)
- Automatic cleanup of old diffs (24h TTL)
- User-specific keys prevent cross-user conflicts

### Position Adjustment
- Only adjusts operations after accepted edits
- Marks adjusted operations to prevent double-adjustment
- Shielded from invalidation during programmatic updates

## API Reference

### Document Diff Store

```javascript
// Set diffs for a document
documentDiffStore.setDiffs(documentId, operations, messageId, contentSnapshot);

// Get diffs for a document
const diffs = documentDiffStore.getDiffs(documentId);

// Clear all diffs
documentDiffStore.clearDiffs(documentId);

// Remove specific diff
documentDiffStore.removeDiff(documentId, operationId);

// Validate diffs (returns { invalidated, isStale })
const validation = documentDiffStore.validateDiffs(documentId, currentContent);
```

### Live Edit Diff Extension

```javascript
// Get plugin instance
const plugin = getLiveEditDiffPlugin(documentId);

// Batch operations
plugin.acceptAllOperations();
plugin.rejectAllOperations();

// Navigation
const next = plugin.findNextDiff(cursorPosition);
const prev = plugin.findPreviousDiff(cursorPosition);
plugin.jumpToPosition(position);

// Individual operations
plugin.acceptOperation(operationId);
plugin.rejectOperation(operationId);
```

### Events

```javascript
// Trigger live diffs
window.dispatchEvent(new CustomEvent('editorOperationsLive', {
  detail: {
    operations: [...],
    messageId: 'msg-123',
    documentId: 'doc-456'
  }
}));

// Accept/reject (handled internally, but can be triggered)
window.dispatchEvent(new CustomEvent('liveEditAccepted', {
  detail: { operationId, operation }
}));

window.dispatchEvent(new CustomEvent('liveEditRejected', {
  detail: { operationId }
}));
```

## Best Practices

### Agent Implementation
- Always provide `original_text` or `anchor_text` with exact, verbatim text
- Include `note` field for explainable diffs (user sees reasoning)
- Use granular edits (word/sentence level) when possible
- Set `pre_hash` in resolver for optimistic concurrency

### Frontend Integration
- Always pass `documentId` to extension for persistence
- Use CodeMirror transactions, not manual string manipulation
- Preserve scroll position before applying changes
- Handle plugin lifecycle (cleanup on unmount)

### Error Recovery
- Implement fuzzy matching fallback in resolver
- Validate positions before applying operations
- Log warnings for debugging position issues
- Gracefully handle missing operations

## Future Enhancements

- [ ] Server-side diff persistence (sync across devices)
- [ ] Collaborative diff reviews (multiple users)
- [ ] Branching/staging mode (preview all diffs applied)
- [ ] Agent-driven re-basing (auto-recalculate on major changes)
- [ ] Keyboard shortcuts for navigation (Ctrl+Shift+Up/Down)
- [ ] Diff statistics (count, scope, confidence scores)

## Troubleshooting

### Diffs Not Appearing
- Check `documentId` is passed to extension
- Verify operations have valid `start`/`end` positions
- Check browser console for errors
- Ensure `editorOperationsLive` event is dispatched

### Positions Incorrect
- Verify `original_text` matches document exactly
- Check frontmatter boundaries (operations should be after frontmatter)
- Enable fuzzy matching in resolver
- Check for manual edits that invalidated positions

### Diffs Disappearing
- Check localStorage quota (may be full)
- Verify user-specific storage key
- Check 24h TTL cleanup
- Ensure documentId is consistent

### Performance Issues
- Reduce max operations (default: 50)
- Check decoration update frequency
- Monitor localStorage write frequency
- Profile fuzzy matching on large documents
