import { EditorView, ViewPlugin, Decoration, WidgetType } from '@codemirror/view';
import { StateField, StateEffect } from '@codemirror/state';
import { documentDiffStore } from '../../../services/documentDiffStore';

// State effects for managing live diffs
const addLiveDiff = StateEffect.define();
const removeLiveDiff = StateEffect.define();
const clearAllLiveDiffs = StateEffect.define();
const setLiveDiffDecorations = StateEffect.define();

// Theme for diff styling - matching chat sidebar colors
const diffTheme = EditorView.baseTheme({
  '.cm-edit-diff-deletion': {
    backgroundColor: 'rgba(211, 47, 47, 0.08)',
    border: '1px solid rgba(211, 47, 47, 0.2)',
    borderRadius: '2px',
    padding: '0 2px',
    textDecoration: 'line-through',
    textDecorationColor: 'rgba(211, 47, 47, 0.6)'
  },
  '.cm-edit-diff-addition': {
    backgroundColor: 'rgba(76, 175, 80, 0.08)',
    border: '1px solid rgba(76, 175, 80, 0.2)',
    borderRadius: '2px',
    padding: '2px 4px',
    marginLeft: '4px',
    display: 'inline-block',
    fontFamily: 'monospace',
    fontSize: 'inherit'
  },
  '.cm-edit-diff-replacement': {
    backgroundColor: 'rgba(211, 47, 47, 0.08)',
    border: '1px solid rgba(211, 47, 47, 0.2)',
    borderRadius: '2px',
    padding: '0 2px',
    textDecoration: 'line-through',
    textDecorationColor: 'rgba(211, 47, 47, 0.6)'
  },
  '.cm-edit-diff-buttons': {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '2px',
    marginLeft: '4px',
    verticalAlign: 'middle'
  },
  '.cm-edit-diff-accept, .cm-edit-diff-reject': {
    width: '18px',
    height: '18px',
    border: 'none',
    borderRadius: '3px',
    cursor: 'pointer',
    fontSize: '11px',
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 0,
    lineHeight: 1,
    fontFamily: 'monospace',
    transition: 'background-color 0.2s'
  },
  '.cm-edit-diff-accept': {
    backgroundColor: '#4caf50',
    color: 'white'
  },
  '.cm-edit-diff-accept:hover': {
    backgroundColor: '#45a049'
  },
  '.cm-edit-diff-reject': {
    backgroundColor: '#f44336',
    color: 'white'
  },
  '.cm-edit-diff-reject:hover': {
    backgroundColor: '#da190b'
  }
});

// Widget for showing proposed addition text
class DiffAdditionWidget extends WidgetType {
  constructor(text, operationId) {
    super();
    this.text = String(text || '');
    this.operationId = operationId;
  }
  
  eq(other) {
    return this.text === other.text && this.operationId === other.operationId;
  }
  
  toDOM() {
    const span = document.createElement('span');
    span.className = 'cm-edit-diff-addition';
    span.textContent = this.text;
    span.setAttribute('data-operation-id', this.operationId);
    return span;
  }
  
  ignoreEvent() {
    return false;
  }
}

// Widget for accept/reject buttons
class DiffButtonWidget extends WidgetType {
  constructor(operationId, onAccept, onReject) {
    super();
    this.operationId = operationId;
    this.onAccept = onAccept;
    this.onReject = onReject;
  }
  
  eq(other) {
    return this.operationId === other.operationId;
  }
  
  toDOM() {
    const wrapper = document.createElement('span');
    wrapper.className = 'cm-edit-diff-buttons';
    
    // Accept button (checkmark)
    const acceptBtn = document.createElement('button');
    acceptBtn.innerHTML = '✓';
    acceptBtn.className = 'cm-edit-diff-accept';
    acceptBtn.title = 'Accept edit';
    acceptBtn.setAttribute('aria-label', 'Accept edit');
    acceptBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (this.onAccept) {
        this.onAccept(this.operationId);
      }
    };
    
    // Reject button (X)
    const rejectBtn = document.createElement('button');
    rejectBtn.innerHTML = '✕';
    rejectBtn.className = 'cm-edit-diff-reject';
    rejectBtn.title = 'Reject edit';
    rejectBtn.setAttribute('aria-label', 'Reject edit');
    rejectBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (this.onReject) {
        this.onReject(this.operationId);
      }
    };
    
    wrapper.appendChild(acceptBtn);
    wrapper.appendChild(rejectBtn);
    return wrapper;
  }
  
  ignoreEvent() {
    return false;
  }
}

// State field for live diffs
const liveDiffField = StateField.define({
  create() {
    return Decoration.none;
  },
  update(decorations, tr) {
    decorations = decorations.map(tr.changes);
    
    for (const effect of tr.effects) {
      if (effect.is(clearAllLiveDiffs)) {
        return Decoration.none;
      }
      if (effect.is(setLiveDiffDecorations)) {
        return effect.value;
      }
    }
    
    return decorations;
  },
  provide: f => EditorView.decorations.from(f)
});

// Plugin class for managing live diffs
// Singleton registry: one plugin instance per document
const documentPluginRegistry = new Map(); // documentId -> plugin instance

const LiveEditDiffPluginClass = class {
  constructor(view, documentId) {
    try {
      // ✅ SINGLETON PATTERN: If a plugin already exists for this document, reuse it
      if (documentId && documentPluginRegistry.has(documentId)) {
        const existingPlugin = documentPluginRegistry.get(documentId);
        console.log('🔍 Reusing existing plugin instance for document:', documentId, 'Registry size:', documentPluginRegistry.size);
        // Update the view reference AND schedule decoration update for new view
        existingPlugin.view = view;
        existingPlugin.scheduleDecorationUpdate();
        return existingPlugin;
      }

      console.log('🔍 Creating NEW plugin instance for document:', documentId, 'Registry size before:', documentPluginRegistry.size, 'Time:', Date.now());

      this.view = view;
      this.documentId = documentId || null; // Store document identity
      this.operations = new Map(); // operationId -> {from, to, original, proposed, opType, messageId, start, end}
      this.maxOperations = 50; // Limit concurrent diffs (increased to prevent blocking)
      this.currentMessageId = null; // Track current message ID to clear old operations
      this.decorations = Decoration.none;
      this.pendingUpdate = false; // Track if decoration update is queued
      this.updateTimeout = null; // Track pending timeout
      this.creationTime = Date.now(); // Track when this instance was created
      this.initialRender = true; // Track if this is the first decoration update

      // Only log in development mode
      if (process.env.NODE_ENV === 'development') {
        console.log('🔍 Live diff extension plugin initialized for view:', view, 'documentId:', documentId, 'creationTime:', this.creationTime);
      }
      
      // Register this instance
      if (documentId) {
        documentPluginRegistry.set(documentId, this);
        console.log('📝 Registered plugin instance for document:', documentId, 'Registry size:', documentPluginRegistry.size);
      }
      
      // Restore persisted diffs for THIS document
      if (this.documentId) {
        const savedDiffs = documentDiffStore.getDiffs(this.documentId);
        if (savedDiffs && savedDiffs.operations && Array.isArray(savedDiffs.operations) && savedDiffs.operations.length > 0) {
          console.log(`Restoring ${savedDiffs.operations.length} diffs for document ${this.documentId}`);
          this.addOperations(savedDiffs.operations, savedDiffs.messageId);
        }
        
        // ✅ Subscribe to store changes to sync with other plugin instances
        this.handleStoreChange = this.handleStoreChange.bind(this);
        documentDiffStore.subscribe(this.handleStoreChange);
        console.log('📢 Plugin subscribed to documentDiffStore for document:', this.documentId);
      }
      
      // Listen for live diff events
      this.handleLiveDiffEvent = this.handleLiveDiffEvent.bind(this);
      window.addEventListener('editorOperationsLive', this.handleLiveDiffEvent);
      window.addEventListener('removeLiveDiff', this.handleLiveDiffEvent);
      window.addEventListener('clearEditorDiffs', this.handleLiveDiffEvent);
      
      // Initial decoration update (immediate for first render)
      this.applyDecorationUpdate();
    } catch (e) {
      console.error('❌ Error initializing live diff plugin:', e);
      this.decorations = Decoration.none;
    }
  }
  
  destroy() {
    try {
      // Clear pending timeout
      if (this.updateTimeout) {
        clearTimeout(this.updateTimeout);
        this.updateTimeout = null;
      }
      
      // ✅ Unregister from singleton registry
      if (this.documentId && documentPluginRegistry.get(this.documentId) === this) {
        documentPluginRegistry.delete(this.documentId);
        console.log('📝 Unregistered plugin instance for document:', this.documentId, 'Registry size:', documentPluginRegistry.size);
      }
      
      // ✅ Unsubscribe from store
      if (this.documentId && this.handleStoreChange) {
        documentDiffStore.unsubscribe(this.handleStoreChange);
        console.log('📢 Plugin unsubscribed from documentDiffStore for document:', this.documentId);
      }
      
      window.removeEventListener('editorOperationsLive', this.handleLiveDiffEvent);
      window.removeEventListener('removeLiveDiff', this.handleLiveDiffEvent);
      window.removeEventListener('clearEditorDiffs', this.handleLiveDiffEvent);
    } catch (e) {
      console.error('❌ Error destroying live diff plugin:', e);
    }
  }
  
  handleLiveDiffEvent(event) {
    try {
      // Only log in development mode
      if (process.env.NODE_ENV === 'development') {
        console.log('🔍 Live diff extension handleLiveDiffEvent called:', event.type, event.detail);
      }
      if (event.type === 'editorOperationsLive') {
        const { operations, messageId, documentId } = event.detail || {};
        
        // Only process operations for THIS document
        if (documentId && documentId !== this.documentId) {
          if (process.env.NODE_ENV === 'development') {
            console.log(`Ignoring operations for different document (${documentId} vs ${this.documentId})`);
          }
          return;
        }
        
        if (process.env.NODE_ENV === 'development') {
          console.log('🔍 Live diff extension received editorOperationsLive:', { 
            operationsCount: operations?.length, 
            messageId,
            documentId: documentId,
            thisDocumentId: this.documentId,
            firstOp: operations?.[0],
            allOps: operations
          });
        }
        if (Array.isArray(operations) && operations.length > 0) {
          // Clearing of old operations now happens in addOperations() before maxOperations check
          // This ensures new operations always have room and aren't blocked by old ones
          console.log('🔍 Calling addOperations with', operations.length, 'operations');
          this.addOperations(operations, messageId);
          
          // ✅ NOTE: ChatSidebarContext already saved to documentDiffStore
          // No need to save again here - that would create duplicates!
        } else {
          console.warn('⚠️ No operations array or empty array received');
        }
      } else if (event.type === 'removeLiveDiff') {
        const { operationId } = event.detail || {};
        if (operationId) {
          this.removeOperation(operationId);
        }
      } else if (event.type === 'clearEditorDiffs') {
        this.clearAllOperations();
      }
    } catch (e) {
      console.error('❌ Error in handleLiveDiffEvent:', e);
    }
  }
  
  handleStoreChange(documentId, changeType) {
    // Only respond to changes for THIS document
    if (documentId !== this.documentId) return;
    
    console.log('📢 Plugin received store change notification:', { documentId, changeType });
    
    if (changeType === 'remove' || changeType === 'clear') {
      // Sync our operations with the store
      const storeDiffs = documentDiffStore.getDiffs(this.documentId);
      const storeOps = storeDiffs?.operations || [];
      
      console.log('📢 Syncing plugin operations with store:', {
        pluginOpsCount: this.operations.size,
        storeOpsCount: storeOps.length
      });
      
      // Build set of operation IDs that should exist
      const validIds = new Set(storeOps.map(op => op.operationId || op.id || `${op.start}-${op.end}`));
      
      // Remove operations that are no longer in the store
      const toRemove = [];
      this.operations.forEach((op, id) => {
        if (!validIds.has(id) && !validIds.has(op.operationId)) {
          toRemove.push(id);
        }
      });
      
      if (toRemove.length > 0) {
        console.log('📢 Removing', toRemove.length, 'operations from plugin that are no longer in store');
        toRemove.forEach(id => this.operations.delete(id));
        this.scheduleDecorationUpdate();
      }
    }
    // ❌ REMOVED: Do NOT sync on 'set' - creates cascade!
    // The plugin already added the operation via the editorOperationsLive event.
    // Store notifications on 'set' are for OTHER components (like TabbedContentManager),
    // not for the plugin that just updated the store!
  }
  
  update(update) {
    // Handle document changes - adjust operation positions instead of removing
    // Only remove operations that are clearly invalid (negative, reversed, or way out of bounds)
    if (update.docChanged && this.operations.size > 0) {
      const docLength = update.state.doc.length;
      const toRemove = [];
      let needsUpdate = false;
      
      // Smart invalidation: Check if user manual edits overlap with pending diffs
      if (update.changes && this.documentId) {
        update.changes.iterChanges((fromA, toA, fromB, toB, inserted) => {
          // User edited range: [fromA, toA]
          const editStart = fromA;
          const editEnd = toA;
          
          // Invalidate overlapping diffs
          this.operations.forEach((op, id) => {
            const opStart = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
            const opEnd = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : opStart);
            
            // Check for overlap
            const overlaps = !(editEnd < opStart || editStart > opEnd);
            
            if (overlaps) {
              console.log(`User edit [${editStart}-${editEnd}] overlaps diff [${opStart}-${opEnd}] - invalidating`);
              toRemove.push(id);
              // Also remove from centralized store
              documentDiffStore.removeDiff(this.documentId, id);
            }
          });
        });
      }
      
      // Adjust positions for remaining operations based on text length changes
      if (update.changes && this.operations.size > 0) {
        update.changes.iterChanges((fromA, toA, fromB, toB) => {
          const lenChange = (toB - fromB) - (toA - fromA);
          
          this.operations.forEach((op, id) => {
            if (toRemove.includes(id)) return; // Skip already-invalidated operations
            
            const opStart = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
            const opEnd = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : opStart);
            
            // If operation is after the edit, adjust its position
            if (opStart >= toA) {
              op.start += lenChange;
              op.end += lenChange;
              op.from = op.start;
              op.to = op.end;
              needsUpdate = true;
            } else if (opEnd > fromA && opStart < toA && !toRemove.includes(id)) {
              // Partial overlap - operation was already invalidated above, but ensure cleanup
              // This shouldn't happen due to overlap check above, but safety check
            }
          });
        });
      }
      
      this.operations.forEach((op, id) => {
        if (toRemove.includes(id)) return; // Skip already-invalidated operations
        
        const from = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
        const to = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : from);
        
        // Only remove operations that are clearly invalid (negative, reversed, or way out of bounds)
        // Operations that are slightly out of bounds will be clamped in applyDecorationUpdate
        if (from < 0 || to < from || from > docLength + 100 || to > docLength + 100) {
          toRemove.push(id);
        } else {
          // Update stored positions to match current document (clamp to valid range)
          const clampedFrom = Math.max(0, Math.min(docLength, from));
          const clampedTo = Math.max(clampedFrom, Math.min(docLength, to));
          if (clampedFrom !== from || clampedTo !== to) {
            op.from = clampedFrom;
            op.to = clampedTo;
            op.start = clampedFrom;
            op.end = clampedTo;
            needsUpdate = true;
          }
        }
      });
      
      // Remove clearly invalid operations
      toRemove.forEach(id => {
        if (!this.operations.has(id)) return; // Already removed
        console.warn('Removing operation with invalid positions:', id);
        this.operations.delete(id);
        if (this.documentId) {
          documentDiffStore.removeDiff(this.documentId, id);
        }
        needsUpdate = true;
      });
      
      // Update decorations if any operations were adjusted or removed
      if (needsUpdate) {
        this.scheduleDecorationUpdate();
      }
    }
    
    try {
      // Operations remain visible until explicitly accepted or rejected
      // This prevents the issue where accepting one edit removes others
    } catch (e) {
      console.error('❌ Error in live diff plugin update:', e);
    }
  }
  
  _calculateSimilarity(str1, str2) {
    // Simple similarity check - count matching characters
    if (!str1 || !str2) return 0;
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    if (longer.length === 0) return 1;
    
    let matches = 0;
    for (let i = 0; i < shorter.length; i++) {
      if (shorter[i] === longer[i]) matches++;
    }
    return matches / longer.length;
  }
  
  _getFrontmatterEnd(docText) {
    // Find frontmatter end (---\n...\n---\n)
    try {
      const match = docText.match(/^(---\s*\n[\s\S]*?\n---\s*\n)/);
      return match ? match[0].length : 0;
    } catch (e) {
      return 0;
    }
  }
  
  addOperations(operations, messageId) {
    // CRITICAL FIX: Clear operations from previous messages FIRST to ensure we have room
    // This must happen before the maxOperations check
    if (messageId && messageId !== this.currentMessageId) {
      const previousMessageId = this.currentMessageId;
      this.currentMessageId = messageId;
      
      // Remove all operations from previous messages
      const toRemove = [];
      this.operations.forEach((op, id) => {
        if (op.messageId && op.messageId !== messageId) {
          toRemove.push(id);
        }
      });
      
      if (toRemove.length > 0) {
        console.log(`🧹 Clearing ${toRemove.length} operations from previous message (${previousMessageId})`);
        toRemove.forEach(id => this.operations.delete(id));
      }
    } else if (!this.currentMessageId && messageId) {
      this.currentMessageId = messageId;
    }
    
    // Now check maxOperations limit AFTER clearing old operations
    const currentCount = this.operations.size;
    const toAdd = operations.slice(0, Math.max(0, this.maxOperations - currentCount));
    
    if (toAdd.length < operations.length) {
      console.warn(`⚠️ Limiting operations: ${operations.length} requested, ${toAdd.length} will be added (${currentCount}/${this.maxOperations} already active)`);
    }
    
    // Get frontmatter boundary
    const docText = this.view.state.doc.toString();
    const frontmatterEnd = this._getFrontmatterEnd(docText);
    
    // Only log in development mode to reduce console noise
    if (process.env.NODE_ENV === 'development') {
      console.log('🔍 Live diff extension adding operations:', { 
        total: operations.length, 
        toAdd: toAdd.length,
        frontmatterEnd,
        operations: toAdd.map(op => ({ 
          start: op.start, 
          end: op.end, 
          op_type: op.op_type,
          hasText: !!op.text,
          textPreview: op.text?.substring(0, 50)
        }))
      });
    }
    
    toAdd.forEach((op, idx) => {
      // ✅ Use stable operation ID based on operation content, not timestamp
      // This ensures the same operation gets the same ID across restores and events
      const operationId = op.operationId || op.id || `${messageId || 'op'}-${idx}-${op.start}-${op.end}`;
      let start = Number(op.start || 0);
      let end = Number(op.end || start);
      const opType = op.op_type || 'replace_range';
      // ✅ CRITICAL: Handle both backend field names and storage field names
      const original = op.original_text || op.original || op.anchor_text || '';
      // ✅ CRITICAL: Handle both 'text' (from backend) and 'proposed' (from storage)
      const proposed = op.text || op.proposed || '';
      
      // CRITICAL: Guard frontmatter - ensure operations never occur before frontmatter end
      if (start < frontmatterEnd) {
        console.warn('⚠️ Operation before frontmatter detected, adjusting:', { 
          originalStart: start, 
          frontmatterEnd,
          opType 
        });
        // For insertions, move to after frontmatter
        if (start === end || opType === 'insert_after_heading' || opType === 'insert_after') {
          start = frontmatterEnd;
          end = frontmatterEnd;
        } else {
          // For replace/delete, clamp start to frontmatter end
          start = frontmatterEnd;
          end = Math.max(end, start);
        }
      }
      
      // Only log in development mode
      if (process.env.NODE_ENV === 'development') {
        console.log('🔍 Processing operation:', { operationId, start, end, opType, proposedLength: proposed.length });
      }
      
      // Validate range - be more lenient for operations that might be slightly out of bounds
      // This can happen when document changes between operation creation and display
      const docLength = this.view.state.doc.length;
      if (start < 0 || end < start) {
        console.warn('Invalid operation range (negative or reversed):', { start, end, docLength });
        return;
      }
      
      // Allow operations that extend slightly beyond document (will be clamped in decoration)
      // Only reject if way out of bounds (more than 100 chars beyond)
      if (start > docLength + 100 || end > docLength + 100) {
        console.warn('Operation range way out of bounds:', { start, end, docLength });
        return;
      }
      
      // Clamp positions to valid range for storage
      start = Math.max(0, Math.min(docLength, start));
      end = Math.max(start, Math.min(docLength, end));
      
      // For replace_range, verify original text matches
      if (opType === 'replace_range' && original) {
        const currentText = this.view.state.doc.sliceString(start, end);
        if (currentText !== original && original.length > 0) {
          console.warn('Operation original text mismatch:', {
            expected: original.substring(0, 50),
            actual: currentText.substring(0, 50)
          });
          // Still add it, but it may be stale
        }
      }
      
      this.operations.set(operationId, {
        operationId: operationId, // ✅ Store the ID with the operation
        from: start,
        to: end,
        original: original,
        proposed: proposed,
        opType: opType,
        messageId: messageId,
        start: start,
        end: end
      });
    });
    
    // ✅ Update the centralized store with operations that now have IDs
    if (this.documentId && toAdd.length > 0) {
      // Get current stored operations and update them with IDs
      const storedOperations = Array.from(this.operations.values());
      const currentContent = this.view.state.doc.toString();
      documentDiffStore.setDiffs(this.documentId, storedOperations, messageId, currentContent);
      console.log('✅ Updated centralized store with', storedOperations.length, 'operations (with IDs)');
    }
    
    this.scheduleDecorationUpdate();
  }
  
  removeOperation(operationId) {
    this.operations.delete(operationId);
    this.scheduleDecorationUpdate();
  }
  
  clearAllOperations() {
    this.operations.clear();
    this.scheduleDecorationUpdate();
  }
  
  scheduleDecorationUpdate() {
    // Prevent multiple updates from being queued
    if (this.pendingUpdate) {
      return;
    }

    this.pendingUpdate = true;

    // Clear any existing timeout
    if (this.updateTimeout) {
      clearTimeout(this.updateTimeout);
    }

    // Use requestAnimationFrame for smoother updates, but immediate for initial render
    if (this.initialRender) {
      this.pendingUpdate = false;
      this.applyDecorationUpdate();
      this.initialRender = false;
    } else {
      this.updateTimeout = requestAnimationFrame(() => {
        this.pendingUpdate = false;
        this.updateTimeout = null;
        this.applyDecorationUpdate();
      });
    }
  }
  
  applyDecorationUpdate() {
    try {
      const decos = [];
      
      this.operations.forEach((op, id) => {
        try {
          // Validate operation positions against current document
          const docLength = this.view.state.doc.length;
          const from = Math.max(0, Math.min(docLength, op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0)));
          const to = Math.max(from, Math.min(docLength, op.to !== undefined ? op.to : (op.end !== undefined ? op.end : from)));
          
          // Skip if positions are invalid
          if (from < 0 || to < from || from > docLength || to > docLength) {
            console.warn('Skipping invalid operation:', id, { from, to, docLength });
            return;
          }
          
          if (op.opType === 'replace_range') {
            // Only create mark decoration if from !== to (mark decorations can't be empty)
            if (from !== to) {
              decos.push(
                Decoration.mark({
                  class: 'cm-edit-diff-replacement',
                  attributes: { 'data-operation-id': id }
                }).range(from, to)
              );
            }
            
            // Add proposed text as widget after the range
            if (op.proposed && op.proposed.length > 0) {
              decos.push(
                Decoration.widget({
                  widget: new DiffAdditionWidget(op.proposed, id),
                  side: 1,
                  block: false  // Ensure not treated as block widget
                }).range(to)
              );
            }

            // Add accept/reject buttons
            decos.push(
              Decoration.widget({
                widget: new DiffButtonWidget(
                  id,
                  () => this.acceptOperation(id),
                  () => this.rejectOperation(id)
                ),
                side: 1,
                block: false  // Ensure not treated as block widget
              }).range(to)
            );
          } else if (op.opType === 'delete_range') {
            // Only create mark decoration if from !== to
            if (from !== to) {
              decos.push(
                Decoration.mark({
                  class: 'cm-edit-diff-deletion',
                  attributes: { 'data-operation-id': id }
                }).range(from, to)
              );
            }
            
            // Add accept/reject buttons
            decos.push(
              Decoration.widget({
                widget: new DiffButtonWidget(
                  id,
                  () => this.acceptOperation(id),
                  () => this.rejectOperation(id)
                ),
                side: 1
              }).range(to)
            );
          } else if (op.opType === 'insert_after_heading' || op.opType === 'insert_after') {
            // For insert operations, show proposed text at the insertion point
            if (op.proposed && op.proposed.length > 0) {
              decos.push(
                Decoration.widget({
                  widget: new DiffAdditionWidget(op.proposed, id),
                  side: 1
                }).range(from)
              );
              
              // Add accept/reject buttons
              decos.push(
                Decoration.widget({
                  widget: new DiffButtonWidget(
                    id,
                    () => this.acceptOperation(id),
                    () => this.rejectOperation(id)
                  ),
                  side: 1
                }).range(from)
              );
            }
          }
        } catch (e) {
          console.warn('Error creating decoration for operation:', id, e);
        }
      });
      
      // CRITICAL: CodeMirror requires decorations to be sorted by `from` position and `startSide`
      // Sort decorations by their start position (from) before creating the set
      // CodeMirror Range objects have `from`, `to`, and `value` properties
      
      // Debug: log decoration positions BEFORE sorting
      if (decos.length > 0) {
        console.log('🔍 Decorations BEFORE sorting:', decos.map(d => ({
          from: d.from,
          to: d.to,
          valueSide: d.value?.side,
          valueWidget: d.value?.widget?.constructor?.name,
          widgetType: d.value?.widget?.type || 'unknown',
          allKeys: Object.keys(d)
        })));
        console.log('🔍 Operations Map state:', {
          size: this.operations.size,
          pluginCreationTime: this.creationTime,
          operations: Array.from(this.operations.entries()).map(([id, op]) => ({
            id,
            operationId: op.operationId,
            messageId: op.messageId,
            from: op.from,
            to: op.to,
            proposedLength: op.proposed?.length || 0,
            start: op.start,
            end: op.end
          }))
        });
      }
      
      decos.sort((a, b) => {
        // Extract `from` position from Range object
        const aFrom = a.from !== undefined ? a.from : 0;
        const bFrom = b.from !== undefined ? b.from : 0;
        
        if (aFrom !== bFrom) {
          return aFrom - bFrom;
        }
        
        // If same position, sort by side
        // Widget decorations store `side` in value.side, default to 0 for marks
        const aSide = (a.value && a.value.side !== undefined) ? a.value.side : 0;
        const bSide = (b.value && b.value.side !== undefined) ? b.value.side : 0;
        
        return aSide - bSide;
      });
      
      // Debug: log decoration positions AFTER sorting
      if (decos.length > 0) {
        console.log('🔍 Decorations AFTER sorting:', decos.map(d => ({
          from: d.from,
          valueSide: d.value?.side
        })));
      }
      
      // Update decorations via state effect
      const decorationSet = Decoration.set(decos);
      this.decorations = decorationSet;
      
      // CRITICAL: Only dispatch if view is still valid
      if (this.view && !this.view.isDestroyed) {
        this.view.dispatch({
          effects: setLiveDiffDecorations.of(decorationSet)
        });
      }
    } catch (e) {
      console.error('❌ Error applying decoration update:', e);
      this.decorations = Decoration.none;
    }
  }
  
  acceptOperation(operationId) {
    const op = this.operations.get(operationId);
    if (!op) return;
    
    // Remove from centralized store
    if (this.documentId) {
      documentDiffStore.removeDiff(this.documentId, operationId);
    }
    
    // CRITICAL: Remove the accepted operation from the map FIRST
    // This prevents it from being affected by position adjustments
    this.operations.delete(operationId);
    
    // Calculate position delta for adjusting other operations
    let positionDelta = 0;
    const opStart = op.start !== undefined ? op.start : op.from;
    const opEnd = op.end !== undefined ? op.end : op.to;
    
    // Build operation object for applyOperations handler
    let operationObj;
    if (op.opType === 'insert_after_heading' || op.opType === 'insert_after') {
      // Normalize text to ensure proper newlines
      let normalizedText = op.proposed || '';
      const insertPos = op.from;
      
      // Check context around insertion point
      const docText = this.view.state.doc.toString();
      const charBefore = insertPos > 0 ? docText[insertPos - 1] : '';
      const charAfter = insertPos < docText.length ? docText[insertPos] : '';
      
      // Check if insertion point is at end of line (has newline after) or middle of line
      const isAtLineEnd = charAfter === '\n' || charAfter === '';
      const needsLeadingNewline = charBefore !== '\n' && charBefore !== '';
      
      // Check if text is a chapter heading
      const isChapterHeading = normalizedText.trim().startsWith('## Chapter');
      
      // Normalize: ensure chapter headings have proper spacing
      if (isChapterHeading) {
        // Chapter headings should have newline(s) before them
        if (!normalizedText.startsWith('\n')) {
          // Add appropriate spacing based on context
          if (needsLeadingNewline) {
            normalizedText = '\n\n' + normalizedText; // Double newline for chapter separation
          } else if (!isAtLineEnd) {
            normalizedText = '\n' + normalizedText; // Single newline if already at line boundary
          }
        }
      } else if (needsLeadingNewline && !normalizedText.startsWith('\n')) {
        // Non-chapter insertions: add newline if inserting mid-line
        normalizedText = '\n' + normalizedText;
      }
      
      // For insert operations, start and end are the same (insertion point)
      operationObj = {
        start: insertPos,
        end: insertPos, // Insert at this position
        text: normalizedText,
        op_type: 'replace_range' // Insert is handled as replace with start == end
      };
      
      // Insertion adds text, so positions after it shift forward
      positionDelta = normalizedText.length;
    } else if (op.opType === 'delete_range') {
      operationObj = {
        start: op.start,
        end: op.end,
        op_type: 'delete_range'
      };
      
      // Deletion removes text, so positions after it shift backward
      positionDelta = -(opEnd - opStart);
    } else {
      // replace_range
      operationObj = {
        start: op.start,
        end: op.end,
        text: op.proposed,
        op_type: 'replace_range',
        original_text: op.original
      };
      
      // Replacement: delta = new length - old length
      const oldLength = opEnd - opStart;
      const newLength = (op.proposed || '').length;
      positionDelta = newLength - oldLength;
    }
    
    // Track operations to remove (overlapping ones)
    const toRemove = [];
    
    // Adjust positions of other operations that come after this one
    if (positionDelta !== 0) {
      this.operations.forEach((otherOp, otherId) => {
        const otherStart = otherOp.start !== undefined ? otherOp.start : otherOp.from;
        const otherEnd = otherOp.end !== undefined ? otherOp.end : otherOp.to;
        
        // If this operation comes after the accepted one, adjust its positions
        if (otherStart >= opEnd) {
          const newStart = otherStart + positionDelta;
          const newEnd = otherEnd + positionDelta;
          
          // Update positions
          otherOp.start = newStart;
          otherOp.end = newEnd;
          if (otherOp.from !== undefined) otherOp.from = newStart;
          if (otherOp.to !== undefined) otherOp.to = newEnd;
          
          // CRITICAL: Mark as adjusted so validation doesn't remove it
          // The operation is still valid, it just targets text at a new position
          otherOp.adjustedForAccept = true;
          
          console.log('✏️ Adjusted operation', otherId, 'by', positionDelta, 'positions');
        } else if (otherEnd > opStart && otherStart < opEnd) {
          // Overlapping operation - mark for removal
          console.log('⚠️ Marking overlapping operation for removal:', otherId);
          toRemove.push(otherId);
        }
      });
    }
    
    // Remove overlapping operations (directly delete to avoid multiple updates)
    toRemove.forEach(id => this.operations.delete(id));
    
    // CRITICAL: Update decorations IMMEDIATELY before applying the change
    // This removes the accepted operation from decorations to prevent invalid positions
    this.scheduleDecorationUpdate();
    
    // Emit event for external handling (MarkdownCMEditor will apply the change)
    // Use setTimeout to ensure decorations are updated first
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('liveEditAccepted', {
        detail: {
          operationId: operationId,
          operation: operationObj
        }
      }));
      
      // After document change, adjust remaining operations and update again
      setTimeout(() => {
        this.scheduleDecorationUpdate();
      }, 0);
    }, 0);
  }
  
  rejectOperation(operationId) {
    console.log('🔍 rejectOperation called with ID:', operationId);
    
    const op = this.operations.get(operationId);
    if (!op) {
      console.warn('⚠️ rejectOperation: Operation not found in plugin map:', operationId);
      return;
    }
    
    console.log('🔍 rejectOperation: Found operation:', {
      operationId: op.operationId,
      start: op.start,
      end: op.end,
      messageId: op.messageId
    });
    
    // Remove from centralized store
    if (this.documentId) {
      console.log('🔍 rejectOperation: Calling documentDiffStore.removeDiff:', {
        documentId: this.documentId,
        operationId: operationId
      });
      documentDiffStore.removeDiff(this.documentId, operationId);
    } else {
      console.warn('⚠️ rejectOperation: No documentId, cannot remove from store');
    }
    
    // Emit event for external handling
    window.dispatchEvent(new CustomEvent('liveEditRejected', {
      detail: {
        operationId: operationId
      }
    }));
    
    // Remove from active diffs
    this.removeOperation(operationId);
  }
};

// Create default plugin instance (for backward compatibility)
const liveEditDiffPlugin = ViewPlugin.fromClass(LiveEditDiffPluginClass);

/**
 * Create live edit diff extension for CodeMirror
 * 
 * Displays edit operations as inline color-coded diffs:
 * - Red strikethrough for deletions
 * - Green background for additions
 * - Accept/reject buttons for each operation
 * 
 * Usage:
 * ```javascript
 * const extension = createLiveEditDiffExtension(documentId);
 * // Add to CodeMirror extensions array
 * ```
 * 
 * To trigger live diffs:
 * ```javascript
 * window.dispatchEvent(new CustomEvent('editorOperationsLive', {
 *   detail: {
 *     operations: [{ start: 100, end: 150, text: 'new text', op_type: 'replace_range' }],
 *     messageId: 'message-id',
 *     documentId: 'doc-123'
 *   }
 * }));
 * ```
 * 
 * @param {string} documentId - Document identifier for persistent diff storage
 */
export function createLiveEditDiffExtension(documentId = null) {
  try {
    console.log('🔍 Creating live edit diff extension for document:', documentId);
    
    // Store documentId in a closure variable that the plugin class can access
    const pluginDocumentId = documentId;
    
    // Create a plugin class that has access to documentId through closure
    class DocumentAwareDiffPlugin extends LiveEditDiffPluginClass {
      constructor(view) {
        // Pass documentId to parent constructor
        super(view, pluginDocumentId);
      }
    }
    
    const extension = [
      diffTheme,
      liveDiffField,
      ViewPlugin.fromClass(DocumentAwareDiffPlugin, {
        decorations: v => v.decorations
      })
    ];
    console.log('🔍 Live edit diff extension created:', extension.length, 'items');
    return extension;
  } catch (e) {
    console.error('❌ Error creating live edit diff extension:', e);
    return [];
  }
}

