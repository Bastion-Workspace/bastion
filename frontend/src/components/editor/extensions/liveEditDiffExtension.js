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
  },
  '.cm-edit-diff-info': {
    cursor: 'help',
    marginLeft: '4px',
    color: '#666',
    fontSize: '12px',
    verticalAlign: 'middle',
    opacity: 0.7,
    transition: 'opacity 0.2s'
  },
  '.cm-edit-diff-info:hover': {
    opacity: 1,
    color: '#1976d2'
  },
  '.cm-edit-diff-tooltip': {
    position: 'absolute',
    background: '#333',
    color: '#fff',
    padding: '8px 12px',
    borderRadius: '4px',
    fontSize: '12px',
    maxWidth: '300px',
    zIndex: 10000,
    pointerEvents: 'none',
    boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    whiteSpace: 'pre-wrap',
    wordWrap: 'break-word'
  },
  '.cm-edit-diff-out-of-bounds': {
    display: 'inline-block',
    marginTop: '8px',
    padding: '6px 10px',
    border: '1px solid rgba(255, 152, 0, 0.5)',
    borderRadius: '4px',
    backgroundColor: 'rgba(255, 152, 0, 0.06)',
    maxWidth: '100%'
  },
  '.cm-edit-diff-out-of-bounds-caption': {
    fontSize: '11px',
    color: 'rgba(140, 80, 0, 0.95)',
    marginBottom: '4px',
    fontWeight: 500
  },
  '.cm-edit-diff-block-container': {
    display: 'block',
    marginTop: '8px',
    marginBottom: '8px',
    border: '1px solid rgba(76, 175, 80, 0.35)',
    borderRadius: '4px',
    backgroundColor: 'rgba(76, 175, 80, 0.06)',
    overflow: 'hidden'
  },
  '.cm-edit-diff-block-header': {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '4px 8px',
    backgroundColor: 'rgba(76, 175, 80, 0.12)',
    borderBottom: '1px solid rgba(76, 175, 80, 0.2)',
    fontSize: '12px',
    fontWeight: 500
  },
  '.cm-edit-diff-block-content': {
    padding: '8px 10px',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    fontSize: 'inherit',
    fontFamily: 'inherit',
    maxHeight: 'none'
  }
});

// Combined widget for proposed text, info icon, and buttons
class DiffActionWidget extends WidgetType {
  constructor(operationId, text, note, onAccept, onReject, outOfBoundsWarning = '') {
    super();
    this.operationId = operationId;
    this.text = String(text || '');
    this.note = note || '';
    this.onAccept = onAccept;
    this.onReject = onReject;
    this.outOfBoundsWarning = outOfBoundsWarning || '';
  }
  
  eq(other) {
    return this.operationId === other.operationId && 
           this.text === other.text && 
           this.note === other.note &&
           this.outOfBoundsWarning === other.outOfBoundsWarning;
  }
  
  toDOM() {
    const container = document.createElement('span');
    container.className = 'cm-edit-diff-container';
    if (this.outOfBoundsWarning) {
      container.classList.add('cm-edit-diff-out-of-bounds');
    }
    container.style.cssText = 'display: inline-flex; flex-direction: column; align-items: flex-start; gap: 4px; vertical-align: middle;';
    
    // 0. Out-of-bounds caption (document changed since proposed)
    if (this.outOfBoundsWarning) {
      const caption = document.createElement('div');
      caption.className = 'cm-edit-diff-out-of-bounds-caption';
      caption.textContent = this.outOfBoundsWarning;
      container.appendChild(caption);
    }
    
    // 1. Proposed text (if any)
    const contentRow = document.createElement('span');
    contentRow.style.cssText = 'display: inline-flex; align-items: center; gap: 4px;';
    if (this.text) {
      const textSpan = document.createElement('span');
      textSpan.className = 'cm-edit-diff-addition';
      textSpan.textContent = this.text;
      textSpan.setAttribute('data-operation-id', this.operationId);
      contentRow.appendChild(textSpan);
    }
    
    // 2. Info icon (if note exists)
    if (this.note && this.note.trim().length > 0) {
      const infoIcon = document.createElement('span');
      infoIcon.className = 'cm-edit-diff-info';
      infoIcon.innerHTML = 'ℹ';
      infoIcon.title = this.note;
      infoIcon.setAttribute('aria-label', `Edit reason: ${this.note}`);
      
      // Hover tooltip logic
      let tooltip = null;
      infoIcon.addEventListener('mouseenter', () => {
        if (tooltip) return;
        tooltip = document.createElement('div');
        tooltip.className = 'cm-edit-diff-tooltip';
        tooltip.textContent = this.note;
        document.body.appendChild(tooltip);
        const rect = infoIcon.getBoundingClientRect();
        tooltip.style.left = `${rect.left}px`;
        tooltip.style.bottom = `${window.innerHeight - rect.top + 8}px`;
      });
      
      infoIcon.addEventListener('mouseleave', () => {
        if (tooltip) {
          document.body.removeChild(tooltip);
          tooltip = null;
        }
      });
      
      contentRow.appendChild(infoIcon);
    }
    
    // 3. Accept/Reject buttons
    const buttonsWrapper = document.createElement('span');
    buttonsWrapper.className = 'cm-edit-diff-buttons';
    
    const acceptBtn = document.createElement('button');
    acceptBtn.innerHTML = '✓';
    acceptBtn.className = 'cm-edit-diff-accept';
    acceptBtn.title = this.outOfBoundsWarning ? 'Accept (will apply at end of document)' : (this.note ? `Accept edit: ${this.note}` : 'Accept edit');
    acceptBtn.onclick = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (this.onAccept) this.onAccept(this.operationId);
    };
    
    const rejectBtn = document.createElement('button');
    rejectBtn.innerHTML = '✕';
    rejectBtn.className = 'cm-edit-diff-reject';
    rejectBtn.title = this.note ? `Reject edit: ${this.note}` : 'Reject edit';
    rejectBtn.onclick = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (this.onReject) this.onReject(this.operationId);
    };
    
    buttonsWrapper.appendChild(acceptBtn);
    buttonsWrapper.appendChild(rejectBtn);
    contentRow.appendChild(buttonsWrapper);
    container.appendChild(contentRow);
    
    return container;
  }
  
  ignoreEvent() {
    return false;
  }
}

// Widget for proposed addition text (kept for backward compatibility or simple use cases)
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

// Widget for info icon with tooltip showing operation note/reasoning
class DiffInfoWidget extends WidgetType {
  constructor(note) {
    super();
    this.note = note || '';
  }
  
  eq(other) {
    return this.note === other.note;
  }
  
  toDOM() {
    if (!this.note || this.note.trim().length === 0) {
      // Return empty span if no note
      return document.createElement('span');
    }
    
    const infoIcon = document.createElement('span');
    infoIcon.className = 'cm-edit-diff-info';
    infoIcon.innerHTML = 'ℹ';
    infoIcon.title = this.note;
    infoIcon.setAttribute('aria-label', `Edit reason: ${this.note}`);
    infoIcon.style.cssText = 'cursor: help; margin-left: 4px; color: #666; font-size: 12px; vertical-align: middle;';
    
    // Add hover tooltip
    let tooltip = null;
    infoIcon.addEventListener('mouseenter', (e) => {
      if (tooltip) return; // Already showing
      
      tooltip = document.createElement('div');
      tooltip.className = 'cm-edit-diff-tooltip';
      tooltip.textContent = this.note;
      tooltip.style.cssText = `
        position: absolute;
        background: #333;
        color: #fff;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        max-width: 300px;
        z-index: 10000;
        pointer-events: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
      `;
      
      document.body.appendChild(tooltip);
      
      // Position tooltip above the icon
      const rect = infoIcon.getBoundingClientRect();
      tooltip.style.left = `${rect.left}px`;
      tooltip.style.bottom = `${window.innerHeight - rect.top + 8}px`;
    });
    
    infoIcon.addEventListener('mouseleave', () => {
      if (tooltip) {
        document.body.removeChild(tooltip);
        tooltip = null;
      }
    });
    
    return infoIcon;
  }
  
  ignoreEvent() {
    return false;
  }
}

// Widget for accept/reject buttons
class DiffButtonWidget extends WidgetType {
  constructor(operationId, onAccept, onReject, note) {
    super();
    this.operationId = operationId;
    this.onAccept = onAccept;
    this.onReject = onReject;
    this.note = note || '';
  }
  
  eq(other) {
    return this.operationId === other.operationId && this.note === other.note;
  }
  
  toDOM() {
    const wrapper = document.createElement('span');
    wrapper.className = 'cm-edit-diff-buttons';
    
    // Accept button (checkmark)
    const acceptBtn = document.createElement('button');
    acceptBtn.innerHTML = '✓';
    acceptBtn.className = 'cm-edit-diff-accept';
    acceptBtn.title = this.note ? `Accept edit: ${this.note}` : 'Accept edit';
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
    rejectBtn.title = this.note ? `Reject edit: ${this.note}` : 'Reject edit';
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

// Thresholds for block-level rendering (large edits)
const BLOCK_THRESHOLD_LINES = 3;
const BLOCK_THRESHOLD_CHARS = 200;

function isLargeEdit(text) {
  if (!text || typeof text !== 'string') return false;
  const lines = text.split('\n').length;
  return lines >= BLOCK_THRESHOLD_LINES || text.length >= BLOCK_THRESHOLD_CHARS;
}

// Block-level widget for large insertions/replacements (full text, no collapse)
class DiffBlockWidget extends WidgetType {
  constructor(operationId, text, note, onAccept, onReject, outOfBoundsWarning = '') {
    super();
    this.operationId = operationId;
    this.text = String(text || '');
    this.note = note || '';
    this.onAccept = onAccept;
    this.onReject = onReject;
    this.outOfBoundsWarning = outOfBoundsWarning || '';
  }

  eq(other) {
    return this.operationId === other.operationId &&
      this.text === other.text &&
      this.note === other.note &&
      this.outOfBoundsWarning === other.outOfBoundsWarning;
  }

  toDOM() {
    const container = document.createElement('div');
    container.className = 'cm-edit-diff-block-container';

    const header = document.createElement('div');
    header.className = 'cm-edit-diff-block-header';
    const label = document.createElement('span');
    label.textContent = 'Proposed addition';
    if (this.outOfBoundsWarning) {
      label.textContent += ' (document changed)';
    }
    header.appendChild(label);
    const buttons = document.createElement('span');
    buttons.className = 'cm-edit-diff-buttons';
    const acceptBtn = document.createElement('button');
    acceptBtn.innerHTML = '✓';
    acceptBtn.className = 'cm-edit-diff-accept';
    acceptBtn.title = 'Accept edit';
    acceptBtn.onclick = (e) => { e.preventDefault(); e.stopPropagation(); if (this.onAccept) this.onAccept(this.operationId); };
    const rejectBtn = document.createElement('button');
    rejectBtn.innerHTML = '✕';
    rejectBtn.className = 'cm-edit-diff-reject';
    rejectBtn.title = 'Reject edit';
    rejectBtn.onclick = (e) => { e.preventDefault(); e.stopPropagation(); if (this.onReject) this.onReject(this.operationId); };
    buttons.appendChild(acceptBtn);
    buttons.appendChild(rejectBtn);
    header.appendChild(buttons);
    container.appendChild(header);

    const content = document.createElement('div');
    content.className = 'cm-edit-diff-block-content';
    content.textContent = this.text;
    container.appendChild(content);

    return container;
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

// Helper function to get plugin instance by documentId (for batch operations)
export function getLiveEditDiffPlugin(documentId) {
  return documentPluginRegistry.get(documentId) || null;
}

const LiveEditDiffPluginClass = class {
  constructor(view, documentId) {
    try {
      // ✅ SINGLETON PATTERN: If a plugin already exists for this document, reuse it
      if (documentId && documentPluginRegistry.has(documentId)) {
        const existingPlugin = documentPluginRegistry.get(documentId);
        console.log('🔍 Reusing existing plugin instance for document:', documentId, 'Registry size:', documentPluginRegistry.size);
        
        // Update the view reference
        existingPlugin.view = view;
        
        // ✅ FIX: Only schedule decoration update if there are operations AND decorations haven't been applied yet
        // This prevents duplicate decorations when extension is recreated (e.g., tab switch, editor remount)
        if (existingPlugin.operations.size > 0 && !existingPlugin.pendingUpdate) {
          console.log('🔍 Scheduling decoration update for reused plugin (operations:', existingPlugin.operations.size, ')');
          existingPlugin.scheduleDecorationUpdate();
        } else if (existingPlugin.operations.size > 0) {
          console.log('🔍 Skipping decoration update - already pending');
        }
        
        return existingPlugin;
      }

      console.log('🔍 Creating NEW plugin instance for document:', documentId, 'Registry size before:', documentPluginRegistry.size, 'Time:', Date.now());

      this.view = view;
      this.documentId = documentId || null; // Store document identity
      this.operations = new Map(); // operationId -> {from, to, original, proposed, opType, messageId, start, end}
      this.maxOperations = 150; // Limit concurrent diffs so large proposals (e.g. 94 ops) can be shown and applied
      this.currentMessageId = null; // Track current message ID to clear old operations
      this.decorations = Decoration.none;
      this.pendingUpdate = false; // Track if decoration update is queued
      this.updateTimeout = null; // Track pending timeout
      this.creationTime = Date.now(); // Track when this instance was created
      this.initialRender = true; // Track if this is the first decoration update
      this.decorationsApplied = false; // Track if decorations have been applied for current operations
      this.lastOperationsHash = ''; // Track hash of operations to detect changes

      // Only log in development mode
      if (import.meta.env.DEV) {
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
          // skipSave=true, preserveExisting=true - load all proposals without wiping any
          this.addOperations(savedDiffs.operations, savedDiffs.messageId, true, true);
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
      window.addEventListener('proposalAcceptDone', this.handleLiveDiffEvent);
      window.addEventListener('proposalRejectDone', this.handleLiveDiffEvent);
      
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
      window.removeEventListener('proposalAcceptDone', this.handleLiveDiffEvent);
      window.removeEventListener('proposalRejectDone', this.handleLiveDiffEvent);
    } catch (e) {
      console.error('❌ Error destroying live diff plugin:', e);
    }
  }

  _clearProposalOps(detail) {
    const { documentId: evDocId, operationIds, proposalId, clearAllForProposal } = detail || {};
    if (evDocId !== this.documentId) return;
    if (clearAllForProposal && proposalId) {
      const idsToRemove = [];
      this.operations.forEach((op, id) => {
        if (op.messageId === proposalId) idsToRemove.push(id);
      });
      idsToRemove.forEach((id) => this.operations.delete(id));
      if (this.documentId) documentDiffStore.removeProposalDiffs(this.documentId, proposalId);
      if (idsToRemove.length > 0) this.scheduleDecorationUpdate();
    } else if (Array.isArray(operationIds)) {
      operationIds.forEach((id) => {
        this.operations.delete(id);
        if (this.documentId) documentDiffStore.removeDiff(this.documentId, id);
      });
      this.scheduleDecorationUpdate();
    }
  }

  handleLiveDiffEvent(event) {
    try {
      // Only log in development mode
      if (import.meta.env.DEV) {
        console.log('🔍 Live diff extension handleLiveDiffEvent called:', event.type, event.detail);
      }
      if (event.type === 'editorOperationsLive') {
        const { operations, messageId, documentId, preserveExisting } = event.detail || {};
        
        // Only process operations for THIS document
        if (documentId && documentId !== this.documentId) {
          if (import.meta.env.DEV) {
            console.log(`Ignoring operations for different document (${documentId} vs ${this.documentId})`);
          }
          return;
        }
        
        if (import.meta.env.DEV) {
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
          // Clearing of old operations now happens in addOperations() (only when !preserveExisting)
          console.log('🔍 Calling addOperations with', operations.length, 'operations');
          this.addOperations(operations, messageId, false, preserveExisting ?? false);
          
          // ✅ NOTE: ChatSidebarContext saves to documentDiffStore BEFORE dispatching event
          // This ensures operations persist even if this document isn't currently open
          // Plugin will also save after generating operation IDs and when adjusting positions
        } else {
          console.warn('⚠️ No operations array or empty array received');
        }
      } else if (event.type === 'removeLiveDiff') {
        const { operationId } = event.detail || {};
        if (operationId) {
          this.removeOperation(operationId);
        }
      } else if (event.type === 'proposalAcceptDone') {
        this._clearProposalOps(event.detail);
      } else if (event.type === 'proposalRejectDone') {
        this._clearProposalOps(event.detail);
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
    if (update.docChanged && this.operations.size > 0) {
      const docLength = update.state.doc.length;
      const toRemove = [];
      let needsUpdate = false;

      if (update.changes && this.documentId) {
        update.changes.iterChanges((fromA, toA, fromB, toB, inserted) => {
          const editStart = fromA;
          const editEnd = toA;

          this.operations.forEach((op, id) => {
            const opStart = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
            const opEnd = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : opStart);
            const overlaps = !(editEnd < opStart || editStart > opEnd);

            if (overlaps) {
              const isLargeEdit = (editEnd - editStart > 500) || (editStart === 0 && editEnd === update.startState.doc.length);
              if (isLargeEdit && op.original) {
                const currentTextAtPos = update.state.doc.sliceString(opStart, opEnd);
                if (currentTextAtPos === op.original) {
                  if (import.meta.env.DEV) {
                    console.log(`Sync edit overlaps diff but content matches - keeping diff.`);
                  }
                  return;
                }
              }
              console.log(`User edit [${editStart}-${editEnd}] overlaps diff [${opStart}-${opEnd}] - invalidating. Message: ${op.messageId}`);
              toRemove.push(id);
              documentDiffStore.removeDiff(this.documentId, id);
            }
          });
        });
      }

      if (update.changes && this.operations.size > 0) {
        update.changes.iterChanges((fromA, toA, fromB, toB) => {
          const lenChange = (toB - fromB) - (toA - fromA);
          this.operations.forEach((op, id) => {
            if (toRemove.includes(id)) return;
            const opStart = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
            const opEnd = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : opStart);
            if (opStart >= toA) {
              op.start += lenChange;
              op.end += lenChange;
              op.from = op.start;
              op.to = op.end;
              needsUpdate = true;
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
      
      // ✅ CRITICAL FIX: Save adjusted positions to store so they persist across tab switches
      if (needsUpdate && this.documentId && this.operations.size > 0) {
        const adjustedOperations = Array.from(this.operations.values());
        const currentContent = update.state.doc.toString();
        documentDiffStore.saveSnapshot(this.documentId, adjustedOperations, currentContent);
      }
      
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
  
  addOperations(operations, messageId, skipSave = false, preserveExisting = false) {
    // ✅ FIX: Reset decorations applied flag when new operations are added
    this.decorationsApplied = false;
    
    // When messageId changes: either clear previous message's ops (writing-assistant) or keep all (multi-proposal)
    if (messageId && messageId !== this.currentMessageId) {
      this.currentMessageId = messageId;
      if (!preserveExisting) {
        // Writing-assistant behavior: clear ops from previous message
        const toRemove = [];
        this.operations.forEach((op, id) => {
          if (op.messageId && op.messageId !== messageId) {
            toRemove.push(id);
          }
        });
        if (toRemove.length > 0) {
          console.log(`🧹 Clearing ${toRemove.length} operations from previous message`);
          toRemove.forEach(id => this.operations.delete(id));
        }
      }
    } else if (!this.currentMessageId && messageId) {
      this.currentMessageId = messageId;
    }

    // Replace all ops for this proposal when re-syncing (e.g. after partial apply — fresh indices/positions)
    if (preserveExisting && messageId) {
      const toReplace = [];
      this.operations.forEach((op, id) => {
        if (op.messageId === messageId) toReplace.push(id);
      });
      toReplace.forEach((id) => this.operations.delete(id));
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
    if (import.meta.env.DEV) {
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
    
    let addedCount = 0;
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
      if (import.meta.env.DEV) {
        console.log('🔍 Processing operation:', { operationId, start, end, opType, proposedLength: proposed.length });
      }
      
      // Validate range - be more lenient for operations that might be slightly out of bounds
      // This can happen when document changes between operation creation and display
      const docLength = this.view.state.doc.length;
      if (start < 0 || end < start) {
        console.warn('Invalid operation range (negative or reversed):', { start, end, docLength });
        return;
      }
      
      // Way out of bounds: show at end of document so user can still accept/reject
      const wayOutOfBounds = start > docLength + 100 || end > docLength + 100;
      if (wayOutOfBounds) {
        console.warn('Operation range way out of bounds — showing at end of document:', { start, end, docLength });
        start = docLength;
        end = docLength;
      } else {
        // Clamp positions to valid range for storage
        start = Math.max(0, Math.min(docLength, start));
        end = Math.max(start, Math.min(docLength, end));
      }
      
      // For replace_range (when not out of bounds), verify original text matches
      if (!wayOutOfBounds && opType === 'replace_range' && original) {
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
        messageId: op.messageId || messageId,
        start: start,
        end: end,
        note: op.note || op.reasoning || '', // ✅ Store note/reasoning for explainable diffs
        outOfBounds: wayOutOfBounds || undefined,
        originalStart: wayOutOfBounds ? Number(op.start ?? 0) : undefined,
        originalEnd: wayOutOfBounds ? Number(op.end ?? op.start ?? 0) : undefined,
        proposal_operation_index: op.proposal_operation_index !== undefined && op.proposal_operation_index !== null
          ? Number(op.proposal_operation_index)
          : undefined
      });
      addedCount += 1;
    });
    
    // ✅ Update the centralized store ONLY when we actually added at least one operation.
    // If all ops were rejected (e.g. out of bounds), do NOT overwrite the store — leave
    // existing pending diffs so the badge and restore remain reliable.
    if (this.documentId && addedCount > 0 && !skipSave) {
      // Get current stored operations and update them with IDs
      const storedOperations = Array.from(this.operations.values());
      const currentContent = this.view.state.doc.toString();
      documentDiffStore.saveSnapshot(this.documentId, storedOperations, currentContent);
    } else if (skipSave) {
      console.log('⏭️ Skipped store update (restoring from storage)');
    }
    
    this.scheduleDecorationUpdate();
  }
  
  removeOperation(operationId) {
    this.operations.delete(operationId);
    // ✅ FIX: Reset decorations applied flag when operations are removed
    this.decorationsApplied = false;
    this.scheduleDecorationUpdate();
  }
  
  clearAllOperations() {
    this.operations.clear();
    // ✅ FIX: Reset decorations applied flag when all operations are cleared
    this.decorationsApplied = false;
    this.lastOperationsHash = '';
    this.scheduleDecorationUpdate();
  }
  
  _hashOperations() {
    // Create a hash of current operations to detect changes
    const opKeys = Array.from(this.operations.keys()).sort();
    return opKeys.join(',');
  }

  scheduleDecorationUpdate() {
    // ✅ FIX: Check if decorations are already applied for current operations
    const currentHash = this._hashOperations();
    if (this.decorationsApplied && this.lastOperationsHash === currentHash) {
      console.log('🔍 Skipping decoration update - already applied for these operations');
      return;
    }

    // Prevent multiple updates from being queued
    if (this.pendingUpdate) {
      return;
    }

    this.pendingUpdate = true;

    // Clear any existing timeout
    if (this.updateTimeout) {
      clearTimeout(this.updateTimeout);
    }

    // ALWAYS use requestAnimationFrame to avoid re-entrancy issues
    // This ensures we're not trying to update while CodeMirror is in an update cycle
    this.updateTimeout = requestAnimationFrame(() => {
      this.pendingUpdate = false;
      this.updateTimeout = null;
      this.initialRender = false; // Clear initial render flag after first update
      this.applyDecorationUpdate();
    });
  }
  
  applyDecorationUpdate() {
    try {
      const decos = [];
      const docLength = this.view.state.doc.length;
      
      this.operations.forEach((op, id) => {
        try {
          // Validate operation positions against current document
          const from = Math.max(0, Math.min(docLength, op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0)));
          const to = Math.max(from, Math.min(docLength, op.to !== undefined ? op.to : (op.end !== undefined ? op.end : from)));
          
          // Out-of-bounds: show only the widget at end of document (no mark)
          const outOfBounds = op.outOfBounds === true;
          const displayFrom = outOfBounds ? docLength : from;
          const displayTo = outOfBounds ? docLength : to;
          
          const outOfBoundsWarning = outOfBounds
            ? 'Document changed since proposed. Accept will apply at end; Reject removes.'
            : '';
          
          // Skip if positions are invalid (and not an out-of-bounds op placed at end)
          if (!outOfBounds && (displayFrom < 0 || displayTo < displayFrom || displayFrom > docLength || displayTo > docLength)) {
            console.warn('Skipping invalid operation:', id, { from: displayFrom, to: displayTo, docLength });
            return;
          }
          
          if (op.opType === 'replace_range') {
            // Only create mark decoration if from !== to and not out-of-bounds (mark decorations can't be empty)
            if (!outOfBounds && displayFrom !== displayTo) {
              decos.push(
                Decoration.mark({
                  class: 'cm-edit-diff-replacement',
                  attributes: { 'data-operation-id': id }
                }).range(displayFrom, displayTo)
              );
            }
            const useBlock = isLargeEdit(op.proposed);
            decos.push(
              Decoration.widget({
                widget: useBlock
                  ? new DiffBlockWidget(
                      id,
                      op.proposed,
                      op.note || op.reasoning || '',
                      () => this.acceptOperation(id),
                      () => this.rejectOperation(id),
                      outOfBoundsWarning
                    )
                  : new DiffActionWidget(
                      id,
                      op.proposed,
                      op.note || op.reasoning || '',
                      () => this.acceptOperation(id),
                      () => this.rejectOperation(id),
                      outOfBoundsWarning
                    ),
                side: 1,
                block: useBlock
              }).range(displayTo)
            );
          } else if (op.opType === 'delete_range') {
            // Only create mark decoration if from !== to and not out-of-bounds
            if (!outOfBounds && displayFrom !== displayTo) {
              decos.push(
                Decoration.mark({
                  class: 'cm-edit-diff-deletion',
                  attributes: { 'data-operation-id': id }
                }).range(displayFrom, displayTo)
              );
            }
            
            // Add action group widget (buttons + info, no addition text)
            decos.push(
              Decoration.widget({
                widget: new DiffActionWidget(
                  id,
                  '',
                  op.note || op.reasoning || '',
                  () => this.acceptOperation(id),
                  () => this.rejectOperation(id),
                  outOfBoundsWarning
                ),
                side: 1
              }).range(displayTo)
            );
          } else if (op.opType === 'insert_after_heading' || op.opType === 'insert_after') {
            const useBlock = isLargeEdit(op.proposed);
            decos.push(
              Decoration.widget({
                widget: useBlock
                  ? new DiffBlockWidget(
                      id,
                      op.proposed,
                      op.note || op.reasoning || '',
                      () => this.acceptOperation(id),
                      () => this.rejectOperation(id),
                      outOfBoundsWarning
                    )
                  : new DiffActionWidget(
                      id,
                      op.proposed,
                      op.note || op.reasoning || '',
                      () => this.acceptOperation(id),
                      () => this.rejectOperation(id),
                      outOfBoundsWarning
                    ),
                side: 1,
                block: useBlock
              }).range(displayFrom)
            );
          }
        } catch (e) {
          console.warn('Error creating decoration for operation:', id, e);
        }
      });
      
      // Decoration.set(decos, true) lets CodeMirror sort by from/startSide (required for RangeSet).
      const decorationSet = Decoration.set(decos, true);
      this.decorations = decorationSet;
      
      // CRITICAL: Only dispatch if view is still valid
      if (this.view && !this.view.isDestroyed) {
        // Use requestAnimationFrame to ensure we are outside of the current update cycle
        // and Promise.resolve().then() as a fallback/additional layer of safety
        requestAnimationFrame(() => {
          if (this.view && !this.view.isDestroyed) {
            this.view.dispatch({
              effects: setLiveDiffDecorations.of(decorationSet)
            });
            
            // ✅ FIX: Mark decorations as applied and save operations hash
            this.decorationsApplied = true;
            this.lastOperationsHash = this._hashOperations();
            console.log('🔍 Decorations applied successfully, hash:', this.lastOperationsHash);
          }
        });
      }
    } catch (e) {
      console.error('❌ Error applying decoration update:', e);
      this.decorations = Decoration.none;
      this.decorationsApplied = false;
    }
  }
  
  acceptOperation(operationId) {
    const op = this.operations.get(operationId);
    if (!op) return;
    const proposalId = op.messageId;
    if (!proposalId) {
      console.warn('acceptOperation: no proposalId on operation, removing locally');
      this.operations.delete(operationId);
      if (this.documentId) documentDiffStore.removeDiff(this.documentId, operationId);
      this.scheduleDecorationUpdate();
      return;
    }
    const proposalOperationIndex = op.proposal_operation_index;
    window.dispatchEvent(new CustomEvent('proposalAccept', {
      detail: {
        documentId: this.documentId,
        proposalId,
        operationId,
        proposalOperationIndex: proposalOperationIndex !== undefined ? proposalOperationIndex : undefined
      }
    }));
  }
  
  rejectOperation(operationId) {
    const op = this.operations.get(operationId);
    if (!op) return;
    const proposalId = op.messageId;
    if (!proposalId) {
      console.warn('rejectOperation: no proposalId on operation, removing locally');
      this.operations.delete(operationId);
      if (this.documentId) documentDiffStore.removeDiff(this.documentId, operationId);
      this.scheduleDecorationUpdate();
      return;
    }
    window.dispatchEvent(new CustomEvent('proposalReject', {
      detail: { documentId: this.documentId, proposalId, operationId }
    }));
  }
  
  acceptAllOperations() {
    const proposalIds = this._getUniqueProposalIds();
    if (proposalIds.length > 0) {
      window.dispatchEvent(new CustomEvent('proposalAcceptAll', {
        detail: { documentId: this.documentId, proposalIds }
      }));
      return;
    }
    if (this.operations.size === 0) return;
    for (const operationId of Array.from(this.operations.keys())) {
      this.acceptOperation(operationId);
    }
  }

  rejectAllOperations() {
    const proposalIds = this._getUniqueProposalIds();
    if (proposalIds.length > 0) {
      window.dispatchEvent(new CustomEvent('proposalRejectAll', {
        detail: { documentId: this.documentId, proposalIds }
      }));
      return;
    }
    if (this.operations.size === 0) return;
    for (const operationId of Array.from(this.operations.keys())) {
      this.rejectOperation(operationId);
    }
  }

  _getUniqueProposalIds() {
    const ids = new Set();
    this.operations.forEach((op) => {
      if (op.messageId) ids.add(op.messageId);
    });
    if (ids.size === 0 && this.operations.size > 0) {
      console.warn('_getUniqueProposalIds: operations exist but none have messageId');
    }
    return Array.from(ids);
  }
  
  /**
   * Find the next diff position after the current cursor
   * @param {number} currentPosition - Current cursor position
   * @returns {Object|null} { operationId, position } or null if no next diff
   */
  findNextDiff(currentPosition = 0) {
    if (this.operations.size === 0) return null;
    
    const docLength = this.view.state.doc.length;
    const cursorPos = Math.max(0, Math.min(docLength, currentPosition));
    
    // Get all operation positions
    const positions = Array.from(this.operations.entries()).map(([id, op]) => {
      const start = op.start !== undefined ? op.start : op.from || 0;
      const end = op.end !== undefined ? op.end : op.to || start;
      return {
        id,
        start: Math.max(0, Math.min(docLength, start)),
        end: Math.max(0, Math.min(docLength, end))
      };
    }).filter(pos => pos.start >= 0 && pos.end >= pos.start);
    
    if (positions.length === 0) return null;
    
    // Find next position after cursor
    const nextPos = positions
      .filter(pos => pos.start > cursorPos)
      .sort((a, b) => a.start - b.start)[0];
    
    // If no next position, wrap to first
    const target = nextPos || positions.sort((a, b) => a.start - b.start)[0];
    
    return {
      operationId: target.id,
      position: target.start
    };
  }
  
  /**
   * Find the previous diff position before the current cursor
   * @param {number} currentPosition - Current cursor position
   * @returns {Object|null} { operationId, position } or null if no previous diff
   */
  findPreviousDiff(currentPosition = 0) {
    if (this.operations.size === 0) return null;
    
    const docLength = this.view.state.doc.length;
    const cursorPos = Math.max(0, Math.min(docLength, currentPosition));
    
    // Get all operation positions
    const positions = Array.from(this.operations.entries()).map(([id, op]) => {
      const start = op.start !== undefined ? op.start : op.from || 0;
      const end = op.end !== undefined ? op.end : op.to || start;
      return {
        id,
        start: Math.max(0, Math.min(docLength, start)),
        end: Math.max(0, Math.min(docLength, end))
      };
    }).filter(pos => pos.start >= 0 && pos.end >= pos.start);
    
    if (positions.length === 0) return null;
    
    // Find previous position before cursor
    const prevPos = positions
      .filter(pos => pos.start < cursorPos)
      .sort((a, b) => b.start - a.start)[0];
    
    // If no previous position, wrap to last
    const target = prevPos || positions.sort((a, b) => b.start - a.start)[0];
    
    return {
      operationId: target.id,
      position: target.start
    };
  }
  
  /**
   * Jump to a specific diff position
   * @param {number} position - Position to jump to
   */
  jumpToPosition(position) {
    if (!this.view || this.view.isDestroyed) return;
    
    const docLength = this.view.state.doc.length;
    const targetPos = Math.max(0, Math.min(docLength, position));
    
    // Scroll to position with margin
    this.view.dispatch({
      effects: EditorView.scrollIntoView(targetPos, { y: 'center', yMargin: 100 })
    });
    
    // Set cursor position
    this.view.dispatch({
      selection: { anchor: targetPos, head: targetPos }
    });
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
      ViewPlugin.fromClass(DocumentAwareDiffPlugin)
    ];
    console.log('🔍 Live edit diff extension created:', extension.length, 'items');
    return extension;
  } catch (e) {
    console.error('❌ Error creating live edit diff extension:', e);
    return [];
  }
}

