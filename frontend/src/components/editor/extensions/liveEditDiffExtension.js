import { EditorView, ViewPlugin, Decoration, WidgetType } from '@codemirror/view';
import { StateField, StateEffect } from '@codemirror/state';

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

// Plugin to manage live diffs
const liveEditDiffPlugin = ViewPlugin.fromClass(class {
  constructor(view) {
    try {
      this.view = view;
      this.operations = new Map(); // operationId -> {from, to, original, proposed, opType, messageId, start, end}
      this.maxOperations = 10; // Limit concurrent diffs
      this.decorations = Decoration.none;
      this.pendingUpdate = false; // Track if decoration update is queued
      this.updateTimeout = null; // Track pending timeout
      
      console.log('🔍 Live diff extension plugin initialized for view:', view);
      
      // Listen for live diff events
      this.handleLiveDiffEvent = this.handleLiveDiffEvent.bind(this);
      window.addEventListener('editorOperationsLive', this.handleLiveDiffEvent);
      window.addEventListener('removeLiveDiff', this.handleLiveDiffEvent);
      window.addEventListener('clearEditorDiffs', this.handleLiveDiffEvent);
      
      // Initial decoration update (deferred)
      this.scheduleDecorationUpdate();
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
      
      window.removeEventListener('editorOperationsLive', this.handleLiveDiffEvent);
      window.removeEventListener('removeLiveDiff', this.handleLiveDiffEvent);
      window.removeEventListener('clearEditorDiffs', this.handleLiveDiffEvent);
    } catch (e) {
      console.error('❌ Error destroying live diff plugin:', e);
    }
  }
  
  handleLiveDiffEvent(event) {
    try {
      console.log('🔍 Live diff extension handleLiveDiffEvent called:', event.type, event.detail);
      if (event.type === 'editorOperationsLive') {
        const { operations, messageId } = event.detail || {};
        console.log('🔍 Live diff extension received editorOperationsLive:', { 
          operationsCount: operations?.length, 
          messageId,
          firstOp: operations?.[0],
          allOps: operations
        });
        if (Array.isArray(operations) && operations.length > 0) {
          console.log('🔍 Calling addOperations with', operations.length, 'operations');
          this.addOperations(operations, messageId);
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
  
  update(update) {
    // OPTION 1: Validation disabled for live edits
    // Operations remain visible until explicitly accepted or rejected
    // This prevents operations from disappearing when document changes
    // 
    // NOTE: If user manually edits the document, decorations may point to wrong positions.
    // In practice, users typically don't edit while live suggestions are pending.
    try {
      // No validation - trust operations remain valid until user acts on them
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
    // Limit to maxOperations
    const currentCount = this.operations.size;
    const toAdd = operations.slice(0, Math.max(0, this.maxOperations - currentCount));
    
    // Get frontmatter boundary
    const docText = this.view.state.doc.toString();
    const frontmatterEnd = this._getFrontmatterEnd(docText);
    
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
    
    toAdd.forEach((op, idx) => {
      const operationId = `${messageId || 'op'}-${idx}-${Date.now()}`;
      let start = Number(op.start || 0);
      let end = Number(op.end || start);
      const opType = op.op_type || 'replace_range';
      const original = op.original_text || op.anchor_text || '';
      const proposed = op.text || '';
      
      // CRITICAL: Guard frontmatter - ensure operations never occur before frontmatter end
      if (start < frontmatterEnd) {
        console.warn('⚠️ Operation before frontmatter detected, adjusting:', { 
          originalStart: start, 
          frontmatterEnd,
          opType 
        });
        // For insertions, move to after frontmatter
        if (start === end || opType === 'insert_after_heading') {
          start = frontmatterEnd;
          end = frontmatterEnd;
        } else {
          // For replace/delete, clamp start to frontmatter end
          start = frontmatterEnd;
          end = Math.max(end, start);
        }
      }
      
      console.log('🔍 Processing operation:', { operationId, start, end, opType, proposedLength: proposed.length });
      
      // Validate range
      if (start < 0 || end < start || end > this.view.state.doc.length) {
        console.warn('Invalid operation range:', { start, end, docLength: this.view.state.doc.length });
        return;
      }
      
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
    
    // Defer the update to avoid dispatching during CodeMirror's update cycle
    this.updateTimeout = setTimeout(() => {
      this.pendingUpdate = false;
      this.updateTimeout = null;
      this.applyDecorationUpdate();
    }, 0);
  }
  
  applyDecorationUpdate() {
    try {
      const decos = [];
      
      this.operations.forEach((op, id) => {
        try {
          if (op.opType === 'replace_range') {
            // Mark original text with deletion styling
            decos.push(
              Decoration.mark({
                class: 'cm-edit-diff-replacement',
                attributes: { 'data-operation-id': id }
              }).range(op.from, op.to)
            );
            
            // Add proposed text as widget after the range
            if (op.proposed && op.proposed.length > 0) {
              decos.push(
                Decoration.widget({
                  widget: new DiffAdditionWidget(op.proposed, id),
                  side: 1
                }).range(op.to)
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
              }).range(op.to)
            );
          } else if (op.opType === 'delete_range') {
            // Mark range with deletion styling
            decos.push(
              Decoration.mark({
                class: 'cm-edit-diff-deletion',
                attributes: { 'data-operation-id': id }
              }).range(op.from, op.to)
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
              }).range(op.to)
            );
          } else if (op.opType === 'insert_after_heading') {
            // For insert operations, show proposed text at the insertion point
            if (op.proposed && op.proposed.length > 0) {
              decos.push(
                Decoration.widget({
                  widget: new DiffAdditionWidget(op.proposed, id),
                  side: 1
                }).range(op.from)
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
                }).range(op.from)
              );
            }
          }
        } catch (e) {
          console.warn('Error creating decoration for operation:', id, e);
        }
      });
      
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
    
    // CRITICAL: Remove the accepted operation from the map FIRST
    // This prevents it from being affected by position adjustments
    this.operations.delete(operationId);
    
    // Calculate position delta for adjusting other operations
    let positionDelta = 0;
    const opStart = op.start !== undefined ? op.start : op.from;
    const opEnd = op.end !== undefined ? op.end : op.to;
    
    // Build operation object for applyOperations handler
    let operationObj;
    if (op.opType === 'insert_after_heading') {
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
    
    // Emit event for external handling (MarkdownCMEditor will apply the change)
    window.dispatchEvent(new CustomEvent('liveEditAccepted', {
      detail: {
        operationId: operationId,
        operation: operationObj
      }
    }));
    
    // Schedule a single decoration update to reflect all changes
    this.scheduleDecorationUpdate();
  }
  
  rejectOperation(operationId) {
    const op = this.operations.get(operationId);
    if (!op) return;
    
    // Emit event for external handling
    window.dispatchEvent(new CustomEvent('liveEditRejected', {
      detail: {
        operationId: operationId
      }
    }));
    
    // Remove from active diffs
    this.removeOperation(operationId);
  }
});

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
 * const extension = createLiveEditDiffExtension();
 * // Add to CodeMirror extensions array
 * ```
 * 
 * To trigger live diffs:
 * ```javascript
 * window.dispatchEvent(new CustomEvent('editorOperationsLive', {
 *   detail: {
 *     operations: [{ start: 100, end: 150, text: 'new text', op_type: 'replace_range' }],
 *     messageId: 'message-id'
 *   }
 * }));
 * ```
 */
export function createLiveEditDiffExtension() {
  try {
    console.log('🔍 Creating live edit diff extension');
    const extension = [
      diffTheme,
      liveDiffField,
      liveEditDiffPlugin
    ];
    console.log('🔍 Live edit diff extension created:', extension.length, 'items');
    return extension;
  } catch (e) {
    console.error('❌ Error creating live edit diff extension:', e);
    return [];
  }
}

