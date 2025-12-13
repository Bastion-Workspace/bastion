import { EditorView, ViewPlugin, Decoration, WidgetType } from '@codemirror/view';
import { StateField, StateEffect } from '@codemirror/state';

// State effects for managing suggestions
const addSuggestion = StateEffect.define();
const removeSuggestion = StateEffect.define();
const clearAllSuggestions = StateEffect.define();

// Theme for suggestion styling
const suggestionTheme = EditorView.baseTheme({
  '.cm-edit-suggestion': {
    backgroundColor: 'rgba(100, 181, 246, 0.15)',
    borderBottom: '1px dashed rgba(100, 181, 246, 0.4)',
    borderRadius: '2px',
    padding: '0 1px'
  },
  '.cm-edit-suggestion-buttons': {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '2px',
    marginLeft: '4px',
    verticalAlign: 'middle'
  },
  '.cm-edit-accept, .cm-edit-reject': {
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
  '.cm-edit-accept': {
    backgroundColor: '#4caf50',
    color: 'white'
  },
  '.cm-edit-accept:hover': {
    backgroundColor: '#45a049'
  },
  '.cm-edit-reject': {
    backgroundColor: '#f44336',
    color: 'white'
  },
  '.cm-edit-reject:hover': {
    backgroundColor: '#da190b'
  }
});

// Widget for accept/reject buttons
class EditSuggestionWidget extends WidgetType {
  constructor(suggestionId, onAccept, onReject) {
    super();
    this.suggestionId = suggestionId;
    this.onAccept = onAccept;
    this.onReject = onReject;
  }
  
  eq(other) {
    return this.suggestionId === other.suggestionId;
  }
  
  toDOM() {
    const wrapper = document.createElement('span');
    wrapper.className = 'cm-edit-suggestion-buttons';
    
    // Accept button (checkmark)
    const acceptBtn = document.createElement('button');
    acceptBtn.innerHTML = '✓';
    acceptBtn.className = 'cm-edit-accept';
    acceptBtn.title = 'Accept suggestion';
    acceptBtn.setAttribute('aria-label', 'Accept suggestion');
    acceptBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.onAccept(this.suggestionId);
    };
    
    // Reject button (X)
    const rejectBtn = document.createElement('button');
    rejectBtn.innerHTML = '✕';
    rejectBtn.className = 'cm-edit-reject';
    rejectBtn.title = 'Reject suggestion';
    rejectBtn.setAttribute('aria-label', 'Reject suggestion');
    rejectBtn.onclick = (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.onReject(this.suggestionId);
    };
    
    wrapper.appendChild(acceptBtn);
    wrapper.appendChild(rejectBtn);
    return wrapper;
  }
  
  ignoreEvent() {
    return false; // Handle click events
  }
}

// State field for suggestions
const suggestionField = StateField.define({
  create() {
    return Decoration.none;
  },
  update(decorations, tr) {
    decorations = decorations.map(tr.changes);
    
    for (const effect of tr.effects) {
      if (effect.is(clearAllSuggestions)) {
        return Decoration.none;
      }
    }
    
    return decorations;
  },
  provide: f => EditorView.decorations.from(f)
});

// Plugin to manage suggestions
const inlineEditSuggestionsPlugin = ViewPlugin.fromClass(class {
  constructor(view) {
    this.view = view;
    this.suggestions = new Map(); // suggestionId -> {from, to, original, suggested, proposalId}
    this.acceptCallbacks = new Map(); // suggestionId -> callback
    this.rejectCallbacks = new Map(); // suggestionId -> callback
    this.autoAcceptTimeout = null;
    
    // Listen for suggestion events
    this.handleSuggestionEvent = this.handleSuggestionEvent.bind(this);
    window.addEventListener('inlineEditSuggestion', this.handleSuggestionEvent);
    window.addEventListener('inlineEditSuggestionRemove', this.handleSuggestionEvent);
  }
  
  destroy() {
    window.removeEventListener('inlineEditSuggestion', this.handleSuggestionEvent);
    window.removeEventListener('inlineEditSuggestionRemove', this.handleSuggestionEvent);
    if (this.autoAcceptTimeout) {
      clearTimeout(this.autoAcceptTimeout);
    }
  }
  
  handleSuggestionEvent(event) {
    if (event.type === 'inlineEditSuggestion') {
      const { suggestionId, from, to, original, suggested, proposalId, onAccept, onReject } = event.detail;
      this.addSuggestion(suggestionId, from, to, original, suggested, proposalId, onAccept, onReject);
    } else if (event.type === 'inlineEditSuggestionRemove') {
      const { suggestionId } = event.detail;
      this.removeSuggestion(suggestionId);
    }
  }
  
  update(update) {
    // Handle state effects - check if effects is iterable
    if (update.effects && typeof update.effects[Symbol.iterator] === 'function') {
      for (const effect of update.effects) {
        if (effect.is(addSuggestion)) {
          const { suggestionId, from, to, original, suggested, proposalId, onAccept, onReject } = effect.value;
          this.addSuggestion(suggestionId, from, to, original, suggested, proposalId, onAccept, onReject);
        } else if (effect.is(removeSuggestion)) {
          this.removeSuggestion(effect.value);
        } else if (effect.is(clearAllSuggestions)) {
          this.suggestions.clear();
          this.acceptCallbacks.clear();
          this.rejectCallbacks.clear();
        }
      }
    }
    
    // Auto-accept suggestions if user edits the document (continues typing)
    // Only if the suggestion range hasn't been manually edited
    if (update.docChanged && this.suggestions.size > 0) {
      const currentDoc = update.state.doc;
      this.suggestions.forEach((suggestion, id) => {
        try {
          const currentText = currentDoc.sliceString(suggestion.from, suggestion.to);
          // If original text still matches, user is continuing elsewhere - auto-accept
          if (currentText === suggestion.original) {
            // Schedule auto-accept after a short delay
            setTimeout(() => {
              if (this.suggestions.has(id)) {
                this.acceptSuggestion(id);
              }
            }, 100);
          } else if (currentText !== suggestion.suggested) {
            // User manually edited - remove suggestion
            this.removeSuggestion(id);
          }
        } catch (e) {
          // Range invalid - remove suggestion
          this.removeSuggestion(id);
        }
      });
    }
    
    this.updateDecorations();
  }
  
  addSuggestion(suggestionId, from, to, original, suggested, proposalId, onAccept, onReject) {
    // Validate range
    if (from < 0 || to < from || to > this.view.state.doc.length) {
      console.warn('Invalid suggestion range:', { from, to, docLength: this.view.state.doc.length });
      return;
    }
    
    // Check if current text matches original
    const currentText = this.view.state.doc.sliceString(from, to);
    if (currentText !== original) {
      console.warn('Suggestion original text does not match current document:', {
        expected: original,
        actual: currentText
      });
      // Still add it, but mark as potentially stale
    }
    
    this.suggestions.set(suggestionId, {
      from,
      to,
      original,
      suggested,
      proposalId
    });
    
    if (onAccept) {
      this.acceptCallbacks.set(suggestionId, onAccept);
    }
    if (onReject) {
      this.rejectCallbacks.set(suggestionId, onReject);
    }
    
    // Auto-accept after 30 seconds if not interacted with
    if (this.autoAcceptTimeout) {
      clearTimeout(this.autoAcceptTimeout);
    }
    this.autoAcceptTimeout = setTimeout(() => {
      this.suggestions.forEach((_, id) => {
        this.acceptSuggestion(id);
      });
    }, 30000); // 30 seconds
    
    this.updateDecorations();
  }
  
  removeSuggestion(suggestionId) {
    this.suggestions.delete(suggestionId);
    this.acceptCallbacks.delete(suggestionId);
    this.rejectCallbacks.delete(suggestionId);
    this.updateDecorations();
  }
  
  updateDecorations() {
    const decos = [];
    
    this.suggestions.forEach((suggestion, id) => {
      try {
        // Shade the original text
        decos.push(
          Decoration.mark({
            class: 'cm-edit-suggestion',
            attributes: { 'data-suggestion-id': id }
          }).range(suggestion.from, suggestion.to)
        );
        
        // Add accept/reject buttons after the suggestion
        decos.push(
          Decoration.widget({
            widget: new EditSuggestionWidget(
              id,
              () => this.acceptSuggestion(id),
              () => this.rejectSuggestion(id)
            ),
            side: 1
          }).range(suggestion.to)
        );
      } catch (e) {
        console.warn('Error creating decoration for suggestion:', id, e);
      }
    });
    
    this.view.dispatch({
      effects: suggestionField.reconfigure(Decoration.set(decos))
    });
  }
  
  acceptSuggestion(suggestionId) {
    const suggestion = this.suggestions.get(suggestionId);
    if (!suggestion) return;
    
    // Apply the suggestion
    this.view.dispatch({
      changes: {
        from: suggestion.from,
        to: suggestion.to,
        insert: suggestion.suggested
      }
    });
    
    // Call accept callback if provided
    const callback = this.acceptCallbacks.get(suggestionId);
    if (callback) {
      try {
        callback(suggestion.proposalId, suggestionId);
      } catch (e) {
        console.error('Error in accept callback:', e);
      }
    }
    
    // Remove suggestion
    this.removeSuggestion(suggestionId);
    
    // Dispatch event for external handling
    window.dispatchEvent(new CustomEvent('inlineEditSuggestionAccepted', {
      detail: { suggestionId, proposalId: suggestion.proposalId }
    }));
  }
  
  rejectSuggestion(suggestionId) {
    const suggestion = this.suggestions.get(suggestionId);
    if (!suggestion) return;
    
    // Call reject callback if provided
    const callback = this.rejectCallbacks.get(suggestionId);
    if (callback) {
      try {
        callback(suggestion.proposalId, suggestionId);
      } catch (e) {
        console.error('Error in reject callback:', e);
      }
    }
    
    // Remove suggestion
    this.removeSuggestion(suggestionId);
    
    // Dispatch event for external handling
    window.dispatchEvent(new CustomEvent('inlineEditSuggestionRejected', {
      detail: { suggestionId, proposalId: suggestion.proposalId }
    }));
  }
});

/**
 * Create inline edit suggestions extension for CodeMirror
 * 
 * Usage:
 * ```javascript
 * const extension = createInlineEditSuggestionsExtension();
 * // Add to CodeMirror extensions array
 * ```
 * 
 * To add a suggestion:
 * ```javascript
 * window.dispatchEvent(new CustomEvent('inlineEditSuggestion', {
 *   detail: {
 *     suggestionId: 'unique-id',
 *     from: 100, // character position
 *     to: 150,
 *     original: 'old text',
 *     suggested: 'new text',
 *     proposalId: 'proposal-id',
 *     onAccept: (proposalId, suggestionId) => { // handle accept },
 *     onReject: (proposalId, suggestionId) => { // handle reject }
 *   }
 * }));
 * ```
 */
export function createInlineEditSuggestionsExtension() {
  return [
    suggestionTheme,
    suggestionField,
    inlineEditSuggestionsPlugin
  ];
}

