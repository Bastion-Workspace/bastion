import { EditorView, ViewPlugin, Decoration, keymap, WidgetType } from '@codemirror/view';
import { StateField, StateEffect, Prec } from '@codemirror/state';

const ghostTheme = EditorView.baseTheme({
  '.cm-ghostText': { opacity: 0.45 }
});

export function createGhostTextExtension(fetchSuggestionFn, options = {}) {
  const debounceMs = Math.max(0, typeof options.debounceMs === 'number' ? options.debounceMs : 350);
  const setGhost = StateEffect.define();
  const ghostField = StateField.define({
    create() { return Decoration.none; },
    update(value, tr) {
      for (const e of tr.effects) if (e.is(setGhost)) return e.value;
      // Preserve ghost across selection/focus changes; clear only on doc changes
      if (tr.docChanged) return Decoration.none;
      return value;
    },
    provide: f => EditorView.decorations.from(f)
  });

  class GhostWidget extends WidgetType {
    constructor(text) {
      super();
      this.text = String(text || '');
    }
    eq(other) {
      return this.text === other.text;
    }
    toDOM() {
      const el = document.createElement('span');
      el.className = 'cm-ghostText';
      el.textContent = this.text;
      el.setAttribute('aria-hidden', 'true');
      // Mobile/touch acceptance: tap the ghost to accept
      const fireAccept = (ev) => {
        try { ev.preventDefault(); } catch {}
        try { window.dispatchEvent(new CustomEvent('codexGhostAccept')); } catch {}
      };
      el.style.cursor = 'pointer';
      el.addEventListener('mousedown', fireAccept);
      el.addEventListener('click', fireAccept);
      el.addEventListener('touchstart', fireAccept, { passive: false });
      el.addEventListener('touchend', fireAccept, { passive: false });
      return el;
    }
    destroy() {
      // no-op: required to satisfy CM6 lifecycle expectations
    }
  }

  const plugin = ViewPlugin.fromClass(class {
    constructor(view) {
      this.view = view;
      this.suggestion = '';
      this.abortController = null;
      this.lastPos = -1;
      this.debounceHandle = null;
      this._handleAccept = this._handleAccept.bind(this);
      this._handleDismiss = this._handleDismiss.bind(this);
      try { window.addEventListener('codexGhostAccept', this._handleAccept); } catch {}
      try { window.addEventListener('codexGhostDismiss', this._handleDismiss); } catch {}
      this.updateNow(view);
    }
    destroy() {
      if (this.abortController) this.abortController.abort();
      try { if (this.debounceHandle) clearTimeout(this.debounceHandle); } catch {}
      try { window.removeEventListener('codexGhostAccept', this._handleAccept); } catch {}
      try { window.removeEventListener('codexGhostDismiss', this._handleDismiss); } catch {}
    }
    update(update) {
      if (update.docChanged || update.selectionSet) {
        this.schedule(update.view);
      }
    }
    _handleAccept() {
      const vp = this;
      if (!vp || !vp.suggestion || vp.suggestion.length === 0) return;
      const view = vp.view;
      const pos = view.state.selection.main.head;
      // Insert suggestion as-is without forcing an extra trailing space to reduce cursor jump
      const insertText = vp.suggestion;
      const tr = view.state.update({ changes: { from: pos, to: pos, insert: insertText }, selection: { anchor: pos + insertText.length, head: pos + insertText.length } });
      view.dispatch(tr);
      vp.clear();
    }
    _handleDismiss() {
      const vp = this;
      if (!vp) return;
      vp.clear();
    }
    schedule(view) {
      try { if (this.debounceHandle) clearTimeout(this.debounceHandle); } catch {}
      const cb = () => this.updateNow(view);
      this.debounceHandle = setTimeout(cb, debounceMs);
    }
    async updateNow(view) {
      try {
        const sel = view.state.selection.main;
        if (!sel.empty) return this.clear();
        const pos = sel.head;
        // Insertion-only: only suggest at end-of-line
        const line = view.state.doc.lineAt(pos);
        if (pos !== line.to) return this.clear();
        if (pos === this.lastPos && this.suggestion) return; // unchanged
        this.lastPos = pos;

        const prefix = view.state.sliceDoc(Math.max(0, pos - 8000), pos);
        const suffix = view.state.sliceDoc(pos, Math.min(view.state.doc.length, pos + 600));
        // Pull latest editor context for filename/frontmatter without tying to React state
        let frontmatter = {};
        let filename = undefined;
        try {
          const raw = localStorage.getItem('editor_ctx_cache');
          if (raw) {
            const ctx = JSON.parse(raw);
            frontmatter = ctx?.frontmatter || {};
            filename = ctx?.filename;
          }
        } catch {}

        if (this.abortController) this.abortController.abort();
        this.abortController = new AbortController();
        const signal = this.abortController.signal;
        const text = await fetchSuggestionFn({ prefix, suffix, position: pos, signal, frontmatter, filename }).catch(() => '');
        if (signal.aborted) return; // superseded
        this.suggestion = (text || '').replace(/^\s+/, '');
        this.applyDeco();
      } catch (e) {
        this.clear();
      }
    }
    applyDeco() {
      const pos = this.view.state.selection.main.head;
      const decos = (!this.suggestion || this.suggestion.length === 0)
        ? Decoration.none
        : Decoration.set([
            Decoration.widget({
              side: 1,
              widget: new GhostWidget(this.suggestion)
            }).range(pos)
          ]);
      this.view.dispatch({ effects: setGhost.of(decos) });
      try {
        const rect = this.view.coordsAtPos(pos);
        if (rect && this.suggestion && this.suggestion.length > 0) {
          window.dispatchEvent(new CustomEvent('codexGhostState', { detail: { has: true, x: rect.right, y: rect.bottom } }));
        } else {
          window.dispatchEvent(new CustomEvent('codexGhostState', { detail: { has: false } }));
        }
      } catch {}
    }
    clear() {
      this.suggestion = '';
      try { this.view.dispatch({ effects: setGhost.of(Decoration.none) }); } catch {}
      try { window.dispatchEvent(new CustomEvent('codexGhostState', { detail: { has: false } })); } catch {}
    }
  });

  const acceptKeymap = Prec.highest(keymap.of([{
    key: 'Tab',
    preventDefault: true,
    run(view) {
      const vp = view.plugin(plugin);
      if (vp && vp.suggestion && vp.suggestion.length > 0) {
        const pos = view.state.selection.main.head;
        // Insert suggestion as-is; no forced trailing space to avoid cursor reposition artifacts
        const insertText = vp.suggestion;
        const tr = view.state.update({ changes: { from: pos, to: pos, insert: insertText }, selection: { anchor: pos + insertText.length, head: pos + insertText.length } });
        view.dispatch(tr);
        vp.clear();
        return true;
      }
      return false; // fallthrough to default Tab behavior
    }
  }, {
    key: 'Escape',
    run(view) {
      const vp = view.plugin(plugin);
      if (vp && vp.suggestion) {
        vp.clear();
        return true;
      }
      return false;
    }
  }]));

  return [ghostTheme, ghostField, plugin, acceptKeymap];
}


