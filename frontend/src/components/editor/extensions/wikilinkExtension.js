/**
 * CodeMirror 6 extensions for markdown [[wikilinks]]: highlight, click, autocomplete.
 */
import { StateField, RangeSetBuilder } from '@codemirror/state';
import { Decoration, EditorView, ViewPlugin } from '@codemirror/view';
import { autocompletion } from '@codemirror/autocomplete';

const WIKILINK_RE = /\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g;

export function isPlainWikilinkTitle(t) {
  const s = (t || '').trim();
  if (!s) return false;
  const low = s.toLowerCase();
  if (
    low.startsWith('file:')
    || low.startsWith('id:')
    || low.startsWith('http://')
    || low.startsWith('https://')
    || low.startsWith('#')
  ) {
    return false;
  }
  if (/^\.?\.\.?\/[^\s]*\.(org|md|txt)$/i.test(s) || /^[^\s/\\]+\.(org|md|txt)$/i.test(s)) {
    return false;
  }
  if (s.includes('/') || s.includes('\\')) return false;
  return true;
}

function wikilinkDecorations(doc) {
  const builder = new RangeSetBuilder();
  for (let i = 1; i <= doc.lines; i += 1) {
    const line = doc.line(i);
    const { text } = line;
    WIKILINK_RE.lastIndex = 0;
    let m = WIKILINK_RE.exec(text);
    while (m !== null) {
      const title = (m[1] || '').trim();
      if (isPlainWikilinkTitle(title)) {
        const from = line.from + m.index;
        const to = from + m[0].length;
        builder.add(
          from,
          to,
          Decoration.mark({
            attributes: {
              class: 'cm-wikilink',
              'data-wikilink': encodeURIComponent(title),
            },
          }),
        );
      }
      m = WIKILINK_RE.exec(text);
    }
  }
  return builder.finish();
}

const wikilinkDecorationField = StateField.define({
  create(state) {
    return wikilinkDecorations(state.doc);
  },
  update(deco, tr) {
    if (tr.docChanged) return wikilinkDecorations(tr.state.doc);
    return deco.map(tr.changes);
  },
  provide: (f) => EditorView.decorations.from(f),
});

/**
 * @param {object} opts
 * @param {boolean} [opts.autocompleteEnabled]
 * @param {(title: string) => Promise<string|null>} opts.resolveTitleToDocumentId
 * @param {(documentId: string, title: string) => void} [opts.onNavigate]
 * @param {(title: string, event: MouseEvent) => void} [opts.onUnresolvedClick]
 * @param {(title: string, anchorEl: HTMLElement | null) => void} [opts.onHoverTitle]
 * @param {() => void} [opts.onHoverLeave]
 */
export function createWikilinkExtensions(opts) {
  const {
    autocompleteEnabled = true,
    resolveTitleToDocumentId,
    searchDocumentsForCompletion,
    onNavigate,
    onUnresolvedClick,
    onHoverTitle,
    onHoverLeave,
  } = opts || {};

  const wikiClick = EditorView.domEventHandlers({
    mousedown(event, view) {
      const t = event.target;
      if (!t || typeof t.closest !== 'function') return false;
      const el = t.closest('.cm-wikilink');
      if (!el || !el.getAttribute) return false;
      const enc = el.getAttribute('data-wikilink');
      if (!enc) return false;
      let title;
      try {
        title = decodeURIComponent(enc);
      } catch {
        title = enc;
      }
      event.preventDefault();
      (async () => {
        const id = await resolveTitleToDocumentId(title);
        if (id && onNavigate) onNavigate(id, title);
        else if (onUnresolvedClick) onUnresolvedClick(title, event);
      })();
      return true;
    },
  });

  let hoverLeaveTimer = null;
  const wikiHover = ViewPlugin.fromClass(
    class {
      constructor(view) {
        this.view = view;
        this._onMove = this.onMove.bind(this);
        this._onLeave = this.onLeave.bind(this);
        view.dom.addEventListener('mousemove', this._onMove);
        view.dom.addEventListener('mouseleave', this._onLeave);
      }

      onMove(event) {
        if (!onHoverTitle) return;
        const t = event.target;
        if (!t || typeof t.closest !== 'function') return;
        const el = t.closest('.cm-wikilink');
        if (!el || !el.getAttribute) {
          this.scheduleLeave();
          return;
        }
        const enc = el.getAttribute('data-wikilink');
        if (!enc) return;
        let title;
        try {
          title = decodeURIComponent(enc);
        } catch {
          title = enc;
        }
        if (hoverLeaveTimer) {
          clearTimeout(hoverLeaveTimer);
          hoverLeaveTimer = null;
        }
        onHoverTitle(title, el);
      }

      onLeave() {
        this.scheduleLeave();
      }

      scheduleLeave() {
        if (!onHoverLeave) return;
        if (hoverLeaveTimer) clearTimeout(hoverLeaveTimer);
        hoverLeaveTimer = setTimeout(() => {
          hoverLeaveTimer = null;
          onHoverLeave();
        }, 280);
      }

      destroy() {
        this.view.dom.removeEventListener('mousemove', this._onMove);
        this.view.dom.removeEventListener('mouseleave', this._onLeave);
        if (hoverLeaveTimer) clearTimeout(hoverLeaveTimer);
      }
    },
  );

  const completionExt = autocompleteEnabled && searchDocumentsForCompletion
    ? [
        autocompletion({
          override: [
            async (context) => {
              const before = context.matchBefore(/\[\[[^\]]*$/);
              if (!before && !context.explicit) return null;
              if (!before) return null;
              if (before.text.length < 2) return null;
              const options = await searchDocumentsForCompletion(
                before.text.slice(2).trim(),
              );
              if (!options || !options.length) return null;
              return {
                from: before.from,
                filter: false,
                options: options.map((o) => ({
                  label: o.label,
                  detail: o.detail,
                  apply(view, _completion, from, to) {
                    const ins = `[[${o.label}]]`;
                    view.dispatch({
                      changes: { from, to, insert: ins },
                      selection: { anchor: from + ins.length },
                    });
                  },
                })),
              };
            },
          ],
        }),
      ]
    : [];

  return [wikilinkDecorationField, wikiClick, wikiHover, ...completionExt];
}
