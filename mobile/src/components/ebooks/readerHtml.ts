import { EPUBJS_INLINE, JSZIP_INLINE } from './readerHtmlVendor.generated';

/**
 * WebView EPUB reader: bundled JSZip + epubjs (no CDN), EPUB bytes embedded as base64.
 * React Native passes the full HTML via WebView `source={{ html }}` so Android file:// XHR is avoided.
 */
function escapeScriptFragmentForHtml(scriptSource: string): string {
  return scriptSource.replace(/<\/script>/gi, '<\\/script>');
}

export function buildReaderHtml(epubBase64: string): string {
  const jszipTag = `<script>${escapeScriptFragmentForHtml(JSZIP_INLINE)}</script>`;
  const epubTag = `<script>${escapeScriptFragmentForHtml(EPUBJS_INLINE)}</script>`;
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/>
  <style>
    html, body { margin: 0; padding: 0; height: 100vh; width: 100vw; overflow: hidden; }
    #area { position: absolute; top: 0; left: 0; right: 0; bottom: 0; touch-action: manipulation; }
  </style>
  ${jszipTag}
  ${epubTag}
</head>
<body>
<div id="area"></div>
<script>
window.__epubB64 = ${JSON.stringify(epubBase64)};
(function () {
  var book = null;
  var rendition = null;
  var themeKey = 'light';
  var fontSize = 18;
  var fontFamily = 'Georgia, serif';

  function send(o) {
    try {
      if (window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify(o));
      }
    } catch (e) {}
  }

  function attachSwipePageTurn() {
    if (!rendition) return;
    rendition.hooks.content.register(function (contents) {
      var doc = contents.document;
      if (!doc) return;
      var startX = 0;
      var startY = 0;
      var tracking = false;
      function onTouchStart(ev) {
        if (!ev.touches || ev.touches.length !== 1) {
          tracking = false;
          return;
        }
        var t = ev.touches[0];
        startX = t.clientX;
        startY = t.clientY;
        tracking = true;
      }
      function onTouchEnd(ev) {
        if (!tracking) return;
        tracking = false;
        if (!ev.changedTouches || ev.changedTouches.length !== 1) return;
        var t = ev.changedTouches[0];
        var dx = t.clientX - startX;
        var dy = t.clientY - startY;
        if (dy > 40 && dy > Math.abs(dx) * 1.8) {
          send({ type: 'OPEN_CHROME' });
          return;
        }
        if (Math.abs(dx) < 56) return;
        if (Math.abs(dy) > Math.abs(dx) * 0.55) return;
        try {
          if (dx < 0) rendition.next();
          else rendition.prev();
        } catch (e2) {}
      }
      doc.addEventListener('touchstart', onTouchStart, { passive: true, capture: true });
      doc.addEventListener('touchend', onTouchEnd, { passive: true, capture: true });
    });
  }

  function applyTheme() {
    if (!rendition) return;
    var themes = {
      light: { body: { background: '#fff', color: '#111' } },
      sepia: { body: { background: '#f4ecd8', color: '#3e3020' } },
      dark: { body: { background: '#1e1e1e', color: '#e6e6e6' } }
    };
    var t = themes[themeKey] || themes.light;
    rendition.themes.default(
      Object.assign({}, t, {
        p: {
          'font-family': fontFamily,
          'font-size': fontSize + 'px !important',
          'line-height': '1.45 !important'
        }
      })
    );
  }

  window.receiveCmd = function (cmd) {
    if (!cmd || !cmd.type) return;
    try {
      if (cmd.type === 'PREV' && rendition) rendition.prev();
      else if (cmd.type === 'NEXT' && rendition) rendition.next();
      else if (cmd.type === 'GOTO_CFI' && rendition && cmd.cfi) rendition.display(String(cmd.cfi));
      else if (cmd.type === 'SET_THEME') {
        if (cmd.theme) themeKey = String(cmd.theme);
        if (typeof cmd.fontSize === 'number') fontSize = cmd.fontSize;
        if (typeof cmd.fontFamily === 'string' && cmd.fontFamily.trim()) fontFamily = cmd.fontFamily.trim();
        applyTheme();
      } else if (cmd.type === 'SET_FONT_SIZE' && typeof cmd.fontSize === 'number') {
        fontSize = cmd.fontSize;
        if (typeof cmd.fontFamily === 'string' && cmd.fontFamily.trim()) fontFamily = cmd.fontFamily.trim();
        applyTheme();
      } else if (cmd.type === 'SEEK_PERCENT' && rendition && book && book.locations) {
        var pctSeek = typeof cmd.pct === 'number' ? cmd.pct : 0;
        var cfiSeek = book.locations.cfiFromPercentage(pctSeek);
        if (cfiSeek) rendition.display(cfiSeek);
      }
    } catch (e) {
      send({ type: 'ERROR', message: String(e) });
    }
  };

  function boot() {
    if (typeof JSZip !== 'function') {
      send({ type: 'ERROR', message: 'JSZip not available (check network / script load)' });
      return;
    }
    if (typeof ePub !== 'function') {
      send({ type: 'ERROR', message: 'epubjs not available' });
      return;
    }
    var b64 = window.__epubB64;
    if (!b64 || typeof b64 !== 'string' || !b64.length) {
      send({ type: 'ERROR', message: 'Missing EPUB data' });
      return;
    }
    try {
      var bin = atob(b64);
      var arr = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      book = ePub(arr.buffer);
    } catch (e) {
      send({ type: 'ERROR', message: String(e) });
      return;
    }
    book.ready
      .then(function () {
        var el = document.getElementById('area');
        rendition = book.renderTo(el, {
          width: '100%',
          height: '100%',
          flow: 'paginated',
          manager: 'default',
          spread: 'none',
          minSpreadWidth: 800
        });
        applyTheme();
        attachSwipePageTurn();
        rendition.on('relocated', function (loc) {
          try {
            var cfi = loc && loc.start && loc.start.cfi;
            var pct = 0;
            if (book.locations && typeof book.locations.length === 'function' && book.locations.length() > 0) {
              var p = book.locations.percentageFromCfi(cfi);
              pct = typeof p === 'number' ? p : 0;
            }
            send({ type: 'RELOCATED', cfi: cfi || '', percentage: pct });
          } catch (err) {
            send({ type: 'RELOCATED', cfi: '', percentage: 0 });
          }
        });
        return book.locations.generate(1024).catch(function () { return null; });
      })
      .then(function () {
        return rendition.display();
      })
      .then(function () {
        send({ type: 'READY' });
      })
      .catch(function (err) {
        send({ type: 'ERROR', message: String(err) });
      });
  }

  boot();
})();
</script>
</body>
</html>`;
}

/**
 * Same reader as {@link buildReaderHtml}, but loads the EPUB via `fetch(relativeUrl)` from disk.
 * HTML must be served from the same directory as the `.epub` file (e.g. `./digest.epub`).
 */
export function buildReaderHtmlFromUri(epubRelativeUrl: string): string {
  const jszipTag = `<script>${escapeScriptFragmentForHtml(JSZIP_INLINE)}</script>`;
  const epubTag = `<script>${escapeScriptFragmentForHtml(EPUBJS_INLINE)}</script>`;
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/>
  <style>
    html, body { margin: 0; padding: 0; height: 100vh; width: 100vw; overflow: hidden; }
    #area { position: absolute; top: 0; left: 0; right: 0; bottom: 0; touch-action: manipulation; }
  </style>
  ${jszipTag}
  ${epubTag}
</head>
<body>
<div id="area"></div>
<script>
var __epubRelUrl = ${JSON.stringify(epubRelativeUrl)};
(function () {
  var book = null;
  var rendition = null;
  var themeKey = 'light';
  var fontSize = 18;
  var fontFamily = 'Georgia, serif';

  function send(o) {
    try {
      if (window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify(o));
      }
    } catch (e) {}
  }

  function attachSwipePageTurn() {
    if (!rendition) return;
    rendition.hooks.content.register(function (contents) {
      var doc = contents.document;
      if (!doc) return;
      var startX = 0;
      var startY = 0;
      var tracking = false;
      function onTouchStart(ev) {
        if (!ev.touches || ev.touches.length !== 1) {
          tracking = false;
          return;
        }
        var t = ev.touches[0];
        startX = t.clientX;
        startY = t.clientY;
        tracking = true;
      }
      function onTouchEnd(ev) {
        if (!tracking) return;
        tracking = false;
        if (!ev.changedTouches || ev.changedTouches.length !== 1) return;
        var t = ev.changedTouches[0];
        var dx = t.clientX - startX;
        var dy = t.clientY - startY;
        if (dy > 40 && dy > Math.abs(dx) * 1.8) {
          send({ type: 'OPEN_CHROME' });
          return;
        }
        if (Math.abs(dx) < 56) return;
        if (Math.abs(dy) > Math.abs(dx) * 0.55) return;
        try {
          if (dx < 0) rendition.next();
          else rendition.prev();
        } catch (e2) {}
      }
      doc.addEventListener('touchstart', onTouchStart, { passive: true, capture: true });
      doc.addEventListener('touchend', onTouchEnd, { passive: true, capture: true });
    });
  }

  function applyTheme() {
    if (!rendition) return;
    var themes = {
      light: { body: { background: '#fff', color: '#111' } },
      sepia: { body: { background: '#f4ecd8', color: '#3e3020' } },
      dark: { body: { background: '#1e1e1e', color: '#e6e6e6' } }
    };
    var t = themes[themeKey] || themes.light;
    rendition.themes.default(
      Object.assign({}, t, {
        p: {
          'font-family': fontFamily,
          'font-size': fontSize + 'px !important',
          'line-height': '1.45 !important'
        }
      })
    );
  }

  window.receiveCmd = function (cmd) {
    if (!cmd || !cmd.type) return;
    try {
      if (cmd.type === 'PREV' && rendition) rendition.prev();
      else if (cmd.type === 'NEXT' && rendition) rendition.next();
      else if (cmd.type === 'GOTO_CFI' && rendition && cmd.cfi) rendition.display(String(cmd.cfi));
      else if (cmd.type === 'SET_THEME') {
        if (cmd.theme) themeKey = String(cmd.theme);
        if (typeof cmd.fontSize === 'number') fontSize = cmd.fontSize;
        if (typeof cmd.fontFamily === 'string' && cmd.fontFamily.trim()) fontFamily = cmd.fontFamily.trim();
        applyTheme();
      } else if (cmd.type === 'SET_FONT_SIZE' && typeof cmd.fontSize === 'number') {
        fontSize = cmd.fontSize;
        if (typeof cmd.fontFamily === 'string' && cmd.fontFamily.trim()) fontFamily = cmd.fontFamily.trim();
        applyTheme();
      } else if (cmd.type === 'SEEK_PERCENT' && rendition && book && book.locations) {
        var pctSeek = typeof cmd.pct === 'number' ? cmd.pct : 0;
        var cfiSeek = book.locations.cfiFromPercentage(pctSeek);
        if (cfiSeek) rendition.display(cfiSeek);
      }
    } catch (e) {
      send({ type: 'ERROR', message: String(e) });
    }
  };

  function boot() {
    if (typeof JSZip !== 'function') {
      send({ type: 'ERROR', message: 'JSZip not available (check network / script load)' });
      return;
    }
    if (typeof ePub !== 'function') {
      send({ type: 'ERROR', message: 'epubjs not available' });
      return;
    }
    var rel = __epubRelUrl;
    if (!rel || typeof rel !== 'string' || !rel.length) {
      send({ type: 'ERROR', message: 'Missing EPUB path' });
      return;
    }
    fetch(rel)
      .then(function (r) {
        if (!r.ok) throw new Error('fetch EPUB failed ' + r.status);
        return r.arrayBuffer();
      })
      .then(function (buf) {
        book = ePub(buf);
        return book.ready;
      })
      .then(function () {
        var el = document.getElementById('area');
        rendition = book.renderTo(el, {
          width: '100%',
          height: '100%',
          flow: 'paginated',
          manager: 'default',
          spread: 'none',
          minSpreadWidth: 800
        });
        applyTheme();
        attachSwipePageTurn();
        rendition.on('relocated', function (loc) {
          try {
            var cfi = loc && loc.start && loc.start.cfi;
            var pct = 0;
            if (book.locations && typeof book.locations.length === 'function' && book.locations.length() > 0) {
              var p = book.locations.percentageFromCfi(cfi);
              pct = typeof p === 'number' ? p : 0;
            }
            send({ type: 'RELOCATED', cfi: cfi || '', percentage: pct });
          } catch (err) {
            send({ type: 'RELOCATED', cfi: '', percentage: 0 });
          }
        });
        return book.locations.generate(1024).catch(function () { return null; });
      })
      .then(function () {
        return rendition.display();
      })
      .then(function () {
        send({ type: 'READY' });
      })
      .catch(function (err) {
        send({ type: 'ERROR', message: String(err) });
      });
  }

  boot();
})();
</script>
</body>
</html>`;
}
