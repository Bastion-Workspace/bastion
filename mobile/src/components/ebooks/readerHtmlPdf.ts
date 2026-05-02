/**
 * WebView PDF reader: loads pdf.js from CDN and renders pages from embedded base64 (no file:// fetch).
 * Exposes window.receiveCmd for PREV/NEXT (same contract as the EPUB WebView reader).
 */

export function buildPdfReaderHtml(pdfBase64: string): string {
  const b64Json = JSON.stringify(pdfBase64);
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/>
  <style>
    body { margin: 0; padding: 0; background: #1e1e1e; color: #eee; font-family: system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; }
    #toolbar { padding: 10px; display: flex; gap: 10px; align-items: center; background: #2d2d2d; flex-shrink: 0; }
    #wrap { flex: 1; overflow: auto; display: flex; justify-content: center; align-items: flex-start; padding: 8px; }
    canvas { box-shadow: 0 2px 12px rgba(0,0,0,0.5); max-width: 100%; height: auto; }
    button { padding: 10px 14px; background: #444; color: #fff; border: 1px solid #666; border-radius: 8px; font-size: 15px; }
    #lab { flex: 1; text-align: center; font-size: 14px; }
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
</head>
<body>
<div id="toolbar">
  <button type="button" id="prev">Prev</button>
  <span id="lab">—</span>
  <button type="button" id="next">Next</button>
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<script>
(function () {
  var b64 = ${b64Json};
  var pdfDoc = null;
  var pageNum = 1;
  var scale = 1.15;
  function send(o) {
    try {
      if (window.ReactNativeWebView) {
        window.ReactNativeWebView.postMessage(JSON.stringify(o));
      }
    } catch (e) {}
  }
  function binToUint8(b64s) {
    var bin = atob(b64s);
    var len = bin.length;
    var arr = new Uint8Array(len);
    for (var i = 0; i < len; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }
  function renderPage() {
    if (!pdfDoc) return;
    pdfDoc
      .getPage(pageNum)
      .then(function (page) {
        var vp = page.getViewport({ scale: scale });
        var canvas = document.getElementById('cv');
        var ctx = canvas.getContext('2d');
        canvas.width = vp.width;
        canvas.height = vp.height;
        return page.render({ canvasContext: ctx, viewport: vp }).promise;
      })
      .then(function () {
        document.getElementById('lab').textContent = pageNum + ' / ' + pdfDoc.numPages;
        var pct = pdfDoc.numPages ? (pageNum - 0.5) / pdfDoc.numPages : 0;
        send({ type: 'RELOCATED', cfi: '', percentage: pct });
      })
      .catch(function (e) {
        send({ type: 'ERROR', message: String(e) });
      });
  }
  window.receiveCmd = function (cmd) {
    if (!cmd || !cmd.type) return;
    try {
      if (cmd.type === 'SET_THEME' || cmd.type === 'SET_FONT_SIZE' || cmd.type === 'GOTO_CFI') {
        return;
      }
      if (cmd.type === 'SEEK_PERCENT' && pdfDoc && typeof cmd.pct === 'number') {
        var target = Math.max(1, Math.min(pdfDoc.numPages, Math.round(cmd.pct * pdfDoc.numPages) || 1));
        pageNum = target;
        renderPage();
        return;
      }
      if (!pdfDoc) return;
      if (cmd.type === 'PREV' && pageNum > 1) {
        pageNum--;
        renderPage();
      } else if (cmd.type === 'NEXT' && pageNum < pdfDoc.numPages) {
        pageNum++;
        renderPage();
      }
    } catch (e2) {}
  };
  if (typeof pdfjsLib === 'undefined') {
    send({ type: 'ERROR', message: 'pdf.js failed to load (network?)' });
    return;
  }
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
  pdfjsLib
    .getDocument({ data: binToUint8(b64) })
    .promise.then(function (pdf) {
      pdfDoc = pdf;
      renderPage();
      send({ type: 'READY' });
    })
    .catch(function (e) {
      send({ type: 'ERROR', message: String(e) });
    });
  document.getElementById('prev').onclick = function () {
    if (pageNum <= 1) return;
    pageNum--;
    renderPage();
  };
  document.getElementById('next').onclick = function () {
    if (!pdfDoc || pageNum >= pdfDoc.numPages) return;
    pageNum++;
    renderPage();
  };
})();
</script>
</body>
</html>`;
}
