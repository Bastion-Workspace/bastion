import React, { useEffect, useLayoutEffect, useState, useMemo, useRef } from 'react';
import { Box, Typography, Alert } from '@mui/material';
import DOMPurify from 'dompurify';
import { useTheme } from '../../contexts/ThemeContext';
import {
  startBridgeListener,
  buildBridgeSdkScript,
  injectBridgeSdkIntoHtmlSrcDoc,
  unregisterArtifactStateWindow,
} from '../../utils/artifactBridge';

/**
 * React artifact runner evaluates code with new Function(), not as an ES module. Babel preset-react
 * only strips JSX; top-level export remains and causes SyntaxError: Unexpected token 'export'.
 */
export function normalizeReactArtifactUserCode(code) {
  let s = String(code ?? '').replace(/\r\n/g, '\n');
  if (!s.trim()) return s;
  s = s.replace(/^\s*export\s+default\s+async\s+function\s+/m, 'async function ');
  s = s.replace(/^\s*export\s+default\s+function\s+/m, 'function ');
  s = s.replace(/^\s*export\s+default\s+class\s+/m, 'class ');
  s = s.replace(/^(\s*)export\s+default\s+([A-Za-z_$][\w$]*)\s*;?\s*$/gm, '$1var App = $2');
  if (/^\s*export\s+default\s+/m.test(s)) {
    s = s.replace(/^\s*export\s+default\s+/m, 'const App = ');
  }
  s = s.replace(/^(\s*)export\s+(?!default\b)/gm, '$1');
  return s;
}

/**
 * Build srcDoc for react artifacts: React 18 + Babel Standalone from CDN inside a sandboxed iframe.
 * Resolves a root component from named App or from export default (via exports shim).
 * @param {string | null | undefined} artifactId - When set, enables shared state and notify APIs in the injected bridge SDK.
 */
export function buildReactSrcDoc(userCode, darkMode, artifactId) {
  const userCodeJson = JSON.stringify(normalizeReactArtifactUserCode(userCode ?? ''));
  const bodyClass = darkMode ? 'dark' : '';
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  ${buildBridgeSdkScript(artifactId)}
  <style>
    *,*::before,*::after{box-sizing:border-box}
    html,body{height:100%;margin:0}
    body{padding:8px;font-family:system-ui,-apple-system,sans-serif;font-size:14px;background:#fff;color:#1a1a1a;overflow:auto}
    body.dark{background:#121212;color:#e3e3e3}
    #root{min-height:100%;min-width:0}
    pre.err{white-space:pre-wrap;word-break:break-word;color:#c62828;background:#ffebee;padding:12px;border-radius:4px;font-size:12px;margin:0}
    body.dark pre.err{color:#ff8a80;background:#2d1b1b}
  </style>
</head>
<body class="${bodyClass}">
  <div id="root"></div>
  <script>
    (function () {
      var USER_CODE = ${userCodeJson};
      function showError(msg) {
        var el = document.getElementById('root');
        var t = String(msg == null ? '' : msg);
        el.innerHTML = '<pre class="err">' + t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;') + '</pre>';
      }
      window.onerror = function (message, source, lineno, colno, error) {
        showError(String(message) + (error && error.stack ? String.fromCharCode(10) + error.stack : ''));
        return true;
      };
      if (!USER_CODE || !String(USER_CODE).trim()) {
        showError('Empty React artifact');
        return;
      }
      try {
        if (typeof Babel === 'undefined' || typeof React === 'undefined' || typeof ReactDOM === 'undefined') {
          showError('Failed to load React or Babel from CDN');
          return;
        }
        var transformed = Babel.transform(USER_CODE, { presets: ['react'], filename: 'artifact.jsx' }).code;
        var tail = [
          '',
          'var _C = typeof App !== "undefined" ? App : null;',
          'if (!_C && typeof exports !== "undefined" && exports.default) _C = exports.default;',
          'return _C;',
        ].join(String.fromCharCode(10));
        var runner = new Function('React', 'ReactDOM', 'exports', transformed + tail);
        var exportsObj = {};
        var Component = runner(React, ReactDOM, exportsObj);
        if (!Component) {
          showError('Define App or export default (function App, const App, class App, or export default App)');
          return;
        }
        var mount = document.getElementById('root');
        mount.innerHTML = '';
        var root = ReactDOM.createRoot(mount);
        root.render(React.createElement(Component));
      } catch (e) {
        showError((e && e.message ? e.message : String(e)) + (e && e.stack ? String.fromCharCode(10) + e.stack : ''));
      }
    })();
  </script>
</body>
</html>`;
}

/** Matches orchestrator create_chart defaults (visualization_tools.py). */
const IFRAME_NOMINAL_WIDTH = 800;
const IFRAME_NOMINAL_HEIGHT = 600;
const IFRAME_MIN_SCALE = 0.18;
/** Cap upscale in viewport mode so CDN canvas/React artifacts do not become unusably huge on large monitors. */
const IFRAME_MAX_VIEWPORT_SCALE = 3;

/**
 * Sandboxed srcDoc iframe scaled to fit the host (no same-origin access required).
 *
 * @param {'panel' | 'viewport'} scaleMode - `panel`: fit width only, never upscale (sidebar / dashboard).
 *   `viewport`: fit both width and height of the container, allow upscale (fullscreen artifact dialog).
 */
export function ScaledSrcDocIframe({ title, srcDoc, sandbox = 'allow-scripts', scaleMode = 'panel', iframeRef }) {
  const containerRef = useRef(null);
  const [scale, setScale] = useState(1);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w <= 0) return;
      let s;
      if (scaleMode === 'viewport') {
        if (h <= 0) return;
        s = Math.min(w / IFRAME_NOMINAL_WIDTH, h / IFRAME_NOMINAL_HEIGHT);
        s = Math.max(IFRAME_MIN_SCALE, Math.min(IFRAME_MAX_VIEWPORT_SCALE, s));
      } else {
        s = Math.max(IFRAME_MIN_SCALE, Math.min(1, w / IFRAME_NOMINAL_WIDTH));
      }
      setScale(s);
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [scaleMode]);

  const clipW = IFRAME_NOMINAL_WIDTH * scale;
  const clipH = IFRAME_NOMINAL_HEIGHT * scale;
  const isViewport = scaleMode === 'viewport';

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        flex: 1,
        minHeight: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: isViewport ? 'center' : 'flex-start',
        overflow: isViewport ? 'hidden' : 'auto',
      }}
    >
      <Box
        sx={{
          width: clipW,
          height: clipH,
          overflow: 'hidden',
          flexShrink: 0,
        }}
      >
        <iframe
          ref={iframeRef}
          title={title}
          srcDoc={srcDoc}
          style={{
            border: 'none',
            width: IFRAME_NOMINAL_WIDTH,
            height: IFRAME_NOMINAL_HEIGHT,
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
            display: 'block',
          }}
          sandbox={sandbox}
        />
      </Box>
    </Box>
  );
}

/**
 * Renders chat artifact payloads: html/chart/react in a sandboxed iframe, mermaid as SVG, svg as sanitized markup.
 * `scaleMode` applies to chart artifacts only (ScaledSrcDocIframe). React artifacts use a responsive iframe.
 */
const ArtifactRenderer = ({
  artifact,
  height = '100%',
  scaleMode = 'panel',
  artifactId: artifactIdProp = null,
  onIframeMount = null,
}) => {
  const { darkMode } = useTheme();
  const [mermaidSvg, setMermaidSvg] = useState('');
  const [mermaidError, setMermaidError] = useState('');
  const type = (artifact?.artifact_type || '').toLowerCase();
  const code = artifact?.code ?? '';
  const htmlIframeRef = useRef(null);
  const reactIframeRef = useRef(null);
  const chartIframeRef = useRef(null);

  useEffect(() => {
    const cleanup = startBridgeListener();
    return cleanup;
  }, []);

  const sanitizedSvg = useMemo(() => {
    if (type !== 'svg' || !code) return '';
    try {
      return DOMPurify.sanitize(code, { USE_PROFILES: { svg: true } });
    } catch {
      return '';
    }
  }, [type, code]);

  useEffect(() => {
    if (type !== 'mermaid') {
      setMermaidSvg('');
      setMermaidError('');
      return undefined;
    }
    let cancelled = false;
    setMermaidError('');
    setMermaidSvg('');
    if (!code.trim()) {
      setMermaidError('Empty diagram source');
      return undefined;
    }
    const id = `mermaid-artifact-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    import('mermaid')
      .then((mod) => {
        if (cancelled) return;
        const mermaid = mod.default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: darkMode ? 'dark' : 'neutral',
        });
        return mermaid.render(id, code);
      })
      .then((result) => {
        if (cancelled || !result) return;
        setMermaidSvg(result.svg || '');
      })
      .catch((err) => {
        if (!cancelled) {
          setMermaidError(err?.message || 'Failed to render diagram');
        }
      });
    return () => {
      cancelled = true;
    };
  }, [type, code, darkMode]);

  const reactSrcDoc = useMemo(() => {
    if (type !== 'react') return '';
    return buildReactSrcDoc(code, darkMode, artifactIdProp);
  }, [type, code, darkMode, artifactIdProp]);

  const htmlOrChartSrcDocWithBridge = useMemo(
    () => injectBridgeSdkIntoHtmlSrcDoc(code, artifactIdProp),
    [code, artifactIdProp]
  );

  useLayoutEffect(() => {
    if (!artifactIdProp || !onIframeMount) return;
    const el =
      type === 'html'
        ? htmlIframeRef.current
        : type === 'react'
          ? reactIframeRef.current
          : type === 'chart'
            ? chartIframeRef.current
            : null;
    if (el) {
      onIframeMount(el);
    }
  }, [artifactIdProp, onIframeMount, type, code, darkMode, reactSrcDoc, htmlOrChartSrcDocWithBridge]);

  useEffect(() => {
    if (!artifactIdProp) return undefined;
    return () => {
      const el =
        type === 'html'
          ? htmlIframeRef.current
          : type === 'react'
            ? reactIframeRef.current
            : type === 'chart'
              ? chartIframeRef.current
              : null;
      const win = el?.contentWindow;
      if (win) {
        unregisterArtifactStateWindow(String(artifactIdProp), win);
      }
    };
  }, [artifactIdProp, type]);

  if (!artifact || !type) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="body2" color="text.secondary">
          No artifact to display
        </Typography>
      </Box>
    );
  }

  if (type === 'html') {
    return (
      <Box sx={{ height, width: '100%', minHeight: 240, display: 'flex', flexDirection: 'column' }}>
        <iframe
          ref={htmlIframeRef}
          title={artifact.title || 'Artifact'}
          srcDoc={htmlOrChartSrcDocWithBridge}
          style={{ border: 'none', width: '100%', height: '100%', flex: 1, minHeight: 200 }}
          sandbox="allow-scripts"
        />
      </Box>
    );
  }

  if (type === 'chart') {
    return (
      <Box sx={{ height, width: '100%', minHeight: 240, display: 'flex', flexDirection: 'column' }}>
        <ScaledSrcDocIframe
          title={artifact.title || 'Chart'}
          srcDoc={htmlOrChartSrcDocWithBridge}
          scaleMode={scaleMode}
          iframeRef={chartIframeRef}
        />
      </Box>
    );
  }

  if (type === 'react') {
    // Responsive iframe: content sees real viewport size (unlike ScaledSrcDocIframe transform, which
    // keeps a fixed 800×600 layout and causes inner scrollbars for wide/tall UIs like expert Minesweeper).
    return (
      <Box sx={{ height, width: '100%', minHeight: 240, display: 'flex', flexDirection: 'column' }}>
        <iframe
          ref={reactIframeRef}
          title={artifact.title || 'React artifact'}
          srcDoc={reactSrcDoc}
          style={{ border: 'none', width: '100%', height: '100%', flex: 1, minHeight: 200 }}
          sandbox="allow-scripts"
        />
      </Box>
    );
  }

  if (type === 'mermaid') {
    return (
      <Box sx={{ height, width: '100%', minHeight: 200, overflow: 'auto', p: 1 }}>
        {mermaidError && (
          <Alert severity="warning" sx={{ mb: 1 }}>
            {mermaidError}
          </Alert>
        )}
        {mermaidSvg ? (
          <Box
            sx={{ '& svg': { maxWidth: '100%', height: 'auto', display: 'block' } }}
            dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(mermaidSvg, { USE_PROFILES: { svg: true } }) }}
          />
        ) : !mermaidError ? (
          <Typography variant="body2" color="text.secondary">
            Rendering diagram…
          </Typography>
        ) : null}
      </Box>
    );
  }

  if (type === 'svg') {
    return (
      <Box sx={{ height, width: '100%', minHeight: 120, overflow: 'auto', p: 1 }}>
        {sanitizedSvg ? (
          <Box sx={{ '& svg': { maxWidth: '100%', height: 'auto', display: 'block' } }} dangerouslySetInnerHTML={{ __html: sanitizedSvg }} />
        ) : (
          <Typography variant="body2" color="text.secondary">
            Invalid or empty SVG
          </Typography>
        )}
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="body2" color="text.secondary">
        Unsupported artifact type: {type}
      </Typography>
    </Box>
  );
};

export default ArtifactRenderer;
