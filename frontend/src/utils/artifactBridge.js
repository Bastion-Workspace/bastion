/**
 * Parent-window bridge for sandboxed artifact iframes (postMessage + allowlisted GET proxy).
 * Tokens stay in the parent; iframe code calls window.bastion.query() from injected SDK.
 * Optional artifact_id enables shared in-memory state (bastion.setState / onStateChange) and notify (badges).
 */

const API_BASE = import.meta.env.VITE_API_URL || '';

/**
 * Tier 1: read-only JSON GET paths requiring auth. Tier 2 stays out until explicitly enabled.
 * When changing this list, update artifact_react in backend/services/builtin_skill_definitions.py
 * and backend/help_docs/chat-and-agents/06-artifacts.md.
 * normalizePath() accepts absolute http(s) URLs (uses pathname) and strips trailing slashes.
 */
const ALLOWED_ENDPOINTS = [
  /^\/api\/todos$/,
  /^\/api\/org\/todos$/,
  /^\/api\/org\/agenda$/,
  /^\/api\/org\/tags$/,
  /^\/api\/org\/search$/,
  /^\/api\/org\/clock\/active$/,
  /^\/api\/org\/journal\/entry$/,
  /^\/api\/calendar\/events$/,
  /^\/api\/calendar\/calendars$/,
  /^\/api\/rss\/feeds$/,
  /^\/api\/rss\/unread-count$/,
  /^\/api\/rss\/feeds\/[^/]+\/articles$/,
  /^\/api\/folders\/tree$/,
  /^\/api\/document-pins$/,
  /^\/api\/folders\/[^/]+\/contents$/,
  /^\/api\/status-bar\/data$/,
];

const BRIDGE_WINDOW_MS = 10000;
const BRIDGE_MAX_PER_WINDOW = 10;
const BRIDGE_MAX_CONCURRENT = 3;

/** Max serialized JSON size for a single state value (bytes, UTF-8 approximation). */
const MAX_STATE_VALUE_BYTES = 65536;

const requestTimestamps = [];
let concurrentBridgeRequests = 0;

let listenerRefCount = 0;
let boundHandler = null;

/** @type {Map<string, Record<string, unknown>>} */
const artifactStateById = new Map();
/** @type {Map<string, Set<Window>>} */
const artifactWindowsById = new Map();

/** @type {((detail: { artifactId: string, payload: Record<string, unknown> }) => void) | null} */
let notifyHandler = null;

function pruneTimestamps() {
  const now = Date.now();
  while (requestTimestamps.length > 0 && requestTimestamps[0] < now - BRIDGE_WINDOW_MS) {
    requestTimestamps.shift();
  }
}

function isUnderRateLimit() {
  pruneTimestamps();
  return requestTimestamps.length < BRIDGE_MAX_PER_WINDOW;
}

function recordBridgeRequest() {
  requestTimestamps.push(Date.now());
}

export function isEndpointAllowed(pathOnly) {
  return ALLOWED_ENDPOINTS.some((re) => re.test(pathOnly));
}

/**
 * Optional: parent registers handler for bastion.notify from iframes (badges, etc.).
 * @param {(detail: { artifactId: string, payload: Record<string, unknown> }) => void | null} fn
 */
export function setArtifactNotifyHandler(fn) {
  notifyHandler = typeof fn === 'function' ? fn : null;
}

/**
 * Register a window for an artifact (for state broadcast). Usually done via bastion_state_init from iframe.
 * @param {string} artifactId
 * @param {Window} win
 */
export function registerArtifactStateWindow(artifactId, win) {
  if (!artifactId || !win) return;
  let set = artifactWindowsById.get(artifactId);
  if (!set) {
    set = new Set();
    artifactWindowsById.set(artifactId, set);
  }
  set.add(win);
}

/**
 * Unregister when iframe unmounts (parent calls from React cleanup).
 * @param {string} artifactId
 * @param {Window} win
 */
export function unregisterArtifactStateWindow(artifactId, win) {
  if (!artifactId || !win) return;
  const set = artifactWindowsById.get(artifactId);
  if (!set) return;
  set.delete(win);
  if (set.size === 0) {
    artifactWindowsById.delete(artifactId);
  }
}

function getOrCreateArtifactState(artifactId) {
  let s = artifactStateById.get(artifactId);
  if (!s) {
    s = {};
    artifactStateById.set(artifactId, s);
  }
  return s;
}

function broadcastStateChange(artifactId, key, value, exceptSource) {
  const state = getOrCreateArtifactState(artifactId);
  const targets = artifactWindowsById.get(artifactId);
  if (!targets) return;
  targets.forEach((w) => {
    if (w === exceptSource) return;
    try {
      if (w.closed) return;
      w.postMessage(
        {
          type: 'bastion_state_changed',
          key,
          value,
          state: { ...state },
        },
        '*'
      );
    } catch {
      // ignore
    }
  });
}

function snapshotToWindow(artifactId, win) {
  if (!win || win.closed) return;
  const state = getOrCreateArtifactState(artifactId);
  try {
    win.postMessage({ type: 'bastion_state_snapshot', state: { ...state } }, '*');
  } catch {
    // ignore
  }
}

function valueSizeOk(value) {
  try {
    const s = JSON.stringify(value);
    return s.length <= MAX_STATE_VALUE_BYTES;
  } catch {
    return false;
  }
}

/**
 * Injects bastion.query() into artifact iframes (ES5 for broad compatibility).
 * @param {string | null | undefined} artifactId - When set, enables shared state + notify APIs.
 * @returns {string} Full <script>...</script> tag to embed in srcDoc.
 */
export function buildBridgeSdkScript(artifactId) {
  const aid = artifactId != null && String(artifactId).trim() ? JSON.stringify(String(artifactId).trim()) : 'null';
  if (aid === 'null') {
    return `<script>
(function () {
  window.bastion = {
    _pending: {},
    query: function (endpoint, params) {
      var self = this;
      return new Promise(function (resolve, reject) {
        var id = Math.random().toString(36).slice(2) + Date.now().toString(36);
        var timer = setTimeout(function () {
          var p = self._pending[id];
          if (p) {
            clearTimeout(p.timer);
            delete self._pending[id];
          }
          reject(new Error('Bridge timeout'));
        }, 15000);
        self._pending[id] = { resolve: resolve, reject: reject, timer: timer };
        window.parent.postMessage(
          { type: 'bastion_query', id: id, endpoint: endpoint, params: params || {} },
          '*'
        );
      });
    },
    _handleResponse: function (event) {
      if (!event.data || event.data.type !== 'bastion_response') return;
      var p = window.bastion._pending[event.data.id];
      if (!p) return;
      clearTimeout(p.timer);
      delete window.bastion._pending[event.data.id];
      if (event.data.error) p.reject(new Error(event.data.error));
      else p.resolve(event.data.result);
    },
  };
  window.addEventListener('message', window.bastion._handleResponse);
})();
</script>`;
  }

  return `<script>
(function () {
  var ARTIFACT_ID = ${aid};
  window.bastion = {
    artifact_id: ARTIFACT_ID,
    _pending: {},
    _stateMirror: {},
    _stateListeners: [],
    query: function (endpoint, params) {
      var self = this;
      return new Promise(function (resolve, reject) {
        var id = Math.random().toString(36).slice(2) + Date.now().toString(36);
        var timer = setTimeout(function () {
          var p = self._pending[id];
          if (p) {
            clearTimeout(p.timer);
            delete self._pending[id];
          }
          reject(new Error('Bridge timeout'));
        }, 15000);
        self._pending[id] = { resolve: resolve, reject: reject, timer: timer };
        window.parent.postMessage(
          { type: 'bastion_query', id: id, endpoint: endpoint, params: params || {} },
          '*'
        );
      });
    },
    _handleResponse: function (event) {
      if (!event.data || event.data.type !== 'bastion_response') return;
      var p = window.bastion._pending[event.data.id];
      if (!p) return;
      clearTimeout(p.timer);
      delete window.bastion._pending[event.data.id];
      if (event.data.error) p.reject(new Error(event.data.error));
      else p.resolve(event.data.result);
    },
    getState: function (key) {
      return window.bastion._stateMirror[key];
    },
    setState: function (key, value) {
      if (key === undefined || key === null) return;
      window.parent.postMessage(
        { type: 'bastion_state_set', artifact_id: ARTIFACT_ID, key: String(key), value: value },
        '*'
      );
    },
    onStateChange: function (cb) {
      if (typeof cb !== 'function') return;
      window.bastion._stateListeners.push(cb);
    },
    notify: function (payload) {
      window.parent.postMessage(
        { type: 'bastion_notify', artifact_id: ARTIFACT_ID, payload: payload && typeof payload === 'object' ? payload : {} },
        '*'
      );
    },
  };
  window.addEventListener('message', window.bastion._handleResponse);
  window.addEventListener('message', function (event) {
    var d = event.data;
    if (!d || typeof d !== 'object') return;
    if (d.type === 'bastion_state_snapshot') {
      window.bastion._stateMirror = d.state && typeof d.state === 'object' ? d.state : {};
      var listeners = window.bastion._stateListeners || [];
      for (var i = 0; i < listeners.length; i++) {
        try {
          listeners[i]({ key: null, value: null, state: window.bastion._stateMirror });
        } catch (e) {}
      }
      return;
    }
    if (d.type === 'bastion_state_changed') {
      if (d.key !== undefined && d.key !== null) {
        window.bastion._stateMirror[d.key] = d.value;
      }
      if (d.state && typeof d.state === 'object') {
        window.bastion._stateMirror = d.state;
      }
      var listeners2 = window.bastion._stateListeners || [];
      for (var j = 0; j < listeners2.length; j++) {
        try {
          listeners2[j]({ key: d.key, value: d.value, state: window.bastion._stateMirror });
        } catch (e2) {}
      }
    }
  });
  window.parent.postMessage({ type: 'bastion_state_init', artifact_id: ARTIFACT_ID }, '*');
})();
</script>`;
}

/**
 * Insert bridge SDK into HTML srcDoc: after <head> if present, else after <html>, else prepend.
 * @param {string} html
 * @param {string | null | undefined} artifactId
 * @returns {string}
 */
export function injectBridgeSdkIntoHtmlSrcDoc(html, artifactId) {
  const sdk = buildBridgeSdkScript(artifactId);
  const s = String(html ?? '');
  const headMatch = s.match(/<head[^>]*>/i);
  if (headMatch) {
    return s.replace(headMatch[0], headMatch[0] + sdk);
  }
  const htmlMatch = s.match(/<html[^>]*>/i);
  if (htmlMatch) {
    return s.replace(htmlMatch[0], htmlMatch[0] + sdk);
  }
  return sdk + s;
}

function normalizePath(endpoint) {
  let raw = String(endpoint).trim();
  if (!raw) return null;
  try {
    if (/^https?:\/\//i.test(raw)) {
      const u = new URL(raw);
      raw = u.pathname + (u.search || '');
    }
  } catch {
    return null;
  }
  let pathOnly = raw.split('?')[0].trim();
  if (pathOnly.includes('..')) return null;
  while (pathOnly.length > 1 && pathOnly.endsWith('/')) {
    pathOnly = pathOnly.slice(0, -1);
  }
  if (!pathOnly.startsWith('/api/')) {
    return null;
  }
  return pathOnly;
}

function buildQueryString(params) {
  if (!params || typeof params !== 'object') return '';
  const query = new URLSearchParams();
  Object.keys(params).forEach((k) => {
    const v = params[k];
    if (v === undefined || v === null) return;
    if (Array.isArray(v)) {
      v.forEach((item) => query.append(k, String(item)));
    } else {
      query.append(k, String(v));
    }
  });
  const qs = query.toString();
  return qs ? `?${qs}` : '';
}

async function handleQueryMessage(event, data) {
  const source = event.source;
  if (!source || typeof source.postMessage !== 'function') return;

  const { id, endpoint, params } = data;
  const respond = (payload) => {
    try {
      source.postMessage({ type: 'bastion_response', id, ...payload }, '*');
    } catch {
      // ignore
    }
  };

  if (!id || typeof endpoint !== 'string') {
    respond({ error: 'Invalid request' });
    return;
  }

  const pathOnly = normalizePath(endpoint);
  if (!pathOnly) {
    respond({ error: 'Endpoint not allowed' });
    return;
  }

  if (!isEndpointAllowed(pathOnly)) {
    respond({ error: 'Endpoint not allowed' });
    return;
  }

  if (concurrentBridgeRequests >= BRIDGE_MAX_CONCURRENT) {
    respond({ error: 'Too many concurrent requests' });
    return;
  }

  if (!isUnderRateLimit()) {
    respond({ error: 'Rate limited' });
    return;
  }

  concurrentBridgeRequests += 1;
  recordBridgeRequest();

  try {
    const token = typeof localStorage !== 'undefined' ? localStorage.getItem('auth_token') : null;
    const url = `${API_BASE}${pathOnly}${buildQueryString(params)}`;
    const headers = { Accept: 'application/json' };
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const res = await fetch(url, { method: 'GET', headers });
    const text = await res.text();
    let body;
    try {
      body = JSON.parse(text);
    } catch {
      respond({ error: 'Non-JSON response' });
      return;
    }

    if (!res.ok) {
      let msg = res.statusText || `HTTP ${res.status}`;
      if (body && body.detail !== undefined) {
        msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      }
      respond({ error: msg });
      return;
    }

    respond({ result: body });
  } catch (e) {
    respond({ error: e?.message || String(e) });
  } finally {
    concurrentBridgeRequests -= 1;
  }
}

function handleStateAndNotifyMessage(event, data) {
  const source = event.source;
  if (!source || typeof source.postMessage !== 'function') return;

  const t = data?.type;
  if (t === 'bastion_state_init') {
    const artifactId = data.artifact_id != null ? String(data.artifact_id).trim() : '';
    if (!artifactId) return;
    registerArtifactStateWindow(artifactId, source);
    snapshotToWindow(artifactId, source);
    return;
  }

  if (t === 'bastion_state_set') {
    const artifactId = data.artifact_id != null ? String(data.artifact_id).trim() : '';
    const key = data.key != null ? String(data.key) : '';
    if (!artifactId || !key) return;
    if (!valueSizeOk(data.value)) return;
    const state = getOrCreateArtifactState(artifactId);
    state[key] = data.value;
    broadcastStateChange(artifactId, key, data.value, source);
    return;
  }

  if (t === 'bastion_notify') {
    const artifactId = data.artifact_id != null ? String(data.artifact_id).trim() : '';
    if (!artifactId || !notifyHandler) return;
    const payload = data.payload && typeof data.payload === 'object' ? data.payload : {};
    try {
      notifyHandler({ artifactId, payload });
    } catch {
      // ignore
    }
  }
}

function handleBridgeMessage(event) {
  const data = event.data;
  if (!data || typeof data !== 'object') return;

  if (data.type === 'bastion_query') {
    handleQueryMessage(event, data);
    return;
  }

  if (
    data.type === 'bastion_state_init' ||
    data.type === 'bastion_state_set' ||
    data.type === 'bastion_notify'
  ) {
    handleStateAndNotifyMessage(event, data);
  }
}

/**
 * Register global message listener (ref-counted). Call cleanup on unmount.
 * @returns {() => void}
 */
export function startBridgeListener() {
  listenerRefCount += 1;
  if (listenerRefCount === 1) {
    boundHandler = handleBridgeMessage;
    window.addEventListener('message', boundHandler);
  }
  return () => {
    listenerRefCount -= 1;
    if (listenerRefCount <= 0) {
      listenerRefCount = 0;
      if (boundHandler) {
        window.removeEventListener('message', boundHandler);
        boundHandler = null;
      }
    }
  };
}
