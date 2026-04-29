import Constants from 'expo-constants';

/** Set from SecureStore on startup or when the user saves “Server URL” in the app. */
let runtimeBaseUrl = '';

/**
 * Override used before build-time env / app.config extra so a packaged APK can target any server.
 */
export function setRuntimeApiBaseUrl(url: string): void {
  runtimeBaseUrl = (url || '').trim().replace(/\/$/, '');
}

/**
 * Bastion HTTP(S) origin without trailing slash, used for REST and WebSocket host.
 * Order: in-app saved URL, then EXPO_PUBLIC_API_BASE_URL, then app.config extra.
 */
export function getApiBaseUrl(): string {
  if (runtimeBaseUrl) {
    return runtimeBaseUrl;
  }
  const fromEnv = process.env.EXPO_PUBLIC_API_BASE_URL?.trim();
  if (fromEnv) {
    return fromEnv.replace(/\/$/, '');
  }
  const extra = Constants.expoConfig?.extra as { apiBaseUrl?: string } | undefined;
  const fromExtra = extra?.apiBaseUrl?.trim();
  if (fromExtra) {
    return fromExtra.replace(/\/$/, '');
  }
  return '';
}

/** Normalize user input to an origin (scheme + host [+ port]); no path or trailing slash. */
export function normalizeBastionOrigin(input: string): string {
  let s = input.trim();
  if (!s) {
    throw new Error('Enter your server URL.');
  }
  if (!/^https?:\/\//i.test(s)) {
    s = `https://${s}`;
  }
  let u: URL;
  try {
    u = new URL(s);
  } catch {
    throw new Error('Invalid URL. Example: https://bastion.example.com or http://10.0.2.2:3051');
  }
  if (!u.hostname) {
    throw new Error('Invalid URL: missing hostname.');
  }
  return `${u.protocol}//${u.host}`.replace(/\/$/, '');
}

export function assertApiBaseUrl(): string {
  const base = getApiBaseUrl();
  if (!base) {
    throw new Error(
      'Configure the Bastion server URL in the app, or set EXPO_PUBLIC_API_BASE_URL at build time.'
    );
  }
  return base;
}

/** Turn a relative `/api/...` path into an absolute URL for images and links. */
export function resolveAbsoluteApiUrl(pathOrUrl: string): string {
  const s = (pathOrUrl || '').trim();
  if (!s) return '';
  if (/^https?:\/\//i.test(s)) return s;
  const base = getApiBaseUrl();
  if (!base) return s;
  if (s.startsWith('/')) return `${base}${s}`;
  return s;
}

/** Rewrite markdown/HTML so asset paths use the configured API origin. */
export function absolutizeMessageMediaRefs(content: string): string {
  const base = getApiBaseUrl();
  if (!base || !content) return content;
  let out = content;
  out = out.replace(/]\(\//g, `](${base}/`);
  out = out.replace(/src="(\/[^"]+)"/gi, (_m, p: string) => `src="${base}${p}"`);
  out = out.replace(/src='(\/[^']+)'/gi, (_m, p: string) => `src='${base}${p}'`);
  return out;
}

export function wsUrlFromHttpBase(pathWithLeadingSlash: string, query: Record<string, string>): string {
  const base = assertApiBaseUrl();
  const u = new URL(base);
  const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
  const q = new URLSearchParams(query).toString();
  const qs = q ? `?${q}` : '';
  return `${proto}//${u.host}${pathWithLeadingSlash}${qs}`;
}
