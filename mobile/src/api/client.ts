import { jwtDecode } from 'jwt-decode';
import { assertApiBaseUrl, getApiBaseUrl } from './config';
import { clearStoredToken, getStoredToken, setStoredToken } from '../session/tokenStore';

type JwtPayload = { exp?: number };

const AUTH_SKIP = ['/api/auth/login', '/api/auth/register'];

function isExpiringSoon(token: string, skewSeconds: number): boolean {
  try {
    const decoded = jwtDecode<JwtPayload>(token);
    if (!decoded.exp) return true;
    const expMs = decoded.exp * 1000;
    return expMs - Date.now() < skewSeconds * 1000;
  } catch {
    return true;
  }
}

async function refreshIfNeeded(): Promise<void> {
  const token = await getStoredToken();
  if (!token) return;
  if (!isExpiringSoon(token, 300)) return;
  const base = getApiBaseUrl();
  if (!base) return;
  const res = await fetch(`${base}/api/auth/refresh`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
  });
  if (!res.ok) {
    await clearStoredToken();
    return;
  }
  const data = (await res.json()) as { access_token?: string };
  if (data.access_token) {
    await setStoredToken(data.access_token);
  }
}

export type ApiError = Error & { status?: number; body?: unknown };

export function isApiError(e: unknown): e is ApiError {
  return e instanceof Error && typeof (e as ApiError).status === 'number';
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit & { skipAuth?: boolean } = {}
): Promise<T> {
  const { skipAuth, ...req } = init;
  const base = assertApiBaseUrl();
  const url = path.startsWith('http') ? path : `${base}${path.startsWith('/') ? path : `/${path}`}`;

  if (!skipAuth && !AUTH_SKIP.some((p) => url.includes(p))) {
    await refreshIfNeeded();
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> | undefined),
  };

  if (!skipAuth) {
    const token = await getStoredToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }

  let res = await fetch(url, { ...req, headers });

  if (res.status === 401 && !skipAuth && !AUTH_SKIP.some((p) => url.includes(p))) {
    const token = await getStoredToken();
    if (token) {
      const refreshRes = await fetch(`${base}/api/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });
      if (refreshRes.ok) {
        const data = (await refreshRes.json()) as { access_token?: string };
        if (data.access_token) {
          await setStoredToken(data.access_token);
          headers.Authorization = `Bearer ${data.access_token}`;
          res = await fetch(url, { ...req, headers });
        }
      } else {
        await clearStoredToken();
      }
    }
  }

  if (!res.ok) {
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      body = await res.text();
    }
    const err = new Error(`HTTP ${res.status}`) as ApiError;
    err.status = res.status;
    err.body = body;
    throw err;
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export async function apiFetchRaw(
  path: string,
  init: RequestInit & { skipAuth?: boolean } = {}
): Promise<Response> {
  const { skipAuth, ...req } = init;
  const base = assertApiBaseUrl();
  const url = path.startsWith('http') ? path : `${base}${path.startsWith('/') ? path : `/${path}`}`;

  if (!skipAuth && !AUTH_SKIP.some((p) => url.includes(p))) {
    await refreshIfNeeded();
  }

  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string> | undefined),
  };
  if (!skipAuth) {
    const token = await getStoredToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
  }
  if (req.body && !headers['Content-Type'] && !(req.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }

  return fetch(url, { ...req, headers });
}
