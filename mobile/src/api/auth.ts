import { apiRequest } from './client';
import { clearStoredToken, setStoredToken } from '../session/tokenStore';

export type LoginResponse = {
  access_token: string;
  token_type?: string;
  user: Record<string, unknown>;
  expires_in: number;
};

export type MeResponse = { user?: Record<string, unknown> } | Record<string, unknown>;

export async function login(username: string, password: string): Promise<LoginResponse> {
  const res = await apiRequest<LoginResponse>('/api/auth/login', {
    method: 'POST',
    skipAuth: true,
    body: JSON.stringify({ username, password }),
  });
  if (res.access_token) {
    await setStoredToken(res.access_token);
  }
  return res;
}

export async function logout(): Promise<void> {
  try {
    await apiRequest('/api/auth/logout', { method: 'POST' });
  } catch {
    /* still clear local session */
  }
  await clearStoredToken();
}

export async function getCurrentUser(): Promise<MeResponse> {
  return apiRequest<MeResponse>('/api/auth/me');
}
