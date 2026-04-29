import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { getCurrentUser, login as apiLogin, logout as apiLogout } from '../api/auth';
import { getApiBaseUrl, normalizeBastionOrigin, setRuntimeApiBaseUrl } from '../api/config';
import { getStoredBaseUrl, setStoredBaseUrl } from '../session/baseUrlStore';
import { clearStoredToken, getStoredToken } from '../session/tokenStore';

export type AuthContextValue = {
  token: string | null;
  user: Record<string, unknown> | null;
  isReady: boolean;
  apiConfigured: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  /** Persist origin, apply for API calls, and clear session when the origin changes. */
  setServerUrl: (rawUrl: string) => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<Record<string, unknown> | null>(null);
  const [isReady, setIsReady] = useState(false);
  /** Bumps when server URL is loaded or saved so apiConfigured recomputes. */
  const [serverEpoch, setServerEpoch] = useState(0);

  const apiConfigured = useMemo(() => !!getApiBaseUrl(), [serverEpoch, isReady]);

  const loadSession = useCallback(async () => {
    const storedBase = await getStoredBaseUrl();
    setRuntimeApiBaseUrl(storedBase ?? '');
    setServerEpoch((n) => n + 1);

    const t = await getStoredToken();
    setToken(t);
    if (t) {
      try {
        const me = await getCurrentUser();
        const u =
          me && typeof me === 'object' && 'user_id' in me
            ? (me as Record<string, unknown>)
            : me && typeof me === 'object' && 'user' in me && me.user && typeof me.user === 'object'
              ? (me.user as Record<string, unknown>)
              : null;
        setUser(u);
      } catch {
        setUser(null);
        await clearStoredToken();
        setToken(null);
      }
    } else {
      setUser(null);
    }
    setIsReady(true);
  }, []);

  useEffect(() => {
    void loadSession();
  }, [loadSession]);

  const setServerUrl = useCallback(async (rawUrl: string) => {
    const prev = getApiBaseUrl();
    const normalized = normalizeBastionOrigin(rawUrl);
    await setStoredBaseUrl(normalized);
    setRuntimeApiBaseUrl(normalized);
    setServerEpoch((n) => n + 1);
    if (normalized !== prev) {
      await clearStoredToken();
      setToken(null);
      setUser(null);
    }
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const res = await apiLogin(username, password);
    setToken(res.access_token);
    setUser((res.user ?? null) as Record<string, unknown> | null);
  }, []);

  const logout = useCallback(async () => {
    await apiLogout();
    setToken(null);
    setUser(null);
  }, []);

  const refreshUser = useCallback(async () => {
    if (!token) return;
    try {
      const me = await getCurrentUser();
      const u =
        me && typeof me === 'object' && 'user_id' in me
          ? (me as Record<string, unknown>)
          : me && typeof me === 'object' && 'user' in me && me.user && typeof me.user === 'object'
            ? (me.user as Record<string, unknown>)
            : null;
      setUser(u);
    } catch {
      setUser(null);
    }
  }, [token]);

  const value = useMemo(
    () => ({
      token,
      user,
      isReady,
      apiConfigured,
      login,
      logout,
      refreshUser,
      setServerUrl,
    }),
    [token, user, isReady, apiConfigured, login, logout, refreshUser, setServerUrl]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return ctx;
}
