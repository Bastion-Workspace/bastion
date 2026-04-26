import Constants from 'expo-constants';

/**
 * Bastion HTTP(S) origin without trailing slash, used for REST and WebSocket host.
 */
export function getApiBaseUrl(): string {
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

export function assertApiBaseUrl(): string {
  const base = getApiBaseUrl();
  if (!base) {
    throw new Error(
      'Set EXPO_PUBLIC_API_BASE_URL to your Bastion server origin (e.g. https://your-host:3051)'
    );
  }
  return base;
}

export function wsUrlFromHttpBase(pathWithLeadingSlash: string, query: Record<string, string>): string {
  const base = assertApiBaseUrl();
  const u = new URL(base);
  const proto = u.protocol === 'https:' ? 'wss:' : 'ws:';
  const q = new URLSearchParams(query).toString();
  const qs = q ? `?${q}` : '';
  return `${proto}//${u.host}${pathWithLeadingSlash}${qs}`;
}
