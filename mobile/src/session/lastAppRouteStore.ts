import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_last_app_route_v1';
/** Default when nothing stored or restore invalid. */
export const DEFAULT_APP_HREF = '/(app)/home';

const SKIP_FIRST_SEGMENT = new Set(['shortcut-send', 'voice']);

/**
 * Build an Expo Router href from current segments (file routes under (app)).
 * Returns null for routes we should not treat as "last screen" (shortcuts, etc.).
 */
export function segmentsToPersistedHref(segments: readonly string[]): string | null {
  if (!segments.length || segments[0] !== '(app)') {
    return null;
  }
  const rest = segments.slice(1).filter(Boolean);
  if (!rest.length) {
    return DEFAULT_APP_HREF;
  }
  if (SKIP_FIRST_SEGMENT.has(rest[0])) {
    return null;
  }
  const path = rest.join('/');
  if (!path || path.includes('..')) {
    return null;
  }
  return `/(app)/${path}`;
}

function isAllowedStoredHref(v: string): boolean {
  if (!v.startsWith('/(app)/') || v.includes('..')) {
    return false;
  }
  if (v.length > 480) {
    return false;
  }
  return true;
}

export async function saveLastAppRoute(href: string): Promise<void> {
  if (!isAllowedStoredHref(href)) {
    return;
  }
  try {
    await SecureStore.setItemAsync(KEY, href);
  } catch {
    /* ignore */
  }
}

export async function loadLastAppRoute(): Promise<string> {
  try {
    const v = await SecureStore.getItemAsync(KEY);
    if (v && isAllowedStoredHref(v)) {
      return v;
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_APP_HREF;
}

export async function clearLastAppRoute(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    /* ignore */
  }
}
