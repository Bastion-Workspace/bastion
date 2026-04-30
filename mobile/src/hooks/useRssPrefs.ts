import * as SecureStore from 'expo-secure-store';
import { useCallback, useEffect, useState } from 'react';

const KEY_SOURCE = 'rss:reader:source';
const KEY_AUTO_MARK = 'rss:reader:autoMarkRead';

/** Sentinel for cross-feed view (must not collide with UUID feed_id). */
export const RSS_SOURCE_ALL = '__all__';

export function useRssPrefs() {
  const [source, setSourceState] = useState<string>(RSS_SOURCE_ALL);
  const [autoMarkRead, setAutoMarkReadState] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [s, a] = await Promise.all([
          SecureStore.getItemAsync(KEY_SOURCE),
          SecureStore.getItemAsync(KEY_AUTO_MARK),
        ]);
        if (cancelled) return;
        if (s && s.trim()) {
          setSourceState(s.trim());
        }
        if (a === 'true' || a === '1') {
          setAutoMarkReadState(true);
        }
      } catch {
        /* use defaults */
      } finally {
        if (!cancelled) setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const setSource = useCallback(async (next: string) => {
    setSourceState(next);
    try {
      await SecureStore.setItemAsync(KEY_SOURCE, next);
    } catch {
      /* ignore persistence failure */
    }
  }, []);

  const setAutoMarkRead = useCallback(async (next: boolean) => {
    setAutoMarkReadState(next);
    try {
      await SecureStore.setItemAsync(KEY_AUTO_MARK, next ? 'true' : 'false');
    } catch {
      /* ignore */
    }
  }, []);

  /** Re-read SecureStore so multiple screens using this hook stay aligned after navigation. */
  const refreshFromStore = useCallback(async () => {
    try {
      const [s, a] = await Promise.all([
        SecureStore.getItemAsync(KEY_SOURCE),
        SecureStore.getItemAsync(KEY_AUTO_MARK),
      ]);
      if (s && s.trim()) {
        setSourceState(s.trim());
      }
      setAutoMarkReadState(a === 'true' || a === '1');
    } catch {
      /* keep current in-memory values */
    }
  }, []);

  return { source, setSource, autoMarkRead, setAutoMarkRead, hydrated, refreshFromStore };
}
