import * as SecureStore from 'expo-secure-store';
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

/** Sentinel for cross-feed view (must not collide with UUID feed_id). */
export const RSS_SOURCE_ALL = '__all__';

const KEY_SOURCE = 'rss:reader:source';
const KEY_AUTO_MARK = 'rss:reader:autoMarkRead';
const KEY_ARTICLE_FONT_SIZE = 'rss:reader:articleFontSize';
const KEY_ARTICLE_FONT = 'rss:reader:articleFontFamily';
const KEY_ARTICLE_THEME = 'rss:reader:articleTheme';

export type RssReaderTheme = 'auto' | 'light' | 'sepia' | 'dark';
export type RssArticleFontFamily = 'sans' | 'serif' | 'mono';

type Ctx = {
  source: string;
  setSource: (next: string) => Promise<void>;
  autoMarkRead: boolean;
  setAutoMarkRead: (next: boolean) => Promise<void>;
  articleFontSize: number;
  setArticleFontSize: (next: number) => Promise<void>;
  articleFontFamily: RssArticleFontFamily;
  setArticleFontFamily: (next: RssArticleFontFamily) => Promise<void>;
  articleTheme: RssReaderTheme;
  setArticleTheme: (next: RssReaderTheme) => Promise<void>;
  hydrated: boolean;
  refreshFromStore: () => Promise<void>;
};

const RssPrefsContext = createContext<Ctx | null>(null);

const FONT_MIN = 12;
const FONT_MAX = 32;

function clampFontSize(n: number): number {
  return Math.min(FONT_MAX, Math.max(FONT_MIN, Math.round(n)));
}

function parseFontFamily(raw: string | null): RssArticleFontFamily {
  if (raw === 'mono') return 'mono';
  if (raw === 'serif') return 'serif';
  if (raw === 'sans') return 'sans';
  return 'serif';
}

function parseArticleTheme(raw: string | null): RssReaderTheme {
  if (raw === 'light' || raw === 'sepia' || raw === 'dark' || raw === 'auto') return raw;
  return 'auto';
}

export function RssPrefsProvider({ children }: { children: ReactNode }) {
  const [source, setSourceState] = useState<string>(RSS_SOURCE_ALL);
  const [autoMarkRead, setAutoMarkReadState] = useState(false);
  const [articleFontSize, setArticleFontSizeState] = useState(20);
  const [articleFontFamily, setArticleFontFamilyState] = useState<RssArticleFontFamily>('serif');
  const [articleTheme, setArticleThemeState] = useState<RssReaderTheme>('auto');
  const [hydrated, setHydrated] = useState(false);

  const applyFromStore = useCallback((keys: {
    s: string | null;
    a: string | null;
    fs: string | null;
    ff: string | null;
    th: string | null;
  }) => {
      const { s, a, fs, ff, th } = keys;
      if (s && s.trim()) {
        setSourceState(s.trim());
      } else {
        setSourceState(RSS_SOURCE_ALL);
      }
      setAutoMarkReadState(a === 'true' || a === '1');
      if (fs != null && fs.trim() !== '') {
        const n = Number.parseInt(fs.trim(), 10);
        if (!Number.isNaN(n)) {
          setArticleFontSizeState(clampFontSize(n));
        }
      }
      setArticleFontFamilyState(parseFontFamily(ff));
      setArticleThemeState(parseArticleTheme(th));
    },
    []
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [s, a, fs, ff, th] = await Promise.all([
          SecureStore.getItemAsync(KEY_SOURCE),
          SecureStore.getItemAsync(KEY_AUTO_MARK),
          SecureStore.getItemAsync(KEY_ARTICLE_FONT_SIZE),
          SecureStore.getItemAsync(KEY_ARTICLE_FONT),
          SecureStore.getItemAsync(KEY_ARTICLE_THEME),
        ]);
        if (cancelled) return;
        applyFromStore({ s, a, fs, ff, th });
      } catch {
        /* keep defaults */
      } finally {
        if (!cancelled) setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [applyFromStore]);

  const setSource = useCallback(async (next: string) => {
    setSourceState(next);
    try {
      await SecureStore.setItemAsync(KEY_SOURCE, next);
    } catch {
      /* ignore */
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

  const setArticleFontSize = useCallback(async (next: number) => {
    const clamped = clampFontSize(next);
    setArticleFontSizeState(clamped);
    try {
      await SecureStore.setItemAsync(KEY_ARTICLE_FONT_SIZE, String(clamped));
    } catch {
      /* ignore */
    }
  }, []);

  const setArticleFontFamily = useCallback(async (next: RssArticleFontFamily) => {
    setArticleFontFamilyState(next);
    try {
      await SecureStore.setItemAsync(KEY_ARTICLE_FONT, next);
    } catch {
      /* ignore */
    }
  }, []);

  const setArticleTheme = useCallback(async (next: RssReaderTheme) => {
    setArticleThemeState(next);
    try {
      await SecureStore.setItemAsync(KEY_ARTICLE_THEME, next);
    } catch {
      /* ignore */
    }
  }, []);

  const refreshFromStore = useCallback(async () => {
    try {
      const [s, a, fs, ff, th] = await Promise.all([
        SecureStore.getItemAsync(KEY_SOURCE),
        SecureStore.getItemAsync(KEY_AUTO_MARK),
        SecureStore.getItemAsync(KEY_ARTICLE_FONT_SIZE),
        SecureStore.getItemAsync(KEY_ARTICLE_FONT),
        SecureStore.getItemAsync(KEY_ARTICLE_THEME),
      ]);
      applyFromStore({ s, a, fs, ff, th });
    } catch {
      /* keep in-memory values */
    }
  }, [applyFromStore]);

  const value = useMemo(
    () => ({
      source,
      setSource,
      autoMarkRead,
      setAutoMarkRead,
      articleFontSize,
      setArticleFontSize,
      articleFontFamily,
      setArticleFontFamily,
      articleTheme,
      setArticleTheme,
      hydrated,
      refreshFromStore,
    }),
    [
      source,
      setSource,
      autoMarkRead,
      setAutoMarkRead,
      articleFontSize,
      setArticleFontSize,
      articleFontFamily,
      setArticleFontFamily,
      articleTheme,
      setArticleTheme,
      hydrated,
      refreshFromStore,
    ]
  );

  return <RssPrefsContext.Provider value={value}>{children}</RssPrefsContext.Provider>;
}

export function useRssPrefs(): Ctx {
  const v = useContext(RssPrefsContext);
  if (!v) {
    throw new Error('useRssPrefs must be used within RssPrefsProvider');
  }
  return v;
}
