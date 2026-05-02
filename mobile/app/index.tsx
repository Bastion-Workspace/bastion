import { Redirect } from 'expo-router';
import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, View } from 'react-native';
import { ScreenShell } from '../src/components/ScreenShell';
import { useAuth } from '../src/context/AuthContext';
import { loadLastAppRoute } from '../src/session/lastAppRouteStore';
import { loadLastEbookParams } from '../src/session/lastEbookParamsStore';

export default function Index() {
  const { token, isReady, apiConfigured } = useAuth();
  const [bootHref, setBootHref] = useState<string | null>(null);

  useEffect(() => {
    if (!isReady || !apiConfigured || !token) {
      return;
    }
    let cancelled = false;
    void (async () => {
      const base = await loadLastAppRoute();
      let href = base;
      if (base === '/(app)/ebooks/reader' || base.startsWith('/(app)/ebooks/reader')) {
        const p = await loadLastEbookParams();
        if (p) {
          const q = new URLSearchParams();
          q.set('catalogId', p.catalogId);
          q.set('acquisitionUrl', p.acquisitionUrl);
          if (p.title) q.set('title', p.title);
          if (p.digest) q.set('digest', p.digest);
          if (p.format) q.set('format', p.format);
          href = `/(app)/ebooks/reader?${q.toString()}`;
        }
      }
      if (!cancelled) {
        setBootHref(href);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [isReady, apiConfigured, token]);

  if (!isReady) {
    return (
      <ScreenShell>
        <View style={styles.center}>
          <ActivityIndicator size="large" />
        </View>
      </ScreenShell>
    );
  }

  if (!apiConfigured) {
    return <Redirect href="/(auth)/server" />;
  }

  if (!token) {
    return <Redirect href="/(auth)/login" />;
  }

  if (bootHref === null) {
    return (
      <ScreenShell>
        <View style={styles.center}>
          <ActivityIndicator size="large" />
        </View>
      </ScreenShell>
    );
  }

  return <Redirect href={bootHref} />;
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 24 },
});
