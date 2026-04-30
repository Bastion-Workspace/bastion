import { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
  useColorScheme,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import {
  fetchOpdsAtom,
  getEbooksSettings,
  kosyncHealth,
  kosyncRegister,
  kosyncTest,
  putEbooksSettings,
  putKosyncSettings,
  type OpdsCatalogEntryInput,
  type OpdsCatalogEntryResponse,
} from '../../../src/api/ebooks';
import { isApiError } from '../../../src/api/client';
import { getColors } from '../../../src/theme/colors';

function newCatalogId(): string {
  return `cat_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function apiErrorDetail(e: unknown): string {
  if (isApiError(e) && e.body && typeof e.body === 'object') {
    const d = (e.body as { detail?: unknown }).detail;
    if (typeof d === 'string') return d;
  }
  return e instanceof Error ? e.message : 'Request failed';
}

type CatalogRow = OpdsCatalogEntryResponse & { http_basic_b64?: string };

export default function EbooksSettingsScreen() {
  const scheme = useColorScheme();
  const c = useMemo(() => getColors(scheme === 'dark' ? 'dark' : 'light'), [scheme]);
  const [loading, setLoading] = useState(true);
  const [catalogs, setCatalogs] = useState<CatalogRow[]>([]);
  const [basicUser, setBasicUser] = useState<Record<string, string>>({});
  const [basicPass, setBasicPass] = useState<Record<string, string>>({});
  const [ksBase, setKsBase] = useState('');
  const [ksUser, setKsUser] = useState('');
  const [ksPass, setKsPass] = useState('');
  const [ksVerify, setKsVerify] = useState(true);
  const [msg, setMsg] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const s = await getEbooksSettings();
      setCatalogs((s.catalogs || []).map((x) => ({ ...x })));
      setKsBase(s.kosync?.base_url || '');
      setKsUser(s.kosync?.username || '');
      setKsPass('');
      setKsVerify(s.kosync?.verify_ssl !== false);
    } catch (e) {
      setErr(apiErrorDetail(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      void load();
    }, [load])
  );

  const onField = useCallback((id: string, field: 'title' | 'root_url' | 'verify_ssl' | 'http_basic_b64', value: string | boolean) => {
    setCatalogs((prev) => prev.map((row) => (row.id === id ? { ...row, [field]: value } : row)));
  }, []);

  const applyBasicAuth = useCallback(
    (id: string) => {
      const u = (basicUser[id] || '').trim();
      const p = basicPass[id] || '';
      if (!u && !p) {
        onField(id, 'http_basic_b64', '');
        return;
      }
      const token = typeof btoa !== 'undefined' ? btoa(`${u}:${p}`) : '';
      onField(id, 'http_basic_b64', token);
    },
    [basicUser, basicPass, onField]
  );

  const onSaveCatalogs = useCallback(async () => {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const trimmed: OpdsCatalogEntryInput[] = catalogs.map((row) => {
        const title = (row.title || '').trim();
        const root_url = (row.root_url || '').trim();
        const entry: OpdsCatalogEntryInput = {
          id: row.id,
          title,
          root_url,
          verify_ssl: row.verify_ssl !== false,
        };
        if (row.http_basic_b64 !== undefined) entry.http_basic_b64 = row.http_basic_b64;
        return entry;
      });
      if (trimmed.some((x) => !x.root_url)) {
        setErr('Each catalog needs a root URL.');
        setBusy(false);
        return;
      }
      await putEbooksSettings({ catalogs: trimmed });
      setMsg('Catalogs saved.');
      await load();
    } catch (e) {
      setErr(apiErrorDetail(e));
    } finally {
      setBusy(false);
    }
  }, [catalogs, load]);

  const onSaveKosync = useCallback(async () => {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      await putKosyncSettings({
        base_url: ksBase.trim(),
        username: ksUser.trim(),
        password: ksPass || undefined,
        verify_ssl: ksVerify,
      });
      setKsPass('');
      setMsg('KoSync settings saved.');
      await load();
    } catch (e) {
      setErr(apiErrorDetail(e));
    } finally {
      setBusy(false);
    }
  }, [ksBase, ksUser, ksPass, ksVerify, load]);

  const onTestKosync = useCallback(async () => {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const data = await kosyncTest({
        base_url: ksBase.trim(),
        username: ksUser.trim(),
        password: ksPass,
        verify_ssl: ksVerify,
      });
      setMsg(data?.ok ? 'KoSync accepted these credentials.' : `Test: ${JSON.stringify(data)}`);
    } catch (e) {
      setErr(apiErrorDetail(e));
    } finally {
      setBusy(false);
    }
  }, [ksBase, ksUser, ksPass, ksVerify]);

  const onHealth = useCallback(async () => {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      const data = await kosyncHealth();
      setMsg(data?.ok ? 'KoSync server reachable.' : JSON.stringify(data));
    } catch (e) {
      setErr(apiErrorDetail(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const onRegister = useCallback(async () => {
    setBusy(true);
    setErr(null);
    setMsg(null);
    try {
      await kosyncRegister({
        username: ksUser.trim(),
        password: ksPass,
        base_url: ksBase.trim() || null,
        verify_ssl: ksVerify,
      });
      setKsPass('');
      setMsg('Registered on KoSync.');
      await load();
    } catch (e) {
      setErr(apiErrorDetail(e));
    } finally {
      setBusy(false);
    }
  }, [ksBase, ksUser, ksPass, ksVerify, load]);

  const onTestCatalog = useCallback(
    async (cat: CatalogRow) => {
      setBusy(true);
      setErr(null);
      setMsg(null);
      try {
        await fetchOpdsAtom({ catalog_id: cat.id, url: cat.root_url.trim(), want: 'atom' });
        setMsg(`OPDS OK: ${cat.title}`);
      } catch (e) {
        setErr(apiErrorDetail(e));
      } finally {
        setBusy(false);
      }
    },
    []
  );

  return (
    <ScrollView
      style={[styles.root, { backgroundColor: c.background }]}
      contentContainerStyle={styles.pad}
      keyboardShouldPersistTaps="handled"
    >
      {loading ? <ActivityIndicator color={c.text} style={styles.spinner} /> : null}
      {msg ? <Text style={[styles.banner, { color: c.link }]}>{msg}</Text> : null}
      {err ? <Text style={[styles.banner, { color: c.danger }]}>{err}</Text> : null}

      <Text style={[styles.section, { color: c.textSecondary }]}>OPDS CATALOGS</Text>
      {catalogs.map((cat) => (
        <View key={cat.id} style={[styles.card, { borderColor: c.border, backgroundColor: c.surface }]}>
          <Text style={[styles.label, { color: c.textSecondary }]}>Title</Text>
          <TextInput
            style={[styles.input, { borderColor: c.border, color: c.text }]}
            value={cat.title}
            onChangeText={(t) => onField(cat.id, 'title', t)}
          />
          <Text style={[styles.label, { color: c.textSecondary }]}>Root URL</Text>
          <TextInput
            style={[styles.input, { borderColor: c.border, color: c.text }]}
            value={cat.root_url}
            onChangeText={(t) => onField(cat.id, 'root_url', t)}
            autoCapitalize="none"
            autoCorrect={false}
          />
          <View style={styles.rowBetween}>
            <Text style={{ color: c.text }}>Verify SSL</Text>
            <Switch value={cat.verify_ssl !== false} onValueChange={(v) => onField(cat.id, 'verify_ssl', v)} />
          </View>
          <Text style={[styles.label, { color: c.textSecondary }]}>HTTP Basic (optional)</Text>
          <TextInput
            style={[styles.input, { borderColor: c.border, color: c.text }]}
            placeholder="Username"
            placeholderTextColor={c.textSecondary}
            value={basicUser[cat.id] || ''}
            onChangeText={(t) => setBasicUser((m) => ({ ...m, [cat.id]: t }))}
            autoCapitalize="none"
          />
          <TextInput
            style={[styles.input, { borderColor: c.border, color: c.text }]}
            placeholder="Password"
            placeholderTextColor={c.textSecondary}
            secureTextEntry
            value={basicPass[cat.id] || ''}
            onChangeText={(t) => setBasicPass((m) => ({ ...m, [cat.id]: t }))}
          />
          <Pressable style={styles.secondaryBtn} onPress={() => applyBasicAuth(cat.id)}>
            <Text style={[styles.secondaryBtnText, { color: c.link }]}>Apply basic auth to this catalog</Text>
          </Pressable>
          {cat.http_basic_configured ? (
            <Text style={[styles.hint, { color: c.textSecondary }]}>HTTP basic is configured on the server.</Text>
          ) : null}
          <View style={styles.rowGap}>
            <Pressable style={[styles.btn, { backgroundColor: c.chipBgActive }]} onPress={() => void onTestCatalog(cat)} disabled={busy}>
              <Text style={[styles.btnText, { color: c.chipTextActive }]}>Test fetch</Text>
            </Pressable>
            <Pressable
              style={[styles.btn, { backgroundColor: c.surfaceMuted }]}
              onPress={() => setCatalogs((p) => p.filter((x) => x.id !== cat.id))}
            >
              <Text style={[styles.btnText, { color: c.danger }]}>Remove</Text>
            </Pressable>
          </View>
        </View>
      ))}
      <Pressable style={styles.linkBtn} onPress={() => setCatalogs((p) => [...p, { id: newCatalogId(), title: 'New catalog', root_url: '', verify_ssl: true, http_basic_configured: false }])}>
        <Text style={[styles.linkBtnText, { color: c.link }]}>Add catalog</Text>
      </Pressable>
      <Pressable style={[styles.btn, { backgroundColor: c.chipBgActive, marginTop: 12 }]} onPress={() => void onSaveCatalogs()} disabled={busy}>
        <Text style={[styles.btnText, { color: c.chipTextActive }]}>Save catalogs</Text>
      </Pressable>

      <Text style={[styles.section, { color: c.textSecondary }]}>KOSYNC</Text>
      <View style={[styles.card, { borderColor: c.border, backgroundColor: c.surface }]}>
        <Text style={[styles.label, { color: c.textSecondary }]}>Server base URL</Text>
        <TextInput
          style={[styles.input, { borderColor: c.border, color: c.text }]}
          value={ksBase}
          onChangeText={setKsBase}
          autoCapitalize="none"
          autoCorrect={false}
        />
        <Text style={[styles.label, { color: c.textSecondary }]}>Username</Text>
        <TextInput
          style={[styles.input, { borderColor: c.border, color: c.text }]}
          value={ksUser}
          onChangeText={setKsUser}
          autoCapitalize="none"
        />
        <Text style={[styles.label, { color: c.textSecondary }]}>Password (set to replace key)</Text>
        <TextInput
          style={[styles.input, { borderColor: c.border, color: c.text }]}
          value={ksPass}
          onChangeText={setKsPass}
          secureTextEntry
        />
        <View style={styles.rowBetween}>
          <Text style={{ color: c.text }}>Verify SSL</Text>
          <Switch value={ksVerify} onValueChange={setKsVerify} />
        </View>
        <View style={styles.rowGap}>
          <Pressable style={[styles.btn, { backgroundColor: c.chipBgActive }]} onPress={() => void onSaveKosync()} disabled={busy}>
            <Text style={[styles.btnText, { color: c.chipTextActive }]}>Save KoSync</Text>
          </Pressable>
          <Pressable style={[styles.btn, { backgroundColor: c.surfaceMuted }]} onPress={() => void onTestKosync()} disabled={busy}>
            <Text style={[styles.btnText, { color: c.text }]}>Test login</Text>
          </Pressable>
          <Pressable style={[styles.btn, { backgroundColor: c.surfaceMuted }]} onPress={() => void onHealth()} disabled={busy}>
            <Text style={[styles.btnText, { color: c.text }]}>Health</Text>
          </Pressable>
          <Pressable style={[styles.btn, { backgroundColor: c.surfaceMuted }]} onPress={() => void onRegister()} disabled={busy}>
            <Text style={[styles.btnText, { color: c.text }]}>Register</Text>
          </Pressable>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  pad: { padding: 16, paddingBottom: 120 },
  spinner: { marginVertical: 16 },
  banner: { marginBottom: 8, fontSize: 14 },
  section: { fontSize: 13, fontWeight: '700', marginTop: 20, letterSpacing: 0.5 },
  card: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 12, padding: 12, marginTop: 10 },
  label: { fontSize: 12, marginTop: 8 },
  input: { borderWidth: StyleSheet.hairlineWidth, borderRadius: 8, paddingHorizontal: 10, paddingVertical: 10, marginTop: 4 },
  rowBetween: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 },
  rowGap: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 12 },
  hint: { fontSize: 12, marginTop: 6 },
  btn: { paddingHorizontal: 14, paddingVertical: 10, borderRadius: 8 },
  btnText: { fontWeight: '700' },
  linkBtn: { marginTop: 10, alignSelf: 'flex-start' },
  linkBtnText: { fontWeight: '600', textDecorationLine: 'underline' },
  secondaryBtn: { marginTop: 8 },
  secondaryBtnText: { fontWeight: '600' },
});
