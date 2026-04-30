import { useCallback, useMemo } from 'react';
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { getApiBaseUrl } from '../../src/api/config';
import { useAuth } from '../../src/context/AuthContext';
import { useAppearancePreference, type AppearancePreference } from '../../src/context/AppearancePreferenceContext';
import { useRssPrefs } from '../../src/hooks/useRssPrefs';
import { getColors } from '../../src/theme/colors';

const APPEARANCE_OPTIONS: { value: AppearancePreference; label: string; sub: string }[] = [
  { value: 'system', label: 'System', sub: 'Match device light or dark mode' },
  { value: 'light', label: 'Light', sub: 'Always light appearance' },
  { value: 'dark', label: 'Dark', sub: 'Always dark appearance' },
];

export default function SettingsScreen() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const colors = useMemo(() => getColors(scheme), [scheme]);
  const { preference, setPreference, hydrated: appearanceHydrated } = useAppearancePreference();
  const { autoMarkRead, setAutoMarkRead, hydrated: rssPrefsHydrated, refreshFromStore } = useRssPrefs();
  const name = (user?.display_name as string) || (user?.username as string) || 'User';

  useFocusEffect(
    useCallback(() => {
      void refreshFromStore();
    }, [refreshFromStore])
  );

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: colors.background }]}
      contentContainerStyle={styles.scrollContent}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={[styles.sectionLabel, { color: colors.textSecondary }]}>ACCOUNT</Text>
      <Text style={[styles.label, { color: colors.textSecondary }]}>Signed in as</Text>
      <Text style={[styles.name, { color: colors.text }]}>{name}</Text>

      <Text style={[styles.sectionLabel, { color: colors.textSecondary }]}>APPEARANCE</Text>
      {!appearanceHydrated ? (
        <ActivityIndicator color={colors.text} style={styles.spinner} />
      ) : (
        APPEARANCE_OPTIONS.map((opt) => {
          const selected = preference === opt.value;
          return (
            <Pressable
              key={opt.value}
              style={[
                styles.appearanceRow,
                { borderColor: colors.border, backgroundColor: colors.surface },
                selected && { borderColor: colors.link, backgroundColor: colors.surfaceMuted },
              ]}
              onPress={() => void setPreference(opt.value)}
              accessibilityRole="radio"
              accessibilityState={{ selected }}
            >
              <View style={styles.flex1}>
                <Text style={[styles.appearanceLabel, { color: colors.text }]}>{opt.label}</Text>
                <Text style={[styles.appearanceSub, { color: colors.textSecondary }]}>{opt.sub}</Text>
              </View>
              {selected ? <Ionicons name="checkmark-circle" size={22} color={colors.link} /> : null}
            </Pressable>
          );
        })
      )}

      <Text style={[styles.sectionLabel, { color: colors.textSecondary }]}>EBOOKS</Text>
      <Text style={[styles.settingSub, { color: colors.textSecondary }]}>
        OPDS catalogs, downloads, and KoSync use the same settings as the web app.
      </Text>
      <Pressable style={styles.linkBtn} onPress={() => router.push('/(app)/ebooks/settings')}>
        <Text style={[styles.linkBtnText, { color: colors.link }]}>eBooks and OPDS settings</Text>
      </Pressable>

      <Text style={[styles.sectionLabel, { color: colors.textSecondary }]}>RSS</Text>
      <View style={styles.settingRow}>
        <View style={styles.settingTextCol}>
          <Text style={[styles.settingTitle, { color: colors.text }]}>Mark read on scroll</Text>
          <Text style={[styles.settingSub, { color: colors.textSecondary }]}>
            Unread items that stay in view in the RSS list are marked read automatically.
          </Text>
        </View>
        {!rssPrefsHydrated ? (
          <ActivityIndicator color={colors.text} />
        ) : (
          <Switch
            value={autoMarkRead}
            onValueChange={(v) => void setAutoMarkRead(v)}
            accessibilityLabel="Mark RSS articles read when scrolled into view"
          />
        )}
      </View>

      <Text style={[styles.sectionLabel, { color: colors.textSecondary }]}>SERVER</Text>
      <Text style={[styles.url, { color: colors.text }]} numberOfLines={3}>
        {getApiBaseUrl() || '—'}
      </Text>
      <Text style={[styles.settingSub, { color: colors.textSecondary }]}>
        Sign-in email and password (or token) are used when you sign in, not on this screen.
      </Text>
      <Pressable style={styles.linkBtn} onPress={() => router.push('/(auth)/server')}>
        <Text style={[styles.linkBtnText, { color: colors.link }]}>Change server URL</Text>
      </Pressable>

      <Pressable style={styles.logout} onPress={() => void logout()}>
        <Text style={[styles.logoutText, { color: colors.danger }]}>Sign out</Text>
      </Pressable>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  scrollContent: { padding: 24, paddingBottom: 40 },
  sectionLabel: { fontSize: 13, fontWeight: '700', marginTop: 20, letterSpacing: 0.5 },
  label: { fontSize: 14, marginTop: 0 },
  name: { fontSize: 22, fontWeight: '700', marginTop: 4 },
  spinner: { marginVertical: 12 },
  appearanceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    paddingHorizontal: 12,
    marginTop: 6,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
  },
  flex1: { flex: 1, minWidth: 0 },
  appearanceLabel: { fontSize: 16, fontWeight: '600' },
  appearanceSub: { fontSize: 12, marginTop: 2 },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 10,
    paddingHorizontal: 4,
  },
  settingTextCol: { flex: 1, minWidth: 0 },
  settingTitle: { fontSize: 16, fontWeight: '600' },
  settingSub: { fontSize: 13, marginTop: 4, lineHeight: 18 },
  url: { fontSize: 13, marginTop: 6 },
  linkBtn: {
    alignSelf: 'flex-start',
    marginTop: 10,
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  linkBtnText: { fontSize: 15, fontWeight: '600', textDecorationLine: 'underline' },
  logout: { marginTop: 36, padding: 14, alignItems: 'center' },
  logoutText: { fontSize: 16, fontWeight: '600' },
});
