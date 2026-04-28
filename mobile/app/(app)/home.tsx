import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useRouter } from 'expo-router';
import { getApiBaseUrl } from '../../src/api/config';
import { useAuth } from '../../src/context/AuthContext';

export default function ProfileScreen() {
  const { user, logout } = useAuth();
  const router = useRouter();
  const name = (user?.display_name as string) || (user?.username as string) || 'User';

  return (
    <View style={styles.container}>
      <Text style={styles.label}>Signed in as</Text>
      <Text style={styles.name}>{name}</Text>

      <Text style={styles.sectionLabel}>Server</Text>
      <Text style={styles.url} numberOfLines={3}>
        {getApiBaseUrl() || '—'}
      </Text>
      <Pressable style={styles.linkBtn} onPress={() => router.push('/(auth)/server')}>
        <Text style={styles.linkBtnText}>Change server</Text>
      </Pressable>

      <Pressable style={styles.logout} onPress={() => void logout()}>
        <Text style={styles.logoutText}>Sign out</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 8 },
  label: { fontSize: 14, color: '#666' },
  name: { fontSize: 22, fontWeight: '700', marginBottom: 16 },
  sectionLabel: { fontSize: 14, color: '#666', marginTop: 8 },
  url: { fontSize: 13, color: '#333', marginTop: 4 },
  linkBtn: {
    alignSelf: 'flex-start',
    marginTop: 8,
    paddingVertical: 8,
    paddingHorizontal: 4,
  },
  linkBtnText: { fontSize: 15, color: '#1a1a2e', fontWeight: '600', textDecorationLine: 'underline' },
  logout: { marginTop: 32, padding: 14, alignItems: 'center' },
  logoutText: { color: '#c00', fontSize: 16, fontWeight: '600' },
});
