import { Pressable, StyleSheet, Text, View } from 'react-native';
import { Link } from 'expo-router';
import { useAuth } from '../../src/context/AuthContext';

export default function HomeScreen() {
  const { user, logout } = useAuth();
  const name = (user?.display_name as string) || (user?.username as string) || 'User';

  return (
    <View style={styles.container}>
      <Text style={styles.greeting}>Signed in as {name}</Text>
      <Link href="/todos" asChild>
        <Pressable style={styles.link}>
          <Text style={styles.linkText}>Todos</Text>
        </Pressable>
      </Link>
      <Link href="/documents" asChild>
        <Pressable style={styles.link}>
          <Text style={styles.linkText}>Documents</Text>
        </Pressable>
      </Link>
      <Link href="/messages" asChild>
        <Pressable style={styles.link}>
          <Text style={styles.linkText}>Messages</Text>
        </Pressable>
      </Link>
      <Link href="/chat" asChild>
        <Pressable style={styles.link}>
          <Text style={styles.linkText}>AI chat</Text>
        </Pressable>
      </Link>
      <Pressable style={styles.logout} onPress={() => void logout()}>
        <Text style={styles.logoutText}>Sign out</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12 },
  greeting: { fontSize: 18, marginBottom: 8 },
  link: {
    backgroundColor: '#1a1a2e',
    padding: 16,
    borderRadius: 8,
  },
  linkText: { color: '#fff', fontSize: 16, fontWeight: '600', textAlign: 'center' },
  logout: { marginTop: 24, padding: 14, alignItems: 'center' },
  logoutText: { color: '#c00', fontSize: 16 },
});
