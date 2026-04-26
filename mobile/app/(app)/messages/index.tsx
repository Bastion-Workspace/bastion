import { useCallback, useEffect, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { useRouter } from 'expo-router';
import { getUserRooms, type Room } from '../../../src/api/messaging';

export default function MessagesListScreen() {
  const router = useRouter();
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const load = useCallback(async () => {
    const list = await getUserRooms(50);
    setRooms(list);
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        await load();
      } finally {
        setLoading(false);
      }
    })();
  }, [load]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await load();
    } finally {
      setRefreshing(false);
    }
  }, [load]);

  if (loading && rooms.length === 0) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <FlatList
      data={rooms}
      keyExtractor={(item) => item.room_id}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      contentContainerStyle={styles.list}
      ListEmptyComponent={<Text style={styles.empty}>No rooms yet.</Text>}
      renderItem={({ item }) => (
        <Pressable
          style={styles.row}
          onPress={() => router.push(`/messages/${item.room_id}`)}
        >
          <Text style={styles.title}>{item.room_name || item.name || 'Room'}</Text>
          {item.last_message_at ? (
            <Text style={styles.sub}>{item.last_message_at}</Text>
          ) : null}
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16 },
  empty: { textAlign: 'center', marginTop: 48, color: '#666' },
  row: {
    backgroundColor: '#fff',
    padding: 14,
    borderRadius: 8,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  title: { fontSize: 16, fontWeight: '600' },
  sub: { fontSize: 12, color: '#666', marginTop: 4 },
});
