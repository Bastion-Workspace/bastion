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
import dayjs from 'dayjs';
import { useRouter } from 'expo-router';
import { getUserRooms, type Room, type RoomParticipant } from '../../../src/api/messaging';
import { useAuth } from '../../../src/context/AuthContext';

function roomTitle(room: Room): string {
  const d = room.display_name?.trim();
  if (d) return d;
  const n = room.room_name?.trim() || room.name?.trim();
  if (n) return n;
  return 'Chat';
}

function participantSubtitle(room: Room, myUserId: string | undefined): string | null {
  const parts = room.participants;
  if (!parts?.length) return null;
  const names = parts.map((p: RoomParticipant) => {
    if (myUserId && p.user_id === myUserId) return 'You';
    return (p.display_name || p.username || 'Member').trim();
  });
  const unique = [...new Set(names)];
  return unique.join(', ');
}

export default function MessagesListScreen() {
  const router = useRouter();
  const { user } = useAuth();
  const myUserId = typeof user?.user_id === 'string' ? user.user_id : undefined;
  const [rooms, setRooms] = useState<Room[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const list = await getUserRooms(50);
      setRooms(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load rooms');
      setRooms([]);
    }
  }, []);

  useEffect(() => {
    void (async () => {
      await load();
      setLoading(false);
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
      ListHeaderComponent={
        error ? (
          <Text style={styles.errorBanner} accessibilityRole="alert">
            {error}
          </Text>
        ) : null
      }
      ListEmptyComponent={<Text style={styles.empty}>No rooms yet.</Text>}
      renderItem={({ item }) => {
        const sub = participantSubtitle(item, myUserId);
        return (
          <Pressable
            style={styles.row}
            onPress={() => router.push(`/messages/${item.room_id}`)}
          >
            <Text style={styles.title}>{roomTitle(item)}</Text>
            {sub ? <Text style={styles.participants}>{sub}</Text> : null}
            {item.last_message_at ? (
              <Text style={styles.sub}>{dayjs(item.last_message_at).fromNow()}</Text>
            ) : null}
          </Pressable>
        );
      }}
    />
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  list: { padding: 16 },
  errorBanner: {
    backgroundColor: '#fee',
    color: '#a00',
    padding: 12,
    borderRadius: 8,
    marginBottom: 12,
    fontSize: 14,
  },
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
  participants: { fontSize: 13, color: '#555', marginTop: 2 },
  sub: { fontSize: 12, color: '#666', marginTop: 4 },
});
