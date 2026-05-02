import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  Dimensions,
  FlatList,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
  useColorScheme,
} from 'react-native';
import { filterActiveConfiguredSources, getMediaSources } from '../api/media';
import { getColors } from '../theme/colors';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useAppLauncher } from '../context/AppLauncherContext';

type LauncherItem = {
  key: string;
  label: string;
  href: `/(app)/${string}`;
  icon: keyof typeof Ionicons.glyphMap;
};

const SECTIONS: LauncherItem[] = [
  { key: 'chat', label: 'Chat', href: '/(app)/chat', icon: 'sparkles-outline' },
  { key: 'messages', label: 'Messages', href: '/(app)/messages', icon: 'chatbubbles-outline' },
  { key: 'documents', label: 'Docs', href: '/(app)/documents', icon: 'document-text-outline' },
  { key: 'todos', label: 'ToDos', href: '/(app)/todos', icon: 'checkbox-outline' },
  { key: 'rss', label: 'RSS', href: '/(app)/rss', icon: 'newspaper-outline' },
  { key: 'media', label: 'Media', href: '/(app)/media', icon: 'musical-notes-outline' },
  { key: 'ebooks', label: 'eBooks', href: '/(app)/ebooks', icon: 'book-outline' },
  { key: 'home', label: 'Settings', href: '/(app)/home', icon: 'settings-outline' },
];

const SHEET_MAX = Math.min(Dimensions.get('window').height * 0.55, 440);

export function AppLauncherSheet() {
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);
  const { open, closeLauncher } = useAppLauncher();
  const slide = useRef(new Animated.Value(SHEET_MAX)).current;
  const [sheetVisible, setSheetVisible] = useState(false);
  const wasOpenRef = useRef(false);
  /** Default true: show Media until a successful fetch proves there are no sources (plan). On API error, stay true. */
  const [showMediaInLauncher, setShowMediaInLauncher] = useState(true);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void (async () => {
      try {
        const src = await getMediaSources();
        if (cancelled) return;
        const filtered = filterActiveConfiguredSources(src.sources);
        setShowMediaInLauncher(filtered.length > 0);
      } catch {
        if (!cancelled) setShowMediaInLauncher(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  const launcherSections = useMemo(
    () => SECTIONS.filter((item) => item.key !== 'media' || showMediaInLauncher),
    [showMediaInLauncher]
  );

  useEffect(() => {
    if (open) {
      wasOpenRef.current = true;
      setSheetVisible(true);
      slide.setValue(SHEET_MAX);
      Animated.spring(slide, {
        toValue: 0,
        useNativeDriver: false,
        friction: 9,
        tension: 65,
      }).start();
    } else if (wasOpenRef.current) {
      Animated.timing(slide, {
        toValue: SHEET_MAX,
        duration: 220,
        useNativeDriver: false,
      }).start(({ finished }) => {
        if (finished) {
          setSheetVisible(false);
          wasOpenRef.current = false;
        }
      });
    }
  }, [open, slide]);

  const onNavigate = useCallback(
    (href: LauncherItem['href']) => {
      closeLauncher();
      router.push(href);
    },
    [closeLauncher, router]
  );

  const renderItem = useCallback(
    ({ item }: { item: LauncherItem }) => (
      <Pressable
        style={[styles.tile, { backgroundColor: c.surface }]}
        onPress={() => onNavigate(item.href)}
        accessibilityRole="button"
        accessibilityLabel={item.label}
      >
        <View style={[styles.tileIconWrap, { backgroundColor: c.chipBg }]}>
          <Ionicons name={item.icon} size={28} color={c.text} />
        </View>
        <Text style={[styles.tileLabel, { color: c.text }]}>{item.label}</Text>
      </Pressable>
    ),
    [onNavigate, c]
  );

  return (
    <Modal visible={sheetVisible} transparent animationType="none" onRequestClose={closeLauncher}>
      <View style={styles.modalRoot}>
        <Pressable style={styles.backdrop} onPress={closeLauncher} accessibilityLabel="Close launcher" />
        <Animated.View
          style={[
            styles.sheet,
            {
              backgroundColor: c.background,
              paddingBottom: Math.max(insets.bottom, 12),
              height: SHEET_MAX,
              transform: [{ translateY: slide }],
            },
          ]}
        >
          <Pressable style={styles.handleHit} onPress={closeLauncher} accessibilityLabel="Close launcher">
            <View style={[styles.handle, { backgroundColor: c.border }]} />
          </Pressable>
          <Text style={[styles.sheetTitle, { color: c.textSecondary }]}>Go to</Text>
          <FlatList
            style={styles.list}
            data={launcherSections}
            keyExtractor={(i) => i.key}
            numColumns={2}
            columnWrapperStyle={styles.rowWrap}
            renderItem={renderItem}
            initialNumToRender={8}
          />
        </Animated.View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  modalRoot: {
    flex: 1,
    justifyContent: 'flex-end',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.4)',
  },
  sheet: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    paddingHorizontal: 16,
    paddingTop: 8,
  },
  handleHit: {
    alignItems: 'center',
    paddingVertical: 10,
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
  },
  sheetTitle: {
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 12,
    textAlign: 'center',
  },
  list: {
    flex: 1,
  },
  rowWrap: {
    gap: 12,
    marginBottom: 12,
  },
  tile: {
    flex: 1,
    minHeight: 88,
    borderRadius: 12,
    padding: 12,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  tileIconWrap: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tileLabel: {
    fontSize: 14,
    fontWeight: '600',
  },
});
