import { Ionicons } from '@expo/vector-icons';
import { useMemo } from 'react';
import { Pressable, StyleSheet, Text, View, useColorScheme } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { DOCK_CONTENT_HEIGHT } from '../constants/dock';
import { useAppLauncher } from '../context/AppLauncherContext';
import { useActiveRoute } from '../hooks/useActiveRoute';
import { getColors } from '../theme/colors';
import { useVoiceModal } from '../voice/VoiceModalContext';

export function BottomDock() {
  const insets = useSafeAreaInsets();
  const { openLauncher } = useAppLauncher();
  const { openVoice } = useVoiceModal();
  const { label, icon } = useActiveRoute();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);

  return (
    <View
      style={[
        styles.wrapper,
        {
          paddingBottom: insets.bottom,
          minHeight: DOCK_CONTENT_HEIGHT + insets.bottom,
          borderTopColor: c.border,
          backgroundColor: c.surface,
        },
      ]}
      pointerEvents="box-none"
    >
      <View style={styles.row}>
        <Pressable
          style={[styles.leftPill, { backgroundColor: c.chipBg }]}
          onPress={openLauncher}
          accessibilityRole="button"
          accessibilityLabel={`Open app launcher. Current section: ${label}`}
        >
          <Ionicons name={icon as keyof typeof Ionicons.glyphMap} size={22} color={c.text} />
          <Text style={[styles.label, { color: c.text }]} numberOfLines={1}>
            {label}
          </Text>
        </Pressable>
        <Pressable
          style={[styles.micBtn, { backgroundColor: c.chipBgActive }]}
          onPress={openVoice}
          accessibilityRole="button"
          accessibilityLabel="Record voice note"
          accessibilityHint="Opens voice recording. Speak, then tap Stop to send to Bastion."
        >
          <Ionicons name="mic" size={24} color={c.chipTextActive} />
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingHorizontal: 14,
    minHeight: DOCK_CONTENT_HEIGHT,
  },
  leftPill: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    minWidth: 0,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  micBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.2,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 2 },
    elevation: 3,
  },
});
