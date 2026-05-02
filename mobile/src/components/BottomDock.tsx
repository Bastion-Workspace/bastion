import { Ionicons } from '@expo/vector-icons';
import { useMemo } from 'react';
import { Pressable, StyleSheet, Text, View, useColorScheme } from 'react-native';
import { useSegments } from 'expo-router';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { DOCK_CONTENT_HEIGHT } from '../constants/dock';
import { useAppLauncher } from '../context/AppLauncherContext';
import { useActiveRoute } from '../hooks/useActiveRoute';
import { getColors } from '../theme/colors';
import { useVoiceModal } from '../voice/VoiceModalContext';

export function BottomDock() {
  const segments = useSegments();
  const insets = useSafeAreaInsets();
  const { openLauncher } = useAppLauncher();
  const { openVoice } = useVoiceModal();
  const { label, icon } = useActiveRoute();
  const scheme = useColorScheme() === 'dark' ? 'dark' : 'light';
  const c = useMemo(() => getColors(scheme), [scheme]);

  const panePillStyle = useMemo(
    () => ({
      backgroundColor: scheme === 'dark' ? c.surfaceMuted : c.chipBg,
      borderColor: scheme === 'dark' ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.1)',
      shadowColor: '#000',
      shadowOpacity: scheme === 'dark' ? 0.42 : 0.16,
      shadowRadius: scheme === 'dark' ? 8 : 7,
      shadowOffset: { width: 0, height: 2 } as const,
      elevation: scheme === 'dark' ? 6 : 5,
    }),
    [c.chipBg, c.surfaceMuted, scheme]
  );

  if (segments.includes('video')) {
    return null;
  }

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
          style={[styles.leftPill, panePillStyle]}
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
    justifyContent: 'space-between',
    gap: 10,
    paddingHorizontal: 14,
    minHeight: DOCK_CONTENT_HEIGHT,
  },
  leftPill: {
    flexDirection: 'row',
    alignItems: 'center',
    maxWidth: '78%',
    gap: 10,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 20,
    borderWidth: StyleSheet.hairlineWidth * 2,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
    flexShrink: 1,
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
