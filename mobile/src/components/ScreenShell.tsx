import type { ReactNode } from 'react';
import { StyleSheet } from 'react-native';
import { SafeAreaView, useSafeAreaInsets, type Edge } from 'react-native-safe-area-context';

type ScreenShellProps = {
  children: ReactNode;
  edges?: Edge[];
};

/**
 * Wraps headerless screens so content does not draw under the status bar,
 * notch, Dynamic Island, or camera cutout. Bottom inset is usually handled
 * separately (e.g. BottomDock).
 */
export function ScreenShell({ children, edges = ['top'] }: ScreenShellProps) {
  return (
    <SafeAreaView edges={edges} style={styles.fill}>
      {children}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1 },
});

/** Bottom padding for slide-up sheets so actions clear the home indicator. */
export function useModalSheetBottomPadding(basePadding = 16): number {
  const insets = useSafeAreaInsets();
  return basePadding + insets.bottom;
}
