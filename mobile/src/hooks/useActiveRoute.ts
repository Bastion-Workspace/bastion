import { useMemo } from 'react';
import { useSegments } from 'expo-router';

export type ActiveRouteInfo = {
  label: string;
  /** Ionicons glyph name */
  icon: string;
};

/**
 * Derives a short label + icon for the bottom dock from the current Expo Router segments.
 */
export function useActiveRoute(): ActiveRouteInfo {
  const segments = useSegments();
  return useMemo(() => {
    const parts = segments.filter((s) => s !== '(app)' && s !== '(auth)');
    if (parts.length === 0) {
      return { label: 'Bastion', icon: 'apps-outline' };
    }
    const root = parts[0];
    if (root === 'chat') {
      return { label: 'Chat', icon: 'sparkles-outline' };
    }
    if (root === 'messages') {
      return { label: 'Messages', icon: 'chatbubbles-outline' };
    }
    if (root === 'documents') {
      return { label: 'Docs', icon: 'document-text-outline' };
    }
    if (root === 'todos') {
      return { label: 'ToDos', icon: 'checkbox-outline' };
    }
    if (root === 'rss') {
      return { label: 'RSS', icon: 'newspaper-outline' };
    }
    if (root === 'media') {
      return { label: 'Media', icon: 'musical-notes-outline' };
    }
    if (root === 'ebooks') {
      return { label: 'eBooks', icon: 'book-outline' };
    }
    if (root === 'home') {
      return { label: 'Settings', icon: 'settings-outline' };
    }
    if (root === 'shortcut-send' || root === 'voice') {
      return { label: 'Bastion', icon: 'apps-outline' };
    }
    return { label: 'Bastion', icon: 'apps-outline' };
  }, [segments]);
}
