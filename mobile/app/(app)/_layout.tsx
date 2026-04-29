import { Ionicons } from '@expo/vector-icons';
import { Redirect, Tabs } from 'expo-router';
import { Platform, StyleSheet, View } from 'react-native';
import { useAuth } from '../../src/context/AuthContext';
import { VoiceFab } from './voice-fab';

export default function AppLayout() {
  const { token, isReady, apiConfigured } = useAuth();

  if (!isReady) {
    return null;
  }
  if (!apiConfigured || !token) {
    return <Redirect href="/(auth)/login" />;
  }

  return (
    <View style={styles.root}>
      <Tabs
        screenOptions={{
          tabBarActiveTintColor: '#1a1a2e',
          tabBarInactiveTintColor: '#888',
          headerShown: true,
          tabBarLabelStyle: styles.tabLabel,
          tabBarStyle: styles.tabBar,
        }}
      >
        <Tabs.Screen
          name="todos"
          options={{
            title: 'ToDos',
            tabBarLabel: 'ToDos',
            tabBarIcon: ({ color, size }) => <Ionicons name="checkbox-outline" size={size} color={color} />,
          }}
        />
        <Tabs.Screen
          name="documents"
          options={{
            title: 'Documents',
            tabBarLabel: 'Docs',
            tabBarIcon: ({ color, size }) => <Ionicons name="document-text-outline" size={size} color={color} />,
          }}
        />
        <Tabs.Screen
          name="messages"
          options={{
            title: 'Messages',
            tabBarLabel: 'Messages',
            tabBarIcon: ({ color, size }) => <Ionicons name="chatbubbles-outline" size={size} color={color} />,
          }}
        />
        <Tabs.Screen
          name="chat"
          options={{
            title: 'Bastion Chat',
            tabBarLabel: 'Chat',
            tabBarIcon: ({ color, size }) => <Ionicons name="sparkles-outline" size={size} color={color} />,
          }}
        />
        <Tabs.Screen
          name="home"
          options={{
            title: 'Profile',
            tabBarLabel: 'Profile',
            tabBarIcon: ({ color, size }) => <Ionicons name="person-circle-outline" size={size} color={color} />,
          }}
        />
      </Tabs>
      <VoiceFab />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  /** Do not set minHeight or extra paddingBottom here — bottom-tabs already applies safe-area insets; duplicating them pushes labels/icons down. */
  tabBar: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#e0e0e0',
    ...(Platform.OS === 'android' ? { elevation: 8 } : {}),
  },
  tabLabel: {
    fontSize: 11,
  },
});
