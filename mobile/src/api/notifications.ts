import * as Notifications from 'expo-notifications';
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import { assertApiBaseUrl } from './config';
import { getStoredToken } from '../session/tokenStore';
import { getOrCreateMobileSurfaceId } from '../session/surfaceIdStore';
import { registerPushToken, revokePushToken } from './notificationsPush';

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: false,
    shouldSetBadge: true,
  }),
});

export async function ensurePushPermissions(): Promise<boolean> {
  const { status: existing } = await Notifications.getPermissionsAsync();
  if (existing === 'granted') return true;
  const { status } = await Notifications.requestPermissionsAsync();
  return status === 'granted';
}

export async function registerDevicePushWithServer(): Promise<void> {
  try {
    const ok = await ensurePushPermissions();
    if (!ok) return;
    const extra = Constants.expoConfig?.extra as
      | { eas?: { projectId?: string }; easProjectId?: string }
      | undefined;
    const projectId =
      extra?.eas?.projectId || extra?.easProjectId || process.env.EAS_PROJECT_ID;
    const expo = await Notifications.getExpoPushTokenAsync(
      projectId ? { projectId: String(projectId) } : undefined
    );
    const token = expo.data;
    const base = assertApiBaseUrl();
    const auth = await getStoredToken();
    if (!auth) return;
    const deviceId = await getOrCreateMobileSurfaceId();
    await registerPushToken(base, auth, {
      token,
      platform: Platform.OS === 'ios' ? 'ios' : 'android',
      device_id: deviceId,
    });
  } catch (e) {
    console.warn('Push registration skipped:', e);
  }
}

export async function revokeDevicePushOnServer(): Promise<void> {
  try {
    const base = assertApiBaseUrl();
    const auth = await getStoredToken();
    if (!auth) return;
    const deviceId = await getOrCreateMobileSurfaceId();
    await revokePushToken(base, auth, deviceId);
  } catch {
    /* ignore */
  }
}
