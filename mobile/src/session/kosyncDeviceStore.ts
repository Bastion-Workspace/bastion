import * as SecureStore from 'expo-secure-store';

const KEY = 'kosync_device_id';

export async function getOrCreateKosyncDeviceId(): Promise<string> {
  let id = await SecureStore.getItemAsync(KEY);
  if (!id) {
    id = `mobile-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    await SecureStore.setItemAsync(KEY, id);
  }
  return id;
}
