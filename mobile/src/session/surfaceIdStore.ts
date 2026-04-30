import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_mobile_surface_id';

export async function getOrCreateMobileSurfaceId(): Promise<string> {
  let existing = await SecureStore.getItemAsync(KEY);
  if (existing && existing.length >= 8) {
    return existing;
  }
  const id =
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : `surf-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  await SecureStore.setItemAsync(KEY, id);
  return id;
}
