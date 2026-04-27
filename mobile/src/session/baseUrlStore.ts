import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_api_base_url';

export async function getStoredBaseUrl(): Promise<string | null> {
  try {
    const v = await SecureStore.getItemAsync(KEY);
    return v?.trim() || null;
  } catch {
    return null;
  }
}

export async function setStoredBaseUrl(url: string): Promise<void> {
  await SecureStore.setItemAsync(KEY, url.trim());
}

export async function clearStoredBaseUrl(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    /* ignore */
  }
}
