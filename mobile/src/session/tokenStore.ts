import * as SecureStore from 'expo-secure-store';

const KEY = 'auth_token';

export async function getStoredToken(): Promise<string | null> {
  try {
    return await SecureStore.getItemAsync(KEY);
  } catch {
    return null;
  }
}

export async function setStoredToken(token: string): Promise<void> {
  await SecureStore.setItemAsync(KEY, token);
}

export async function clearStoredToken(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    /* ignore */
  }
}
