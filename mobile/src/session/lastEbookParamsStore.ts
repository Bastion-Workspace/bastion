import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_last_ebook_reader_params_v1';

export type LastEbookParams = {
  catalogId: string;
  acquisitionUrl: string;
  title?: string;
  digest?: string;
  format?: string;
};

function isValid(p: LastEbookParams | null): p is LastEbookParams {
  return Boolean(
    p &&
      typeof p.catalogId === 'string' &&
      p.catalogId.length > 0 &&
      p.catalogId.length < 512 &&
      typeof p.acquisitionUrl === 'string' &&
      p.acquisitionUrl.length > 0 &&
      p.acquisitionUrl.length < 4096
  );
}

export async function saveLastEbookParams(params: LastEbookParams): Promise<void> {
  if (!isValid(params)) return;
  try {
    await SecureStore.setItemAsync(KEY, JSON.stringify(params));
  } catch {
    /* ignore */
  }
}

export async function loadLastEbookParams(): Promise<LastEbookParams | null> {
  try {
    const raw = await SecureStore.getItemAsync(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as LastEbookParams;
    return isValid(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export async function clearLastEbookParams(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    /* ignore */
  }
}
