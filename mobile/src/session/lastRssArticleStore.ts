import * as SecureStore from 'expo-secure-store';

const KEY = 'bastion_last_rss_article_id_v1';

export async function saveLastOpenRssArticleId(articleId: string): Promise<void> {
  if (!articleId || articleId.length > 256) return;
  try {
    await SecureStore.setItemAsync(KEY, articleId);
  } catch {
    /* ignore */
  }
}

export async function loadLastOpenRssArticleId(): Promise<string | null> {
  try {
    const v = await SecureStore.getItemAsync(KEY);
    if (v && v.length > 0 && v.length < 256) return v;
  } catch {
    /* ignore */
  }
  return null;
}

export async function clearLastOpenRssArticleId(): Promise<void> {
  try {
    await SecureStore.deleteItemAsync(KEY);
  } catch {
    /* ignore */
  }
}
