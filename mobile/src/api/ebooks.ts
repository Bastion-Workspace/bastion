import { apiFetchRaw, apiRequest } from './client';

export type OpdsCatalogEntryResponse = {
  id: string;
  title: string;
  root_url: string;
  verify_ssl: boolean;
  http_basic_configured: boolean;
};

export type OpdsCatalogEntryInput = {
  id: string;
  title: string;
  root_url: string;
  verify_ssl?: boolean;
  http_basic_b64?: string | null;
};

export type EbooksSettingsResponse = {
  catalogs: OpdsCatalogEntryResponse[];
  reader_prefs: Record<string, unknown>;
  recently_opened: RecentlyOpenedEntry[];
  kosync: {
    configured: boolean;
    base_url: string;
    username: string;
    verify_ssl: boolean;
  };
};

export type RecentlyOpenedEntry = {
  digest?: string;
  title?: string;
  catalog_id?: string;
  acquisition_url?: string;
  opened_at?: string;
  acquisition_format?: 'epub' | 'pdf';
};

export type EbooksSettingsUpdate = {
  catalogs?: OpdsCatalogEntryInput[];
  reader_prefs?: Record<string, unknown>;
  recently_opened?: RecentlyOpenedEntry[];
};

export type OpdsFetchAtomBody = {
  catalog_id: string;
  url: string;
  want?: 'atom';
};

export type OpdsFetchBinaryBody = {
  catalog_id: string;
  url: string;
  want: 'binary';
};

export type OpdsLink = {
  href?: string;
  rel?: string;
  type?: string;
};

export type OpdsFeedEntry = {
  id?: string;
  title?: string;
  acquisition_href?: string;
  acquisition_type?: string;
  navigation_links?: OpdsLink[];
  links?: OpdsLink[];
};

export type OpdsFeed = {
  feed_title?: string;
  feed_id?: string;
  entries?: OpdsFeedEntry[];
  search_template?: string;
  fetched_url?: string;
};

export type OpdsFetchAtomResponse = {
  feed?: OpdsFeed;
  fetched_url?: string;
};

export type KosyncSettingsBody = {
  base_url: string;
  username: string;
  password?: string | null;
  verify_ssl?: boolean;
};

export type KosyncTestBody = {
  base_url: string;
  username: string;
  password: string;
  verify_ssl?: boolean;
};

export type KosyncRegisterBody = {
  username: string;
  password: string;
  base_url?: string | null;
  verify_ssl?: boolean;
};

export type KosyncProgressPutBody = {
  document: string;
  progress: string;
  percentage: number;
  device?: string;
  device_id?: string;
};

export type KosyncProgressRemote = {
  percentage?: number;
  progress?: string;
  device?: string;
  [key: string]: unknown;
};

export async function getEbooksSettings(): Promise<EbooksSettingsResponse> {
  return apiRequest<EbooksSettingsResponse>('/api/ebooks/settings');
}

export async function putEbooksSettings(body: EbooksSettingsUpdate): Promise<EbooksSettingsResponse> {
  return apiRequest<EbooksSettingsResponse>('/api/ebooks/settings', {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

export async function putKosyncSettings(body: KosyncSettingsBody): Promise<EbooksSettingsResponse> {
  return apiRequest<EbooksSettingsResponse>('/api/ebooks/kosync/settings', {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}

export async function fetchOpdsAtom(body: OpdsFetchAtomBody): Promise<OpdsFetchAtomResponse> {
  return apiRequest<OpdsFetchAtomResponse>('/api/ebooks/opds/fetch', {
    method: 'POST',
    body: JSON.stringify({ ...body, want: body.want ?? 'atom' }),
  });
}

export async function fetchOpdsBinary(body: OpdsFetchBinaryBody): Promise<ArrayBuffer> {
  const res = await apiFetchRaw('/api/ebooks/opds/fetch', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let bodyText: unknown;
    try {
      bodyText = await res.json();
    } catch {
      bodyText = await res.text();
    }
    const err = new Error(`HTTP ${res.status}`) as Error & { status?: number; body?: unknown };
    err.status = res.status;
    err.body = bodyText;
    throw err;
  }
  return res.arrayBuffer();
}

export async function kosyncTest(body: KosyncTestBody): Promise<{ ok: boolean; status: number; body: unknown }> {
  return apiRequest('/api/ebooks/kosync/test', { method: 'POST', body: JSON.stringify(body) });
}

export async function kosyncHealth(): Promise<{ ok: boolean; detail: string }> {
  return apiRequest('/api/ebooks/kosync/health');
}

export async function kosyncRegister(body: KosyncRegisterBody): Promise<{ ok: boolean; username: string }> {
  return apiRequest('/api/ebooks/kosync/register', { method: 'POST', body: JSON.stringify(body) });
}

export async function getKosyncProgress(documentDigest: string): Promise<KosyncProgressRemote> {
  return apiRequest<KosyncProgressRemote>(
    `/api/ebooks/kosync/progress/${encodeURIComponent(documentDigest)}`
  );
}

export async function putKosyncProgress(body: KosyncProgressPutBody): Promise<unknown> {
  return apiRequest('/api/ebooks/kosync/progress', {
    method: 'PUT',
    body: JSON.stringify(body),
  });
}
