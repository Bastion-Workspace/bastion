import { assertApiBaseUrl } from './config';
import { getStoredToken } from '../session/tokenStore';

type TranscribeErrorBody = {
  detail?: string | Array<{ msg?: string }>;
};

function formatHttpError(status: number, body: TranscribeErrorBody | null): string {
  const d = body?.detail;
  if (typeof d === 'string' && d.trim()) return d.trim();
  if (Array.isArray(d) && d.length > 0) {
    const parts = d.map((x) => (typeof x?.msg === 'string' ? x.msg : '')).filter(Boolean);
    if (parts.length) return parts.join('; ');
  }
  return `Transcription failed (${status})`;
}

/**
 * Upload a local audio file to POST /api/audio/transcribe and return plain text.
 */
export async function transcribeAudio(uri: string): Promise<string> {
  const base = assertApiBaseUrl();
  const token = await getStoredToken();
  if (!token) {
    throw new Error('Not authenticated');
  }

  const form = new FormData();
  form.append('file', {
    uri,
    name: 'recording.m4a',
    type: 'audio/m4a',
  } as unknown as Blob);

  const res = await fetch(`${base}/api/audio/transcribe`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: form,
  });

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON */
  }

  if (!res.ok) {
    throw new Error(formatHttpError(res.status, body as TranscribeErrorBody | null));
  }

  const data = body as { success?: boolean; text?: string } | null;
  if (!data?.success || typeof data.text !== 'string') {
    throw new Error('Transcription failed: invalid response');
  }
  return data.text;
}
