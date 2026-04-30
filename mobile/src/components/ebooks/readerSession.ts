import { buildReaderHtml } from './readerHtml';

/**
 * Builds self-contained reader HTML (embedded EPUB base64) for WebView `source={{ html }}`.
 */
export function prepareReaderSession(epubBytes: ArrayBuffer): { sourceHtml: string } {
  const b64 = arrayBufferToBase64(epubBytes);
  const html = buildReaderHtml(b64);
  return { sourceHtml: html };
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode.apply(null, Array.from(sub) as unknown as number[]);
  }
  return btoa(binary);
}
