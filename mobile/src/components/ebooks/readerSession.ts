import * as FileSystem from 'expo-file-system';
import { buildReaderHtmlFromUri } from './readerHtml';

function digestFromEpubUri(epubUri: string): string {
  const m = epubUri.match(/\/([^/]+)\.epub$/);
  if (!m) {
    throw new Error('Invalid EPUB URI (expected .../<digest>.epub)');
  }
  return m[1];
}

function epubDirFromUri(epubUri: string): string {
  const idx = epubUri.lastIndexOf('/');
  if (idx < 0) {
    throw new Error('Invalid EPUB URI');
  }
  return epubUri.slice(0, idx + 1);
}

/**
 * Writes a small self-contained reader HTML next to the EPUB and returns its `file://` URI for WebView `source={{ uri }}`.
 * The EPUB is loaded inside the WebView via `fetch()` relative to that HTML file (same directory as the `.epub`).
 */
export async function prepareReaderSession(
  epubUri: string
): Promise<{ sourceUri: string; allowingReadAccessToURL: string }> {
  const digest = digestFromEpubUri(epubUri);
  const dir = epubDirFromUri(epubUri);
  const htmlPath = `${dir}ebook_reader_${digest}.html`;
  const relativeEpub = `./${digest}.epub`;
  const html = buildReaderHtmlFromUri(relativeEpub);
  await FileSystem.writeAsStringAsync(htmlPath, html, { encoding: FileSystem.EncodingType.UTF8 });
  return { sourceUri: htmlPath, allowingReadAccessToURL: dir };
}
