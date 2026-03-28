/**
 * Helpers for streaming TTS (MediaSource MIME selection, long-text chunking).
 */

/** Minimum text length before sentence chunking is used. */
export const TTS_CHUNK_THRESHOLD_CHARS = 500;

/** Target max characters per TTS chunk (merged from sentences). */
export const TTS_CHUNK_MAX_CHARS = 900;

/** Above this chunk count, audio export uses backend Celery job instead of browser synthesis. */
export const AUDIO_EXPORT_BACKEND_THRESHOLD_CHUNKS = 50;

/**
 * Map X-Audio-Format / stream label to MediaSource SourceBuffer MIME type.
 * @param {string} fmt
 * @returns {string|null}
 */
export function mimeForStreamFormat(fmt) {
  const f = String(fmt || 'ogg').toLowerCase();
  if (f === 'mp3') {
    return 'audio/mpeg';
  }
  if (f === 'ogg' || f === 'opus') {
    return 'audio/ogg; codecs=opus';
  }
  return null;
}

/**
 * Split long text into sentence-ish chunks for faster time-to-first-audio.
 * @param {string} text
 * @returns {string[]}
 */
export function splitTextForTts(text, threshold = TTS_CHUNK_THRESHOLD_CHARS, maxChunk = TTS_CHUNK_MAX_CHARS) {
  const t = String(text || '').trim();
  if (!t || t.length <= threshold) {
    return [t];
  }

  const sentences = [];
  const re = /[^.!?\n]+(?:[.!?]+|\n+)/g;
  let m;
  while ((m = re.exec(t)) !== null) {
    const s = m[0].trim();
    if (s) sentences.push(s);
  }

  if (sentences.length === 0) {
    const out = [];
    for (let i = 0; i < t.length; i += maxChunk) {
      out.push(t.slice(i, i + maxChunk).trim());
    }
    return out.filter(Boolean);
  }

  const merged = [];
  let acc = '';
  for (const s of sentences) {
    const next = acc ? `${acc} ${s}` : s;
    if (next.length <= maxChunk) {
      acc = next;
    } else {
      if (acc) merged.push(acc.trim());
      let remaining = s;
      while (remaining.length > maxChunk) {
        merged.push(remaining.slice(0, maxChunk).trim());
        remaining = remaining.slice(maxChunk).trim();
      }
      acc = remaining;
    }
  }
  if (acc) merged.push(acc.trim());
  return merged.length ? merged : [t];
}
