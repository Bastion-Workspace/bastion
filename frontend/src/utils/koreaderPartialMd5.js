/**
 * KOReader-compatible partial file digest (matches util.partialMD5 in KOReader).
 *
 * Parity with backend `koreader_partial_md5` / KOReader: 10240 bytes where data[i] = i % 256
 * must yield digest `a52dff8366b1473d2e13edd2415def67`.
 */
import SparkMD5 from 'spark-md5';

/** Golden digest for `buildKoreaderPartialMd5GoldenArrayBuffer()` (algorithm regression check). */
export const KOREADER_PARTIAL_MD5_REF_HEX = 'a52dff8366b1473d2e13edd2415def67';

/** Deterministic 10KiB buffer used to verify the implementation matches KOReader / server. */
export function buildKoreaderPartialMd5GoldenArrayBuffer() {
  const u = new Uint8Array(10240);
  for (let i = 0; i < u.length; i += 1) {
    u[i] = i % 256;
  }
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength);
}

/**
 * @param {ArrayBuffer} arrayBuffer
 * @returns {string} 32-char hex MD5
 */
export function koreaderPartialMd5(arrayBuffer) {
  const spark = new SparkMD5.ArrayBuffer();
  const step = 1024;
  const size = 1024;
  const len = arrayBuffer.byteLength;
  for (let i = -1; i <= 10; i += 1) {
    const shift = 2 * i;
    const offset = shift >= 0 ? step << shift : step >> -shift;
    if (offset < 0 || offset >= len) {
      break;
    }
    const end = Math.min(offset + size, len);
    spark.append(arrayBuffer.slice(offset, end));
    if (end <= offset) {
      break;
    }
  }
  return spark.end();
}
