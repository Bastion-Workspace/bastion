const KEY = 'bastion_ebook_positions_v1';

function readAll() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
  } catch (_) {
    /* ignore */
  }
  return {};
}

function writeAll(data) {
  try {
    localStorage.setItem(KEY, JSON.stringify(data));
  } catch (_) {
    /* ignore */
  }
}

function clampUnitPct(n) {
  if (typeof n !== 'number' || !Number.isFinite(n)) return null;
  let x = n;
  if (x > 1) x /= 100;
  return Math.max(0, Math.min(1, x));
}

/**
 * @param {string} digest Koreader partial MD5 hex
 * @returns {{ percentage: number, cfi: string|null, updated_at?: string }|null}
 */
export function loadLocalEbookPosition(digest) {
  if (!digest || typeof digest !== 'string') return null;
  const row = readAll()[digest];
  if (!row || typeof row !== 'object') return null;
  const percentage = clampUnitPct(row.percentage);
  if (percentage === null) return null;
  const cfi = typeof row.cfi === 'string' && row.cfi.startsWith('epubcfi(') ? row.cfi : null;
  return {
    percentage,
    cfi,
    updated_at: typeof row.updated_at === 'string' ? row.updated_at : undefined,
  };
}

/**
 * @param {string} digest Koreader partial MD5 hex
 * @param {number} percentage 0..1
 * @param {string|null|undefined} cfi epub CFI or null
 */
export function saveLocalEbookPosition(digest, percentage, cfi) {
  if (!digest || typeof digest !== 'string') return;
  const p = clampUnitPct(percentage);
  if (p === null) return;
  const all = readAll();
  all[digest] = {
    percentage: p,
    cfi: cfi && String(cfi).startsWith('epubcfi(') ? String(cfi) : null,
    updated_at: new Date().toISOString(),
  };
  writeAll(all);
}
