import * as FileSystem from 'expo-file-system';

const POSITIONS_FILE = 'positions.json';

function cacheDir(): string {
  const base = FileSystem.documentDirectory;
  if (!base) {
    throw new Error('documentDirectory is not available');
  }
  return `${base}ebooks/`;
}

async function positionsFilePath(): Promise<string> {
  const dir = cacheDir();
  const info = await FileSystem.getInfoAsync(dir);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(dir, { intermediates: true });
  }
  return `${dir}${POSITIONS_FILE}`;
}

type Row = { percentage: number; cfi: string | null; updated_at: string };

function clampUnitPct(n: unknown): number | null {
  if (typeof n !== 'number' || !Number.isFinite(n)) return null;
  let x = n;
  if (x > 1) x /= 100;
  return Math.max(0, Math.min(1, x));
}

async function readAll(): Promise<Record<string, Row>> {
  try {
    const p = await positionsFilePath();
    const info = await FileSystem.getInfoAsync(p);
    if (!info.exists) return {};
    const raw = await FileSystem.readAsStringAsync(p);
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, Row>;
    }
  } catch {
    /* ignore */
  }
  return {};
}

async function writeAll(data: Record<string, Row>): Promise<void> {
  try {
    await FileSystem.writeAsStringAsync(await positionsFilePath(), JSON.stringify(data), {
      encoding: FileSystem.EncodingType.UTF8,
    });
  } catch {
    /* ignore */
  }
}

export type LocalEbookPosition = {
  percentage: number;
  cfi: string | null;
  updated_at?: string;
};

export async function loadLocalEbookPosition(digest: string): Promise<LocalEbookPosition | null> {
  if (!digest) return null;
  const all = await readAll();
  const row = all[digest];
  if (!row || typeof row !== 'object') return null;
  const percentage = clampUnitPct(row.percentage);
  if (percentage === null) return null;
  const cfi = typeof row.cfi === 'string' && row.cfi.startsWith('epubcfi(') ? row.cfi : null;
  return { percentage, cfi, updated_at: row.updated_at };
}

export async function saveLocalEbookPosition(
  digest: string,
  percentage: number,
  cfi: string | null
): Promise<void> {
  if (!digest) return;
  const p = clampUnitPct(percentage);
  if (p === null) return;
  const all = await readAll();
  all[digest] = {
    percentage: p,
    cfi: cfi && cfi.startsWith('epubcfi(') ? cfi : null,
    updated_at: new Date().toISOString(),
  };
  await writeAll(all);
}
