export const BASTION_REF_KEY = '_bastion_ref';

export function parseRefFromCell(value) {
  if (value == null) return null;
  let v = value;
  if (typeof v === 'string') {
    const s = v.trim();
    if (!s) return null;
    try {
      v = JSON.parse(s);
    } catch {
      return null;
    }
  }
  if (typeof v !== 'object' || Array.isArray(v)) return null;
  const inner = v[BASTION_REF_KEY];
  if (inner && typeof inner === 'object') return inner;
  if (v.table_id && v.row_id) return v;
  return null;
}

export function buildRefCell(inner) {
  if (!inner || typeof inner !== 'object') return null;
  return { [BASTION_REF_KEY]: { v: 1, ...inner } };
}

export function formatRefLabel(value) {
  const inner = parseRefFromCell(value);
  if (!inner) return '';
  return inner.label != null && String(inner.label).trim() !== ''
    ? String(inner.label)
    : String(inner.row_id || '');
}
