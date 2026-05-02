/**
 * Extracts and parses YAML frontmatter from a markdown document body (--- delimited).
 * Handles a small subset used by Bastion: scalars, quoted scalars, inline [a, b] lists,
 * and indented dash lists. Returns {} when there is no frontmatter block.
 */
export function parseFrontmatter(content: string): Record<string, unknown> {
  const match = content.match(/^\s*---\r?\n([\s\S]*?)\r?\n---\s*(?:\r?\n|$)/);
  if (!match) return {};

  const lines = match[1].split(/\r?\n/);
  const result: Record<string, unknown> = {};
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      i += 1;
      continue;
    }

    const kv = line.match(/^([a-zA-Z_][\w-]*):\s*(.*)$/);
    if (!kv) {
      i += 1;
      continue;
    }

    const key = kv[1];
    let raw = kv[2].trim();
    if (
      (raw.startsWith('"') && raw.endsWith('"')) ||
      (raw.startsWith("'") && raw.endsWith("'"))
    ) {
      raw = raw.slice(1, -1);
    }

    if (raw.startsWith('[')) {
      const inner = raw.replace(/^\[|\]$/g, '');
      result[key] = inner
        .split(',')
        .map((s) => s.trim().replace(/^["']|["']$/g, ''))
        .filter(Boolean);
      i += 1;
      continue;
    }

    if (raw === '') {
      const items: string[] = [];
      let j = i + 1;
      while (j < lines.length && /^\s+-\s+/.test(lines[j])) {
        items.push(lines[j].replace(/^\s+-\s+/, '').trim());
        j += 1;
      }
      if (items.length > 0) {
        result[key] = items;
        i = j;
      } else {
        result[key] = '';
        i += 1;
      }
      continue;
    }

    result[key] = raw;
    i += 1;
  }

  return result;
}
