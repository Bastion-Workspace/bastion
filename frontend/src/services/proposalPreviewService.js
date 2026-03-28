/**
 * Applies resolved edit operations to document content to produce a preview.
 * Operations are applied in reverse position order so earlier offsets remain valid.
 * @param {string} content - Current document content
 * @param {Array<{ from?: number, to?: number, start?: number, end?: number, proposed?: string, opType: string }>} resolvedOps - Resolved operations with start/end (or from/to) and proposed text
 * @returns {string} Content with all operations applied
 */
export function applyProposalsToContent(content, resolvedOps) {
  if (!content || !Array.isArray(resolvedOps) || resolvedOps.length === 0) {
    return content;
  }
  const normalized = resolvedOps
    .map((op) => {
      const start = op.from !== undefined ? op.from : (op.start !== undefined ? op.start : 0);
      const end = op.to !== undefined ? op.to : (op.end !== undefined ? op.end : start);
      return { ...op, start, end };
    })
    .filter((op) => op.start >= 0 && op.end >= op.start);
  const sorted = [...normalized].sort((a, b) => b.start - a.start);
  let result = content;
  for (const op of sorted) {
    const { start, end, proposed, opType } = op;
    if (opType === 'replace_range') {
      result = result.slice(0, start) + (proposed ?? '') + result.slice(end);
    } else if (opType === 'delete_range') {
      result = result.slice(0, start) + result.slice(end);
    } else if (opType === 'insert_after_heading' || opType === 'insert_after' || opType === 'append') {
      result = result.slice(0, start) + (proposed ?? '') + result.slice(start);
    }
  }
  return result;
}
