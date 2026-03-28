/**
 * Build markdown and helpers for exporting agent line timeline (PDF / library).
 */

function slugifySegment(s) {
  return (s || 'line')
    .replace(/[^\w\s\-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 72) || 'line';
}

function formatHeaderTime(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function lineLabel(msg) {
  const from = msg.from_agent_name || msg.from_agent_handle || (msg.from_agent_id ? 'Agent' : 'System');
  const to = msg.to_agent_id
    ? msg.to_agent_name || msg.to_agent_handle || 'Agent'
    : null;
  return to ? `${from} → ${to}` : from;
}

/**
 * Fetch all timeline rows matching filters (paginated).
 */
export async function fetchAllTimelineMessages(getLineTimeline, lineId, filters) {
  const { messageType, agentFilter, since } = filters;
  const batch = 200;
  let offset = 0;
  const byId = new Map();
  while (true) {
    const data = await getLineTimeline(lineId, {
      limit: batch,
      offset,
      message_type: messageType,
      agent: agentFilter,
      since,
    });
    const items = data?.items ?? [];
    for (const m of items) {
      if (m?.id) byId.set(m.id, m);
    }
    if (items.length < batch) break;
    offset += batch;
    if (offset > 100000) break;
  }
  const list = Array.from(byId.values());
  list.sort((a, b) => {
    const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
    const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
    return ta - tb;
  });
  return list;
}

/**
 * Build markdown document: chronological, readable export.
 */
export function buildTimelineMarkdown(teamName, messages, filterNote) {
  const title = `Timeline — ${teamName || 'Agent line'}`;
  const exportedAt = new Date().toISOString();
  const lines = [
    '---',
    `title: ${JSON.stringify(title)}`,
    `exported_at: ${exportedAt}`,
    '---',
    '',
    `# ${title}`,
    '',
    `*Exported ${formatHeaderTime(exportedAt)}*`,
  ];
  if (filterNote) {
    lines.push('', `*${filterNote}*`);
  }
  lines.push('', '---', '');

  const typeLabels = {
    task_assignment: 'Task',
    status_update: 'Status',
    request: 'Request',
    response: 'Response',
    delegation: 'Delegation',
    escalation: 'Escalation',
    report: 'Report',
    system: 'System',
  };

  for (const msg of messages) {
    const mt = typeLabels[msg.message_type] || msg.message_type || 'Message';
    const when = formatHeaderTime(msg.created_at);
    lines.push(`## ${mt} · ${lineLabel(msg)}`);
    lines.push(`*${when}*`, '');
    const body = (msg.content || '—').trim();
    lines.push(body);
    lines.push('', '---', '');
  }

  return lines.join('\n');
}

export function timelineExportBasename(teamName) {
  const d = new Date().toISOString().slice(0, 10);
  return `timeline-${slugifySegment(teamName)}-${d}`;
}
