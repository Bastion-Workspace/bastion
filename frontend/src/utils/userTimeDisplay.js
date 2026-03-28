/**
 * Time display helpers aligned with StatusBar: user time_format (12h / 24h) and optional IANA timezone.
 */

/**
 * @param {Date} date
 * @param {{ timeFormat?: string }} options timeFormat '12h' | '24h'
 * @returns {string}
 */
export function formatWallClockHm(date, { timeFormat = '24h' } = {}) {
  if (!date || !(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  const hour12 = timeFormat === '12h';
  const opts = { hour12, hour: 'numeric', minute: '2-digit' };
  try {
    return date.toLocaleTimeString('en-US', opts);
  } catch {
    return date.toLocaleTimeString('en-US', { hour12, hour: 'numeric', minute: '2-digit' });
  }
}

/**
 * Format an absolute instant in the user's timezone (when provided).
 * @param {Date} date
 * @param {{ timeFormat?: string, timeZone?: string }} options
 * @returns {string}
 */
export function formatInstantHm(date, { timeFormat = '24h', timeZone } = {}) {
  if (!date || !(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  const hour12 = timeFormat === '12h';
  const opts = { hour12, hour: 'numeric', minute: '2-digit' };
  if (timeZone) opts.timeZone = timeZone;
  try {
    return date.toLocaleTimeString('en-US', opts);
  } catch {
    return date.toLocaleTimeString('en-US', { hour12, hour: 'numeric', minute: '2-digit' });
  }
}

/**
 * Format an absolute instant as date + time using user 12h/24h and optional IANA timezone (StatusBar-aligned).
 * @param {Date|string|number} input
 * @param {{ timeFormat?: string, timeZone?: string }} options
 * @returns {string}
 */
export function formatInstantDateTime(input, { timeFormat = '24h', timeZone } = {}) {
  const date = input instanceof Date ? input : new Date(input);
  if (!date || Number.isNaN(date.getTime())) return '';
  const hour12 = timeFormat === '12h';
  const opts = {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12,
  };
  if (timeZone) opts.timeZone = timeZone;
  try {
    return date.toLocaleString('en-US', opts);
  } catch {
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12,
    });
  }
}

/**
 * @param {string} agendaDate YYYY-MM-DD
 * @param {string} timePart HH:MM
 * @returns {Date|null}
 */
export function parseOrgAgendaWallTime(agendaDate, timePart) {
  if (!agendaDate || !timePart) return null;
  const m = String(timePart).trim().match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return null;
  const h = parseInt(m[1], 10);
  const min = parseInt(m[2], 10);
  if (Number.isNaN(h) || Number.isNaN(min)) return null;
  const isoLocal = `${agendaDate}T${String(h).padStart(2, '0')}:${String(min).padStart(2, '0')}:00`;
  const d = new Date(isoLocal);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Civil date + wall clock interpreted in an IANA zone → UTC instant (for org agenda lines).
 * Falls back to browser-local parse when timeZone is omitted.
 */
export function zonedWallTimeToDate(ymd, hm, timeZone) {
  if (!timeZone) return parseOrgAgendaWallTime(ymd, hm);
  const [y, month, day] = ymd.split('-').map(Number);
  const match = String(hm).trim().match(/^(\d{1,2}):(\d{2})$/);
  if (!match) return null;
  const h = parseInt(match[1], 10);
  const mi = parseInt(match[2], 10);
  if (Number.isNaN(y) || Number.isNaN(month) || Number.isNaN(day) || Number.isNaN(h) || Number.isNaN(mi)) {
    return null;
  }
  let dtf;
  try {
    dtf = new Intl.DateTimeFormat('en-US', {
      timeZone,
      year: 'numeric',
      month: 'numeric',
      day: 'numeric',
      hour: 'numeric',
      minute: 'numeric',
      hour12: false,
    });
  } catch {
    return parseOrgAgendaWallTime(ymd, hm);
  }
  const readZoned = (instant) => {
    const z = {};
    for (const p of dtf.formatToParts(instant)) {
      if (p.type !== 'literal') z[p.type] = parseInt(p.value, 10);
    }
    return z;
  };
  const targetDay = new Date(y, month - 1, day).getTime();
  let t = Date.UTC(y, month - 1, day, h, mi, 0);
  for (let i = 0; i < 56; i += 1) {
    const d = new Date(t);
    const z = readZoned(d);
    if (
      Number.isFinite(z.year) &&
      z.year === y &&
      z.month === month &&
      z.day === day &&
      z.hour === h &&
      z.minute === mi
    ) {
      return d;
    }
    if (!Number.isFinite(z.year) || !Number.isFinite(z.month) || !Number.isFinite(z.day)) {
      return parseOrgAgendaWallTime(ymd, hm);
    }
    const zDay = new Date(z.year, z.month - 1, z.day).getTime();
    const dayDiffDays = Math.round((targetDay - zDay) / 86400000);
    t += dayDiffDays * 86400000;
    t += ((h - z.hour) * 60 + (mi - z.minute)) * 60 * 1000;
  }
  return new Date(t);
}

/**
 * Org agenda `time` field: "14:00", "14:00-15:00", or "14:00–15:00".
 * @param {string} rawTime
 * @param {string} agendaDate YYYY-MM-DD
 * @param {{ timeFormat?: string, timeZone?: string }} options timeZone = user's settings IANA zone (org times are wall times in that zone)
 * @returns {string|null}
 */
export function formatOrgAgendaTimeForDisplay(rawTime, agendaDate, { timeFormat = '24h', timeZone } = {}) {
  if (rawTime == null || rawTime === '') return null;
  const trimmed = String(rawTime).trim();
  if (trimmed === '00:00' || /^00:00\s*[-–]/.test(trimmed) || trimmed === '00:00 – 00:00') {
    return 'All day';
  }
  const range = trimmed.match(/^(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})$/);
  const startPart = range ? range[1] : trimmed;
  const endPart = range ? range[2] : null;
  const d0 = zonedWallTimeToDate(agendaDate, startPart, timeZone);
  if (!d0) return trimmed;
  const fmtOpts = { timeFormat, timeZone };
  const startStr = timeZone
    ? formatInstantHm(d0, fmtOpts)
    : formatWallClockHm(d0, { timeFormat });
  if (!endPart) return startStr;
  const d1 = zonedWallTimeToDate(agendaDate, endPart, timeZone);
  if (!d1) return `${startStr}–${endPart}`;
  const endStr = timeZone ? formatInstantHm(d1, fmtOpts) : formatWallClockHm(d1, { timeFormat });
  return `${startStr}–${endStr}`;
}

/**
 * Microsoft 365 / ISO agenda row.
 * @param {object} item merged row with start_datetime, end_datetime, is_all_day
 * @param {{ timeFormat?: string, timeZone?: string }} options
 * @returns {string|null}
 */
export function formatO365AgendaTimeLabel(item, { timeFormat = '24h', timeZone } = {}) {
  if (item.is_all_day) return 'All day';
  if (!item.start_datetime) return null;
  const startRaw = String(item.start_datetime).slice(11, 16);
  const endRaw = String(item.end_datetime || '').slice(11, 16);
  if (startRaw === '00:00' && (!endRaw || endRaw === '00:00' || endRaw === '23:59')) {
    return 'All day';
  }
  const ds = new Date(item.start_datetime);
  if (Number.isNaN(ds.getTime())) return null;
  const startStr = formatInstantHm(ds, { timeFormat, timeZone });
  if (!item.end_datetime) return startStr;
  const de = new Date(item.end_datetime);
  if (Number.isNaN(de.getTime())) return startStr;
  const endStr = formatInstantHm(de, { timeFormat, timeZone });
  return startStr === endStr ? startStr : `${startStr}–${endStr}`;
}

/**
 * Unified label for OrgAgendaView list rows.
 * @param {object} item
 * @param {{ timeFormat?: string, timeZone?: string }} options
 * @returns {string|null}
 */
export function getAgendaRowTimeLabel(item, options = {}) {
  if (item.source === 'o365') {
    return formatO365AgendaTimeLabel(item, options);
  }
  if (item.source === 'org' && item.time) {
    return formatOrgAgendaTimeForDisplay(item.time, item.agenda_date, options);
  }
  return null;
}

/**
 * Popover / detail line for a timed external event.
 * @param {object} ev
 * @param {{ timeFormat?: string, timeZone?: string }} options
 * @returns {string}
 */
export function formatAgendaPopoverWhenLine(ev, { timeFormat = '24h', timeZone } = {}) {
  if (!ev.start_datetime) return '';
  const rowLabel = formatO365AgendaTimeLabel(ev, { timeFormat, timeZone });
  if (rowLabel === 'All day') {
    return `All day · ${ev.start_datetime.slice(0, 10)}`;
  }
  const ds = new Date(ev.start_datetime);
  const de = ev.end_datetime ? new Date(ev.end_datetime) : null;
  if (Number.isNaN(ds.getTime())) return ev.start_datetime.slice(0, 16).replace('T', ' ');
  const a = formatInstantHm(ds, { timeFormat, timeZone });
  if (!de || Number.isNaN(de.getTime())) return a;
  const b = formatInstantHm(de, { timeFormat, timeZone });
  return `${a} – ${b}`;
}
