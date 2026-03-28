import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  ToggleButtonGroup,
  ToggleButton,
  Paper,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Collapse,
  Popover
} from '@mui/material';
import {
  CalendarToday,
  Schedule,
  Error as ErrorIcon,
  Description,
  Repeat,
  ExpandMore,
  ExpandLess,
  Event as EventIcon
} from '@mui/icons-material';
import apiService from '../services/apiService';
import { getAgendaRowTimeLabel, formatAgendaPopoverWhenLine } from '../utils/userTimeDisplay';

/**
 * Agenda View: org-mode agenda + O365 (and future CalDAV) calendars.
 * Shows available calendar connections and lets user pick which calendars to display.
 */
const OrgAgendaView = ({ onOpenDocument }) => {
  const { data: timeFormatData } = useQuery(
    'userTimeFormat',
    () => apiService.settings.getUserTimeFormat(),
    { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false }
  );
  const { data: timezoneData } = useQuery(
    'userTimezone',
    () => apiService.getUserTimezone(),
    { staleTime: 5 * 60 * 1000, refetchOnWindowFocus: false }
  );
  const timeFormat = timeFormatData?.time_format || '24h';
  const userTimezone = timezoneData?.timezone || undefined;
  const agendaTimeOptions = useMemo(
    () => ({ timeFormat, timeZone: userTimezone }),
    [timeFormat, userTimezone]
  );

  const [viewMode, setViewMode] = useState('week');
  const [loading, setLoading] = useState(true);
  const [agendaData, setAgendaData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedOrgFiles, setSelectedOrgFiles] = useState(new Set(['calendar.org']));
  const [calendarConnections, setCalendarConnections] = useState([]);
  const [calendars, setCalendars] = useState([]);
  const [selectedCalendarIds, setSelectedCalendarIds] = useState(new Set());
  const [o365Events, setO365Events] = useState([]);
  const [loadingCalendars, setLoadingCalendars] = useState(false);
  const [calendarsExpanded, setCalendarsExpanded] = useState(true);
  const [eventPopover, setEventPopover] = useState({ open: false, anchor: null, event: null });
  const hasMergedOrgFilesFromResponse = useRef(false);

  const getDaysForView = useCallback((mode) => {
    switch (mode) {
      case 'day': return 1;
      case 'week': return 7;
      case 'month': return 30;
      default: return 7;
    }
  }, []);

  const connectionId = calendarConnections[0]?.id ?? null;

  const loadCalendarConnectionsAndCalendars = useCallback(async () => {
    if (!apiService.calendar) return;
    try {
      setLoadingCalendars(true);
      const connRes = await apiService.calendar.getConnections();
      const conns = connRes?.connections ?? [];
      setCalendarConnections(conns);
      if (conns.length > 0) {
        const calRes = await apiService.calendar.getCalendars(conns[0].id);
        const cals = calRes?.calendars ?? [];
        setCalendars(cals);
        if (selectedCalendarIds.size === 0 && cals.length > 0) {
          const defaultId = cals.find((c) => c.is_default)?.id ?? cals[0]?.id;
          if (defaultId) setSelectedCalendarIds(new Set([defaultId]));
        }
      }
    } catch (err) {
      console.error('Calendar connections/calendars error:', err);
    } finally {
      setLoadingCalendars(false);
    }
  }, []);

  const loadO365Events = useCallback(async () => {
    if (!apiService.calendar || !connectionId || selectedCalendarIds.size === 0) {
      setO365Events([]);
      return;
    }
    const days = getDaysForView(viewMode);
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    const end = new Date(start);
    end.setDate(end.getDate() + days);
    const startStr = start.toISOString().slice(0, 19);
    const endStr = end.toISOString().slice(0, 19);
    try {
      const allEvents = [];
      for (const calId of selectedCalendarIds) {
        const res = await apiService.calendar.getEvents(startStr, endStr, connectionId, calId, 100);
        const list = res?.events ?? [];
        allEvents.push(...list.map((e) => ({ ...e, _calendarId: calId })));
      }
      allEvents.sort((a, b) => (a.start_datetime || '').localeCompare(b.start_datetime || ''));
      setO365Events(allEvents);
    } catch (err) {
      console.error('O365 events error:', err);
      setO365Events([]);
    }
  }, [connectionId, selectedCalendarIds, viewMode, getDaysForView]);

  const loadAgenda = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const days = getDaysForView(viewMode);
      const params = new URLSearchParams({ days_ahead: days });
      if (selectedOrgFiles.size > 0) {
        selectedOrgFiles.forEach((f) => params.append('include_org_files', f));
      } else {
        params.append('include_org_files', '');
      }
      const response = await apiService.get(`/api/org/agenda?${params.toString()}`);
      if (response.success) {
        setAgendaData(response);
        if (response.org_files_present?.length && !hasMergedOrgFilesFromResponse.current) {
          hasMergedOrgFilesFromResponse.current = true;
          setSelectedOrgFiles((prev) => {
            const next = new Set([...prev, ...response.org_files_present]);
            if (next.size === prev.size && [...next].every((f) => prev.has(f))) return prev;
            return next;
          });
        }
      } else {
        setError(response.error || 'Failed to load agenda');
      }
    } catch (err) {
      console.error('Agenda error:', err);
      setError(err.message || 'Failed to load agenda');
    } finally {
      setLoading(false);
    }
  }, [viewMode, getDaysForView, selectedOrgFiles]);

  useEffect(() => {
    loadCalendarConnectionsAndCalendars();
  }, [loadCalendarConnectionsAndCalendars]);

  useEffect(() => {
    loadAgenda();
  }, [loadAgenda]);

  useEffect(() => {
    loadO365Events();
  }, [loadO365Events]);

  const toggleCalendar = (calId) => {
    setSelectedCalendarIds((prev) => {
      const next = new Set(prev);
      if (next.has(calId)) next.delete(calId);
      else next.add(calId);
      return next;
    });
  };

  const toggleOrgFile = (filename) => {
    setSelectedOrgFiles((prev) => {
      const next = new Set(prev);
      if (next.has(filename)) next.delete(filename);
      else next.add(filename);
      return next;
    });
  };

  const knownOrgFiles = React.useMemo(
    () => Array.from(new Set(['calendar.org', ...(agendaData?.org_files_present || [])])).filter(Boolean).sort(),
    [agendaData?.org_files_present]
  );

  const mergedGroupedByDate = React.useMemo(() => {
    const grouped = {};
    const days = getDaysForView(viewMode);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    for (let d = 0; d < days; d++) {
      const d2 = new Date(today);
      d2.setDate(d2.getDate() + d);
      grouped[d2.toISOString().slice(0, 10)] = [];
    }
    if (agendaData?.grouped_by_date) {
      for (const [dateKey, items] of Object.entries(agendaData.grouped_by_date)) {
        if (!grouped[dateKey]) grouped[dateKey] = [];
        grouped[dateKey].push(...items.map((i) => ({ ...i, source: 'org' })));
      }
    }
    for (const ev of o365Events) {
      const dateKey = (ev.start_datetime || '').slice(0, 10);
      if (!dateKey) continue;
      if (!grouped[dateKey]) grouped[dateKey] = [];
      const start = ev.start_datetime || '';
      const end = ev.end_datetime || '';
      grouped[dateKey].push({
        source: 'o365',
        id: ev.id,
        heading: ev.subject || '(No title)',
        agenda_date: dateKey,
        start_datetime: start,
        end_datetime: end,
        location: ev.location,
        is_all_day: ev.is_all_day,
        web_link: ev.web_link,
        body_preview: ev.body_preview,
        _raw: ev
      });
    }
    for (const key of Object.keys(grouped)) {
      grouped[key].sort((a, b) => {
        const aTime = a.source === 'o365' ? (a.start_datetime || '') : (a.sort_datetime ? new Date(a.sort_datetime).toISOString() : (a.agenda_date || ''));
        const bTime = b.source === 'o365' ? (b.start_datetime || '') : (b.sort_datetime ? new Date(b.sort_datetime).toISOString() : (b.agenda_date || ''));
        return aTime.localeCompare(bTime);
      });
    }
    return grouped;
  }, [agendaData, o365Events, viewMode, getDaysForView]);

  const handleItemClick = async (item, clickEvent) => {
    if (item.source === 'o365') {
      setEventPopover({ open: true, anchor: clickEvent?.currentTarget ?? null, event: item });
      return;
    }
    if (!onOpenDocument) return;

    let documentId = item.document_id;

    // If document_id is missing, try to look it up from filename
    if (!documentId && item.filename) {
      try {
        console.log(`🔍 Looking up document_id for: ${item.filename}`);
        const lookupResult = await apiService.get(`/api/org/lookup-document?filename=${encodeURIComponent(item.filename)}`);
        
        if (lookupResult.success && lookupResult.document_id) {
          documentId = lookupResult.document_id;
          console.log(`✅ Found document_id: ${documentId}`);
        } else {
          console.error('❌ Could not find document ID for:', item.filename);
          alert(`❌ Could not find document ID for: ${item.filename}`);
          return;
        }
      } catch (err) {
        console.error('❌ Failed to lookup document:', err);
        alert(`❌ Failed to find document: ${item.filename}`);
        return;
      }
    }

    if (!documentId) {
      console.error('❌ Agenda item missing document_id:', item);
      alert(`❌ Could not find document ID for: ${item.filename}`);
      return;
    }

    console.log('✅ Opening org file:', documentId);
    
    // Open document with scroll parameters
    onOpenDocument({
      documentId: documentId,
      documentName: item.filename,
      scrollToLine: item.line_number,
      scrollToHeading: item.heading
    });
  };

  // Format date for display
  const formatDate = (dateStr) => {
    const date = new Date(dateStr);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const itemDate = new Date(date);
    itemDate.setHours(0, 0, 0, 0);

    const diffDays = Math.floor((itemDate - today) / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Tomorrow';
    if (diffDays === -1) return 'Yesterday';

    return date.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' });
  };

  // Get badge color for TODO state
  const getTodoStateColor = (state) => {
    const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
    return doneStates.includes(state) ? 'success' : 'error';
  };

  // Detect org repeater syntax on scheduled/deadline timestamps
  const isRecurring = (item) => {
    // Check for repeater syntax: +1w, .+2d, ++1m, etc.
    const repeaterPattern = /[.+]+\d+[dwmy]/;
    return repeaterPattern.test(item.scheduled || '') || repeaterPattern.test(item.deadline || '');
  };

  // Extract repeater info for display
  const getRepeaterInfo = (item) => {
    const timestamp = item.scheduled || item.deadline || '';
    const match = timestamp.match(/([.+]+)(\d+)([dwmy])/);
    if (!match) return null;
    
    const [, type, count, unit] = match;
    const unitNames = { d: 'day', w: 'week', m: 'month', y: 'year' };
    const unitName = unitNames[unit] || unit;
    const plural = count > 1 ? 's' : '';
    
    return `Every ${count} ${unitName}${plural}`;
  };

  const totalMergedCount = Object.values(mergedGroupedByDate).reduce((n, arr) => n + arr.length, 0);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 1.5,
            mb: 1
          }}
        >
          <Typography variant="h6" sx={{ m: 0, display: 'flex', alignItems: 'center', gap: 1 }}>
            <CalendarToday /> Agenda
          </Typography>
          <ToggleButtonGroup
            value={viewMode}
            exclusive
            onChange={(e, newMode) => newMode && setViewMode(newMode)}
            size="small"
            sx={{ flexShrink: 0 }}
          >
            <ToggleButton value="day">Day</ToggleButton>
            <ToggleButton value="week">Week</ToggleButton>
            <ToggleButton value="month">Month</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        <Box sx={{ mt: 1 }}>
          <ListItemButton onClick={() => setCalendarsExpanded(!calendarsExpanded)} sx={{ py: 0, px: 0 }}>
            {calendarsExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
            <Typography variant="subtitle2" sx={{ ml: 0.5 }}>
              Calendars
            </Typography>
          </ListItemButton>
          <Collapse in={calendarsExpanded}>
            {loadingCalendars && calendarConnections.length > 0 ? (
              <CircularProgress size={20} sx={{ mt: 0.5 }} />
            ) : (
              <FormGroup row sx={{ mt: 0.5, gap: 0.5, flexWrap: 'wrap' }}>
                {knownOrgFiles.map((filename) => (
                  <FormControlLabel
                    key={filename}
                    control={
                      <Checkbox
                        size="small"
                        checked={selectedOrgFiles.has(filename)}
                        onChange={() => toggleOrgFile(filename)}
                      />
                    }
                    label={
                      <Typography variant="caption">
                        {filename}
                      </Typography>
                    }
                  />
                ))}
                {calendarConnections.length > 0 && calendars.map((cal) => (
                  <FormControlLabel
                    key={cal.id}
                    control={
                      <Checkbox
                        size="small"
                        checked={selectedCalendarIds.has(cal.id)}
                        onChange={() => toggleCalendar(cal.id)}
                      />
                    }
                    label={
                      <Typography variant="caption">
                        {cal.name}
                        {cal.is_default ? ' (default)' : ''}
                      </Typography>
                    }
                  />
                ))}
              </FormGroup>
            )}
          </Collapse>
        </Box>

        {(agendaData || totalMergedCount > 0) && (
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            {totalMergedCount} items
            {agendaData?.date_range ? ` • ${agendaData.date_range.start} to ${agendaData.date_range.end}` : ''}
          </Typography>
        )}
      </Box>

      {/* Content Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Alert severity="error" icon={<ErrorIcon />}>
            {error}
          </Alert>
        )}

        {!loading && (agendaData || o365Events.length > 0) && (
          <>
            {totalMergedCount === 0 ? (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <CalendarToday sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No Agenda Items
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  No scheduled or deadline items in the selected time range. Connect Microsoft 365 in Settings to see calendars.
                </Typography>
              </Box>
            ) : (
              <List disablePadding>
                {Object.entries(mergedGroupedByDate)
                  .filter(([, items]) => items.length > 0)
                  .map(([date, items]) => (
                  <Box key={date} sx={{ mb: 3 }}>
                    {/* Date Header */}
                    <Typography
                      variant="subtitle2"
                      sx={{
                        fontWeight: 600,
                        mb: 1,
                        color: 'primary.main',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1
                      }}
                    >
                      <CalendarToday fontSize="small" />
                      {formatDate(date)}
                      <Chip label={items.length} size="small" sx={{ ml: 'auto' }} />
                    </Typography>

                    {/* Items for this date */}
                    <Paper variant="outlined">
                      <List disablePadding>
                        {items.map((item, idx) => {
                          const timeLabel = getAgendaRowTimeLabel(item, agendaTimeOptions);
                          return (
                          <React.Fragment key={item.source === 'o365' ? item.id : idx}>
                            {idx > 0 && <Divider />}
                            <ListItem disablePadding>
                              <ListItemButton onClick={(e) => handleItemClick(item, e)} sx={{ py: 0.75 }}>
                                <Box
                                  sx={{
                                    width: '100%',
                                    minWidth: 0,
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 0.75,
                                    flexWrap: 'wrap'
                                  }}
                                >
                                  {item.source === 'o365' ? (
                                    <EventIcon fontSize="small" color="action" sx={{ flexShrink: 0 }} />
                                  ) : null}
                                  <Typography
                                    variant="body2"
                                    sx={{
                                      fontWeight: 500,
                                      minWidth: 0,
                                      flex: '1 1 140px',
                                      maxWidth: '100%',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap'
                                    }}
                                  >
                                    {item.source === 'org' ? ('•'.repeat(item.level || 1) + ' ') : ''}
                                    {item.heading}
                                  </Typography>
                                  {timeLabel && (
                                    <Chip
                                      icon={<Schedule sx={{ fontSize: 14 }} />}
                                      label={timeLabel}
                                      size="small"
                                      variant="outlined"
                                      sx={{
                                        flexShrink: 0,
                                        fontSize: '0.65rem',
                                        height: 22,
                                        fontWeight: 600,
                                        backgroundColor: 'action.hover',
                                        '& .MuiChip-label': { px: 0.75 }
                                      }}
                                    />
                                  )}
                                  {item.source === 'o365' && (
                                    <Chip
                                      label="Microsoft 365"
                                      size="small"
                                      color="primary"
                                      variant="outlined"
                                      sx={{ flexShrink: 0, fontSize: '0.65rem', height: 22, '& .MuiChip-label': { px: 0.75 } }}
                                    />
                                  )}
                                  {item.source === 'org' && item.agenda_type === 'DEADLINE' && (
                                    <Chip
                                      label={item.is_urgent ? `URGENT (${item.days_until}d)` : `DEADLINE (${item.days_until}d)`}
                                      size="small"
                                      color={item.is_urgent ? 'error' : 'warning'}
                                      sx={{
                                        flexShrink: 0,
                                        fontWeight: 600,
                                        fontSize: '0.65rem',
                                        height: 22,
                                        '& .MuiChip-label': { px: 0.75 }
                                      }}
                                    />
                                  )}
                                  {item.source === 'org' && item.agenda_type === 'SCHEDULED' && (
                                    <Chip
                                      label="SCHEDULED"
                                      size="small"
                                      color="info"
                                      sx={{
                                        flexShrink: 0,
                                        fontWeight: 600,
                                        fontSize: '0.65rem',
                                        height: 22,
                                        '& .MuiChip-label': { px: 0.75 }
                                      }}
                                    />
                                  )}
                                  {item.source === 'org' && item.todo_state && (
                                    <Chip
                                      label={item.todo_state}
                                      size="small"
                                      color={getTodoStateColor(item.todo_state)}
                                      sx={{
                                        flexShrink: 0,
                                        fontWeight: 600,
                                        fontSize: '0.65rem',
                                        height: 22,
                                        '& .MuiChip-label': { px: 0.75 }
                                      }}
                                    />
                                  )}
                                  {item.source === 'org' && isRecurring(item) && (
                                    <Chip
                                      icon={<Repeat sx={{ fontSize: 14 }} />}
                                      label={getRepeaterInfo(item) || 'Recurring'}
                                      size="small"
                                      color="secondary"
                                      variant="outlined"
                                      sx={{
                                        flexShrink: 0,
                                        fontWeight: 600,
                                        fontSize: '0.65rem',
                                        height: 22,
                                        '& .MuiChip-label': { px: 0.75 }
                                      }}
                                    />
                                  )}
                                  {item.source === 'org' && item.filename !== 'calendar.org' && (
                                    <Chip
                                      icon={<Description sx={{ fontSize: 14 }} />}
                                      label={item.filename}
                                      size="small"
                                      variant="outlined"
                                      sx={{ flexShrink: 0, fontSize: '0.65rem', height: 22, '& .MuiChip-label': { px: 0.75 } }}
                                    />
                                  )}
                                  {item.source === 'org' && item.tags && item.tags.length > 0 && item.tags.length <= 3
                                    ? item.tags.map((tag) => (
                                        <Chip
                                          key={tag}
                                          label={tag}
                                          size="small"
                                          color="primary"
                                          variant="outlined"
                                          sx={{ flexShrink: 0, fontSize: '0.65rem', height: 22, '& .MuiChip-label': { px: 0.5 } }}
                                        />
                                      ))
                                    : null}
                                  {item.source === 'org' && item.tags && item.tags.length > 3 && (
                                    <Chip
                                      label={`${item.tags.slice(0, 2).join(', ')}…`}
                                      size="small"
                                      color="primary"
                                      variant="outlined"
                                      sx={{ flexShrink: 0, fontSize: '0.65rem', height: 22, '& .MuiChip-label': { px: 0.5 } }}
                                    />
                                  )}
                                </Box>
                              </ListItemButton>
                            </ListItem>
                          </React.Fragment>
                          );
                        })}
                      </List>
                    </Paper>
                  </Box>
                ))}
              </List>
            )}
          </>
        )}
      </Box>

      <Popover
        open={eventPopover.open}
        anchorEl={eventPopover.anchor}
        onClose={() => setEventPopover({ open: false, anchor: null, event: null })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
      >
        {eventPopover.event && (
          <Box sx={{ p: 2, maxWidth: 360 }}>
            <Typography variant="subtitle1" fontWeight={600}>{eventPopover.event.heading}</Typography>
            {eventPopover.event.start_datetime && (
              <Typography variant="body2" color="text.secondary">
                {formatAgendaPopoverWhenLine(eventPopover.event, agendaTimeOptions)}
              </Typography>
            )}
            {eventPopover.event.location && (
              <Typography variant="body2" sx={{ mt: 0.5 }}>{eventPopover.event.location}</Typography>
            )}
            {eventPopover.event.body_preview && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }} noWrap>
                {eventPopover.event.body_preview.slice(0, 120)}{eventPopover.event.body_preview.length > 120 ? '…' : ''}
              </Typography>
            )}
            {eventPopover.event.web_link && (
              <Typography component="a" href={eventPopover.event.web_link} target="_blank" rel="noopener noreferrer" variant="body2" sx={{ mt: 1, display: 'block' }}>
                Open in calendar
              </Typography>
            )}
          </Box>
        )}
      </Popover>
    </Box>
  );
};

export default OrgAgendaView;

