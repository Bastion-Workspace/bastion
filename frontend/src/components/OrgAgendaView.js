import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  ToggleButtonGroup,
  ToggleButton,
  Paper
} from '@mui/material';
import {
  CalendarToday,
  Schedule,
  Error as ErrorIcon,
  LocalOffer,
  Description,
  Repeat,
  TrendingUp
} from '@mui/icons-material';
import apiService from '../services/apiService';

/**
 * ROOSEVELT'S ORG AGENDA VIEW
 * View scheduled and deadline items across org files
 */
const OrgAgendaView = ({ onOpenDocument }) => {
  const [viewMode, setViewMode] = useState('week'); // day, week, month
  const [loading, setLoading] = useState(true);
  const [agendaData, setAgendaData] = useState(null);
  const [error, setError] = useState(null);

  // Map view mode to days
  const getDaysForView = useCallback((mode) => {
    switch (mode) {
      case 'day': return 1;
      case 'week': return 7;
      case 'month': return 30;
      default: return 7;
    }
  }, []);

  // Load agenda data
  const loadAgenda = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const days = getDaysForView(viewMode);
      const response = await apiService.get(`/api/org/agenda?days_ahead=${days}`);

      if (response.success) {
        setAgendaData(response);
      } else {
        setError(response.error || 'Failed to load agenda');
      }
    } catch (err) {
      console.error('❌ Agenda error:', err);
      setError(err.message || 'Failed to load agenda');
    } finally {
      setLoading(false);
    }
  }, [viewMode, getDaysForView]);

  // Load on mount and when view mode changes
  useEffect(() => {
    loadAgenda();
  }, [loadAgenda]);

  // Handle clicking an agenda item
  const handleItemClick = (item) => {
    if (!onOpenDocument) return;

    // document_id is already in the search results!
    if (!item.document_id) {
      console.error('❌ Agenda item missing document_id:', item);
      alert(`❌ Could not find document ID for: ${item.filename}`);
      return;
    }

    console.log('✅ ROOSEVELT: Opening org file:', item.document_id);
    
    // Open document with scroll parameters
    onOpenDocument({
      documentId: item.document_id,
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

  // **ROOSEVELT RECURRING TASKS!** Check if task has a repeater
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

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <CalendarToday /> Org Agenda
        </Typography>

        <ToggleButtonGroup
          value={viewMode}
          exclusive
          onChange={(e, newMode) => newMode && setViewMode(newMode)}
          size="small"
          sx={{ mb: 1 }}
        >
          <ToggleButton value="day">Day</ToggleButton>
          <ToggleButton value="week">Week</ToggleButton>
          <ToggleButton value="month">Month</ToggleButton>
        </ToggleButtonGroup>

        {agendaData && (
          <Typography variant="caption" color="text.secondary" display="block">
            {agendaData.count} items • {agendaData.date_range.start} to {agendaData.date_range.end}
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

        {!loading && agendaData && (
          <>
            {agendaData.count === 0 ? (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <CalendarToday sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  No Agenda Items
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  No scheduled or deadline items in the selected time range
                </Typography>
              </Box>
            ) : (
              <List disablePadding>
                {Object.entries(agendaData.grouped_by_date).map(([date, items]) => (
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
                        {items.map((item, idx) => (
                          <React.Fragment key={idx}>
                            {idx > 0 && <Divider />}
                            <ListItem disablePadding>
                              <ListItemButton onClick={() => handleItemClick(item)}>
                                <Box sx={{ width: '100%' }}>
                                  {/* Heading and Badges */}
                                  <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.5 }}>
                                    <Typography variant="body1" sx={{ flex: 1, fontWeight: 500 }}>
                                      {'•'.repeat(item.level)} {item.heading}
                                    </Typography>

                                    {item.agenda_type === 'DEADLINE' && (
                                      <Chip
                                        label={item.is_urgent ? `URGENT (${item.days_until}d)` : `DEADLINE (${item.days_until}d)`}
                                        size="small"
                                        color={item.is_urgent ? 'error' : 'warning'}
                                        sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                                      />
                                    )}

                                    {item.agenda_type === 'SCHEDULED' && (
                                      <Chip
                                        label="SCHEDULED"
                                        size="small"
                                        color="info"
                                        sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                                      />
                                    )}

                                    {item.todo_state && (
                                      <Chip
                                        label={item.todo_state}
                                        size="small"
                                        color={getTodoStateColor(item.todo_state)}
                                        sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                                      />
                                    )}

                                    {/* **ROOSEVELT RECURRING!** Show repeater badge */}
                                    {isRecurring(item) && (
                                      <Chip
                                        icon={<Repeat fontSize="small" />}
                                        label={getRepeaterInfo(item) || 'Recurring'}
                                        size="small"
                                        color="secondary"
                                        variant="outlined"
                                        sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                                      />
                                    )}
                                  </Box>

                                  {/* Metadata Row */}
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                                    <Chip
                                      icon={<Description fontSize="small" />}
                                      label={item.filename}
                                      size="small"
                                      variant="outlined"
                                      sx={{ fontSize: '0.7rem' }}
                                    />

                                    {item.tags && item.tags.length > 0 && (
                                      <Box sx={{ display: 'flex', gap: 0.5 }}>
                                        {item.tags.map(tag => (
                                          <Chip
                                            key={tag}
                                            icon={<LocalOffer sx={{ fontSize: 12 }} />}
                                            label={tag}
                                            size="small"
                                            color="primary"
                                            variant="outlined"
                                            sx={{ fontSize: '0.7rem', height: 20 }}
                                          />
                                        ))}
                                      </Box>
                                    )}
                                  </Box>
                                </Box>
                              </ListItemButton>
                            </ListItem>
                          </React.Fragment>
                        ))}
                      </List>
                    </Paper>
                  </Box>
                ))}
              </List>
            )}
          </>
        )}
      </Box>
    </Box>
  );
};

export default OrgAgendaView;

