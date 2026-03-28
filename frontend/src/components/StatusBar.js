import React, { useState, useEffect } from 'react';
import { Box, Typography, Tooltip } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useQuery } from 'react-query';
import statusBarService from '../services/statusBarService';
import apiService from '../services/apiService';
import MusicStatusBarControls from './music/MusicStatusBarControls';
import ControlPaneIcons from './control-panes/ControlPaneIcons';
import LineStatusIndicators from './agent-factory/LineStatusIndicators';

const StatusBar = () => {
  const theme = useTheme();
  const [statusData, setStatusData] = useState({
    current_time: '',
    date_formatted: '',
    weather: null,
    app_version: ''
  });

  // Fetch user time format preference
  const { data: timeFormatData } = useQuery(
    'userTimeFormat',
    () => apiService.settings.getUserTimeFormat(),
    {
      onSuccess: (data) => {
        // Time format is available in data.time_format
      },
      onError: (error) => {
        console.error('Failed to fetch user time format:', error);
      },
      staleTime: 5 * 60 * 1000, // Cache for 5 minutes
      refetchOnWindowFocus: false
    }
  );

  // Fetch user timezone so status bar clock reflects settings (shared cache with Settings page)
  const { data: timezoneData } = useQuery(
    'userTimezone',
    () => apiService.getUserTimezone(),
    {
      onError: (error) => {
        console.error('Failed to fetch user timezone:', error);
      },
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false
    }
  );

  const timeFormat = timeFormatData?.time_format || '24h';
  const userTimezone = timezoneData?.timezone || undefined;

  const fetchStatusData = async () => {
    try {
      const data = await statusBarService.getStatusBarData();
      if (data && typeof data === 'object') {
        setStatusData(prev => ({
          ...prev,
          ...data,
          // Ensure we always have required fields
          current_time: data.current_time || prev.current_time || '',
          date_formatted: data.date_formatted || prev.date_formatted || '',
          app_version: data.app_version || prev.app_version
        }));
      }
    } catch (error) {
      console.error('Error fetching status bar data:', error);
      // Don't update state on error, keep existing data
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchStatusData();

    const use12Hour = timeFormat === '12h';
    const timeOpts = { hour12: use12Hour, hour: '2-digit', minute: '2-digit', second: '2-digit' };
    const dateOpts = { month: '2-digit', day: '2-digit', year: 'numeric' };
    if (userTimezone) {
      timeOpts.timeZone = userTimezone;
      dateOpts.timeZone = userTimezone;
    }

    // Update time every second using user's timezone from settings
    const timeInterval = setInterval(() => {
      const now = new Date();
      let currentTimeStr = '';
      let dateFormattedStr = '';
      try {
        currentTimeStr = now.toLocaleTimeString('en-US', timeOpts);
        dateFormattedStr = now.toLocaleDateString('en-US', dateOpts);
      } catch (e) {
        currentTimeStr = now.toLocaleTimeString('en-US', { hour12: use12Hour, hour: '2-digit', minute: '2-digit', second: '2-digit' });
        dateFormattedStr = now.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' });
      }
      setStatusData(prev => ({
        ...prev,
        current_time: currentTimeStr,
        date_formatted: dateFormattedStr
      }));
    }, 1000);

    // Refresh weather data every 10 minutes
    const weatherInterval = setInterval(() => {
      fetchStatusData();
    }, 10 * 60 * 1000);

    return () => {
      clearInterval(timeInterval);
      clearInterval(weatherInterval);
    };
  }, [timeFormat, userTimezone]);

  const formatWeatherDisplay = () => {
    if (!statusData.weather) {
      return null;
    }

    const { location, temperature, conditions, moon_phase } = statusData.weather;
    const moonIcon = moon_phase?.phase_icon || '🌙';
    const moonPhaseName = moon_phase?.phase_name || 'Moon';
    
    return (
      <>
        {location}, {temperature}°F, {conditions}{' '}
        <Tooltip title={moonPhaseName} arrow>
          <span style={{ cursor: 'help' }}>{moonIcon}</span>
        </Tooltip>
      </>
    );
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        height: '32px',
        backgroundColor: theme.palette.mode === 'dark' 
          ? theme.palette.grey[900] 
          : theme.palette.grey[100],
        borderTop: `1px solid ${theme.palette.divider}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingX: 2,
        zIndex: 1300,
        fontSize: '0.75rem',
      }}
    >
      {/* Left side: Date/Time and Weather */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, minWidth: 0, flexShrink: 0 }}>
        <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
          {statusData?.date_formatted || ''} - {statusData?.current_time || ''}
        </Typography>
        {statusData?.weather && (
          <>
            <Typography variant="caption" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
              |
            </Typography>
            <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
              {formatWeatherDisplay()}
            </Typography>
          </>
        )}
      </Box>

      {/* Center: Music Controls */}
      <MusicStatusBarControls />

      {/* Right segment: Team status + Control Pane Icons + Version (right-aligned together) */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexShrink: 0, marginLeft: 'auto' }}>
        <LineStatusIndicators />
        <ControlPaneIcons />
        <Typography variant="caption" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
          v{statusData?.app_version || '…'}
        </Typography>
      </Box>
    </Box>
  );
};

export default StatusBar;

