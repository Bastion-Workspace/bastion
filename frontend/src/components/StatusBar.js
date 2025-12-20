import React, { useState, useEffect } from 'react';
import { Box, Typography, Tooltip } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { useQuery } from 'react-query';
import statusBarService from '../services/statusBarService';
import apiService from '../services/apiService';
import { APP_VERSION } from '../config/version';
import MusicStatusBarControls from './music/MusicStatusBarControls';

const StatusBar = () => {
  const theme = useTheme();
  const [statusData, setStatusData] = useState({
    current_time: '',
    date_formatted: '',
    weather: null,
    app_version: APP_VERSION
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

  const timeFormat = timeFormatData?.time_format || '24h';

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
          app_version: data.app_version || prev.app_version || APP_VERSION
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

    // Update time every second
    const timeInterval = setInterval(() => {
      const now = new Date();
      const use12Hour = timeFormat === '12h';
      setStatusData(prev => ({
        ...prev,
        current_time: now.toLocaleTimeString('en-US', { hour12: use12Hour, hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        date_formatted: now.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' })
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
  }, [timeFormat]);

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

      {/* Right side: App Version */}
      <Typography variant="caption" sx={{ fontSize: '0.75rem', color: 'text.secondary', flexShrink: 0 }}>
        v{statusData?.app_version || APP_VERSION}
      </Typography>
    </Box>
  );
};

export default StatusBar;

