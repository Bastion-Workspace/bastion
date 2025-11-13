/**
 * Roosevelt's Presence Indicator
 * Status dot component for online/offline/away states
 * 
 * BULLY! A simple but effective visual indicator!
 */

import React from 'react';
import { Box, Tooltip } from '@mui/material';

const PresenceIndicator = ({ status, lastSeenAt, statusMessage, size = 'small', showTooltip = true }) => {
  const getColor = () => {
    switch (status) {
      case 'online':
        return '#4caf50'; // Green
      case 'away':
        return '#ff9800'; // Yellow/Orange
      case 'offline':
      default:
        return '#9e9e9e'; // Gray
    }
  };

  const getSize = () => {
    switch (size) {
      case 'large':
        return 16;
      case 'medium':
        return 12;
      case 'small':
      default:
        return 8;
    }
  };

  const formatLastSeen = () => {
    if (!lastSeenAt || status === 'online') return null;
    
    try {
      const date = new Date(lastSeenAt);
      const now = new Date();
      const diffMs = now - date;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);
      
      if (diffMins < 1) return 'Just now';
      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      if (diffDays < 7) return `${diffDays}d ago`;
      return date.toLocaleDateString();
    } catch (error) {
      return null;
    }
  };

  const getTooltipText = () => {
    const statusText = status.charAt(0).toUpperCase() + status.slice(1);
    const lastSeen = formatLastSeen();
    
    let text = statusText;
    if (statusMessage) {
      text += ` - ${statusMessage}`;
    }
    if (lastSeen && status !== 'online') {
      text += ` (${lastSeen})`;
    }
    
    return text;
  };

  const indicator = (
    <Box
      sx={{
        width: getSize(),
        height: getSize(),
        borderRadius: '50%',
        backgroundColor: getColor(),
        border: '2px solid white',
        boxShadow: status === 'online' ? '0 0 4px rgba(76, 175, 80, 0.5)' : 'none',
        flexShrink: 0,
      }}
    />
  );

  if (showTooltip) {
    return (
      <Tooltip title={getTooltipText()} placement="top" arrow>
        {indicator}
      </Tooltip>
    );
  }

  return indicator;
};

export default PresenceIndicator;

