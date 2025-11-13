import React from 'react';
import { Box } from '@mui/material';
import { useTheme } from '../contexts/ThemeContext';

const ThemeAwareBox = ({ 
  children, 
  variant = 'default', 
  elevation = 0,
  sx = {}, 
  ...props 
}) => {
  const { darkMode } = useTheme();

  const getVariantStyles = () => {
    const baseStyles = {
      borderRadius: 2,
      transition: 'all 0.3s ease',
    };

    switch (variant) {
      case 'card':
        return {
          ...baseStyles,
          backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
          border: `1px solid ${darkMode ? '#424242' : '#e0e0e0'}`,
          boxShadow: elevation > 0 
            ? darkMode 
              ? `0px ${elevation * 2}px ${elevation * 4}px rgba(0,0,0,0.3)`
              : `0px ${elevation * 2}px ${elevation * 4}px rgba(0,0,0,0.1)`
            : 'none',
        };
      
      case 'surface':
        return {
          ...baseStyles,
          backgroundColor: darkMode ? '#2d2d2d' : '#fafafa',
          border: `1px solid ${darkMode ? '#424242' : '#e0e0e0'}`,
        };
      
      case 'elevated':
        return {
          ...baseStyles,
          backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
          boxShadow: darkMode 
            ? '0px 4px 8px rgba(0,0,0,0.3)'
            : '0px 4px 8px rgba(0,0,0,0.1)',
        };
      
      case 'outlined':
        return {
          ...baseStyles,
          backgroundColor: 'transparent',
          border: `2px solid ${darkMode ? '#424242' : '#e0e0e0'}`,
        };
      
      case 'glass':
        return {
          ...baseStyles,
          backgroundColor: darkMode 
            ? 'rgba(30, 30, 30, 0.8)' 
            : 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(10px)',
          border: `1px solid ${darkMode ? 'rgba(66, 66, 66, 0.3)' : 'rgba(224, 224, 224, 0.3)'}`,
        };
      
      default:
        return {
          ...baseStyles,
          backgroundColor: 'transparent',
        };
    }
  };

  const combinedSx = {
    ...getVariantStyles(),
    ...sx,
  };

  return (
    <Box sx={combinedSx} {...props}>
      {children}
    </Box>
  );
};

export default ThemeAwareBox; 