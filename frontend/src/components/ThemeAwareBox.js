import React from 'react';
import { Box } from '@mui/material';
import { useTheme as useMuiTheme } from '@mui/material/styles';
import { useTheme } from '../contexts/ThemeContext';

const ThemeAwareBox = ({ 
  children, 
  variant = 'default', 
  elevation = 0,
  sx = {}, 
  ...props 
}) => {
  const { darkMode } = useTheme();
  const muiTheme = useMuiTheme();

  const getVariantStyles = () => {
    const baseStyles = {
      borderRadius: 2,
      transition: 'all 0.3s ease',
    };
    const bgPaper = muiTheme.palette.background.paper;
    const bgSecondary = muiTheme.palette.background.secondary;
    const divider = muiTheme.palette.divider;

    switch (variant) {
      case 'card':
        return {
          ...baseStyles,
          backgroundColor: bgPaper,
          border: `1px solid ${divider}`,
          boxShadow: elevation > 0 
            ? (darkMode 
              ? `0px ${elevation * 2}px ${elevation * 4}px rgba(0,0,0,0.3)`
              : `0px ${elevation * 2}px ${elevation * 4}px rgba(0,0,0,0.1)`)
            : 'none',
        };
      
      case 'surface':
        return {
          ...baseStyles,
          backgroundColor: bgSecondary,
          border: `1px solid ${divider}`,
        };
      
      case 'elevated':
        return {
          ...baseStyles,
          backgroundColor: bgPaper,
          boxShadow: darkMode 
            ? '0px 4px 8px rgba(0,0,0,0.3)'
            : '0px 4px 8px rgba(0,0,0,0.1)',
        };
      
      case 'outlined':
        return {
          ...baseStyles,
          backgroundColor: 'transparent',
          border: `2px solid ${divider}`,
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