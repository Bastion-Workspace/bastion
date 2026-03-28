import { createTheme } from '@mui/material/styles';

/**
 * Accent palettes: primary (and optional secondary) per mode.
 * Used by createAppTheme(darkMode, accentId).
 */
export const ACCENT_PALETTES = {
  blue: {
    light: {
      primary: { main: '#1976d2', light: '#42a5f5', dark: '#1565c0', contrastText: '#fff' },
      secondary: { main: '#dc004e', light: '#ff5983', dark: '#9a0036', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#90caf9', light: '#e3f2fd', dark: '#42a5f5', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#fce4ec', dark: '#c2185b', contrastText: '#000' },
    },
  },
  teal: {
    light: {
      primary: { main: '#00897b', light: '#4db6ac', dark: '#00695c', contrastText: '#fff' },
      secondary: { main: '#dc004e', light: '#ff5983', dark: '#9a0036', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#80cbc4', light: '#b2dfdb', dark: '#4db6ac', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#fce4ec', dark: '#c2185b', contrastText: '#000' },
    },
  },
  green: {
    light: {
      primary: { main: '#43a047', light: '#66bb6a', dark: '#2e7d32', contrastText: '#fff' },
      secondary: { main: '#dc004e', light: '#ff5983', dark: '#9a0036', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#81c784', light: '#a5d6a7', dark: '#66bb6a', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#fce4ec', dark: '#c2185b', contrastText: '#000' },
    },
  },
  purple: {
    light: {
      primary: { main: '#7b1fa2', light: '#9c27b0', dark: '#6a1b9a', contrastText: '#fff' },
      secondary: { main: '#dc004e', light: '#ff5983', dark: '#9a0036', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#ce93d8', light: '#e1bee7', dark: '#ba68c8', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#fce4ec', dark: '#c2185b', contrastText: '#000' },
    },
  },
  orange: {
    light: {
      primary: { main: '#e65100', light: '#ff9800', dark: '#bf360c', contrastText: '#fff' },
      secondary: { main: '#dc004e', light: '#ff5983', dark: '#9a0036', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#ffb74d', light: '#ffcc80', dark: '#ffa726', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#fce4ec', dark: '#c2185b', contrastText: '#000' },
    },
  },
  indigo: {
    light: {
      primary: { main: '#3949ab', light: '#5c6bc0', dark: '#283593', contrastText: '#fff' },
      secondary: { main: '#dc004e', light: '#ff5983', dark: '#9a0036', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#9fa8da', light: '#c5cae9', dark: '#7986cb', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#fce4ec', dark: '#c2185b', contrastText: '#000' },
    },
  },
  rose: {
    light: {
      primary: { main: '#d81b60', light: '#ec407a', dark: '#ad1457', contrastText: '#fff' },
      secondary: { main: '#7b1fa2', light: '#9c27b0', dark: '#6a1b9a', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#f48fb1', light: '#f8bbd0', dark: '#ec407a', contrastText: '#000' },
      secondary: { main: '#ce93d8', light: '#e1bee7', dark: '#ba68c8', contrastText: '#000' },
    },
  },
  cyan: {
    light: {
      primary: { main: '#00838f', light: '#26c6da', dark: '#006064', contrastText: '#fff' },
      secondary: { main: '#00796b', light: '#26a69a', dark: '#004d40', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#80deea', light: '#b2ebf2', dark: '#4dd0e1', contrastText: '#000' },
      secondary: { main: '#80cbc4', light: '#b2dfdb', dark: '#4db6ac', contrastText: '#000' },
    },
  },
  amber: {
    light: {
      primary: { main: '#ff8f00', light: '#ffb300', dark: '#ff6f00', contrastText: '#fff' },
      secondary: { main: '#6d4c41', light: '#8d6e63', dark: '#5d4037', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#ffd54f', light: '#ffe082', dark: '#ffca28', contrastText: '#000' },
      secondary: { main: '#bcaaa4', light: '#d7ccc8', dark: '#a1887f', contrastText: '#000' },
    },
  },
  red: {
    light: {
      primary: { main: '#c62828', light: '#ef5350', dark: '#b71c1c', contrastText: '#fff' },
      secondary: { main: '#ad1457', light: '#d81b60', dark: '#880e4f', contrastText: '#fff' },
    },
    dark: {
      primary: { main: '#ef9a9a', light: '#ffcdd2', dark: '#e57373', contrastText: '#000' },
      secondary: { main: '#f48fb1', light: '#f8bbd0', dark: '#ec407a', contrastText: '#000' },
    },
  },
};

const getAccentPalette = (darkMode, accentId) => {
  const palettes = ACCENT_PALETTES[accentId] || ACCENT_PALETTES.blue;
  return darkMode ? palettes.dark : palettes.light;
};

export const createAppTheme = (darkMode, accentId = 'blue') => {
  const accent = getAccentPalette(darkMode, accentId);
  return createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: accent.primary,
      secondary: accent.secondary,
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
        secondary: darkMode ? '#2d2d2d' : '#fafafa',
      },
      surface: {
        main: darkMode ? '#2d2d2d' : '#ffffff',
        light: darkMode ? '#424242' : '#f5f5f5',
        dark: darkMode ? '#1e1e1e' : '#e0e0e0',
      },
      text: {
        primary: darkMode ? '#ffffff' : '#212121',
        secondary: darkMode ? '#b3b3b3' : '#757575',
        disabled: darkMode ? '#666666' : '#bdbdbd',
      },
      divider: darkMode ? '#424242' : '#e0e0e0',
      action: {
        active: darkMode ? '#ffffff' : '#212121',
        hover: darkMode ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
        selected: darkMode ? 'rgba(255, 255, 255, 0.16)' : 'rgba(0, 0, 0, 0.08)',
        disabled: darkMode ? 'rgba(255, 255, 255, 0.3)' : 'rgba(0, 0, 0, 0.26)',
        disabledBackground: darkMode ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)',
      },
      success: {
        main: darkMode ? '#66bb6a' : '#4caf50',
        light: darkMode ? '#81c784' : '#81c784',
        dark: darkMode ? '#388e3c' : '#388e3c',
      },
      warning: {
        main: darkMode ? '#ffa726' : '#ff9800',
        light: darkMode ? '#ffb74d' : '#ffb74d',
        dark: darkMode ? '#f57c00' : '#f57c00',
      },
      error: {
        main: darkMode ? '#f44336' : '#f44336',
        light: darkMode ? '#e57373' : '#e57373',
        dark: darkMode ? '#d32f2f' : '#d32f2f',
      },
      info: {
        main: darkMode ? '#29b6f6' : '#2196f3',
        light: darkMode ? '#4fc3f7' : '#64b5f6',
        dark: darkMode ? '#0288d1' : '#1976d2',
      },
    },
    typography: {
      fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
      h1: {
        fontWeight: 700,
        color: darkMode ? '#ffffff' : '#212121',
      },
      h2: {
        fontWeight: 600,
        color: darkMode ? '#ffffff' : '#212121',
      },
      h3: {
        fontWeight: 600,
        color: darkMode ? '#ffffff' : '#212121',
      },
      h4: {
        fontWeight: 600,
        color: darkMode ? '#ffffff' : '#212121',
      },
      h5: {
        fontWeight: 500,
        color: darkMode ? '#ffffff' : '#212121',
      },
      h6: {
        fontWeight: 500,
        color: darkMode ? '#ffffff' : '#212121',
      },
      body1: {
        color: darkMode ? '#e0e0e0' : '#424242',
      },
      body2: {
        color: darkMode ? '#b3b3b3' : '#757575',
      },
      caption: {
        color: darkMode ? '#b3b3b3' : '#757575',
      },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarColor: darkMode ? '#424242 #121212' : '#bdbdbd #f5f5f5',
            '&::-webkit-scrollbar, & *::-webkit-scrollbar': {
              width: '8px',
              height: '8px',
            },
            '&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb': {
              borderRadius: 4,
              backgroundColor: darkMode ? '#424242' : '#bdbdbd',
              minHeight: 24,
            },
            '&::-webkit-scrollbar-thumb:focus, & *::-webkit-scrollbar-thumb:focus': {
              backgroundColor: darkMode ? '#616161' : '#9e9e9e',
            },
            '&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track': {
              backgroundColor: darkMode ? '#121212' : '#f5f5f5',
            },
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: ({ theme }) => ({
            backgroundColor: darkMode ? '#1e1e1e' : theme.palette.primary.main,
            color: darkMode ? '#ffffff' : '#ffffff',
            boxShadow: darkMode
              ? '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)'
              : '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)',
          }),
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
            border: darkMode ? '1px solid #424242' : '1px solid #e0e0e0',
            boxShadow: darkMode 
              ? '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)'
              : '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)',
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
            border: darkMode ? '1px solid #424242' : '1px solid #e0e0e0',
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            borderRadius: 8,
            fontWeight: 500,
          },
          contained: {
            boxShadow: darkMode 
              ? '0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 2px 0px rgba(0,0,0,0.14), 0px 1px 5px 0px rgba(0,0,0,0.12)'
              : '0px 3px 1px -2px rgba(0,0,0,0.2), 0px 2px 2px 0px rgba(0,0,0,0.14), 0px 1px 5px 0px rgba(0,0,0,0.12)',
          },
        },
      },
      MuiTextField: {
        styleOverrides: {
          root: ({ theme }) => ({
            '& .MuiOutlinedInput-root': {
              backgroundColor: darkMode ? '#2d2d2d' : '#ffffff',
              '& fieldset': {
                borderColor: darkMode ? '#424242' : '#e0e0e0',
              },
              '&:hover fieldset': {
                borderColor: darkMode ? '#616161' : '#bdbdbd',
              },
              '&.Mui-focused fieldset': {
                borderColor: theme.palette.primary.main,
              },
            },
          }),
        },
      },
      MuiInputBase: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? '#2d2d2d' : '#ffffff',
            '& .MuiInputBase-input': {
              color: darkMode ? '#ffffff' : '#212121',
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? '#424242' : '#e0e0e0',
            color: darkMode ? '#ffffff' : '#212121',
          },
        },
      },
      MuiDivider: {
        styleOverrides: {
          root: {
            borderColor: darkMode ? '#424242' : '#e0e0e0',
          },
        },
      },
      MuiMenu: {
        styleOverrides: {
          paper: {
            backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
            border: darkMode ? '1px solid #424242' : '1px solid #e0e0e0',
            boxShadow: darkMode 
              ? '0px 5px 5px -3px rgba(0,0,0,0.2), 0px 8px 10px 1px rgba(0,0,0,0.14), 0px 3px 14px 2px rgba(0,0,0,0.12)'
              : '0px 5px 5px -3px rgba(0,0,0,0.2), 0px 8px 10px 1px rgba(0,0,0,0.14), 0px 3px 14px 2px rgba(0,0,0,0.12)',
          },
        },
      },
      MuiMenuItem: {
        styleOverrides: {
          root: {
            '&:hover': {
              backgroundColor: darkMode ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)',
            },
          },
        },
      },
      MuiTableHead: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? '#2d2d2d' : '#f5f5f5',
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderBottom: darkMode ? '1px solid #424242' : '1px solid #e0e0e0',
          },
        },
      },
      MuiDialog: {
        styleOverrides: {
          paper: {
            backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
            border: darkMode ? '1px solid #424242' : '1px solid #e0e0e0',
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: darkMode ? '#1e1e1e' : '#ffffff',
            border: darkMode ? '1px solid #424242' : '1px solid #e0e0e0',
          },
        },
      },
    },
  });
}; 