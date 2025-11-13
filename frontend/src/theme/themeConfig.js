import { createTheme } from '@mui/material/styles';

export const createAppTheme = (darkMode) => {
  return createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: darkMode ? '#90caf9' : '#1976d2',
        light: darkMode ? '#e3f2fd' : '#42a5f5',
        dark: darkMode ? '#42a5f5' : '#1565c0',
        contrastText: darkMode ? '#000' : '#fff',
      },
      secondary: {
        main: darkMode ? '#f48fb1' : '#dc004e',
        light: darkMode ? '#fce4ec' : '#ff5983',
        dark: darkMode ? '#c2185b' : '#9a0036',
        contrastText: darkMode ? '#000' : '#fff',
      },
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
          root: {
            backgroundColor: darkMode ? '#1e1e1e' : '#1976d2',
            color: darkMode ? '#ffffff' : '#ffffff',
            boxShadow: darkMode 
              ? '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)'
              : '0px 2px 4px -1px rgba(0,0,0,0.2), 0px 4px 5px 0px rgba(0,0,0,0.14), 0px 1px 10px 0px rgba(0,0,0,0.12)',
          },
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
          root: {
            '& .MuiOutlinedInput-root': {
              backgroundColor: darkMode ? '#2d2d2d' : '#ffffff',
              '& fieldset': {
                borderColor: darkMode ? '#424242' : '#e0e0e0',
              },
              '&:hover fieldset': {
                borderColor: darkMode ? '#616161' : '#bdbdbd',
              },
              '&.Mui-focused fieldset': {
                borderColor: darkMode ? '#90caf9' : '#1976d2',
              },
            },
          },
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