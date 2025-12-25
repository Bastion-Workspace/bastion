import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { QueryClient, QueryClientProvider } from 'react-query';
import { ThemeProvider as CustomThemeProvider, useTheme } from './contexts/ThemeContext';
import { createAppTheme } from './theme/themeConfig';
import './styles/global.css';
import App from './App';

// Viewport height fix for iOS Safari and mobile browsers
const ViewportFix = () => {
  React.useEffect(() => {
    const setVh = () => {
      try {
        const vh = window.innerHeight;
        document.documentElement.style.setProperty('--appvh', `${vh}px`);
      } catch {}
    };
    setVh();
    window.addEventListener('resize', setVh);
    window.addEventListener('orientationchange', setVh);
    return () => {
      window.removeEventListener('resize', setVh);
      window.removeEventListener('orientationchange', setVh);
    };
  }, []);
  return null;
};

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Dynamic Theme Component
const DynamicThemeProvider = ({ children }) => {
  const { darkMode } = useTheme();
  const theme = React.useMemo(() => createAppTheme(darkMode), [darkMode]);
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  // ❌ REMOVED React.StrictMode: Causes double-mounting which duplicates CodeMirror diff decorations
  // StrictMode is useful in development but causes issues with CodeMirror plugins that manage global state
  <QueryClientProvider client={queryClient}>
    <CustomThemeProvider>
      <DynamicThemeProvider>
        <BrowserRouter>
          <ViewportFix />
          <App />
        </BrowserRouter>
      </DynamicThemeProvider>
    </CustomThemeProvider>
  </QueryClientProvider>
);
