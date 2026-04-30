import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

type AppLauncherContextValue = {
  open: boolean;
  openLauncher: () => void;
  closeLauncher: () => void;
};

const AppLauncherContext = createContext<AppLauncherContextValue | null>(null);

export function AppLauncherProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const openLauncher = useCallback(() => setOpen(true), []);
  const closeLauncher = useCallback(() => setOpen(false), []);
  const value = useMemo(
    () => ({ open, openLauncher, closeLauncher }),
    [open, openLauncher, closeLauncher]
  );
  return <AppLauncherContext.Provider value={value}>{children}</AppLauncherContext.Provider>;
}

export function useAppLauncher(): AppLauncherContextValue {
  const ctx = useContext(AppLauncherContext);
  if (!ctx) {
    throw new Error('useAppLauncher must be used within AppLauncherProvider');
  }
  return ctx;
}
