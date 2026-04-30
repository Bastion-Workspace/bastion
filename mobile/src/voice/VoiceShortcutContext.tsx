import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

type VoiceShortcutContextValue = {
  /** Incremented when an external shortcut requests the voice capture UI. */
  voiceOpenRequestId: number;
  requestVoiceOpen: () => void;
};

const VoiceShortcutContext = createContext<VoiceShortcutContextValue | null>(null);

export function VoiceShortcutProvider({ children }: { children: ReactNode }) {
  const [voiceOpenRequestId, setVoiceOpenRequestId] = useState(0);

  const requestVoiceOpen = useCallback(() => {
    setVoiceOpenRequestId((n) => n + 1);
  }, []);

  const value = useMemo(
    () => ({ voiceOpenRequestId, requestVoiceOpen }),
    [voiceOpenRequestId, requestVoiceOpen]
  );

  return <VoiceShortcutContext.Provider value={value}>{children}</VoiceShortcutContext.Provider>;
}

export function useVoiceShortcut(): VoiceShortcutContextValue {
  const ctx = useContext(VoiceShortcutContext);
  if (!ctx) {
    throw new Error('useVoiceShortcut must be used within VoiceShortcutProvider');
  }
  return ctx;
}
