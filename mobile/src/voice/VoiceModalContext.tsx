import { createContext, useContext, useMemo, type ReactNode } from 'react';
import { useVoiceModalController } from './useVoiceModal';

type VoiceModalContextValue = {
  openVoice: () => void;
};

const VoiceModalContext = createContext<VoiceModalContextValue | null>(null);

function VoiceModalBridge({ children }: { children: ReactNode }) {
  const { openVoice, modalElement } = useVoiceModalController();
  const value = useMemo(() => ({ openVoice }), [openVoice]);

  return (
    <VoiceModalContext.Provider value={value}>
      {children}
      {modalElement}
    </VoiceModalContext.Provider>
  );
}

/** Provides voice capture modal and `openVoice` for the dock mic button and deep links. */
export function VoiceModalProvider({ children }: { children: ReactNode }) {
  return <VoiceModalBridge>{children}</VoiceModalBridge>;
}

export function useVoiceModal(): VoiceModalContextValue {
  const ctx = useContext(VoiceModalContext);
  if (!ctx) {
    throw new Error('useVoiceModal must be used within VoiceModalProvider');
  }
  return ctx;
}
