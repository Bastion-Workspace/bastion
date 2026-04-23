import React, { createContext, useContext, useState } from 'react';

const ModelContext = createContext();

export const useModel = () => {
  const ctx = useContext(ModelContext);
  if (!ctx) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return ctx;
};

/**
 * Legacy context for selected model. Chat UI state and persistence live in
 * ChatSidebarContext (per-user localStorage + selectModel). This provider remains
 * so imports do not break; consumers should prefer useChatSidebar for chat model.
 */
export const ModelProvider = ({ children }) => {
  const [selectedModel, setSelectedModel] = useState('');

  const value = {
    selectedModel,
    setSelectedModel,
  };

  return (
    <ModelContext.Provider value={value}>
      {children}
    </ModelContext.Provider>
  );
};
