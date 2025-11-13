import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { useMutation } from 'react-query';
import apiService from '../services/apiService';

const ModelContext = createContext();

export const useModel = () => {
  const ctx = useContext(ModelContext);
  if (!ctx) {
    throw new Error('useModel must be used within a ModelProvider');
  }
  return ctx;
};

export const ModelProvider = ({ children }) => {
  const [selectedModel, setSelectedModel] = useState('');
  const saveTimerRef = useRef(null);

  // Load saved model and notify backend once on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem('chatSidebarSelectedModel');
      if (saved) {
        setSelectedModel(saved);
        apiService.selectModel(saved).catch(() => {});
      }
    } catch {}
  }, []);

  // Debounce persistence of selected model
  useEffect(() => {
    if (!selectedModel) return;
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      try { localStorage.setItem('chatSidebarSelectedModel', selectedModel); } catch {}
    }, 200);
    return () => saveTimerRef.current && clearTimeout(saveTimerRef.current);
  }, [selectedModel]);

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


