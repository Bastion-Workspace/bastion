import { useState, useRef } from 'react';

export const useChatState = () => {
  // Basic state management
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [sessionId, setSessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  
  // UI State
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [savingNoteFor, setSavingNoteFor] = useState(null);
  const [copiedMessageId, setCopiedMessageId] = useState(null);
  const [forceExecutionMode, setForceExecutionMode] = useState(null);
  
  // Job service state
  const [backgroundJobService, setBackgroundJobService] = useState(null);
  
  // Track plans that are currently being executed
  const [executingPlans, setExecutingPlans] = useState(new Set());

  // Refs
  const messagesEndRef = useRef(null);
  const textFieldRef = useRef(null);
  const lastLoadedConversationRef = useRef(null);

  return {
    // State
    query,
    setQuery,
    messages,
    setMessages,
    currentConversationId,
    setCurrentConversationId,
    sessionId,
    setSessionId,
    isLoading,
    setIsLoading,
    sidebarCollapsed,
    setSidebarCollapsed,
    savingNoteFor,
    setSavingNoteFor,
    copiedMessageId,
    setCopiedMessageId,
    forceExecutionMode,
    setForceExecutionMode,
    backgroundJobService,
    setBackgroundJobService,
    executingPlans,
    setExecutingPlans,
    
    // Refs
    messagesEndRef,
    textFieldRef,
    lastLoadedConversationRef,
  };
}; 