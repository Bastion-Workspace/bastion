/**
 * Team Execution Context
 * Holds latest team execution status (running/idle) per team_id, updated from central WebSocket.
 * Consumed by StatusBar (TeamStatusIndicators) and TeamDashboardPage.
 */

import React, { createContext, useContext, useState, useCallback } from 'react';

const TeamExecutionContext = createContext();

export const useTeamExecution = () => {
  const context = useContext(TeamExecutionContext);
  if (!context) {
    throw new Error('useTeamExecution must be used within a TeamExecutionProvider');
  }
  return context;
};

export const TeamExecutionProvider = ({ children }) => {
  const [teamStatusMap, setTeamStatusMap] = useState({});

  const setTeamExecutionStatus = useCallback((teamId, data) => {
    if (!teamId) return;
    setTeamStatusMap((prev) => ({
      ...prev,
      [teamId]: {
        status: data.status,
        teamName: data.team_name || prev[teamId]?.teamName || 'Team',
        agentId: data.agent_id ?? null,
        timestamp: data.timestamp || new Date().toISOString(),
      },
    }));
  }, []);

  const value = {
    teamStatusMap,
    setTeamExecutionStatus,
  };

  return (
    <TeamExecutionContext.Provider value={value}>
      {children}
    </TeamExecutionContext.Provider>
  );
};
