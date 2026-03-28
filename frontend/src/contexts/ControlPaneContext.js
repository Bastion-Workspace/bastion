import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { useQuery } from 'react-query';
import apiService from '../services/apiService';

const ControlPaneContext = createContext(null);

function getByPath(obj, path) {
  if (!obj || !path) return undefined;
  const parts = String(path).split('.');
  let current = obj;
  for (const key of parts) {
    if (current == null) return undefined;
    current = current[key];
  }
  return current;
}

export const useControlPanes = () => {
  const context = useContext(ControlPaneContext);
  if (!context) {
    throw new Error('useControlPanes must be used within a ControlPaneProvider');
  }
  return context;
};

export const ControlPaneProvider = ({ children }) => {
  const [paneStates, setPaneStates] = useState({});
  const paneStatesRef = useRef(paneStates);
  useEffect(() => {
    paneStatesRef.current = paneStates;
  }, [paneStates]);

  const { data: allPanes = [], refetch: refetchPanes } = useQuery(
    'controlPanes',
    () => apiService.controlPanes.listPanes(),
    { staleTime: 60 * 1000 }
  );

  const visiblePanes = (allPanes || []).filter((p) => p.is_visible !== false);

  const executeAction = useCallback(async (paneId, endpointId, params = {}) => {
    const result = await apiService.controlPanes.executeAction(paneId, endpointId, params);
    const responseKeys = result && typeof result === 'object' ? Object.keys(result) : [];
    if (import.meta.env.DEV) {
      console.debug('[ControlPane] execute', { paneId, endpointId, params, responseKeys });
    }
    return result;
  }, []);

  const refreshPaneState = useCallback(async (paneId) => {
    const pane = allPanes.find((p) => p.id === paneId);
    if (!pane || !pane.controls || !pane.controls.length) return;

    const controlsNeedingRefresh = pane.controls.filter(
      (c) => c.refresh_endpoint_id && c.value_path
    );
    if (controlsNeedingRefresh.length === 0) return;

    const byEndpoint = new Map();
    for (const control of controlsNeedingRefresh) {
      const refId = control.refresh_endpoint_id;
      if (!byEndpoint.has(refId)) byEndpoint.set(refId, []);
      byEndpoint.get(refId).push(control);
    }

    const uniqueEndpoints = [...byEndpoint.keys()];
    const currentPaneState = paneStatesRef.current[paneId] || {};
    const paramsForEndpoint = (controlsForEndpoint) => {
      const merged = {};
      for (const c of controlsForEndpoint) {
        for (const src of c.refresh_param_source || []) {
          if (src.param && src.from_control_id) {
            const v = currentPaneState[src.from_control_id];
            if (v !== undefined && v !== null && v !== '') merged[src.param] = v;
          }
        }
      }
      return merged;
    };

    let responses;
    try {
      responses = await Promise.all(
        uniqueEndpoints.map((refId) =>
          apiService.controlPanes.executeAction(
            paneId,
            refId,
            paramsForEndpoint(byEndpoint.get(refId))
          )
        )
      );
    } catch (e) {
      if (import.meta.env.DEV) {
        console.warn('[ControlPane] refreshPaneState failed', { paneId, error: e?.message });
      }
      return;
    }
    const responseByEndpoint = new Map(
      uniqueEndpoints.map((refId, i) => [refId, responses[i]])
    );

    setPaneStates((prev) => {
      const next = { ...prev };
      if (!next[paneId]) next[paneId] = {};

      for (const refId of uniqueEndpoints) {
        const res = responseByEndpoint.get(refId);
        const raw = res?.raw_response ?? (Array.isArray(res?.records) ? res.records[0] : res?.records) ?? res;
        const responseKeys = raw && typeof raw === 'object' ? Object.keys(raw) : [];

        for (const control of byEndpoint.get(refId)) {
          const valuePath = control.value_path;
          try {
            const extracted = getByPath(raw, valuePath);
            if (extracted !== undefined) {
              next[paneId][control.id] = extracted;
              if (import.meta.env.DEV) {
                console.debug('[ControlPane] refresh extracted', {
                  label: control.label,
                  value_path: valuePath,
                  value: extracted,
                });
              }
            } else {
              if (import.meta.env.DEV) {
                console.warn('[ControlPane] refresh value_path not found', {
                  label: control.label,
                  value_path: valuePath,
                  responseKeys,
                });
              }
            }
          } catch (e) {
            if (import.meta.env.DEV) {
              console.warn('[ControlPane] refresh extract error', {
                label: control.label,
                value_path: valuePath,
                error: e?.message,
              });
            }
          }
        }
      }
      return next;
    });
  }, [allPanes]);

  const setPaneControlValue = useCallback((paneId, controlId, value) => {
    setPaneStates((prev) => ({
      ...prev,
      [paneId]: {
        ...(prev[paneId] || {}),
        [controlId]: value,
      },
    }));
  }, []);

  const value = {
    visiblePanes,
    allPanes,
    paneStates,
    refetchPanes,
    executeAction,
    refreshPaneState,
    setPaneControlValue,
  };

  return (
    <ControlPaneContext.Provider value={value}>
      {children}
    </ControlPaneContext.Provider>
  );
};
