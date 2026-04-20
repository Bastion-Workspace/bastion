import React, {
  createContext,
  useContext,
  useRef,
  useCallback,
  useMemo,
  useState,
  useEffect,
} from 'react';
import { useQuery } from 'react-query';
import { Box } from '@mui/material';
import { useControlPanes } from './ControlPaneContext';
import savedArtifactService from '../services/savedArtifactService';
import ArtifactRenderer from '../components/chat/ArtifactRenderer';
import { setArtifactNotifyHandler } from '../utils/artifactBridge';

const ArtifactInstanceContext = createContext(null);

function ArtifactPanePoolSlot({ pane, setSlotRef, onIframeReady }) {
  const w = Math.max(200, Number(pane.artifact_popover_width) || 360);
  const h = Math.max(120, Number(pane.artifact_popover_height) || 400);
  const { data, isLoading } = useQuery(
    ['savedArtifactFull', pane.artifact_id],
    () => savedArtifactService.get(pane.artifact_id),
    { enabled: Boolean(pane.artifact_id), staleTime: 2 * 60 * 1000 }
  );

  if (!pane.artifact_id) return null;
  if (isLoading || !data) return null;

  const artifact = {
    artifact_type: data.artifact_type,
    title: data.title,
    code: data.code,
    language: data.language,
  };

  return (
    <Box
      ref={setSlotRef}
      sx={{
        width: w,
        height: h,
        minWidth: w,
        minHeight: h,
        flexShrink: 0,
        overflow: 'hidden',
        '& iframe': { display: 'block' },
      }}
    >
      <ArtifactRenderer
        artifact={artifact}
        artifactId={pane.artifact_id}
        height="100%"
        onIframeMount={onIframeReady}
      />
    </Box>
  );
}

export function ArtifactInstanceProvider({ children }) {
  const { visiblePanes } = useControlPanes();
  const artifactPanes = useMemo(
    () =>
      (visiblePanes || []).filter(
        (p) => (p.pane_type || 'connector') === 'artifact' && p.artifact_id
      ),
    [visiblePanes]
  );

  const poolSlotRefs = useRef(new Map());
  const iframeByPaneId = useRef(new Map());
  const [badges, setBadges] = useState({});

  const setPoolSlotRef = useCallback((paneId) => (el) => {
    if (el) {
      poolSlotRefs.current.set(paneId, el);
    } else {
      poolSlotRefs.current.delete(paneId);
    }
  }, []);

  const registerIframe = useCallback((paneId, el) => {
    if (paneId && el) {
      iframeByPaneId.current.set(paneId, el);
    }
  }, []);

  useEffect(() => {
    setArtifactNotifyHandler(({ artifactId, payload }) => {
      setBadges((prev) => ({
        ...prev,
        [artifactId]: {
          show: Boolean(payload?.badge),
          text: typeof payload?.text === 'string' ? payload.text : '',
        },
      }));
    });
    return () => setArtifactNotifyHandler(null);
  }, []);

  const mountArtifactInContainer = useCallback((paneId, containerEl) => {
    const iframe = iframeByPaneId.current.get(paneId);
    if (!iframe || !containerEl) return;
    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.minHeight = '200px';
    iframe.style.display = 'block';
    containerEl.appendChild(iframe);
  }, []);

  const restoreArtifactToPool = useCallback((paneId) => {
    const iframe = iframeByPaneId.current.get(paneId);
    const slot = poolSlotRefs.current.get(paneId);
    if (iframe && slot) {
      slot.appendChild(iframe);
    }
  }, []);

  const getBadgeForArtifact = useCallback(
    (artifactId) => (artifactId ? badges[artifactId] : null),
    [badges]
  );

  const clearBadgeForArtifact = useCallback((artifactId) => {
    if (!artifactId) return;
    setBadges((prev) => {
      const next = { ...prev };
      delete next[artifactId];
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({
      mountArtifactInContainer,
      restoreArtifactToPool,
      registerIframe,
      getBadgeForArtifact,
      clearBadgeForArtifact,
    }),
    [
      mountArtifactInContainer,
      restoreArtifactToPool,
      registerIframe,
      getBadgeForArtifact,
      clearBadgeForArtifact,
    ]
  );

  return (
    <ArtifactInstanceContext.Provider value={value}>
      {children}
      <Box
        aria-hidden
        sx={{
          position: 'fixed',
          left: -99999,
          top: 0,
          width: 1,
          height: 1,
          overflow: 'hidden',
          pointerEvents: 'none',
          visibility: 'hidden',
          zIndex: -1,
        }}
      >
        {artifactPanes.map((pane) => (
          <ArtifactPanePoolSlot
            key={pane.id}
            pane={pane}
            setSlotRef={setPoolSlotRef(pane.id)}
            onIframeReady={(el) => registerIframe(pane.id, el)}
          />
        ))}
      </Box>
    </ArtifactInstanceContext.Provider>
  );
}

export function useArtifactInstance() {
  const ctx = useContext(ArtifactInstanceContext);
  if (!ctx) {
    throw new Error('useArtifactInstance must be used within ArtifactInstanceProvider');
  }
  return ctx;
}
