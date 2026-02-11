import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { Modal, Fade, Box, IconButton, Typography, Tooltip, FormControlLabel, Switch, Popover, CircularProgress } from '@mui/material';
import { Close, Download, Subject } from '@mui/icons-material';
import apiService from '../../services/apiService';

const ImageLightboxContext = createContext();

export const useImageLightbox = () => {
  const context = useContext(ImageLightboxContext);
  if (!context) {
    throw new Error('useImageLightbox must be used within ImageLightboxProvider');
  }
  return context;
};

const getBbox = (item) => {
  const x = item.bbox_x ?? item.left ?? item.x ?? 0;
  const y = item.bbox_y ?? item.top ?? item.y ?? 0;
  const w = item.bbox_width ?? (item.right != null && item.left != null ? item.right - item.left : 0) ?? item.width ?? 0;
  const h = item.bbox_height ?? (item.bottom != null && item.top != null ? item.bottom - item.top : 0) ?? item.height ?? 0;
  return { x, y, w, h };
};

export const ImageLightboxProvider = ({ children }) => {
  const [lightboxState, setLightboxState] = useState({
    isOpen: false,
    imageUrl: null,
    alt: null,
    filename: null,
    documentId: null
  });

  const openLightbox = (imageUrl, { alt, filename, documentId } = {}) => {
    setLightboxState({
      isOpen: true,
      imageUrl,
      alt: alt || 'Image',
      filename,
      documentId: documentId ?? null
    });
  };

  const closeLightbox = () => {
    setLightboxState({
      isOpen: false,
      imageUrl: null,
      alt: null,
      filename: null,
      documentId: null
    });
  };

  return (
    <ImageLightboxContext.Provider value={{ openLightbox, closeLightbox }}>
      {children}
      <ImageLightboxModal {...lightboxState} onClose={closeLightbox} />
    </ImageLightboxContext.Provider>
  );
};

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 8;
const ZOOM_SENSITIVITY = 0.0012;

const ImageLightboxModal = ({ isOpen, imageUrl, alt, filename, documentId, onClose }) => {
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(false);
  const [showFaces, setShowFaces] = useState(true);
  const [showObjects, setShowObjects] = useState(true);
  const [boxesData, setBoxesData] = useState({ faces: [], objects: [] });
  const [imageDisplaySize, setImageDisplaySize] = useState({ w: 0, h: 0 });
  const imageRef = useRef(null);
  const viewportRef = useRef(null);
  const [descriptionAnchorEl, setDescriptionAnchorEl] = useState(null);
  const [descriptionContent, setDescriptionContent] = useState(null);
  const [descriptionLoading, setDescriptionLoading] = useState(false);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!isOpen) {
      setShowBoundingBoxes(false);
      setShowFaces(true);
      setShowObjects(true);
      setBoxesData({ faces: [], objects: [] });
      setDescriptionAnchorEl(null);
      setZoom(1);
      setPan({ x: 0, y: 0 });
    }
  }, [isOpen]);

  useEffect(() => {
    if (!showBoundingBoxes || !documentId || !isOpen) {
      setBoxesData({ faces: [], objects: [] });
      return;
    }
    let cancelled = false;
    const load = async () => {
      try {
        const [metaRes, facesRes, objectsRes] = await Promise.all([
          apiService.get(`/api/documents/${documentId}/image-metadata`).catch(() => ({ exists: false })),
          apiService.get(`/api/documents/${documentId}/faces`).catch(() => ({ success: false })),
          apiService.getDetectedObjects(documentId).catch(() => ({ success: false }))
        ]);
        if (cancelled) return;
        const faces = (facesRes?.success && Array.isArray(facesRes.faces) && facesRes.faces.length > 0)
          ? facesRes.faces
          : (metaRes?.exists && metaRes?.metadata?.faces && Array.isArray(metaRes.metadata.faces))
            ? metaRes.metadata.faces
            : [];
        const objects = (objectsRes?.success && Array.isArray(objectsRes.objects) && objectsRes.objects.length > 0)
          ? objectsRes.objects
          : (metaRes?.exists && metaRes?.metadata?.objects && Array.isArray(metaRes.metadata.objects))
            ? metaRes.metadata.objects
            : [];
        setBoxesData({ faces, objects });
      } catch {
        if (!cancelled) setBoxesData({ faces: [], objects: [] });
      }
    };
    load();
    return () => { cancelled = true; };
  }, [showBoundingBoxes, documentId, isOpen]);

  const hasBoxes = boxesData.faces.length > 0 || boxesData.objects.length > 0;

  const measureImage = useCallback(() => {
    if (imageRef.current && viewportRef.current) {
      const vRect = viewportRef.current.getBoundingClientRect();
      const nw = imageRef.current.naturalWidth || 1;
      const nh = imageRef.current.naturalHeight || 1;
      const scale = Math.min(vRect.width / nw, vRect.height / nh, 1);
      const w = nw * scale;
      const h = nh * scale;
      if (w > 0 && h > 0) {
        setImageDisplaySize({ w, h });
        setPan({ x: (vRect.width - w) / 2, y: (vRect.height - h) / 2 });
      }
    } else if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setImageDisplaySize({ w: rect.width, h: rect.height });
      }
    }
  }, []);

  useEffect(() => {
    if (isOpen && showBoundingBoxes && hasBoxes) measureImage();
  }, [isOpen, showBoundingBoxes, hasBoxes, boxesData.faces.length, boxesData.objects.length, measureImage]);

  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [imageUrl]);

  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  zoomRef.current = zoom;
  panRef.current = pan;

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onWheel = (e) => {
      if (imageDisplaySize.w <= 0) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;
      const { x: px, y: py } = panRef.current;
      const z = zoomRef.current;
      const contentX = (mouseX - px) / z;
      const contentY = (mouseY - py) / z;
      const delta = -e.deltaY * ZOOM_SENSITIVITY;
      const newZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z * (1 + delta)));
      setZoom(newZoom);
      setPan({
        x: mouseX - contentX * newZoom,
        y: mouseY - contentY * newZoom
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [imageDisplaySize.w]);

  const handleDescriptionClick = useCallback(async (e) => {
    setDescriptionAnchorEl(e.currentTarget);
    setDescriptionLoading(true);
    setDescriptionContent(null);
    try {
      const res = await apiService.get(`/api/documents/${documentId}/image-metadata`);
      const content = res?.exists && res?.metadata?.content ? res.metadata.content : null;
      setDescriptionContent(content);
    } catch {
      setDescriptionContent(null);
    } finally {
      setDescriptionLoading(false);
    }
  }, [documentId]);

  const showHeader = filename || (documentId && imageUrl);
  const maxImgHeight = showHeader ? '85vh' : '95vh';

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'rgba(0, 0, 0, 0.9)'
      }}
    >
      <Fade in={isOpen}>
        <Box
          sx={{
            position: 'relative',
            maxWidth: '95vw',
            maxHeight: '95vh',
            outline: 'none',
            display: 'flex',
            flexDirection: 'column',
            gap: 1
          }}
        >
          {/* Header with filename, optional bounding-box switch, and actions */}
          {showHeader && (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: 1,
                bgcolor: 'rgba(0, 0, 0, 0.7)',
                px: 2,
                py: 1,
                borderRadius: 1
              }}
            >
              <Typography variant="body2" sx={{ color: 'white' }}>
                {filename || 'Image'}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {documentId && (
                  <>
                    <FormControlLabel
                      control={
                        <Switch
                          size="small"
                          checked={showBoundingBoxes}
                          onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                        />
                      }
                      label={<Typography variant="body2" sx={{ color: 'white' }}>Boxes</Typography>}
                    />
                    {showBoundingBoxes && (
                      <>
                        <FormControlLabel
                          control={
                            <Switch
                              size="small"
                              checked={showFaces}
                              onChange={(e) => setShowFaces(e.target.checked)}
                            />
                          }
                          label={<Typography variant="body2" sx={{ color: 'white' }}>Faces</Typography>}
                        />
                        <FormControlLabel
                          control={
                            <Switch
                              size="small"
                              checked={showObjects}
                              onChange={(e) => setShowObjects(e.target.checked)}
                            />
                          }
                          label={<Typography variant="body2" sx={{ color: 'white' }}>Objects</Typography>}
                        />
                      </>
                    )}
                  </>
                )}
                {documentId && (
                  <Tooltip title="Show description from metadata">
                    <IconButton size="small" onClick={handleDescriptionClick} sx={{ color: 'white' }}>
                      <Subject fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
                <Tooltip title="Download">
                  <IconButton
                    component="a"
                    href={imageUrl}
                    download={filename}
                    size="small"
                    sx={{ color: 'white' }}
                  >
                    <Download fontSize="small" />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Close">
                  <IconButton
                    onClick={onClose}
                    size="small"
                    sx={{ color: 'white' }}
                  >
                    <Close fontSize="small" />
                  </IconButton>
                </Tooltip>
              </Box>
            </Box>
          )}

          {/* Description popover */}
          <Popover
            open={Boolean(descriptionAnchorEl)}
            anchorEl={descriptionAnchorEl}
            onClose={() => setDescriptionAnchorEl(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            transformOrigin={{ vertical: 'top', horizontal: 'left' }}
            PaperProps={{ sx: { bgcolor: 'grey.900', color: 'white' } }}
          >
            <Box sx={{ p: 2, maxWidth: 420, maxHeight: 360, overflow: 'auto', minWidth: 200 }}>
              {descriptionLoading ? (
                <CircularProgress size={24} sx={{ color: 'white' }} />
              ) : (
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {descriptionContent || 'No description in metadata.'}
                </Typography>
              )}
            </Box>
          </Popover>

          {/* Image with optional bounding-box overlay when documentId provided */}
          {imageUrl && (
            <Box
              ref={viewportRef}
              sx={{
                position: 'relative',
                width: '95vw',
                height: maxImgHeight,
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Box
                sx={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  width: imageDisplaySize.w || '100%',
                  height: imageDisplaySize.h || '100%',
                  transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                  transformOrigin: '0 0'
                }}
              >
                <Box sx={{ position: 'relative', display: 'inline-block' }}>
                  <Box
                    ref={imageRef}
                    component="img"
                    src={imageUrl}
                    alt={alt}
                    onLoad={measureImage}
                    sx={{
                      display: 'block',
                      maxWidth: '100%',
                      maxHeight: '100%',
                      width: imageDisplaySize.w || 'auto',
                      height: imageDisplaySize.h || 'auto',
                      objectFit: 'contain',
                      borderRadius: 1,
                      boxShadow: 24,
                      bgcolor: 'rgba(255, 255, 255, 0.05)',
                      pointerEvents: 'none'
                    }}
                  />
                  {documentId && showBoundingBoxes && hasBoxes && imageDisplaySize.w > 0 && imageRef.current && (
                    <svg
                      style={{
                        position: 'absolute',
                        left: 0,
                        top: 0,
                        width: imageDisplaySize.w,
                        height: imageDisplaySize.h,
                        pointerEvents: 'none'
                      }}
                      width={imageDisplaySize.w}
                      height={imageDisplaySize.h}
                      viewBox={`0 0 ${imageDisplaySize.w} ${imageDisplaySize.h}`}
                    >
                  {showFaces && boxesData.faces.map((face, idx) => {
                    const nw = imageRef.current?.naturalWidth || 1;
                    const nh = imageRef.current?.naturalHeight || 1;
                    const sx = imageDisplaySize.w / nw;
                    const sy = imageDisplaySize.h / nh;
                    const { x, y, w, h } = getBbox(face);
                    const rx = x * sx;
                    const ry = y * sy;
                    const rw = w * sx;
                    const rh = h * sy;
                    const label = face.identity_name || face.suggested_identity || null;
                    return (
                      <g key={face.id ?? `face-${idx}`}>
                        <rect x={rx} y={ry} width={rw} height={rh} fill="none" stroke="lime" strokeWidth={2} />
                        {label && (
                          <text
                            x={rx}
                            y={Math.max(ry - 4, 12)}
                            fill="white"
                            fontSize={12}
                            fontWeight="bold"
                            style={{ paintOrder: 'stroke', stroke: 'black', strokeWidth: 2 }}
                          >
                            {face.identity_name ? label : `${label}?`}
                          </text>
                        )}
                      </g>
                    );
                  })}
                  {showObjects && boxesData.objects.map((obj, idx) => {
                    const nw = imageRef.current?.naturalWidth || 1;
                    const nh = imageRef.current?.naturalHeight || 1;
                    const sx = imageDisplaySize.w / nw;
                    const sy = imageDisplaySize.h / nh;
                    const { x, y, w, h } = getBbox(obj);
                    const rx = x * sx;
                    const ry = y * sy;
                    const rw = w * sx;
                    const rh = h * sy;
                    const label = obj.user_tag ?? obj.matched_description ?? obj.class_name ?? obj.label ?? obj.name ?? null;
                    return (
                      <g key={obj.id ?? `obj-${idx}`}>
                        <rect x={rx} y={ry} width={rw} height={rh} fill="none" stroke={obj.detection_method === 'user_defined' ? '#9333ea' : '#f97316'} strokeWidth={2} />
                        {label && (
                          <text
                            x={rx}
                            y={Math.max(ry - 4, 12)}
                            fill="white"
                            fontSize={12}
                            fontWeight="bold"
                            style={{ paintOrder: 'stroke', stroke: 'black', strokeWidth: 2 }}
                          >
                            {label}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </svg>
              )}
                </Box>
              </Box>
            </Box>
          )}

          {/* Close button overlay (when no header) */}
          {!showHeader && (
            <IconButton
              onClick={onClose}
              sx={{
                position: 'absolute',
                top: 8,
                right: 8,
                bgcolor: 'rgba(255, 255, 255, 0.9)',
                '&:hover': { bgcolor: 'white' }
              }}
            >
              <Close />
            </IconButton>
          )}
        </Box>
      </Fade>
    </Modal>
  );
};
