/**
 * ObjectAnnotationCanvas - Interactive canvas for drawing bounding boxes on images
 * Click and drag to draw a rectangle; enter description and save as user-defined object.
 */

import React, { useState, useRef, useCallback } from 'react';
import {
  Box,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
} from '@mui/material';
import apiService from '../../services/apiService';

const COLORS = {
  yolo: '#f97316',
  user_defined: '#9333ea',
  clip_semantic: '#0ea5e9',
  unconfirmed: '#eab308',
};

const ObjectAnnotationCanvas = ({
  documentId,
  imageUrl,
  imageWidth,
  imageHeight,
  detectedObjects = [],
  userAnnotations = [],
  onAnnotationSaved,
  onClose,
}) => {
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const [naturalSize, setNaturalSize] = useState({ width: imageWidth || 0, height: imageHeight || 0 });
  const [drawing, setDrawing] = useState(false);
  const [start, setStart] = useState(null);
  const [currentBox, setCurrentBox] = useState(null);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [objectName, setObjectName] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const scaleX = imageWidth && containerRef.current?.clientWidth ? containerRef.current.clientWidth / imageWidth : 1;
  const scaleY = imageHeight && containerRef.current?.clientHeight ? containerRef.current.clientHeight / imageHeight : 1;

  const toDisplay = useCallback(
    (bbox) => ({
      x: (bbox.bbox_x ?? bbox.x) * scaleX,
      y: (bbox.bbox_y ?? bbox.y) * scaleY,
      width: (bbox.bbox_width ?? bbox.width) * scaleX,
      height: (bbox.bbox_height ?? bbox.height) * scaleY,
    }),
    [scaleX, scaleY]
  );

  const toServer = useCallback(
    (displayX, displayY, displayW, displayH) => ({
      x: Math.round(displayX / scaleX),
      y: Math.round(displayY / scaleY),
      width: Math.round(displayW / scaleX),
      height: Math.round(displayH / scaleY),
    }),
    [scaleX, scaleY]
  );

  const handleMouseDown = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setDrawing(true);
    setStart({ x, y });
    setCurrentBox({ x, y, width: 0, height: 0 });
  };

  const handleMouseMove = (e) => {
    if (!drawing || !start || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const minX = Math.min(start.x, x);
    const minY = Math.min(start.y, y);
    const width = Math.abs(x - start.x);
    const height = Math.abs(y - start.y);
    setCurrentBox({ x: minX, y: minY, width, height });
  };

  const handleMouseUp = () => {
    if (!drawing || !start || !currentBox) return;
    setDrawing(false);
    const { width, height } = currentBox;
    if (width < 5 || height < 5) {
      setStart(null);
      setCurrentBox(null);
      return;
    }
    setShowSaveDialog(true);
  };

  const handleSaveAnnotation = async () => {
    if (!objectName.trim() || !currentBox) return;
    setSaving(true);
    setError(null);
    try {
      const serverBbox = toServer(currentBox.x, currentBox.y, currentBox.width, currentBox.height);
      const response = await apiService.post(`/api/documents/${documentId}/annotate-object`, {
        object_name: objectName.trim(),
        description: description.trim(),
        bbox: serverBbox,
      });
      if (response.success && onAnnotationSaved) {
        onAnnotationSaved(response.annotation);
      }
      setShowSaveDialog(false);
      setObjectName('');
      setDescription('');
      setStart(null);
      setCurrentBox(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to save annotation');
    } finally {
      setSaving(false);
    }
  };

  const handleCancelSave = () => {
    setShowSaveDialog(false);
    setObjectName('');
    setDescription('');
    setError(null);
    setStart(null);
    setCurrentBox(null);
  };

  const allBoxes = [
    ...detectedObjects.map((obj) => ({ ...obj, type: obj.detection_method || 'yolo' })),
    ...userAnnotations.map((a) => ({ ...a, type: 'user_defined' })),
  ];

  return (
    <Box sx={{ position: 'relative', display: 'inline-block' }}>
      <Box
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          if (drawing) setDrawing(false);
        }}
        sx={{
          position: 'relative',
          cursor: 'crosshair',
          maxWidth: '100%',
          maxHeight: '70vh',
          overflow: 'hidden',
        }}
      >
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Annotate"
          draggable={false}
          style={{ display: 'block', maxWidth: '100%', maxHeight: '70vh', pointerEvents: 'none' }}
          onLoad={(e) => {
            if (imageWidth && imageHeight) return;
            const img = e.target;
            if (img.naturalWidth && img.naturalHeight) {
              setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
            }
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            pointerEvents: 'none',
          }}
        >
          {allBoxes.map((obj, idx) => {
            const d = toDisplay(obj);
            const color = COLORS[obj.type] || COLORS.unconfirmed;
            return (
              <Box
                key={obj.id || idx}
                sx={{
                  position: 'absolute',
                  left: d.x,
                  top: d.y,
                  width: d.width,
                  height: d.height,
                  border: `2px solid ${color}`,
                  borderRadius: 1,
                  boxSizing: 'border-box',
                }}
              />
            );
          })}
          {currentBox && (currentBox.width > 0 || currentBox.height > 0) && (
            <Box
              sx={{
                position: 'absolute',
                left: currentBox.x,
                top: currentBox.y,
                width: currentBox.width,
                height: currentBox.height,
                border: '2px dashed #22c55e',
                borderRadius: 1,
                boxSizing: 'border-box',
                pointerEvents: 'none',
              }}
            />
          )}
        </Box>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
        Drag to draw a box, then name and save the object.
      </Typography>

      <Dialog open={showSaveDialog} onClose={handleCancelSave} maxWidth="sm" fullWidth>
        <DialogTitle>Save object annotation</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Object name"
            fullWidth
            value={objectName}
            onChange={(e) => setObjectName(e.target.value)}
            placeholder="e.g. My coffee mug"
          />
          <TextField
            margin="dense"
            label="Description (optional)"
            fullWidth
            multiline
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g. Blue ceramic mug with logo"
          />
          {error && (
            <Typography color="error" sx={{ mt: 1 }}>
              {error}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelSave}>Cancel</Button>
          <Button onClick={handleSaveAnnotation} variant="contained" disabled={!objectName.trim() || saving}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ObjectAnnotationCanvas;
