/**
 * Dialog to generate and save LLM descriptions for all images in a folder.
 * Processes images one-by-one; shows progress and allows cancel.
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  Alert,
  Box
} from '@mui/material';
import { AutoAwesome } from '@mui/icons-material';
import apiService from '../../services/apiService';

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.gif'];

function isImageDocument(doc) {
  const filename = (doc?.filename || '').toLowerCase();
  if (!filename) return false;
  if (filename.endsWith('.metadata.json')) return false;
  return IMAGE_EXTENSIONS.some(ext => filename.endsWith(ext));
}

const DescribeFolderImagesDialog = ({ open, onClose, folderId, folderName }) => {
  const [imageDocs, setImageDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentFilename, setCurrentFilename] = useState('');
  const [doneCount, setDoneCount] = useState(0);
  const [failCount, setFailCount] = useState(0);
  const [complete, setComplete] = useState(false);
  const abortedRef = useRef(false);

  useEffect(() => {
    if (!open || !folderId) return;
    setLoading(true);
    setError(null);
    setImageDocs([]);
    setRunning(false);
    setCurrentIndex(0);
    setDoneCount(0);
    setFailCount(0);
    setComplete(false);
    abortedRef.current = false;

    apiService.getFolderContents(folderId)
      .then((data) => {
        const docs = data?.documents || [];
        const images = docs.filter(isImageDocument);
        setImageDocs(images);
        setError(images.length === 0 ? 'No image files found in this folder.' : null);
      })
      .catch((err) => {
        setError(err?.response?.data?.detail || err?.message || 'Failed to load folder contents.');
      })
      .finally(() => setLoading(false));
  }, [open, folderId]);

  const handleStart = async () => {
    if (imageDocs.length === 0) return;
    setRunning(true);
    setError(null);
    setFailCount(0);
    setDoneCount(0);
    abortedRef.current = false;

    let succeeded = 0;
    let failed = 0;

    for (let i = 0; i < imageDocs.length; i++) {
      if (abortedRef.current) break;

      const doc = imageDocs[i];
      const documentId = doc.document_id;
      const filename = doc.filename || '';

      setCurrentIndex(i + 1);
      setCurrentFilename(filename);

      try {
        let existing = {};
        try {
          const metaRes = await apiService.get(`/api/documents/${documentId}/image-metadata`);
          if (metaRes?.metadata) existing = metaRes.metadata;
        } catch {
          // No existing metadata
        }

        const describeRes = await apiService.post(`/api/documents/${documentId}/describe-image`);
        const description = (describeRes?.description || '').trim();
        const modelUsed = describeRes?.model_used;
        const confidence = describeRes?.confidence;

        const payload = {
          ...existing,
          content: description,
          llm_metadata: modelUsed
            ? {
                model: modelUsed,
                timestamp: new Date().toISOString(),
                confidence: typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0.92
              }
            : (existing.llm_metadata || null)
        };

        Object.keys(payload).forEach(key => {
          if (payload[key] === null || payload[key] === '') delete payload[key];
        });

        await apiService.post(`/api/documents/${documentId}/image-metadata`, payload);
        succeeded++;
        setDoneCount(succeeded);
      } catch (err) {
        failed++;
        setFailCount(failed);
        console.error(`Describe/save failed for ${filename}:`, err);
      }
    }

    setRunning(false);
    setComplete(true);
    setCurrentFilename('');
  };

  const handleCancel = () => {
    abortedRef.current = true;
  };

  const handleClose = () => {
    if (!running) {
      onClose?.();
    }
  };

  const total = imageDocs.length;
  const progress = total > 0 ? (currentIndex / total) * 100 : 0;

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <AutoAwesome fontSize="small" />
          Describe folder images with LLM
        </Box>
      </DialogTitle>
      <DialogContent>
        {folderName && (
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Folder: {folderName}
          </Typography>
        )}

        {loading && (
          <Typography color="text.secondary">Loading folder contents…</Typography>
        )}

        {!loading && error && (
          <Alert severity="warning" sx={{ mt: 1 }}>{error}</Alert>
        )}

        {!loading && !error && imageDocs.length > 0 && !complete && (
          <>
            <Typography variant="body2" sx={{ mb: 1 }}>
              {imageDocs.length} image{imageDocs.length !== 1 ? 's' : ''} in this folder. Generate LLM descriptions and save to metadata one by one?
            </Typography>
            {running && (
              <>
                <LinearProgress variant="determinate" value={progress} sx={{ my: 2 }} />
                <Typography variant="body2" color="text.secondary">
                  {currentIndex} of {total}: {currentFilename}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Done: {doneCount} · Failed: {failCount}
                </Typography>
              </>
            )}
          </>
        )}

        {!loading && complete && (
          <Typography variant="body2" sx={{ mt: 1 }}>
            Finished. Described {doneCount} image{doneCount !== 1 ? 's' : ''}.
            {failCount > 0 && ` ${failCount} failed.`}
          </Typography>
        )}

        {!loading && !error && imageDocs.length > 0 && (
          <List dense sx={{ maxHeight: 200, overflow: 'auto', mt: 1 }}>
            {imageDocs.slice(0, 20).map((doc) => (
              <ListItem key={doc.document_id} disablePadding>
                <ListItemText primary={doc.filename || doc.document_id} primaryTypographyProps={{ variant: 'body2' }} />
              </ListItem>
            ))}
            {imageDocs.length > 20 && (
              <ListItem disablePadding>
                <ListItemText primary={`… and ${imageDocs.length - 20} more`} primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }} />
              </ListItem>
            )}
          </List>
        )}
      </DialogContent>
      <DialogActions>
        {!complete && running && (
          <Button onClick={handleCancel} color="secondary">Cancel</Button>
        )}
        {!complete && !running && imageDocs.length > 0 && (
          <Button onClick={handleStart} variant="contained" startIcon={<AutoAwesome />}>Start</Button>
        )}
        <Button onClick={handleClose}>{complete ? 'Close' : 'Cancel'}</Button>
      </DialogActions>
    </Dialog>
  );
};

export default DescribeFolderImagesDialog;
