import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useQuery } from 'react-query';
import {
  Box,
  Button,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import { ChevronLeft, ChevronRight, Shuffle, Casino } from '@mui/icons-material';
import apiService from '../../services/apiService';
import { AuthDocumentFileImage } from './homeDashboardAuthImage';
import { flattenFolderTree } from './homeDashboardUtils';

const IMAGE_EXT = /\.(jpe?g|png|gif|webp|bmp|tiff?|svg)$/i;
/** Safety cap on distinct folders visited when include_subfolders is on. */
const MAX_FOLDERS_TO_TRAVERSE = 2000;

export function isImageDocument(doc) {
  if (!doc?.document_id) return false;
  const t = (doc.doc_type || '').toLowerCase();
  if (t === 'image') return true;
  const name = doc.filename || doc.title || '';
  return IMAGE_EXT.test(name);
}

async function collectImagesInOneFolder(folderId, maxScan, collected, pageSize) {
  let offset = 0;
  let subfolders = null;
  while (collected.length < maxScan) {
    const res = await apiService.getFolderContents(folderId, pageSize, offset);
    const docs = res?.documents || [];
    if (offset === 0) {
      subfolders = res?.subfolders || [];
    }
    for (const d of docs) {
      if (isImageDocument(d)) collected.push(d);
      if (collected.length >= maxScan) break;
    }
    if (docs.length < pageSize) break;
    offset += pageSize;
  }
  return subfolders;
}

async function fetchImageDocumentsFromFolder(folderId, maxScan, includeSubfolders) {
  const pageSize = 250;
  const collected = [];
  if (!includeSubfolders) {
    await collectImagesInOneFolder(folderId, maxScan, collected, pageSize);
    return collected;
  }
  const queue = [folderId];
  const seenFolders = new Set();
  while (queue.length > 0 && collected.length < maxScan && seenFolders.size < MAX_FOLDERS_TO_TRAVERSE) {
    const fid = queue.shift();
    if (!fid || seenFolders.has(fid)) continue;
    seenFolders.add(fid);
    const subfolders = await collectImagesInOneFolder(fid, maxScan, collected, pageSize);
    for (const s of subfolders || []) {
      const sid = s?.folder_id;
      if (sid && !seenFolders.has(sid)) queue.push(sid);
    }
  }
  const seen = new Set();
  return collected.filter((d) => {
    const id = d?.document_id;
    if (!id || seen.has(id)) return false;
    seen.add(id);
    return true;
  });
}

function shuffleIds(ids) {
  const arr = [...ids];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function FolderImageSlideshowBlock({ config, navigate }) {
  const folderId = (config?.folder_id || '').trim();
  const maxScan = Math.min(5000, Math.max(50, config?.scan_limit ?? 500));
  const includeSubfolders = Boolean(config?.include_subfolders);

  const { data: imageDocs = [], isLoading, error, refetch } = useQuery(
    ['homeDashboardFolderImages', folderId, maxScan, includeSubfolders],
    () => fetchImageDocumentsFromFolder(folderId, maxScan, includeSubfolders),
    {
      enabled: Boolean(folderId),
      staleTime: 2 * 60 * 1000,
    }
  );

  const ids = useMemo(() => imageDocs.map((d) => d.document_id), [imageDocs]);

  const [order, setOrder] = useState([]);
  const [cursor, setCursor] = useState(0);

  const idsKey = useMemo(() => [...ids].sort().join(','), [ids]);

  useEffect(() => {
    if (!ids.length) {
      setOrder([]);
      setCursor(0);
      return;
    }
    setOrder(shuffleIds(ids));
    setCursor(0);
  }, [folderId, idsKey]);

  const n = order.length;
  const currentId = n ? order[cursor] : null;
  const currentMeta = useMemo(
    () => imageDocs.find((d) => d.document_id === currentId),
    [imageDocs, currentId]
  );

  const goPrev = useCallback(() => {
    if (n < 2) return;
    setCursor((c) => (c - 1 + n) % n);
  }, [n]);

  const goNext = useCallback(() => {
    if (n < 2) return;
    setCursor((c) => (c + 1) % n);
  }, [n]);

  const reshuffle = useCallback(() => {
    if (!ids.length) return;
    setOrder(shuffleIds(ids));
    setCursor(0);
  }, [ids]);

  const randomJump = useCallback(() => {
    if (n < 2) return;
    let next = cursor;
    let guard = 0;
    while (next === cursor && guard < 32) {
      next = Math.floor(Math.random() * n);
      guard += 1;
    }
    setCursor(next);
  }, [n, cursor]);

  const openInDocuments = useCallback(() => {
    if (!currentId) return;
    const title = currentMeta?.title || currentMeta?.filename || 'Image';
    navigate(
      `/documents?document=${encodeURIComponent(currentId)}&doc_title=${encodeURIComponent(title)}`
    );
  }, [currentId, currentMeta, navigate]);

  if (!folderId) {
    return (
      <Typography variant="body2" color="text.secondary">
        Choose a folder in Edit layout.
      </Typography>
    );
  }

  if (isLoading) {
    return (
      <Typography variant="body2" color="text.secondary">
        Loading images…
      </Typography>
    );
  }

  if (error) {
    return (
      <Typography color="error" variant="body2">
        Could not load folder contents.
      </Typography>
    );
  }

  if (!n) {
    return (
      <Typography variant="body2" color="text.secondary">
        No image files in this folder
        {includeSubfolders ? ' or its subfolders' : ''} (supports common image types and documents with{' '}
        <code>doc_type=image</code>).
      </Typography>
    );
  }

  return (
    <Stack spacing={1.5}>
      <Box sx={{ position: 'relative' }}>
        <AuthDocumentFileImage
          documentId={currentId}
          alt={currentMeta?.title || currentMeta?.filename || 'Image'}
        />
      </Box>
      <Stack direction="row" flexWrap="wrap" alignItems="center" gap={0.5}>
        <Tooltip title="Previous image (shuffled order)">
          <span>
            <Button size="small" variant="outlined" onClick={goPrev} disabled={n < 2} startIcon={<ChevronLeft />}>
              Back
            </Button>
          </span>
        </Tooltip>
        <Tooltip title="Next image (shuffled order)">
          <span>
            <Button size="small" variant="outlined" onClick={goNext} disabled={n < 2} endIcon={<ChevronRight />}>
              Next
            </Button>
          </span>
        </Tooltip>
        <Tooltip title="Pick a different random image from the folder">
          <span>
            <Button size="small" variant="outlined" onClick={randomJump} disabled={n < 2} startIcon={<Casino />}>
              Random
            </Button>
          </span>
        </Tooltip>
        <Tooltip title="Shuffle order again (starts from first of new order)">
          <span>
            <Button size="small" onClick={reshuffle} startIcon={<Shuffle />}>
              Reshuffle
            </Button>
          </span>
        </Tooltip>
        <Button size="small" onClick={() => refetch()}>
          Refresh
        </Button>
        <Button size="small" onClick={openInDocuments} disabled={!currentId}>
          Open in Documents
        </Button>
        <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
          {cursor + 1} / {n}
        </Typography>
      </Stack>
      {currentMeta?.filename ? (
        <Typography variant="caption" color="text.secondary" noWrap title={currentMeta.filename}>
          {currentMeta.filename}
        </Typography>
      ) : null}
    </Stack>
  );
}

export function FolderImageSlideshowWidgetEditor({ widget, onChange }) {
  const { data: treeRes, isLoading } = useQuery(
    ['homeDashboardFolderTreeImageWidget'],
    () => apiService.getFolderTree('user', false),
    { staleTime: 5 * 60 * 1000 }
  );

  const flat = useMemo(
    () => flattenFolderTree(treeRes?.folders || []),
    [treeRes?.folders]
  );

  const updateConfig = (patch) => {
    onChange({ ...widget, config: { ...widget.config, ...patch } });
  };

  return (
    <Stack spacing={2}>
      {isLoading ? (
        <Typography variant="body2" color="text.secondary">
          Loading folders…
        </Typography>
      ) : null}
      <FormControl size="small" fullWidth>
        <InputLabel>Folder</InputLabel>
        <Select
          label="Folder"
          value={widget.config?.folder_id || ''}
          onChange={(e) => updateConfig({ folder_id: e.target.value || null })}
        >
          <MenuItem value="">
            <em>Select folder</em>
          </MenuItem>
          {flat.map((f) => (
            <MenuItem key={f.folder_id} value={f.folder_id}>
              {f.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={Boolean(widget.config?.include_subfolders)}
            onChange={(e) => updateConfig({ include_subfolders: e.target.checked })}
          />
        }
        label="Include images in subfolders"
      />
      <Typography variant="caption" color="text.secondary" sx={{ mt: -1 }}>
        When on, walks nested folders (breadth-first) until the max image count or folder visit limit is
        reached.
      </Typography>
      <TextField
        size="small"
        type="number"
        label="Max images to collect"
        helperText="Stops after this many image files (50–5000). Applies across the folder and subfolders when enabled above."
        inputProps={{ min: 50, max: 5000 }}
        value={widget.config?.scan_limit ?? 500}
        onChange={(e) =>
          updateConfig({
            scan_limit: Math.min(5000, Math.max(50, parseInt(e.target.value, 10) || 500)),
          })
        }
      />
    </Stack>
  );
}
