/**
 * Agent line reference files/folders (context injection for playbooks).
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Chip,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  Folder as FolderIcon,
  Description as FileIcon,
  Delete as DeleteIcon,
  Lock as LockIcon,
} from '@mui/icons-material';
import ReferencePicker from './ReferencePicker';

const defaultConfig = () => ({
  folders: [],
  documents: [],
  load_strategy: 'full',
});

function normalizeIncoming(raw) {
  if (!raw || typeof raw !== 'object') return defaultConfig();
  return {
    folders: Array.isArray(raw.folders) ? raw.folders : [],
    documents: Array.isArray(raw.documents) ? raw.documents : [],
    load_strategy: raw.load_strategy === 'metadata_first' ? 'metadata_first' : 'full',
  };
}

export default function LineReferenceSection({ team, onSave, saving }) {
  const [config, setConfig] = useState(defaultConfig);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerMode, setPickerMode] = useState('folder');

  useEffect(() => {
    if (team?.reference_config) {
      setConfig(normalizeIncoming(team.reference_config));
    } else {
      setConfig(defaultConfig());
    }
  }, [team]);

  const addFolderEntry = (meta) => {
    const { folder_id, name, collection_type, team_id } = meta;
    const access = collection_type === 'global' ? 'read' : 'read_write';
    setConfig((c) => {
      if (c.folders.some((f) => f.folder_id === folder_id)) return c;
      return {
        ...c,
        folders: [
          ...c.folders,
          {
            folder_id,
            name: name || folder_id,
            collection_type,
            team_id: team_id || undefined,
            access,
          },
        ],
      };
    });
  };

  const addDocumentEntry = (meta) => {
    const { document_id, title, collection_type, team_id } = meta;
    const access = collection_type === 'global' ? 'read' : 'read_write';
    setConfig((c) => {
      if (c.documents.some((d) => d.document_id === document_id)) return c;
      return {
        ...c,
        documents: [
          ...c.documents,
          {
            document_id,
            title: title || document_id,
            collection_type,
            team_id: team_id || undefined,
            access,
          },
        ],
      };
    });
  };

  const removeFolder = (folderId) => {
    setConfig((c) => ({
      ...c,
      folders: c.folders.filter((f) => f.folder_id !== folderId),
    }));
  };

  const removeDocument = (documentId) => {
    setConfig((c) => ({
      ...c,
      documents: c.documents.filter((d) => d.document_id !== documentId),
    }));
  };

  const setFolderAccess = (folderId, access) => {
    setConfig((c) => ({
      ...c,
      folders: c.folders.map((f) =>
        f.folder_id === folderId ? { ...f, access } : f
      ),
    }));
  };

  const setDocumentAccess = (documentId, access) => {
    setConfig((c) => ({
      ...c,
      documents: c.documents.map((d) =>
        d.document_id === documentId ? { ...d, access } : d
      ),
    }));
  };

  const collectionLabel = (ct) => {
    if (ct === 'global') return 'Global';
    if (ct === 'team') return 'Team';
    return 'My Docs';
  };

  return (
    <Box component="form" onSubmit={(e) => { e.preventDefault(); onSave(config); }}>
      <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
        Injected into playbook prompts as {'{line_refs}'} and per-entry {'{line_ref_*}'} variables. Does not restrict tool access.
      </Typography>
      <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <Button
          size="small"
          variant="outlined"
          startIcon={<FolderIcon />}
          onClick={() => { setPickerMode('folder'); setPickerOpen(true); }}
        >
          Add folder
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<FileIcon />}
          onClick={() => { setPickerMode('file'); setPickerOpen(true); }}
        >
          Add file
        </Button>
        <FormControl size="small" sx={{ minWidth: 180 }}>
          <InputLabel id="line-ref-load-strategy">Load strategy</InputLabel>
          <Select
            labelId="line-ref-load-strategy"
            label="Load strategy"
            value={config.load_strategy}
            onChange={(e) => setConfig((c) => ({ ...c, load_strategy: e.target.value }))}
          >
            <MenuItem value="full">Full content</MenuItem>
            <MenuItem value="metadata_first">Metadata first</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {config.folders.map((f) => (
        <Box
          key={f.folder_id}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            py: 0.5,
            flexWrap: 'wrap',
          }}
        >
          <FolderIcon fontSize="small" color="action" />
          <Typography variant="body2" sx={{ flex: 1, minWidth: 120 }}>
            {f.name}
          </Typography>
          <Chip size="small" label={collectionLabel(f.collection_type)} />
          {f.collection_type !== 'global' ? (
            <ToggleButtonGroup
              size="small"
              value={f.access === 'read_write' ? 'rw' : 'r'}
              exclusive
              onChange={(_, v) => v && setFolderAccess(f.folder_id, v === 'rw' ? 'read_write' : 'read')}
            >
              <ToggleButton value="r">Read</ToggleButton>
              <ToggleButton value="rw">R/W</ToggleButton>
            </ToggleButtonGroup>
          ) : (
            <Chip size="small" icon={<LockIcon sx={{ fontSize: 14 }} />} label="Read only" variant="outlined" />
          )}
          <IconButton size="small" aria-label="Remove" onClick={() => removeFolder(f.folder_id)}>
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Box>
      ))}

      {config.documents.map((d) => (
        <Box
          key={d.document_id}
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            py: 0.5,
            flexWrap: 'wrap',
          }}
        >
          <FileIcon fontSize="small" color="action" />
          <Typography variant="body2" sx={{ flex: 1, minWidth: 120 }}>
            {d.title}
          </Typography>
          <Chip size="small" label={collectionLabel(d.collection_type)} />
          {d.collection_type !== 'global' ? (
            <ToggleButtonGroup
              size="small"
              value={d.access === 'read_write' ? 'rw' : 'r'}
              exclusive
              onChange={(_, v) => v && setDocumentAccess(d.document_id, v === 'rw' ? 'read_write' : 'read')}
            >
              <ToggleButton value="r">Read</ToggleButton>
              <ToggleButton value="rw">R/W</ToggleButton>
            </ToggleButtonGroup>
          ) : (
            <Chip size="small" icon={<LockIcon sx={{ fontSize: 14 }} />} label="Read only" variant="outlined" />
          )}
          <IconButton size="small" aria-label="Remove" onClick={() => removeDocument(d.document_id)}>
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Box>
      ))}

      {config.folders.length === 0 && config.documents.length === 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ py: 1 }}>
          No references yet. Add folders or files from My Documents, Global Documents, or Teams.
        </Typography>
      )}

      <Box sx={{ mt: 2 }}>
        <Button type="submit" variant="contained" size="small" disabled={saving}>
          Save references
        </Button>
      </Box>

      <ReferencePicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        mode={pickerMode}
        onConfirm={(payload) => {
          if (payload.kind === 'folder') addFolderEntry(payload);
          else addDocumentEntry(payload);
        }}
      />
    </Box>
  );
}
