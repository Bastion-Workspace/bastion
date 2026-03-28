/**
 * Modal to pick a folder or a file from My Documents, Global Documents, or Teams.
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Tabs,
  Tab,
  Box,
  Typography,
  List,
  ListItemButton,
  ListItemText,
  CircularProgress,
  Radio,
  RadioGroup,
  FormControlLabel,
} from '@mui/material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

const TAB_USER = 0;
const TAB_GLOBAL = 1;
const TAB_TEAM = 2;

function tabToCollection(tabIndex) {
  if (tabIndex === TAB_GLOBAL) return 'global';
  if (tabIndex === TAB_TEAM) return 'team';
  return 'user';
}

function FolderTreeList({
  folders,
  depth,
  collectionType,
  onPickFolder,
  selectedFolderId,
  mode,
}) {
  if (!folders || !folders.length) return null;
  return (
    <>
      {folders.map((f) => (
        <Box key={f.folder_id}>
          <ListItemButton
            sx={{ pl: 1 + depth * 2 }}
            selected={mode === 'folder' && selectedFolderId === f.folder_id}
            onClick={() =>
              onPickFolder({
                folder_id: f.folder_id,
                name: f.name,
                collection_type: f.collection_type || collectionType,
                team_id: f.team_id || null,
              })
            }
          >
            <ListItemText primary={`📁 ${f.name}`} secondary={f.document_count != null ? `${f.document_count} docs` : null} />
          </ListItemButton>
          {f.children && f.children.length > 0 && (
            <FolderTreeList
              folders={f.children}
              depth={depth + 1}
              collectionType={collectionType}
              onPickFolder={onPickFolder}
              selectedFolderId={selectedFolderId}
              mode={mode}
            />
          )}
        </Box>
      ))}
    </>
  );
}

export default function ReferencePicker({ open, onClose, mode, onConfirm }) {
  const [tab, setTab] = useState(TAB_USER);
  const collectionType = tabToCollection(tab);

  const [selectedFolder, setSelectedFolder] = useState(null);
  const [fileBrowseFolder, setFileBrowseFolder] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState('');

  const { data: treeData, isLoading: treeLoading } = useQuery(
    ['referencePickerTree', collectionType, open],
    () => apiService.folders.getFolderTree(collectionType, false),
    { enabled: open, staleTime: 30_000 }
  );

  const folders = treeData?.folders || [];

  const { data: contentsData, isLoading: contentsLoading } = useQuery(
    ['referencePickerContents', fileBrowseFolder?.folder_id, open],
    () => apiService.folders.getFolderContents(fileBrowseFolder.folder_id, 250, 0),
    { enabled: open && mode === 'file' && !!fileBrowseFolder?.folder_id, staleTime: 10_000 }
  );

  const documents = contentsData?.documents || [];

  React.useEffect(() => {
    if (!open) {
      setTab(0);
      setSelectedFolder(null);
      setFileBrowseFolder(null);
      setSelectedDocId('');
    }
  }, [open]);

  const handlePickFolder = (meta) => {
    setSelectedFolder(meta);
    if (mode === 'file') {
      setFileBrowseFolder(meta);
      setSelectedDocId('');
    }
  };

  const canConfirm =
    mode === 'folder'
      ? selectedFolder?.folder_id
      : selectedDocId && fileBrowseFolder;

  const handleConfirm = () => {
    if (mode === 'folder' && selectedFolder) {
      onConfirm({ kind: 'folder', ...selectedFolder });
      onClose();
      return;
    }
    if (mode === 'file' && fileBrowseFolder && selectedDocId) {
      const doc = documents.find((d) => d.document_id === selectedDocId);
      onConfirm({
        kind: 'document',
        document_id: selectedDocId,
        title: doc?.title || doc?.filename || selectedDocId,
        collection_type: doc?.collection_type || fileBrowseFolder.collection_type || collectionType,
        team_id: fileBrowseFolder.team_id || doc?.team_id || null,
      });
      onClose();
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{mode === 'folder' ? 'Choose folder' : 'Choose file'}</DialogTitle>
      <DialogContent dividers>
        <Tabs value={tab} onChange={(_, v) => { setTab(v); setSelectedFolder(null); setFileBrowseFolder(null); setSelectedDocId(''); }} sx={{ mb: 1 }}>
          <Tab label="My Documents" />
          <Tab label="Global" />
          <Tab label="Teams" />
        </Tabs>
        {treeLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
            <CircularProgress size={28} />
          </Box>
        )}
        {!treeLoading && (
          <List dense disablePadding>
            <FolderTreeList
              folders={folders}
              depth={0}
              collectionType={collectionType}
              onPickFolder={handlePickFolder}
              selectedFolderId={mode === 'folder' ? selectedFolder?.folder_id : null}
              mode={mode}
            />
          </List>
        )}
        {mode === 'file' && fileBrowseFolder && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Files in: {fileBrowseFolder.name}
            </Typography>
            {contentsLoading && <CircularProgress size={24} sx={{ mt: 1 }} />}
            {!contentsLoading && documents.length === 0 && (
              <Typography variant="body2" color="text.secondary">No files in this folder.</Typography>
            )}
            {!contentsLoading && documents.length > 0 && (
              <RadioGroup value={selectedDocId} onChange={(e) => setSelectedDocId(e.target.value)}>
                {documents.map((d) => (
                  <FormControlLabel
                    key={d.document_id}
                    value={d.document_id}
                    control={<Radio size="small" />}
                    label={`${d.filename || d.title || d.document_id}`}
                  />
                ))}
              </RadioGroup>
            )}
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleConfirm} disabled={!canConfirm}>
          Add
        </Button>
      </DialogActions>
    </Dialog>
  );
}
