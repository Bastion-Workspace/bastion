/**
 * Folder Watch section: expandable folder tree with watch toggles per folder.
 * Persists to profile.watch_config.folder_watches; backend syncs to agent_folder_watches on save.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  List,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Switch,
  TextField,
  CircularProgress,
  Collapse,
  IconButton,
} from '@mui/material';
import FolderIcon from '@mui/icons-material/Folder';
import FolderOpenIcon from '@mui/icons-material/FolderOpen';
import ExpandMore from '@mui/icons-material/ExpandMore';
import ChevronRight from '@mui/icons-material/ChevronRight';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';
import { useAuth } from '../../contexts/AuthContext';

const VIRTUAL_ROOT_IDS = ['my_documents_root', 'global_documents_root'];

function FolderTreeItem({ folder, depth, watchMap, onToggle, onFilterChange, expanded, onExpand }) {
  const isVirtual = VIRTUAL_ROOT_IDS.includes(folder.folder_id);
  const children = folder.children || [];
  const hasChildren = children.length > 0;
  const isExpanded = expanded.has(folder.folder_id);
  const watch = watchMap[String(folder.folder_id)];
  const watching = !!watch;
  const activeChildCount = countActiveWatches(folder, watchMap);

  return (
    <>
      <ListItemButton
        disableRipple={isVirtual}
        sx={{
          pl: isVirtual ? 1 : 1 + depth * 2.5,
          py: 0.5,
          minHeight: 36,
          '&:hover': { bgcolor: 'action.hover' },
        }}
        onClick={() => hasChildren && onExpand(folder.folder_id)}
      >
        {hasChildren ? (
          <IconButton size="small" sx={{ mr: 0.5, p: 0.25 }} onClick={(e) => { e.stopPropagation(); onExpand(folder.folder_id); }}>
            {isExpanded ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
          </IconButton>
        ) : (
          <Box sx={{ width: 28 }} />
        )}
        <ListItemIcon sx={{ minWidth: 28 }}>
          {isExpanded ? <FolderOpenIcon fontSize="small" color="action" /> : <FolderIcon fontSize="small" color="action" />}
        </ListItemIcon>
        <ListItemText
          primary={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="body2" sx={{ fontWeight: isVirtual ? 600 : 400 }}>
                {folder.name}
              </Typography>
              {activeChildCount > 0 && !isExpanded && (
                <Typography variant="caption" color="primary.main" sx={{ fontWeight: 500 }}>
                  ({activeChildCount} watched)
                </Typography>
              )}
            </Box>
          }
          sx={{ my: 0 }}
        />
        {!isVirtual && (
          <Switch
            size="small"
            checked={watching}
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => onToggle(folder.folder_id, e.target.checked)}
          />
        )}
      </ListItemButton>
      {!isVirtual && watching && (
        <Box sx={{ pl: (isVirtual ? 1 : 1 + depth * 2.5) + 7, pr: 2, pb: 0.5 }}>
          <TextField
            size="small"
            label="File types (optional)"
            placeholder="pdf, docx, md"
            value={watch?.file_type_filter ?? ''}
            onChange={(e) => onFilterChange(folder.folder_id, e.target.value)}
            sx={{ width: 220 }}
          />
        </Box>
      )}
      {hasChildren && (
        <Collapse in={isExpanded} timeout="auto" unmountOnExit>
          <List disablePadding>
            {children.map((child) => (
              <FolderTreeItem
                key={child.folder_id}
                folder={child}
                depth={isVirtual ? depth : depth + 1}
                watchMap={watchMap}
                onToggle={onToggle}
                onFilterChange={onFilterChange}
                expanded={expanded}
                onExpand={onExpand}
              />
            ))}
          </List>
        </Collapse>
      )}
    </>
  );
}

function countActiveWatches(folder, watchMap) {
  let count = watchMap[String(folder.folder_id)] ? 1 : 0;
  for (const child of folder.children || []) {
    count += countActiveWatches(child, watchMap);
  }
  return count;
}

export default function FolderWatchSection({ profile, onChange, compact }) {
  const { user } = useAuth();
  const { data: treeData, isLoading } = useQuery(
    ['folders', 'tree', user?.user_id, user?.role],
    () => apiService.getFolderTree('user'),
    { staleTime: 60 * 1000, enabled: !!user }
  );

  const [expanded, setExpanded] = useState(new Set());

  const folders = treeData?.folders ?? [];

  if (!profile) return null;

  const watchConfig = profile.watch_config || {};
  const folderWatches = watchConfig.folder_watches || [];

  const watchMap = {};
  for (const w of folderWatches) {
    watchMap[String(w.folder_id)] = w;
  }

  const setFolderWatches = (next) => {
    onChange({
      ...profile,
      watch_config: { ...watchConfig, folder_watches: next },
    });
  };

  const handleToggle = (folderId, enabled) => {
    const rest = folderWatches.filter((w) => String(w.folder_id) !== String(folderId));
    if (!enabled) {
      setFolderWatches(rest);
      return;
    }
    const existing = watchMap[String(folderId)];
    setFolderWatches([
      ...rest,
      { folder_id: folderId, file_type_filter: existing?.file_type_filter ?? '' },
    ]);
  };

  const handleFilterChange = (folderId, value) => {
    setFolderWatches(
      folderWatches.map((w) =>
        String(w.folder_id) === String(folderId)
          ? { ...w, file_type_filter: value }
          : w
      )
    );
  };

  const handleExpand = (folderId) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(folderId)) next.delete(folderId);
      else next.add(folderId);
      return next;
    });
  };

  const activeCount = folderWatches.length;

  return (
    <Box>
      {!compact && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Trigger when a new file is added to a watched folder. Optionally filter by file type.
        </Typography>
      )}
      {isLoading ? (
        <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
          <CircularProgress size={24} />
        </Box>
      ) : folders.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No folders found. Create folders in Documents first.
        </Typography>
      ) : (
        <List disablePadding dense>
          {folders.map((folder) => (
            <FolderTreeItem
              key={folder.folder_id}
              folder={folder}
              depth={0}
              watchMap={watchMap}
              onToggle={handleToggle}
              onFilterChange={handleFilterChange}
              expanded={expanded}
              onExpand={handleExpand}
            />
          ))}
        </List>
      )}
      {!compact && activeCount > 0 && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          {activeCount} folder{activeCount !== 1 ? 's' : ''} watched
        </Typography>
      )}
    </Box>
  );
}

export function folderWatchSummary(profile) {
  const watches = profile?.watch_config?.folder_watches || [];
  return watches.length ? `${watches.length} folder${watches.length !== 1 ? 's' : ''}` : '';
}
