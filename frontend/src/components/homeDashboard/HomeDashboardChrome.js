import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  IconButton,
  InputLabel,
  Menu,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import { Add, Edit, MoreVert, Save } from '@mui/icons-material';

/**
 * Toolbar: dashboard picker, layout edit controls, dashboard CRUD menu, add-widget entry.
 */
export default function HomeDashboardChrome({
  dashboards,
  dashboardId,
  onSelectDashboard,
  editMode,
  onStartEdit,
  onCancelEdit,
  onSaveLayout,
  saveLayoutLoading,
  onOpenCreate,
  onOpenRename,
  onSetDefault,
  onOpenDelete,
  canDeleteDashboard,
  isDefaultDashboard,
  onAddWidgetClick,
  addMenuAnchor,
  onCloseAddMenu,
  onPickWidgetType,
  widgetTypeOptions,
}) {
  const [dashMenuAnchor, setDashMenuAnchor] = useState(null);

  const current = dashboards?.find((d) => d.id === dashboardId);

  return (
    <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" sx={{ mb: 2 }}>
      <Typography variant="h5" component="h1" sx={{ flexGrow: 1, minWidth: 120 }}>
        Home
      </Typography>

      {dashboards?.length ? (
        <FormControl size="small" sx={{ minWidth: 180, maxWidth: 280 }}>
          <InputLabel id="home-dash-select-label">Dashboard</InputLabel>
          <Select
            labelId="home-dash-select-label"
            label="Dashboard"
            value={dashboardId || ''}
            onChange={(e) => onSelectDashboard(e.target.value)}
          >
            {dashboards.map((d) => (
              <MenuItem key={d.id} value={d.id}>
                {d.name}
                {d.is_default ? ' (default)' : ''}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      ) : null}

      <Button size="small" onClick={onOpenCreate} disabled={!dashboardId}>
        New dashboard
      </Button>

      <IconButton
        aria-label="Dashboard actions"
        onClick={(e) => setDashMenuAnchor(e.currentTarget)}
        size="small"
        disabled={!dashboardId}
      >
        <MoreVert />
      </IconButton>
      <Menu
        anchorEl={dashMenuAnchor}
        open={Boolean(dashMenuAnchor)}
        onClose={() => setDashMenuAnchor(null)}
      >
        <MenuItem
          onClick={() => {
            setDashMenuAnchor(null);
            onOpenCreate();
          }}
        >
          New dashboard
        </MenuItem>
        <MenuItem
          onClick={() => {
            setDashMenuAnchor(null);
            onOpenRename();
          }}
          disabled={!current}
        >
          Rename
        </MenuItem>
        <MenuItem
          onClick={() => {
            setDashMenuAnchor(null);
            onSetDefault();
          }}
          disabled={!current || isDefaultDashboard}
        >
          Set as default
        </MenuItem>
        <MenuItem
          onClick={() => {
            setDashMenuAnchor(null);
            onOpenDelete();
          }}
          disabled={!canDeleteDashboard}
        >
          Delete dashboard
        </MenuItem>
      </Menu>

      {!editMode ? (
        <Button startIcon={<Edit />} variant="outlined" onClick={onStartEdit} disabled={!dashboardId}>
          Edit layout
        </Button>
      ) : (
        <>
          <Button variant="outlined" startIcon={<Add />} onClick={onAddWidgetClick}>
            Add widget
          </Button>
          <Menu anchorEl={addMenuAnchor} open={Boolean(addMenuAnchor)} onClose={onCloseAddMenu}>
            {widgetTypeOptions.map((wt) => (
              <MenuItem
                key={wt.type}
                onClick={() => {
                  onPickWidgetType(wt.type);
                  onCloseAddMenu();
                }}
              >
                {wt.label}
              </MenuItem>
            ))}
          </Menu>
          <Button
            variant="contained"
            startIcon={<Save />}
            onClick={onSaveLayout}
            disabled={saveLayoutLoading}
          >
            Save
          </Button>
          <Button variant="text" onClick={onCancelEdit} disabled={saveLayoutLoading}>
            Cancel
          </Button>
        </>
      )}
    </Stack>
  );
}

export function HomeDashboardDialogs({
  createOpen,
  onCloseCreate,
  onSubmitCreate,
  createLoading,
  duplicateByDefault,
  onDuplicateByDefaultChange,
  renameOpen,
  onCloseRename,
  renameName,
  onRenameNameChange,
  onSubmitRename,
  renameLoading,
  deleteOpen,
  onCloseDelete,
  onConfirmDelete,
  deleteLoading,
  deleteTargetName,
}) {
  const [newName, setNewName] = useState('');

  useEffect(() => {
    if (createOpen) setNewName('');
  }, [createOpen]);

  return (
    <>
      <Dialog open={createOpen} onClose={onCloseCreate} maxWidth="xs" fullWidth>
        <DialogTitle>New dashboard</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <Box sx={{ mt: 1 }}>
            <label>
              <input
                type="checkbox"
                checked={duplicateByDefault}
                onChange={(e) => onDuplicateByDefaultChange(e.target.checked)}
              />{' '}
              Copy layout from current dashboard
            </label>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={onCloseCreate}>Cancel</Button>
          <Button
            onClick={() => onSubmitCreate(newName, duplicateByDefault)}
            variant="contained"
            disabled={createLoading}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={renameOpen} onClose={onCloseRename} maxWidth="xs" fullWidth>
        <DialogTitle>Rename dashboard</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            value={renameName}
            onChange={(e) => onRenameNameChange(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={onCloseRename}>Cancel</Button>
          <Button onClick={onSubmitRename} variant="contained" disabled={renameLoading}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={deleteOpen} onClose={onCloseDelete}>
        <DialogTitle>Delete dashboard?</DialogTitle>
        <DialogContent>
          <Typography variant="body2">
            Permanently delete &quot;{deleteTargetName}&quot;? This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={onCloseDelete}>Cancel</Button>
          <Button onClick={onConfirmDelete} color="error" variant="contained" disabled={deleteLoading}>
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
