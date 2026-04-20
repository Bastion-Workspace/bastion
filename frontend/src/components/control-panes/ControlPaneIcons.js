import React, { useState } from 'react';
import { Box, IconButton, Popover, Tooltip, Badge } from '@mui/material';
import {
  Tune,
  Settings,
  VolumeUp,
  PlayArrow,
  TouchApp,
  Dashboard,
  ToggleOn,
  SmartToy,
} from '@mui/icons-material';
import { useControlPanes } from '../../contexts/ControlPaneContext';
import { useArtifactInstance } from '../../contexts/ArtifactInstanceContext';
import ControlPanePopover from './ControlPanePopover';

const ICON_MAP = {
  Tune,
  Settings,
  VolumeUp,
  PlayArrow,
  TouchApp,
  Dashboard,
  ToggleOn,
  SmartToy,
};

function getIconComponent(iconName) {
  if (!iconName) return Tune;
  const name = String(iconName).trim();
  return ICON_MAP[name] || Tune;
}

const ControlPaneIcons = () => {
  const { visiblePanes } = useControlPanes();
  const { getBadgeForArtifact, clearBadgeForArtifact } = useArtifactInstance();
  const [anchorEl, setAnchorEl] = useState(null);
  const [openPaneId, setOpenPaneId] = useState(null);

  const handleOpen = (event, pane) => {
    setAnchorEl(event.currentTarget);
    setOpenPaneId(pane.id);
    if ((pane.pane_type || 'connector') === 'artifact' && pane.artifact_id) {
      clearBadgeForArtifact(pane.artifact_id);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
    setOpenPaneId(null);
  };

  const open = Boolean(anchorEl);
  const openPane = openPaneId ? visiblePanes.find((p) => p.id === openPaneId) : null;

  if (!visiblePanes || visiblePanes.length === 0) return null;

  return (
    <>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
        {visiblePanes.map((pane) => {
          const IconComponent = getIconComponent(pane.icon);
          const isArtifact = (pane.pane_type || 'connector') === 'artifact';
          const badgeInfo =
            isArtifact && pane.artifact_id ? getBadgeForArtifact(pane.artifact_id) : null;
          const showDot = Boolean(badgeInfo?.show);
          const tip = showDot && badgeInfo?.text ? `${pane.name || 'Control pane'}: ${badgeInfo.text}` : pane.name || 'Control pane';
          return (
            <Tooltip key={pane.id} title={tip}>
              <Badge
                color="error"
                variant="dot"
                invisible={!showDot}
                overlap="circular"
                anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
              >
                <IconButton
                  size="small"
                  onClick={(e) => handleOpen(e, pane)}
                  aria-label={pane.name}
                  sx={{ p: 0.25 }}
                >
                  <IconComponent sx={{ fontSize: '1.1rem' }} />
                </IconButton>
              </Badge>
            </Tooltip>
          );
        })}
      </Box>
      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        slotProps={{ paper: { sx: { mt: -1 } } }}
      >
        {openPane && (
          <ControlPanePopover pane={openPane} onClose={handleClose} />
        )}
      </Popover>
    </>
  );
};

export default ControlPaneIcons;
