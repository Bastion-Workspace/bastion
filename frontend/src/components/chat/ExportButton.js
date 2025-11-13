import React, { useState } from 'react';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
  CircularProgress,
  Snackbar,
  Alert,
} from '@mui/material';
import {
  FileDownload,
  ContentCopy,
  PictureAsPdf,
  Description,
  Edit,
  BookmarkAdd,
} from '@mui/icons-material';
import exportService from '../../services/exportService';

const ExportButton = ({
  message,
  onCopyMessage,
  onSaveAsNote,
  copiedMessageId,
  savingNoteFor,
  currentConversationId,
  isUser,
}) => {
  const [anchorEl, setAnchorEl] = useState(null);
  const [exportingPdf, setExportingPdf] = useState(false);
  const [exportingDocx, setExportingDocx] = useState(false);
  const [exportingForEditor, setExportingForEditor] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleCopy = () => {
    onCopyMessage(message);
    handleClose();
  };

  const handleSaveAsNote = () => {
    onSaveAsNote(message);
    handleClose();
  };

  const handleExportPDF = async () => {
    setExportingPdf(true);
    try {
      await exportService.exportAsPDF(message);
      setSnackbar({ open: true, message: 'PDF exported successfully!', severity: 'success' });
    } catch (error) {
      setSnackbar({ open: true, message: error.message, severity: 'error' });
    } finally {
      setExportingPdf(false);
      handleClose();
    }
  };

  const handleExportDOCX = async () => {
    setExportingDocx(true);
    try {
      await exportService.exportAsDOCX(message);
      setSnackbar({ open: true, message: 'DOCX exported successfully!', severity: 'success' });
    } catch (error) {
      setSnackbar({ open: true, message: error.message, severity: 'error' });
    } finally {
      setExportingDocx(false);
      handleClose();
    }
  };

  const handleExportForEditor = async () => {
    setExportingForEditor(true);
    try {
      const result = await exportService.exportForEditor(message);
      setSnackbar({ open: true, message: result.message, severity: 'success' });
    } catch (error) {
      setSnackbar({ open: true, message: error.message, severity: 'error' });
    } finally {
      setExportingForEditor(false);
      handleClose();
    }
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const open = Boolean(anchorEl);

  return (
    <>
      <Tooltip title="Export Message">
        <IconButton
          size="small"
          onClick={handleClick}
          sx={{ 
            color: isUser ? 'primary.contrastText' : 'text.secondary',
            '&:hover': {
              backgroundColor: isUser ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.04)'
            }
          }}
        >
          <FileDownload fontSize="small" />
        </IconButton>
      </Tooltip>
      
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={handleCopy}>
          <ListItemIcon>
            {copiedMessageId === message.id ? (
              <CircularProgress size={16} />
            ) : (
              <ContentCopy fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText>
            {copiedMessageId === message.id ? "Copied!" : "Copy to Clipboard"}
          </ListItemText>
        </MenuItem>
        
        <MenuItem onClick={handleExportPDF} disabled={exportingPdf}>
          <ListItemIcon>
            {exportingPdf ? (
              <CircularProgress size={16} />
            ) : (
              <PictureAsPdf fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText>
            {exportingPdf ? "Exporting..." : "Export as PDF"}
          </ListItemText>
        </MenuItem>
        
        <MenuItem onClick={handleExportDOCX} disabled={exportingDocx}>
          <ListItemIcon>
            {exportingDocx ? (
              <CircularProgress size={16} />
            ) : (
              <Description fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText>
            {exportingDocx ? "Exporting..." : "Export as DOCX"}
          </ListItemText>
        </MenuItem>
        
        <MenuItem onClick={handleExportForEditor} disabled={exportingForEditor}>
          <ListItemIcon>
            {exportingForEditor ? (
              <CircularProgress size={16} />
            ) : (
              <Edit fontSize="small" />
            )}
          </ListItemIcon>
          <ListItemText>
            {exportingForEditor ? "Copying..." : "Copy for Editor"}
          </ListItemText>
        </MenuItem>
        
        {currentConversationId && message.message_id && (
          <MenuItem onClick={handleSaveAsNote} disabled={savingNoteFor === message.id}>
            <ListItemIcon>
              {savingNoteFor === message.id ? (
                <CircularProgress size={16} />
              ) : (
                <BookmarkAdd fontSize="small" />
              )}
            </ListItemIcon>
            <ListItemText>
              {savingNoteFor === message.id ? "Saving..." : "Save as Markdown"}
            </ListItemText>
          </MenuItem>
        )}
      </Menu>
      
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default ExportButton; 