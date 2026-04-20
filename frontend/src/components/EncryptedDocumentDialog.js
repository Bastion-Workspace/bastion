import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
  IconButton,
  InputAdornment,
  CircularProgress,
  Alert,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { Visibility, VisibilityOff, ExpandMore, HelpOutline } from '@mui/icons-material';
import apiService from '../services/apiService';
import {
  HELP_TOPIC_DOCUMENT_ENCRYPTION,
  openHelpTopic,
} from '../constants/helpTopics';

/**
 * @param {'unlock'|'encrypt'|'remove'} mode
 * @param {string|null} documentId
 * @param {boolean} open
 * @param {() => void} onClose
 * @param {(result: { sessionToken?: string, ttlSeconds?: number }) => void} onSuccess — unlock returns session
 */
export default function EncryptedDocumentDialog({
  mode = 'unlock',
  documentId,
  open,
  onClose,
  onSuccess,
}) {
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (open) {
      setPassword('');
      setConfirmPassword('');
      setError(null);
      setShowPassword(false);
    }
  }, [open, documentId, mode]);

  const title =
    mode === 'encrypt'
      ? 'Encrypt document'
      : mode === 'remove'
        ? 'Remove encryption'
        : 'Unlock encrypted document';

  const handleSubmit = async () => {
    setError(null);
    if (!documentId) {
      setError('No document selected');
      return;
    }
    if (mode === 'encrypt') {
      if (password.length < 8) {
        setError('Password must be at least 8 characters');
        return;
      }
      if (password !== confirmPassword) {
        setError('Passwords do not match');
        return;
      }
    }
    if (!password) {
      setError('Enter a password');
      return;
    }

    setLoading(true);
    try {
      if (mode === 'encrypt') {
        await apiService.encryptDocument(documentId, password, confirmPassword);
        onSuccess?.({});
        onClose?.();
      } else if (mode === 'remove') {
        await apiService.removeEncryption(documentId, password);
        onSuccess?.({});
        onClose?.();
      } else {
        const res = await apiService.createDecryptSession(documentId, password);
        onSuccess?.({
          sessionToken: res.session_token,
          ttlSeconds: res.ttl_seconds,
        });
        onClose?.();
      }
    } catch (e) {
      const msg =
        e?.response?.data?.detail ||
        e?.message ||
        (typeof e === 'string' ? e : 'Request failed');
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg));
    } finally {
      setLoading(false);
    }
  };

  const openFullHelp = () => {
    openHelpTopic(HELP_TOPIC_DOCUMENT_ENCRYPTION);
  };

  return (
    <Dialog open={open} onClose={loading ? undefined : onClose} maxWidth="sm" fullWidth>
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1,
          pr: 1,
        }}
      >
        <Typography component="span" variant="h6" sx={{ pr: 1 }}>
          {title}
        </Typography>
        <IconButton
          size="small"
          onClick={openFullHelp}
          aria-label="Open Document encryption help"
          title="Open full help: Document encryption"
        >
          <HelpOutline />
        </IconButton>
      </DialogTitle>
      <DialogContent>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {mode === 'unlock' && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Enter the document password to edit and save. Your unlock session stays active while you
            use this browser; use <strong>Lock</strong> in the toolbar or close the last tab for this
            file to end it. Other encrypted files each need their own password.
          </Alert>
        )}

        {mode === 'encrypt' && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            The file is encrypted on the server, <strong>excluded from search</strong>, and cannot use
            real-time collaboration. Use a strong password you can recover (a password manager helps).
            If you lose the password, the content cannot be restored from Bastion.
          </Alert>
        )}

        {mode === 'remove' && (
          <Alert severity="info" sx={{ mb: 2 }}>
            After removal, the file is stored in plain form again and can be indexed for search (unless
            the file is separately marked exempt from search). You must enter the current document
            password.
          </Alert>
        )}

        <Accordion disableGutters elevation={0} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, mb: 2, '&:before': { display: 'none' } }}>
          <AccordionSummary expandIcon={<ExpandMore />} aria-controls="encryption-help-details" id="encryption-help-header">
            <Typography variant="subtitle2">What to expect (quick reference)</Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ pt: 0 }}>
            {mode === 'unlock' && (
              <List dense disablePadding>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText
                    primary="Sessions and tabs"
                    secondary="Leaving this document tab open in the tab bar usually keeps your session so you are not asked again when you switch back, until it expires or you lock."
                  />
                </ListItem>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText
                    primary="Linked or referenced files"
                    secondary="If this file points to another encrypted document, that other file still needs its own unlock (or its own open tab with a valid session)."
                  />
                </ListItem>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText
                    primary="Save and lock"
                    secondary="If your session ends while editing, save may fail until you unlock again. Use Lock when you step away from an unlocked document."
                  />
                </ListItem>
              </List>
            )}
            {mode === 'encrypt' && (
              <List dense disablePadding>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText primary="Formats" secondary="Only .md, .txt, and .org can be encrypted from the library." />
                </ListItem>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText
                    primary="Search and tools"
                    secondary="Search and automatic processing skip ciphertext. After removal, indexing can resume if applicable."
                  />
                </ListItem>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText primary="Logout" secondary="Logging out clears unlock sessions in this browser." />
                </ListItem>
              </List>
            )}
            {mode === 'remove' && (
              <List dense disablePadding>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText
                    primary="One-time operation"
                    secondary="The file is decrypted and rewritten; choose a safe network environment if your policy requires it."
                  />
                </ListItem>
                <ListItem disableGutters sx={{ display: 'block' }}>
                  <ListItemText primary="Password" secondary="You need the current encryption password, not your login password." />
                </ListItem>
              </List>
            )}
            <Button size="small" onClick={openFullHelp} sx={{ mt: 1 }}>
              Open full &quot;Document encryption&quot; help
            </Button>
          </AccordionDetails>
        </Accordion>

        <TextField
          autoFocus
          margin="dense"
          label={mode === 'encrypt' ? 'New password' : 'Password'}
          type={showPassword ? 'text' : 'password'}
          fullWidth
          value={password}
          onChange={(ev) => setPassword(ev.target.value)}
          disabled={loading}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label="toggle password visibility"
                  onClick={() => setShowPassword((s) => !s)}
                  edge="end"
                >
                  {showPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
        {mode === 'encrypt' && (
          <TextField
            margin="dense"
            label="Confirm password"
            type={showPassword ? 'text' : 'password'}
            fullWidth
            value={confirmPassword}
            onChange={(ev) => setConfirmPassword(ev.target.value)}
            disabled={loading}
            sx={{ mt: 1 }}
          />
        )}
        <Box sx={{ mt: 1.5 }}>
          <Typography variant="caption" color="text.secondary">
            You can also open <strong>Help</strong> from the user menu (top right) and search for{' '}
            <strong>Document encryption</strong>.
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button onClick={handleSubmit} variant="contained" disabled={loading}>
          {loading ? (
            <CircularProgress size={22} />
          ) : mode === 'unlock' ? (
            'Unlock'
          ) : mode === 'encrypt' ? (
            'Encrypt'
          ) : (
            'Decrypt'
          )}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
