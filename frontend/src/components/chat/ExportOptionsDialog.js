import React, { useEffect, useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  ToggleButton,
  ToggleButtonGroup,
  Stack,
} from '@mui/material';
import { getDefaultDocxExportOptions } from '../../utils/docxUtils';

const FONT_CHOICES = ['Calibri', 'Times New Roman', 'Arial', 'Georgia'];
const SIZE_CHOICES = [10, 11, 12, 14];

export default function ExportOptionsDialog({
  open,
  title = 'Export options',
  onClose,
  onConfirm,
  initialOptions = {},
}) {
  const defaults = getDefaultDocxExportOptions();
  const [fontFamily, setFontFamily] = useState(defaults.fontFamily);
  const [fontSizePt, setFontSizePt] = useState(defaults.fontSizePt);
  const [orientation, setOrientation] = useState(defaults.orientation);
  const [paper, setPaper] = useState(defaults.paper);
  const [includeTimestamps, setIncludeTimestamps] = useState(true);
  const [includeMetadata, setIncludeMetadata] = useState(false);

  useEffect(() => {
    if (!open) return;
    const d = getDefaultDocxExportOptions();
    setFontFamily(initialOptions.fontFamily ?? d.fontFamily);
    setFontSizePt(initialOptions.fontSizePt ?? d.fontSizePt);
    setOrientation(initialOptions.orientation ?? d.orientation);
    setPaper(initialOptions.paper ?? d.paper);
    setIncludeTimestamps(initialOptions.includeTimestamps ?? true);
    setIncludeMetadata(initialOptions.includeMetadata ?? false);
  }, [open, initialOptions]);

  const handleConfirm = () => {
    onConfirm({
      fontFamily,
      fontSizePt,
      orientation,
      paper,
      includeTimestamps,
      includeMetadata,
    });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{title}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ pt: 1 }}>
          <FormControl fullWidth size="small">
            <InputLabel id="export-font-label">Font</InputLabel>
            <Select
              labelId="export-font-label"
              label="Font"
              value={fontFamily}
              onChange={(e) => setFontFamily(e.target.value)}
            >
              {FONT_CHOICES.map((f) => (
                <MenuItem key={f} value={f}>
                  {f}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth size="small">
            <InputLabel id="export-size-label">Size</InputLabel>
            <Select
              labelId="export-size-label"
              label="Size"
              value={fontSizePt}
              onChange={(e) => setFontSizePt(Number(e.target.value))}
            >
              {SIZE_CHOICES.map((s) => (
                <MenuItem key={s} value={s}>
                  {s} pt
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth size="small">
            <InputLabel id="export-paper-label">Paper</InputLabel>
            <Select
              labelId="export-paper-label"
              label="Paper"
              value={paper}
              onChange={(e) => setPaper(e.target.value)}
            >
              <MenuItem value="letter">Letter (US)</MenuItem>
              <MenuItem value="a4">A4</MenuItem>
            </Select>
          </FormControl>

          <div>
            <InputLabel shrink sx={{ mb: 0.5 }}>
              Orientation
            </InputLabel>
            <ToggleButtonGroup
              exclusive
              fullWidth
              size="small"
              value={orientation}
              onChange={(_, v) => v && setOrientation(v)}
            >
              <ToggleButton value="portrait">Portrait</ToggleButton>
              <ToggleButton value="landscape">Landscape</ToggleButton>
            </ToggleButtonGroup>
          </div>

          <FormControlLabel
            control={
              <Checkbox
                checked={includeTimestamps}
                onChange={(e) => setIncludeTimestamps(e.target.checked)}
              />
            }
            label="Include timestamps in header"
          />

          <FormControlLabel
            control={
              <Checkbox
                checked={includeMetadata}
                onChange={(e) => setIncludeMetadata(e.target.checked)}
              />
            }
            label="Include message metadata (when available)"
          />
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleConfirm}>
          Export
        </Button>
      </DialogActions>
    </Dialog>
  );
}
