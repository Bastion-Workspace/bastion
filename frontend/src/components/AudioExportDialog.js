import React, { useEffect, useRef, useState } from 'react';
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  LinearProgress,
  Stack,
  Typography,
} from '@mui/material';
import apiService from '../services/apiService';
import { useVoiceAvailability } from '../contexts/VoiceAvailabilityContext';
import { stripTextForSpeech } from '../utils/textForSpeech';
import {
  AUDIO_EXPORT_BACKEND_THRESHOLD_CHUNKS,
  splitTextForTts,
} from '../utils/ttsStreamUtils';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function buildDownloadFilename(title, filename) {
  const base = String(title || filename || 'document').replace(/\.[^/.]+$/, '');
  const safe = base.replace(/[^\w\s\-_.]+/g, '').trim() || 'document';
  return safe.toLowerCase().endsWith('.mp3') ? safe : `${safe}.mp3`;
}

function triggerBlobDownload(blob, fname) {
  const url = URL.createObjectURL(blob);
  const a = window.document.createElement('a');
  a.href = url;
  a.download = fname;
  window.document.body.appendChild(a);
  a.click();
  window.document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Export document text as MP3: short docs in-browser (chunked synthesize), long docs via Celery.
 */
export default function AudioExportDialog({
  open,
  onClose,
  documentId,
  documentContent,
  documentTitle,
  documentFilename,
  speechMode = 'markdown',
}) {
  const { refreshPrefs } = useVoiceAvailability();
  const [phase, setPhase] = useState('idle');
  const [message, setMessage] = useState('');
  const [current, setCurrent] = useState(0);
  const [total, setTotal] = useState(0);
  const abortRef = useRef(null);

  useEffect(() => {
    if (!open) {
      setPhase('idle');
      setMessage('');
      setCurrent(0);
      setTotal(0);
      return undefined;
    }

    const ac = new AbortController();
    abortRef.current = ac;

    (async () => {
      const [avail, prefs] = await Promise.all([
        apiService.voice.checkAvailability(true),
        refreshPrefs(),
      ]);
      if (ac.signal.aborted) return;

      if (prefs.prefer_browser_tts || !avail?.available) {
        setPhase('error');
        setMessage(
          'Server text-to-speech is required. Configure a voice provider in settings (not browser-only).'
        );
        return;
      }

      const sanitized = stripTextForSpeech(String(documentContent || ''), speechMode).trim();
      if (!sanitized) {
        setPhase('error');
        setMessage('No speakable text in this document.');
        return;
      }

      const chunks = splitTextForTts(sanitized);
      const fname = buildDownloadFilename(documentTitle, documentFilename);
      setPhase('running');
      setTotal(Math.max(chunks.length, 1));
      setCurrent(0);

      try {
        if (chunks.length > AUDIO_EXPORT_BACKEND_THRESHOLD_CHUNKS) {
          if (!documentId) {
            setPhase('error');
            setMessage('This document is too long for browser export; document id is missing.');
            return;
          }
          setMessage('Synthesizing on server (this may take a while)...');
          const start = await apiService.voice.startAudioExport(
            { documentId },
            { signal: ac.signal }
          );
          const taskId = start?.task_id;
          if (!taskId) {
            setPhase('error');
            setMessage('Failed to start server export.');
            return;
          }
          while (!ac.signal.aborted) {
            const st = await apiService.voice.getAudioExportStatus(taskId, {
              signal: ac.signal,
            });
            if (st.status === 'complete') {
              const blob = await apiService.voice.downloadAudioExportBlob(taskId, {
                signal: ac.signal,
              });
              triggerBlobDownload(blob, fname);
              setPhase('done');
              setMessage('Download started.');
              return;
            }
            if (st.status === 'failure') {
              setPhase('error');
              setMessage(st.error || 'Server export failed.');
              return;
            }
            if (st.status === 'progress') {
              setCurrent(st.current_chunk || 0);
              setTotal(st.total_chunks || chunks.length);
              setMessage(
                `Server: chunk ${st.current_chunk || 0} of ${st.total_chunks || '?'}...`
              );
            }
            await sleep(1500);
          }
          return;
        }

        const blobs = [];
        for (let i = 0; i < chunks.length; i++) {
          if (ac.signal.aborted) return;
          const seg = chunks[i].trim();
          if (!seg) continue;
          setCurrent(i + 1);
          setMessage(`Synthesizing chunk ${i + 1} of ${chunks.length}...`);
          const { blob } = await apiService.voice.synthesize(seg, {
            outputFormat: 'mp3',
            signal: ac.signal,
          });
          blobs.push(blob);
        }
        if (ac.signal.aborted) return;
        const merged = new Blob(blobs, { type: 'audio/mpeg' });
        triggerBlobDownload(merged, fname);
        setPhase('done');
        setMessage('Download started.');
      } catch (e) {
        if (e?.name === 'AbortError') return;
        setPhase('error');
        setMessage(e?.message || 'Export failed.');
      }
    })();

    return () => {
      ac.abort();
      abortRef.current = null;
    };
  }, [open, documentId, documentContent, documentTitle, documentFilename, speechMode, refreshPrefs]);

  const handleCancel = () => {
    abortRef.current?.abort();
    onClose();
  };

  const progress = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;

  return (
    <Dialog open={open} onClose={phase === 'running' ? undefined : onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Export as audio</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          {phase === 'running' && <LinearProgress variant="determinate" value={progress} />}
          <Typography variant="body2" color="text.secondary">
            {message || (phase === 'idle' ? 'Preparing…' : '')}
          </Typography>
        </Stack>
      </DialogContent>
      <DialogActions>
        {phase === 'running' ? (
          <Button onClick={handleCancel}>Cancel</Button>
        ) : (
          <Button onClick={onClose} variant="contained">
            Close
          </Button>
        )}
      </DialogActions>
    </Dialog>
  );
}
