import { useState, useRef, useCallback } from 'react';
import apiService from '../services/apiService';

const DICTATION_PROMPT = 'Dictated prose. Capitalize sentences and use proper punctuation.';
const SILENCE_MS = 1200;
const MIN_RECORD_MS = 700;
const PARAGRAPH_PAUSE_MS = 2500;
const PARTIAL_THROTTLE_MS = 1200;
const SILENCE_THRESHOLD = 0.01;

function postProcess(text) {
  if (!text || typeof text !== 'string') return '';
  let t = text.trim();
  t = t.replace(/\bnew paragraph\b/gi, '\n\n');
  t = t.replace(/\bnew line\b/gi, '\n');
  return t.replace(/\n{3,}/g, '\n\n').trim();
}

export function useDictation() {
  const [isDictating, setIsDictating] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [segmentCount, setSegmentCount] = useState(0);

  const insertFnRef = useRef(null);
  const isDictatingRef = useRef(false);
  const streamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const rafIdRef = useRef(null);
  const lastVoiceTimeRef = useRef(0);
  const lastSegmentStopTimeRef = useRef(0);
  const prefixNextInsertWithParagraphRef = useRef(false);
  const partialInFlightRef = useRef(false);
  const lastPartialTimeRef = useRef(0);

  const stopCurrentSegment = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop();
    }
  }, []);

  const startRecordingSegment = useCallback(() => {
    const stream = streamRef.current;
    if (!stream || !insertFnRef.current) return;

    const now = Date.now();
    if (now - lastSegmentStopTimeRef.current > PARAGRAPH_PAUSE_MS) {
      prefixNextInsertWithParagraphRef.current = true;
    }

    const mimeType = MediaRecorder.isTypeSupported('audio/webm')
      ? 'audio/webm'
      : MediaRecorder.isTypeSupported('audio/ogg')
        ? 'audio/ogg'
        : '';
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mediaRecorderRef.current = recorder;
    recordedChunksRef.current = [];

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunksRef.current.push(e.data);
        if (recorder.state !== 'recording') return;
        const now = Date.now();
        if (
          !partialInFlightRef.current &&
          now - (lastPartialTimeRef.current || 0) > PARTIAL_THROTTLE_MS
        ) {
          partialInFlightRef.current = true;
          lastPartialTimeRef.current = now;
          const chunkBlob = new Blob([e.data], { type: e.data.type || 'audio/webm' });
          apiService.audio
            .transcribeAudio(chunkBlob, { prompt: DICTATION_PROMPT })
            .then((partial) => {
              setLiveTranscript((prev) => (prev ? `${prev} ${partial}` : partial));
            })
            .catch(() => {})
            .finally(() => {
              partialInFlightRef.current = false;
            });
        }
      }
    };

    recorder.onstop = async () => {
      try {
        const blob = new Blob(recordedChunksRef.current, {
          type: mimeType || 'audio/webm',
        });
        if (blob.size === 0) {
          setLiveTranscript('');
          if (isDictatingRef.current) startRecordingSegment();
          return;
        }
        const transcript = await apiService.audio.transcribeAudio(blob, {
          prompt: DICTATION_PROMPT,
        });
        let text = postProcess(transcript || '');
        if (!text) {
          setLiveTranscript('');
          if (isDictatingRef.current) startRecordingSegment();
          return;
        }
        if (prefixNextInsertWithParagraphRef.current) {
          text = '\n\n' + text;
          prefixNextInsertWithParagraphRef.current = false;
        }
        const insert = insertFnRef.current;
        if (insert) insert(text);
        lastSegmentStopTimeRef.current = Date.now();
        setSegmentCount((c) => c + 1);
      } catch (err) {
        console.error('Dictation segment transcription failed:', err);
      } finally {
        setLiveTranscript('');
        if (isDictatingRef.current) startRecordingSegment();
      }
    };

    recorder.start(1000);
  }, []);

  const startDictation = useCallback(
    (insertFn) => {
      if (typeof insertFn !== 'function') return;
      insertFnRef.current = insertFn;
      isDictatingRef.current = true;
      setSegmentCount(0);
      setLiveTranscript('');
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then((stream) => {
          streamRef.current = stream;
          startRecordingSegment();
          setIsDictating(true);

          const AudioContext = window.AudioContext || window.webkitAudioContext;
          const ctx = new AudioContext();
          audioContextRef.current = ctx;
          const source = ctx.createMediaStreamSource(stream);
          const analyser = ctx.createAnalyser();
          analyser.fftSize = 2048;
          analyserRef.current = analyser;
          source.connect(analyser);

          const data = new Float32Array(analyser.fftSize);
          const startTime = Date.now();
          lastVoiceTimeRef.current = Date.now();

          const check = () => {
            if (!isDictatingRef.current) return;
            analyser.getFloatTimeDomainData(data);
            let sumSquares = 0;
            for (let i = 0; i < data.length; i++) {
              const v = data[i];
              sumSquares += v * v;
            }
            const rms = Math.sqrt(sumSquares / data.length);
            const now = Date.now();
            if (rms > SILENCE_THRESHOLD) {
              lastVoiceTimeRef.current = now;
            }
            const elapsed = now - lastVoiceTimeRef.current;
            const recordedMs = now - startTime;
            if (
              recordedMs > MIN_RECORD_MS &&
              elapsed > SILENCE_MS &&
              mediaRecorderRef.current &&
              mediaRecorderRef.current.state === 'recording'
            ) {
              stopCurrentSegment();
              return;
            }
            rafIdRef.current = requestAnimationFrame(check);
          };
          rafIdRef.current = requestAnimationFrame(check);
        })
        .catch((err) => {
          console.error('Microphone access failed:', err);
          isDictatingRef.current = false;
          setIsDictating(false);
        });
    },
    [startRecordingSegment, stopCurrentSegment]
  );

  const stopDictation = useCallback(() => {
    isDictatingRef.current = false;
    try {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
      }
    } catch (_) {}
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch (_) {}
    streamRef.current = null;
    try {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
      if (audioContextRef.current?.state !== 'closed') {
        audioContextRef.current?.close();
      }
    } catch (_) {}
    audioContextRef.current = null;
    analyserRef.current = null;
    rafIdRef.current = null;
    setLiveTranscript('');
    setIsDictating(false);
  }, [stopCurrentSegment]);

  return {
    isDictating,
    startDictation,
    stopDictation,
    liveTranscript,
    segmentCount,
  };
}
