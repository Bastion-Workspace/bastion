import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  TextField,
  IconButton,
  FormControl,
  Select,
  MenuItem,
  InputLabel,
  Tooltip,
  Typography,
  Chip,
} from '@mui/material';
import {
  Send,
  Clear,
  Stop,
  Mic,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useChatSidebar } from '../../contexts/ChatSidebarContext';
import { useModel } from '../../contexts/ModelContext';
import apiService from '../../services/apiService';

const ChatInputArea = () => {
  const {
    query,
    setQuery,
    sendMessage,
    isLoading,
    currentConversationId,
    clearChat,
    createNewConversation,
    currentJobId,
    cancelCurrentJob,
  } = useChatSidebar();
  const { selectedModel, setSelectedModel } = useModel();

  const textFieldRef = useRef(null);
  // Use local input state to avoid global context updates on each keystroke
  const [inputValue, setInputValue] = useState(query || '');
  const queryClient = useQueryClient();
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const rafIdRef = useRef(null);
  const lastVoiceTimeRef = useRef(0);
  const streamRef = useRef(null);
  const [liveTranscript, setLiveTranscript] = useState('');
  const lastPartialTimeRef = useRef(0);
  const partialInFlightRef = useRef(false);

  // Model selection mutation to notify backend
  const selectModelMutation = useMutation(
    (modelName) => apiService.selectModel(modelName),
    {
      onSuccess: (data) => {
        console.log('✅ Model selected successfully:', data);
        queryClient.invalidateQueries('currentModel');
      },
      onError: (error) => {
        console.error('❌ Failed to select model:', error);
      },
    }
  );

  // Fetch enabled models
  const { data: enabledModelsData } = useQuery(
    ['enabledModels'],
    () => apiService.getEnabledModels(),
    {
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
    }
  );

  // Fetch available models
  const { data: availableModelsData } = useQuery(
    ['availableModels'],
    () => apiService.getAvailableModels(),
    {
      refetchOnWindowFocus: false,
      staleTime: 300000, // 5 minutes
    }
  );

  // Keep local input in sync when context query changes externally (e.g., clear, conversation switch)
  useEffect(() => {
    setInputValue(query || '');
  }, [query]);

  // Set default model when data loads
  useEffect(() => {
    if (enabledModelsData?.enabled_models?.length > 0 && !selectedModel) {
      const defaultModel = enabledModelsData.enabled_models[0];
      setSelectedModel(defaultModel);
      // Also notify backend of the default model selection
      selectModelMutation.mutate(defaultModel);
    }
  }, [enabledModelsData, selectedModel]);

  // Handle model selection change
  const handleModelChange = (newModel) => {
    console.log('🎯 User selected model:', newModel);
    setSelectedModel(newModel);
    // Notify backend of the model change
    selectModelMutation.mutate(newModel);
  };

  const handleSendMessage = () => {
    const trimmed = (inputValue || '').trim();
    if (!trimmed) return;
    // Use override to avoid relying on context query
    sendMessage('auto', trimmed);
    // Clear local and context input after sending
    setInputValue('');
    setQuery('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = () => {
    clearChat();
    textFieldRef.current?.focus();
  };

  const handleCancelJob = async () => {
    if (currentJobId && cancelCurrentJob) {
      await cancelCurrentJob();
    }
  };

  const getModelInfo = (modelId) => {
    return availableModelsData?.models?.find(m => m.id === modelId);
  };

  // Format cost for display (per 1M tokens by default)
  const formatCost = (cost) => {
    if (!cost) return 'Free';
    if (cost < 0.001) return `$${(cost * 1000000).toFixed(2)}`;
    if (cost < 1) return `$${(cost * 1000).toFixed(2)}`;
    return `$${cost.toFixed(3)}`;
  };

  // Format pricing string for display
  const formatPricing = (modelInfo) => {
    if (!modelInfo) return '';
    
    const parts = [];
    
    // Add context length
    if (modelInfo.context_length) {
      parts.push(`${modelInfo.context_length.toLocaleString()} ctx`);
    }
    
    // Add pricing if available
    if (modelInfo.input_cost || modelInfo.output_cost) {
      const inputPrice = modelInfo.input_cost ? formatCost(modelInfo.input_cost) : 'Free';
      const outputPrice = modelInfo.output_cost ? formatCost(modelInfo.output_cost) : 'Free';
      parts.push(`I/O: ${inputPrice} / ${outputPrice}`);
    }
    
    return parts.join(' • ');
  };

  const isSendDisabled = !inputValue.trim() || isLoading;

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : (MediaRecorder.isTypeSupported('audio/ogg') ? 'audio/ogg' : '');
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recordedChunksRef.current = [];
      recorder.ondataavailable = async (e) => {
        if (e.data && e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
          // Lightweight partial transcription while recording (throttled)
          try {
            if (recorder.state === 'recording') {
              const now = Date.now();
              const throttleMs = 1200;
              if (!partialInFlightRef.current && now - (lastPartialTimeRef.current || 0) > throttleMs) {
                partialInFlightRef.current = true;
                lastPartialTimeRef.current = now;
                const chunkBlob = new Blob([e.data], { type: e.data.type || 'audio/webm' });
                const partial = await apiService.audio.transcribeAudio(chunkBlob);
                setLiveTranscript(prev => prev ? `${prev} ${partial}` : partial);
              }
            }
          } catch (pe) {
            // Non-blocking
          } finally {
            partialInFlightRef.current = false;
          }
        }
      };
      recorder.onstop = async () => {
        try {
          const blob = new Blob(recordedChunksRef.current, { type: mimeType || 'audio/webm' });
          // Call transcription service
          const transcript = await apiService.audio.transcribeAudio(blob);
          setInputValue(prev => (prev ? `${prev}\n${transcript}` : transcript));
          textFieldRef.current?.focus();
        } catch (err) {
          console.error('❌ Transcription error:', err);
        } finally {
          // Stop all tracks
          try { (streamRef.current || stream).getTracks().forEach(t => t.stop()); } catch {}
          // Cleanup audio context and analyser
          try {
            if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
            if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
              audioContextRef.current.close();
            }
          } catch {}
          audioContextRef.current = null;
          analyserRef.current = null;
          rafIdRef.current = null;
          streamRef.current = null;
          lastPartialTimeRef.current = 0;
          partialInFlightRef.current = false;
          setLiveTranscript('');
          setIsRecording(false);
        }
      };
      // Request periodic chunks to enable partial transcription
      recorder.start(1000);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);

      // Initialize simple VAD (silence detection) to auto-stop
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const ctx = new AudioContext();
        audioContextRef.current = ctx;
        const source = ctx.createMediaStreamSource(stream);
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        analyserRef.current = analyser;
        source.connect(analyser);

        const data = new Float32Array(analyser.fftSize);
        const silenceThreshold = 0.01; // RMS threshold
        const silenceMs = 1200; // auto-stop after this much silence
        const minRecordMs = 700; // don't stop too early
        const startTime = Date.now();
        lastVoiceTimeRef.current = Date.now();

        const check = () => {
          analyser.getFloatTimeDomainData(data);
          let sumSquares = 0;
          for (let i = 0; i < data.length; i++) {
            const v = data[i];
            sumSquares += v * v;
          }
          const rms = Math.sqrt(sumSquares / data.length);
          const now = Date.now();
          if (rms > silenceThreshold) {
            lastVoiceTimeRef.current = now;
          }
          const elapsed = now - lastVoiceTimeRef.current;
          const recordedMs = now - startTime;
          if (recordedMs > minRecordMs && elapsed > silenceMs && mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            stopRecording();
            return;
          }
          rafIdRef.current = requestAnimationFrame(check);
        };
        rafIdRef.current = requestAnimationFrame(check);
      } catch (vadErr) {
        console.warn('⚠️ VAD init failed (non-blocking):', vadErr);
      }
    } catch (err) {
      console.error('❌ Microphone access error:', err);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    try {
      const recorder = mediaRecorderRef.current;
      if (recorder && recorder.state !== 'inactive') {
        recorder.stop();
      }
    } catch {}
  };

  return (
    <Box sx={{ p: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
      {/* Model Selection */}
      {enabledModelsData?.enabled_models?.length > 0 && (
        <Box sx={{ mb: 1.5 }}>
          <FormControl size="small" fullWidth>
            <InputLabel>AI Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              label="AI Model"
            >
              {enabledModelsData.enabled_models.map((modelId) => {
                const modelInfo = getModelInfo(modelId);
                const pricingInfo = formatPricing(modelInfo);
                const isSelected = selectedModel === modelId;
                return (
                  <MenuItem key={modelId} value={modelId}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%" sx={{ gap: 1 }}>
                      <Typography 
                        variant="body2" 
                        sx={{ 
                          fontWeight: isSelected ? 'bold' : 'normal',
                          flex: 1,
                          textAlign: 'left'
                        }}
                      >
                        {modelInfo?.name || modelId}
                      </Typography>
                      {pricingInfo && (
                        <Typography 
                          variant="caption" 
                          color="text.secondary"
                          sx={{ textAlign: 'right', whiteSpace: 'nowrap' }}
                        >
                          {pricingInfo}
                        </Typography>
                      )}
                    </Box>
                  </MenuItem>
                );
              })}
            </Select>
          </FormControl>
        </Box>
      )}



      {/* Input Area */}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
        <TextField
          ref={textFieldRef}
          multiline
          maxRows={4}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          variant="outlined"
          size="small"
          fullWidth
          disabled={isLoading}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 2,
            },
          }}
        />
        
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <Tooltip title={isRecording ? 'Stop recording' : 'Record voice'}>
            <IconButton
              onClick={isRecording ? stopRecording : startRecording}
              color={isRecording ? 'error' : 'default'}
              size="small"
              sx={{
                backgroundColor: isRecording ? 'error.main' : 'action.hover',
                color: isRecording ? 'white' : 'inherit',
                '&:hover': {
                  backgroundColor: isRecording ? 'error.dark' : 'action.selected',
                },
              }}
            >
              <Mic fontSize="small" />
            </IconButton>
          </Tooltip>

          {isLoading && currentJobId ? (
            <Tooltip title="Cancel processing">
              <IconButton
                onClick={handleCancelJob}
                color="error"
                size="small"
                sx={{ 
                  backgroundColor: 'error.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'error.dark',
                  },
                }}
              >
                <Stop fontSize="small" />
              </IconButton>
            </Tooltip>
          ) : (
            <Tooltip title="Send message (Enter)">
              <IconButton
                onClick={handleSendMessage}
                disabled={isSendDisabled}
                color="primary"
                size="small"
                sx={{ 
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                  '&:disabled': {
                    backgroundColor: 'action.disabledBackground',
                    color: 'action.disabled',
                  },
                }}
              >
                <Send fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Box>

      {liveTranscript && isRecording && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Listening: {liveTranscript}
        </Typography>
      )}

      {/* Character count */}
      {inputValue && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          {inputValue.length} characters
        </Typography>
      )}
    </Box>
  );
};

export default ChatInputArea; 