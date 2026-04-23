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
  Popper,
  Paper,
  List,
  ListItemButton,
  ListItemText,
} from '@mui/material';
import {
  Send,
  Clear,
  Stop,
  Mic,
  AttachFile,
  SmartToy,
  AutoMode,
  Groups,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useChatSidebar } from '../../contexts/ChatSidebarContext';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import { ACCENT_PALETTES } from '../../theme/themeConfig';
import apiService from '../../services/apiService';
import { getSelectableChatModels } from '../../utils/chatSelectableModels';
import agentFactoryService, { AGENT_HANDLES_QUERY_KEY } from '../../services/agentFactoryService';
import ConversationService from '../../services/conversation/ConversationService';

const ChatInputArea = () => {
  const { authLoading, user } = useAuth();
  const {
    query,
    setQuery,
    sendMessage,
    isLoading,
    currentConversationId,
    clearChat,
    createNewConversation,
    cancelCurrentJob,
    replyToMessage,
    setReplyToMessage,
    selectedModel,
    setSelectedModel,
  } = useChatSidebar();
  const { darkMode, accentId } = useTheme();
  const accentPalette = ACCENT_PALETTES[accentId] || ACCENT_PALETTES.blue;
  const accentTone = darkMode ? accentPalette.dark : accentPalette.light;
  const sendButtonMain = accentTone?.primary?.main || '#1976d2';
  const sendButtonDark = accentTone?.primary?.dark || sendButtonMain;

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
  
  // File attachment state
  const [selectedFiles, setSelectedFiles] = useState([]);
  const fileInputRef = useRef(null);

  // @mention autocomplete for Agent Factory
  const [selectedMentionIndex, setSelectedMentionIndex] = useState(0);
  const { data: agentHandles = [] } = useQuery(
    AGENT_HANDLES_QUERY_KEY,
    () => agentFactoryService.fetchAgentHandles(),
    { staleTime: 60000 }
  );
  const allMentionOptions = [
    { handle: 'auto', name: 'Auto-route (clear agent lock)', isAuto: true },
    ...agentHandles,
  ];
  const lastAt = (inputValue || '').lastIndexOf('@');
  const afterAt = lastAt >= 0 ? (inputValue || '').slice(lastAt + 1) : '';
  const mentionOpen = lastAt >= 0 && !afterAt.includes(' ');
  const mentionFilter = (afterAt.split(/\s/)[0] || '').toLowerCase();
  const filteredHandles = mentionFilter
    ? allMentionOptions.filter(
        (h) =>
          (h.handle || '').toLowerCase().startsWith(mentionFilter) ||
          (h.name || '').toLowerCase().includes(mentionFilter)
      )
    : allMentionOptions;
  const clampedMentionIndex = Math.min(
    Math.max(0, selectedMentionIndex),
    Math.max(0, filteredHandles.length - 1)
  );
  useEffect(() => {
    if (mentionOpen) setSelectedMentionIndex(0);
  }, [mentionFilter, mentionOpen]);

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

  const chatModels = getSelectableChatModels(enabledModelsData);

  // Set default model when data loads (catalog-verified list excludes orphans and image-gen model)
  useEffect(() => {
    if (authLoading || !user?.user_id) return;
    const list = getSelectableChatModels(enabledModelsData);
    if (list.length === 0) return;
    const validSelection = list.includes(selectedModel);
    if (!selectedModel || !validSelection) {
      const defaultModel = list[0];
      setSelectedModel(defaultModel);
      selectModelMutation.mutate(defaultModel);
    }
  }, [authLoading, user?.user_id, enabledModelsData, selectedModel]);

  // Handle model selection change
  const handleModelChange = (newModel) => {
    console.log('🎯 User selected model:', newModel);
    setSelectedModel(newModel);
    // Notify backend of the model change
    selectModelMutation.mutate(newModel);
  };

  const handleSendMessage = async () => {
    const trimmed = (inputValue || '').trim();
    if (!trimmed && selectedFiles.length === 0) return;
    
    // If we have files, we need to send message first to get message_id, then upload files
    if (selectedFiles.length > 0 && currentConversationId) {
      try {
        // Send message first (even if empty, to get message_id)
        const messageContent = trimmed || '📎 Attached files';
        sendMessage('auto', messageContent);
        
        // Get the message_id from the response (we'll need to wait for it)
        // For now, we'll upload files after a short delay
        // In production, we'd get message_id from sendMessage response
        setTimeout(async () => {
          try {
              // Get latest message for this conversation
              const messagesResponse = await ConversationService.getConversationMessages(currentConversationId, 0, 1);
              const messages = messagesResponse?.messages || [];
              if (messages.length > 0) {
                const messageId = messages[0].message_id;
                
                // Upload each file
                for (const file of selectedFiles) {
                  const formData = new FormData();
                  formData.append('file', file);
                  
                  await ConversationService.uploadAttachment(currentConversationId, messageId, file);
                }
              }
          } catch (error) {
            console.error('❌ Failed to upload attachments:', error);
          }
        }, 1000);
      } catch (error) {
        console.error('❌ Failed to send message with attachments:', error);
      }
    } else {
      // No files, send normally
      sendMessage('auto', trimmed);
    }
    
    // Clear local and context input after sending
    setInputValue('');
    setQuery('');
    setSelectedFiles([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files || []);
    
    // Validate file count (max 5)
    if (selectedFiles.length + files.length > 5) {
      alert('Maximum 5 files allowed per message');
      return;
    }
    
    // Validate file sizes (max 10MB each)
    const maxSize = 10 * 1024 * 1024; // 10MB
    const oversizedFiles = files.filter(f => f.size > maxSize);
    if (oversizedFiles.length > 0) {
      alert(`Some files exceed 10MB limit: ${oversizedFiles.map(f => f.name).join(', ')}`);
      return;
    }
    
    setSelectedFiles(prev => [...prev, ...files]);
  };
  
  const handleRemoveFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };
  
  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const insertMention = (handle) => {
    const before = (inputValue || '').slice(0, lastAt);
    setInputValue(before + '@' + handle + ' ');
  };

  const handleKeyDown = (e) => {
    if (mentionOpen && filteredHandles.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedMentionIndex((i) => Math.min(i + 1, filteredHandles.length - 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedMentionIndex((i) => Math.max(i - 1, 0));
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        insertMention(filteredHandles[clampedMentionIndex].handle);
        return;
      }
    }
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
    if (cancelCurrentJob) {
      await cancelCurrentJob();
    }
  };

  const getModelInfo = (modelId) => {
    return availableModelsData?.models?.find(m => m.id === modelId);
  };

  const getModelSourceTag = (modelInfo) => {
    if (!modelInfo?.source) return null;
    const sourceLabel = modelInfo.source === 'admin' ? 'Admin' : 'My providers';
    const provider = (modelInfo.provider_type || '').replace(/-/g, ' ');
    return provider ? `${sourceLabel} · ${provider}` : sourceLabel;
  };

  // One-line caption for model: show provider only when it adds info (e.g. vendor "Anthropic" vs provider_type "openrouter"), else just source tag.
  const getModelCaption = (modelInfo) => {
    const sourceTag = getModelSourceTag(modelInfo);
    if (!modelInfo?.provider) return sourceTag || null;
    const providerTypeFormatted = (modelInfo.provider_type || '').replace(/-/g, ' ');
    const providerDisplay = modelInfo.provider.trim();
    if (providerDisplay.toLowerCase() === providerTypeFormatted.toLowerCase()) return sourceTag || null;
    return sourceTag ? `${modelInfo.provider} · ${sourceTag}` : modelInfo.provider;
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
    <Box key={darkMode ? 'dark' : 'light'} sx={{ p: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
      {/* Model Selection */}
      {chatModels.length > 0 && (
        <Box sx={{ mb: 1.5 }}>
          <FormControl size="small" fullWidth>
            <InputLabel>AI Model</InputLabel>
            <Select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              label="AI Model"
            >
              {chatModels.map((modelId) => {
                const modelInfo = getModelInfo(modelId);
                const pricingInfo = formatPricing(modelInfo);
                const isSelected = selectedModel === modelId;
                return (
                  <MenuItem key={modelId} value={modelId}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%" sx={{ gap: 1 }}>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography
                          variant="body2"
                          sx={{
                            fontWeight: isSelected ? 'bold' : 'normal',
                            textAlign: 'left'
                          }}
                        >
                          {modelInfo?.name || modelId}
                        </Typography>
                        {getModelCaption(modelInfo) && (
                          <Typography variant="caption" color="text.secondary" display="block">
                            {getModelCaption(modelInfo)}
                          </Typography>
                        )}
                      </Box>
                      {pricingInfo && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ whiteSpace: 'nowrap', ml: 1 }}
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

      {/* Reply Indicator */}
      {replyToMessage && (
        <Box sx={{ 
          mb: 1, 
          p: 1, 
          bgcolor: 'action.hover', 
          borderRadius: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 1
        }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
              Replying to:
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
            >
              {replyToMessage.content?.substring(0, 100) || 'Message'}
              {replyToMessage.content && replyToMessage.content.length > 100 ? '...' : ''}
            </Typography>
          </Box>
          <IconButton
            size="small"
            onClick={() => setReplyToMessage(null)}
            sx={{ flexShrink: 0 }}
          >
            <Clear fontSize="small" />
          </IconButton>
        </Box>
      )}

      {/* File Preview Area */}
      {selectedFiles.length > 0 && (
        <Box sx={{ mb: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {selectedFiles.map((file, index) => (
            <Chip
              key={index}
              label={`${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`}
              onDelete={() => handleRemoveFile(index)}
              size="small"
              color="primary"
              variant="outlined"
            />
          ))}
        </Box>
      )}

      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        style={{ display: 'none' }}
        onChange={handleFileSelect}
        accept="image/*,application/pdf,application/vnd.openxmlformats-officedocument.presentationml.presentation,text/*,audio/*"
      />

      {/* Input Area */}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
        <Tooltip title="Attach file">
          <IconButton
            onClick={handleAttachClick}
            size="small"
            sx={{
              backgroundColor: 'action.hover',
              '&:hover': {
                backgroundColor: 'action.selected',
              },
            }}
          >
            <AttachFile fontSize="small" />
          </IconButton>
        </Tooltip>
        
        <TextField
          ref={textFieldRef}
          multiline
          maxRows={4}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onKeyPress={handleKeyPress}
          placeholder="Type your message... Use @ to mention an agent"
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
        <Popper
          open={mentionOpen && filteredHandles.length > 0}
          anchorEl={textFieldRef.current}
          placement="top-start"
          style={{ zIndex: 1300 }}
        >
          <Paper elevation={2} sx={{ maxHeight: 220, overflow: 'auto', minWidth: 220 }}>
            <List dense>
              {filteredHandles.slice(0, 8).map((h, i) => (
                <ListItemButton
                  key={h.isAuto ? 'auto' : (h.id || h.handle)}
                  selected={i === clampedMentionIndex}
                  onClick={() => insertMention(h.handle)}
                >
                  {h.isAuto ? (
                    <AutoMode sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
                  ) : h.type === 'team' || h.type === 'line' ? (
                    <Groups sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
                  ) : (
                    <SmartToy sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
                  )}
                  <ListItemText
                    primary={'@' + h.handle}
                    secondary={
                      h.type === 'team' || h.type === 'line'
                        ? (h.name ? `${h.name} (line)` : '(line)')
                        : (h.name || '')
                    }
                  />
                </ListItemButton>
              ))}
            </List>
          </Paper>
        </Popper>
        
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

          {isLoading ? (
            <Tooltip title="Stop generation">
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
                disabled={!inputValue.trim() && selectedFiles.length === 0 || isLoading}
                color="primary"
                size="small"
                sx={{ 
                  backgroundColor: sendButtonMain,
                  color: 'white',
                  border: 'none',
                  boxShadow: 'none',
                  '&:hover': {
                    backgroundColor: sendButtonDark,
                    boxShadow: 'none',
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

    </Box>
  );
};

export default ChatInputArea; 