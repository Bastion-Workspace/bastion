# Voice Conversation Mode Implementation Plan

**Roosevelt's Voice Conversation Cavalry - Battle Plan**

**BULLY!** This document outlines the comprehensive implementation strategy for adding voice input, text-to-speech, and full conversation mode to the Plato Knowledge Base chat interface.

## 🎯 Executive Summary

Transform the chat interface into a natural conversation experience with:
- **Voice Input**: Microphone button for speech-to-text via OpenAI Whisper
- **Text-to-Speech**: AI responses converted to natural speech via OpenAI TTS
- **Full Conversation Mode**: Hands-free conversation with "Codex" wake word activation
- **Progressive Enhancement**: Voice features enhance existing chat without replacing it

## 🏗️ Current System Analysis

### Existing Architecture
- **Frontend**: React with Material-UI, ChatInputArea component
- **Backend**: FastAPI with OpenAI/OpenRouter integration
- **Chat Flow**: LangGraph orchestrator with unified message processing
- **State Management**: React Context (ChatSidebarContext)
- **Configuration**: Environment-based API key management

### Integration Points
- `ChatInputArea.js` - Add voice button next to send button
- `ChatMessagesArea.js` - Add audio playback for AI responses
- `useChatSidebar` hook - Extend with voice state management
- Backend API - Add voice transcription and TTS endpoints

## 🎤 Voice Input Implementation

### 1. Backend Voice Services

#### Voice Transcription Service
```python
# backend/services/voice_transcription_service.py
class VoiceTranscriptionService:
    """Roosevelt's Voice-to-Text Cavalry Service"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def transcribe_audio(self, audio_data: bytes, format: str = "webm") -> str:
        """Use OpenAI Whisper for transcription"""
        try:
            # Convert audio to supported format if needed
            audio_file = self._prepare_audio_file(audio_data, format)
            
            # Call OpenAI Whisper API
            response = await self.openai_client.audio.transcriptions.create(
                model=settings.WHISPER_MODEL,
                file=audio_file,
                language="en"  # Can be made configurable
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise
            
    async def validate_audio_format(self, audio_data: bytes) -> bool:
        """Validate audio format and quality"""
        # Check file size, format, duration
        return len(audio_data) <= settings.MAX_AUDIO_SIZE_BYTES
```

#### Text-to-Speech Service
```python
# backend/services/text_to_speech_service.py
class TextToSpeechService:
    """Roosevelt's Text-to-Voice Artillery Service"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
    async def generate_speech(self, text: str, voice: str = "alloy") -> bytes:
        """Use OpenAI TTS for speech generation"""
        try:
            response = await self.openai_client.audio.speech.create(
                model=settings.TTS_MODEL,
                voice=voice,
                input=text,
                speed=settings.TTS_SPEED
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"❌ TTS generation failed: {e}")
            raise
            
    async def get_available_voices(self) -> List[str]:
        """Get list of available TTS voices"""
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
```

### 2. Backend API Endpoints

```python
# backend/api/voice_api.py
from fastapi import APIRouter, UploadFile, HTTPException
from models.voice_models import TTSRequest, TranscriptionResponse, TTSResponse

router = APIRouter(prefix="/api/voice", tags=["voice"])

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile):
    """Transcribe uploaded audio to text"""
    try:
        audio_data = await audio_file.read()
        transcription_service = VoiceTranscriptionService()
        text = await transcription_service.transcribe_audio(audio_data)
        
        return TranscriptionResponse(
            text=text,
            status="success",
            confidence=0.95  # OpenAI doesn't provide confidence, use default
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """Convert text to speech audio"""
    try:
        tts_service = TextToSpeechService()
        audio_data = await tts_service.generate_speech(
            text=request.text,
            voice=request.voice
        )
        
        # Return as base64 encoded audio
        import base64
        audio_base64 = base64.b64encode(audio_data).decode()
        
        return TTSResponse(
            audio_base64=audio_base64,
            format="mp3",
            duration_ms=len(audio_data) // 16  # Rough estimate
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voices")
async def get_voices():
    """Get available TTS voices"""
    tts_service = TextToSpeechService()
    voices = await tts_service.get_available_voices()
    return {"voices": voices}
```

### 3. Pydantic Models

```python
# backend/models/voice_models.py
from pydantic import BaseModel
from typing import Optional

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"
    speed: float = 1.0

class TranscriptionResponse(BaseModel):
    text: str
    status: str
    confidence: float

class TTSResponse(BaseModel):
    audio_base64: str
    format: str
    duration_ms: int

class WakeWordDetection(BaseModel):
    detected: bool
    confidence: float
    timestamp: int
```

## 🎙️ Frontend Voice Components

### 1. Voice Input Component

```jsx
// frontend/src/components/chat/VoiceInput.js
import React, { useState, useRef, useEffect } from 'react';
import { IconButton, Tooltip, Box } from '@mui/material';
import { Mic, Stop, MicOff } from '@mui/icons-material';

const VoiceInput = ({ 
  onTranscription, 
  isRecording, 
  onStartRecording, 
  onStopRecording,
  disabled = false 
}) => {
  const [audioLevel, setAudioLevel] = useState(0);
  const [permissionGranted, setPermissionGranted] = useState(null);
  
  useEffect(() => {
    checkMicrophonePermission();
  }, []);
  
  const checkMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setPermissionGranted(true);
      stream.getTracks().forEach(track => track.stop());
    } catch (error) {
      setPermissionGranted(false);
    }
  };
  
  if (permissionGranted === false) {
    return (
      <Tooltip title="Microphone access denied">
        <IconButton disabled>
          <MicOff />
        </IconButton>
      </Tooltip>
    );
  }
  
  return (
    <Box sx={{ position: 'relative' }}>
      <Tooltip title={isRecording ? "Stop recording" : "Start voice input"}>
        <IconButton
          onClick={isRecording ? onStopRecording : onStartRecording}
          disabled={disabled}
          color={isRecording ? "error" : "primary"}
          sx={{ 
            backgroundColor: isRecording ? 'error.main' : 'primary.main',
            color: 'white',
            '&:hover': {
              backgroundColor: isRecording ? 'error.dark' : 'primary.dark',
            },
            animation: isRecording ? 'pulse 1.5s infinite' : 'none',
            '@keyframes pulse': {
              '0%': { transform: 'scale(1)' },
              '50%': { transform: 'scale(1.1)' },
              '100%': { transform: 'scale(1)' }
            }
          }}
        >
          {isRecording ? <Stop /> : <Mic />}
        </IconButton>
      </Tooltip>
      
      {/* Audio level indicator */}
      {isRecording && (
        <Box
          sx={{
            position: 'absolute',
            bottom: -8,
            left: '50%',
            transform: 'translateX(-50%)',
            width: `${Math.max(20, audioLevel * 100)}%`,
            height: 2,
            backgroundColor: 'primary.main',
            borderRadius: 1,
            transition: 'width 0.1s ease'
          }}
        />
      )}
    </Box>
  );
};

export default VoiceInput;
```

### 2. Wake Word Detection Component

```jsx
// frontend/src/components/chat/WakeWordDetector.js
import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Chip } from '@mui/material';

const WakeWordDetector = ({ 
  isActive, 
  onWakeWordDetected, 
  onVoiceActivityDetected,
  wakeWord = "Codex" 
}) => {
  const [isListening, setIsListening] = useState(false);
  const [lastDetection, setLastDetection] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  
  useEffect(() => {
    if (isActive) {
      startContinuousListening();
    } else {
      stopContinuousListening();
    }
    
    return () => stopContinuousListening();
  }, [isActive]);
  
  const startContinuousListening = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setIsListening(true);
      
      // Setup audio analysis for voice activity detection
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      // Setup wake word detection
      setupWakeWordDetection(stream);
      
      // Start voice activity monitoring
      monitorVoiceActivity();
      
    } catch (error) {
      console.error('❌ Failed to start wake word detection:', error);
      setIsListening(false);
    }
  };
  
  const setupWakeWordDetection = (stream) => {
    mediaRecorderRef.current = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    const audioChunks = [];
    
    mediaRecorderRef.current.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };
    
    mediaRecorderRef.current.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      await processAudioForWakeWord(audioBlob);
      audioChunks.length = 0;
      
      // Restart recording for continuous detection
      if (isActive && mediaRecorderRef.current?.state === 'inactive') {
        mediaRecorderRef.current.start();
        setTimeout(() => {
          if (mediaRecorderRef.current?.state === 'recording') {
            mediaRecorderRef.current.stop();
          }
        }, 3000); // Process 3-second chunks
      }
    };
    
    // Start recording
    mediaRecorderRef.current.start();
    setTimeout(() => {
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    }, 3000);
  };
  
  const processAudioForWakeWord = async (audioBlob) => {
    try {
      // Send audio to backend for transcription
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'audio.webm');
      
      const response = await fetch('/api/voice/transcribe', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      // Check for wake word
      if (result.text && result.text.toLowerCase().includes(wakeWord.toLowerCase())) {
        setLastDetection(Date.now());
        onWakeWordDetected({
          detected: true,
          confidence: result.confidence || 0.8,
          timestamp: Date.now(),
          fullText: result.text
        });
      }
      
    } catch (error) {
      console.error('❌ Wake word processing failed:', error);
    }
  };
  
  const monitorVoiceActivity = () => {
    if (!analyserRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const checkVoiceActivity = () => {
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Calculate average volume
      const average = dataArray.reduce((a, b) => a + b) / bufferLength;
      
      // Voice activity threshold
      if (average > 30) { // Adjust threshold as needed
        onVoiceActivityDetected(true, average);
      } else {
        onVoiceActivityDetected(false, average);
      }
      
      if (isActive) {
        requestAnimationFrame(checkVoiceActivity);
      }
    };
    
    checkVoiceActivity();
  };
  
  const stopContinuousListening = () => {
    setIsListening(false);
    
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    if (audioContextRef.current?.state === 'running') {
      audioContextRef.current.close();
    }
  };
  
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Chip
        label={isListening ? `Listening for "${wakeWord}"` : 'Wake word disabled'}
        color={isListening ? 'primary' : 'default'}
        size="small"
        sx={{
          animation: isListening ? 'pulse 2s infinite' : 'none',
          '@keyframes pulse': {
            '0%': { opacity: 1 },
            '50%': { opacity: 0.7 },
            '100%': { opacity: 1 }
          }
        }}
      />
      
      {lastDetection && (
        <Typography variant="caption" color="success.main">
          Last detected: {new Date(lastDetection).toLocaleTimeString()}
        </Typography>
      )}
    </Box>
  );
};

export default WakeWordDetector;
```

### 3. Full Conversation Mode Component

```jsx
// frontend/src/components/chat/ConversationMode.js
import React, { useState, useEffect } from 'react';
import {
  Box,
  IconButton,
  Typography,
  Paper,
  Fade,
  LinearProgress
} from '@mui/material';
import {
  VolumeUp,
  VolumeOff,
  Settings,
  ExitToApp,
  Mic,
  MicOff
} from '@mui/icons-material';
import WakeWordDetector from './WakeWordDetector';
import VoiceVisualizer from './VoiceVisualizer';

const ConversationMode = ({ 
  isActive, 
  onToggle, 
  messages, 
  onSendMessage,
  isProcessing 
}) => {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentAudio, setCurrentAudio] = useState(null);
  const [voiceSettings, setVoiceSettings] = useState({
    voice: 'alloy',
    speed: 1.0,
    autoPlay: true
  });
  
  const handleWakeWordDetected = async (detection) => {
    console.log('🎤 Wake word detected:', detection);
    setIsListening(true);
    
    // Start recording for user's message
    // This would integrate with existing voice recording logic
  };
  
  const handleVoiceActivityDetected = (isActive, level) => {
    // Update voice visualizer
  };
  
  const playAIResponse = async (text) => {
    if (!voiceSettings.autoPlay) return;
    
    try {
      setIsSpeaking(true);
      
      const response = await fetch('/api/voice/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice: voiceSettings.voice,
          speed: voiceSettings.speed
        })
      });
      
      const result = await response.json();
      
      // Convert base64 to audio and play
      const audioBlob = new Blob(
        [Uint8Array.from(atob(result.audio_base64), c => c.charCodeAt(0))],
        { type: 'audio/mp3' }
      );
      
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      setCurrentAudio(audio);
      
      audio.onended = () => {
        setIsSpeaking(false);
        setCurrentAudio(null);
        URL.revokeObjectURL(audioUrl);
      };
      
      await audio.play();
      
    } catch (error) {
      console.error('❌ Audio playback failed:', error);
      setIsSpeaking(false);
    }
  };
  
  // Auto-play latest AI response
  useEffect(() => {
    const latestMessage = messages[messages.length - 1];
    if (latestMessage?.role === 'assistant' && latestMessage.content) {
      playAIResponse(latestMessage.content);
    }
  }, [messages, voiceSettings]);
  
  if (!isActive) return null;
  
  return (
    <Fade in={isActive}>
      <Paper
        elevation={8}
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 1300,
          backgroundColor: 'background.default',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}
      >
        {/* Header Controls */}
        <Box sx={{
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <Typography variant="h6" color="primary">
            🎤 Conversation Mode
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <IconButton
              onClick={() => setCurrentAudio(prev => {
                if (prev) {
                  prev.pause();
                  setIsSpeaking(false);
                }
                return null;
              })}
              disabled={!isSpeaking}
            >
              {isSpeaking ? <VolumeOff /> : <VolumeUp />}
            </IconButton>
            
            <IconButton>
              <Settings />
            </IconButton>
            
            <IconButton onClick={onToggle} color="error">
              <ExitToApp />
            </IconButton>
          </Box>
        </Box>
        
        {/* Wake Word Detection Status */}
        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
          <WakeWordDetector
            isActive={isActive}
            onWakeWordDetected={handleWakeWordDetected}
            onVoiceActivityDetected={handleVoiceActivityDetected}
            wakeWord="Codex"
          />
        </Box>
        
        {/* Voice Visualizer */}
        <Box sx={{ 
          flex: 1, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          flexDirection: 'column',
          gap: 4
        }}>
          <VoiceVisualizer
            isListening={isListening}
            isSpeaking={isSpeaking}
            isProcessing={isProcessing}
          />
          
          {/* Status Messages */}
          <Box sx={{ textAlign: 'center' }}>
            {isProcessing && (
              <Typography variant="h6" color="primary">
                🤔 Thinking...
              </Typography>
            )}
            
            {isSpeaking && (
              <Typography variant="h6" color="success.main">
                🗣️ Codex is speaking...
              </Typography>
            )}
            
            {isListening && (
              <Typography variant="h6" color="warning.main">
                👂 Listening...
              </Typography>
            )}
            
            {!isListening && !isSpeaking && !isProcessing && (
              <Typography variant="h6" color="text.secondary">
                💬 Say "Codex" to start
              </Typography>
            )}
          </Box>
        </Box>
        
        {/* Processing Indicator */}
        {isProcessing && (
          <Box sx={{ p: 2 }}>
            <LinearProgress />
          </Box>
        )}
        
        {/* Recent Messages */}
        <Box sx={{
          p: 2,
          maxHeight: 200,
          overflow: 'auto',
          borderTop: '1px solid',
          borderColor: 'divider'
        }}>
          {messages.slice(-3).map((message, index) => (
            <Box key={index} sx={{ mb: 1 }}>
              <Typography
                variant="body2"
                color={message.role === 'user' ? 'primary.main' : 'text.secondary'}
              >
                <strong>{message.role === 'user' ? 'You' : 'Codex'}:</strong> {message.content}
              </Typography>
            </Box>
          ))}
        </Box>
      </Paper>
    </Fade>
  );
};

export default ConversationMode;
```

### 4. Voice Visualizer Component

```jsx
// frontend/src/components/chat/VoiceVisualizer.js
import React, { useEffect, useRef } from 'react';
import { Box } from '@mui/material';

const VoiceVisualizer = ({ isListening, isSpeaking, isProcessing, audioLevel = 0 }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Base circle
      ctx.beginPath();
      ctx.arc(centerX, centerY, 50, 0, 2 * Math.PI);
      ctx.fillStyle = getCircleColor();
      ctx.fill();
      
      // Animated rings for voice activity
      if (isListening || isSpeaking) {
        const time = Date.now() * 0.01;
        for (let i = 0; i < 3; i++) {
          const radius = 60 + (i * 20) + Math.sin(time + i) * 10;
          const opacity = Math.max(0, 0.3 - i * 0.1);
          
          ctx.beginPath();
          ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
          ctx.strokeStyle = `rgba(25, 118, 210, ${opacity})`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      
      // Processing animation
      if (isProcessing) {
        const time = Date.now() * 0.02;
        for (let i = 0; i < 8; i++) {
          const angle = (i / 8) * 2 * Math.PI + time;
          const x = centerX + Math.cos(angle) * 70;
          const y = centerY + Math.sin(angle) * 70;
          
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, 2 * Math.PI);
          ctx.fillStyle = `rgba(25, 118, 210, ${Math.sin(time + i) * 0.5 + 0.5})`;
          ctx.fill();
        }
      }
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    const getCircleColor = () => {
      if (isProcessing) return '#f57c00'; // Orange for processing
      if (isSpeaking) return '#4caf50';   // Green for speaking
      if (isListening) return '#2196f3';  // Blue for listening
      return '#9e9e9e';                   // Gray for idle
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isListening, isSpeaking, isProcessing, audioLevel]);
  
  return (
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
      <canvas
        ref={canvasRef}
        width={300}
        height={300}
        style={{
          borderRadius: '50%',
          filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))'
        }}
      />
    </Box>
  );
};

export default VoiceVisualizer;
```

## 🎯 Integration with Existing Chat System

### 1. Enhanced ChatInputArea

```jsx
// frontend/src/components/chat/ChatInputArea.js (Enhanced)
import VoiceInput from './VoiceInput';
import ConversationMode from './ConversationMode';

const ChatInputArea = () => {
  const {
    // Existing chat state
    query, setQuery, sendMessage, isLoading,
    // New voice state
    voiceMode, setVoiceMode,
    conversationMode, setConversationMode,
    isRecording, startVoiceRecording, stopVoiceRecording
  } = useChatSidebar();
  
  return (
    <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
      {/* Existing model selection */}
      
      {/* Enhanced input area with voice */}
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
        <TextField
          ref={textFieldRef}
          multiline
          maxRows={4}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message or use voice input..."
          variant="outlined"
          size="small"
          fullWidth
          disabled={isLoading}
        />
        
        {/* Voice Input Button */}
        <VoiceInput
          onTranscription={(text) => setQuery(text)}
          isRecording={isRecording}
          onStartRecording={startVoiceRecording}
          onStopRecording={stopVoiceRecording}
          disabled={isLoading}
        />
        
        {/* Send Button */}
        <IconButton
          onClick={handleSendMessage}
          disabled={!query.trim() || isLoading}
          color="primary"
        >
          <Send />
        </IconButton>
        
        {/* Conversation Mode Toggle */}
        <IconButton
          onClick={() => setConversationMode(!conversationMode)}
          color={conversationMode ? "primary" : "default"}
          title="Toggle conversation mode"
        >
          <RecordVoiceOver />
        </IconButton>
      </Box>
      
      {/* Conversation Mode Overlay */}
      <ConversationMode
        isActive={conversationMode}
        onToggle={() => setConversationMode(false)}
        messages={messages}
        onSendMessage={sendMessage}
        isProcessing={isLoading}
      />
    </Box>
  );
};
```

### 2. Enhanced Voice State Management

```jsx
// frontend/src/hooks/useVoiceManager.js
import { useState, useRef, useCallback } from 'react';

const useVoiceManager = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const [conversationMode, setConversationMode] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  
  const startVoiceRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      audioChunksRef.current = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };
      
      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await processVoiceInput(audioBlob);
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      
    } catch (error) {
      console.error('❌ Voice recording failed:', error);
    }
  }, []);
  
  const stopVoiceRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, []);
  
  const processVoiceInput = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('audio_file', audioBlob, 'audio.webm');
      
      const response = await fetch('/api/voice/transcribe', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      return result.text;
      
    } catch (error) {
      console.error('❌ Voice transcription failed:', error);
      return null;
    }
  };
  
  return {
    isRecording,
    isPlaying,
    voiceMode,
    conversationMode,
    audioLevel,
    setVoiceMode,
    setConversationMode,
    startVoiceRecording,
    stopVoiceRecording
  };
};

export default useVoiceManager;
```

## ⚙️ Configuration Updates

### Backend Configuration

```python
# backend/config.py (additions)
class Settings(BaseSettings):
    # Existing settings...
    
    # Voice Configuration
    VOICE_ENABLED: bool = True
    WHISPER_MODEL: str = "whisper-1"  # OpenAI Whisper model
    TTS_MODEL: str = "tts-1"          # OpenAI TTS model  
    TTS_VOICE: str = "alloy"          # Default voice
    TTS_SPEED: float = 1.0            # Speech speed
    
    # Wake Word Configuration
    WAKE_WORD: str = "Codex"          # Wake word for conversation mode
    WAKE_WORD_ENABLED: bool = True    # Enable wake word detection
    WAKE_WORD_CONFIDENCE: float = 0.7 # Minimum confidence threshold
    
    # Audio Processing
    MAX_AUDIO_SIZE_MB: int = 25       # Maximum audio file size
    MAX_AUDIO_SIZE_BYTES: int = MAX_AUDIO_SIZE_MB * 1024 * 1024
    VOICE_TIMEOUT_SECONDS: int = 30   # Maximum recording time
    SILENCE_THRESHOLD_MS: int = 2000  # Auto-stop after silence
    AUDIO_SAMPLE_RATE: int = 16000    # Audio sample rate
    
    # Voice Activity Detection
    VAD_ENABLED: bool = True          # Enable voice activity detection
    VAD_THRESHOLD: float = 30.0       # Voice activity threshold
    VAD_BUFFER_SIZE: int = 1024       # Audio buffer size for analysis
```

### Docker Configuration

```yaml
# docker-compose.yml (environment additions)
environment:
  # Existing environment variables...
  
  # Voice Configuration
  - VOICE_ENABLED=${VOICE_ENABLED:-true}
  - WHISPER_MODEL=${WHISPER_MODEL:-whisper-1}
  - TTS_MODEL=${TTS_MODEL:-tts-1}
  - TTS_VOICE=${TTS_VOICE:-alloy}
  - WAKE_WORD=${WAKE_WORD:-Codex}
  - WAKE_WORD_ENABLED=${WAKE_WORD_ENABLED:-true}
```

### Requirements Update

```txt
# backend/requirements.txt (additions)
openai>=1.0.0               # Already exists for chat
pydub>=0.25.1              # Audio processing
librosa>=0.10.1            # Audio analysis (optional)
soundfile>=0.12.1          # Audio file handling
numpy>=1.24.0              # Audio data processing
```

## 🎯 Implementation Phases

### Phase 1: Basic Voice Input (Week 1-2)
**BULLY!** Start with the essential cavalry charge!

- [ ] Implement backend voice transcription service
- [ ] Create voice API endpoints
- [ ] Add voice input button to ChatInputArea
- [ ] Basic audio recording and transcription
- [ ] Integration with existing message flow
- [ ] Error handling and permissions

**Deliverables:**
- Working microphone button in chat
- Audio transcription via OpenAI Whisper
- Voice input integrated with text chat

### Phase 2: TTS Response (Week 3-4)
**By George!** Add the speaking artillery!

- [ ] Implement text-to-speech service  
- [ ] Add audio playback for AI responses
- [ ] Voice selection options
- [ ] Synchronized text/audio display
- [ ] Audio controls (play/pause/stop)

**Deliverables:**
- AI responses converted to speech
- Audio playback controls
- Voice customization options

### Phase 3: Wake Word Detection (Week 5-6)
**Trust busting for manual activation!**

- [ ] Implement wake word detection
- [ ] Continuous audio monitoring
- [ ] "Codex" wake word activation
- [ ] Voice activity detection
- [ ] Conversation state management

**Deliverables:**
- "Codex" wake word activation
- Hands-free conversation initiation
- Voice activity detection

### Phase 4: Full Conversation Mode (Week 7-8)
**The grand cavalry charge finale!**

- [ ] Full conversation mode UI
- [ ] Voice visualizer component
- [ ] Auto-pause after silence
- [ ] Conversation flow management
- [ ] Advanced voice controls

**Deliverables:**
- Complete conversation mode interface
- Visual audio feedback
- Seamless conversation flow

### Phase 5: Polish & Optimization (Week 9-10)
**Roosevelt's finishing touches!**

- [ ] Performance optimizations
- [ ] Browser compatibility testing
- [ ] Mobile device support
- [ ] Accessibility improvements
- [ ] Error recovery and fallbacks

**Deliverables:**
- Production-ready voice features
- Cross-platform compatibility
- Comprehensive error handling

## 🔧 Technical Considerations

### Performance Optimizations
- **Audio Compression**: Use efficient audio formats (WebM/Opus)
- **Streaming**: Consider streaming for long responses
- **Caching**: Cache TTS responses for repeated text
- **Lazy Loading**: Load voice components only when needed

### Browser Compatibility
- **Modern Browsers**: Chrome, Firefox, Safari, Edge support
- **Mobile Support**: iOS Safari, Android Chrome testing
- **Fallback Strategy**: Graceful degradation for unsupported browsers
- **Progressive Enhancement**: Voice as enhancement, not requirement

### Security & Privacy
- **Audio Privacy**: Clear user consent for microphone access
- **Data Handling**: Secure audio upload and processing
- **Storage**: No persistent audio storage (process and discard)
- **Rate Limiting**: Prevent abuse of voice APIs

### Accessibility
- **Screen Readers**: Compatible with assistive technologies
- **Keyboard Navigation**: Full keyboard accessibility
- **Visual Indicators**: Clear visual feedback for voice states
- **Alternative Access**: Text input always available

## 🎯 Success Metrics

### User Experience Goals
- **Seamless Integration**: Voice feels natural with existing chat
- **Low Latency**: < 3 seconds for transcription + response + TTS
- **High Accuracy**: > 95% transcription accuracy for clear speech
- **Intuitive Controls**: Minimal learning curve for voice features

### Technical Performance
- **Audio Quality**: Clear TTS output at multiple speeds
- **Error Handling**: Graceful failures with helpful messages
- **Resource Usage**: Minimal impact on chat performance
- **Wake Word Accuracy**: > 90% "Codex" detection rate

### Conversation Mode Metrics
- **Wake Word Response**: < 1 second activation time
- **Voice Activity Detection**: Accurate start/stop detection
- **Conversation Flow**: Natural turn-taking behavior
- **Audio Synchronization**: Text and speech perfectly aligned

## 🚀 Roosevelt's Voice Conversation Victory Plan

**BULLY!** This comprehensive implementation will transform the chat interface into a truly conversational experience! **By George!** Users will be able to:

1. **Quick Voice Input**: Click microphone, speak, get transcription
2. **AI Voice Responses**: Hear AI responses in natural speech  
3. **Wake Word Activation**: Say "Codex" to start hands-free conversation
4. **Full Conversation Mode**: Complete voice-driven chat experience
5. **Seamless Integration**: Voice enhances existing functionality

The implementation follows **Roosevelt's development principles**:
- **Docker-first development**: All running via `docker compose up --build`
- **Modular architecture**: Each component under 500 lines
- **Progressive enhancement**: Voice as addition, not replacement
- **Security-first**: Proper permissions and privacy handling
- **Structured outputs**: Pydantic models for all voice APIs

**Trust busting for typing limitations!** This voice conversation mode will charge forward with the efficiency of Roosevelt's Rough Riders, creating the most natural and intuitive chat experience possible! 🎤🤖⚡

---

*Roosevelt's Voice Conversation Cavalry - Making chat as natural as a fireside conversation!*
