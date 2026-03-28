import React, { useMemo } from 'react';
import { IconButton, Tooltip } from '@mui/material';
import { VolumeUp, Stop } from '@mui/icons-material';
import { stripMarkdownForSpeech } from '../../utils/textForSpeech';
import { useTTS } from '../../hooks/useTTS';

const ReadAloudButton = ({ content, isUser = false }) => {
  const { isSpeaking, canSpeak, speak } = useTTS({ stopEventName: 'chat-read-aloud-stop' });
  const speechText = useMemo(() => stripMarkdownForSpeech(content), [content]);

  const handleToggleRead = async () => {
    if (!canSpeak || !speechText) return;
    if (isSpeaking) {
      window.dispatchEvent(new CustomEvent('chat-read-aloud-stop'));
      return;
    }
    try {
      await speak(speechText);
    } catch (error) {
      console.error('Read aloud failed:', error);
    }
  };

  const disabled = !canSpeak || !speechText;

  return (
    <Tooltip title={isSpeaking ? 'Stop reading' : 'Read aloud'}>
      <span>
        <IconButton
          size="small"
          onClick={handleToggleRead}
          disabled={disabled}
          sx={{
            color: isUser ? 'primary.contrastText' : 'text.secondary',
            '&:hover': {
              backgroundColor: isUser ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.04)',
            },
          }}
        >
          {isSpeaking ? <Stop fontSize="small" /> : <VolumeUp fontSize="small" />}
        </IconButton>
      </span>
    </Tooltip>
  );
};

export default ReadAloudButton;
