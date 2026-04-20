import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Box, IconButton, TextField, Typography, Tooltip } from '@mui/material';
import { Close, KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';
import { markdownToPlainText } from '../../utils/chatUtils';

function messageHaystack(message) {
  const role = (message.role || message.type || '').toString();
  const content = markdownToPlainText(message.content || '');
  return `${role}\n${content}`.toLowerCase();
}

function computeMatchIndices(messages, q) {
  const trimmed = q.trim();
  if (!trimmed) return [];
  const needle = trimmed.toLowerCase();
  const indices = [];
  messages.forEach((m, i) => {
    if (messageHaystack(m).includes(needle)) indices.push(i);
  });
  return indices;
}

/**
 * Compact find bar for the active conversation: filter messages by plain-text content, jump with prev/next.
 */
export default function ChatInConversationSearchBar({
  messages,
  scrollContainerRef,
  open,
  onClose,
  onActiveMatchChange,
}) {
  const [query, setQuery] = useState('');
  const [activeMatchIdx, setActiveMatchIdx] = useState(0);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open) {
      setQuery('');
      setActiveMatchIdx(0);
      return;
    }
    const t = setTimeout(() => inputRef.current?.focus(), 50);
    return () => clearTimeout(t);
  }, [open]);

  const matchIndices = useMemo(() => computeMatchIndices(messages, query), [messages, query]);

  useEffect(() => {
    if (matchIndices.length === 0) {
      setActiveMatchIdx(0);
      return;
    }
    setActiveMatchIdx((i) => (i >= matchIndices.length ? 0 : i));
  }, [matchIndices]);

  useEffect(() => {
    if (!open) {
      onActiveMatchChange?.({ query: '', messageListIndex: null });
      return;
    }
    const q = query.trim();
    if (!q || matchIndices.length === 0) {
      onActiveMatchChange?.({ query: '', messageListIndex: null });
      return;
    }
    const idx = matchIndices[activeMatchIdx];
    onActiveMatchChange?.({ query: q, messageListIndex: idx });
  }, [open, query, activeMatchIdx, matchIndices, onActiveMatchChange]);

  useEffect(() => {
    if (!open || !query.trim() || matchIndices.length === 0) return;
    const listIndex = matchIndices[activeMatchIdx];
    const container = scrollContainerRef?.current;
    if (!container) return;

    const messageEl = container.querySelector(`[data-chat-message-index="${listIndex}"]`);
    messageEl?.scrollIntoView({ block: 'center', behavior: 'smooth', inline: 'nearest' });

    const scrollToMark = () => {
      const mark = container.querySelector('[data-chat-scroll-target="1"]');
      mark?.scrollIntoView({ block: 'center', behavior: 'smooth', inline: 'nearest' });
    };

    const raf = requestAnimationFrame(() => {
      requestAnimationFrame(scrollToMark);
    });
    const t = setTimeout(scrollToMark, 200);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(t);
    };
  }, [open, query, activeMatchIdx, matchIndices, scrollContainerRef]);

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
    setActiveMatchIdx(0);
  };

  const goNext = () => {
    if (matchIndices.length === 0) return;
    setActiveMatchIdx((i) => (i + 1) % matchIndices.length);
  };

  const goPrev = () => {
    if (matchIndices.length === 0) return;
    setActiveMatchIdx((i) => (i - 1 + matchIndices.length) % matchIndices.length);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) goPrev();
      else goNext();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  };

  if (!open) return null;

  const statusLabel = !query.trim()
    ? 'Type to search'
    : matchIndices.length === 0
      ? 'No matches'
      : `${activeMatchIdx + 1} / ${matchIndices.length}`;

  return (
    <Box
      sx={{
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        gap: 0.5,
        px: 1,
        py: 0.5,
        borderBottom: '1px solid',
        borderColor: 'divider',
        bgcolor: 'background.paper',
      }}
    >
      <TextField
        inputRef={inputRef}
        size="small"
        placeholder="Find in chat…"
        value={query}
        onChange={handleQueryChange}
        onKeyDown={handleKeyDown}
        variant="outlined"
        sx={{
          flex: 1,
          minWidth: 0,
          '& .MuiInputBase-root': { fontSize: '0.8125rem' },
        }}
        inputProps={{ 'aria-label': 'Find in chat' }}
      />
      <Typography variant="caption" color="text.secondary" sx={{ flexShrink: 0, minWidth: 72, textAlign: 'center' }}>
        {statusLabel}
      </Typography>
      <Tooltip title="Previous match (Shift+Enter)">
        <span>
          <IconButton size="small" onClick={goPrev} disabled={matchIndices.length === 0} aria-label="Previous match">
            <KeyboardArrowUp fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Next match (Enter)">
        <span>
          <IconButton size="small" onClick={goNext} disabled={matchIndices.length === 0} aria-label="Next match">
            <KeyboardArrowDown fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>
      <Tooltip title="Close search">
        <IconButton size="small" onClick={onClose} aria-label="Close search">
          <Close fontSize="small" />
        </IconButton>
      </Tooltip>
    </Box>
  );
}
