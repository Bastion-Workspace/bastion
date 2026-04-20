/**
 * In-document find bar: highlights matches in a DOM subtree (read-only viewers).
 * Optional cross-slide mode for PPTX counts matches across all slides and switches slides when navigating.
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Typography,
  Stack,
  InputAdornment,
} from '@mui/material';
import {
  Close,
  KeyboardArrowUp,
  KeyboardArrowDown,
} from '@mui/icons-material';
import { useTheme } from '../contexts/ThemeContext';

const STORAGE_KEY = 'viewer_last_search';

const MARK_CLASS = 'find-in-doc-highlight';
const MARK_ACTIVE_CLASS = 'find-in-doc-highlight-active';

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Collect text nodes under root (skip script/style/textarea and our marks). */
function collectTextNodes(root) {
  if (!root) return [];
  const out = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
      const el = node.parentElement;
      if (!el) return NodeFilter.FILTER_REJECT;
      if (el.closest('script, style, textarea, noscript')) return NodeFilter.FILTER_REJECT;
      if (el.closest(`mark.${MARK_CLASS}, mark.${MARK_ACTIVE_CLASS}`)) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  while (walker.nextNode()) {
    out.push(walker.currentNode);
  }
  return out;
}

export function clearFindHighlights(root) {
  if (!root) return;
  const marks = root.querySelectorAll(`mark.${MARK_CLASS}, mark.${MARK_ACTIVE_CLASS}`);
  marks.forEach((mark) => {
    const text = document.createTextNode(mark.textContent);
    mark.parentNode.replaceChild(text, mark);
  });
  root.normalize();
}

/**
 * Wrap all case-insensitive matches in root; set activeMarkIndex-th match as active (0-based). Returns number of marks.
 */
export function applyFindHighlights(root, query, activeMarkIndex = 0) {
  clearFindHighlights(root);
  const q = (query || '').trim();
  if (!q || !root) return 0;

  const re = new RegExp(escapeRegExp(q), 'gi');
  const textNodes = collectTextNodes(root);
  const marks = [];
  let matchCount = 0;

  for (const textNode of textNodes) {
    const text = textNode.nodeValue;
    if (!text) continue;
    re.lastIndex = 0;
    let lastIndex = 0;
    let m;
    const frag = document.createDocumentFragment();
    let hasFrag = false;
    while ((m = re.exec(text)) !== null) {
      if (m.index === lastIndex && m[0].length === 0) break;
      if (m.index > lastIndex) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex, m.index)));
      }
      const mark = document.createElement('mark');
      mark.className = MARK_CLASS;
      mark.textContent = m[0];
      if (matchCount === activeMarkIndex) {
        mark.classList.add(MARK_ACTIVE_CLASS);
      }
      frag.appendChild(mark);
      marks.push(mark);
      matchCount += 1;
      lastIndex = m.index + m[0].length;
      if (m[0].length === 0) re.lastIndex++;
      hasFrag = true;
    }
    if (hasFrag) {
      if (lastIndex < text.length) {
        frag.appendChild(document.createTextNode(text.slice(lastIndex)));
      }
      textNode.parentNode.replaceChild(frag, textNode);
    }
  }

  return marks.length;
}

function countMatchesInHtmlString(htmlString, query) {
  const q = (query || '').trim();
  if (!q || !htmlString) return 0;
  const div = document.createElement('div');
  div.innerHTML = htmlString;
  let count = 0;
  const re = new RegExp(escapeRegExp(q), 'gi');
  const textNodes = collectTextNodes(div);
  for (const textNode of textNodes) {
    const text = textNode.nodeValue || '';
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      count += 1;
      if (m[0].length === 0) break;
    }
  }
  return count;
}

/** From global index and per-slide counts, return { slideIndex, localIndex } */
function globalIndexToSlideLocal(perSlideCounts, globalIdx) {
  let remaining = globalIdx;
  for (let s = 0; s < perSlideCounts.length; s++) {
    const c = perSlideCounts[s];
    if (remaining < c) {
      return { slideIndex: s, localIndex: remaining };
    }
    remaining -= c;
  }
  const last = perSlideCounts.length - 1;
  if (last < 0) return { slideIndex: 0, localIndex: 0 };
  return { slideIndex: last, localIndex: Math.max(0, perSlideCounts[last] - 1) };
}

/**
 * @param {Object} props
 * @param {React.RefObject<HTMLElement|null>} props.containerRef - scroll/highlight root
 * @param {boolean} props.open
 * @param {() => void} props.onClose
 * @param {boolean} [props.darkMode] - optional override; else from ThemeContext
 * @param {{ slidesHtml: string[], slideIndex: number, setSlideIndex: (i: number) => void } | null} [props.crossSlideSearch] - PPTX: search all slides, navigate across slides
 */
export default function FindInDocumentBar({
  containerRef,
  open,
  onClose,
  darkMode: darkModeProp,
  crossSlideSearch = null,
}) {
  const { darkMode: ctxDark } = useTheme();
  const darkMode = darkModeProp !== undefined ? darkModeProp : ctxDark;
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [activeGlobalIndex, setActiveGlobalIndex] = useState(0);
  /** Match count after last highlight apply (single-root mode; cross-slide uses perSlideCounts). */
  const [singleRootMatchCount, setSingleRootMatchCount] = useState(0);
  const inputRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setDebouncedQuery(query);
      setActiveGlobalIndex(0);
    }, 150);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  const perSlideCounts = useMemo(() => {
    if (!crossSlideSearch || !crossSlideSearch.slidesHtml?.length) return null;
    const q = debouncedQuery.trim();
    if (!q) return crossSlideSearch.slidesHtml.map(() => 0);
    return crossSlideSearch.slidesHtml.map((html) => countMatchesInHtmlString(html, q));
  }, [crossSlideSearch, debouncedQuery]);

  const totalMatches = useMemo(() => {
    if (perSlideCounts) {
      return perSlideCounts.reduce((a, b) => a + b, 0);
    }
    return null;
  }, [perSlideCounts]);

  const { localActiveIndex, targetSlide } = useMemo(() => {
    if (perSlideCounts && crossSlideSearch) {
      const total = perSlideCounts.reduce((a, b) => a + b, 0);
      if (total === 0) {
        return { localActiveIndex: 0, targetSlide: crossSlideSearch.slideIndex };
      }
      const idx = Math.min(Math.max(0, activeGlobalIndex), total - 1);
      const { slideIndex, localIndex } = globalIndexToSlideLocal(perSlideCounts, idx);
      return { localActiveIndex: localIndex, targetSlide: slideIndex };
    }
    return { localActiveIndex: activeGlobalIndex, targetSlide: null };
  }, [perSlideCounts, crossSlideSearch, activeGlobalIndex]);

  // PPTX: switch slide when active match is on another slide
  useEffect(() => {
    if (!open || !crossSlideSearch || targetSlide === null) return;
    if (crossSlideSearch.slideIndex !== targetSlide) {
      crossSlideSearch.setSlideIndex(targetSlide);
    }
  }, [open, crossSlideSearch, targetSlide]);

  // Apply highlights in DOM
  useEffect(() => {
    if (!open) return;
    const root = containerRef?.current;
    if (!root) return;
    const q = debouncedQuery.trim();
    if (!q) {
      clearFindHighlights(root);
      setSingleRootMatchCount(0);
      return;
    }
    if (perSlideCounts && crossSlideSearch) {
      const onCurrent =
        crossSlideSearch.slideIndex === targetSlide || targetSlide === null;
      if (!onCurrent) {
        clearFindHighlights(root);
        setSingleRootMatchCount(0);
        return;
      }
      const n = applyFindHighlights(root, q, localActiveIndex);
      setSingleRootMatchCount(n);
      return;
    }
    const n = applyFindHighlights(root, q, activeGlobalIndex);
    setSingleRootMatchCount(n);
  }, [
    open,
    debouncedQuery,
    containerRef,
    activeGlobalIndex,
    localActiveIndex,
    perSlideCounts,
    crossSlideSearch,
    targetSlide,
  ]);

  // Scroll active mark into view
  useEffect(() => {
    if (!open || !containerRef?.current) return;
    const root = containerRef.current;
    const active = root.querySelector(`mark.${MARK_ACTIVE_CLASS}`);
    if (active && typeof active.scrollIntoView === 'function') {
      active.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
    }
  }, [open, containerRef, debouncedQuery, activeGlobalIndex, localActiveIndex, crossSlideSearch?.slideIndex]);

  useEffect(() => {
    if (open) {
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
          setQuery(saved);
          setDebouncedQuery(saved);
        }
      } catch {
        /* ignore */
      }
      const t = setTimeout(() => inputRef.current?.focus?.(), 50);
      return () => clearTimeout(t);
    }
    setQuery('');
    setDebouncedQuery('');
    setActiveGlobalIndex(0);
    setSingleRootMatchCount(0);
    const root = containerRef?.current;
    if (root) clearFindHighlights(root);
  }, [open, containerRef]);

  const persistQuery = useCallback((value) => {
    try {
      if (value.trim()) localStorage.setItem(STORAGE_KEY, value);
    } catch {
      /* ignore */
    }
  }, []);

  const displayTotal = totalMatches !== null ? totalMatches : singleRootMatchCount;
  const displayIndex = displayTotal > 0 ? Math.min(activeGlobalIndex + 1, displayTotal) : 0;

  const goPrev = useCallback(() => {
    const total = totalMatches !== null ? totalMatches : singleRootMatchCount;
    if (total <= 0) return;
    setActiveGlobalIndex((i) => (i - 1 + total) % total);
  }, [totalMatches, singleRootMatchCount]);

  const goNext = useCallback(() => {
    const total = totalMatches !== null ? totalMatches : singleRootMatchCount;
    if (total <= 0) return;
    setActiveGlobalIndex((i) => (i + 1) % total);
  }, [totalMatches, singleRootMatchCount]);

  useEffect(() => {
    if (!open) return undefined;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === 'Enter') {
        if (e.shiftKey) {
          e.preventDefault();
          goPrev();
        } else {
          e.preventDefault();
          goNext();
        }
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [open, onClose, goPrev, goNext]);

  if (!open) return null;

  const inactiveBg = darkMode ? 'rgba(255, 235, 59, 0.25)' : 'rgba(255, 235, 59, 0.4)';
  const activeBg = darkMode ? 'rgba(255, 152, 0, 0.4)' : 'rgba(255, 152, 0, 0.6)';

  return (
    <>
      <Box
        component="style"
        dangerouslySetInnerHTML={{
          __html: `
            mark.${MARK_CLASS} { background-color: ${inactiveBg}; color: inherit; }
            mark.${MARK_ACTIVE_CLASS} { background-color: ${activeBg}; color: inherit; outline: 2px solid ${darkMode ? '#ffb74d' : '#f57c00'}; }
          `,
        }}
      />
      <Box
        sx={{
          px: 1.5,
          py: 1,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'action.hover',
          flexShrink: 0,
        }}
      >
        <Stack direction="row" spacing={1} alignItems="center">
          <TextField
            inputRef={inputRef}
            size="small"
            placeholder="Find in document…"
            value={query}
            onChange={(e) => {
              const v = e.target.value;
              setQuery(v);
              persistQuery(v);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                if (e.shiftKey) goPrev();
                else goNext();
              }
            }}
            sx={{ flex: 1, maxWidth: 360 }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: 'nowrap' }}>
                    {displayTotal > 0 ? `${displayIndex} of ${displayTotal}` : debouncedQuery.trim() ? '0 matches' : ''}
                  </Typography>
                </InputAdornment>
              ),
            }}
            autoComplete="off"
          />
          <IconButton size="small" onClick={goPrev} aria-label="Previous match" disabled={displayTotal === 0}>
            <KeyboardArrowUp />
          </IconButton>
          <IconButton size="small" onClick={goNext} aria-label="Next match" disabled={displayTotal === 0}>
            <KeyboardArrowDown />
          </IconButton>
          <IconButton size="small" onClick={onClose} aria-label="Close find">
            <Close />
          </IconButton>
        </Stack>
      </Box>
    </>
  );
}

export const VIEWER_FIND_STORAGE_KEY = STORAGE_KEY;
