import React, { useState, useCallback, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  Box,
  Typography,
  IconButton,
  Link,
  Button,
} from '@mui/material';
import { Close, GitHub, MenuBook, BugReport } from '@mui/icons-material';
import apiService from '../services/apiService';

const GITHUB_BASE = 'https://github.com/adamsih300u/bastion';
const APACHE_LICENSE_URL = 'https://www.apache.org/licenses/LICENSE-2.0';

const DUCK_HOTSPOTS = [
  { id: 0, left: '28%', top: '32.9%', label: 'Left duck' },
  { id: 1, left: '42%', top: '25.3%', label: 'Center duck' },
  { id: 2, left: '79%', top: '33.5%', label: 'Right duck' },
];

// One quack per duck; paths under public/ so URLs are /audio/...
const QUACK_AUDIO_URLS = [
  '/audio/duck-left.mp3',
  '/audio/duck-center.mp3',
  '/audio/duck-right.mp3',
];

const AboutDialog = ({ open, onClose }) => {
  const [quackingDuck, setQuackingDuck] = useState(null);
  const [version, setVersion] = useState(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    apiService.get('/api/version').then((res) => {
      if (!cancelled && res?.version) setVersion(res.version);
    }).catch(() => {
      if (!cancelled) setVersion('—');
    });
    return () => { cancelled = true; };
  }, [open]);

  const handleDuckClick = useCallback((duckId) => {
    setQuackingDuck(duckId);
    const url = QUACK_AUDIO_URLS[duckId];
    if (url) {
      const audio = new Audio(url);
      audio.play().catch(() => {});
    }
    setTimeout(() => setQuackingDuck(null), 1200);
  }, []);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 2,
          '& .quack-bubble': {
            animation: 'quackFade 1.2s ease-out forwards',
          },
          '@keyframes quackFade': {
            '0%': { opacity: 0, transform: 'scale(0.8) translateY(4px)' },
            '15%': { opacity: 1, transform: 'scale(1) translateY(0)' },
            '85%': { opacity: 1 },
            '100%': { opacity: 0, transform: 'scale(0.9) translateY(-4px)' },
          },
          '@keyframes wobble': {
            '0%, 100%': { transform: 'rotate(0deg)' },
            '25%': { transform: 'rotate(-8deg)' },
            '75%': { transform: 'rotate(8deg)' },
          },
        },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          p: 2,
          borderBottom: '1px solid',
          borderColor: 'divider',
        }}
      >
        <Typography variant="h6">About Bastion Workspace</Typography>
        <IconButton onClick={onClose} aria-label="Close" size="small">
          <Close />
        </IconButton>
      </Box>
      <DialogContent sx={{ pt: 2, pb: 3 }}>
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
          <Box
            sx={{
              position: 'relative',
              width: 250,
              flexShrink: 0,
            }}
          >
            <Box
              component="img"
              src="/images/ducks.png"
              alt="Bastion mascots"
              sx={{
                width: '100%',
                height: 'auto',
                display: 'block',
                borderRadius: '50%',
              }}
            />
            {DUCK_HOTSPOTS.map(({ id, left, top, label }) => (
              <Box
                key={id}
                aria-label={label}
                role="button"
                tabIndex={0}
                onClick={() => handleDuckClick(id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleDuckClick(id);
                  }
                }}
                sx={{
                  position: 'absolute',
                  left,
                  top,
                  width: '12%',
                  height: '15%',
                  borderRadius: '50%',
                  cursor: 'pointer',
                  outline: 'none',
                  animation: quackingDuck === id ? 'wobble 0.4s ease-in-out' : 'none',
                  '&:hover': {
                    backgroundColor: 'rgba(255,255,255,0.15)',
                  },
                  '&:focus': {
                    outline: 'none',
                  },
                }}
              >
                {quackingDuck === id && (
                  <Typography
                    className="quack-bubble"
                    variant="caption"
                    sx={{
                      position: 'absolute',
                      bottom: '100%',
                      left: '50%',
                      transform: 'translateX(-50%)',
                      mb: 0.5,
                      whiteSpace: 'nowrap',
                      fontWeight: 600,
                      color: 'text.primary',
                    }}
                  >
                    Quack!
                  </Typography>
                )}
              </Box>
            ))}
          </Box>

          <Typography variant="h5" component="div">
            Bastion Workspace
          </Typography>
          <Typography variant="body2" color="text.secondary">
            v{version ?? '…'}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            The next generation nexus for personal and team work
          </Typography>

          <Box sx={{ width: '100%', my: 1 }}>
            <Box
              sx={{
                display: 'flex',
                flexWrap: 'wrap',
                justifyContent: 'center',
                gap: 1,
              }}
            >
              <Button
                component={Link}
                href={GITHUB_BASE}
                target="_blank"
                rel="noopener noreferrer"
                startIcon={<GitHub />}
                color="inherit"
                sx={{ textTransform: 'none' }}
              >
                Source Code
              </Button>
              <Button
                component={Link}
                href={`${GITHUB_BASE}/wiki`}
                target="_blank"
                rel="noopener noreferrer"
                startIcon={<MenuBook />}
                color="inherit"
                sx={{ textTransform: 'none' }}
              >
                Documentation
              </Button>
              <Button
                component={Link}
                href={`${GITHUB_BASE}/issues`}
                target="_blank"
                rel="noopener noreferrer"
                startIcon={<BugReport />}
                color="inherit"
                sx={{ textTransform: 'none' }}
              >
                Report Issue
              </Button>
            </Box>
          </Box>

          <Typography variant="caption" color="text.secondary" sx={{ textAlign: 'center' }}>
            <Link
              href={APACHE_LICENSE_URL}
              target="_blank"
              rel="noopener noreferrer"
              color="inherit"
              underline="hover"
            >
              Licensed under Apache-2.0
            </Link>
          </Typography>
          <Typography variant="caption" color="text.secondary">
            © 2024-2026 Bastion Contributors
          </Typography>
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default AboutDialog;
