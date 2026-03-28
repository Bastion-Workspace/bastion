/**
 * Unified Monitors section: single card with collapsible accordions for
 * Team, Email, Folder, and Conversation watches.
 * Replaces four separate sprawling cards in the agent profile editor.
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Box,
  Chip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import GroupIcon from '@mui/icons-material/Group';
import EmailIcon from '@mui/icons-material/Email';
import FolderIcon from '@mui/icons-material/Folder';
import ChatIcon from '@mui/icons-material/Chat';

import LineWatchSection, { teamWatchSummary } from './LineWatchSection';
import EmailWatchSection, { emailWatchSummary } from './EmailWatchSection';
import FolderWatchSection, { folderWatchSummary } from './FolderWatchSection';
import ConversationWatchSection, { conversationWatchSummary } from './ConversationWatchSection';

const SECTIONS = [
  { key: 'lines', label: 'Lines', Icon: GroupIcon, Summary: teamWatchSummary, Component: LineWatchSection },
  { key: 'email', label: 'Email', Icon: EmailIcon, Summary: emailWatchSummary, Component: EmailWatchSection },
  { key: 'folders', label: 'Folders', Icon: FolderIcon, Summary: folderWatchSummary, Component: FolderWatchSection },
  { key: 'conversations', label: 'Conversations', Icon: ChatIcon, Summary: conversationWatchSummary, Component: ConversationWatchSection },
];

export default function MonitorsSection({ profile, onChange, readOnly = false }) {
  if (!profile) return null;

  const totalActive = SECTIONS.reduce((sum, s) => {
    const txt = s.Summary(profile);
    return sum + (txt ? 1 : 0);
  }, 0);

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent sx={{ pb: '12px !important' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <MonitorHeartIcon fontSize="small" color="action" />
          <Typography variant="h6" color="text.secondary" sx={{ flex: 1 }}>
            Monitors
          </Typography>
          {totalActive > 0 && (
            <Chip
              label={`${totalActive} active`}
              size="small"
              color="primary"
              variant="outlined"
            />
          )}
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Event triggers that automatically run this agent&apos;s playbook.
          {readOnly && ' This agent is read-only; unlock a custom agent to edit monitors.'}
        </Typography>

        {SECTIONS.map(({ key, label, Icon, Summary, Component }) => {
          const summary = Summary(profile);
          return (
            <Accordion
              key={key}
              disableGutters
              elevation={0}
              sx={{
                '&:before': { display: 'none' },
                border: 1,
                borderColor: 'divider',
                borderRadius: 1,
                mb: 1,
                '&:last-of-type': { mb: 0 },
                overflow: 'hidden',
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                sx={{ minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5, alignItems: 'center', gap: 1 } }}
              >
                <Icon fontSize="small" color="action" />
                <Typography variant="subtitle2">{label}</Typography>
                {summary && (
                  <Chip label={summary} size="small" variant="outlined" sx={{ ml: 'auto', mr: 1 }} />
                )}
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0, pb: 1.5 }}>
                <Box
                  sx={
                    readOnly
                      ? { pointerEvents: 'none', opacity: 0.72, userSelect: 'none' }
                      : undefined
                  }
                >
                  <Component profile={profile} onChange={onChange} compact />
                </Box>
              </AccordionDetails>
            </Accordion>
          );
        })}
      </CardContent>
    </Card>
  );
}
