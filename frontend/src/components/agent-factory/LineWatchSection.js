/**
 * Team Watch section: which teams this agent watches, trigger on new post, respond as comment/post.
 * Persists to profile.team_config.team_memberships; backend syncs to agent_team_watches on save.
 */

import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Switch,
  FormControlLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
} from '@mui/material';
import { useQuery } from 'react-query';
import apiService from '../../services/apiService';

export default function TeamWatchSection({ profile, onChange, compact }) {
  const { data: teamsData, isLoading } = useQuery(
    'agent-factory-teams',
    () => apiService.get('/api/agent-factory/lines'),
    { staleTime: 60 * 1000 }
  );
  const teams = Array.isArray(teamsData) ? teamsData : (teamsData?.teams ?? []);

  if (!profile) return null;

  const teamConfig = profile.team_config || {};
  const memberships = teamConfig.team_memberships || [];

  const getMembership = (lineId) =>
    memberships.find((m) => String(m.team_id) === String(lineId));

  const lineId = (team) => team.id ?? team.team_id;

  const setMemberships = (next) => {
    onChange({
      ...profile,
      team_config: { ...teamConfig, team_memberships: next },
    });
  };

  const handleWatchToggle = (lineId, enabled) => {
    if (enabled) {
      setMemberships([
        ...memberships.filter((m) => String(m.team_id) !== String(lineId)),
        { team_id: lineId, trigger_on_new_post: true, respond_as: 'comment' },
      ]);
    } else {
      setMemberships(memberships.filter((m) => String(m.team_id) !== String(lineId)));
    }
  };

  const handleTriggerToggle = (lineId, triggerOnNewPost) => {
    const next = memberships.map((m) =>
      String(m.team_id) === String(lineId) ? { ...m, trigger_on_new_post: triggerOnNewPost } : m
    );
    if (!next.find((m) => String(m.team_id) === String(lineId))) {
      next.push({ team_id: lineId, trigger_on_new_post: triggerOnNewPost, respond_as: 'comment' });
    }
    setMemberships(next);
  };

  const handleRespondAs = (lineId, respondAs) => {
    const next = memberships.map((m) =>
      String(m.team_id) === String(lineId) ? { ...m, respond_as: respondAs } : m
    );
    if (!next.find((m) => String(m.team_id) === String(lineId))) {
      next.push({ team_id: lineId, trigger_on_new_post: true, respond_as: respondAs });
    }
    setMemberships(next);
  };

  return (
    <Box>
      {!compact && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Watch teams and trigger on new posts. Use "Post to team" output to reply.
        </Typography>
      )}
      {isLoading ? (
        <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
          <CircularProgress size={24} />
        </Box>
      ) : teams.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No teams found. Join a team first to watch it here.
        </Typography>
      ) : (
        <List disablePadding dense>
          {teams.map((team) => {
            const id = lineId(team);
            const membership = getMembership(id);
            const watching = !!membership;
            return (
              <ListItem key={id} disablePadding sx={{ flexWrap: 'wrap', py: 0.5 }}>
                <ListItemText
                  primary={team.name || id}
                  primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                />
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                  <Switch
                    size="small"
                    checked={watching}
                    onChange={(e) => handleWatchToggle(id, e.target.checked)}
                  />
                  {watching && (
                    <>
                      <FormControlLabel
                        control={
                          <Switch
                            size="small"
                            checked={membership?.trigger_on_new_post !== false}
                            onChange={(e) => handleTriggerToggle(id, e.target.checked)}
                          />
                        }
                        label={<Typography variant="caption">Trigger on post</Typography>}
                      />
                      <FormControl size="small" sx={{ minWidth: 140 }}>
                        <InputLabel>Respond as</InputLabel>
                        <Select
                          value={membership?.respond_as || 'comment'}
                          label="Respond as"
                          onChange={(e) => handleRespondAs(id, e.target.value)}
                        >
                          <MenuItem value="comment">Comment</MenuItem>
                          <MenuItem value="post">New post</MenuItem>
                        </Select>
                      </FormControl>
                    </>
                  )}
                </Box>
              </ListItem>
            );
          })}
        </List>
      )}
    </Box>
  );
}

export function teamWatchSummary(profile) {
  const memberships = profile?.team_config?.team_memberships || [];
  return memberships.length ? `${memberships.length} team${memberships.length !== 1 ? 's' : ''}` : '';
}
