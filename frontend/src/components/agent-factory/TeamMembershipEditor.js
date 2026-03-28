/**
 * Add/remove agents from team, assign roles, reporting lines, and per-member additional tools.
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Card,
  CardContent,
  Collapse,
} from '@mui/material';
import { Add, Delete, ExpandMore, ExpandLess } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';
import CollapsibleToolPicker from './CollapsibleToolPicker';

const ROLE_OPTIONS = [
  { value: 'ceo', label: 'CEO' },
  { value: 'manager', label: 'Manager' },
  { value: 'worker', label: 'Worker' },
  { value: 'specialist', label: 'Specialist' },
];

function MemberRow({ member, reportsToOptions, actions, onRemove, onUpdate, updateLoading }) {
  const [expanded, setExpanded] = useState(false);
  const [advancedToolsOpen, setAdvancedToolsOpen] = useState(false);
  const [editRole, setEditRole] = useState(member.role || 'worker');
  const [editReportsTo, setEditReportsTo] = useState(member.reports_to || '');
  const [editAdditionalTools, setEditAdditionalTools] = useState(
    Array.isArray(member.additional_tools) ? [...member.additional_tools] : []
  );

  React.useEffect(() => {
    setEditRole(member.role || 'worker');
    setEditReportsTo(member.reports_to || '');
    setEditAdditionalTools(Array.isArray(member.additional_tools) ? [...member.additional_tools] : []);
  }, [member.id, member.role, member.reports_to, member.additional_tools]);

  const hasChanges =
    editRole !== (member.role || 'worker') ||
    (editReportsTo || '') !== (member.reports_to || '') ||
    JSON.stringify([...editAdditionalTools].sort()) !==
      JSON.stringify([...(member.additional_tools || [])].sort());

  const handleSave = () => {
    onUpdate({
      membershipId: member.id,
      body: {
        role: editRole,
        reports_to: editReportsTo || null,
        additional_tools: editAdditionalTools,
      },
    });
  };

  const toggleTool = (nextList) => {
    setEditAdditionalTools(Array.isArray(nextList) ? nextList : []);
  };

  return (
    <>
      <ListItem
        sx={{ flexWrap: 'wrap' }}
        secondaryAction={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0 }}>
            <IconButton
              size="small"
              onClick={() => setExpanded((e) => !e)}
              aria-label={expanded ? 'Collapse' : 'Expand'}
            >
              {expanded ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
            <IconButton
              edge="end"
              size="small"
              onClick={() => onRemove(member.id)}
              aria-label="Remove"
            >
              <Delete />
            </IconButton>
          </Box>
        }
      >
        <ListItemText
          primary={member.agent_name || member.agent_handle || member.agent_profile_id}
          secondary={
            <Box component="span" sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
              {member.agent_handle && <Typography component="span" variant="body2" color="text.secondary">@{member.agent_handle}</Typography>}
              <Typography component="span" variant="body2" color="text.secondary"> · </Typography>
              <Typography component="span" variant="body2" color="text.secondary">{member.role}</Typography>
              {(member.additional_tools || []).length > 0 && (
                <Typography component="span" variant="caption" color="text.secondary">
                  {' · +'}{(member.additional_tools || []).length} tools
                </Typography>
              )}
            </Box>
          }
        />
      </ListItem>
      <Collapse in={expanded} timeout="auto" unmountOnExit>
        <Box sx={{ pl: 2, pr: 2, pb: 2 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 1 }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Role</InputLabel>
              <Select
                value={editRole}
                label="Role"
                onChange={(e) => setEditRole(e.target.value)}
              >
                {ROLE_OPTIONS.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>Reports to</InputLabel>
              <Select
                value={editReportsTo}
                label="Reports to"
                onChange={(e) => setEditReportsTo(e.target.value)}
              >
                <MenuItem value="">— None (root)</MenuItem>
                {reportsToOptions.map((m) => (
                  <MenuItem key={m.id} value={m.id}>
                    {m.agent_name || m.agent_handle || m.id} ({m.role})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {hasChanges && (
              <Button
                size="small"
                variant="contained"
                onClick={handleSave}
                disabled={updateLoading}
              >
                Save
              </Button>
            )}
          </Box>
          <Box sx={{ mt: 1 }}>
            <Button
              size="small"
              startIcon={advancedToolsOpen ? <ExpandLess /> : <ExpandMore />}
              onClick={() => setAdvancedToolsOpen((o) => !o)}
              sx={{ textTransform: 'none', color: 'text.secondary', fontSize: '0.8125rem' }}
            >
              Advanced: Additional tools
            </Button>
            <Collapse in={advancedToolsOpen}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5, mt: 0.5 }}>
                Tools here are added on top of this agent&apos;s playbook and team packs. In most cases, configure tools in the playbook or team packs instead.
              </Typography>
              <CollapsibleToolPicker
                actions={actions}
                selectedTools={editAdditionalTools}
                onToggleTool={toggleTool}
              />
            </Collapse>
          </Box>
        </Box>
      </Collapse>
    </>
  );
}

export default function TeamMembershipEditor({ lineId, teamId, members, orgChart, onMembersChange }) {
  const resolvedLineId = lineId ?? teamId;
  const queryClient = useQueryClient();
  const [selectedProfileId, setSelectedProfileId] = useState('');
  const [role, setRole] = useState('worker');
  const [reportsTo, setReportsTo] = useState('');

  const { data: profiles = [] } = useQuery(
    'agentFactoryProfiles',
    () => apiService.agentFactory.listProfiles(),
    { retry: false }
  );

  const addMemberMutation = useMutation(
    (body) => apiService.agentFactory.addLineMember(resolvedLineId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeam', resolvedLineId]);
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries(['agentFactoryOrgChart', resolvedLineId]);
        setSelectedProfileId('');
        setRole('worker');
        setReportsTo('');
        onMembersChange?.();
      },
    }
  );

  const removeMemberMutation = useMutation(
    (membershipId) => apiService.agentFactory.removeLineMember(resolvedLineId, membershipId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeam', resolvedLineId]);
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries(['agentFactoryOrgChart', resolvedLineId]);
        onMembersChange?.();
      },
    }
  );

  const updateMemberMutation = useMutation(
    ({ membershipId, body }) =>
      apiService.agentFactory.updateLineMember(resolvedLineId, membershipId, body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['agentFactoryTeam', resolvedLineId]);
        queryClient.invalidateQueries('agentFactoryTeams');
        queryClient.invalidateQueries(['agentFactoryOrgChart', resolvedLineId]);
        onMembersChange?.();
      },
    }
  );

  const { data: actions = [] } = useQuery(
    'agentFactoryActions',
    () => apiService.agentFactory.getActions(),
    { staleTime: 60000 }
  );

  const memberProfileIds = (members || []).map((m) => m.agent_profile_id);
  const availableProfiles = profiles.filter((p) => !memberProfileIds.includes(p.id));

  const flattenMembersForReportsTo = (nodes) => {
    if (!nodes || !nodes.length) return [];
    const out = [];
    for (const n of nodes) {
      out.push({ id: n.id, agent_name: n.agent_name, agent_handle: n.agent_handle, role: n.role });
      if (n.children?.length) out.push(...flattenMembersForReportsTo(n.children));
    }
    return out;
  };
  const reportsToOptions = flattenMembersForReportsTo(orgChart || []);

  const handleAdd = () => {
    if (!selectedProfileId || !resolvedLineId) return;
    addMemberMutation.mutate({
      agent_profile_id: selectedProfileId,
      role,
      reports_to: reportsTo || null,
    });
  };

  const handleRemove = (membershipId) => {
    removeMemberMutation.mutate(membershipId);
  };

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Members
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'flex-end', mb: 2 }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Agent</InputLabel>
            <Select
              value={selectedProfileId}
              label="Agent"
              onChange={(e) => setSelectedProfileId(e.target.value)}
            >
              <MenuItem value="">Select agent</MenuItem>
              {availableProfiles.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  {p.name || p.handle || p.id}
                  {p.handle ? ` @${p.handle}` : ''}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Role</InputLabel>
            <Select value={role} label="Role" onChange={(e) => setRole(e.target.value)}>
              {ROLE_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <InputLabel>Reports to</InputLabel>
            <Select
              value={reportsTo}
              label="Reports to"
              onChange={(e) => setReportsTo(e.target.value)}
            >
              <MenuItem value="">— None (root)</MenuItem>
              {reportsToOptions.map((m) => (
                <MenuItem key={m.id} value={m.id}>
                  {m.agent_name || m.agent_handle || m.id} ({m.role})
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={handleAdd}
            disabled={!selectedProfileId || !resolvedLineId || addMemberMutation.isLoading}
          >
            Add
          </Button>
        </Box>
        {addMemberMutation.error && (
          <Typography color="error" variant="body2" sx={{ mb: 1 }}>
            {addMemberMutation.error.message}
          </Typography>
        )}
        <List dense>
          {(members || []).map((m) => (
            <MemberRow
              key={m.id}
              member={m}
              reportsToOptions={reportsToOptions}
              actions={actions}
              onRemove={handleRemove}
              onUpdate={updateMemberMutation.mutate}
              updateLoading={updateMemberMutation.isLoading}
            />
          ))}
        </List>
        {(!members || members.length === 0) && (
          <Typography variant="body2" color="text.secondary">
            No members yet. Add agents above.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}
