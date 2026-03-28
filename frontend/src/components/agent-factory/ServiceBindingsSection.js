/**
 * Service Bindings: bind external email accounts to this agent profile.
 * Bound accounts get scoped playbook tools (email:<connection_id>:send_email, etc.).
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  Button,
  CircularProgress,
} from '@mui/material';
import { Email, Add, Delete } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

export default function ServiceBindingsSection({ profileId }) {
  const queryClient = useQueryClient();

  const { data: bindings = [], isLoading: bindingsLoading } = useQuery(
    ['agentFactoryServiceBindings', profileId],
    () => apiService.agentFactory.listServiceBindings(profileId),
    { enabled: !!profileId, retry: false }
  );

  const { data: availableData, isLoading: availableLoading } = useQuery(
    'agentFactoryAvailableEmailConnections',
    () => apiService.agentFactory.getAvailableEmailConnections(),
    { enabled: !!profileId, retry: false }
  );

  const createBindingMutation = useMutation(
    ({ profileId: id, connection_id, service_type }) =>
      apiService.agentFactory.createServiceBinding(id, { connection_id, service_type: service_type || 'email' }),
    {
      onSuccess: (_, { profileId: id }) => {
        queryClient.invalidateQueries(['agentFactoryServiceBindings', id]);
        queryClient.invalidateQueries(['agentFactoryActions', id]);
      },
    }
  );

  const deleteBindingMutation = useMutation(
    ({ profileId: id, bindingId }) => apiService.agentFactory.deleteServiceBinding(id, bindingId),
    {
      onSuccess: (_, { profileId: id }) => {
        queryClient.invalidateQueries(['agentFactoryServiceBindings', id]);
        queryClient.invalidateQueries(['agentFactoryActions', id]);
      },
    }
  );

  const connections = availableData?.connections ?? [];
  const boundIds = new Set((bindings || []).map((b) => Number(b.connection_id)));

  if (!profileId) return null;

  const loading = bindingsLoading || availableLoading;

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Email fontSize="small" />
          Connected accounts
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Bind email accounts to this agent. Scoped tools (e.g. &quot;Send email from alice@company.com&quot;) will
          appear in the playbook composer.
        </Typography>
        {loading ? (
          <Box sx={{ py: 2, display: 'flex', justifyContent: 'center' }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <List dense>
            {bindings.map((b) => (
              <ListItem
                key={b.id}
                secondaryAction={
                  <Button
                    size="small"
                    color="secondary"
                    startIcon={<Delete fontSize="small" />}
                    onClick={() =>
                      window.confirm('Unbind this account from the agent?') &&
                      deleteBindingMutation.mutate({ profileId, bindingId: b.id })
                    }
                  >
                    Unbind
                  </Button>
                }
              >
                <ListItemText
                  primary={b.display_name || b.account_identifier || `Connection ${b.connection_id}`}
                  secondary={b.provider ? `${b.account_identifier} (${b.provider})` : b.account_identifier}
                />
              </ListItem>
            ))}
            {connections
              .filter((c) => !boundIds.has(Number(c.id)))
              .map((c) => (
                <ListItem
                  key={c.id}
                  secondaryAction={
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<Add fontSize="small" />}
                      onClick={() =>
                        createBindingMutation.mutate({
                          profileId,
                          connection_id: Number(c.id),
                          service_type: 'email',
                        })
                      }
                      disabled={createBindingMutation.isLoading}
                    >
                      Bind
                    </Button>
                  }
                >
                  <ListItemText
                    primary={c.display_name || c.account_identifier || `Connection ${c.id}`}
                    secondary={c.provider ? `${c.account_identifier} (${c.provider})` : c.account_identifier}
                  />
                </ListItem>
              ))}
            {connections.length === 0 && (!bindings || bindings.length === 0) && (
              <ListItem>
                <ListItemText
                  primary="No email accounts"
                  secondary="Connect an email account in Settings to bind it here."
                />
              </ListItem>
            )}
          </List>
        )}
      </CardContent>
    </Card>
  );
}
