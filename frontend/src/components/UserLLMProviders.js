import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  InputAdornment,
  InputLabel,
  FormControl,
  Select,
  MenuItem,
  Paper,
} from '@mui/material';
import ExpandMore from '@mui/icons-material/ExpandMore';
import Search from '@mui/icons-material/Search';
import Key from '@mui/icons-material/Key';
import Add from '@mui/icons-material/Add';
import Delete from '@mui/icons-material/Delete';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const PROVIDER_LABELS = {
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
  ollama: 'Ollama',
  vllm: 'vLLM',
};

const PROVIDER_HINTS = {
  openai: 'API key from platform.openai.com',
  openrouter: 'API key from openrouter.ai (access many models including Anthropic)',
  ollama: 'Base URL e.g. http://localhost:11434 (no key required)',
  vllm: 'Base URL of your vLLM server e.g. http://localhost:8000',
};

export default function UserLLMProviders() {
  const queryClient = useQueryClient();
  const [useAdminModels, setUseAdminModels] = useState(true);
  const [addingProvider, setAddingProvider] = useState(null);
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [addError, setAddError] = useState('');

  const { data: useAdminData, isLoading: useAdminLoading } = useQuery(
    'useAdminModels',
    () => apiService.getUseAdminModels(),
    { onSuccess: (d) => d?.use_admin_models !== undefined && setUseAdminModels(d.use_admin_models) }
  );

  const { data: providersData, isLoading: providersLoading } = useQuery(
    'userLlmProviders',
    () => apiService.getUserLlmProviders(),
    { enabled: !useAdminModels }
  );

  const setUseAdminMutation = useMutation(
    (value) => apiService.setUseAdminModels(value),
    {
      onSuccess: (_, value) => {
        setUseAdminModels(value);
        queryClient.invalidateQueries('useAdminModels');
        queryClient.invalidateQueries('enabledModels');
        queryClient.invalidateQueries('availableModels');
      },
    }
  );

  const addProviderMutation = useMutation(
    (body) => apiService.addUserLlmProvider(body),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('userLlmProviders');
        queryClient.invalidateQueries('enabledModels');
        queryClient.invalidateQueries('availableModels');
        setAddingProvider(null);
        setApiKey('');
        setBaseUrl('');
        setAddError('');
      },
      onError: (err) => {
        setAddError(err?.message || err?.response?.data?.detail || 'Failed to add provider');
      },
    }
  );

  const removeProviderMutation = useMutation(
    (providerId) => apiService.removeUserLlmProvider(providerId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('userLlmProviders');
        queryClient.invalidateQueries('enabledModels');
        queryClient.invalidateQueries('availableModels');
      },
    }
  );

  const handleConnect = (providerType) => {
    setAddError('');
    const body = { provider_type: providerType };
    if (providerType === 'openai' || providerType === 'openrouter') {
      body.api_key = apiKey?.trim() || undefined;
    } else {
      body.base_url = baseUrl?.trim() || undefined;
      if (apiKey?.trim()) body.api_key = apiKey.trim();
    }
    if ((providerType === 'ollama' || providerType === 'vllm') && !body.base_url) {
      setAddError('URL is required for Ollama and vLLM');
      return;
    }
    addProviderMutation.mutate(body);
  };

  const providers = providersData?.providers || [];
  const activeUseAdmin = useAdminData?.use_admin_models ?? useAdminModels;

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <Key sx={{ mr: 1 }} />
          <Typography variant="h6">My AI Providers</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Use shared admin-configured models, or your own API keys and self-hosted endpoints.
        </Typography>

        {useAdminLoading ? (
          <CircularProgress size={24} sx={{ mt: 1 }} />
        ) : (
          <>
            <FormControlLabel
              control={
                <Switch
                  checked={!activeUseAdmin}
                  onChange={(e) => setUseAdminMutation.mutate(!e.target.checked)}
                  disabled={setUseAdminMutation.isLoading}
                />
              }
              label="Use my own API keys"
            />

            {!activeUseAdmin && (
              <Box sx={{ mt: 2 }}>
                {addError && (
                  <Alert severity="error" onClose={() => setAddError('')} sx={{ mb: 2 }}>
                    {addError}
                  </Alert>
                )}

                {providersLoading ? (
                  <CircularProgress size={24} />
                ) : (
                  <>
                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                      Add provider
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
                      {['openai', 'openrouter', 'ollama', 'vllm'].map((type) => {
                        const isCloud = type === 'openai' || type === 'openrouter';
                        const hasProvider = isCloud && providers.some((p) => p.provider_type === type);
                        const isAdding = addingProvider === type;
                        return (
                          <Chip
                            key={type}
                            label={PROVIDER_LABELS[type]}
                            onClick={() => setAddingProvider(isAdding ? null : type)}
                            color={isAdding ? 'primary' : 'default'}
                            variant={isAdding ? 'filled' : 'outlined'}
                            disabled={hasProvider}
                          />
                        );
                      })}
                    </Box>

                    {addingProvider && (
                      <Box sx={{ p: 2, bgcolor: 'action.hover', borderRadius: 1, mb: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          {PROVIDER_LABELS[addingProvider]} — {PROVIDER_HINTS[addingProvider]}
                        </Typography>
                        {(addingProvider === 'ollama' || addingProvider === 'vllm') && (
                          <TextField
                            fullWidth
                            size="small"
                            label="Base URL"
                            placeholder={addingProvider === 'ollama' ? 'http://localhost:11434' : 'http://localhost:8000'}
                            value={baseUrl}
                            onChange={(e) => setBaseUrl(e.target.value)}
                            sx={{ mt: 1, mb: 1 }}
                          />
                        )}
                        <TextField
                          fullWidth
                          size="small"
                          type="password"
                          label={addingProvider === 'ollama' || addingProvider === 'vllm' ? 'API key (optional)' : 'API key'}
                          value={apiKey}
                          onChange={(e) => setApiKey(e.target.value)}
                          sx={{ mt: 1, mb: 1 }}
                        />
                        <Button
                          size="small"
                          variant="contained"
                          startIcon={addProviderMutation.isLoading ? <CircularProgress size={16} /> : <Add />}
                          onClick={() => handleConnect(addingProvider)}
                          disabled={addProviderMutation.isLoading}
                        >
                          Connect
                        </Button>
                      </Box>
                    )}

                    {providers.length > 0 && (
                      <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                        Your providers
                      </Typography>
                    )}
                    {providers.map((p) => (
                      <ProviderModelsCard
                        key={p.id}
                        provider={p}
                        onRemove={() => removeProviderMutation.mutate(p.id)}
                        onEnabledChange={() => queryClient.invalidateQueries('enabledModels')}
                      />
                    ))}
                  </>
                )}
              </Box>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

/** Match backend composite "provider_id:rest" where rest may contain ':' (e.g. ollama tags). */
function rawModelIdFromComposite(compositeModelId, providerIdNum) {
  const s = String(compositeModelId || '');
  const prefix = `${providerIdNum}:`;
  if (s.startsWith(prefix)) {
    return s.slice(prefix.length);
  }
  const i = s.indexOf(':');
  if (i >= 0 && /^\d+$/.test(s.slice(0, i))) {
    return s.slice(i + 1);
  }
  return s;
}

function ProviderModelsCard({ provider, onRemove, onEnabledChange }) {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState(false);
  const [search, setSearch] = useState('');
  const [subProviderFilter, setSubProviderFilter] = useState('');
  const [showOnlyEnabled, setShowOnlyEnabled] = useState(false);
  const providerIdNum = Number(provider.id);

  const { data: modelsData, isLoading: modelsLoading } = useQuery(
    ['userLlmProviderModels', provider.id],
    () => apiService.getUserLlmProviderModels(provider.id),
    { enabled: expanded }
  );

  const { data: userEnabledData } = useQuery(
    'userEnabledModels',
    () => apiService.getUserEnabledModels(),
    { enabled: expanded }
  );

  const models = modelsData?.models || [];
  const userEnabled = userEnabledData?.enabled_models || [];
  // API returns composite model_id "provider_id:model_id"; provider models use raw id. Strip prefix for this provider.
  const enabledForThisProvider = userEnabled
    .filter((e) => Number(e.provider_id) === providerIdNum)
    .map((e) => rawModelIdFromComposite(e.model_id, providerIdNum));

  const subProviders = useMemo(
    () =>
      [...new Set(models.map((m) => (m.id.includes('/') ? m.id.split('/')[0] : '')))].filter(Boolean).sort(),
    [models]
  );

  const grouped = useMemo(() => {
    let list = models;
    if (search) list = list.filter((m) => (m.name || m.id).toLowerCase().includes(search.toLowerCase()));
    if (subProviderFilter) list = list.filter((m) => m.id.startsWith(subProviderFilter + '/'));
    if (showOnlyEnabled) list = list.filter((m) => enabledForThisProvider.includes(m.id));
    return list.reduce((acc, m) => {
      const grp = m.id.includes('/') ? m.id.split('/')[0] : 'Other';
      acc[grp] = acc[grp] || [];
      acc[grp].push(m);
      return acc;
    }, {});
  }, [models, search, subProviderFilter, showOnlyEnabled, enabledForThisProvider]);

  const filteredCount = useMemo(
    () => Object.values(grouped).reduce((sum, arr) => sum + arr.length, 0),
    [grouped]
  );
  const enabledInFiltered = useMemo(
    () => Object.values(grouped).flat().filter((m) => enabledForThisProvider.includes(m.id)).length,
    [grouped, enabledForThisProvider]
  );

  const staleEnabledRawIds = useMemo(
    () => enabledForThisProvider.filter((id) => !models.some((m) => m.id === id)),
    [enabledForThisProvider, models]
  );

  const setEnabledMutation = useMutation(
    (modelIds) => apiService.setUserEnabledModels(provider.id, modelIds),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('userEnabledModels');
        queryClient.invalidateQueries('enabledModels');
        queryClient.invalidateQueries('availableModels');
        queryClient.invalidateQueries(['userLlmProviderModels', provider.id]);
        onEnabledChange?.();
      },
    }
  );

  const handleToggle = (modelId) => {
    const next = new Set(enabledForThisProvider);
    if (next.has(modelId)) next.delete(modelId);
    else next.add(modelId);
    setEnabledMutation.mutate(Array.from(next));
  };

  const pruneStaleEnabled = () => {
    const kept = enabledForThisProvider.filter((id) => models.some((m) => m.id === id));
    setEnabledMutation.mutate(kept);
  };

  return (
    <Accordion expanded={expanded} onChange={() => setExpanded(!expanded)} sx={{ mb: 1 }}>
      <AccordionSummary expandIcon={<ExpandMore />}>
        <Box display="flex" alignItems="center" justifyContent="space-between" width="100%" pr={1}>
          <Typography>
            {provider.display_name || PROVIDER_LABELS[provider.provider_type] || provider.provider_type}
            {provider.base_url && (
              <Typography component="span" variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                {provider.base_url}
              </Typography>
            )}
          </Typography>
          <Button
            size="small"
            color="error"
            startIcon={<Delete />}
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
          >
            Remove
          </Button>
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        {modelsLoading ? (
          <CircularProgress size={24} />
        ) : models.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No models returned from this provider. Check the URL and key.
          </Typography>
        ) : (
          <>
            {staleEnabledRawIds.length > 0 && (
              <Alert
                severity="warning"
                sx={{ mb: 2 }}
                action={
                  <Button color="inherit" size="small" onClick={pruneStaleEnabled} disabled={setEnabledMutation.isLoading}>
                    Remove stale
                  </Button>
                }
              >
                <Typography variant="body2">
                  {staleEnabledRawIds.length} enabled model(s) are not in this provider&apos;s current list (renamed or removed).
                  Remove them to keep your enabled set accurate.
                </Typography>
              </Alert>
            )}
            <Box sx={{ mb: 2 }}>
              <Box display="flex" flexWrap="wrap" gap={2} alignItems="center" sx={{ mb: 2 }}>
                <TextField
                  size="small"
                  placeholder="Search models..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <Search sx={{ color: 'text.secondary' }} />
                      </InputAdornment>
                    ),
                  }}
                  sx={{ minWidth: 200 }}
                />
                <FormControl size="small" sx={{ minWidth: 160 }}>
                  <InputLabel shrink>Provider</InputLabel>
                  <Select
                    value={subProviderFilter}
                    onChange={(e) => setSubProviderFilter(e.target.value)}
                    label="Provider"
                    displayEmpty
                    notched
                  >
                    <MenuItem value="">
                      <em>All</em>
                    </MenuItem>
                    {subProviders.map((sp) => (
                      <MenuItem key={sp} value={sp}>
                        {sp}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={showOnlyEnabled}
                      onChange={(e) => setShowOnlyEnabled(e.target.checked)}
                    />
                  }
                  label="Show only enabled"
                />
                <Typography variant="body2" color="text.secondary">
                  {filteredCount} models / {enabledInFiltered} enabled
                </Typography>
              </Box>
            </Box>
            {Object.keys(grouped).length === 0 ? (
              <Paper sx={{ p: 3, textAlign: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  No models match the current filters.
                </Typography>
              </Paper>
            ) : (
              Object.entries(grouped).map(([grp, grpModels]) => {
                const enabledInGrp = grpModels.filter((m) => enabledForThisProvider.includes(m.id)).length;
                return (
                  <Accordion
                    key={grp}
                    defaultExpanded={enabledInGrp > 0}
                    sx={{ '&:before': { display: 'none' }, mb: 0.5 }}
                  >
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2">{grp}</Typography>
                        <Chip label={`${grpModels.length} models`} size="small" variant="outlined" />
                        <Chip label={`${enabledInGrp} enabled`} size="small" color="primary" />
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails sx={{ pt: 0 }}>
                      <List dense>
                        {grpModels.map((m) => (
                          <ListItem key={m.id}>
                            <ListItemText primary={m.name || m.id} secondary={m.id} />
                            <ListItemSecondaryAction>
                              <Switch
                                edge="end"
                                checked={enabledForThisProvider.includes(m.id)}
                                onChange={() => handleToggle(m.id)}
                                disabled={setEnabledMutation.isLoading}
                              />
                            </ListItemSecondaryAction>
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>
                );
              })
            )}
          </>
        )}
      </AccordionDetails>
    </Accordion>
  );
}
