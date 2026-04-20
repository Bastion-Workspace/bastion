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
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import VolumeUp from '@mui/icons-material/VolumeUp';
import Mic from '@mui/icons-material/Mic';
import Add from '@mui/icons-material/Add';
import Delete from '@mui/icons-material/Delete';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const TTS_TYPES = [
  { id: 'elevenlabs', label: 'ElevenLabs' },
  { id: 'hedra', label: 'Hedra' },
  { id: 'openai', label: 'OpenAI TTS' },
];
const STT_TYPES = [
  { id: 'openai', label: 'OpenAI (Whisper)' },
  { id: 'whisper_api', label: 'Whisper API (OpenAI-compatible)' },
  { id: 'deepgram', label: 'Deepgram' },
];

/** Built-in TTS: local services only (server default uses deployer-configured cloud/local). */
const ADMIN_LOCAL_TTS_CHIPS = [
  { id: 'server', label: 'Server default' },
  { id: 'piper', label: 'Piper (local)' },
  { id: 'browser', label: 'Browser TTS' },
];

/** Must match backend utils/elevenlabs_tts_model.py allowlist. */
const ELEVENLABS_TTS_MODEL_OPTIONS = [
  { value: '', label: 'Platform default' },
  { value: 'eleven_flash_v2_5', label: 'Flash (faster, lower cost)' },
  { value: 'eleven_turbo_v2_5', label: 'Turbo' },
  { value: 'eleven_multilingual_v2', label: 'Multilingual v2' },
  { value: 'eleven_v3', label: 'Eleven v3' },
];

export default function UserVoiceProviders() {
  const queryClient = useQueryClient();
  const [addError, setAddError] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [adding, setAdding] = useState(null);

  const { data: settingsData, isLoading: settingsLoading } = useQuery(
    'userVoiceSettings',
    () => apiService.getUserVoiceSettings()
  );

  const { data: providersData, isLoading: providersLoading } = useQuery(
    'userVoiceProviders',
    () => apiService.getUserVoiceProviders()
  );

  const useAdminTts = settingsData?.use_admin_tts !== false;
  const useAdminStt = settingsData?.use_admin_stt !== false;
  const ttsProviderId = settingsData?.user_tts_provider_id || '';
  const sttProviderId = settingsData?.user_stt_provider_id || '';
  const ttsVoiceId = settingsData?.user_tts_voice_id || '';
  const byokPiperVoiceId = settingsData?.user_byok_tts_voice_piper || '';
  const byokTtsEngine = (settingsData?.user_byok_tts_engine || 'cloud').toLowerCase();

  const ttsProviders = useMemo(
    () => (providersData?.providers || []).filter((p) => p.provider_role === 'tts'),
    [providersData]
  );
  const sttProviders = useMemo(
    () => (providersData?.providers || []).filter((p) => p.provider_role === 'stt'),
    [providersData]
  );

  const notifyVoicePrefsChanged = () => {
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('bastion-voice-settings-changed'));
    }
  };

  const setSettingsMutation = useMutation(
    (body) => apiService.setUserVoiceSettings(body),
    {
      onSuccess: (payload) => {
        if (payload && typeof payload === 'object' && 'use_admin_tts' in payload) {
          queryClient.setQueryData('userVoiceSettings', payload);
        }
        queryClient.invalidateQueries('userVoiceSettings');
        queryClient.invalidateQueries('userVoiceProviders');
        queryClient.invalidateQueries('voiceVoicesAdmin');
        queryClient.invalidateQueries('voiceVoicesByokPiper');
        notifyVoicePrefsChanged();
      },
    }
  );

  const addMutation = useMutation(
    (body) => apiService.addUserVoiceProvider(body),
    {
      onSuccess: (data, variables) => {
        queryClient.invalidateQueries('userVoiceProviders');
        setAdding(null);
        setApiKey('');
        setBaseUrl('');
        setAddError('');
        const newId = data?.provider_id;
        const role = variables?.provider_role;
        if (role === 'tts' && newId) {
          setSettingsMutation.mutate({
            use_admin_tts: false,
            user_byok_tts_engine: 'cloud',
            user_tts_provider_id: String(newId),
          });
        }
        if (role === 'stt' && newId) {
          setSettingsMutation.mutate({
            use_admin_stt: false,
            user_stt_provider_id: String(newId),
          });
        }
      },
      onError: (err) => {
        setAddError(
          err?.message || err?.response?.data?.detail || 'Failed to add provider'
        );
      },
    }
  );

  const removeMutation = useMutation(
    (id) => apiService.removeUserVoiceProvider(id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('userVoiceProviders');
        queryClient.invalidateQueries('userVoiceSettings');
        notifyVoicePrefsChanged();
      },
    }
  );

  const { data: voicesData, isLoading: voicesLoading } = useQuery(
    ['userVoiceProviderVoices', ttsProviderId],
    () => apiService.getUserVoiceProviderVoices(Number(ttsProviderId)),
    {
      enabled:
        Boolean(ttsProviderId) &&
        !useAdminTts &&
        byokTtsEngine === 'cloud',
    }
  );
  const voices = voicesData?.voices || [];

  const { data: byokPiperVoices = [], isLoading: byokPiperLoading } = useQuery(
    ['voiceVoicesByokPiper'],
    () => apiService.voice.listVoices('piper'),
    { enabled: !settingsLoading && !useAdminTts && byokTtsEngine === 'piper' }
  );

  const adminProvRaw = (settingsData?.user_admin_tts_provider ?? '').trim();
  const adminProvLower = adminProvRaw.toLowerCase();
  const adminLocalChipId =
    adminProvLower === 'piper' || adminProvLower === 'browser' ? adminProvLower : 'server';
  const adminVoicesListProvider =
    adminProvLower === 'browser' ? null : adminProvLower === 'piper' ? 'piper' : '';
  const { data: adminVoices = [], isLoading: adminVoicesLoading } = useQuery(
    ['voiceVoicesAdmin', adminVoicesListProvider ?? 'none'],
    () => apiService.voice.listVoices(adminVoicesListProvider || ''),
    { enabled: !settingsLoading && useAdminTts && adminVoicesListProvider !== null }
  );

  const adminServerVoiceId = settingsData?.user_admin_tts_voice_server ?? '';
  const adminPiperVoiceId = settingsData?.user_admin_tts_voice_piper ?? '';
  const voiceForAdminBackend =
    adminLocalChipId === 'piper' ? adminPiperVoiceId : adminServerVoiceId;
  const adminVoiceSelectValue =
    adminLocalChipId === 'browser'
      ? ''
      : adminVoices.some((v) => v.voice_id === voiceForAdminBackend)
        ? voiceForAdminBackend
        : '';

  const byokPiperResolved = byokPiperVoiceId || ttsVoiceId;
  const byokPiperSelectValue = byokPiperVoices.some((v) => v.voice_id === byokPiperResolved)
    ? byokPiperResolved
    : '';

  const adminSelectedVoice = useMemo(
    () => adminVoices.find((v) => v.voice_id === voiceForAdminBackend),
    [adminVoices, voiceForAdminBackend]
  );
  const adminUsesElevenLabs =
    useAdminTts &&
    adminLocalChipId !== 'browser' &&
    adminLocalChipId !== 'piper' &&
    (adminProvLower === 'elevenlabs' ||
      (adminSelectedVoice?.provider || '').toLowerCase() === 'elevenlabs');

  const adminUsesHedra =
    useAdminTts &&
    adminLocalChipId !== 'browser' &&
    adminLocalChipId !== 'piper' &&
    (adminProvLower === 'hedra' ||
      (adminSelectedVoice?.provider || '').toLowerCase() === 'hedra');

  const byokSelectedTtsProvider = useMemo(
    () => ttsProviders.find((p) => String(p.id) === String(ttsProviderId)),
    [ttsProviders, ttsProviderId]
  );
  const byokUsesElevenLabs =
    !useAdminTts &&
    byokTtsEngine === 'cloud' &&
    (byokSelectedTtsProvider?.provider_type || '').toLowerCase() === 'elevenlabs';

  const byokUsesHedra =
    !useAdminTts &&
    byokTtsEngine === 'cloud' &&
    (byokSelectedTtsProvider?.provider_type || '').toLowerCase() === 'hedra';

  const userElevenlabsModelId = settingsData?.user_elevenlabs_tts_model_id ?? '';
  const userAdminElevenlabsModelId = settingsData?.user_admin_elevenlabs_tts_model_id ?? '';
  const userHedraModelId = settingsData?.user_hedra_tts_model_id ?? '';
  const userAdminHedraModelId = settingsData?.user_admin_hedra_tts_model_id ?? '';

  const hedraModelsEnabled = adminUsesHedra || byokUsesHedra;
  const { data: hedraModelsData, isLoading: hedraModelsLoading } = useQuery(
    ['hedraTtsModels', hedraModelsEnabled, useAdminTts, ttsProviderId],
    () => apiService.getUserHedraTtsModels(),
    { enabled: hedraModelsEnabled, retry: false }
  );
  const hedraModels = hedraModelsData?.models || [];

  const handleAdd = () => {
    setAddError('');
    if (!adding) return;
    const { role, type } = adding;
    if (!apiKey.trim()) {
      setAddError('API key is required');
      return;
    }
    addMutation.mutate({
      provider_type: type,
      provider_role: role,
      api_key: apiKey.trim(),
      base_url: baseUrl.trim() || undefined,
    });
  };

  if (settingsLoading) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <CircularProgress size={24} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <VolumeUp sx={{ mr: 1 }} />
          <Typography variant="h6">Voice and speech (your keys)</Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Use the server default voice and transcription, or your own TTS/STT API keys (e.g.
          ElevenLabs, OpenAI, Deepgram).
        </Typography>

        {addError && (
          <Alert severity="error" onClose={() => setAddError('')} sx={{ my: 2 }}>
            {addError}
          </Alert>
        )}

        <Box sx={{ mt: 2, mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <VolumeUp sx={{ mr: 0.5, fontSize: 18 }} />
            Text-to-speech
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={!useAdminTts}
                onChange={(e) =>
                  setSettingsMutation.mutate({ use_admin_tts: !e.target.checked })
                }
                disabled={setSettingsMutation.isLoading}
              />
            }
            label="Use my own API keys"
          />

          {useAdminTts && (
            <Box sx={{ mt: 2 }}>
              <Alert severity="info" sx={{ mb: 2 }}>
                Built-in TTS: choose server default (deployer-configured), local Piper, or browser speech
                synthesis only—no cloud API keys required. These preferences are saved to your account. Turn on
                &quot;Use my own API keys&quot; below to add cloud TTS alongside Piper and browser.
              </Alert>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                Speech backend
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
                {ADMIN_LOCAL_TTS_CHIPS.map(({ id, label }) => {
                  const selected = adminLocalChipId === id;
                  return (
                    <Chip
                      key={id}
                      label={label}
                      onClick={() => {
                        const prov = id === 'server' ? '' : id;
                        setSettingsMutation.mutate({ user_admin_tts_provider: prov });
                      }}
                      color={selected ? 'primary' : 'default'}
                      variant={selected ? 'filled' : 'outlined'}
                      disabled={setSettingsMutation.isLoading}
                      aria-pressed={selected}
                      sx={
                        selected
                          ? { fontWeight: 600, border: '1px solid', borderColor: 'primary.main' }
                          : undefined
                      }
                    />
                  );
                })}
              </Box>
              {adminLocalChipId !== 'browser' && (
                <>
                  <FormControl fullWidth size="small" sx={{ mb: 1 }}>
                    <InputLabel>Preferred voice</InputLabel>
                    <Select
                      label="Preferred voice"
                      value={adminVoiceSelectValue}
                      onChange={(e) => {
                        const vid = e.target.value;
                        const sel = adminVoices.find((v) => v.voice_id === vid);
                        const payload =
                          adminLocalChipId === 'piper'
                            ? { user_admin_tts_voice_piper: vid }
                            : { user_admin_tts_voice_server: vid };
                        if (sel?.provider && adminLocalChipId !== 'piper') {
                          payload.user_admin_tts_provider = sel.provider;
                        }
                        setSettingsMutation.mutate(payload);
                      }}
                      disabled={adminVoicesLoading || setSettingsMutation.isLoading}
                    >
                      <MenuItem value="">
                        <em>Default (provider default)</em>
                      </MenuItem>
                      {adminVoices.map((v) => (
                        <MenuItem key={`${v.provider || 'x'}-${v.voice_id}`} value={v.voice_id}>
                          {(v.name || v.voice_id) + (v.provider ? ` (${v.provider})` : '')}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  {adminVoicesLoading && <CircularProgress size={20} sx={{ ml: 1 }} />}
                  {adminLocalChipId === 'piper' &&
                    !adminVoicesLoading &&
                    adminVoices.length === 0 && (
                      <Alert severity="warning" sx={{ mt: 1 }}>
                        No Piper voices were returned. Ensure Piper .onnx models are installed under the
                        voice-service model path (see deployer docs / PIPER_MODEL_PATH).
                      </Alert>
                    )}
                  {adminUsesElevenLabs && (
                    <FormControl fullWidth size="small" sx={{ mb: 2, mt: 1 }}>
                      <InputLabel>ElevenLabs synthesis model</InputLabel>
                      <Select
                        label="ElevenLabs synthesis model"
                        value={
                          ELEVENLABS_TTS_MODEL_OPTIONS.some(
                            (o) => o.value === userAdminElevenlabsModelId
                          )
                            ? userAdminElevenlabsModelId
                            : ''
                        }
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_admin_elevenlabs_tts_model_id: e.target.value,
                          })
                        }
                        disabled={setSettingsMutation.isLoading}
                      >
                        {ELEVENLABS_TTS_MODEL_OPTIONS.map((o) => (
                          <MenuItem key={o.value || 'default'} value={o.value}>
                            {o.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                  {adminUsesHedra && (
                    <FormControl fullWidth size="small" sx={{ mb: 2, mt: 1 }}>
                      <InputLabel>Hedra TTS engine</InputLabel>
                      <Select
                        label="Hedra TTS engine"
                        value={
                          hedraModels.some((m) => m.id === userAdminHedraModelId)
                            ? userAdminHedraModelId
                            : ''
                        }
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_admin_hedra_tts_model_id: e.target.value,
                          })
                        }
                        disabled={setSettingsMutation.isLoading || hedraModelsLoading}
                      >
                        <MenuItem value="">
                          <em>Platform default</em>
                        </MenuItem>
                        {hedraModels.map((m) => (
                          <MenuItem key={m.id} value={m.id}>
                            {m.name || m.id}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}
                </>
              )}
              {adminLocalChipId === 'browser' && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Read-aloud uses your browser&apos;s built-in voices only; the server is not used for TTS.
                </Typography>
              )}
            </Box>
          )}

          {!useAdminTts && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                Speech backend
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                Piper and Browser work without API keys. Cloud options appear after your saved providers load.
              </Typography>
              <Box
                display="flex"
                flexWrap="wrap"
                gap={1}
                alignItems="center"
                sx={{ mb: 2 }}
              >
                <Chip
                  label="Piper (local)"
                  onClick={() =>
                    setSettingsMutation.mutate({
                      user_byok_tts_engine: 'piper',
                    })
                  }
                  color={byokTtsEngine === 'piper' ? 'primary' : 'default'}
                  variant={byokTtsEngine === 'piper' ? 'filled' : 'outlined'}
                  disabled={setSettingsMutation.isLoading}
                  aria-pressed={byokTtsEngine === 'piper'}
                  sx={
                    byokTtsEngine === 'piper'
                      ? { fontWeight: 600, border: '1px solid', borderColor: 'primary.main' }
                      : undefined
                  }
                />
                <Chip
                  label="Browser TTS"
                  onClick={() =>
                    setSettingsMutation.mutate({
                      user_byok_tts_engine: 'browser',
                    })
                  }
                  color={byokTtsEngine === 'browser' ? 'primary' : 'default'}
                  variant={byokTtsEngine === 'browser' ? 'filled' : 'outlined'}
                  disabled={setSettingsMutation.isLoading}
                  aria-pressed={byokTtsEngine === 'browser'}
                  sx={
                    byokTtsEngine === 'browser'
                      ? { fontWeight: 600, border: '1px solid', borderColor: 'primary.main' }
                      : undefined
                  }
                />
                {providersLoading && <CircularProgress size={20} />}
                {!providersLoading &&
                  ttsProviders.map((p) => {
                    const pid = String(p.id);
                    const selected =
                      byokTtsEngine === 'cloud' && String(ttsProviderId) === pid;
                    return (
                      <Chip
                        key={p.id}
                        label={`${p.display_name || p.provider_type} (${p.provider_type})`}
                        onClick={() => {
                          const body = {
                            user_byok_tts_engine: 'cloud',
                            user_tts_provider_id: pid,
                          };
                          if (String(ttsProviderId) !== pid) {
                            body.user_tts_voice_id = '';
                          }
                          setSettingsMutation.mutate(body);
                        }}
                        color={selected ? 'primary' : 'default'}
                        variant={selected ? 'filled' : 'outlined'}
                        disabled={setSettingsMutation.isLoading}
                        aria-pressed={selected}
                        sx={
                          selected
                            ? { fontWeight: 600, border: '1px solid', borderColor: 'primary.main' }
                            : undefined
                        }
                      />
                    );
                  })}
              </Box>

              {byokTtsEngine === 'piper' && (
                    <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                      <InputLabel>Preferred Piper voice</InputLabel>
                      <Select
                        label="Preferred Piper voice"
                        value={byokPiperSelectValue}
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_byok_tts_voice_piper: e.target.value,
                          })
                        }
                        disabled={byokPiperLoading}
                      >
                        <MenuItem value="">
                          <em>Default</em>
                        </MenuItem>
                        {byokPiperVoices.map((v) => (
                          <MenuItem key={v.voice_id} value={v.voice_id}>
                            {v.name || v.voice_id}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}

                  {byokTtsEngine === 'piper' && !byokPiperLoading && byokPiperVoices.length === 0 && (
                    <Alert severity="warning" sx={{ mb: 2 }}>
                      No Piper voices were returned. Ensure Piper .onnx models are on the server, or ask
                      your admin to set PIPER_MODEL_PATH.
                    </Alert>
                  )}

                  {byokTtsEngine === 'cloud' && Boolean(ttsProviderId) && (
                    <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                      <InputLabel>Preferred voice</InputLabel>
                      <Select
                        label="Preferred voice"
                        value={ttsVoiceId}
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_tts_voice_id: e.target.value,
                          })
                        }
                        disabled={voicesLoading}
                      >
                        <MenuItem value="">
                          <em>Default</em>
                        </MenuItem>
                        {voices.map((v) => (
                          <MenuItem key={v.voice_id} value={v.voice_id}>
                            {v.name || v.voice_id}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}

                  {byokUsesElevenLabs && Boolean(ttsProviderId) && (
                    <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                      <InputLabel>ElevenLabs synthesis model</InputLabel>
                      <Select
                        label="ElevenLabs synthesis model"
                        value={
                          ELEVENLABS_TTS_MODEL_OPTIONS.some(
                            (o) => o.value === userElevenlabsModelId
                          )
                            ? userElevenlabsModelId
                            : ''
                        }
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_elevenlabs_tts_model_id: e.target.value,
                          })
                        }
                        disabled={setSettingsMutation.isLoading}
                      >
                        {ELEVENLABS_TTS_MODEL_OPTIONS.map((o) => (
                          <MenuItem key={o.value || 'byok-default'} value={o.value}>
                            {o.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}

                  {byokUsesHedra && Boolean(ttsProviderId) && (
                    <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                      <InputLabel>Hedra TTS engine</InputLabel>
                      <Select
                        label="Hedra TTS engine"
                        value={
                          hedraModels.some((m) => m.id === userHedraModelId)
                            ? userHedraModelId
                            : ''
                        }
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_hedra_tts_model_id: e.target.value,
                          })
                        }
                        disabled={setSettingsMutation.isLoading || hedraModelsLoading}
                      >
                        <MenuItem value="">
                          <em>Platform default</em>
                        </MenuItem>
                        {hedraModels.map((m) => (
                          <MenuItem key={m.id} value={m.id}>
                            {m.name || m.id}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}

                  {byokTtsEngine === 'browser' && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      Read-aloud uses your browser&apos;s built-in voices only; the server is not used for TTS.
                    </Typography>
                  )}

              {byokTtsEngine === 'cloud' &&
                    !providersLoading &&
                    ttsProviders.length === 0 && (
                      <Alert severity="info" sx={{ mb: 2 }}>
                        Add a cloud TTS provider below, or choose Piper or Browser for speech.
                      </Alert>
                    )}
              {byokTtsEngine === 'cloud' &&
                    !providersLoading &&
                    ttsProviders.length > 0 &&
                    !ttsProviderId && (
                      <Alert severity="warning" sx={{ mb: 2 }}>
                        Select a cloud provider chip, or switch to Piper / Browser.
                      </Alert>
                    )}

              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1, mt: 2 }}>
                Add cloud TTS provider (validated before save)
              </Typography>
              {providersLoading ? (
                <CircularProgress size={22} sx={{ mb: 2 }} />
              ) : (
                <>
                  <Box display="flex" flexWrap="wrap" gap={1} sx={{ mb: 1 }}>
                    {TTS_TYPES.map(({ id, label }) => (
                      <Chip
                        key={id}
                        label={label}
                        onClick={() =>
                          setAdding(
                            adding?.role === 'tts' && adding?.type === id
                              ? null
                              : { role: 'tts', type: id }
                          )
                        }
                        color={
                          adding?.role === 'tts' && adding?.type === id ? 'primary' : 'default'
                        }
                        variant={
                          adding?.role === 'tts' && adding?.type === id ? 'filled' : 'outlined'
                        }
                      />
                    ))}
                  </Box>
                  {adding?.role === 'tts' && (
                    <Box sx={{ p: 2, bgcolor: 'action.hover', borderRadius: 1, mb: 2 }}>
                      <TextField
                        fullWidth
                        size="small"
                        label="API key"
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        sx={{ mb: 1 }}
                      />
                      <TextField
                        fullWidth
                        size="small"
                        label="Base URL (optional, OpenAI-compatible)"
                        value={baseUrl}
                        onChange={(e) => setBaseUrl(e.target.value)}
                        sx={{ mb: 1 }}
                      />
                      <Button
                        size="small"
                        variant="contained"
                        startIcon={addMutation.isLoading ? <CircularProgress size={16} /> : <Add />}
                        onClick={handleAdd}
                        disabled={addMutation.isLoading}
                      >
                        Add TTS provider
                      </Button>
                    </Box>
                  )}
                </>
              )}

              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                Saved cloud TTS providers
              </Typography>
              <List dense>
                {providersLoading ? (
                  <ListItem>
                    <ListItemText primary="Loading…" />
                  </ListItem>
                ) : (
                  ttsProviders.map((p) => (
                    <ListItem key={p.id}>
                      <ListItemText
                        primary={p.display_name || p.provider_type}
                        secondary={`${p.provider_type} · id ${p.id}`}
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          aria-label="delete"
                          onClick={() => removeMutation.mutate(p.id)}
                          size="small"
                        >
                          <Delete />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))
                )}
              </List>
            </Box>
          )}
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
            <Mic sx={{ mr: 0.5, fontSize: 18 }} />
            Speech-to-text
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={!useAdminStt}
                onChange={(e) =>
                  setSettingsMutation.mutate({ use_admin_stt: !e.target.checked })
                }
                disabled={setSettingsMutation.isLoading}
              />
            }
            label="Use my own API keys"
          />

          {!useAdminStt && (
            <Box sx={{ mt: 2 }}>
              {providersLoading ? (
                <CircularProgress size={22} />
              ) : (
                <>
                  <Box display="flex" flexWrap="wrap" gap={1} sx={{ mb: 1 }}>
                    {STT_TYPES.map(({ id, label }) => (
                      <Chip
                        key={id}
                        label={label}
                        onClick={() =>
                          setAdding(adding?.role === 'stt' && adding?.type === id ? null : { role: 'stt', type: id })
                        }
                        color={adding?.role === 'stt' && adding?.type === id ? 'primary' : 'default'}
                        variant={adding?.role === 'stt' && adding?.type === id ? 'filled' : 'outlined'}
                      />
                    ))}
                  </Box>
                  {adding?.role === 'stt' && (
                    <Box sx={{ p: 2, bgcolor: 'action.hover', borderRadius: 1, mb: 2 }}>
                      <TextField
                        fullWidth
                        size="small"
                        label="API key"
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        sx={{ mb: 1 }}
                      />
                      <TextField
                        fullWidth
                        size="small"
                        label="Base URL (optional, OpenAI-compatible)"
                        value={baseUrl}
                        onChange={(e) => setBaseUrl(e.target.value)}
                        sx={{ mb: 1 }}
                      />
                      <Button
                        size="small"
                        variant="contained"
                        startIcon={addMutation.isLoading ? <CircularProgress size={16} /> : <Add />}
                        onClick={handleAdd}
                        disabled={addMutation.isLoading}
                      >
                        Add STT provider
                      </Button>
                    </Box>
                  )}

                  {sttProviders.length > 0 && (
                    <FormControl fullWidth size="small" sx={{ mb: 2 }}>
                      <InputLabel>Active STT provider</InputLabel>
                      <Select
                        label="Active STT provider"
                        value={sttProviderId}
                        onChange={(e) =>
                          setSettingsMutation.mutate({
                            user_stt_provider_id: e.target.value,
                          })
                        }
                      >
                        <MenuItem value="">
                          <em>Select…</em>
                        </MenuItem>
                        {sttProviders.map((p) => (
                          <MenuItem key={p.id} value={String(p.id)}>
                            {p.display_name || p.provider_type} ({p.provider_type})
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  )}

                  <List dense>
                    {sttProviders.map((p) => (
                      <ListItem key={p.id}>
                        <ListItemText
                          primary={p.display_name || p.provider_type}
                          secondary={`${p.provider_type} · id ${p.id}`}
                        />
                        <ListItemSecondaryAction>
                          <IconButton
                            edge="end"
                            aria-label="delete"
                            onClick={() => removeMutation.mutate(p.id)}
                            size="small"
                          >
                            <Delete />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}
