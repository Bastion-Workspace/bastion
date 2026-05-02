import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  LinearProgress,
  Chip,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  Divider,
  Tooltip,
  IconButton,
  Paper,
  Radio,
  Badge,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  CircularProgress,
  useTheme as useMuiTheme,
  ToggleButtonGroup,
  ToggleButton,
} from '@mui/material';
import { 
  Settings, 
  Psychology, 
  Speed, 
  ExpandMore, 
  Search, 
  Refresh,
  Security,
  DeleteSweep,
  Warning,
  Book,
  Person,
  Add,
  Edit as EditIcon,
  ListAlt,
  FolderOpen,
  MusicNote,
  RssFeed as RssFeedIcon,
  MenuBook,
  Info,
  Visibility,
  VisibilityOff,
  Email,
  Link as LinkIcon,
  Lock,
  Palette,
  BrightnessAuto,
  Hub,
  AccountTree,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useNavigate, useSearchParams } from 'react-router-dom';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { ACCENT_IDS, DEFAULT_ACCENT_ID } from '../contexts/ThemeContext';
import { ACCENT_PALETTES } from '../theme/themeConfig';
import UserManagement from './UserManagement';
import ClassificationModelSelector from './ClassificationModelSelector';
import ImageGenerationModelSelector from './ImageGenerationModelSelector';
import TextCompletionModelSelector from './TextCompletionModelSelector';
import OrgModeSettingsTab from './OrgModeSettingsTab';
import ZettelkastenSettingsTab from './ZettelkastenSettingsTab';
import MediaSettingsTab from './music/MediaSettingsTab';
import ExternalConnectionsSettings from './ExternalConnectionsSettings';
import FederationSettings from './FederationSettings';
import UserLLMProviders from './UserLLMProviders';
import UserVoiceProviders from './UserVoiceProviders';
import BrowserSessionManagement from './agent-factory/BrowserSessionManagement';
import RSSFeedSettings from './RSSFeedSettings';
import SettingsEbooksOpdsSection from './settings/SettingsEbooksOpdsSection';
import BbsWallpaperSettingsTab from './settings/BbsWallpaperSettingsTab';
import UiWallpaperSettingsSection from './settings/UiWallpaperSettingsSection';

// Model Status Display Component
const ModelStatusDisplay = () => {
  const muiTheme = useMuiTheme();
  const isDark = muiTheme.palette.mode === 'dark';
  const successBg = isDark ? 'success.dark' : 'success.light';
  const warningBg = isDark ? 'warning.dark' : 'warning.light';

  const { data: classificationData, isLoading: loadingClassification } = useQuery(
    'classificationModel',
    () => apiService.get('/api/models/classification')
  );

  if (loadingClassification) {
    return (
      <Box display="flex" alignItems="center" gap={2} p={2}>
        <CircularProgress size={20} />
        <Typography variant="body2">Loading model status...</Typography>
      </Box>
    );
  }

  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, bgcolor: classificationData?.chat_model_is_fallback ? warningBg : successBg }}>
          <Typography variant="subtitle2" gutterBottom>
            Main Chat Model
            {classificationData?.chat_model_is_fallback && (
              <Chip label="Fallback" size="small" color="warning" sx={{ ml: 1 }} />
            )}
          </Typography>
          <Typography variant="body2">
            {classificationData?.effective_chat_model || 'Not configured'}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Global default main model; chat usually uses sidebar selection and per-user preferences instead
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, bgcolor: classificationData?.classification_model_is_fallback ? warningBg : successBg }}>
          <Typography variant="subtitle2" gutterBottom>
            Classification Model
            {classificationData?.classification_model_is_fallback && (
              <Chip label="Fallback" size="small" color="warning" sx={{ ml: 1 }} />
            )}
          </Typography>
          <Typography variant="body2">
            {classificationData?.effective_classification_model || 'Not configured'}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Org default for the fast model slot (metadata user_fast_model) when the user has no fast preference
          </Typography>
        </Paper>
      </Grid>

      {(classificationData?.chat_model_is_fallback || classificationData?.classification_model_is_fallback) && (
        <Grid item xs={12}>
          <Alert severity="warning">
            <Typography variant="body2">
              <strong>Models Using Fallbacks:</strong> Some models are using system defaults instead of explicitly configured models.
              Configure specific models above to ensure consistent behavior.
            </Typography>
          </Alert>
        </Grid>
      )}
    </Grid>
  );
};


const SettingsPage = () => {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const { user, updateUser, loading: authLoading } = useAuth();
  const {
    darkMode,
    accentId,
    setAccentId,
    setAppearance,
    themePreference,
    setThemePreference,
    systemPrefersDark,
  } = useTheme();
  const [currentTab, setCurrentTab] = useState(0);
  const [enabledModels, setEnabledModels] = useState(new Set());
  const [selectedModel, setSelectedModel] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [providerFilter, setProviderFilter] = useState('');
  const [providerSelectFocused, setProviderSelectFocused] = useState(false);
  const [showOnlyEnabled, setShowOnlyEnabled] = useState(false);
  
  // Database cleanup dialog states
  const [qdrantDialogOpen, setQdrantDialogOpen] = useState(false);
  const [neo4jDialogOpen, setNeo4jDialogOpen] = useState(false);
  const [documentsDialogOpen, setDocumentsDialogOpen] = useState(false);
  const [documentsDatabaseOnlyDialogOpen, setDocumentsDatabaseOnlyDialogOpen] = useState(false);
  const [faceIdentitiesDialogOpen, setFaceIdentitiesDialogOpen] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // User profile state
  const [userTimezone, setUserTimezone] = useState('UTC');
  const [timezoneLoading, setTimezoneLoading] = useState(false);
  const [userZipCode, setUserZipCode] = useState('');
  const [userTimeFormat, setUserTimeFormat] = useState('24h');
  const [userPreferredName, setUserPreferredName] = useState('');
  const [userAiContext, setUserAiContext] = useState('');
  const [userFacts, setUserFacts] = useState([]);
  const [newFactKey, setNewFactKey] = useState('');
  const [newFactValue, setNewFactValue] = useState('');
  const [factsInjectEnabled, setFactsInjectEnabled] = useState(true);
  const [factsWriteEnabled, setFactsWriteEnabled] = useState(true);
  const [userEpisodes, setUserEpisodes] = useState([]);
  const [episodesInjectEnabled, setEpisodesInjectEnabled] = useState(true);
  const [pendingFacts, setPendingFacts] = useState([]);
  const [factHistory, setFactHistory] = useState([]);
  const [visionFeaturesEnabled, setVisionFeaturesEnabled] = useState(false);
  const [visionServiceAvailable, setVisionServiceAvailable] = useState(false);
  const [profileEmail, setProfileEmail] = useState('');
  const [profilePhoneNumber, setProfilePhoneNumber] = useState('');
  const [profileBirthday, setProfileBirthday] = useState('');
  const [profileDisplayName, setProfileDisplayName] = useState('');
  
  // Password change state
  const [passwordChange, setPasswordChange] = useState({
    current_password: '',
    new_password: '',
    confirm_password: ''
  });
  const [showPasswords, setShowPasswords] = useState({
    current: false,
    new: false,
    confirm: false
  });


  // AI Personality state
  const [promptSettings, setPromptSettings] = useState({
    ai_name: 'Alex',
    political_bias: 'neutral',
    persona_style: 'professional'
  });

  const [promptOptions, setPromptOptions] = useState({
    political_biases: [],
    persona_styles: [],
    historical_figures: []
  });

  // Stock persona state (legacy - kept for promptOptions.political_biases in persona form)
  const [stockPersonaMode, setStockPersonaMode] = useState('custom');
  const [customSettings, setCustomSettings] = useState({
    ai_name: 'Alex',
    political_bias: 'neutral',
    persona_style: 'professional'
  });

  // Persona manager state
  const [personaDialogOpen, setPersonaDialogOpen] = useState(false);
  const [personaDialogMode, setPersonaDialogMode] = useState('create');
  const [personaDialogEditingId, setPersonaDialogEditingId] = useState(null);
  const [personaForm, setPersonaForm] = useState({
    name: '',
    ai_name: 'Alex',
    style_instruction: '',
    political_bias: 'neutral',
    description: ''
  });

  // Sync profile fields from auth user (email and display_name editable in profile)
  React.useEffect(() => {
    if (user) {
      setProfileEmail(user.email || '');
      setProfileDisplayName(user.display_name || '');
    }
  }, [user?.user_id, user?.email, user?.display_name]);

  // Handle OAuth callback redirect: connections=success | connections=error&message=...
  React.useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const status = urlParams.get('connections');
    if (status === 'success') {
      setSnackbar({ open: true, message: 'Email connected successfully.', severity: 'success' });
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          p.delete('connections');
          p.set('tab', 'connections');
          return p;
        },
        { replace: true }
      );
    } else if (status === 'error') {
      const msg = urlParams.get('message') || 'Connection failed.';
      setSnackbar({ open: true, message: msg, severity: 'error' });
      setSearchParams(
        (prev) => {
          const p = new URLSearchParams(prev);
          p.delete('connections');
          p.delete('message');
          p.set('tab', 'connections');
          return p;
        },
        { replace: true }
      );
    }
  }, [setSearchParams]);

  const { data: useAdminModelsData } = useQuery(
    'useAdminModels',
    () => apiService.getUseAdminModels(),
    { staleTime: 60000 }
  );
  const useOwnProviders = useAdminModelsData?.use_admin_models === false;

  const { data: userModelRolesData, refetch: refetchUserModelRoles } = useQuery(
    'userModelRoles',
    () => apiService.getUserModelRoles(),
    { staleTime: 60000 }
  );
  const userModelRoles = userModelRolesData || {
    user_chat_model: '',
    user_fast_model: '',
    user_image_gen_model: '',
    user_image_analysis_model: '',
    send_while_streaming_behavior: 'queue',
  };

  const setUserModelRolesMutation = useMutation(
    (roles) => apiService.setUserModelRoles(roles),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('userModelRoles');
      },
    }
  );

  const handleUserModelRoleChange = (key, value) => {
    const next =
      key === 'send_while_streaming_behavior'
        ? { ...userModelRoles, [key]: value }
        : { ...userModelRoles, [key]: value || '' };
    setUserModelRolesMutation.mutate(next);
  };

  // Fetch enabled models from backend
  const { data: enabledModelsData, isLoading: enabledModelsLoading } = useQuery(
    ['enabledModels', user?.user_id],
    () => apiService.getEnabledModels(),
    {
      enabled: !!(user?.user_id && !authLoading),
      onSuccess: (data) => {
        if (data?.enabled_models) {
          setEnabledModels(new Set(data.enabled_models));
        }
      }
    }
  );

  // Fetch available models - only after enabled models are loaded
  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useQuery(
    ['availableModels', user?.user_id],
    () => apiService.getAvailableModels(),
    {
      enabled: !!(user?.user_id && !authLoading && !enabledModelsLoading), // Wait for enabled models to load first
      onSuccess: (data) => {
        // Initialize with some popular models enabled by default if no enabled models exist
        if (data?.models && enabledModelsData && (!enabledModelsData?.enabled_models || enabledModelsData.enabled_models.length === 0)) {
          const defaultEnabled = new Set();
          const popularModels = [
            'anthropic/claude-3-sonnet',
            'anthropic/claude-3-haiku',
            'openai/gpt-4-turbo-preview',
            'openai/gpt-3.5-turbo',
            'meta-llama/llama-3-70b-instruct'
          ];
          
          data.models.forEach(model => {
            if (popularModels.includes(model.id)) {
              defaultEnabled.add(model.id);
            }
          });
          
          setEnabledModels(defaultEnabled);
          
          // Set first enabled model as selected
          if (defaultEnabled.size > 0 && !selectedModel) {
            setSelectedModel(Array.from(defaultEnabled)[0]);
          }
        }
      }
    }
  );

  const orphanedEnabledModels = useMemo(
    () => enabledModelsData?.orphaned_enabled_models || [],
    [enabledModelsData?.orphaned_enabled_models]
  );

  const orphanedRoleModelEntries = useMemo(() => {
    const raw = enabledModelsData?.orphaned_role_models;
    if (!raw || typeof raw !== 'object') return [];
    return Object.entries(raw).filter(([, v]) => v);
  }, [enabledModelsData?.orphaned_role_models]);

  const enabledModelIds = useMemo(
    () => new Set(enabledModelsData?.enabled_models || []),
    [enabledModelsData?.enabled_models]
  );
  const enabledModelsList = useMemo(
    () => (modelsData?.models || []).filter((m) => enabledModelIds.has(m.id)),
    [modelsData?.models, enabledModelIds]
  );
  const imageGenerationModels = useMemo(
    () =>
      enabledModelsList.filter((m) => {
        if (Array.isArray(m.output_modalities) && m.output_modalities.includes('image')) return true;
        const id = (m.id || '').toLowerCase();
        const name = (m.name || '').toLowerCase();
        return id.includes('image') || id.includes('vision') || name.includes('image') || name.includes('vision') || id.includes('gemini') || name.includes('gemini');
      }),
    [enabledModelsList]
  );
  const imageAnalysisModels = useMemo(
    () =>
      enabledModelsList.filter((m) => {
        if (Array.isArray(m.input_modalities) && m.input_modalities.includes('image')) return true;
        const id = (m.id || '').toLowerCase();
        const name = (m.name || '').toLowerCase();
        return id.includes('vision') || name.includes('vision') || id.includes('gemini') || name.includes('gemini') || id.includes('gpt-4o') || id.includes('claude-3');
      }),
    [enabledModelsList]
  );

  // Fetch user timezone
  const { data: timezoneData, refetch: refetchTimezone } = useQuery(
    'userTimezone',
    () => apiService.getUserTimezone(),
    {
      onSuccess: (data) => {
        if (data?.timezone) {
          setUserTimezone(data.timezone);
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user timezone:', error);
      }
    }
  );

  // Fetch user zip code
  const { data: zipCodeData, refetch: refetchZipCode } = useQuery(
    'userZipCode',
    () => apiService.settings.getUserZipCode(),
    {
      onSuccess: (data) => {
        if (data?.zip_code) {
          setUserZipCode(data.zip_code);
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user zip code:', error);
      }
    }
  );

  // Fetch user time format
  const { data: timeFormatData, refetch: refetchTimeFormat } = useQuery(
    'userTimeFormat',
    () => apiService.settings.getUserTimeFormat(),
    {
      onSuccess: (data) => {
        if (data?.time_format) {
          setUserTimeFormat(data.time_format);
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user time format:', error);
      }
    }
  );

  // Fetch user preferred name
  const { data: preferredNameData, refetch: refetchPreferredName } = useQuery(
    'userPreferredName',
    () => apiService.settings.getUserPreferredName(),
    {
      onSuccess: (data) => {
        if (data?.preferred_name !== undefined) {
          setUserPreferredName(data.preferred_name || '');
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user preferred name:', error);
      }
    }
  );

  // Fetch user phone number
  const { refetch: refetchPhoneNumber } = useQuery(
    'userPhoneNumber',
    () => apiService.settings.getUserPhoneNumber(),
    {
      onSuccess: (data) => {
        if (data?.phone_number !== undefined) {
          setProfilePhoneNumber(data.phone_number || '');
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user phone number:', error);
      }
    }
  );

  // Fetch user birthday
  const { refetch: refetchBirthday } = useQuery(
    'userBirthday',
    () => apiService.settings.getUserBirthday(),
    {
      onSuccess: (data) => {
        if (data?.birthday !== undefined) {
          setProfileBirthday(data.birthday || '');
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user birthday:', error);
      }
    }
  );

  // Fetch user AI context
  const { data: aiContextData, refetch: refetchAiContext } = useQuery(
    'userAiContext',
    () => apiService.settings.getUserAiContext(),
    {
      onSuccess: (data) => {
        if (data?.ai_context !== undefined) {
          setUserAiContext(data.ai_context || '');
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user AI context:', error);
      }
    }
  );

  const { data: userFactsData, refetch: refetchUserFacts } = useQuery(
    'userFacts',
    () => apiService.settings.getUserFacts(),
    {
      onSuccess: (data) => {
        if (data?.facts && Array.isArray(data.facts)) {
          setUserFacts(data.facts);
        }
      },
      onError: (error) => {
        console.error('Failed to fetch user facts:', error);
      }
    }
  );

  const { refetch: refetchFactsPreferences } = useQuery(
    'factsPreferences',
    () => apiService.settings.getFactsPreferences(),
    {
      onSuccess: (data) => {
        if (data?.facts_inject_enabled !== undefined) setFactsInjectEnabled(data.facts_inject_enabled);
        if (data?.facts_write_enabled !== undefined) setFactsWriteEnabled(data.facts_write_enabled);
      }
    }
  );

  const { data: episodesData, refetch: refetchEpisodes } = useQuery(
    'userEpisodes',
    () => apiService.settings.getEpisodes({ limit: 50, days: 30 }),
    {
      onSuccess: (data) => {
        if (data?.episodes && Array.isArray(data.episodes)) setUserEpisodes(data.episodes);
      }
    }
  );

  const { refetch: refetchEpisodesPreferences } = useQuery(
    'episodesPreferences',
    () => apiService.settings.getEpisodesPreferences(),
    {
      onSuccess: (data) => {
        if (data?.episodes_inject_enabled !== undefined) setEpisodesInjectEnabled(data.episodes_inject_enabled);
      }
    }
  );

  const { data: pendingFactsData, refetch: refetchPendingFacts } = useQuery(
    'pendingFacts',
    () => apiService.settings.getPendingFacts(),
    {
      onSuccess: (data) => {
        if (data?.pending && Array.isArray(data.pending)) setPendingFacts(data.pending);
      }
    }
  );

  const { data: factHistoryData, refetch: refetchFactHistory } = useQuery(
    'factHistory',
    () => apiService.settings.getFactHistory({ limit: 30 }),
    {
      onSuccess: (data) => {
        if (data?.history && Array.isArray(data.history)) setFactHistory(data.history);
      }
    }
  );

  // Fetch vision service status
  const { data: visionServiceStatusData } = useQuery(
    'visionServiceStatus',
    () => apiService.settings.getVisionServiceStatus(),
    {
      onSuccess: (data) => {
        setVisionServiceAvailable(data?.available || false);
      },
      onError: (error) => {
        console.error('Failed to fetch vision service status:', error);
        setVisionServiceAvailable(false);
      }
    }
  );

  const { data: visionFeaturesData, refetch: refetchVisionFeatures } = useQuery(
    'visionFeatures',
    () => apiService.settings.getVisionFeaturesEnabled(),
    {
      onSuccess: (data) => {
        if (data?.enabled !== undefined) {
          setVisionFeaturesEnabled(data.enabled);
        }
      },
      onError: (error) => {
        console.error('Failed to fetch vision features setting:', error);
      }
    }
  );

  // Fetch prompt settings options
  const { data: promptOptionsData } = useQuery(
    'promptOptions',
    () => apiService.getPromptOptions(),
    {
      onSuccess: (data) => {
        if (data) {
          setPromptOptions({
            political_biases: data.political_biases || [],
            persona_styles: data.persona_styles || [],
            historical_figures: data.historical_figures || []
          });
        }
      },
      onError: (error) => {
        console.error('Failed to fetch prompt options:', error);
      }
    }
  );

  const { data: personasData, refetch: refetchPersonas } = useQuery(
    'personas',
    () => apiService.settings.getPersonas(),
    { staleTime: 60000 }
  );
  const personasList = personasData?.personas || [];

  const { data: defaultPersonaData, refetch: refetchDefaultPersona } = useQuery(
    'defaultPersona',
    () => apiService.settings.getDefaultPersona(),
    { staleTime: 60000 }
  );
  const defaultPersona = defaultPersonaData?.persona;

  const setDefaultPersonaMutation = useMutation(
    (personaId) => apiService.settings.setDefaultPersona(personaId),
    {
      onSuccess: () => {
        refetchDefaultPersona();
        queryClient.invalidateQueries('defaultPersona');
        setSnackbar({ open: true, message: 'Default persona updated.', severity: 'success' });
      },
      onError: (err) => {
        setSnackbar({ open: true, message: err?.message || 'Failed to set default persona', severity: 'error' });
      }
    }
  );

  const createPersonaMutation = useMutation(
    (data) => apiService.settings.createPersona(data),
    {
      onSuccess: () => {
        refetchPersonas();
        refetchDefaultPersona();
        queryClient.invalidateQueries('personas');
        setPersonaDialogOpen(false);
        setPersonaForm({ name: '', ai_name: 'Alex', style_instruction: '', political_bias: 'neutral', description: '' });
        setSnackbar({ open: true, message: 'Persona created.', severity: 'success' });
      },
      onError: (err) => {
        setSnackbar({ open: true, message: err?.message || 'Failed to create persona', severity: 'error' });
      }
    }
  );

  const updatePersonaMutation = useMutation(
    ({ id, data }) => apiService.settings.updatePersona(id, data),
    {
      onSuccess: () => {
        refetchPersonas();
        refetchDefaultPersona();
        queryClient.invalidateQueries('personas');
        setPersonaDialogOpen(false);
        setPersonaDialogEditingId(null);
        setSnackbar({ open: true, message: 'Persona updated.', severity: 'success' });
      },
      onError: (err) => {
        setSnackbar({ open: true, message: err?.message || 'Failed to update persona', severity: 'error' });
      }
    }
  );

  const deletePersonaMutation = useMutation(
    (personaId) => apiService.settings.deletePersona(personaId),
    {
      onSuccess: () => {
        refetchPersonas();
        refetchDefaultPersona();
        queryClient.invalidateQueries('personas');
        setSnackbar({ open: true, message: 'Persona deleted.', severity: 'success' });
      },
      onError: (err) => {
        setSnackbar({ open: true, message: err?.message || 'Failed to delete persona', severity: 'error' });
      }
    }
  );

  const handleOpenCreatePersona = () => {
    setPersonaDialogMode('create');
    setPersonaDialogEditingId(null);
    setPersonaForm({ name: '', ai_name: 'Alex', style_instruction: '', political_bias: 'neutral', description: '' });
    setPersonaDialogOpen(true);
  };

  const handleOpenEditPersona = (p) => {
    setPersonaDialogMode('edit');
    setPersonaDialogEditingId(p.id);
    setPersonaForm({
      name: p.name || '',
      ai_name: p.ai_name || 'Alex',
      style_instruction: p.style_instruction || '',
      political_bias: p.political_bias || 'neutral',
      description: p.description || ''
    });
    setPersonaDialogOpen(true);
  };

  const handleSavePersonaDialog = () => {
    if (personaDialogMode === 'create') {
      createPersonaMutation.mutate(personaForm);
    } else {
      updatePersonaMutation.mutate({ id: personaDialogEditingId, data: personaForm });
    }
  };

  // Fetch user prompt settings
  const { data: promptSettingsData, refetch: refetchPromptSettings } = useQuery(
    'promptSettings',
    () => apiService.getPromptSettings(),
    {
      onSuccess: (data) => {
        if (data) {
          setPromptSettings({
            ai_name: data.ai_name || 'Alex',
            political_bias: data.political_bias || 'neutral',
            persona_style: data.persona_style || 'professional',
            bias_intensity: data.bias_intensity || 0.5,
            formality_level: data.formality_level || 0.7,
            technical_depth: data.technical_depth || 0.5
          });
          
          // Check if current settings match a stock persona
          const isStockPersona = promptOptions.historical_figures.some(
            figure => figure.value === data.persona_style
          );
          
          if (isStockPersona) {
            setStockPersonaMode(data.persona_style);
            setCustomSettings({
              ai_name: 'Alex',
              political_bias: 'neutral',
              persona_style: 'professional'
            });
          } else {
            setStockPersonaMode('custom');
            setCustomSettings({
              ai_name: data.ai_name || 'Alex',
              political_bias: data.political_bias || 'neutral',
              persona_style: data.persona_style || 'professional'
            });
          }
        }
      },
      onError: (error) => {
        console.error('Failed to fetch prompt settings:', error);
      }
    }
  );

  // Timezone update mutation
  const timezoneMutation = useMutation(
    (timezone) => apiService.setUserTimezone({ timezone }),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: `Timezone updated to ${data.timezone}`,
          severity: 'success'
        });
        refetchTimezone();
        queryClient.invalidateQueries('userTimezone');
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update timezone: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Zip code update mutation
  const zipCodeMutation = useMutation(
    (zipCode) => apiService.settings.setUserZipCode({ zip_code: zipCode }),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: `Zip code updated to ${data.zip_code}`,
          severity: 'success'
        });
        refetchZipCode();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update zip code: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Time format update mutation
  const timeFormatMutation = useMutation(
    (timeFormat) => apiService.settings.setUserTimeFormat({ time_format: timeFormat }),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: `Time format updated to ${data.time_format === '12h' ? '12-hour (AM/PM)' : '24-hour (Military)'}`,
          severity: 'success'
        });
        refetchTimeFormat();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update time format: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Profile (email, display name) update mutation – same data admins see in User Management
  const updateProfileMutation = useMutation(
    (data) => apiService.updateProfile({ email: data.email, display_name: data.display_name }),
    {
      onSuccess: (data) => {
        updateUser({ email: data.email, display_name: data.display_name });
        setSnackbar({
          open: true,
          message: 'Profile updated successfully',
          severity: 'success'
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: Array.isArray(error.response?.data?.detail) ? error.response.data.detail[0] : (error.response?.data?.detail || error.message || 'Failed to update profile'),
          severity: 'error'
        });
      }
    }
  );

  // Preferred name update mutation
  const preferredNameMutation = useMutation(
    (preferredName) => apiService.settings.setUserPreferredName({ preferred_name: preferredName }),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: 'Preferred name updated successfully',
          severity: 'success'
        });
        refetchPreferredName();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update preferred name: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Phone number update mutation
  const phoneNumberMutation = useMutation(
    (phoneNumber) => apiService.settings.setUserPhoneNumber({ phone_number: phoneNumber }),
    {
      onSuccess: () => {
        setSnackbar({
          open: true,
          message: 'Phone number updated successfully',
          severity: 'success'
        });
        refetchPhoneNumber();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update phone number: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Birthday update mutation
  const birthdayMutation = useMutation(
    (birthday) => apiService.settings.setUserBirthday({ birthday }),
    {
      onSuccess: () => {
        setSnackbar({
          open: true,
          message: 'Birthday updated successfully',
          severity: 'success'
        });
        refetchBirthday();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update birthday: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // AI context update mutation
  const aiContextMutation = useMutation(
    (aiContext) => apiService.settings.setUserAiContext({ ai_context: aiContext }),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: 'AI context updated successfully',
          severity: 'success'
        });
        refetchAiContext();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update AI context: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const addUserFactMutation = useMutation(
    (data) => apiService.settings.addUserFact(data),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Fact saved', severity: 'success' });
        setNewFactKey('');
        setNewFactValue('');
        refetchUserFacts();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to save fact: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const deleteUserFactMutation = useMutation(
    (factKey) => apiService.settings.deleteUserFact(factKey),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Fact removed', severity: 'success' });
        refetchUserFacts();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to remove fact: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const setFactsPreferencesMutation = useMutation(
    (data) => apiService.settings.setFactsPreferences(data),
    {
      onSuccess: (data) => {
        if (data?.facts_inject_enabled !== undefined) setFactsInjectEnabled(data.facts_inject_enabled);
        if (data?.facts_write_enabled !== undefined) setFactsWriteEnabled(data.facts_write_enabled);
        setSnackbar({ open: true, message: 'Facts preferences saved', severity: 'success' });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to save preferences: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const deleteEpisodeMutation = useMutation(
    (episodeId) => apiService.settings.deleteEpisode(episodeId),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Activity removed', severity: 'success' });
        refetchEpisodes();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to remove: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const deleteAllEpisodesMutation = useMutation(
    () => apiService.settings.deleteAllEpisodes(),
    {
      onSuccess: (data) => {
        const n = data?.deleted_count ?? 0;
        setSnackbar({ open: true, message: `Cleared ${n} activity entries`, severity: 'success' });
        refetchEpisodes();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to clear: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const setEpisodesPreferencesMutation = useMutation(
    (data) => apiService.settings.setEpisodesPreferences(data),
    {
      onSuccess: (data) => {
        if (data?.episodes_inject_enabled !== undefined) setEpisodesInjectEnabled(data.episodes_inject_enabled);
        setSnackbar({ open: true, message: 'Activity preferences saved', severity: 'success' });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to save: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const resolvePendingFactMutation = useMutation(
    ({ historyId, action }) => apiService.settings.resolvePendingFact(historyId, action),
    {
      onSuccess: () => {
        setSnackbar({ open: true, message: 'Update resolved', severity: 'success' });
        refetchPendingFacts();
        refetchUserFacts();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Vision features update mutation
  const visionFeaturesMutation = useMutation(
    (enabled) => apiService.settings.setVisionFeaturesEnabled(enabled),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: 'Vision features setting updated successfully',
          severity: 'success'
        });
        refetchVisionFeatures();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update vision features: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Password change mutation
  const passwordChangeMutation = useMutation(
    ({ userId, currentPassword, newPassword }) => apiService.changePassword(userId, currentPassword, newPassword),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: 'Password changed successfully',
          severity: 'success'
        });
        setPasswordChange({
          current_password: '',
          new_password: '',
          confirm_password: ''
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to change password: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Toggle password visibility
  const togglePasswordVisibility = (field) => {
    setShowPasswords(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  // Handle password change submit
  const handlePasswordChange = () => {
    if (!passwordChange.current_password || !passwordChange.new_password || !passwordChange.confirm_password) {
      setSnackbar({
        open: true,
        message: 'Please fill in all password fields',
        severity: 'error'
      });
      return;
    }

    if (passwordChange.new_password !== passwordChange.confirm_password) {
      setSnackbar({
        open: true,
        message: 'New password and confirm password do not match',
        severity: 'error'
      });
      return;
    }

    if (passwordChange.new_password.length < 8) {
      setSnackbar({
        open: true,
        message: 'Password must be at least 8 characters long',
        severity: 'error'
      });
      return;
    }

    if (!user?.user_id) {
      setSnackbar({
        open: true,
        message: 'User information not available',
        severity: 'error'
      });
      return;
    }

    passwordChangeMutation.mutate({
      userId: user.user_id,
      currentPassword: passwordChange.current_password,
      newPassword: passwordChange.new_password
    });
  };

  // Prompt settings update mutation
  const promptSettingsMutation = useMutation(
    (settings) => apiService.updatePromptSettings(settings),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: `AI personality settings updated successfully!`,
          severity: 'success'
        });
        refetchPromptSettings();
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to update AI personality settings: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Stock persona handlers
  const handleStockPersonaChange = (persona) => {
    setStockPersonaMode(persona);
    
    if (persona === 'custom') {
      // Switch back to custom mode with saved custom settings
      setPromptSettings(customSettings);
    } else {
      // Apply stock persona settings
      const stockSettings = {
        ai_name: getStockPersonaName(persona),
        political_bias: getStockPersonaBias(persona),
        persona_style: persona
      };
      setPromptSettings(stockSettings);
    }
  };

  const handleCustomSettingChange = (field, value) => {
    const newCustomSettings = { ...customSettings, [field]: value };
    setCustomSettings(newCustomSettings);
    
    if (stockPersonaMode === 'custom') {
      setPromptSettings(newCustomSettings);
    }
  };

  // Stock persona helper functions
  const getStockPersonaName = (persona) => {
    const nameMap = {
      'amelia_earhart': 'Amelia',
      'theodore_roosevelt': 'Teddy',
      'winston_churchill': 'Winston',
      'mr_spock': 'Spock',
      'abraham_lincoln': 'Abe',
      'napoleon_bonaparte': 'Napoleon',
      'isaac_newton': 'Isaac',
      'george_washington': 'George',
      'mark_twain': 'Mark',
      'edgar_allan_poe': 'Edgar',
      'jane_austen': 'Jane',
      'albert_einstein': 'Albert',
      'nikola_tesla': 'Tesla'
    };
    return nameMap[persona] || 'Alex';
  };

  const getStockPersonaBias = (persona) => {
    const biasMap = {
      'amelia_earhart': 'mildly_left',
      'theodore_roosevelt': 'mildly_right',
      'winston_churchill': 'mildly_right',
      'mr_spock': 'neutral',
      'abraham_lincoln': 'mildly_left',
      'napoleon_bonaparte': 'extreme_right',
      'isaac_newton': 'neutral',
      'george_washington': 'mildly_right',
      'mark_twain': 'mildly_left',
      'edgar_allan_poe': 'neutral',
      'jane_austen': 'mildly_right',
      'albert_einstein': 'mildly_left',
      'nikola_tesla': 'neutral'
    };
    return biasMap[persona] || 'neutral';
  };

  const getStockPersonaDescription = (persona) => {
    const descriptions = {
      'amelia_earhart': 'A pioneering aviator and adventurer, known for her record-breaking flights and fearless spirit.',
      'theodore_roosevelt': 'A charismatic and energetic leader, known for his conservation efforts and adventurous spirit.',
      'winston_churchill': 'A legendary British statesman and wartime leader, known for his resilience and powerful oratory.',
      'mr_spock': 'A logical and analytical Vulcan, known for his calm demeanor and rational approach to conflict.',
      'abraham_lincoln': 'A compassionate and wise leader, known for his leadership during the Civil War and his role in abolishing slavery.',
      'napoleon_bonaparte': 'A brilliant and ambitious military leader, known for his military genius and his downfall.',
      'isaac_newton': 'A brilliant mathematician and physicist, known for his laws of motion and universal gravitation.',
      'george_washington': 'A wise and experienced leader, known for his leadership during the Revolutionary War and his role as the first President of the United States.',
      'mark_twain': 'A witty and insightful author, known for his humor and his portrayal of American society.',
      'edgar_allan_poe': 'A mysterious and brilliant author, known for his short stories and his influence on the detective genre.',
      'jane_austen': 'A witty and insightful author, known for her novels and her portrayal of English society.',
      'albert_einstein': 'A brilliant physicist and author, known for his theory of relativity and his famous equation E=mc².',
      'nikola_tesla': 'A brilliant inventor and electrical engineer, known for his contributions to the field of electrical power and his work on alternating current.'
    };
    return descriptions[persona] || 'No specific description available for this persona.';
  };

  // Model selection mutation
  const selectModelMutation = useMutation(
    (modelName) => apiService.selectModel(modelName),
    {
      onSuccess: (data, variables) => {
        setSelectedModel(variables);
      },
    }
  );

  // Save enabled models mutation
  const saveEnabledModelsMutation = useMutation(
    (modelIds) => apiService.setEnabledModels(modelIds),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('enabledModels');
      },
    }
  );

  const handleRemoveOrphanedEnabledModels = (idsToRemove) => {
    const removeSet = new Set(idsToRemove);
    const next = new Set(enabledModels);
    removeSet.forEach((id) => next.delete(id));
    if (selectedModel && removeSet.has(selectedModel)) {
      const remaining = Array.from(next);
      const nextSel = remaining[0] || '';
      setSelectedModel(nextSel);
      if (nextSel) {
        selectModelMutation.mutate(nextSel);
      }
    }
    setEnabledModels(next);
    saveEnabledModelsMutation.mutate(Array.from(next));
  };

  // Database cleanup mutations
  const clearQdrantMutation = useMutation(
    () => apiService.clearQdrantDatabase(),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('systemStatus');
        setQdrantDialogOpen(false);
      },
    }
  );

  const clearNeo4jMutation = useMutation(
    () => apiService.clearNeo4jDatabase(),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('systemStatus');
        setNeo4jDialogOpen(false);
      },
    }
  );

  const clearFaceIdentitiesMutation = useMutation(
    () => apiService.delete('/api/vision/clear-all-identities'),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('systemStatus');
        setFaceIdentitiesDialogOpen(false);
      },
    }
  );

  const cleanupOrphanedVectorsMutation = useMutation(
    () => apiService.post('/api/vision/cleanup-orphaned-vectors'),
    {
      onSuccess: (data) => {
        setSnackbar({
          open: true,
          message: data.message || `Cleaned up ${data.vectors_cleaned} orphaned vectors`,
          severity: 'success'
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to cleanup vectors: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const clearDocumentsMutation = useMutation(
    () => apiService.clearAllDocuments(),
    {
      onSuccess: (data) => {
        console.log('✅ Clear documents success:', data);
        
                 // Force complete refresh of ONLY document-related queries (not settings)
         queryClient.invalidateQueries('documents');
         queryClient.invalidateQueries('documents-hierarchy');
         
         // Force immediate refetch of documents to ensure UI updates
         queryClient.refetchQueries('documents');
         queryClient.refetchQueries('documents-hierarchy');
         
         // Remove any cached document data (but preserve settings cache)
         queryClient.removeQueries('documents');
         queryClient.removeQueries('documents-hierarchy');
        
        setDocumentsDialogOpen(false);
        
                 // Show success message  
         setSnackbar({
           open: true,
           message: data.message || 'All documents deleted successfully!',
           severity: 'success'
         });
         
         // Only refresh system status, don't reload the entire page to preserve settings
         if (data.refresh_required) {
           setTimeout(() => {
             queryClient.invalidateQueries('systemStatus');
           }, 1000);
         }
      },
      onError: (error) => {
        console.error('❌ Clear documents failed:', error);
        setSnackbar({
          open: true,
          message: `Failed to delete documents: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      },
    }
  );

  const clearDocumentsDatabaseOnlyMutation = useMutation(
    () => apiService.clearDocumentsDatabaseOnly(true),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('documents');
        queryClient.invalidateQueries('documents-hierarchy');
        queryClient.refetchQueries('documents');
        queryClient.refetchQueries('documents-hierarchy');
        queryClient.removeQueries('documents');
        queryClient.removeQueries('documents-hierarchy');
        setDocumentsDatabaseOnlyDialogOpen(false);
        setSnackbar({
          open: true,
          message: data.message || 'Document database cleared; files on disk will be re-read.',
          severity: 'success'
        });
        if (data.refresh_required) {
          setTimeout(() => queryClient.invalidateQueries('systemStatus'), 1000);
        }
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to clear document database: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      },
    }
  );

  const rebuildAllLinksMutation = useMutation(
    () => apiService.rebuildAllLinks(),
    {
      onSuccess: (data) => {
        const { processed = 0, errors = 0, total = 0 } = data || {};
        setSnackbar({
          open: true,
          message: `Links rebuilt: ${processed} processed, ${errors} errors, ${total} total documents.`,
          severity: errors > 0 ? 'warning' : 'success'
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to rebuild links: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      },
    }
  );

  // Filter and group models
  const { filteredModels, groupedModels, displayGroupedModels, providers } = useMemo(() => {
    if (!modelsData?.models) return { filteredModels: [], groupedModels: {}, displayGroupedModels: {}, providers: [] };

    // Get unique providers - ALWAYS show all providers in dropdown
    const uniqueProviders = [...new Set(modelsData.models.map(m => m.provider))].sort();

    // When "Show Enabled" is ON, we want to show all providers with enabled models
    // When "Show Enabled" is OFF, we only show when a provider is selected
    if (showOnlyEnabled) {
      // Filter to only enabled models, then group by provider
      let enabledFiltered = modelsData.models.filter(model => enabledModels.has(model.id));
      
      // Apply search filter if present
      if (searchTerm) {
        const search = searchTerm.toLowerCase();
        enabledFiltered = enabledFiltered.filter(model => 
          model.name.toLowerCase().includes(search) ||
          model.provider.toLowerCase().includes(search) ||
          model.id.toLowerCase().includes(search)
        );
      }
      
      // Group enabled models by provider
      const enabledGrouped = enabledFiltered.reduce((acc, model) => {
        if (!acc[model.provider]) {
          acc[model.provider] = [];
        }
        acc[model.provider].push(model);
        return acc;
      }, {});
      
      return {
        filteredModels: enabledFiltered,
        groupedModels: enabledGrouped,
        displayGroupedModels: enabledGrouped, // Show all providers with enabled models
        providers: uniqueProviders
      };
    }

    // When "Show Enabled" is OFF, only show when provider is selected
    if (!providerFilter) {
      return { 
        filteredModels: [], 
        groupedModels: {}, 
        displayGroupedModels: {}, // Don't show any provider cards
        providers: uniqueProviders 
      };
    }

    let filtered = modelsData.models;

    // Apply search filter
    if (searchTerm) {
      const search = searchTerm.toLowerCase();
      filtered = filtered.filter(model => 
        model.name.toLowerCase().includes(search) ||
        model.provider.toLowerCase().includes(search) ||
        model.id.toLowerCase().includes(search)
      );
    }

    // Apply provider filter
    if (providerFilter !== 'all') {
      filtered = filtered.filter(model => model.provider === providerFilter);
    }

    // Group by provider
    const grouped = filtered.reduce((acc, model) => {
      if (!acc[model.provider]) {
        acc[model.provider] = [];
      }
      acc[model.provider].push(model);
      return acc;
    }, {});

    return { 
      filteredModels: filtered, 
      groupedModels: grouped, 
      displayGroupedModels: grouped, // Only show selected provider
      providers: uniqueProviders 
    };
  }, [modelsData, searchTerm, providerFilter, showOnlyEnabled, enabledModels]);

  const handleModelToggle = (modelId) => {
    const newEnabled = new Set(enabledModels);
    if (newEnabled.has(modelId)) {
      newEnabled.delete(modelId);
      // If we're disabling the selected model, select another enabled one
      if (selectedModel === modelId) {
        const remainingEnabled = Array.from(newEnabled);
        setSelectedModel(remainingEnabled.length > 0 ? remainingEnabled[0] : '');
      }
    } else {
      newEnabled.add(modelId);
      // If no model is selected, select this one
      if (!selectedModel) {
        setSelectedModel(modelId);
      }
    }
    setEnabledModels(newEnabled);
    
    // Save to backend
    saveEnabledModelsMutation.mutate(Array.from(newEnabled));
  };

  const handleActiveModelChange = (modelId) => {
    setSelectedModel(modelId);
    selectModelMutation.mutate(modelId);
  };

  const formatCost = (cost) => {
    if (!cost) return 'Free';
    if (cost < 0.001) return `$${(cost * 1000000).toFixed(2)}/1M tokens`;
    if (cost < 1) return `$${(cost * 1000).toFixed(2)}/1K tokens`;
    return `$${cost.toFixed(3)}/token`;
  };

  const getModelSourceTag = (model) => {
    if (!model?.source) return null;
    const sourceLabel = model.source === 'admin' ? 'Admin' : 'My providers';
    const provider = (model.provider_type || '').replace(/-/g, ' ');
    return provider ? `${sourceLabel} · ${provider}` : sourceLabel;
  };

  const ModelCard = ({ model }) => {
    const isEnabled = enabledModels.has(model.id);
    const isSelected = selectedModel === model.id;

    return (
      <Card
        sx={{
          mb: 1,
          border: isSelected ? '2px solid #1976d2' : '1px solid',
          borderColor: isSelected ? '#1976d2' : 'divider',
          backgroundColor: isEnabled ? 'background.secondary' : 'background.paper'
        }}
      >
        <CardContent sx={{ py: 2 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box flex={1}>
              <Box display="flex" alignItems="center" mb={1}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  {model.name}
                </Typography>
                <Chip 
                  label={model.provider} 
                  size="small" 
                  sx={{ ml: 1 }}
                  color="primary"
                  variant="outlined"
                />
                {getModelSourceTag(model) && (
                  <Chip 
                    label={getModelSourceTag(model)} 
                    size="small" 
                    sx={{ ml: 1 }}
                    variant="outlined"
                    color="default"
                  />
                )}
                {isSelected && (
                  <Chip 
                    label="Active" 
                    size="small" 
                    sx={{ ml: 1 }}
                    color="success"
                  />
                )}
              </Box>
              
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {model.description || model.id}
              </Typography>
              
              <Box display="flex" gap={2} flexWrap="wrap">
                <Typography variant="caption">
                  Context: {model.context_length?.toLocaleString() || 'Unknown'}
                </Typography>
                <Typography variant="caption">
                  Input: {formatCost(model.input_cost)}
                </Typography>
                <Typography variant="caption">
                  Output: {formatCost(model.output_cost)}
                </Typography>
              </Box>
            </Box>
            
            <Box display="flex" alignItems="center" gap={1}>
              <FormControlLabel
                control={
                  <Switch
                    checked={isEnabled}
                    onChange={() => handleModelToggle(model.id)}
                    size="small"
                  />
                }
                label="Enable"
                sx={{ mr: 1 }}
              />
              
              {isEnabled && (
                <Radio
                  checked={isSelected}
                  onChange={() => handleActiveModelChange(model.id)}
                  value={model.id}
                  size="small"
                  disabled={selectModelMutation.isLoading}
                />
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>
    );
  };

  const tabs = [
    { id: 'profile', label: 'User Profile', icon: <Person /> },
    { id: 'appearance', label: 'Appearance', icon: <Palette /> },
    { id: 'personas', label: 'Personas', icon: <Psychology /> },
    { id: 'models', label: 'Models', icon: <Settings /> },
    { id: 'rss-feeds', label: 'RSS Feeds', icon: <RssFeedIcon /> },
    { id: 'ebooks-opds', label: 'Ebooks (OPDS)', icon: <MenuBook /> },
    { id: 'org', label: 'Org-Mode', icon: <ListAlt /> },
    { id: 'zettelkasten', label: 'Zettelkasten', icon: <AccountTree /> },
    { id: 'media', label: 'Media', icon: <MusicNote /> },
    { id: 'connections', label: 'Connections', icon: <Email /> },
    { id: 'sessions', label: 'Browser Sessions', icon: <Lock /> },
    ...(user?.role === 'admin' ? [
      { id: 'database', label: 'Database', icon: <DeleteSweep /> },
      { id: 'users', label: 'User Management', icon: <Security /> },
      { id: 'federation', label: 'Federation', icon: <Hub /> },
    ] : [])
  ];

  // Sync currentTab from URL (e.g. /settings?tab=appearance)
  React.useEffect(() => {
    const tabId = searchParams.get('tab');
    if (tabId === 'control-panes') {
      navigate('/control-panes', { replace: true });
      return;
    }
    if (tabId === 'wallpaper' || tabId === 'news') {
      const next = tabId === 'wallpaper' ? 'appearance' : 'rss-feeds';
      setSearchParams(
        (prev) => {
          const params = new URLSearchParams(prev);
          params.set('tab', next);
          return params;
        },
        { replace: true }
      );
      return;
    }
    if (tabId) {
      const idx = tabs.findIndex(t => t.id === tabId);
      if (idx >= 0) setCurrentTab(idx);
    }
  }, [searchParams, tabs, navigate, setSearchParams]);

  const handleTabSelect = (newIndex) => {
    setCurrentTab(newIndex);
    const nextTabId = tabs[newIndex]?.id;
    if (!nextTabId) return;
    setSearchParams((prev) => {
      const params = new URLSearchParams(prev);
      params.set('tab', nextTabId);
      return params;
    });
  };

  const modelsTabIndex = useMemo(() => tabs.findIndex((t) => t.id === 'models'), [tabs]);

  const roleSettingSectionId = (roleKey) => {
    const map = {
      classification_model: 'settings-classification-model-card',
      image_generation_model: 'settings-image-gen-model-card',
      text_completion_model: 'settings-text-completion-model-card',
      image_analysis_model: 'settings-image-analysis-model-card',
    };
    return map[roleKey] || null;
  };

  const scrollToRoleSettingSection = (roleKey) => {
    if (modelsTabIndex >= 0) {
      handleTabSelect(modelsTabIndex);
    }
    const elId = roleSettingSectionId(roleKey);
    if (elId) {
      setTimeout(() => {
        document.getElementById(elId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 150);
    }
  };

  const clearRoleOrphanMutation = useMutation(
    async (roleKey) => {
      if (roleKey === 'classification_model') {
        await apiService.delete('/api/models/classification');
      } else if (roleKey === 'text_completion_model') {
        await apiService.settings.setSettingValue('text_completion_model', {
          value: '',
          description: 'Fast text-completion model for editor/proofreading tasks',
          category: 'llm',
        });
      } else if (roleKey === 'image_generation_model') {
        await apiService.settings.setSettingValue('image_generation_model', {
          value: '',
          description: 'OpenRouter model used for image generation',
          category: 'llm',
        });
      } else if (roleKey === 'image_analysis_model') {
        await apiService.settings.setSettingValue('image_analysis_model', {
          value: '',
          description: 'Vision model for image description and analysis',
          category: 'llm',
        });
      }
      await apiService.invalidateCatalogSlice();
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('enabledModels');
        queryClient.invalidateQueries('classificationModel');
        queryClient.invalidateQueries('textCompletionModelSetting');
        queryClient.invalidateQueries('imageGenerationModelSetting');
        queryClient.invalidateQueries('imageAnalysisModelSetting');
        setSnackbar({
          open: true,
          message: 'Setting cleared. Pick a new model in the section below.',
          severity: 'success',
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: error?.response?.data?.detail || error?.message || 'Failed to clear setting',
          severity: 'error',
        });
      },
    }
  );

  const roleOrphanLabels = {
    classification_model: 'Intent classification model',
    image_generation_model: 'Image generation model',
    text_completion_model: 'Text completion model',
    image_analysis_model: 'Image analysis (vision) model',
  };

  return (
    <Box sx={{ px: { xs: 2, sm: 3 }, py: 2 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure your Bastion Workspace settings and manage users.
      </Typography>

      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', md: 'row' },
          alignItems: 'flex-start',
          gap: 3,
        }}
      >
        <Paper
          variant="outlined"
          sx={{
            width: { xs: '100%', md: 260 },
            flexShrink: 0,
            position: { md: 'sticky' },
            top: { md: 16 },
            maxHeight: { md: 'calc(100vh - 160px)' },
            overflow: 'auto',
          }}
        >
          <List dense>
            {tabs.map((tab, index) => (
              <ListItemButton
                key={tab.id}
                selected={currentTab === index}
                onClick={() => handleTabSelect(index)}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>{tab.icon}</ListItemIcon>
                <ListItemText primary={tab.label} />
              </ListItemButton>
            ))}
          </List>
        </Paper>

        <Box sx={{ flex: 1, minWidth: 0 }}>
      {/* Settings Content */}
      {currentTab === 0 && (
        <Grid container spacing={3}>
          {/* User Profile Settings */}
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Person sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">User Profile Settings</Typography>
                  </Box>

                  <Grid container spacing={3} sx={{ mb: 3 }}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom>
                        Preferred Name
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        How would you like our AI agents to address you?
                      </Typography>
                      <TextField
                        fullWidth
                        label="Preferred Name"
                        value={userPreferredName}
                        onChange={(e) => setUserPreferredName(e.target.value)}
                        sx={{ mb: 2 }}
                        disabled={preferredNameMutation.isLoading}
                      />
                      <Button
                        variant="contained"
                        onClick={() => preferredNameMutation.mutate(userPreferredName)}
                        disabled={preferredNameMutation.isLoading}
                        startIcon={preferredNameMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                      >
                        {preferredNameMutation.isLoading ? 'Updating...' : 'Update Preferred Name'}
                      </Button>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom>
                        Display Name
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        Name shown across your profile and user-facing areas.
                      </Typography>
                      <TextField
                        fullWidth
                        label="Display Name"
                        value={profileDisplayName}
                        onChange={(e) => setProfileDisplayName(e.target.value)}
                        sx={{ mb: 2 }}
                        disabled={updateProfileMutation.isLoading}
                        placeholder="Optional"
                      />
                      <Button
                        variant="contained"
                        onClick={() => {
                          const email = profileEmail.trim();
                          if (!email) {
                            setSnackbar({ open: true, message: 'Email is required', severity: 'error' });
                            return;
                          }
                          updateProfileMutation.mutate({ email, display_name: profileDisplayName.trim() || null });
                        }}
                        disabled={updateProfileMutation.isLoading}
                        startIcon={updateProfileMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                      >
                        {updateProfileMutation.isLoading ? 'Saving…' : 'Save Display Name'}
                      </Button>
                    </Grid>
                  </Grid>

                  <Grid container spacing={3} sx={{ mb: 3 }}>
                    <Grid item xs={12} md={6}>
                      {/* Account information */}
                      <Box>
                        <Typography variant="h6" gutterBottom>
                          Account Information
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          <strong>Username:</strong> {user?.username} · <strong>Role:</strong> {user?.role}
                        </Typography>
                        <TextField
                          fullWidth
                          type="email"
                          label="Email"
                          value={profileEmail}
                          onChange={(e) => setProfileEmail(e.target.value)}
                          sx={{ mb: 2 }}
                          disabled={updateProfileMutation.isLoading}
                          helperText="Used for notifications and account communication."
                        />
                        <Button
                          variant="contained"
                          onClick={() => {
                            const email = profileEmail.trim();
                            if (!email) {
                              setSnackbar({ open: true, message: 'Email is required', severity: 'error' });
                              return;
                            }
                            updateProfileMutation.mutate({ email, display_name: profileDisplayName.trim() || null });
                          }}
                          disabled={updateProfileMutation.isLoading}
                          startIcon={updateProfileMutation.isLoading ? <CircularProgress size={20} /> : <Email />}
                          sx={{ mb: 2 }}
                        >
                          {updateProfileMutation.isLoading ? 'Saving…' : 'Save Email'}
                        </Button>

                        <TextField
                          fullWidth
                          label="Phone Number"
                          value={profilePhoneNumber}
                          onChange={(e) => setProfilePhoneNumber(e.target.value)}
                          sx={{ mb: 2 }}
                          disabled={phoneNumberMutation.isLoading}
                          placeholder="Optional"
                          helperText="Optional, for future notification channels."
                        />
                        <Button
                          variant="contained"
                          onClick={() => phoneNumberMutation.mutate(profilePhoneNumber.trim())}
                          disabled={phoneNumberMutation.isLoading}
                          startIcon={phoneNumberMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                          sx={{ mb: 2 }}
                        >
                          {phoneNumberMutation.isLoading ? 'Saving…' : 'Save Phone Number'}
                        </Button>

                        <TextField
                          fullWidth
                          type="date"
                          label="Birthday"
                          value={profileBirthday}
                          onChange={(e) => setProfileBirthday(e.target.value)}
                          sx={{ mb: 2 }}
                          disabled={birthdayMutation.isLoading}
                          InputLabelProps={{ shrink: true }}
                          helperText="Optional"
                        />
                        <Button
                          variant="contained"
                          onClick={() => birthdayMutation.mutate(profileBirthday)}
                          disabled={birthdayMutation.isLoading}
                          startIcon={birthdayMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                        >
                          {birthdayMutation.isLoading ? 'Saving…' : 'Save Birthday'}
                        </Button>
                      </Box>
                    </Grid>

                    <Grid item xs={12} md={6}>
                      {/* AI Context Setting */}
                      <Box>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="h6" gutterBottom sx={{ mb: 0, mr: 1 }}>
                            AI Context
                          </Typography>
                          <Tooltip title={
                            <Box>
                              <Typography variant="body2" gutterBottom>Examples:</Typography>
                              <Typography variant="body2">• "I'm a software developer working in Python and React"</Typography>
                              <Typography variant="body2">• "I prefer detailed technical explanations"</Typography>
                              <Typography variant="body2">• "I'm learning programming - explain step-by-step"</Typography>
                              <Typography variant="body2">• "I have ADHD - concise responses help me focus"</Typography>
                            </Box>
                          } arrow placement="right">
                            <Info sx={{ fontSize: 18, color: 'text.secondary', cursor: 'help' }} />
                          </Tooltip>
                        </Box>

                        <TextField
                          fullWidth
                          multiline
                          rows={6}
                          label="Tell your AI about yourself"
                          value={userAiContext}
                          onChange={(e) => {
                            if (e.target.value.length <= 500) {
                              setUserAiContext(e.target.value);
                            }
                          }}
                          inputProps={{ maxLength: 500 }}
                          sx={{ mb: 2 }}
                          disabled={aiContextMutation.isLoading}
                          helperText={`${userAiContext.length}/500 characters`}
                        />

                        <Alert severity="warning" sx={{ mb: 2 }}>
                          <strong>Privacy Notice:</strong> This information will be included in all agent conversations
                          and may be transmitted to external AI providers (OpenAI, Anthropic, etc.) as part of system prompts.
                        </Alert>

                        <Button
                          variant="contained"
                          onClick={() => aiContextMutation.mutate(userAiContext)}
                          disabled={aiContextMutation.isLoading}
                          startIcon={aiContextMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                        >
                          {aiContextMutation.isLoading ? 'Updating...' : 'Update AI Context'}
                        </Button>
                      </Box>
                    </Grid>
                  </Grid>

                  {/* Remembered Facts */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Remembered Facts
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Facts stored here are included in your AI context automatically. You can add or remove them below; agents can also save facts when you ask them to remember something.
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mb: 2 }}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={factsInjectEnabled}
                            onChange={(e) => {
                              const v = e.target.checked;
                              setFactsInjectEnabled(v);
                              setFactsPreferencesMutation.mutate({ facts_inject_enabled: v, facts_write_enabled: factsWriteEnabled });
                            }}
                            color="primary"
                          />
                        }
                        label="Include facts in AI conversations"
                      />
                      <FormControlLabel
                        control={
                          <Switch
                            checked={factsWriteEnabled}
                            onChange={(e) => {
                              const v = e.target.checked;
                              setFactsWriteEnabled(v);
                              setFactsPreferencesMutation.mutate({ facts_inject_enabled: factsInjectEnabled, facts_write_enabled: v });
                            }}
                            color="primary"
                          />
                        }
                        label="Allow agents to save new facts"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                      {userFacts.map((f) => (
                        <Chip
                          key={f.fact_key}
                          label={`${f.fact_key}: ${f.value}`}
                          onDelete={() => deleteUserFactMutation.mutate(f.fact_key)}
                          disabled={deleteUserFactMutation.isLoading}
                          size="small"
                        />
                      ))}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                      <TextField
                        size="small"
                        placeholder="Key (e.g. job_title)"
                        value={newFactKey}
                        onChange={(e) => setNewFactKey(e.target.value)}
                        sx={{ minWidth: 140 }}
                      />
                      <TextField
                        size="small"
                        placeholder="Value"
                        value={newFactValue}
                        onChange={(e) => setNewFactValue(e.target.value)}
                        sx={{ minWidth: 180 }}
                      />
                      <Button
                        variant="outlined"
                        size="small"
                        onClick={() => {
                          if (newFactKey.trim() && newFactValue.trim()) {
                            addUserFactMutation.mutate({
                              fact_key: newFactKey.trim(),
                              value: newFactValue.trim(),
                              category: 'general'
                            });
                          }
                        }}
                        disabled={addUserFactMutation.isLoading || !newFactKey.trim() || !newFactValue.trim()}
                      >
                        Add Fact
                      </Button>
                    </Box>
                  </Box>

                  {/* Activity History (episodic memory) */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Activity History
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Recent conversation activity is stored to help the AI remember what you worked on. You can clear entries below.
                    </Typography>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={episodesInjectEnabled}
                          onChange={(e) => {
                            const v = e.target.checked;
                            setEpisodesInjectEnabled(v);
                            setEpisodesPreferencesMutation.mutate({ episodes_inject_enabled: v });
                          }}
                          color="primary"
                        />
                      }
                      label="Track activity for AI context"
                    />
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1, mt: 1 }}>
                      {userEpisodes.slice(0, 20).map((ep) => (
                        <Chip
                          key={ep.id}
                          label={ep.summary?.slice(0, 40) + (ep.summary?.length > 40 ? '...' : '')}
                          onDelete={() => deleteEpisodeMutation.mutate(ep.id)}
                          disabled={deleteEpisodeMutation.isLoading}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                    {userEpisodes.length > 0 && (
                      <Button
                        size="small"
                        color="secondary"
                        onClick={() => deleteAllEpisodesMutation.mutate()}
                        disabled={deleteAllEpisodesMutation.isLoading}
                      >
                        Clear all activity
                      </Button>
                    )}
                  </Box>

                  {/* Pending Fact Updates */}
                  {pendingFacts.length > 0 && (
                    <Box mb={3}>
                      <Typography variant="h6" gutterBottom>
                        Pending Fact Updates
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        The AI suggested changes to facts you set. Accept or reject below.
                      </Typography>
                      {pendingFacts.map((p) => (
                        <Box key={p.id} sx={{ mb: 2, p: 1.5, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                          <Typography variant="body2"><strong>{p.fact_key}</strong></Typography>
                          <Typography variant="body2" color="text.secondary">Current (yours): {p.old_value}</Typography>
                          <Typography variant="body2" color="text.secondary">Proposed: {p.new_value}</Typography>
                          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                            <Button size="small" variant="contained" onClick={() => resolvePendingFactMutation.mutate({ historyId: p.id, action: 'accept' })} disabled={resolvePendingFactMutation.isLoading}>Accept</Button>
                            <Button size="small" variant="outlined" onClick={() => resolvePendingFactMutation.mutate({ historyId: p.id, action: 'reject' })} disabled={resolvePendingFactMutation.isLoading}>Reject</Button>
                          </Box>
                        </Box>
                      ))}
                    </Box>
                  )}

                  {/* Fact Change History */}
                  {factHistory.length > 0 && (
                    <Box mb={3}>
                      <Typography variant="h6" gutterBottom>
                        Fact Change History
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        Recent changes to your remembered facts.
                      </Typography>
                      {factHistory.slice(0, 10).map((h) => (
                        <Typography key={h.id} variant="body2" sx={{ mb: 0.5 }}>
                          <strong>{h.fact_key}</strong>: {h.old_value} → {h.new_value} ({h.resolution || 'updated'})
                        </Typography>
                      ))}
                    </Box>
                  )}

                  {/* Vision Features Setting */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Face Detection & Tagging
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Enable face detection and tagging features for images. This allows you to identify and search for people in your photos.
                    </Typography>

                    {!visionServiceAvailable && (
                      <Alert severity="warning" sx={{ mb: 2 }}>
                        Vision service is unavailable. The image-vision-service container may not be running.
                        Face detection features will be disabled until the service is available.
                      </Alert>
                    )}

                    <FormControlLabel
                      control={
                        <Switch
                          checked={visionFeaturesEnabled}
                          onChange={(e) => {
                            const newValue = e.target.checked;
                            setVisionFeaturesEnabled(newValue);
                            visionFeaturesMutation.mutate(newValue);
                          }}
                          disabled={!visionServiceAvailable || visionFeaturesMutation.isLoading}
                        />
                      }
                      label={
                        visionServiceAvailable
                          ? "Enable Face Detection & Tagging"
                          : "Face Detection (Service Unavailable)"
                      }
                    />

                    {visionServiceAvailable && visionFeaturesEnabled && (
                      <Alert severity="info" sx={{ mt: 2 }}>
                        Face detection is enabled. You can now use the "Tag Faces" button when viewing images.
                        Processing may take 10-30 seconds per image on CPU.
                      </Alert>
                    )}
                  </Box>

                  {/* Time and Locale Settings */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Time and Locale
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Configure timezone and time display format for status bar and AI time-aware responses.
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Timezone</InputLabel>
                          <Select
                            value={userTimezone}
                            onChange={(e) => setUserTimezone(e.target.value)}
                            label="Timezone"
                            disabled={timezoneMutation.isLoading}
                          >
                            <MenuItem value="UTC">UTC (Coordinated Universal Time)</MenuItem>
                            <MenuItem value="America/New_York">Eastern Time (ET)</MenuItem>
                            <MenuItem value="America/Chicago">Central Time (CT)</MenuItem>
                            <MenuItem value="America/Denver">Mountain Time (MT)</MenuItem>
                            <MenuItem value="America/Los_Angeles">Pacific Time (PT)</MenuItem>
                            <MenuItem value="Europe/London">London (GMT/BST)</MenuItem>
                            <MenuItem value="Europe/Paris">Paris (CET/CEST)</MenuItem>
                            <MenuItem value="Europe/Berlin">Berlin (CET/CEST)</MenuItem>
                            <MenuItem value="Asia/Tokyo">Tokyo (JST)</MenuItem>
                            <MenuItem value="Asia/Shanghai">Shanghai (CST)</MenuItem>
                            <MenuItem value="Australia/Sydney">Sydney (AEST/AEDT)</MenuItem>
                            <MenuItem value="Pacific/Auckland">Auckland (NZST/NZDT)</MenuItem>
                          </Select>
                        </FormControl>
                        <Button
                          variant="contained"
                          onClick={() => timezoneMutation.mutate(userTimezone)}
                          disabled={timezoneMutation.isLoading}
                          startIcon={timezoneMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                        >
                          {timezoneMutation.isLoading ? 'Updating...' : 'Update Timezone'}
                        </Button>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Time Format</InputLabel>
                          <Select
                            value={userTimeFormat}
                            onChange={(e) => setUserTimeFormat(e.target.value)}
                            label="Time Format"
                            disabled={timeFormatMutation.isLoading}
                          >
                            <MenuItem value="12h">12-hour (AM/PM)</MenuItem>
                            <MenuItem value="24h">24-hour (Military)</MenuItem>
                          </Select>
                        </FormControl>
                        <Button
                          variant="contained"
                          onClick={() => timeFormatMutation.mutate(userTimeFormat)}
                          disabled={timeFormatMutation.isLoading}
                          startIcon={timeFormatMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                        >
                          {timeFormatMutation.isLoading ? 'Updating...' : 'Update Time Format'}
                        </Button>
                      </Grid>
                    </Grid>
                  </Box>

                  {/* Zip Code Setting */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Zip Code
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Enter your zip code to display local weather conditions in the status bar.
                    </Typography>
                    
                    <TextField
                      fullWidth
                      label="Zip Code"
                      value={userZipCode}
                      onChange={(e) => {
                        const value = e.target.value.replace(/\D/g, '').slice(0, 5);
                        setUserZipCode(value);
                      }}
                      inputProps={{ maxLength: 5 }}
                      sx={{ mb: 2 }}
                      helperText="5-digit US zip code"
                      disabled={zipCodeMutation.isLoading}
                    />

                    <Button
                      variant="contained"
                      onClick={() => {
                        if (userZipCode.length === 5) {
                          zipCodeMutation.mutate(userZipCode);
                        } else {
                          setSnackbar({
                            open: true,
                            message: 'Zip code must be 5 digits',
                            severity: 'error'
                          });
                        }
                      }}
                      disabled={zipCodeMutation.isLoading || userZipCode.length !== 5}
                      startIcon={zipCodeMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                    >
                      {zipCodeMutation.isLoading ? 'Updating...' : 'Update Zip Code'}
                    </Button>
                  </Box>

                  {/* Password Change Setting */}
                  <Box mb={3}>
                    <Box display="flex" alignItems="center" mb={1}>
                      <Security sx={{ mr: 1, color: 'primary.main' }} />
                      <Typography variant="h6" gutterBottom sx={{ mb: 0 }}>
                        Change Password
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Update your account password. You will be logged out of all sessions after changing your password.
                    </Typography>
                    
                    <TextField
                      fullWidth
                      label="Current Password"
                      type={showPasswords.current ? 'text' : 'password'}
                      value={passwordChange.current_password}
                      onChange={(e) => setPasswordChange({ ...passwordChange, current_password: e.target.value })}
                      sx={{ mb: 2 }}
                      disabled={passwordChangeMutation.isLoading}
                      InputProps={{
                        endAdornment: (
                          <IconButton
                            onClick={() => togglePasswordVisibility('current')}
                            edge="end"
                          >
                            {showPasswords.current ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        )
                      }}
                    />

                    <TextField
                      fullWidth
                      label="New Password"
                      type={showPasswords.new ? 'text' : 'password'}
                      value={passwordChange.new_password}
                      onChange={(e) => setPasswordChange({ ...passwordChange, new_password: e.target.value })}
                      sx={{ mb: 2 }}
                      disabled={passwordChangeMutation.isLoading}
                      helperText="Must be at least 8 characters long"
                      InputProps={{
                        endAdornment: (
                          <IconButton
                            onClick={() => togglePasswordVisibility('new')}
                            edge="end"
                          >
                            {showPasswords.new ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        )
                      }}
                    />

                    <TextField
                      fullWidth
                      label="Confirm New Password"
                      type={showPasswords.confirm ? 'text' : 'password'}
                      value={passwordChange.confirm_password}
                      onChange={(e) => setPasswordChange({ ...passwordChange, confirm_password: e.target.value })}
                      sx={{ mb: 2 }}
                      disabled={passwordChangeMutation.isLoading}
                      InputProps={{
                        endAdornment: (
                          <IconButton
                            onClick={() => togglePasswordVisibility('confirm')}
                            edge="end"
                          >
                            {showPasswords.confirm ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        )
                      }}
                    />

                    <Button
                      variant="contained"
                      onClick={handlePasswordChange}
                      disabled={passwordChangeMutation.isLoading}
                      startIcon={passwordChangeMutation.isLoading ? <CircularProgress size={20} /> : <Security />}
                    >
                      {passwordChangeMutation.isLoading ? 'Changing Password...' : 'Change Password'}
                    </Button>
                  </Box>

                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      )}

      {currentTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Palette sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">Appearance</Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Theme mode, accent color, and optional wallpapers for the web app and BBS. Changes apply
                    immediately where applicable.
                  </Typography>

                  <Box mb={3}>
                    <Typography variant="subtitle2" gutterBottom>Theme mode</Typography>
                    <ToggleButtonGroup
                      value={themePreference}
                      exclusive
                      onChange={(_, value) => {
                        if (value !== null) setThemePreference(value);
                      }}
                      aria-label="Theme mode"
                      size="small"
                    >
                      <ToggleButton value="light" aria-label="Light">Light</ToggleButton>
                      <ToggleButton value="dark" aria-label="Dark">Dark</ToggleButton>
                      <ToggleButton value="system" aria-label="Match system">
                        <BrightnessAuto sx={{ mr: 0.75, fontSize: '1.125rem', verticalAlign: 'text-bottom' }} />
                        System
                      </ToggleButton>
                    </ToggleButtonGroup>
                    {themePreference === 'system' && (
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                        Matches this device ({systemPrefersDark ? 'dark' : 'light'} right now). Quick light/dark toggle
                        is hidden from the user menu until you choose Light or Dark here.
                      </Typography>
                    )}
                  </Box>

                  <Box mb={3}>
                    <Typography variant="subtitle2" gutterBottom sx={{ mb: 1 }}>Accent color</Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {ACCENT_IDS.map((id) => {
                        const palette = ACCENT_PALETTES[id];
                        const mainColor = darkMode ? palette?.dark?.primary?.main : palette?.light?.primary?.main;
                        const isSelected = accentId === id;
                        return (
                          <Tooltip key={id} title={id.charAt(0).toUpperCase() + id.slice(1)}>
                            <Box
                              component="button"
                              type="button"
                              onClick={() => setAccentId(id)}
                              aria-label={`Accent ${id}`}
                              aria-pressed={isSelected}
                              sx={{
                                width: 40,
                                height: 40,
                                borderRadius: '50%',
                                border: 2,
                                borderColor: isSelected ? 'primary.main' : 'divider',
                                bgcolor: mainColor || '#1976d2',
                                cursor: 'pointer',
                                p: 0,
                                '&:hover': { opacity: 0.9 },
                                '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 2 },
                              }}
                            />
                          </Tooltip>
                        );
                      })}
                    </Box>
                  </Box>

                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setAppearance({ mode: false, accentId: DEFAULT_ACCENT_ID })}
                  >
                    Reset to default (Light, Blue)
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
          </Grid>
          <Grid item xs={12}>
            <UiWallpaperSettingsSection />
          </Grid>
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
          </Grid>
          <Grid item xs={12}>
            <BbsWallpaperSettingsTab />
          </Grid>
        </Grid>
      )}

      {currentTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
                    <Box display="flex" alignItems="center">
                      <Psychology sx={{ mr: 2, color: 'primary.main' }} />
                      <Typography variant="h6">Personas</Typography>
                    </Box>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={handleOpenCreatePersona}
                    >
                      Create persona
                    </Button>
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    <strong>Default persona:</strong> This is used for main chat and for Agent Factory agents that use &quot;default persona&quot;. You can also create custom personas and assign a specific one per agent.
                  </Alert>

                  <Box mb={3}>
                    <Typography variant="subtitle1" gutterBottom>Default persona</Typography>
                    <FormControl fullWidth sx={{ maxWidth: 400 }}>
                      <InputLabel>Default persona</InputLabel>
                      <Select
                        value={defaultPersona?.id ?? ''}
                        onChange={(e) => setDefaultPersonaMutation.mutate(e.target.value || null)}
                        label="Default persona"
                        disabled={setDefaultPersonaMutation.isLoading}
                      >
                        <MenuItem value="">
                          <em>None (use first built-in)</em>
                        </MenuItem>
                        {personasList.map((p) => (
                          <MenuItem key={p.id} value={p.id}>
                            {p.name}{p.is_builtin ? ' (built-in)' : ''}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>

                  <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>Your custom personas</Typography>
                  {personasList.filter((p) => !p.is_builtin).length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                      No custom personas yet. Create one to use your own name, style instructions, and political bias.
                    </Typography>
                  ) : (
                    <Box display="flex" flexDirection="column" gap={1}>
                      {personasList.filter((p) => !p.is_builtin).map((p) => (
                        <Paper key={p.id} variant="outlined" sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <Box>
                            <Typography fontWeight="medium">{p.name}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              AI name: {p.ai_name} · Bias: {promptOptions.political_biases?.find(b => b.value === p.political_bias)?.label || p.political_bias}
                            </Typography>
                            {p.style_instruction && (
                              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5, maxWidth: 600 }} noWrap>
                                {p.style_instruction.slice(0, 80)}...
                              </Typography>
                            )}
                          </Box>
                          <Box>
                            <IconButton size="small" onClick={() => handleOpenEditPersona(p)} title="Edit">
                              <EditIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={() => deletePersonaMutation.mutate(p.id)} title="Delete">
                              <DeleteSweep fontSize="small" />
                            </IconButton>
                          </Box>
                        </Paper>
                      ))}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      )}

      {/* Create/Edit Persona Dialog */}
      <Dialog open={personaDialogOpen} onClose={() => setPersonaDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>{personaDialogMode === 'create' ? 'Create persona' : 'Edit persona'}</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Persona name"
            fullWidth
            value={personaForm.name}
            onChange={(e) => setPersonaForm((f) => ({ ...f, name: e.target.value }))}
          />
          <TextField
            margin="dense"
            label="AI name"
            fullWidth
            value={personaForm.ai_name}
            onChange={(e) => setPersonaForm((f) => ({ ...f, ai_name: e.target.value }))}
          />
          <TextField
            margin="dense"
            label="Style instruction (free-form)"
            fullWidth
            multiline
            rows={4}
            value={personaForm.style_instruction}
            onChange={(e) => setPersonaForm((f) => ({ ...f, style_instruction: e.target.value }))}
            placeholder="e.g. Speak like a pirate captain who loves cooking."
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Political bias</InputLabel>
            <Select
              value={personaForm.political_bias}
              label="Political bias"
              onChange={(e) => setPersonaForm((f) => ({ ...f, political_bias: e.target.value }))}
            >
              {(promptOptions.political_biases || []).map((b) => (
                <MenuItem key={b.value} value={b.value}>{b.label}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <TextField
            margin="dense"
            label="Description (optional)"
            fullWidth
            multiline
            rows={2}
            value={personaForm.description}
            onChange={(e) => setPersonaForm((f) => ({ ...f, description: e.target.value }))}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPersonaDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSavePersonaDialog}
            disabled={!personaForm.name.trim() || createPersonaMutation.isLoading || updatePersonaMutation.isLoading}
          >
            {personaDialogMode === 'create' ? 'Create' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      {currentTab === 3 && (
        <Grid container spacing={3}>

        {/* User-level LLM providers (toggle + own API keys / models) */}
        <Grid item xs={12}>
          <UserLLMProviders />
        </Grid>

        <Grid item xs={12}>
          <UserVoiceProviders />
        </Grid>

        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.04 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <Psychology sx={{ mr: 1 }} />
                  While the AI is responding
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  If you type another message and send before the current reply finishes, choose what happens.
                </Typography>
                <ToggleButtonGroup
                  exclusive
                  fullWidth
                  size="small"
                  value={userModelRoles.send_while_streaming_behavior === 'stop_and_send' ? 'stop_and_send' : 'queue'}
                  onChange={(_, v) => {
                    if (v != null) {
                      handleUserModelRoleChange('send_while_streaming_behavior', v);
                    }
                  }}
                  disabled={setUserModelRolesMutation.isLoading}
                  sx={{ mt: 2 }}
                >
                  <ToggleButton value="queue">
                    Queue next message
                  </ToggleButton>
                  <ToggleButton value="stop_and_send">
                    Stop and send now
                  </ToggleButton>
                </ToggleButtonGroup>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1.5 }}>
                  Queue waits for the current stream to end, then sends. Stop and send cancels the in-flight reply
                  and sends your new message (checkpoints for that turn are cleared on stop).
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Global model status (admin): reflects shared settings table, not per-user chat/sidebar choice */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                  <Psychology sx={{ mr: 1 }} />
                  Current AI Model Configuration
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Server-wide defaults from the shared settings store. Per-user chat uses the sidebar and model
                  preference keys, not this summary.
                </Typography>
                <ModelStatusDisplay />
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* My Model Preferences - non-admin, own providers only */}
        {useOwnProviders && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Psychology sx={{ mr: 1 }} />
                    My Model Preferences
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    Set which model to use for each role. Only your enabled models are shown. Leave as &quot;Use admin default&quot; to inherit system defaults.
                  </Typography>
                  <Grid container spacing={2} sx={{ mt: 1 }}>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Main chat model</InputLabel>
                        <Select
                          value={userModelRoles.user_chat_model || ''}
                          onChange={(e) => handleUserModelRoleChange('user_chat_model', e.target.value)}
                          label="Main chat model"
                          disabled={setUserModelRolesMutation.isLoading}
                        >
                          <MenuItem value="">
                            <em>Use admin default</em>
                          </MenuItem>
                          {enabledModelsList.map((m) => (
                            <MenuItem key={m.id} value={m.id}>
                              {m.name || m.id}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                        Default for chat; you can change it in the chat sidebar.
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Fast / classification model</InputLabel>
                        <Select
                          value={userModelRoles.user_fast_model || ''}
                          onChange={(e) => handleUserModelRoleChange('user_fast_model', e.target.value)}
                          label="Fast / classification model"
                          disabled={setUserModelRolesMutation.isLoading}
                        >
                          <MenuItem value="">
                            <em>Use admin default</em>
                          </MenuItem>
                          {enabledModelsList.map((m) => (
                            <MenuItem key={m.id} value={m.id}>
                              {m.name || m.id}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Image generation model</InputLabel>
                        <Select
                          value={userModelRoles.user_image_gen_model || ''}
                          onChange={(e) => handleUserModelRoleChange('user_image_gen_model', e.target.value)}
                          label="Image generation model"
                          disabled={setUserModelRolesMutation.isLoading}
                        >
                          <MenuItem value="">
                            <em>Use admin default</em>
                          </MenuItem>
                          {imageGenerationModels.map((m) => (
                            <MenuItem key={m.id} value={m.id}>
                              {m.name || m.id}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Image analysis model</InputLabel>
                        <Select
                          value={userModelRoles.user_image_analysis_model || ''}
                          onChange={(e) => handleUserModelRoleChange('user_image_analysis_model', e.target.value)}
                          label="Image analysis model"
                          disabled={setUserModelRolesMutation.isLoading}
                        >
                          <MenuItem value="">
                            <em>Use admin default</em>
                          </MenuItem>
                          {imageAnalysisModels.map((m) => (
                            <MenuItem key={m.id} value={m.id}>
                              {m.name || m.id}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}

        {/* Classification Model Selection - Admin Only */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card id="settings-classification-model-card" sx={{ border: '2px solid #2196f3', borderRadius: 2 }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Speed sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6" color="primary">
                      Intent Classification Model
                    </Typography>
                    <Chip 
                      label="Performance Critical" 
                      size="small" 
                      color="primary" 
                      sx={{ ml: 2 }}
                    />
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    <strong>Fast Classification:</strong> This model is used for quick intent classification to determine 
                    execution mode (chat/direct/plan/execute). Choose a lightweight, fast model for best performance.
                  </Alert>

                  <ClassificationModelSelector 
                    enabledModels={enabledModels}
                    modelsData={modelsData}
                    modelsLoading={modelsLoading}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}

        {/* Image Generation Model Selection - Admin Only */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.06 }}
            >
              <Card id="settings-image-gen-model-card" sx={{ border: '2px solid #4caf50', borderRadius: 2 }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Psychology sx={{ mr: 2, color: 'success.main' }} />
                    <Typography variant="h6" color="success.main">
                      Image Generation Model
                    </Typography>
                    <Chip 
                      label="Used by Image Generation Agent" 
                      size="small" 
                      color="success" 
                      sx={{ ml: 2 }}
                    />
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    Select the OpenRouter model the Image Generation Agent will use to create images.
                  </Alert>

                  <ImageGenerationModelSelector 
                    enabledModels={enabledModels}
                    modelsData={modelsData}
                    modelsLoading={modelsLoading}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}

        {/* Text Completion Model Selection - Admin Only */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card id="settings-text-completion-model-card" sx={{ border: '2px solid #00bcd4', borderRadius: 2 }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Speed sx={{ mr: 2, color: 'info.main' }} />
                    <Typography variant="h6" color="info.main">
                      Text Completion Model
                    </Typography>
                    <Chip 
                      label="Performance Critical" 
                      size="small" 
                      color="info" 
                      sx={{ ml: 2 }}
                    />
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    <strong>Fast Completions:</strong> This model is used for editor suggestions and proofreading.
                    Choose a lightweight, fast model separate from the main chat model.
                  </Alert>

                  <TextCompletionModelSelector 
                    enabledModels={enabledModels}
                    modelsData={modelsData}
                    modelsLoading={modelsLoading}
                  />
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}

        {/* Image Analysis Model Selection - Admin Only */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card id="settings-image-analysis-model-card" sx={{ border: '2px solid #9c27b0', borderRadius: 2 }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Visibility sx={{ mr: 2, color: 'secondary.main' }} />
                    <Typography variant="h6" color="secondary.main">
                      Image Analysis Model
                    </Typography>
                    <Chip 
                      label="Vision / Describe" 
                      size="small" 
                      color="secondary" 
                      sx={{ ml: 2 }}
                    />
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    Select the vision model used for image description (metadata overlay and chat).
                  </Alert>

                  <ImageGenerationModelSelector 
                    enabledModels={enabledModels}
                    modelsData={modelsData}
                    modelsLoading={modelsLoading}
                    variant="analysis"
                  />
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}

        {/* Enhanced Model Management - Admin Only */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
                  <Box display="flex" alignItems="center">
                    <Psychology sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">OpenRouter Model Management</Typography>
                    <Badge badgeContent={enabledModels.size} color="primary" sx={{ ml: 2 }}>
                      <Chip label="Enabled" size="small" />
                    </Badge>
                  </Box>
                  
                  <Box display="flex" gap={1}>
                    <Tooltip title="Refresh models from OpenRouter">
                      <IconButton onClick={() => refetchModels()} disabled={modelsLoading}>
                        <Refresh />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>

                {orphanedEnabledModels.length > 0 && (
                  <Alert
                    severity="warning"
                    sx={{ mb: 2 }}
                    action={
                      <Button
                        color="inherit"
                        size="small"
                        disabled={saveEnabledModelsMutation.isLoading}
                        onClick={() => handleRemoveOrphanedEnabledModels(orphanedEnabledModels)}
                      >
                        Remove all stale
                      </Button>
                    }
                  >
                    <Typography variant="subtitle2" gutterBottom>
                      {orphanedEnabledModels.length} enabled model ID(s) are no longer returned by the live provider catalog
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      They remain in the enabled list until removed. Use Remove on each row or clear all at once.
                    </Typography>
                    <List dense disablePadding sx={{ bgcolor: 'action.hover', borderRadius: 1 }}>
                      {orphanedEnabledModels.map((id) => (
                        <ListItem
                          key={id}
                          secondaryAction={
                            <Button
                              size="small"
                              disabled={saveEnabledModelsMutation.isLoading}
                              onClick={() => handleRemoveOrphanedEnabledModels([id])}
                            >
                              Remove
                            </Button>
                          }
                        >
                          <ListItemText
                            primary={id}
                            primaryTypographyProps={{ variant: 'body2', fontFamily: 'monospace' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Alert>
                )}

                {orphanedRoleModelEntries.length > 0 && (
                  <Alert severity="warning" sx={{ mb: 2 }} id="settings-stale-role-models">
                    <Typography variant="subtitle2" gutterBottom>
                      Role-specific model settings point at IDs not in the current provider catalog
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Jump to the section to pick a valid model, or clear the saved value (classification resets to follow chat model).
                    </Typography>
                    <List dense disablePadding>
                      {orphanedRoleModelEntries.map(([key, modelId]) => (
                        <ListItem
                          key={key}
                          secondaryAction={
                            <Box display="flex" gap={0.5} flexWrap="wrap" justifyContent="flex-end">
                              <Button
                                size="small"
                                onClick={() => scrollToRoleSettingSection(key)}
                              >
                                Jump
                              </Button>
                              <Button
                                size="small"
                                color="warning"
                                disabled={clearRoleOrphanMutation.isLoading}
                                onClick={() => clearRoleOrphanMutation.mutate(key)}
                              >
                                Clear
                              </Button>
                            </Box>
                          }
                        >
                          <ListItemText
                            primary={roleOrphanLabels[key] || key}
                            secondary={modelId}
                            secondaryTypographyProps={{ variant: 'body2', fontFamily: 'monospace' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Alert>
                )}

                {enabledModelsData?.catalog_verified === false &&
                  (enabledModelsData?.enabled_models?.length ?? 0) > 0 && (
                    <Alert severity="info" sx={{ mb: 2 }}>
                      The provider catalog is empty or could not be loaded, so stale enabled entries cannot be detected.
                      Fix connectivity or API keys, then use Refresh — or remove models manually from the list.
                    </Alert>
                  )}

                {/* Search and Filter Controls */}
                <Box mb={3}>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} md={4}>
                      <TextField
                        fullWidth
                        size="small"
                        placeholder="Search models..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        InputProps={{
                          startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
                        }}
                      />
                    </Grid>
                    
                    <Grid item xs={12} md={3}>
                      <FormControl fullWidth size="small">
                        <InputLabel shrink>Provider</InputLabel>
                        <Select
                          value={providerFilter}
                          onChange={(e) => setProviderFilter(e.target.value)}
                          label="Provider"
                          displayEmpty
                          notched
                        >
                          <MenuItem value="">
                            <em>Select a provider</em>
                          </MenuItem>
                          {providers.map(provider => (
                            <MenuItem key={provider} value={provider}>
                              {provider}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                    </Grid>
                    
                    <Grid item xs={12} md={3}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={showOnlyEnabled}
                            onChange={(e) => setShowOnlyEnabled(e.target.checked)}
                            size="small"
                          />
                        }
                        label="Show only enabled"
                      />
                    </Grid>
                    
                    <Grid item xs={12} md={2}>
                      <Typography variant="body2" color="text.secondary">
                        {filteredModels.length} models
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>

                {/* Status Messages */}
                {selectModelMutation.isSuccess && (
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Active model updated successfully!
                  </Alert>
                )}

                {selectModelMutation.isError && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    Failed to update model: {selectModelMutation.error?.response?.data?.detail}
                  </Alert>
                )}

                {modelsLoading ? (
                  <Box display="flex" alignItems="center" gap={2} p={3}>
                    <LinearProgress sx={{ flex: 1 }} />
                    <Typography variant="body2">Loading models from OpenRouter...</Typography>
                  </Box>
                ) : (
                  <>
                    {Object.keys(displayGroupedModels).length === 0 ? (
                      <Paper sx={{ p: 4, textAlign: 'center' }}>
                        <Typography variant="h6" color="text.secondary" gutterBottom>
                          {showOnlyEnabled 
                            ? 'No enabled models found' 
                            : 'Select a provider to view models'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {showOnlyEnabled
                            ? 'Enable some models to see them organized by provider'
                            : 'Choose a provider from the dropdown above to see available models'}
                        </Typography>
                      </Paper>
                    ) : (
                      <>
                        {/* Active Model Summary */}
                        {selectedModel && (
                          <Paper sx={{ p: 2, mb: 3, backgroundColor: '#e3f2fd' }}>
                            <Typography variant="subtitle2" gutterBottom>
                              🎯 Active Model
                            </Typography>
                            <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                              {modelsData?.models?.find(m => m.id === selectedModel)?.name || selectedModel}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {modelsData?.models?.find(m => m.id === selectedModel)?.provider} • 
                              {modelsData?.models?.find(m => m.id === selectedModel)?.context_length?.toLocaleString()} context
                            </Typography>
                          </Paper>
                        )}

                        {/* Models by Provider */}
                        {Object.entries(displayGroupedModels).map(([provider, models]) => (
                          <Accordion key={provider} defaultExpanded={models.some(m => enabledModels.has(m.id))}>
                            <AccordionSummary expandIcon={<ExpandMore />}>
                              <Box display="flex" alignItems="center" gap={2}>
                                <Typography variant="h6">{provider}</Typography>
                                <Chip 
                                  label={`${models.length} models`} 
                                  size="small" 
                                  variant="outlined" 
                                />
                                <Chip 
                                  label={`${models.filter(m => enabledModels.has(m.id)).length} enabled`} 
                                  size="small" 
                                  color="primary"
                                />
                              </Box>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Box>
                                {models.map(model => (
                                  <ModelCard key={model.id} model={model} />
                                ))}
                              </Box>
                            </AccordionDetails>
                          </Accordion>
                        ))}

                        {filteredModels.length === 0 && (
                          <Paper sx={{ p: 4, textAlign: 'center' }}>
                            <Typography variant="h6" color="text.secondary" gutterBottom>
                              No models found
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              Try adjusting your search or filter criteria
                            </Typography>
                          </Paper>
                        )}
                      </>
                    )}
                  </>
                )}

                <Divider sx={{ my: 3 }} />
                
                <Typography variant="body2" color="text.secondary">
                  💡 <strong>Tip:</strong> Enable multiple models to have options available in chat.
                  Only one model can be active at a time. Models are fetched live from the provider API.
                  Stale IDs that disappear from the catalog appear above with one-click removal.
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        )}

        {/* Model catalog for non-admins on shared (admin) models only — BYOK users enable models per provider in My AI Providers */}
        {user?.role !== 'admin' && !useOwnProviders && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Psychology sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">Available AI Models</Typography>
                    <Badge badgeContent={enabledModels.size} color="primary" sx={{ ml: 2 }}>
                      <Chip label="Enabled" size="small" />
                    </Badge>
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    Model management is restricted to administrators. You can select from the enabled models in the chat interface.
                  </Alert>

                  {orphanedEnabledModels.length > 0 && (
                    <Alert severity="warning" sx={{ mb: 3 }}>
                      <Typography variant="body2">
                        Some enabled models are no longer returned by the provider catalog. Ask an administrator to open
                        AI Models settings and remove stale entries ({orphanedEnabledModels.length}).
                      </Typography>
                    </Alert>
                  )}

                  {modelsLoading ? (
                    <Box display="flex" alignItems="center" gap={2} p={3}>
                      <LinearProgress sx={{ flex: 1 }} />
                      <Typography variant="body2">Loading models...</Typography>
                    </Box>
                  ) : (
                    <Box>
                      {Object.entries(groupedModels).map(([provider, models]) => {
                        const enabledModelsInProvider = models.filter(m => enabledModels.has(m.id));
                        if (enabledModelsInProvider.length === 0) return null;
                        
                        return (
                          <Accordion key={provider} defaultExpanded>
                            <AccordionSummary expandIcon={<ExpandMore />}>
                              <Box display="flex" alignItems="center" gap={2}>
                                <Typography variant="h6">{provider}</Typography>
                                <Chip 
                                  label={`${enabledModelsInProvider.length} enabled`} 
                                  size="small" 
                                  color="primary"
                                />
                              </Box>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Box>
                                {enabledModelsInProvider.map(model => (
                                  <Paper key={model.id} sx={{ p: 2, mb: 1, backgroundColor: '#f5f5f5' }}>
                                    <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                                      {model.name}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">
                                      {model.provider} • {model.context_length?.toLocaleString()} context
                                    </Typography>
                                  </Paper>
                                ))}
                              </Box>
                            </AccordionDetails>
                          </Accordion>
                        );
                      })}
                    </Box>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        )}

      </Grid>
      )}

      {currentTab === 4 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <RSSFeedSettings />
        </motion.div>
      )}

      {currentTab === 5 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <SettingsEbooksOpdsSection />
        </motion.div>
      )}

      {/* Org-Mode Settings Tab */}
      {currentTab === 6 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <OrgModeSettingsTab />
        </motion.div>
      )}

      {currentTab === 7 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <ZettelkastenSettingsTab />
        </motion.div>
      )}

      {/* Media Settings Tab */}
      {currentTab === 8 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <MediaSettingsTab />
        </motion.div>
      )}

      {/* Connections Tab */}
      {currentTab === 9 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <ExternalConnectionsSettings />
        </motion.div>
      )}

      {/* Browser Sessions Tab */}
      {currentTab === 10 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <BrowserSessionManagement />
        </motion.div>
      )}

      {/* Database Management Tab */}
      {currentTab === 11 && user?.role === 'admin' && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card sx={{ border: '2px solid #ff9800', borderRadius: 2 }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <DeleteSweep sx={{ mr: 2, color: 'warning.main' }} />
                    <Typography variant="h6" color="warning.main">
                      Database Management
                    </Typography>
                    <Chip 
                      label="Admin Only" 
                      size="small" 
                      color="warning" 
                      sx={{ ml: 2 }}
                    />
                  </Box>

                  <Alert severity="warning" sx={{ mb: 3 }}>
                    <strong>Caution:</strong> These operations will permanently delete all data from the respective databases. 
                    Use only when you want to start completely fresh.
                  </Alert>

                  <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 3, textAlign: 'center', border: '1px solid #e0e0e0' }}>
                        <Typography variant="h4" sx={{ mb: 1 }}>
                          📄
                        </Typography>
                        <Typography variant="h6" gutterBottom>
                          Document Database
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          All documents, notes, and associated files
                        </Typography>
                        <Button
                          variant="outlined"
                          color="error"
                          startIcon={<DeleteSweep />}
                          onClick={() => setDocumentsDialogOpen(true)}
                          disabled={clearDocumentsMutation.isLoading}
                          fullWidth
                          sx={{ mb: 1 }}
                        >
                          {clearDocumentsMutation.isLoading ? 'Clearing...' : 'Delete All Documents'}
                        </Button>
                        <Button
                          variant="outlined"
                          color="warning"
                          startIcon={<DeleteSweep />}
                          onClick={() => setDocumentsDatabaseOnlyDialogOpen(true)}
                          disabled={clearDocumentsDatabaseOnlyMutation.isLoading}
                          fullWidth
                        >
                          {clearDocumentsDatabaseOnlyMutation.isLoading ? 'Clearing...' : 'Clear Database Only (Re-sync from disk)'}
                        </Button>
                      </Paper>
                    </Grid>

                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 3, textAlign: 'center', border: '1px solid #e0e0e0' }}>
                        <Typography variant="h4" sx={{ mb: 1 }}>
                          🔍
                        </Typography>
                        <Typography variant="h6" gutterBottom>
                          Qdrant Vector Database
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Contains all document embeddings and search vectors
                        </Typography>
                        <Button
                          variant="outlined"
                          color="warning"
                          startIcon={<DeleteSweep />}
                          onClick={() => setQdrantDialogOpen(true)}
                          disabled={clearQdrantMutation.isLoading}
                          fullWidth
                        >
                          {clearQdrantMutation.isLoading ? 'Clearing...' : 'Clear Qdrant Database'}
                        </Button>
                      </Paper>
                    </Grid>

                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 3, textAlign: 'center', border: '1px solid #e0e0e0' }}>
                        <Typography variant="h4" sx={{ mb: 1 }}>
                          🕸️
                        </Typography>
                        <Typography variant="h6" gutterBottom>
                          Neo4j Knowledge Graph
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Contains extracted entities and their relationships
                        </Typography>
                        <Button
                          variant="outlined"
                          color="warning"
                          startIcon={<DeleteSweep />}
                          onClick={() => setNeo4jDialogOpen(true)}
                          disabled={clearNeo4jMutation.isLoading}
                          fullWidth
                        >
                          {clearNeo4jMutation.isLoading ? 'Clearing...' : 'Clear Neo4j Database'}
                        </Button>
                      </Paper>
                    </Grid>

                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 3, textAlign: 'center', border: '1px solid #e0e0e0' }}>
                        <Typography variant="h4" sx={{ mb: 1 }}>
                          👤
                        </Typography>
                        <Typography variant="h6" gutterBottom>
                          Face Recognition Data
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Contains detected faces and known identities
                        </Typography>
                        <Button
                          variant="outlined"
                          color="warning"
                          startIcon={<DeleteSweep />}
                          onClick={() => setFaceIdentitiesDialogOpen(true)}
                          disabled={clearFaceIdentitiesMutation.isLoading}
                          fullWidth
                          sx={{ mb: 1 }}
                        >
                          {clearFaceIdentitiesMutation.isLoading ? 'Clearing...' : 'Clear Face Identities'}
                        </Button>
                        <Button
                          variant="outlined"
                          size="small"
                          onClick={() => cleanupOrphanedVectorsMutation.mutate()}
                          disabled={cleanupOrphanedVectorsMutation.isLoading}
                          fullWidth
                        >
                          {cleanupOrphanedVectorsMutation.isLoading ? 'Cleaning...' : 'Cleanup Orphaned Vectors'}
                        </Button>
                      </Paper>
                    </Grid>

                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 3, textAlign: 'center', border: '1px solid #e0e0e0' }}>
                        <Typography variant="h4" sx={{ mb: 1 }}>
                          🔗
                        </Typography>
                        <Typography variant="h6" gutterBottom>
                          Document Links
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          File relation graph (org, markdown, and frontmatter links)
                        </Typography>
                        <Button
                          variant="outlined"
                          color="primary"
                          startIcon={<LinkIcon />}
                          onClick={() => rebuildAllLinksMutation.mutate()}
                          disabled={rebuildAllLinksMutation.isLoading}
                          fullWidth
                        >
                          {rebuildAllLinksMutation.isLoading ? 'Rebuilding...' : 'Rebuild All Links'}
                        </Button>
                      </Paper>
                    </Grid>
                  </Grid>

                  {/* Success/Error Messages */}
                  {clearDocumentsMutation.isSuccess && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Document database cleared successfully! All documents, notes, and associated data have been removed.
                    </Alert>
                  )}
                  
                  {clearDocumentsMutation.isError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      Failed to clear document database: {clearDocumentsMutation.error?.response?.data?.detail}
                    </Alert>
                  )}

                  {clearDocumentsDatabaseOnlyMutation.isSuccess && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Document database cleared (files left on disk). Documents will be re-read from disk.
                    </Alert>
                  )}
                  {clearDocumentsDatabaseOnlyMutation.isError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      Failed to clear document database only: {clearDocumentsDatabaseOnlyMutation.error?.response?.data?.detail}
                    </Alert>
                  )}

                  {clearQdrantMutation.isSuccess && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Qdrant database cleared successfully! All embeddings have been removed.
                    </Alert>
                  )}
                  
                  {clearQdrantMutation.isError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      Failed to clear Qdrant database: {clearQdrantMutation.error?.response?.data?.detail}
                    </Alert>
                  )}

                  {clearNeo4jMutation.isSuccess && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Neo4j database cleared successfully! All entities and relationships have been removed.
                    </Alert>
                  )}
                  
                  {clearNeo4jMutation.isError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      Failed to clear Neo4j database: {clearNeo4jMutation.error?.response?.data?.detail}
                    </Alert>
                  )}

                  {clearFaceIdentitiesMutation.isSuccess && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Face recognition data cleared successfully! All detected faces and known identities have been removed.
                    </Alert>
                  )}
                  
                  {clearFaceIdentitiesMutation.isError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      Failed to clear face identities: {clearFaceIdentitiesMutation.error?.response?.data?.detail}
                    </Alert>
                  )}

                  {rebuildAllLinksMutation.isSuccess && (
                    <Alert severity="success" sx={{ mt: 2 }}>
                      Document links rebuilt. {rebuildAllLinksMutation.data?.processed ?? 0} documents processed, {rebuildAllLinksMutation.data?.errors ?? 0} errors, {rebuildAllLinksMutation.data?.total ?? 0} total.
                    </Alert>
                  )}
                  {rebuildAllLinksMutation.isError && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      Failed to rebuild links: {rebuildAllLinksMutation.error?.response?.data?.detail}
                    </Alert>
                  )}
                </CardContent>
              </Card>
        </motion.div>
          </Grid>
        </Grid>
      )}

      {/* User Management Tab */}
      {currentTab === 12 && user?.role === 'admin' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <UserManagement />
        </motion.div>
      )}

      {currentTab === 13 && user?.role === 'admin' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <FederationSettings />
        </motion.div>
      )}

      {/* Confirmation Dialogs */}
      
      {/* Document Database Clear Confirmation */}
      <Dialog
        open={documentsDialogOpen}
        onClose={() => setDocumentsDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning sx={{ mr: 1, color: 'error.main' }} />
          Delete All Documents and Data
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            <strong>⚠️ EXTREME CAUTION: This action cannot be undone!</strong>
          </DialogContentText>
          <DialogContentText sx={{ mt: 2 }}>
            You are about to permanently delete <strong>ALL</strong> data from your knowledge base. 
            This comprehensive operation will:
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, mb: 2, pl: 2 }}>
            <li><strong>Delete all documents</strong> from PostgreSQL database</li>
            <li><strong>Remove all free-form notes</strong> and user-created content</li>
            <li><strong>Clear all document embeddings</strong> from Qdrant vector database</li>
            <li><strong>Delete all extracted entities</strong> from Neo4j knowledge graph</li>
            <li><strong>Remove all PDF segmentation data</strong> and annotations</li>
            <li><strong>Delete all uploaded files</strong> and processed content</li>
            <li><strong>Reset all database sequences</strong> to start fresh</li>
          </Box>
          <Alert severity="error" sx={{ my: 2 }}>
            <strong>After this operation:</strong><br/>
            • Your knowledge base will be completely empty<br/>
            • All search functionality will be reset<br/>
            • You will need to re-upload and re-process all documents<br/>
            • All user notes and annotations will be permanently lost
          </Alert>
          <DialogContentText>
            <strong>Are you absolutely sure you want to delete everything?</strong>
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setDocumentsDialogOpen(false)} 
            disabled={clearDocumentsMutation.isLoading}
          >
            Cancel
          </Button>
          <Button 
            onClick={() => clearDocumentsMutation.mutate()} 
            color="error" 
            variant="contained"
            disabled={clearDocumentsMutation.isLoading}
            startIcon={clearDocumentsMutation.isLoading ? null : <DeleteSweep />}
          >
            {clearDocumentsMutation.isLoading ? 'Deleting All Data...' : 'Delete Everything'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Document Database – Clear Database Only (re-sync from disk) */}
      <Dialog
        open={documentsDatabaseOnlyDialogOpen}
        onClose={() => setDocumentsDatabaseOnlyDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning sx={{ mr: 1, color: 'warning.main' }} />
          Clear Document Database Only (Re-sync from Disk)
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            This will remove all document <strong>records and indexes</strong> but <strong>leave files on disk</strong>.
            Use this when the database is out of sync with disk.
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, mb: 2, pl: 2 }}>
            <li>Remove all document records from PostgreSQL</li>
            <li>Clear embeddings and knowledge graph data</li>
            <li><strong>Do not delete any files</strong> from the upload directory</li>
            <li>Run a rescan so files on disk are re-added (or restart the app)</li>
          </Box>
          <Alert severity="info" sx={{ my: 2 }}>
            After this operation, documents on disk will be re-read and re-indexed. Your files are safe.
          </Alert>
          <DialogContentText>
            Continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDocumentsDatabaseOnlyDialogOpen(false)}
            disabled={clearDocumentsDatabaseOnlyMutation.isLoading}
          >
            Cancel
          </Button>
          <Button
            onClick={() => clearDocumentsDatabaseOnlyMutation.mutate()}
            color="warning"
            variant="contained"
            disabled={clearDocumentsDatabaseOnlyMutation.isLoading}
            startIcon={clearDocumentsDatabaseOnlyMutation.isLoading ? null : <DeleteSweep />}
          >
            {clearDocumentsDatabaseOnlyMutation.isLoading ? 'Clearing...' : 'Clear Database Only'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Qdrant Database Clear Confirmation */}
      <Dialog
        open={qdrantDialogOpen}
        onClose={() => setQdrantDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning sx={{ mr: 1, color: 'warning.main' }} />
          Clear Qdrant Vector Database
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            <strong>This action cannot be undone!</strong>
          </DialogContentText>
          <DialogContentText sx={{ mt: 2 }}>
            You are about to permanently delete all embeddings and search vectors from the Qdrant database. 
            This will:
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, mb: 2, pl: 2 }}>
            <li>Remove all document embeddings</li>
            <li>Clear all search indexes</li>
            <li>Require re-processing of all documents for search functionality</li>
          </Box>
          <DialogContentText>
            Are you sure you want to continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setQdrantDialogOpen(false)} 
            disabled={clearQdrantMutation.isLoading}
          >
            Cancel
          </Button>
          <Button 
            onClick={() => clearQdrantMutation.mutate()} 
            color="warning" 
            variant="contained"
            disabled={clearQdrantMutation.isLoading}
            startIcon={clearQdrantMutation.isLoading ? null : <DeleteSweep />}
          >
            {clearQdrantMutation.isLoading ? 'Clearing...' : 'Clear Database'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Neo4j Database Clear Confirmation */}
      <Dialog
        open={neo4jDialogOpen}
        onClose={() => setNeo4jDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning sx={{ mr: 1, color: 'warning.main' }} />
          Clear Neo4j Knowledge Graph
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            <strong>This action cannot be undone!</strong>
          </DialogContentText>
          <DialogContentText sx={{ mt: 2 }}>
            You are about to permanently delete all entities and relationships from the Neo4j knowledge graph. 
            This will:
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, mb: 2, pl: 2 }}>
            <li>Remove all extracted entities (people, places, organizations, etc.)</li>
            <li>Delete all entity relationships</li>
            <li>Clear the knowledge graph visualization data</li>
            <li>Require re-processing of all documents for entity extraction</li>
          </Box>
          <DialogContentText>
            Are you sure you want to continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setNeo4jDialogOpen(false)} 
            disabled={clearNeo4jMutation.isLoading}
          >
            Cancel
          </Button>
          <Button 
            onClick={() => clearNeo4jMutation.mutate()} 
            color="warning" 
            variant="contained"
            disabled={clearNeo4jMutation.isLoading}
            startIcon={clearNeo4jMutation.isLoading ? null : <DeleteSweep />}
          >
            {clearNeo4jMutation.isLoading ? 'Clearing...' : 'Clear Database'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Face Recognition Data Clear Confirmation */}
      <Dialog
        open={faceIdentitiesDialogOpen}
        onClose={() => setFaceIdentitiesDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center' }}>
          <Warning sx={{ mr: 1, color: 'warning.main' }} />
          Clear Face Recognition Data
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            <strong>This action cannot be undone!</strong>
          </DialogContentText>
          <DialogContentText sx={{ mt: 2 }}>
            You are about to permanently delete all face recognition data. This will:
          </DialogContentText>
          <Box component="ul" sx={{ mt: 1, mb: 2, pl: 2 }}>
            <li>Remove all detected faces and their bounding boxes</li>
            <li>Delete all known face identities and their encodings</li>
            <li>Clear all face tagging data</li>
            <li>Require re-analyzing images to detect and tag faces again</li>
          </Box>
          <DialogContentText>
            This is useful when face encodings have drifted or you want to start fresh with face recognition.
          </DialogContentText>
          <DialogContentText sx={{ mt: 2 }}>
            Are you sure you want to continue?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={() => setFaceIdentitiesDialogOpen(false)} 
            disabled={clearFaceIdentitiesMutation.isLoading}
          >
            Cancel
          </Button>
          <Button 
            onClick={() => clearFaceIdentitiesMutation.mutate()} 
            color="warning" 
            variant="contained"
            disabled={clearFaceIdentitiesMutation.isLoading}
            startIcon={clearFaceIdentitiesMutation.isLoading ? null : <DeleteSweep />}
          >
            {clearFaceIdentitiesMutation.isLoading ? 'Clearing...' : 'Clear Face Data'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
        </Box>
      </Box>
    </Box>
  );
};

export default SettingsPage;
