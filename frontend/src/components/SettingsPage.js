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
  Tabs,
  Tab,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  CircularProgress,
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
  Description as DescriptionIcon,
  ListAlt,
  FolderOpen,
  MusicNote,
  Movie,
  Info
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { useThemeMode } from '../hooks/useThemeMode';
import UserManagement from './UserManagement';
import ClassificationModelSelector from './ClassificationModelSelector';
import ImageGenerationModelSelector from './ImageGenerationModelSelector';
import TextCompletionModelSelector from './TextCompletionModelSelector';
import { useModel } from '../contexts/ModelContext';
import OrgModeSettingsTab from './OrgModeSettingsTab';
import MediaSettingsTab from './music/MediaSettingsTab';
import EntertainmentSyncManager from './EntertainmentSyncManager';

// Model Status Display Component
const ModelStatusDisplay = () => {
  const { data: classificationData, isLoading: loadingClassification } = useQuery(
    'classificationModel',
    () => apiService.get('/api/models/classification')
  );

  const { data: currentModelData, isLoading: loadingCurrent } = useQuery(
    'currentModel',
    () => apiService.get('/api/models/current')
  );

  if (loadingClassification || loadingCurrent) {
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
        <Paper sx={{ p: 2, bgcolor: classificationData?.chat_model_is_fallback ? 'warning.light' : 'success.light' }}>
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
            Used for general AI conversations and responses
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, bgcolor: classificationData?.classification_model_is_fallback ? 'warning.light' : 'success.light' }}>
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
            Used for fast intent classification and routing
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

// Pending Submissions Component for Admin
const PendingSubmissions = () => {
  const [pendingSubmissions, setPendingSubmissions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // Fetch pending submissions
  const { data: submissionsData, isLoading, refetch } = useQuery(
    'pendingSubmissions',
    () => apiService.getPendingSubmissions(),
    {
      refetchInterval: 30000, // Refresh every 30 seconds
      onSuccess: (data) => {
        setPendingSubmissions(data.submissions || []);
        setLoading(false);
      },
      onError: () => {
        setLoading(false);
      }
    }
  );

  // Review submission mutation
  const reviewMutation = useMutation(
    ({ documentId, action, comment }) => apiService.reviewSubmission(documentId, action, comment),
    {
      onSuccess: (data) => {
        refetch();
        setSnackbar({
          open: true,
          message: `Document ${data.action}ed successfully!`,
          severity: 'success'
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to review submission: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      },
    }
  );

  const handleApprove = (documentId, comment = '') => {
    reviewMutation.mutate({ documentId, action: 'approve', comment });
  };

  const handleReject = (documentId, comment = '') => {
    reviewMutation.mutate({ documentId, action: 'reject', comment });
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent sx={{ textAlign: 'center', py: 4 }}>
          <CircularProgress />
          <Typography variant="body1" sx={{ mt: 2 }}>
            Loading pending submissions...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          <Warning sx={{ mr: 1, verticalAlign: 'middle' }} />
          Pending Global Submissions
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          Review user submissions for inclusion in the global knowledge base.
        </Typography>

        {pendingSubmissions.length === 0 ? (
          <Alert severity="info">
            No pending submissions at this time.
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {pendingSubmissions.map((doc) => (
              <Grid item xs={12} key={doc.document_id}>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Grid container spacing={2} alignItems="center">
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6">{doc.title || doc.filename}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Submitted by: {doc.submitted_by}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Submitted: {new Date(doc.submitted_at).toLocaleString()}
                      </Typography>
                      {doc.submission_reason && (
                        <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
                          Reason: {doc.submission_reason}
                        </Typography>
                      )}
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Box display="flex" flexDirection="column" gap={1}>
                        <Chip label={doc.doc_type.toUpperCase()} size="small" />
                        <Chip label={`${(doc.file_size / 1024).toFixed(1)} KB`} size="small" variant="outlined" />
                        {doc.category && <Chip label={doc.category} size="small" variant="outlined" />}
                      </Box>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Box display="flex" flexDirection="column" gap={1}>
                        <Button
                          variant="contained"
                          color="success"
                          size="small"
                          onClick={() => handleApprove(doc.document_id)}
                          disabled={reviewMutation.isLoading}
                        >
                          Approve
                        </Button>
                        <Button
                          variant="outlined"
                          color="error"
                          size="small"
                          onClick={() => handleReject(doc.document_id)}
                          disabled={reviewMutation.isLoading}
                        >
                          Reject
                        </Button>
                      </Box>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>
            ))}
          </Grid>
        )}

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
      </CardContent>
    </Card>
  );
};

const SettingsPage = () => {
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const { darkMode, toggleDarkMode } = useTheme();
  const { systemPrefersDark, syncWithSystem, isSystemTheme } = useThemeMode();
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
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // User profile state
  const [userTimezone, setUserTimezone] = useState('UTC');
  const [timezoneLoading, setTimezoneLoading] = useState(false);
  const [userZipCode, setUserZipCode] = useState('');
  const [userTimeFormat, setUserTimeFormat] = useState('24h');
  const [userPreferredName, setUserPreferredName] = useState('');
  const [userAiContext, setUserAiContext] = useState('');


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

  // Stock persona state
  const [stockPersonaMode, setStockPersonaMode] = useState('custom'); // 'custom' or specific persona
  const [customSettings, setCustomSettings] = useState({
    ai_name: 'Alex',
    political_bias: 'neutral',
    persona_style: 'professional'
  });

  // Check for users tab from URL params
  React.useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('tab') === 'users' && user?.role === 'admin') {
      setCurrentTab(8); // Database tab is at 7, User Management is at 8
    }
  }, [user]);

  // Fetch enabled models from backend
  const { data: enabledModelsData, isLoading: enabledModelsLoading } = useQuery(
    'enabledModels',
    () => apiService.getEnabledModels(),
    {
      onSuccess: (data) => {
        if (data?.enabled_models) {
          setEnabledModels(new Set(data.enabled_models));
        }
      }
    }
  );

  // Fetch available models - only after enabled models are loaded
  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useQuery(
    'availableModels',
    () => apiService.getAvailableModels(),
    {
      enabled: !enabledModelsLoading, // Wait for enabled models to load first
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

  // Fetch system settings
  const { data: systemSettings, isLoading: systemSettingsLoading } = useQuery(
    'systemSettings',
    () => apiService.getSettings(),
    {
      onSuccess: (data) => {
        console.log('System settings loaded:', data);
      },
      onError: (error) => {
        console.error('Failed to load system settings:', error);
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
    { label: 'User Profile', icon: <Person /> },
    { label: 'AI Personality', icon: <Psychology /> },
    { label: 'Models', icon: <Settings /> },
    { label: 'News', icon: <DescriptionIcon /> },
    { label: 'Org-Mode', icon: <ListAlt /> },
    { label: 'Media', icon: <MusicNote /> },
    { label: 'Entertainment', icon: <Movie /> },
    ...(user?.role === 'admin' ? [
      { label: 'Database', icon: <DeleteSweep /> },
      { label: 'User Management', icon: <Security /> },
      { label: 'Pending Submissions', icon: <Warning /> }
    ] : [])
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure your Bastion Workspace settings and manage users.
      </Typography>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs 
          value={currentTab} 
          onChange={(_, newValue) => setCurrentTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          allowScrollButtonsMobile
          sx={{
            '& .MuiTabs-flexContainer': {
              flexWrap: { xs: 'nowrap', sm: 'nowrap', md: 'wrap' },
            },
            '& .MuiTab-root': {
              minWidth: { xs: 'auto', sm: 120 },
              fontSize: { xs: '0.75rem', sm: '0.875rem' },
              px: { xs: 1, sm: 2 },
            }
          }}
        >
          {tabs.map((tab, index) => (
            <Tab 
              key={index}
              label={tab.label} 
              icon={tab.icon} 
              iconPosition="start"
            />
          ))}
        </Tabs>
      </Box>

      {/* Tab Content */}
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

                  <Alert severity="info" sx={{ mb: 3 }}>
                    <strong>Personal Settings:</strong> Configure your personal preferences including timezone for accurate time displays.
                  </Alert>

                  {/* Preferred Name Setting */}
                  <Box mb={3}>
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
                  </Box>

                  {/* AI Context Setting */}
                  <Box mb={3}>
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
                    
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Provide context to help AI agents tailor their responses to your needs.
                    </Typography>

                    <Alert severity="warning" sx={{ mb: 2 }}>
                      <strong>Privacy Notice:</strong> This information will be included in all agent conversations 
                      and may be transmitted to external AI providers (OpenAI, Anthropic, etc.) as part of system prompts.
                    </Alert>
                    
                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      label="Tell your AI about yourself"
                      value={userAiContext}
                      onChange={(e) => {
                        if (e.target.value.length <= 500) {
                          setUserAiContext(e.target.value);
                        }
                      }}
                      inputProps={{ maxLength: 500 }}
                      sx={{ mb: 1 }}
                      disabled={aiContextMutation.isLoading}
                      helperText={`${userAiContext.length}/500 characters`}
                    />

                    <Button
                      variant="contained"
                      onClick={() => aiContextMutation.mutate(userAiContext)}
                      disabled={aiContextMutation.isLoading}
                      startIcon={aiContextMutation.isLoading ? <CircularProgress size={20} /> : <Settings />}
                    >
                      {aiContextMutation.isLoading ? 'Updating...' : 'Update AI Context'}
                    </Button>
                  </Box>

                  {/* Timezone Setting */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Timezone
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Set your timezone to ensure accurate time displays when asking questions like "what time is it?"
                    </Typography>
                    
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

                  {/* Time Format Setting */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Time Format
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Choose how time is displayed in the status bar: 12-hour format (AM/PM) or 24-hour format (Military).
                    </Typography>
                    
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
                  </Box>

                  <Divider sx={{ my: 3 }} />

                  {/* User Info Display */}
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Account Information
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Username:</strong> {user?.username}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Email:</strong> {user?.email}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Role:</strong> {user?.role}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      )}

      {currentTab === 1 && (
        <Grid container spacing={3}>
          {/* AI Personality Settings */}
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <Psychology sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">AI Personality Settings</Typography>
                  </Box>

                  <Alert severity="info" sx={{ mb: 3 }}>
                    <strong>Customize Your AI:</strong> Configure the personality, political bias, and communication style of your AI assistant. 
                    Note: You must change the AI name from "Alex" when using non-default bias or persona settings.
                  </Alert>

                  {/* Stock Persona Selection */}
                  <Box mb={3}>
                    <Typography variant="h6" gutterBottom>
                      Stock Personas
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      Choose a pre-configured historical figure persona, or select "Custom" to configure individual settings.
                    </Typography>
                    
                    <FormControl fullWidth sx={{ mb: 2 }}>
                      <InputLabel>Stock Persona</InputLabel>
                      <Select
                        value={stockPersonaMode}
                        onChange={(e) => handleStockPersonaChange(e.target.value)}
                        label="Stock Persona"
                      >
                        <MenuItem value="custom">
                          <strong>Custom</strong> - Configure individual settings
                        </MenuItem>
                        <Divider />
                        {promptOptions.historical_figures.map((figure) => (
                          <MenuItem key={figure.value} value={figure.value}>
                            <strong>{figure.label}</strong> - {getStockPersonaDescription(figure.value)}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>

                  {/* Custom Settings (only shown when Custom is selected) */}
                  {stockPersonaMode === 'custom' && (
                    <>
                      {/* AI Name */}
                      <Box mb={3}>
                        <Typography variant="h6" gutterBottom>
                          AI Assistant Name
                        </Typography>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          Choose a name for your AI assistant. "Alex" is always neutral and professional.
                        </Typography>
                        
                        <TextField
                          fullWidth
                          label="AI Name"
                          value={promptSettings.ai_name}
                          onChange={(e) => handleCustomSettingChange('ai_name', e.target.value)}
                          sx={{ mb: 2 }}
                        />
                      </Box>

                      {/* Political Bias */}
                      <Box mb={3}>
                        <Typography variant="h6" gutterBottom>
                          Political Bias
                        </Typography>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          Choose the political perspective that will influence your AI's analysis and responses. Extreme biases may twist facts to suit their worldview.
                        </Typography>
                        
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Political Bias</InputLabel>
                          <Select
                            value={promptSettings.political_bias}
                            onChange={(e) => handleCustomSettingChange('political_bias', e.target.value)}
                            label="Political Bias"
                          >
                            {promptOptions.political_biases.map((bias) => (
                              <MenuItem key={bias.value} value={bias.value}>
                                {bias.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Box>

                      {/* Persona Style */}
                      <Box mb={3}>
                        <Typography variant="h6" gutterBottom>
                          Communication Style
                        </Typography>
                        <Typography variant="body2" color="text.secondary" paragraph>
                          Choose how your AI communicates and interacts with you.
                        </Typography>
                        
                        <FormControl fullWidth sx={{ mb: 2 }}>
                          <InputLabel>Persona Style</InputLabel>
                          <Select
                            value={promptSettings.persona_style}
                            onChange={(e) => handleCustomSettingChange('persona_style', e.target.value)}
                            label="Persona Style"
                          >
                            {promptOptions.persona_styles.filter(persona => 
                              !promptOptions.historical_figures.some(figure => figure.value === persona.value)
                            ).map((persona) => (
                              <MenuItem key={persona.value} value={persona.value}>
                                {persona.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Box>
                    </>
                  )}

                  {/* Stock Persona Info (shown when stock persona is selected) */}
                  {stockPersonaMode !== 'custom' && (
                    <Box mb={3}>
                      <Alert severity="info">
                        <Typography variant="h6" gutterBottom>
                          {promptOptions.historical_figures.find(f => f.value === stockPersonaMode)?.label}
                        </Typography>
                        <Typography variant="body2">
                          <strong>AI Name:</strong> {promptSettings.ai_name}<br />
                          <strong>Political Bias:</strong> {promptOptions.political_biases.find(b => b.value === promptSettings.political_bias)?.label}<br />
                          <strong>Communication Style:</strong> {promptOptions.historical_figures.find(f => f.value === stockPersonaMode)?.label}
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 1 }}>
                          {getStockPersonaDescription(stockPersonaMode)}
                        </Typography>
                      </Alert>
                    </Box>
                  )}

                  <Divider sx={{ my: 3 }} />

                  {/* Save Button */}
                  <Button
                    variant="contained"
                    onClick={() => promptSettingsMutation.mutate(promptSettings)}
                    disabled={promptSettingsMutation.isLoading}
                    startIcon={promptSettingsMutation.isLoading ? <CircularProgress size={20} /> : <Psychology />}
                  >
                    {promptSettingsMutation.isLoading ? 'Updating...' : 'Save AI Personality Settings'}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      )}

      {currentTab === 2 && (
        <Grid container spacing={3}>

        {/* Current Model Status - Shows what agents are actually using */}
        <Grid item xs={12}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                <Psychology sx={{ mr: 1 }} />
                Current AI Model Configuration
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                This shows the models currently being used by AI agents. Models marked with "Fallback" are system defaults.
              </Typography>

              {/* We'll add the model status display here */}
              <ModelStatusDisplay />
            </CardContent>
          </Card>
        </Grid>

        {/* Classification Model Selection - Admin Only */}
        {user?.role === 'admin' && (
          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
            >
              <Card sx={{ border: '2px solid #2196f3', borderRadius: 2 }}>
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
              <Card sx={{ border: '2px solid #4caf50', borderRadius: 2 }}>
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
              <Card sx={{ border: '2px solid #00bcd4', borderRadius: 2 }}>
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
                  Only one model can be active at a time. Models are fetched live from OpenRouter API.
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
        )}

        {/* Model Information for Non-Admin Users */}
        {user?.role !== 'admin' && (
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
                    <strong>Note:</strong> Model management is restricted to administrators. 
                    You can select from the enabled models in the chat interface.
                  </Alert>

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

      {/* News Settings Tab */}
      {currentTab === 3 && (
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
                    <DescriptionIcon sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">News Synthesis</Typography>
                  </Box>

                  <Alert severity="info" sx={{ mb: 2 }}>
                    Configure how the News background agent synthesizes balanced articles from your RSS sources.
                  </Alert>

                  <Grid container spacing={2}>
                    <Grid item xs={12} md={6}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Synthesis Model</InputLabel>
                        <Select
                          label="Synthesis Model"
                          value={systemSettings?.settings?.news?.synthesis_model?.value || ''}
                          onChange={async (e) => {
                            try {
                              await apiService.setSetting('news.synthesis_model', { key: 'news.synthesis_model', value: e.target.value, category: 'news', description: 'Model used for news synthesis' });
                              // Refresh settings
                              try { window?.requestIdleCallback?.(() => {}); } catch {}
                              // Using react-query client to invalidate
                              // Note: queryClient is available in scope
                              try { queryClient.invalidateQueries('systemSettings'); } catch {}
                            } catch {}
                          }}
                        >
                          {(modelsData?.models || []).filter(m => enabledModels.has(m.id)).map(m => (
                            <MenuItem key={m.id} value={m.id}>{m.name} ({m.provider})</MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <Typography variant="caption" color="text.secondary">Model used by the news agent to write balanced articles</Typography>
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <TextField
                        fullWidth
                        label="Min Sources"
                        type="number"
                        size="small"
                        value={systemSettings?.settings?.news?.min_sources?.value ?? 3}
                        onChange={async (e) => {
                          const val = parseInt(e.target.value, 10);
                          if (Number.isNaN(val)) return;
                          try {
                            await apiService.setSetting('news.min_sources', { key: 'news.min_sources', value: val, category: 'news', description: 'Minimum sources per cluster' });
                            try { queryClient.invalidateQueries('systemSettings'); } catch {}
                          } catch {}
                        }}
                        helperText="Cluster must include at least this many distinct outlets"
                      />
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <TextField
                        fullWidth
                        label="Recency (min)"
                        type="number"
                        size="small"
                        value={systemSettings?.settings?.news?.recency_minutes?.value ?? 60}
                        onChange={async (e) => {
                          const val = parseInt(e.target.value, 10);
                          if (Number.isNaN(val)) return;
                          try {
                            await apiService.setSetting('news.recency_minutes', { key: 'news.recency_minutes', value: val, category: 'news', description: 'Recency window in minutes' });
                            try { queryClient.invalidateQueries('systemSettings'); } catch {}
                          } catch {}
                        }}
                        helperText="Time window for clustering fresh stories"
                      />
                    </Grid>
                    <Grid item xs={12} md={2}>
                      <TextField
                        fullWidth
                        label="Min Diversity (0-1)"
                        type="number"
                        size="small"
                        inputProps={{ min: 0, max: 1, step: 0.1 }}
                        value={systemSettings?.settings?.news?.min_diversity?.value ?? 0.4}
                        onChange={async (e) => {
                          const val = parseFloat(e.target.value);
                          if (Number.isNaN(val)) return;
                          try {
                            await apiService.setSetting('news.min_diversity', { key: 'news.min_diversity', value: val, category: 'news', description: 'Required diversity score for synthesis' });
                            try { queryClient.invalidateQueries('systemSettings'); } catch {}
                          } catch {}
                        }}
                        helperText="Higher requires more outlet diversity"
                      />
                    </Grid>
                    <Grid item xs={12}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={!!(systemSettings?.settings?.news?.notifications_enabled?.value)}
                            onChange={(e) => apiService.setSetting('news.notifications_enabled', { key: 'news.notifications_enabled', value: e.target.checked, category: 'news', description: 'Enable desktop notifications for breaking/urgent' })}
                          />
                        }
                        label="Enable browser notifications for breaking/urgent headlines"
                      />
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>

          <Grid item xs={12}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 }}
            >
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={3}>
                    <DescriptionIcon sx={{ mr: 2, color: 'primary.main' }} />
                    <Typography variant="h6">RSS Sources for News</Typography>
                  </Box>

                  <Alert severity="info" sx={{ mb: 2 }}>
                    Manage your RSS feeds in the RSS manager; the news agent will use feeds tagged for News. Toggle "Include in News synthesis" on a feed.
                  </Alert>
                </CardContent>
              </Card>
            </motion.div>
          </Grid>
        </Grid>
      )}

      {/* Org-Mode Settings Tab */}
      {currentTab === 4 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <OrgModeSettingsTab />
        </motion.div>
      )}

      {/* Media Settings Tab */}
      {currentTab === 5 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <MediaSettingsTab />
        </motion.div>
      )}

      {/* Entertainment Sync Tab */}
      {currentTab === 6 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <EntertainmentSyncManager />
        </motion.div>
      )}

      {/* Database Management Tab */}
      {currentTab === 7 && user?.role === 'admin' && (
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
                        >
                          {clearDocumentsMutation.isLoading ? 'Clearing...' : 'Delete All Documents'}
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
                </CardContent>
              </Card>
        </motion.div>
          </Grid>
        </Grid>
      )}

      {/* User Management Tab */}
      {currentTab === 8 && user?.role === 'admin' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
        >
          <UserManagement />
        </motion.div>
      )}

      {/* Pending Submissions Tab */}
      {currentTab === 9 && user?.role === 'admin' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
            >
          <PendingSubmissions />
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
  );
};

export default SettingsPage;
