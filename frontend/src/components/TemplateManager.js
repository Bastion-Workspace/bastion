import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Alert,
  Tabs,
  Tab,
  Fab,
  Tooltip,
  Paper,
  Divider,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  CircularProgress,
  Snackbar
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  FileCopy as CopyIcon,
  Description as DescriptionIcon,
  ExpandMore as ExpandMoreIcon,
  DragIndicator as DragIcon,
  Preview as PreviewIcon,
  Public as PublicIcon,
  Lock as PrivateIcon,
  Build as SystemIcon
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';

const TemplateManager = () => {
  const queryClient = useQueryClient();
  const [currentTab, setCurrentTab] = useState(0);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [builderOpen, setBuilderOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [templateToDelete, setTemplateToDelete] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

  // Fetch user templates
  const { data: userTemplates, isLoading: userTemplatesLoading, refetch: refetchUserTemplates } = useQuery(
    'userTemplates',
    () => apiService.getUserTemplates(),
    {
      onError: (error) => {
        console.error('Failed to fetch user templates:', error);
        setSnackbar({
          open: true,
          message: `Failed to load templates: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Fetch public templates
  const { data: publicTemplates, isLoading: publicTemplatesLoading } = useQuery(
    'publicTemplates',
    () => apiService.getPublicTemplates(),
    {
      onError: (error) => {
        console.error('Failed to fetch public templates:', error);
      }
    }
  );

  // Delete template mutation
  const deleteTemplateMutation = useMutation(
    (templateId) => apiService.deleteTemplate(templateId),
    {
      onSuccess: (data) => {
        refetchUserTemplates();
        setSnackbar({
          open: true,
          message: 'Template deleted successfully!',
          severity: 'success'
        });
        setDeleteDialogOpen(false);
        setTemplateToDelete(null);
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to delete template: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  // Duplicate template mutation
  const duplicateTemplateMutation = useMutation(
    ({ templateId, newName }) => apiService.duplicateTemplate(templateId, { new_name: newName }),
    {
      onSuccess: (data) => {
        refetchUserTemplates();
        setSnackbar({
          open: true,
          message: `Template duplicated as '${data.template.template_name}'!`,
          severity: 'success'
        });
      },
      onError: (error) => {
        setSnackbar({
          open: true,
          message: `Failed to duplicate template: ${error.response?.data?.detail || error.message}`,
          severity: 'error'
        });
      }
    }
  );

  const handleEditTemplate = (template) => {
    setSelectedTemplate(template);
    setBuilderOpen(true);
  };

  const handleDeleteTemplate = (template) => {
    setTemplateToDelete(template);
    setDeleteDialogOpen(true);
  };

  const handleDuplicateTemplate = (template) => {
    const newName = prompt(`Enter name for duplicated template:`, `${template.template_name} (Copy)`);
    if (newName && newName.trim()) {
      duplicateTemplateMutation.mutate({ 
        templateId: template.template_id, 
        newName: newName.trim() 
      });
    }
  };

  const handleCreateNew = () => {
    setSelectedTemplate(null);
    setBuilderOpen(true);
  };

  const getScopeIcon = (scope) => {
    switch (scope) {
      case 'public': return <PublicIcon fontSize="small" />;
      case 'system': return <SystemIcon fontSize="small" />;
      default: return <PrivateIcon fontSize="small" />;
    }
  };

  const getScopeColor = (scope) => {
    switch (scope) {
      case 'public': return 'primary';
      case 'system': return 'secondary';
      default: return 'default';
    }
  };

  const TemplateCard = ({ template, showActions = true }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card 
        sx={{ 
          mb: 2, 
          border: selectedTemplate?.template_id === template.template_id ? '2px solid #1976d2' : '1px solid #e0e0e0',
          cursor: 'pointer',
          '&:hover': { boxShadow: 3 }
        }}
        onClick={() => setSelectedTemplate(template)}
      >
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box flex={1}>
              <Typography variant="h6" gutterBottom>
                {template.template_name}
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                {template.description}
              </Typography>
              
              <Box display="flex" gap={1} mb={2} flexWrap="wrap">
                <Chip 
                  icon={getScopeIcon(template.scope)}
                  label={template.scope}
                  size="small"
                  color={getScopeColor(template.scope)}
                />
                <Chip 
                  label={template.category}
                  size="small"
                  variant="outlined"
                />
                <Chip 
                  label={`${template.sections?.length || 0} sections`}
                  size="small"
                  variant="outlined"
                />
              </Box>

              {template.keywords && template.keywords.length > 0 && (
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Keywords: {template.keywords.join(', ')}
                  </Typography>
                </Box>
              )}
            </Box>

            {showActions && (
              <Box display="flex" flexDirection="column" gap={1}>
                <Tooltip title="Edit Template">
                  <IconButton 
                    size="small" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEditTemplate(template);
                    }}
                    disabled={template.scope === 'system'}
                  >
                    <EditIcon />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="Duplicate Template">
                  <IconButton 
                    size="small" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDuplicateTemplate(template);
                    }}
                  >
                    <CopyIcon />
                  </IconButton>
                </Tooltip>
                
                <Tooltip title="Delete Template">
                  <IconButton 
                    size="small" 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteTemplate(template);
                    }}
                    disabled={template.scope === 'system'}
                    color="error"
                  >
                    <DeleteIcon />
                  </IconButton>
                </Tooltip>
              </Box>
            )}
          </Box>

          {selectedTemplate?.template_id === template.template_id && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              transition={{ duration: 0.3 }}
            >
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Template Sections:
              </Typography>
              <Box>
                {template.sections?.map((section, index) => (
                  <Accordion key={section.section_id} variant="outlined">
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                      <Box display="flex" alignItems="center" gap={2}>
                        <Typography variant="body2">
                          {section.section_name}
                        </Typography>
                        {section.required && (
                          <Chip label="Required" size="small" color="error" />
                        )}
                        <Chip 
                          label={`${section.fields?.length || 0} fields`} 
                          size="small" 
                          variant="outlined" 
                        />
                      </Box>
                    </AccordionSummary>
                    <AccordionDetails>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {section.description}
                      </Typography>
                      {section.fields && section.fields.length > 0 && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Fields:
                          </Typography>
                          <List dense>
                            {section.fields.map((field) => (
                              <ListItem key={field.field_id}>
                                <ListItemText
                                  primary={field.field_name}
                                  secondary={`${field.field_type} - ${field.description}`}
                                />
                                {field.required && (
                                  <Chip label="Required" size="small" color="error" />
                                )}
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}
                    </AccordionDetails>
                  </Accordion>
                ))}
              </Box>
            </motion.div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );

  const EmptyState = ({ title, description, action }) => (
    <Paper sx={{ p: 4, textAlign: 'center' }}>
      <DescriptionIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        {description}
      </Typography>
      {action}
    </Paper>
  );

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h5" gutterBottom>
            Report Templates
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Create and manage structured report templates for research tasks.
          </Typography>
        </Box>
        
        <Tooltip title="Create New Template">
          <Fab 
            color="primary" 
            aria-label="create template"
            onClick={handleCreateNew}
          >
            <AddIcon />
          </Fab>
        </Tooltip>
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={currentTab} onChange={(_, newValue) => setCurrentTab(newValue)}>
          <Tab label="My Templates" />
          <Tab label="Public Templates" />
        </Tabs>
      </Box>

      {/* Content */}
      {currentTab === 0 && (
        <Box>
          {userTemplatesLoading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : userTemplates?.templates?.length > 0 ? (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                {userTemplates.templates.map((template) => (
                  <TemplateCard key={template.template_id} template={template} />
                ))}
              </Grid>
            </Grid>
          ) : (
            <EmptyState
              title="No Templates Yet"
              description="Create your first report template to get started with structured research."
              action={
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={handleCreateNew}
                >
                  Create First Template
                </Button>
              }
            />
          )}
        </Box>
      )}

      {currentTab === 1 && (
        <Box>
          {publicTemplatesLoading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : publicTemplates?.templates?.length > 0 ? (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Alert severity="info" sx={{ mb: 2 }}>
                  These are built-in and community templates. You can duplicate them to create your own versions.
                </Alert>
                {publicTemplates.templates.map((template) => (
                  <TemplateCard 
                    key={template.template_id} 
                    template={template} 
                    showActions={template.scope !== 'system'} 
                  />
                ))}
              </Grid>
            </Grid>
          ) : (
            <EmptyState
              title="No Public Templates"
              description="No public or system templates are currently available."
            />
          )}
        </Box>
      )}

      {/* Template Builder Dialog */}
      <TemplateBuilder
        open={builderOpen}
        onClose={() => {
          setBuilderOpen(false);
          setSelectedTemplate(null);
        }}
        template={selectedTemplate}
        onSave={() => {
          refetchUserTemplates();
          setBuilderOpen(false);
          setSelectedTemplate(null);
        }}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Delete Template</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the template "{templateToDelete?.template_name}"? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>
            Cancel
          </Button>
          <Button 
            onClick={() => deleteTemplateMutation.mutate(templateToDelete?.template_id)}
            color="error"
            variant="contained"
            disabled={deleteTemplateMutation.isLoading}
          >
            {deleteTemplateMutation.isLoading ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Snackbar */}
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

// Placeholder for Template Builder component
const TemplateBuilder = ({ open, onClose, template, onSave }) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      fullScreen
    >
      <DialogTitle>
        {template ? 'Edit Template' : 'Create New Template'}
      </DialogTitle>
      <DialogContent>
        <Alert severity="info" sx={{ mb: 2 }}>
          Template Builder UI coming soon! This will include drag-and-drop section management, 
          field configuration, and real-time preview.
        </Alert>
        <Typography variant="body2" color="text.secondary">
          For now, templates can be managed via the backend API. The full visual template 
          builder will be implemented in the next phase.
        </Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default TemplateManager;

