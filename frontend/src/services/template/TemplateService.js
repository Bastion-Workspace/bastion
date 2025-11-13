import ApiServiceBase from '../base/ApiServiceBase';

class TemplateService extends ApiServiceBase {
  // ===== TEMPLATE MANAGEMENT API =====
  
  // Get user's templates
  getUserTemplates = async () => {
    return this.get('/api/templates/');
  }

  // Get public/system templates
  getPublicTemplates = async () => {
    return this.get('/api/templates/public');
  }

  // Get specific template by ID
  getTemplate = async (templateId) => {
    return this.get(`/api/templates/${templateId}`);
  }

  // Create new template
  createTemplate = async (templateData) => {
    return this.post('/api/templates/', templateData);
  }

  // Update existing template
  updateTemplate = async (templateId, updates) => {
    return this.put(`/api/templates/${templateId}`, updates);
  }

  // Delete template
  deleteTemplate = async (templateId) => {
    return this.delete(`/api/templates/${templateId}`);
  }

  // Duplicate template
  duplicateTemplate = async (templateId, duplicateData) => {
    return this.post(`/api/templates/${templateId}/duplicate`, duplicateData);
  }

  // Search templates by keywords
  searchTemplates = async (keywords) => {
    return this.get(`/api/templates/search?keywords=${encodeURIComponent(keywords)}`);
  }

  // Get template statistics (admin only)
  getTemplateStats = async () => {
    return this.get('/api/templates/stats/overview');
  }

  // Get available field types
  getAvailableFieldTypes = async () => {
    return this.get('/api/templates/field-types/available');
  }

  // Validate template structure
  validateTemplate = async (templateData) => {
    return this.post('/api/templates/validate', templateData);
  }

  // Template execution methods
  executeTemplateResearch = async (conversationId, templateId, query, customInstructions = null) => {
    return this.post('/api/template-execution/execute', {
      conversation_id: conversationId,
      template_id: templateId,
      query: query,
      custom_instructions: customInstructions
    });
  }

  confirmTemplateUsage = async (conversationId, templateId, action, modifications = null) => {
    return this.post('/api/template-execution/confirm', {
      conversation_id: conversationId,
      template_id: templateId,
      action: action,
      modifications: modifications
    });
  }

  getTemplateExecutionStatus = async (conversationId) => {
    return this.get(`/api/template-execution/status/${conversationId}`);
  }

  generateTemplatePlan = async (templateId, query) => {
    return this.get(`/api/template-execution/plan/${templateId}?query=${encodeURIComponent(query)}`);
  }
}

export default new TemplateService();
