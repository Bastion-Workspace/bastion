import ApiServiceBase from './base/ApiServiceBase';

const AGENT_FACTORY_PREFIX = '/api/agent-factory';
const BROWSER_AUTH_PREFIX = '/api/browser-auth';

class AgentFactoryService extends ApiServiceBase {
  getActions = (profileId = null) =>
    profileId
      ? this.get(`${AGENT_FACTORY_PREFIX}/actions?profile_id=${encodeURIComponent(profileId)}`)
      : this.get(`${AGENT_FACTORY_PREFIX}/actions`);
  getToolPacks = () => this.get(`${AGENT_FACTORY_PREFIX}/tool-packs`);

  listProfiles = () => this.get(`${AGENT_FACTORY_PREFIX}/profiles`);
  fetchAgentHandles = () => this.get(`${AGENT_FACTORY_PREFIX}/handles`);
  createProfile = (body) => this.post(`${AGENT_FACTORY_PREFIX}/profiles`, body);
  getProfile = (profileId) => this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}`);
  updateProfile = (profileId, body) => this.put(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}`, body);
  deleteProfile = (profileId) => this.delete(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}`);
  pauseProfile = (profileId) => this.post(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/pause`);
  resumeProfile = (profileId) => this.post(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/resume`);
  getProfileBudget = (profileId) => this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/budget`);
  updateProfileBudget = (profileId, body) => this.put(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/budget`, body);
  getProfileMemory = (profileId) => this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/memory`);
  clearProfileMemory = (profileId) => this.delete(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/memory`);
  resetProfileDefaults = (profileId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/reset-defaults`);
  exportAgentBundle = (profileId) =>
    this.getRaw(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/export-bundle`);
  importAgentBundle = (yamlString) =>
    this.post(`${AGENT_FACTORY_PREFIX}/profiles/import-bundle`, { yaml: yamlString });
  listProfileExecutions = (profileId, limit = 20) =>
    this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/executions?limit=${limit}`);
  getExecution = (profileId, executionId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/executions/${executionId}`);
  deleteExecution = (profileId, executionId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/executions/${executionId}`);
  clearExecutions = (profileId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/executions`);

  dashboardFleetStatus = () => this.get(`${AGENT_FACTORY_PREFIX}/dashboard/fleet-status`);
  dashboardCostSummary = (period = 'month') =>
    this.get(`${AGENT_FACTORY_PREFIX}/dashboard/cost-summary?period=${encodeURIComponent(period)}`);
  dashboardActivityFeed = (limit = 50) =>
    this.get(`${AGENT_FACTORY_PREFIX}/dashboard/activity-feed?limit=${limit}`);

  listPendingApprovals = () => this.get(`${AGENT_FACTORY_PREFIX}/approvals/pending`);
  respondApproval = (approvalId, approved) =>
    this.post(`${AGENT_FACTORY_PREFIX}/approvals/${approvalId}/respond`, { approved });

  listPlaybooks = () => this.get(`${AGENT_FACTORY_PREFIX}/playbooks`);
  createPlaybook = (body) => this.post(`${AGENT_FACTORY_PREFIX}/playbooks`, body);
  getPlaybook = (playbookId) => this.get(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}`);
  updatePlaybook = (playbookId, body) => this.put(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}`, body);
  deletePlaybook = (playbookId) => this.delete(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}`);
  getPlaybookUsage = (playbookId) => this.get(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}/usage`);
  listPlaybookVersions = (playbookId) => this.get(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}/versions`);
  getPlaybookVersion = (playbookId, versionId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}/versions/${versionId}`);
  restorePlaybookVersion = (playbookId, versionId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}/restore/${versionId}`);
  clonePlaybook = (playbookId) => this.post(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}/clone`);
  exportPlaybook = (playbookId) => this.get(`${AGENT_FACTORY_PREFIX}/playbooks/${playbookId}/export`);
  importPlaybook = (body) => this.post(`${AGENT_FACTORY_PREFIX}/playbooks/import`, body);

  getConnectorTemplates = () => this.get(`${AGENT_FACTORY_PREFIX}/connector-templates`);
  listConnectors = () => this.get(`${AGENT_FACTORY_PREFIX}/connectors`);
  exportConnectors = (ids = null) => {
    const url = ids && ids.length
      ? `${AGENT_FACTORY_PREFIX}/connectors/export?ids=${encodeURIComponent(ids.join(','))}`
      : `${AGENT_FACTORY_PREFIX}/connectors/export`;
    return this.getRaw(url);
  };
  importConnectors = (yamlString) =>
    this.post(`${AGENT_FACTORY_PREFIX}/connectors/import`, { yaml: yamlString });
  getConnector = (connectorId) => this.get(`${AGENT_FACTORY_PREFIX}/connectors/${connectorId}`);
  getConnectorUsage = (connectorId) => this.get(`${AGENT_FACTORY_PREFIX}/connectors/${connectorId}/usage`);
  createConnector = (body) => this.post(`${AGENT_FACTORY_PREFIX}/connectors`, body);
  createConnectorFromTemplate = (body) => this.post(`${AGENT_FACTORY_PREFIX}/connectors/from-template`, body);
  updateConnector = (connectorId, body) => this.put(`${AGENT_FACTORY_PREFIX}/connectors/${connectorId}`, body);
  deleteConnector = (connectorId, force = false) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/connectors/${connectorId}${force ? '?force=true' : ''}`);
  testConnector = (connectorId, body) => this.post(`${AGENT_FACTORY_PREFIX}/connectors/${connectorId}/test`, body);
  listDataSources = (profileId) => this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/data-sources`);
  createDataSourceFromTemplate = (profileId, body) =>
    this.post(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/data-sources-from-template`, body);
  updateDataSource = (profileId, sourceId, body) =>
    this.put(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/data-sources/${sourceId}`, body);
  deleteDataSource = (profileId, sourceId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/data-sources/${sourceId}`);
  executeConnector = (body) => this.post(`${AGENT_FACTORY_PREFIX}/execute-connector`, body);

  /** @deprecated Prefer playbook step external tool packs; kept for migration and legacy UIs. */
  listServiceBindings = (profileId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/service-bindings`);
  /** @deprecated Prefer playbook step external tool packs. */
  createServiceBinding = (profileId, body) =>
    this.post(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/service-bindings`, body);
  /** @deprecated Prefer playbook step external tool packs. */
  deleteServiceBinding = (profileId, bindingId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/service-bindings/${bindingId}`);
  getAvailableEmailConnections = () =>
    this.get(`${AGENT_FACTORY_PREFIX}/available-email-connections`);

  listMcpServers = () => this.get('/api/mcp-servers');
  createMcpServer = (body) => this.post('/api/mcp-servers', body);
  updateMcpServer = (serverId, body) => this.put(`/api/mcp-servers/${serverId}`, body);
  deleteMcpServer = (serverId) => this.delete(`/api/mcp-servers/${serverId}`);
  discoverMcpServer = (serverId) => this.post(`/api/mcp-servers/${serverId}/discover`, {});
  testMcpServer = (serverId) => this.post(`/api/mcp-servers/${serverId}/test`, {});

  getPlugins = () => this.get(`${AGENT_FACTORY_PREFIX}/plugins`);
  listPluginConfigs = (profileId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/plugin-configs`);
  upsertPluginConfigs = (profileId, body) =>
    this.put(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/plugin-configs`, body);

  listSchedules = (profileId) => this.get(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/schedules`);
  createSchedule = (profileId, body) => this.post(`${AGENT_FACTORY_PREFIX}/profiles/${profileId}/schedules`, body);
  getSchedule = (scheduleId) => this.get(`${AGENT_FACTORY_PREFIX}/schedules/${scheduleId}`);
  updateSchedule = (scheduleId, body) => this.put(`${AGENT_FACTORY_PREFIX}/schedules/${scheduleId}`, body);
  deleteSchedule = (scheduleId) => this.delete(`${AGENT_FACTORY_PREFIX}/schedules/${scheduleId}`);
  pauseSchedule = (scheduleId) => this.post(`${AGENT_FACTORY_PREFIX}/schedules/${scheduleId}/pause`);
  resumeSchedule = (scheduleId) => this.post(`${AGENT_FACTORY_PREFIX}/schedules/${scheduleId}/resume`);

  listSkills = (params = {}) => {
    const q = new URLSearchParams();
    if (params.category != null) q.set('category', params.category);
    if (params.include_builtin != null) q.set('include_builtin', String(params.include_builtin));
    const query = q.toString();
    return this.get(`${AGENT_FACTORY_PREFIX}/skills${query ? `?${query}` : ''}`);
  };
  getSkill = (skillId) => this.get(`${AGENT_FACTORY_PREFIX}/skills/${skillId}`);
  createSkill = (body) => this.post(`${AGENT_FACTORY_PREFIX}/skills`, body);
  updateSkill = (skillId, body) => this.put(`${AGENT_FACTORY_PREFIX}/skills/${skillId}`, body);
  deleteSkill = (skillId) => this.delete(`${AGENT_FACTORY_PREFIX}/skills/${skillId}`);
  listSkillVersions = (skillId) => this.get(`${AGENT_FACTORY_PREFIX}/skills/${skillId}/versions`);
  revertSkill = (skillId, versionId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/skills/${skillId}/revert/${versionId}`);

  listSidebarCategories = (section = null) => {
    const url = section
      ? `${AGENT_FACTORY_PREFIX}/sidebar-categories?section=${encodeURIComponent(section)}`
      : `${AGENT_FACTORY_PREFIX}/sidebar-categories`;
    return this.get(url);
  };
  createSidebarCategory = (body) =>
    this.post(`${AGENT_FACTORY_PREFIX}/sidebar-categories`, body);
  updateSidebarCategory = (categoryId, body) =>
    this.patch(`${AGENT_FACTORY_PREFIX}/sidebar-categories/${categoryId}`, body);
  deleteSidebarCategory = (categoryId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/sidebar-categories/${categoryId}`);

  // Agent Lines (autonomous line container and org chart)
  listLines = () => this.get(`${AGENT_FACTORY_PREFIX}/lines`);
  createLine = (body) => this.post(`${AGENT_FACTORY_PREFIX}/lines`, body);
  getLine = (lineId) => this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}`);
  updateLine = (lineId, body) => this.put(`${AGENT_FACTORY_PREFIX}/lines/${lineId}`, body);
  deleteLine = (lineId) => this.delete(`${AGENT_FACTORY_PREFIX}/lines/${lineId}`);
  addLineMember = (lineId, body) => this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/members`, body);
  updateLineMember = (lineId, membershipId, body) =>
    this.put(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/members/${membershipId}`, body);
  removeLineMember = (lineId, membershipId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/members/${membershipId}`);
  removeLineMemberByAgent = (lineId, agentProfileId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/members/by-agent/${agentProfileId}`);
  getLineOrgChart = (lineId) => this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/org-chart`);
  getLineBudgetSummary = (lineId) => this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/budget-summary`);
  getLineTimeline = (lineId, params = {}) => {
    const q = new URLSearchParams();
    if (params.limit != null) q.set('limit', params.limit);
    if (params.offset != null) q.set('offset', params.offset);
    if (params.message_type != null) q.set('message_type', params.message_type);
    if (params.agent != null) q.set('agent', params.agent);
    if (params.since != null) q.set('since', params.since);
    const query = q.toString();
    return this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/timeline${query ? `?${query}` : ''}`);
  };
  getLineTimelineSummary = (lineId) => this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/timeline/summary`);
  getLineMessageThread = (lineId, messageId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/messages/${messageId}/thread`);
  postLineMessage = (lineId, body) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/timeline/message`, body);
  clearLineTimeline = (lineId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/timeline/clear`);
  getLineWorkspace = (lineId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/workspace`);
  getLineWorkspaceEntry = (lineId, key) =>
    this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/workspace/${encodeURIComponent(key)}`);
  resetLine = (lineId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/reset`);

  getLineGoals = (lineId) => this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/goals`);
  createGoal = (lineId, body) => this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/goals`, body);
  updateGoal = (lineId, goalId, body) =>
    this.put(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/goals/${goalId}`, body);
  deleteGoal = (lineId, goalId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/goals/${goalId}`);

  listLineTasks = (lineId, params = {}) => {
    const q = new URLSearchParams();
    if (params.status != null) q.set('status', params.status);
    if (params.agent_id != null) q.set('agent_id', params.agent_id);
    const query = q.toString();
    return this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks${query ? `?${query}` : ''}`);
  };
  getTask = (lineId, taskId) => this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks/${taskId}`);
  createTask = (lineId, body) => this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks`, body);
  updateTask = (lineId, taskId, body) =>
    this.put(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks/${taskId}`, body);
  assignTask = (lineId, taskId, agentProfileId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks/${taskId}/assign?agent_profile_id=${encodeURIComponent(agentProfileId)}`);
  transitionTask = (lineId, taskId, newStatus) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks/${taskId}/transition?new_status=${encodeURIComponent(newStatus)}`);
  deleteTask = (lineId, taskId) =>
    this.delete(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/tasks/${taskId}`);

  triggerHeartbeat = (lineId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/heartbeat`);
  /** Pause line, disable autonomous scheduling, revoke in-flight heartbeat (same as legacy emergency-stop). */
  stopAutonomous = (lineId) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/emergency-stop`);
  emergencyStop = (lineId) => this.stopAutonomous(lineId);
  invokeLineAgent = (lineId, body) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/invoke-agent`, body);
  startLineDiscussion = (lineId, body) =>
    this.post(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/start-discussion`, body);
  getLineApprovals = (lineId) =>
    this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/approvals`);
  getLineAgentHealth = (lineId, days = 7) =>
    this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/agent-health?days=${days}`);
  getLineAnalytics = (lineId, days = 30) =>
    this.get(`${AGENT_FACTORY_PREFIX}/lines/${lineId}/analytics?days=${days}`);

  // Browser auth (interactive login capture for playbooks)
  browserAuthInteract = (body) => this.post(`${BROWSER_AUTH_PREFIX}/interact`, body);
  browserAuthScreenshot = (sessionId) =>
    this.get(`${BROWSER_AUTH_PREFIX}/${encodeURIComponent(sessionId)}/screenshot`);
  browserAuthCapture = (sessionId, body) =>
    this.post(`${BROWSER_AUTH_PREFIX}/${encodeURIComponent(sessionId)}/capture`, body);
  browserAuthListSessions = () => this.get(`${BROWSER_AUTH_PREFIX}/sessions`);
  browserAuthDeleteSession = (domain) =>
    this.delete(`${BROWSER_AUTH_PREFIX}/sessions/${encodeURIComponent(domain)}`);
  browserAuthStartSession = (body) => this.post(`${BROWSER_AUTH_PREFIX}/start-session`, body);
  browserAuthCloseSession = (sessionId) =>
    this.post(`${BROWSER_AUTH_PREFIX}/${encodeURIComponent(sessionId)}/close`);
}

export default new AgentFactoryService();
