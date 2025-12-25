import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || '';

class DataWorkspaceService {
  constructor() {
    this.api = axios.create({
      baseURL: `${API_BASE_URL}/api/data`,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.api.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  // Workspace methods
  async createWorkspace(data) {
    const response = await this.api.post('/workspaces', data);
    return response.data;
  }

  async listWorkspaces() {
    const response = await this.api.get('/workspaces');
    return response.data;
  }

  async getWorkspace(workspaceId) {
    const response = await this.api.get(`/workspaces/${workspaceId}`);
    return response.data;
  }

  async updateWorkspace(workspaceId, data) {
    const response = await this.api.put(`/workspaces/${workspaceId}`, data);
    return response.data;
  }

  async deleteWorkspace(workspaceId) {
    const response = await this.api.delete(`/workspaces/${workspaceId}`);
    return response.data;
  }

  // Database methods
  async createDatabase(data) {
    const response = await this.api.post('/databases', data);
    return response.data;
  }

  async listDatabases(workspaceId) {
    const response = await this.api.get(`/workspaces/${workspaceId}/databases`);
    return response.data;
  }

  async getDatabase(databaseId) {
    const response = await this.api.get(`/databases/${databaseId}`);
    return response.data;
  }

  async deleteDatabase(databaseId) {
    const response = await this.api.delete(`/databases/${databaseId}`);
    return response.data;
  }

  // Import methods
  async uploadFile(workspaceId, file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.api.post(`/import/upload?workspace_id=${workspaceId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async previewImport(data) {
    const response = await this.api.post('/import/preview', data);
    return response.data;
  }

  async executeImport(data) {
    const response = await this.api.post('/import/execute', data);
    return response.data;
  }

  async getImportStatus(jobId) {
    const response = await this.api.get(`/import/jobs/${jobId}`);
    return response.data;
  }

  // Table methods
  async createTable(data) {
    const response = await this.api.post('/tables', data);
    return response.data;
  }

  async listTables(databaseId) {
    const response = await this.api.get(`/databases/${databaseId}/tables`);
    return response.data;
  }

  async getTable(tableId) {
    const response = await this.api.get(`/tables/${tableId}`);
    return response.data;
  }

  async deleteTable(tableId) {
    const response = await this.api.delete(`/tables/${tableId}`);
    return response.data;
  }

  async getTableData(tableId, offset = 0, limit = 100) {
    const response = await this.api.get(`/tables/${tableId}/data`, {
      params: { offset, limit },
    });
    return response.data;
  }

  async insertTableRow(tableId, rowData) {
    const response = await this.api.post(`/tables/${tableId}/rows`, { row_data: rowData });
    return response.data;
  }

  async updateTableRow(tableId, rowId, rowData) {
    const response = await this.api.put(`/tables/${tableId}/rows/${rowId}`, { row_data: rowData });
    return response.data;
  }

  async updateTableCell(tableId, rowId, columnName, value, formula = null) {
    const response = await this.api.patch(`/tables/${tableId}/rows/${rowId}/cells`, {
      column_name: columnName,
      value: value,
      formula: formula
    });
    return response.data;
  }

  async recalculateTable(tableId) {
    const response = await this.api.post(`/tables/${tableId}/recalculate`);
    return response.data;
  }

  async deleteTableRow(tableId, rowId) {
    const response = await this.api.delete(`/tables/${tableId}/rows/${rowId}`);
    return response.data;
  }

  // Sharing methods
  async shareWorkspace(workspaceId, shareData) {
    const response = await this.api.post(`/workspaces/${workspaceId}/share`, shareData);
    return response.data;
  }

  async listWorkspaceShares(workspaceId) {
    const response = await this.api.get(`/workspaces/${workspaceId}/shares`);
    return response.data;
  }

  async revokeShare(workspaceId, shareId) {
    const response = await this.api.delete(`/workspaces/${workspaceId}/shares/${shareId}`);
    return response.data;
  }

  async listSharedWorkspaces() {
    const response = await this.api.get('/workspaces/shared');
    return response.data;
  }
}

export default new DataWorkspaceService();

