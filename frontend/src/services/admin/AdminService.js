import ApiServiceBase from '../base/ApiServiceBase';

class AdminService extends ApiServiceBase {
  // System admin methods
  clearQdrantDatabase = async () => {
    return this.post('/api/admin/clear-qdrant');
  }

  clearNeo4jDatabase = async () => {
    return this.post('/api/admin/clear-neo4j');
  }

  clearAllDocuments = async () => {
    return this.post('/api/admin/clear-documents');
  }

  // User management methods
  getUsers = async () => {
    return this.get('/api/admin/users');
  }

  createUser = async (userData) => {
    return this.post('/api/admin/users', userData);
  }

  updateUser = async (userId, userData) => {
    return this.put(`/api/admin/users/${userId}`, userData);
  }

  deleteUser = async (userId) => {
    return this.delete(`/api/admin/users/${userId}`);
  }

  changePassword = async (userId, passwordData) => {
    return this.post(`/api/admin/users/${userId}/change-password`, passwordData);
  }

}

export default new AdminService();
