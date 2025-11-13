import ApiServiceBase from '../base/ApiServiceBase';

class AuthService extends ApiServiceBase {
  // Authentication methods
  getToken = () => {
    return localStorage.getItem('auth_token');
  }

  login = async (username, password) => {
    return this.post('/api/auth/login', {
      username,
      password,
    });
  }

  logout = async () => {
    return this.post('/api/auth/logout');
  }

  getCurrentUser = async () => {
    return this.get('/api/auth/me');
  }

  register = async (userData) => {
    return this.post('/api/auth/register', userData);
  }

  changePassword = async (currentPassword, newPassword) => {
    return this.post('/api/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  }

  // Admin change password for any user
  adminChangePassword = async (userId, passwordData) => {
    return this.post(`/api/admin/users/${userId}/change-password`, passwordData);
  }
}

export default new AuthService();
