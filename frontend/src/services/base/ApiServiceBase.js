// Base API Service for frontend-backend communication
const API_BASE = process.env.REACT_APP_API_URL || '';

class ApiServiceBase {
  constructor() {
    this.baseURL = API_BASE;
    this.refreshingToken = false;
    this.refreshPromise = null;
  }

  // Decode JWT token to check expiration
  decodeToken(token) {
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(
        atob(base64)
          .split('')
          .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
          .join('')
      );
      return JSON.parse(jsonPayload);
    } catch (e) {
      return null;
    }
  }

  // Check if token is expired or will expire soon (within 5 minutes)
  isTokenExpiringSoon(token) {
    if (!token) return true;
    
    const decoded = this.decodeToken(token);
    if (!decoded || !decoded.exp) return true;
    
    const expirationTime = decoded.exp * 1000; // Convert to milliseconds
    const currentTime = Date.now();
    const timeUntilExpiry = expirationTime - currentTime;
    const fiveMinutes = 5 * 60 * 1000; // 5 minutes in milliseconds
    
    return timeUntilExpiry < fiveMinutes;
  }

  // Refresh token if needed
  async refreshTokenIfNeeded() {
    // Prevent multiple simultaneous refresh attempts
    if (this.refreshingToken && this.refreshPromise) {
      return this.refreshPromise;
    }

    const token = localStorage.getItem('auth_token');
    if (!token) {
      return false;
    }

    // Check if token needs refresh
    if (!this.isTokenExpiringSoon(token)) {
      return false;
    }

    this.refreshingToken = true;
    this.refreshPromise = (async () => {
      try {
        // Call refresh endpoint directly to avoid circular dependency
        const refreshResponse = await fetch(`${this.baseURL}/api/auth/refresh`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          }
        });

        if (refreshResponse.ok) {
          const response = await refreshResponse.json();
          if (response && response.access_token) {
            localStorage.setItem('auth_token', response.access_token);
            console.log('✅ Token refreshed successfully');
            return true;
          }
        }
        
        console.warn('⚠️ Token refresh failed');
        return false;
      } catch (error) {
        console.error('❌ Token refresh failed:', error);
        return false;
      } finally {
        this.refreshingToken = false;
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  async request(url, options = {}) {
    // Skip token refresh for auth endpoints
    const isAuthEndpoint = url.includes('/api/auth/login') || 
                          url.includes('/api/auth/refresh') ||
                          url.includes('/api/auth/logout');
    
    // Refresh token if needed (before making request)
    if (!isAuthEndpoint) {
      await this.refreshTokenIfNeeded();
    }

    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    const fullURL = `${this.baseURL}${url}`;
    
    console.log('🌐 API Request:', {
      url: fullURL,
      method: config.method || 'GET',
      headers: Object.keys(config.headers),
      hasBody: !!config.body
    });

    const response = await fetch(fullURL, config);

    console.log('🌐 API Response:', {
      status: response.status,
      ok: response.ok,
      url: response.url
    });

    // Handle 401 errors by attempting token refresh
    if (response.status === 401 && !isAuthEndpoint) {
      const token = localStorage.getItem('auth_token');
      if (token) {
        console.log('🔄 Received 401, attempting token refresh...');
        
        try {
          // Call refresh endpoint directly to avoid circular dependency
          const refreshResponse = await fetch(`${this.baseURL}/api/auth/refresh`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`
            }
          });
          
          if (refreshResponse.ok) {
            const refreshData = await refreshResponse.json();
            if (refreshData && refreshData.access_token) {
              localStorage.setItem('auth_token', refreshData.access_token);
              console.log('✅ Token refreshed, retrying original request...');
              
              // Retry original request with new token
              config.headers.Authorization = `Bearer ${refreshData.access_token}`;
              const retryResponse = await fetch(fullURL, config);
              
              if (!retryResponse.ok) {
                const error = new Error(`HTTP error! status: ${retryResponse.status}`);
                try {
                  const errorData = await retryResponse.json();
                  error.response = { data: errorData };
                  console.error('🌐 API Error after refresh:', errorData);
                } catch (e) {
                  error.response = { data: { detail: retryResponse.statusText } };
                  console.error('🌐 API Error after refresh (no JSON):', retryResponse.statusText);
                }
                throw error;
              }
              
              const result = await retryResponse.json();
              console.log('🌐 API Success after refresh:', result);
              return result;
            }
          }
          
          // Refresh failed, clear token and throw error
          console.error('❌ Token refresh failed');
          localStorage.removeItem('auth_token');
          throw new Error('Token refresh failed - please log in again');
        } catch (refreshError) {
          console.error('❌ Token refresh error:', refreshError);
          localStorage.removeItem('auth_token');
          throw refreshError;
        }
      }
    }

    if (!response.ok) {
      const error = new Error(`HTTP error! status: ${response.status}`);
      try {
        const errorData = await response.json();
        error.response = { data: errorData };
        console.error('🌐 API Error:', errorData);
      } catch (e) {
        error.response = { data: { detail: response.statusText } };
        console.error('🌐 API Error (no JSON):', response.statusText);
      }
      throw error;
    }

    const result = await response.json();
    console.log('🌐 API Success:', result);
    return result;
  }

  get = async (url, options = {}) => {
    return this.request(url, { method: 'GET', ...options });
  }

  post = async (url, data, options = {}) => {
    return this.request(url, {
      method: 'POST',
      body: JSON.stringify(data),
      ...options,
    });
  }

  put = async (url, data, options = {}) => {
    return this.request(url, {
      method: 'PUT',
      body: JSON.stringify(data),
      ...options,
    });
  }

  delete = async (url, options = {}) => {
    return this.request(url, { method: 'DELETE', ...options });
  }
}

export default ApiServiceBase;
