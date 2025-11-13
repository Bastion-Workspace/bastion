// Base API Service for frontend-backend communication
const API_BASE = process.env.REACT_APP_API_URL || '';

class ApiServiceBase {
  constructor() {
    this.baseURL = API_BASE;
  }

  async request(url, options = {}) {
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
