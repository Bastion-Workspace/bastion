import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import apiService from '../services/apiService';
import { lockAllRegistered } from '../services/encryptionSessionRegistry';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

/** Same as useContext(AuthContext); returns null outside AuthProvider (no throw). For providers that must not crash the tree if context is missing. */
export const useAuthOptional = () => useContext(AuthContext);

export const AuthProvider = ({ children, queryClient }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const userRef = useRef(null);
  userRef.current = user;

  // Initialize auth state from localStorage
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const token = localStorage.getItem('auth_token');
        if (token) {
          // Verify token is still valid
          const currentUser = await apiService.getCurrentUser();
          if (currentUser) {
            setUser(currentUser);
            setIsAuthenticated(true);
          } else {
            // Token is invalid, remove it
            localStorage.removeItem('auth_token');
          }
        }
      } catch (error) {
        console.error('Auth initialization error:', error);
        localStorage.removeItem('auth_token');
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const login = async (username, password) => {
    try {
      const response = await apiService.login(username, password);
      
      // Store token
      localStorage.setItem('auth_token', response.access_token);

      const prevId = userRef.current?.user_id;
      const nextId = response.user?.user_id;
      if (queryClient && prevId && nextId && prevId !== nextId) {
        try {
          queryClient.clear();
        } catch {
          /* ignore */
        }
      }

      // Set user state
      setUser(response.user);
      setIsAuthenticated(true);
      
      return { success: true };
    } catch (error) {
      console.error('Login error:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed' 
      };
    }
  };

  const logout = async () => {
    try {
      // Call logout endpoint
      await apiService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      try {
        await lockAllRegistered();
      } catch {
        /* best-effort */
      }
      try {
        queryClient?.clear();
      } catch {
        /* ignore */
      }
      // Clear local state regardless of API call success
      localStorage.removeItem('auth_token');
      setUser(null);
      setIsAuthenticated(false);
    }
  };

  const updateUser = (updatedUser) => {
    setUser(prevUser => ({ ...prevUser, ...updatedUser }));
  };

  const value = {
    user,
    loading,
    isAuthenticated,
    login,
    logout,
    updateUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 