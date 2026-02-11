import React, { createContext, useContext, useEffect, useState } from 'react';
import apiService from '../services/apiService';
import { useAuth } from './AuthContext';

const CapabilitiesContext = createContext({});

export const useCapabilities = () => useContext(CapabilitiesContext);

export const CapabilitiesProvider = ({ children }) => {
  const { user } = useAuth();
  const [caps, setCaps] = useState({});
  const isAdmin = (user?.role || '').toLowerCase() === 'admin';

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        if (!user) { setCaps({}); return; }
        if (isAdmin) { setCaps({ admin: true }); return; }
        const res = await apiService.get(`/api/admin/users/${user.user_id}/capabilities`);
        if (!mounted) return;
        setCaps(res?.capabilities || {});
      } catch (error) {
        console.error('Failed to load capabilities:', error);
        if (mounted) setCaps({});
      }
    };
    load();
    return () => { mounted = false; };
  }, [user, isAdmin]);

  const value = {
    isAdmin,
    has: (key) => isAdmin ? true : !!caps[key],
    raw: caps,
  };

  return (
    <CapabilitiesContext.Provider value={value}>
      {children}
    </CapabilitiesContext.Provider>
  );
};


