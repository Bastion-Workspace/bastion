import React, { useLayoutEffect, useRef } from 'react';
import { useQueryClient } from 'react-query';
import { useAuth } from '../contexts/AuthContext';

/**
 * Clears React Query cache on authenticated user id change (defensive; normal flow is logout then login).
 * Logout also clears via AuthProvider when queryClient is passed in.
 */
export default function AuthQueryCacheBoundary() {
  const queryClient = useQueryClient();
  const { user, loading, isAuthenticated } = useAuth();
  const sessionRef = useRef(null);

  useLayoutEffect(() => {
    if (loading) return;
    const uid = user?.user_id ?? null;
    const snapshot = { authed: !!(isAuthenticated && uid), uid };
    const prev = sessionRef.current;
    if (prev != null) {
      const hadUser = prev.authed && prev.uid;
      const hasUser = snapshot.authed && snapshot.uid;
      const switchedUser = hadUser && hasUser && prev.uid !== snapshot.uid;
      if (switchedUser) {
        try {
          queryClient.clear();
        } catch {
          /* ignore */
        }
      }
    }
    sessionRef.current = snapshot;
  }, [loading, isAuthenticated, user?.user_id, queryClient]);

  return null;
}
