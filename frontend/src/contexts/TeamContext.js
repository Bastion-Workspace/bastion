import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import { useQueryClient } from 'react-query';
import { useAuth } from './AuthContext';
import teamService from '../services/teams/TeamService';
import { useMessaging } from './MessagingContext';

const TeamContext = createContext();

export const useTeam = () => {
  const context = useContext(TeamContext);
  if (!context) {
    throw new Error('useTeam must be used within TeamProvider');
  }
  return context;
};

export const TeamProvider = ({ children }) => {
  const { user, isAuthenticated } = useAuth();
  const { loadRooms } = useMessaging();
  const queryClient = useQueryClient();
  
  // State
  const [teams, setTeams] = useState([]);
  const [currentTeam, setCurrentTeam] = useState(null);
  const [teamPosts, setTeamPosts] = useState({}); // team_id -> posts array
  const [teamMembers, setTeamMembers] = useState({}); // team_id -> members array
  const [pendingInvitations, setPendingInvitations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const wsRef = useRef(null);

  // =====================
  // TEAM OPERATIONS
  // =====================

  const loadUserTeams = useCallback(async () => {
    if (!isAuthenticated) return;
    
    try {
      setIsLoading(true);
      const response = await teamService.getTeams();
      setTeams(response.teams || []);
      setError(null);
    } catch (error) {
      console.error('Failed to load teams:', error);
      setError('Failed to load teams');
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  const createTeam = useCallback(async (teamData) => {
    try {
      const newTeam = await teamService.createTeam(teamData);
      await loadUserTeams();
      await loadRooms(); // Reload rooms to get new team room
      // Invalidate folder tree to show new team root folder
      queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
      return newTeam;
    } catch (error) {
      console.error('Failed to create team:', error);
      throw error;
    }
  }, [loadUserTeams, loadRooms, queryClient, user?.user_id, user?.role]);

  const selectTeam = useCallback(async (teamId) => {
    try {
      setIsLoading(true);
      const team = await teamService.getTeam(teamId);
      setCurrentTeam(team);
      
      // Load team posts and members
      await Promise.all([
        loadTeamPosts(teamId),
        loadTeamMembers(teamId)
      ]);
      
      setError(null);
    } catch (error) {
      console.error('Failed to load team:', error);
      setError('Failed to load team');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateTeam = useCallback(async (teamId, updates) => {
    try {
      const updated = await teamService.updateTeam(teamId, updates);
      setTeams(prev => prev.map(t => t.team_id === teamId ? updated : t));
      if (currentTeam?.team_id === teamId) {
        setCurrentTeam(updated);
      }
      return updated;
    } catch (error) {
      console.error('Failed to update team:', error);
      throw error;
    }
  }, [currentTeam]);

  const deleteTeam = useCallback(async (teamId) => {
    try {
      await teamService.deleteTeam(teamId);
      setTeams(prev => prev.filter(t => t.team_id !== teamId));
      if (currentTeam?.team_id === teamId) {
        setCurrentTeam(null);
      }
      await loadRooms(); // Reload rooms after team deletion
    } catch (error) {
      console.error('Failed to delete team:', error);
      throw error;
    }
  }, [currentTeam, loadRooms]);

  // =====================
  // MEMBER OPERATIONS
  // =====================

  const loadTeamMembers = useCallback(async (teamId) => {
    try {
      const response = await teamService.getTeamMembers(teamId);
      setTeamMembers(prev => ({
        ...prev,
        [teamId]: response.members || []
      }));
    } catch (error) {
      console.error('Failed to load team members:', error);
    }
  }, []);

  const inviteMember = useCallback(async (teamId, userId) => {
    try {
      await teamService.createInvitation(teamId, userId);
      await loadTeamMembers(teamId);
    } catch (error) {
      console.error('Failed to invite member:', error);
      throw error;
    }
  }, [loadTeamMembers]);

  const removeMember = useCallback(async (teamId, userId) => {
    try {
      await teamService.removeMember(teamId, userId);
      await loadTeamMembers(teamId);
    } catch (error) {
      console.error('Failed to remove member:', error);
      throw error;
    }
  }, [loadTeamMembers]);

  const updateMemberRole = useCallback(async (teamId, userId, role) => {
    try {
      await teamService.updateMemberRole(teamId, userId, role);
      await loadTeamMembers(teamId);
    } catch (error) {
      console.error('Failed to update member role:', error);
      throw error;
    }
  }, [loadTeamMembers]);

  // =====================
  // INVITATION OPERATIONS
  // =====================

  const loadPendingInvitations = useCallback(async () => {
    if (!isAuthenticated) return;
    
    try {
      const invitations = await teamService.getPendingInvitations();
      setPendingInvitations(invitations || []);
    } catch (error) {
      console.error('Failed to load pending invitations:', error);
    }
  }, [isAuthenticated]);

  const acceptInvitation = useCallback(async (invitationId) => {
    try {
      const team = await teamService.acceptInvitation(invitationId);
      await loadPendingInvitations();
      await loadUserTeams();
      await loadRooms(); // Reload rooms to get new team room
      // Invalidate folder tree to show new team root folder
      queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
      return team;
    } catch (error) {
      console.error('Failed to accept invitation:', error);
      throw error;
    }
  }, [loadPendingInvitations, loadUserTeams, loadRooms, queryClient, user?.user_id, user?.role]);

  const rejectInvitation = useCallback(async (invitationId) => {
    try {
      await teamService.rejectInvitation(invitationId);
      await loadPendingInvitations();
    } catch (error) {
      console.error('Failed to reject invitation:', error);
      throw error;
    }
  }, [loadPendingInvitations]);

  // =====================
  // POST OPERATIONS
  // =====================

  const loadTeamPosts = useCallback(async (teamId, limit = 20, beforePostId = null) => {
    try {
      const response = await teamService.getTeamPosts(teamId, limit, beforePostId);
      setTeamPosts(prev => ({
        ...prev,
        [teamId]: response.posts || []
      }));
      return response;
    } catch (error) {
      console.error('Failed to load team posts:', error);
      throw error;
    }
  }, []);

  const createPost = useCallback(async (teamId, content, postType = 'text', attachments = []) => {
    try {
      const post = await teamService.createPost(teamId, content, postType, attachments);
      setTeamPosts(prev => ({
        ...prev,
        [teamId]: [post, ...(prev[teamId] || [])]
      }));
      return post;
    } catch (error) {
      console.error('Failed to create post:', error);
      throw error;
    }
  }, []);

  const deletePost = useCallback(async (teamId, postId) => {
    try {
      await teamService.deletePost(teamId, postId);
      setTeamPosts(prev => ({
        ...prev,
        [teamId]: (prev[teamId] || []).filter(p => p.post_id !== postId)
      }));
    } catch (error) {
      console.error('Failed to delete post:', error);
      throw error;
    }
  }, []);

  // =====================
  // REACTION OPERATIONS
  // =====================

  const addReaction = useCallback(async (teamId, postId, reactionType) => {
    try {
      await teamService.addReaction(teamId, postId, reactionType);
      // Reload posts to get updated reactions
      await loadTeamPosts(teamId);
    } catch (error) {
      console.error('Failed to add reaction:', error);
      throw error;
    }
  }, [loadTeamPosts]);

  const removeReaction = useCallback(async (teamId, postId, reactionType) => {
    try {
      await teamService.removeReaction(teamId, postId, reactionType);
      // Reload posts to get updated reactions
      await loadTeamPosts(teamId);
    } catch (error) {
      console.error('Failed to remove reaction:', error);
      throw error;
    }
  }, [loadTeamPosts]);

  // =====================
  // COMMENT OPERATIONS
  // =====================

  const createComment = useCallback(async (teamId, postId, content) => {
    try {
      const comment = await teamService.createComment(teamId, postId, content);
      // Reload posts to get updated comment count
      await loadTeamPosts(teamId);
      return comment;
    } catch (error) {
      console.error('Failed to create comment:', error);
      throw error;
    }
  }, [loadTeamPosts]);

  const deleteComment = useCallback(async (teamId, postId, commentId) => {
    try {
      await teamService.deleteComment(teamId, postId, commentId);
      // Reload posts to get updated comment count
      await loadTeamPosts(teamId);
    } catch (error) {
      console.error('Failed to delete comment:', error);
      throw error;
    }
  }, [loadTeamPosts]);

  // =====================
  // EFFECTS
  // =====================

  useEffect(() => {
    if (isAuthenticated) {
      loadUserTeams();
      loadPendingInvitations();
    }
  }, [isAuthenticated, loadUserTeams, loadPendingInvitations]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!isAuthenticated || !user) return;

    let reconnectAttempts = 0;
    let reconnectTimeout = null;
    const maxReconnectDelay = 30000; // 30 seconds max
    const initialReconnectDelay = 1000; // Start with 1 second

    const connectWebSocket = () => {
      const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
      if (!token) {
        console.warn('No auth token available for team WebSocket');
        return;
      }
      
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws?token=${token}`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('Team WebSocket connected');
        reconnectAttempts = 0; // Reset on successful connection
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Handle team-related events
          if (data.type === 'team.post.created') {
            const { team_id, post } = data;
            setTeamPosts(prev => ({
              ...prev,
              [team_id]: [post, ...(prev[team_id] || [])]
            }));
          } else if (data.type === 'team.post.deleted') {
            const { team_id, post_id } = data;
            setTeamPosts(prev => ({
              ...prev,
              [team_id]: (prev[team_id] || []).filter(p => p.post_id !== post_id)
            }));
          } else if (data.type === 'team.post.reaction') {
            const { team_id } = data;
            loadTeamPosts(team_id);
          } else if (data.type === 'team.post.comment') {
            const { team_id } = data;
            loadTeamPosts(team_id);
          } else if (data.type === 'team.member.joined') {
            const { team_id } = data;
            loadTeamMembers(team_id);
          } else if (data.type === 'team.member.left') {
            const { team_id } = data;
            loadTeamMembers(team_id);
          } else if (data.type === 'team.invitation.received') {
            loadPendingInvitations();
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        // Only log errors in development to reduce console noise
        if (process.env.NODE_ENV === 'development') {
          console.error('Team WebSocket error:', error);
        }
      };

      ws.onclose = () => {
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s (max)
        const delay = Math.min(
          initialReconnectDelay * Math.pow(2, reconnectAttempts),
          maxReconnectDelay
        );
        reconnectAttempts++;
        
        // Only log reconnection attempts in development
        if (process.env.NODE_ENV === 'development') {
          console.log(`Team WebSocket disconnected, reconnecting in ${delay}ms (attempt ${reconnectAttempts})...`);
        }
        
        reconnectTimeout = setTimeout(connectWebSocket, delay);
      };
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isAuthenticated, user, loadTeamPosts, loadTeamMembers, loadPendingInvitations]);

  const value = {
    // State
    teams,
    currentTeam,
    teamPosts,
    teamMembers,
    pendingInvitations,
    isLoading,
    error,
    
    // Team operations
    loadUserTeams,
    createTeam,
    selectTeam,
    updateTeam,
    deleteTeam,
    
    // Member operations
    loadTeamMembers,
    inviteMember,
    removeMember,
    updateMemberRole,
    
    // Invitation operations
    loadPendingInvitations,
    acceptInvitation,
    rejectInvitation,
    
    // Post operations
    loadTeamPosts,
    createPost,
    deletePost,
    
    // Reaction operations
    addReaction,
    removeReaction,
    
    // Comment operations
    createComment,
    deleteComment
  };

  return (
    <TeamContext.Provider value={value}>
      {children}
    </TeamContext.Provider>
  );
};

