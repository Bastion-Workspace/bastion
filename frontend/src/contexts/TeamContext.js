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
  const [unreadCounts, setUnreadCounts] = useState({}); // team_id -> unread count
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [newPostIds, setNewPostIds] = useState(new Set()); // post_id -> true (for highlighting new posts)
  
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

  const addMember = useCallback(async (teamId, userId, role = 'member') => {
    try {
      await teamService.addMember(teamId, userId, role);
      await loadTeamMembers(teamId);
    } catch (error) {
      console.error('Failed to add member:', error);
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
  // UNREAD TRACKING OPERATIONS
  // =====================

  const loadUnreadCounts = useCallback(async () => {
    if (!isAuthenticated) return;
    
    try {
      console.log('📊 Loading unread counts...');
      const counts = await teamService.getUnreadPostCounts();
      console.log('📊 Unread counts loaded:', counts);
      setUnreadCounts(counts || {});
    } catch (error) {
      console.error('Failed to load unread counts:', error);
    }
  }, [isAuthenticated]);

  const markTeamPostsAsRead = useCallback(async (teamId) => {
    try {
      await teamService.markTeamPostsAsRead(teamId);
      await loadUnreadCounts();
    } catch (error) {
      console.error('Failed to mark team posts as read:', error);
      throw error;
    }
  }, [loadUnreadCounts]);

  const muteTeam = useCallback(async (teamId, muted = true) => {
    try {
      console.log(`🔔 ${muted ? 'Muting' : 'Unmuting'} team ${teamId}... (current state: ${teams.find(t => t.team_id === teamId)?.muted})`);
      
      // Update local state FIRST for immediate UI feedback
      setTeams(prev => prev.map(t => 
        t.team_id === teamId ? { ...t, muted } : t
      ));
      
      // Then call the API
      await teamService.muteTeam(teamId, muted);
      console.log(`✅ Team ${teamId} ${muted ? 'muted' : 'unmuted'} successfully`);
      
      // Reload unread counts (don't reload teams to avoid overwriting local state)
      await loadUnreadCounts();
    } catch (error) {
      console.error('Failed to mute/unmute team:', error);
      // Revert local state on error
      setTeams(prev => prev.map(t => 
        t.team_id === teamId ? { ...t, muted: !muted } : t
      ));
      throw error;
    }
  }, [loadUnreadCounts, teams]);

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
  // TEAM SELECTION (moved after dependencies)
  // =====================

  const selectTeam = useCallback(async (teamId) => {
    try {
      setIsLoading(true);
      setError(null);
      // Clear current team immediately to show loading state when switching teams
      // This prevents showing stale data from the previous team
      setCurrentTeam(null);
      
      const team = await teamService.getTeam(teamId);
      setCurrentTeam(team);
      
      // Clear new post highlights when switching teams
      setNewPostIds(new Set());
      
      // Load team posts and members, mark as read
      await Promise.all([
        loadTeamPosts(teamId),
        loadTeamMembers(teamId),
        markTeamPostsAsRead(teamId)
      ]);
      
      setError(null);
    } catch (error) {
      console.error('Failed to load team:', error);
      setError('Failed to load team');
      setCurrentTeam(null); // Clear team on error
    } finally {
      setIsLoading(false);
    }
  }, [loadTeamPosts, loadTeamMembers, markTeamPostsAsRead]);

  // =====================
  // EFFECTS
  // =====================

  useEffect(() => {
    if (isAuthenticated) {
      loadUserTeams();
      loadPendingInvitations();
      loadUnreadCounts();
    }
  }, [isAuthenticated, loadUserTeams, loadPendingInvitations, loadUnreadCounts]);

  // Refresh unread counts periodically (reduced frequency to minimize re-renders)
  useEffect(() => {
    if (!isAuthenticated) return;
    
    const interval = setInterval(() => {
      loadUnreadCounts();
    }, 300000); // Every 5 minutes (reduced from 30s to prevent editor lag)
    
    return () => clearInterval(interval);
  }, [isAuthenticated, loadUnreadCounts]);

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
      const wsUrl = `${protocol}//${window.location.host}/api/ws/conversations?token=${token}`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ Team WebSocket connected');
        reconnectAttempts = 0; // Reset on successful connection
        // Load unread counts when WebSocket connects
        loadUnreadCounts();
        
        // Start heartbeat to keep connection alive
        const heartbeatInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            console.log('💓 Sending WebSocket heartbeat');
            ws.send(JSON.stringify({ type: 'heartbeat' }));
          } else {
            console.warn('⚠️ WebSocket not open, clearing heartbeat interval');
            clearInterval(heartbeatInterval);
          }
        }, 30000); // Every 30 seconds
        
        // Store interval reference for cleanup
        ws.heartbeatInterval = heartbeatInterval;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📨 WebSocket message received:', data.type, data);
          
          // Handle team-related events
          if (data.type === 'team.post.created') {
            const { team_id, post } = data;
            console.log('📝 New post created in team:', team_id, 'by user:', post?.author_id);
            
            // Only add post if it doesn't already exist (avoid duplicates from createPost + WebSocket)
            setTeamPosts(prev => {
              const existingPosts = prev[team_id] || [];
              // Check if post already exists by post_id
              const postExists = existingPosts.some(p => p.post_id === post?.post_id);
              if (postExists) {
                console.log('📝 Post already exists, skipping duplicate:', post?.post_id);
                return prev;
              }
              return {
                ...prev,
                [team_id]: [post, ...existingPosts]
              };
            });
            
            // If viewing this team's feed, mark post as new for highlight effect
            // Only mark if post is from another user (not your own post)
            if (currentTeam?.team_id === team_id && post?.author_id !== user?.user_id && post?.post_id) {
              setNewPostIds(prev => new Set([...prev, post.post_id]));
            }
            
            // Refresh unread counts when new post is created (for other users)
            // Only refresh if the post author is not the current user
            if (post?.author_id !== user?.user_id) {
              console.log('🔄 Reloading unread counts (new post from other user)...');
              loadUnreadCounts();
            }
          } else if (data.type === 'team.post.deleted') {
            const { team_id, post_id } = data;
            setTeamPosts(prev => ({
              ...prev,
              [team_id]: (prev[team_id] || []).filter(p => p.post_id !== post_id)
            }));
          } else if (data.type === 'team.post.reaction') {
            const { team_id, post_id, reaction_type, user_id, action } = data;
            console.log('📨 Reaction event:', { team_id, post_id, reaction_type, user_id, action });
            
            // Update the specific post's reactions in real-time
            setTeamPosts(prev => {
              const teamPosts = prev[team_id] || [];
              return {
                ...prev,
                [team_id]: teamPosts.map(p => {
                  if (p.post_id === post_id) {
                    const currentReactions = p.reactions || [];
                    let updatedReactions = [...currentReactions];
                    
                    if (action === 'add') {
                      // Find existing reaction of this type
                      const existingReactionIndex = updatedReactions.findIndex(
                        r => r.reaction_type === reaction_type
                      );
                      
                      if (existingReactionIndex >= 0) {
                        // Reaction type exists, add user to users array and increment count
                        const existingReaction = updatedReactions[existingReactionIndex];
                        const users = existingReaction.users || [];
                        if (!users.includes(user_id)) {
                          updatedReactions[existingReactionIndex] = {
                            ...existingReaction,
                            users: [...users, user_id],
                            count: (existingReaction.count || 0) + 1
                          };
                        }
                      } else {
                        // New reaction type, add it
                        updatedReactions.push({
                          reaction_type: reaction_type,
                          users: [user_id],
                          count: 1
                        });
                      }
                    } else if (action === 'remove') {
                      // Find existing reaction of this type
                      const existingReactionIndex = updatedReactions.findIndex(
                        r => r.reaction_type === reaction_type
                      );
                      
                      if (existingReactionIndex >= 0) {
                        const existingReaction = updatedReactions[existingReactionIndex];
                        const users = existingReaction.users || [];
                        const userIndex = users.indexOf(user_id);
                        
                        if (userIndex >= 0) {
                          // Remove user from users array and decrement count
                          const newUsers = users.filter(u => u !== user_id);
                          const newCount = Math.max(0, (existingReaction.count || 1) - 1);
                          
                          if (newCount === 0) {
                            // Remove reaction entirely if count is 0
                            updatedReactions = updatedReactions.filter(
                              r => r.reaction_type !== reaction_type
                            );
                          } else {
                            updatedReactions[existingReactionIndex] = {
                              ...existingReaction,
                              users: newUsers,
                              count: newCount
                            };
                          }
                        }
                      }
                    }
                    
                    return {
                      ...p,
                      reactions: updatedReactions
                    };
                  }
                  return p;
                })
              };
            });
            
            // Dispatch custom event for TeamPostCard to handle reaction update
            window.dispatchEvent(new CustomEvent('teamPostReactionUpdated', {
              detail: { post_id, reaction_type, user_id, action }
            }));
          } else if (data.type === 'team.post.comment') {
            const { team_id, post_id, comment } = data;
            // Update the specific post's comment count and add comment if needed
            setTeamPosts(prev => {
              const teamPosts = prev[team_id] || [];
              return {
                ...prev,
                [team_id]: teamPosts.map(p => {
                  if (p.post_id === post_id) {
                    return {
                      ...p,
                      comment_count: (p.comment_count || 0) + 1
                    };
                  }
                  return p;
                })
              };
            });
            // Dispatch custom event for TeamPostCard to handle comment addition
            window.dispatchEvent(new CustomEvent('teamPostCommentAdded', {
              detail: { post_id, comment }
            }));
          } else if (data.type === 'team.member.joined') {
            const { team_id, user_id } = data;
            // Reload team members for all viewers of this team
            loadTeamMembers(team_id);
            // If the added user is the current user, reload their teams list
            if (user_id === user?.user_id) {
              loadUserTeams();
              loadRooms(); // Reload rooms to get new team room
            }
          } else if (data.type === 'team.member.left') {
            const { team_id } = data;
            loadTeamMembers(team_id);
          } else if (data.type === 'team.invitation.received') {
            loadPendingInvitations();
          } else if (data.type === 'team.deleted') {
            const { team_id } = data;
            console.log('🗑️ Team deleted:', team_id);
            
            // Remove team from state
            setTeams(prev => prev.filter(t => t.team_id !== team_id));
            
            // Clear team posts for deleted team
            setTeamPosts(prev => {
              const next = { ...prev };
              delete next[team_id];
              return next;
            });
            
            // Clear team members for deleted team
            setTeamMembers(prev => {
              const next = { ...prev };
              delete next[team_id];
              return next;
            });
            
            // Clear unread counts for deleted team
            setUnreadCounts(prev => {
              const next = { ...prev };
              delete next[team_id];
              return next;
            });
            
            // Clear new post highlights for deleted team
            setNewPostIds(new Set());
            
            // If viewing this team, clear current team (will trigger navigation)
            if (currentTeam?.team_id === team_id) {
              setCurrentTeam(null);
            }
            
            // Reload rooms to remove team room
            loadRooms();
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
        
        // Clear heartbeat interval on error
        if (ws.heartbeatInterval) {
          clearInterval(ws.heartbeatInterval);
        }
      };

      ws.onclose = () => {
        console.log('Team WebSocket disconnected');
        
        // Clear heartbeat interval
        if (ws.heartbeatInterval) {
          clearInterval(ws.heartbeatInterval);
        }
        
        wsRef.current = null;
        
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
    }, [isAuthenticated, user, currentTeam, loadTeamPosts, loadTeamMembers, loadPendingInvitations, loadUnreadCounts, loadUserTeams, loadRooms]);

  // Function to mark a post as "viewed" (remove from new posts set)
  const markPostAsViewed = useCallback((postId) => {
    setNewPostIds(prev => {
      const next = new Set(prev);
      next.delete(postId);
      return next;
    });
  }, []);

  const value = {
    // State
    teams,
    currentTeam,
    teamPosts,
    teamMembers,
    pendingInvitations,
    unreadCounts,
    newPostIds,
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
    addMember,
    removeMember,
    updateMemberRole,
    
    // Invitation operations
    loadPendingInvitations,
    acceptInvitation,
    rejectInvitation,
    
    // Unread tracking operations
    loadUnreadCounts,
    markTeamPostsAsRead,
    muteTeam,
    
    // Post operations
    loadTeamPosts,
    createPost,
    deletePost,
    
    // Reaction operations
    addReaction,
    removeReaction,
    
    // Comment operations
    createComment,
    deleteComment,
    
    // New post highlighting
    markPostAsViewed
  };

  return (
    <TeamContext.Provider value={value}>
      {children}
    </TeamContext.Provider>
  );
};

