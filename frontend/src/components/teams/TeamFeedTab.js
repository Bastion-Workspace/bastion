import React, { useState, useEffect, useRef } from 'react';
import { Box, CircularProgress, Alert } from '@mui/material';
import { useTeam } from '../../contexts/TeamContext';
import TeamPostComposer from './TeamPostComposer';
import TeamPostCard from './TeamPostCard';

const TeamFeedTab = ({ teamId }) => {
  const {
    teamPosts,
    loadTeamPosts,
    isLoading
  } = useTeam();
  const [hasMore, setHasMore] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const observerRef = useRef(null);
  const lastPostRef = useRef(null);

  useEffect(() => {
    if (teamId) {
      loadTeamPosts(teamId).then((response) => {
        setHasMore(response.has_more || false);
      });
    }
  }, [teamId, loadTeamPosts]);

  useEffect(() => {
    // Intersection Observer for infinite scroll
    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loadingMore) {
          loadMorePosts();
        }
      },
      { threshold: 0.1 }
    );

    if (lastPostRef.current) {
      observerRef.current.observe(lastPostRef.current);
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, loadingMore, teamId]);

  const loadMorePosts = async () => {
    if (loadingMore || !hasMore) return;

    setLoadingMore(true);
    try {
      const posts = teamPosts[teamId] || [];
      const lastPost = posts[posts.length - 1];
      const response = await loadTeamPosts(teamId, 20, lastPost?.post_id);
      setHasMore(response.has_more || false);
    } catch (error) {
      console.error('Failed to load more posts:', error);
    } finally {
      setLoadingMore(false);
    }
  };

  const posts = teamPosts[teamId] || [];

  return (
    <Box>
      <TeamPostComposer teamId={teamId} />
      
      <Box sx={{ mt: 3 }}>
        {isLoading && posts.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : posts.length === 0 ? (
          <Alert severity="info">No posts yet. Be the first to post!</Alert>
        ) : (
          <>
            {posts.map((post, index) => (
              <Box
                key={post.post_id}
                ref={index === posts.length - 1 ? lastPostRef : null}
                sx={{ mb: 2 }}
              >
                <TeamPostCard post={post} teamId={teamId} />
              </Box>
            ))}
            
            {loadingMore && (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
                <CircularProgress size={24} />
              </Box>
            )}
          </>
        )}
      </Box>
    </Box>
  );
};

export default TeamFeedTab;

