import React, { useState, useEffect, useRef } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Avatar,
  Typography,
  Box,
  IconButton,
  Chip,
  Button,
  Collapse,
  TextField,
  Divider,
  Tooltip
} from '@mui/material';
import {
  ThumbUp,
  Comment,
  Delete,
  MoreVert,
  ExpandMore,
  ExpandLess,
  AttachFile
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';
import { useAuth } from '../../contexts/AuthContext';
import { formatDistanceToNow } from 'date-fns';
import AudioPlayer from '../AudioPlayer';
import EmbeddedContent from './EmbeddedContent';
import { parseContentForEmbeds } from '../../utils/embedUtils';

const TeamPostCard = ({ post, teamId }) => {
  const { user } = useAuth();
  const {
    deletePost,
    addReaction,
    removeReaction,
    createComment,
    loadTeamPosts,
    newPostIds,
    markPostAsViewed,
    teamMembers
  } = useTeam();
  const [showComments, setShowComments] = useState(false);
  const [comments, setComments] = useState([]);
  const [commentContent, setCommentContent] = useState('');
  const [isSubmittingComment, setIsSubmittingComment] = useState(false);
  const [imageBlobUrls, setImageBlobUrls] = useState({});
  const [newCommentHighlight, setNewCommentHighlight] = useState(false);
  const [isHighlighted, setIsHighlighted] = useState(false);
  const cardRef = useRef(null);
  
  const isNewPost = newPostIds.has(post.post_id);

  const isAuthor = post.author_id === user?.user_id;
  const userReaction = post.reactions?.find(r => r.users?.includes(user?.user_id));
  
  // Get team members for this team to map user IDs to names
  const members = teamMembers[teamId] || [];
  
  // Helper function to get user name from user_id
  const getUserName = (userId) => {
    const member = members.find(m => m.user_id === userId);
    if (member) {
      return member.display_name || member.username || 'Unknown';
    }
    // Fallback: if it's the current user, use their info
    if (userId === user?.user_id) {
      return user?.display_name || user?.username || 'You';
    }
    return 'Unknown';
  };
  
  // Helper function to format reaction tooltip text
  const getReactionTooltip = (reaction) => {
    if (!reaction.users || reaction.users.length === 0) {
      return '';
    }
    
    const userNames = reaction.users.map(userId => {
      if (userId === user?.user_id) {
        return 'You';
      }
      return getUserName(userId);
    });
    
    if (userNames.length === 1) {
      return userNames[0];
    } else if (userNames.length <= 5) {
      return userNames.join(', ');
    } else {
      return `${userNames.slice(0, 5).join(', ')} and ${userNames.length - 5} more`;
    }
  };
  
  // Load attachment images with auth token (create blob URLs)
  useEffect(() => {
    const loadImages = async () => {
      if (!post.attachments) return;
      
      const blobUrlsMap = {};
      
      for (const att of post.attachments) {
        if (att.mime_type?.startsWith('image/') && !imageBlobUrls[att.file_path]) {
          try {
            const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
            const response = await fetch(att.file_path, {
              headers: token ? { 'Authorization': `Bearer ${token}` } : {}
            });
            
            if (response.ok) {
              const blob = await response.blob();
              const blobUrl = URL.createObjectURL(blob);
              blobUrlsMap[att.file_path] = blobUrl;
            } else {
              console.error('Failed to load image:', response.status, att.file_path);
            }
          } catch (error) {
            console.error('Failed to load image blob:', error, att.file_path);
          }
        }
      }
      
      if (Object.keys(blobUrlsMap).length > 0) {
        setImageBlobUrls(prev => ({ ...prev, ...blobUrlsMap }));
      }
    };
    
    loadImages();
    
    // Cleanup blob URLs on unmount
    return () => {
      Object.values(imageBlobUrls).forEach(url => {
        if (url) URL.revokeObjectURL(url);
      });
    };
  }, [post.attachments]); // eslint-disable-line react-hooks/exhaustive-deps

  // Set highlight state when post is marked as new
  useEffect(() => {
    if (isNewPost) {
      setIsHighlighted(true);
    }
  }, [isNewPost]);

  // Intersection Observer to detect when post comes into view and fade highlight
  useEffect(() => {
    if (!isHighlighted || !cardRef.current) return;

    // Check if already in view immediately (for posts that appear at top while user is scrolled to top)
    const checkIfInView = () => {
      const rect = cardRef.current.getBoundingClientRect();
      const isInView = rect.top < window.innerHeight && rect.bottom > 0;
      if (isInView) {
        // Post is already visible - fade out highlight after a brief delay
        setTimeout(() => {
          setIsHighlighted(false);
          markPostAsViewed(post.post_id);
        }, 500);
        return true;
      }
      return false;
    };

    // Check immediately
    if (checkIfInView()) {
      return; // Already in view, no need for observer
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // Post is now visible - fade out highlight after a brief delay
            setTimeout(() => {
              setIsHighlighted(false);
              markPostAsViewed(post.post_id);
            }, 500); // Small delay so user sees the highlight
          }
        });
      },
      { threshold: 0.1 } // Trigger when 10% of post is visible
    );

    observer.observe(cardRef.current);

    return () => {
      observer.disconnect();
    };
  }, [isHighlighted, post.post_id, markPostAsViewed]);

  // Listen for new comments on this post
  useEffect(() => {
    const handleCommentAdded = (event) => {
      if (event.detail?.post_id === post.post_id) {
        const newComment = event.detail.comment;
        // Add comment to list if comments are expanded
        if (showComments) {
          setComments(prev => {
            // Check if comment already exists (avoid duplicates)
            if (prev.some(c => c.comment_id === newComment.comment_id)) {
              return prev;
            }
            return [...prev, newComment];
          });
          // Highlight the new comment briefly
          setNewCommentHighlight(true);
          setTimeout(() => setNewCommentHighlight(false), 2000);
        }
      }
    };
    
    window.addEventListener('teamPostCommentAdded', handleCommentAdded);
    return () => window.removeEventListener('teamPostCommentAdded', handleCommentAdded);
  }, [post.post_id, showComments]);

  const handleReaction = async (reactionType) => {
    try {
      if (userReaction?.reaction_type === reactionType) {
        await removeReaction(teamId, post.post_id, reactionType);
      } else {
        await addReaction(teamId, post.post_id, reactionType);
      }
    } catch (error) {
      console.error('Failed to toggle reaction:', error);
    }
  };

  const handleDelete = async () => {
    if (window.confirm('Are you sure you want to delete this post?')) {
      try {
        await deletePost(teamId, post.post_id);
      } catch (error) {
        console.error('Failed to delete post:', error);
      }
    }
  };

  const handleToggleComments = async () => {
    if (!showComments && post.comment_count > 0) {
      try {
        const response = await fetch(`/api/teams/${teamId}/posts/${post.post_id}/comments`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
          }
        });
        if (response.ok) {
          const data = await response.json();
          setComments(data.comments || []);
        }
      } catch (error) {
        console.error('Failed to load comments:', error);
      }
    }
    setShowComments(!showComments);
  };

  const handleSubmitComment = async (e) => {
    e.preventDefault();
    if (!commentContent.trim()) return;

    setIsSubmittingComment(true);
    try {
      const comment = await createComment(teamId, post.post_id, commentContent.trim());
      setComments(prev => [...prev, comment]);
      setCommentContent('');
    } catch (error) {
      console.error('Failed to create comment:', error);
    } finally {
      setIsSubmittingComment(false);
    }
  };

  return (
    <Card
      ref={cardRef}
      sx={{
        transition: 'background-color 3s cubic-bezier(0.4, 0, 0.2, 1)',
        backgroundColor: isHighlighted 
          ? (theme) => theme.palette.mode === 'dark' 
            ? 'rgba(255, 255, 255, 0.08)' 
            : 'rgba(25, 118, 210, 0.08)'
          : 'transparent'
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'start', mb: 2 }}>
          <Avatar
            src={post.author_avatar}
            sx={{ width: 40, height: 40, mr: 2 }}
          >
            {post.author_name?.[0]?.toUpperCase()}
          </Avatar>
          
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="subtitle1" fontWeight="bold">
              {post.author_name}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {formatDistanceToNow(new Date(post.created_at), { addSuffix: true })}
            </Typography>
          </Box>
          
          {isAuthor && (
            <IconButton size="small" onClick={handleDelete}>
              <Delete fontSize="small" />
            </IconButton>
          )}
        </Box>
        
        <Box sx={{ mb: 2 }}>
          {(() => {
            const segments = parseContentForEmbeds(post.content || '');
            return segments.map((segment, index) => {
              if (segment.type === 'embed') {
                return (
                  <EmbeddedContent
                    key={`embed-${index}`}
                    embedType={segment.embedType}
                    embedData={segment.embedData}
                  />
                );
              } else {
                // Render text with clickable URLs and line breaks preserved
                const renderTextWithLinks = (text) => {
                  // URL regex pattern
                  const urlRegex = /(https?:\/\/[^\s]+)/g;
                  const parts = [];
                  let lastIndex = 0;
                  let match;
                  
                  while ((match = urlRegex.exec(text)) !== null) {
                    // Add text before URL
                    if (match.index > lastIndex) {
                      parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
                    }
                    // Add URL as link
                    parts.push({ type: 'link', content: match[0], url: match[0] });
                    lastIndex = match.index + match[0].length;
                  }
                  
                  // Add remaining text
                  if (lastIndex < text.length) {
                    parts.push({ type: 'text', content: text.substring(lastIndex) });
                  }
                  
                  // If no URLs found, return text as-is
                  if (parts.length === 0) {
                    parts.push({ type: 'text', content: text });
                  }
                  
                  return parts;
                };
                
                const textLines = segment.content.split('\n');
                return (
                  <Typography
                    key={`text-${index}`}
                    variant="body1"
                    component="div"
                    sx={{ whiteSpace: 'pre-wrap', mb: segment.content.trim() ? 1 : 0 }}
                  >
                    {textLines.map((line, lineIndex) => {
                      const parts = renderTextWithLinks(line);
                      return (
                        <React.Fragment key={lineIndex}>
                          {parts.map((part, partIndex) => {
                            if (part.type === 'link') {
                              return (
                                <a
                                  key={partIndex}
                                  href={part.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  style={{
                                    color: '#1976d2',
                                    textDecoration: 'none',
                                    borderBottom: '1px solid #1976d2'
                                  }}
                                  onMouseEnter={(e) => {
                                    e.target.style.textDecoration = 'underline';
                                  }}
                                  onMouseLeave={(e) => {
                                    e.target.style.textDecoration = 'none';
                                  }}
                                >
                                  {part.content}
                                </a>
                              );
                            }
                            return <React.Fragment key={partIndex}>{part.content}</React.Fragment>;
                          })}
                          {lineIndex < textLines.length - 1 && <br />}
                        </React.Fragment>
                      );
                    })}
                  </Typography>
                );
              }
            });
          })()}
        </Box>
        
        {post.attachments && post.attachments.length > 0 && (
          <Box sx={{ mb: 2 }}>
            {post.attachments.map((att, index) => {
              // Debug: log attachment data
              console.log('Attachment data:', JSON.stringify(att, null, 2), 'teamId:', teamId);
              
              // Construct file URL - file_path should be the full API path
              // If file_path exists and starts with /api/, use it directly
              // Otherwise, construct from filename (fallback for legacy data)
              let fileUrl;
              if (att.file_path) {
                if (att.file_path.startsWith('/api/')) {
                  fileUrl = att.file_path;
                } else if (att.file_path.startsWith('http')) {
                  fileUrl = att.file_path;
                } else {
                  // If file_path is just a filename, construct the full path
                  fileUrl = `/api/teams/${teamId}/posts/attachments/${att.file_path}`;
                }
              } else {
                // Fallback: construct from filename (shouldn't happen with new posts)
                fileUrl = `/api/teams/${teamId}/posts/attachments/${att.filename || `attachment_${index}`}`;
              }
              
              // Add token as query parameter for direct image access (e.g., when opening in new tab)
              const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
              if (token && !fileUrl.includes('token=')) {
                const separator = fileUrl.includes('?') ? '&' : '?';
                fileUrl = `${fileUrl}${separator}token=${encodeURIComponent(token)}`;
              }
              
              console.log('Constructed fileUrl:', fileUrl, 'from file_path:', att.file_path, 'filename:', att.filename);
              
              return (
                <Box key={index} sx={{ mb: 1 }}>
                  {att.mime_type?.startsWith('audio/') ? (
                    <AudioPlayer
                      src={fileUrl}
                      filename={att.filename || 'Audio attachment'}
                    />
                  ) : att.mime_type?.startsWith('image/') ? (
                    imageBlobUrls[att.file_path] ? (
                      <img
                        src={imageBlobUrls[att.file_path]}
                        alt={att.filename || 'Attachment'}
                        style={{ maxWidth: '100%', maxHeight: '400px', borderRadius: 4, cursor: 'pointer' }}
                        onClick={async () => {
                          // Fetch image and open as blob URL in new tab to display (not download)
                          try {
                            const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
                            if (!token) {
                              console.error('No auth token available');
                              return;
                            }
                            
                            const response = await fetch(fileUrl, {
                              headers: { 'Authorization': `Bearer ${token}` }
                            });
                            
                            if (response.ok) {
                              const blob = await response.blob();
                              const blobUrl = URL.createObjectURL(blob);
                              
                              // Open in new tab - blob URLs display images instead of downloading
                              const newWindow = window.open(blobUrl, '_blank');
                              
                              // Clean up blob URL after a delay (window should have loaded it)
                              setTimeout(() => {
                                // Revoke after window opens (give it time to load)
                                setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
                              }, 100);
                              
                              // If popup was blocked, clean up immediately
                              if (!newWindow) {
                                URL.revokeObjectURL(blobUrl);
                                alert('Please allow popups to view images in a new tab.');
                              }
                            } else {
                              console.error('Failed to load full-size image:', response.status);
                              alert('Failed to load image. Please check your authentication.');
                            }
                          } catch (error) {
                            console.error('Error opening image:', error);
                            alert('Failed to open image. Please try again.');
                          }
                        }}
                        onError={(e) => {
                          console.error('Failed to load image:', imageBlobUrls[att.file_path], att);
                          e.target.style.display = 'none';
                        }}
                      />
                    ) : (
                      <Box sx={{ 
                        maxWidth: '100%', 
                        maxHeight: '400px', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        backgroundColor: '#f0f0f0',
                        borderRadius: 1,
                        p: 2
                      }}>
                        <Typography variant="body2" color="text.secondary">
                          Loading image...
                        </Typography>
                      </Box>
                    )
                  ) : (
                    <Button
                      variant="outlined"
                      size="small"
                      href={fileUrl}
                      download={att.filename || 'attachment'}
                    >
                      <AttachFile sx={{ mr: 1, fontSize: 16 }} />
                      {att.filename || 'Download attachment'}
                    </Button>
                  )}
                </Box>
              );
            })}
          </Box>
        )}
        
        {post.reactions && post.reactions.length > 0 && (
          <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
            {post.reactions.map((reaction) => {
              const tooltipText = getReactionTooltip(reaction);
              const chip = (
                <Chip
                  key={reaction.reaction_type}
                  label={`${reaction.reaction_type} ${reaction.count}`}
                  size="small"
                  onClick={() => handleReaction(reaction.reaction_type)}
                  color={userReaction?.reaction_type === reaction.reaction_type ? 'primary' : 'default'}
                />
              );
              
              return tooltipText ? (
                <Tooltip key={reaction.reaction_type} title={tooltipText} arrow>
                  {chip}
                </Tooltip>
              ) : (
                chip
              );
            })}
          </Box>
        )}
      </CardContent>
      
      <Divider />
      
      <CardActions>
        <IconButton
          size="small"
          onClick={() => handleReaction('👍')}
          color={userReaction?.reaction_type === '👍' ? 'primary' : 'default'}
        >
          <ThumbUp fontSize="small" />
        </IconButton>
        
        <Button
          size="small"
          startIcon={showComments ? <ExpandLess /> : <ExpandMore />}
          onClick={handleToggleComments}
        >
          {post.comment_count || 0} Comments
        </Button>
      </CardActions>
      
      <Collapse in={showComments}>
        <Box sx={{ p: 2 }}>
          {comments.map((comment, index) => (
            <Box 
              key={comment.comment_id} 
              sx={{ 
                mb: 2,
                ...(newCommentHighlight && index === comments.length - 1 ? {
                  bgcolor: (theme) => theme.palette.mode === 'dark' 
                    ? 'rgba(25, 118, 210, 0.1)' 
                    : 'rgba(25, 118, 210, 0.05)',
                  borderRadius: 1,
                  p: 1,
                  transition: 'background-color 0.3s ease'
                } : {})
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'start' }}>
                <Avatar sx={{ width: 32, height: 32, mr: 1 }}>
                  {comment.author_name?.[0]?.toUpperCase()}
                </Avatar>
                <Box sx={{ flexGrow: 1 }}>
                  <Typography variant="subtitle2">
                    {comment.author_name}
                  </Typography>
                  <Typography variant="body2">
                    {comment.content}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatDistanceToNow(new Date(comment.created_at), { addSuffix: true })}
                  </Typography>
                </Box>
              </Box>
            </Box>
          ))}
          
          <form onSubmit={handleSubmitComment}>
            <TextField
              placeholder="Write a comment..."
              value={commentContent}
              onChange={(e) => setCommentContent(e.target.value)}
              fullWidth
              size="small"
              disabled={isSubmittingComment}
              sx={{ mb: 1 }}
            />
            <Button
              type="submit"
              size="small"
              variant="outlined"
              disabled={isSubmittingComment || !commentContent.trim()}
            >
              Comment
            </Button>
          </form>
        </Box>
      </Collapse>
    </Card>
  );
};

export default TeamPostCard;

