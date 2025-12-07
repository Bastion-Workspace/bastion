import React, { useState, useEffect } from 'react';
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
  Divider
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
    loadTeamPosts
  } = useTeam();
  const [showComments, setShowComments] = useState(false);
  const [comments, setComments] = useState([]);
  const [commentContent, setCommentContent] = useState('');
  const [isSubmittingComment, setIsSubmittingComment] = useState(false);
  const [imageBlobUrls, setImageBlobUrls] = useState({});

  const isAuthor = post.author_id === user?.user_id;
  const userReaction = post.reactions?.find(r => r.users?.includes(user?.user_id));
  
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
    <Card>
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
                        onClick={() => {
                          // Open original URL in new tab with auth
                          const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
                          if (token) {
                            // For downloads, we might need a different approach
                            window.open(fileUrl, '_blank');
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
            {post.reactions.map((reaction) => (
              <Chip
                key={reaction.reaction_type}
                label={`${reaction.reaction_type} ${reaction.count}`}
                size="small"
                onClick={() => handleReaction(reaction.reaction_type)}
                color={userReaction?.reaction_type === reaction.reaction_type ? 'primary' : 'default'}
              />
            ))}
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
          {comments.map((comment) => (
            <Box key={comment.comment_id} sx={{ mb: 2 }}>
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

