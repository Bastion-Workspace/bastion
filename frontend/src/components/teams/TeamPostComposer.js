import React, { useState, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  IconButton,
  Paper,
  Typography,
  LinearProgress
} from '@mui/material';
import {
  Send,
  Image,
  AttachFile,
  Close
} from '@mui/icons-material';
import { useTeam } from '../../contexts/TeamContext';

const TeamPostComposer = ({ teamId }) => {
  const { createPost } = useTeam();
  const [content, setContent] = useState('');
  const [postType, setPostType] = useState('text');
  const [attachments, setAttachments] = useState([]);
  const [previewFiles, setPreviewFiles] = useState([]); // Local file previews before upload
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!content.trim() && previewFiles.length === 0) {
      return;
    }

    setIsSubmitting(true);
    setUploadProgress(0);

    try {
      // Upload files and get attachment metadata
      const uploadedAttachments = [];
      
      for (const preview of previewFiles) {
        const formData = new FormData();
        formData.append('file', preview.file);
        
        try {
          const response = await fetch(`/api/teams/${teamId}/posts/upload`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
            },
            body: formData
          });
          
          if (response.ok) {
            const data = await response.json();
            // Ensure file_path is set - it should always be returned by the backend
            if (!data.file_path && !data.url) {
              console.error('Upload response missing file_path:', data);
              throw new Error('Upload response missing file_path');
            }
            uploadedAttachments.push({
              filename: preview.filename,
              file_path: data.file_path || data.url,
              mime_type: preview.mime_type || data.mime_type,
              file_size: preview.file_size || data.file_size,
              width: data.width,
              height: data.height
            });
          } else {
            const errorText = await response.text();
            console.error('Failed to upload file:', errorText);
            throw new Error(`Failed to upload file: ${errorText}`);
          }
        } catch (error) {
          console.error('Failed to upload file:', error);
          // Continue with other files even if one fails
        }
      }
      
      // Create post with uploaded attachments
      await createPost(teamId, content.trim() || '', postType, uploadedAttachments);
      
      // Cleanup preview URLs
      previewFiles.forEach(preview => {
        if (preview.preview) {
          URL.revokeObjectURL(preview.preview);
        }
      });
      
      setContent('');
      setAttachments([]);
      setPreviewFiles([]);
      setPostType('text');
      setUploadProgress(0);
    } catch (error) {
      console.error('Failed to create post:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFileSelect = (e, type) => {
    const files = Array.from(e.target.files || []);
    
    // Create previews immediately (before upload)
    files.forEach(file => {
      const preview = {
        file: file,
        preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
        filename: file.name,
        mime_type: file.type,
        file_size: file.size,
        type: type
      };
      
      setPreviewFiles(prev => [...prev, preview]);
      
      // Set post type
      if (type === 'image') {
        setPostType('image');
      } else if (type === 'file') {
        setPostType('file');
      }
    });
    
    // Clear input so same file can be selected again
    e.target.value = '';
  };

  const removePreview = (index) => {
    const preview = previewFiles[index];
    if (preview && preview.preview) {
      URL.revokeObjectURL(preview.preview);
    }
    setPreviewFiles(prev => {
      const newFiles = prev.filter((_, i) => i !== index);
      if (newFiles.length === 0) {
        setPostType('text');
      }
      return newFiles;
    });
  };

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <form onSubmit={handleSubmit}>
        <TextField
          placeholder="What's on your mind?"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          fullWidth
          multiline
          rows={3}
          variant="outlined"
          disabled={isSubmitting}
          sx={{ mb: 2 }}
        />
        
        {previewFiles.length > 0 && (
          <Box sx={{ mb: 2 }}>
            {previewFiles.map((preview, index) => (
              <Box
                key={index}
                sx={{
                  display: 'inline-block',
                  position: 'relative',
                  mr: 1,
                  mb: 1,
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  overflow: 'hidden'
                }}
              >
                {preview.preview ? (
                  <Box sx={{ position: 'relative' }}>
                    <img
                      src={preview.preview}
                      alt={preview.filename}
                      style={{
                        maxWidth: '200px',
                        maxHeight: '200px',
                        display: 'block'
                      }}
                    />
                    <IconButton
                      size="small"
                      onClick={() => removePreview(index)}
                      disabled={isSubmitting}
                      sx={{
                        position: 'absolute',
                        top: 4,
                        right: 4,
                        bgcolor: 'rgba(0,0,0,0.5)',
                        color: 'white',
                        '&:hover': {
                          bgcolor: 'rgba(0,0,0,0.7)'
                        }
                      }}
                    >
                      <Close fontSize="small" />
                    </IconButton>
                  </Box>
                ) : (
                  <Box
                    sx={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      bgcolor: 'background.paper',
                      p: 1
                    }}
                  >
                    <AttachFile sx={{ mr: 1, fontSize: 16 }} />
                    <Typography variant="body2" sx={{ mr: 1 }}>
                      {preview.filename}
                    </Typography>
                    <IconButton
                      size="small"
                      onClick={() => removePreview(index)}
                      disabled={isSubmitting}
                    >
                      <Close fontSize="small" />
                    </IconButton>
                  </Box>
                )}
              </Box>
            ))}
          </Box>
        )}
        
        {uploadProgress > 0 && uploadProgress < 100 && (
          <Box sx={{ mb: 2 }}>
            <LinearProgress variant="determinate" value={uploadProgress} />
          </Box>
        )}
        
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <input
              ref={imageInputRef}
              type="file"
              accept="image/*"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => handleFileSelect(e, 'image')}
            />
            <input
              ref={fileInputRef}
              type="file"
              multiple
              style={{ display: 'none' }}
              onChange={(e) => handleFileSelect(e, 'file')}
            />
            
            <IconButton
              onClick={() => imageInputRef.current?.click()}
              disabled={isSubmitting}
              title="Add image"
            >
              <Image />
            </IconButton>
            
            <IconButton
              onClick={() => fileInputRef.current?.click()}
              disabled={isSubmitting}
              title="Add file"
            >
              <AttachFile />
            </IconButton>
          </Box>
          
          <Button
            type="submit"
            variant="contained"
            startIcon={<Send />}
            disabled={isSubmitting || (!content.trim() && previewFiles.length === 0)}
          >
            Post
          </Button>
        </Box>
      </form>
    </Paper>
  );
};

export default TeamPostComposer;

