import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Collapse,
  Tooltip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Chip,
  Divider,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  useTheme
} from '@mui/material';
import {
  Folder,
  FolderOpen,
  InsertDriveFile,
  ExpandMore,
  ExpandLess,
  Add,
  MoreVert,
  CreateNewFolder,
  Upload,
  Edit,
  Delete,
  DriveFileMove,
  Description,
  Article,
  CloudUpload,
  Refresh,
  CheckCircle,
  HourglassEmpty,
  Error,
  CloudSync,
  RssFeed,
  ChevronLeft,
  ChevronRight,
  Newspaper,
  FindInPage,
  Block,
  Audiotrack
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import apiService from '../services/apiService';
import folderService from '../services/folder/FolderService';
import { useAuth } from '../contexts/AuthContext';
import DocumentMetadataPane from './DocumentMetadataPane';
import FolderMetadataPane from './FolderMetadataPane';
import rssService from '../services/rssService';
import DataWorkspacesSection from './data_workspace/DataWorkspacesSection';

const FileTreeSidebar = ({ 
  selectedFolderId, 
  onFolderSelect, 
  onFileSelect,
  onRSSFeedClick,
  onAddRSSFeed,
  width = 280,
  isCollapsed: collapsedProp,
  onToggleCollapse
}) => {
  const theme = useTheme();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  
  // State
  const [expandedFolders, setExpandedFolders] = useState(() => {
    // Load expanded folders from localStorage
    try {
      const saved = localStorage.getItem('expandedFolders');
      const defaultExpanded = ['my_documents_root', 'global_documents_root'];
      return saved ? new Set([...defaultExpanded, ...JSON.parse(saved)]) : new Set(defaultExpanded);
    } catch (error) {
      console.error('Failed to load expanded folders from localStorage:', error);
      return new Set(['my_documents_root', 'global_documents_root']);
    }
  });
  const [contextMenu, setContextMenu] = useState(null);
  const [contextMenuTarget, setContextMenuTarget] = useState(null);
  const [vectorizationSubmenu, setVectorizationSubmenu] = useState(null);
  const [createFolderDialog, setCreateFolderDialog] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [newFolderParent, setNewFolderParent] = useState(null);
  const [newFolderCollectionType, setNewFolderCollectionType] = useState('user');
  const [newProjectDialog, setNewProjectDialog] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [projectType, setProjectType] = useState('electronics');
  const [projectParentFolder, setProjectParentFolder] = useState(null);
  const [uploadDialog, setUploadDialog] = useState(false);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploadTargetFolder, setUploadTargetFolder] = useState(null);
  const [uploadCategory, setUploadCategory] = useState('');
  const [uploadTags, setUploadTags] = useState([]);
  const [uploadTagInput, setUploadTagInput] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [metadataPane, setMetadataPane] = useState({ open: false, document: null, position: { x: 0, y: 0 } });
  const [folderMetadataPane, setFolderMetadataPane] = useState({ open: false, folder: null, position: { x: 0, y: 0 } });
  const [categories, setCategories] = useState([]);
  const [allTags, setAllTags] = useState([]);
  const [folderContents, setFolderContents] = useState({});
  const [hasOrgFiles, setHasOrgFiles] = useState(false);
  const [orgToolsExpanded, setOrgToolsExpanded] = useState(() => {
    const saved = localStorage.getItem('orgToolsExpanded');
    return saved !== null ? JSON.parse(saved) : true;
  });
  const [moveDialogOpen, setMoveDialogOpen] = useState(false);
  const [moveTarget, setMoveTarget] = useState(null); // { type: 'folder'|'file', data }
  const [moveDestinationId, setMoveDestinationId] = useState(null);
  const [rssFeedsExpanded, setRssFeedsExpanded] = useState(false);
  // Allow parent-controlled collapse to reclaim layout space
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isControlledCollapse = typeof collapsedProp === 'boolean';
  const isCollapsed = isControlledCollapse ? collapsedProp : internalCollapsed;
  const [processingFiles, setProcessingFiles] = useState([]);

  // Save expanded folders to localStorage whenever they change
  useEffect(() => {
    try {
      const expandedArray = Array.from(expandedFolders);
      localStorage.setItem('expandedFolders', JSON.stringify(expandedArray));
    } catch (error) {
      console.error('Failed to save expanded folders to localStorage:', error);
    }
  }, [expandedFolders]);

  // Toast notification function
  const showToast = useCallback((message, type = 'info') => {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#2196f3'};
      color: white;
      padding: 12px 20px;
      border-radius: 4px;
      z-index: 9999;
      font-size: 14px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      animation: slideIn 0.3s ease-out;
    `;
    
    // Add animation keyframes
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
      if (document.body.contains(toast)) {
        document.body.removeChild(toast);
      }
    }, 4000);
  }, []);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!user?.user_id) return;

    const token = apiService.getToken();
    if (!token) {
      console.error('❌ No authentication token available for folder WebSocket');
      return;
    }

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/folders?token=${encodeURIComponent(token)}`;
    let ws = null;

    try {
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('📡 Connected to folder updates WebSocket');
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          
          if (update.type === 'document_status_update') {
            console.log('🔄 ROOSEVELT: Received document status update:', update);
            
            // Update specific folder contents when document status changes - REAL-TIME!
            if (update.folder_id) {
              console.log(`📁 BULLY! Real-time status update for folder ${update.folder_id}`);
              
              // Immediately update folder contents to show new status
              apiService.getFolderContents(update.folder_id)
                .then(contents => {
                  setFolderContents(prev => {
                    const newContents = { ...prev, [update.folder_id]: contents };
                    console.log(`✅ ROOSEVELT: Real-time folder contents updated!`, {
                      folderId: update.folder_id,
                      documentCount: contents.documents?.length || 0,
                      documents: contents.documents?.map(d => ({ 
                        filename: d.filename, 
                        status: d.status || d.processing_status,
                        actual_status_field: d.status,
                        actual_processing_status_field: d.processing_status
                      }))
                    });
                    return newContents;
                  });
                  
                  // Remove from processing files list if status is completed
                  if (update.status === 'completed' && update.filename) {
                    setProcessingFiles(prev => prev.filter(f => f.filename !== update.filename));
                    showToast(`✅ "${update.filename}" processing completed!`, 'success');
                  }
                })
                .catch(error => {
                  console.error(`❌ Failed to refresh folder ${update.folder_id}:`, error);
                });
            }
          } else if (update.type === 'file_deleted') {
            console.log('🗑️ Received file deleted notification:', update);
            const folderId = update.folder_id;
            const deletedId = update.document_id;
            if (folderId && deletedId) {
              // Remove from in-memory folder contents if present
              setFolderContents(prev => {
                const current = prev[folderId];
                if (!current) return prev;
                return {
                  ...prev,
                  [folderId]: {
                    ...current,
                    documents: (current.documents || []).filter(d => d.document_id !== deletedId)
                  }
                };
              });
              // Refresh that folder's contents to ensure consistency
              queryClient.invalidateQueries(['folders', 'contents', folderId]);
              showToast('🗑️ File deleted', 'success');
            }

          } else if (update.type === 'folder_tree_refresh') {
            console.log('🔄 Received folder tree refresh request:', update);
            // Folder was deleted from disk but not found in DB - refresh tree to ensure consistency
            queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
            showToast('🔄 Folder tree refreshed', 'info');
            
          } else if (update.type === 'folder_event') {
            console.log('🔄 Received folder event notification:', update);
            
            // Handle different folder events
            if (update.action === 'created') {
              const folderName = update.folder?.name || 'New folder';
              showToast(`📁 ${folderName} created`, 'success');
              
              // Optimistic update - add folder to tree immediately
              queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
                if (!oldData) return oldData;
                
                const newFolder = {
                  folder_id: update.folder.folder_id,
                  name: update.folder.name,
                  parent_folder_id: update.folder.parent_folder_id,
                  user_id: update.folder.user_id,
                  collection_type: update.folder.collection_type,
                  created_at: update.folder.created_at,
                  updated_at: update.folder.created_at,
                  document_count: 0,
                  subfolder_count: 0,
                  children: [],
                  is_virtual_source: false
                };
                
                // Helper function to recursively add folder to the correct parent
                const addFolderToTree = (folders) => {
                  return folders.map(folder => {
                    // If this is the parent folder, add the new folder to its children
                    if (folder.folder_id === update.folder.parent_folder_id) {
                      return {
                        ...folder,
                        children: [...(folder.children || []), newFolder],
                        subfolder_count: (folder.subfolder_count || 0) + 1
                      };
                    }
                    // If this folder has children, recursively search them
                    else if (folder.children && folder.children.length > 0) {
                      const updatedChildren = addFolderToTree(folder.children);
                      // Check if any child was updated (meaning we found the parent)
                      if (updatedChildren !== folder.children) {
                        return { ...folder, children: updatedChildren };
                      }
                    }
                    // If this is a virtual root and folder has no parent (root level)
                    else if (!update.folder.parent_folder_id) {
                      if ((folder.folder_id === 'my_documents_root' && update.folder?.collection_type === 'user') ||
                          (folder.folder_id === 'global_documents_root' && update.folder?.collection_type === 'global')) {
                        return {
                          ...folder,
                          children: [...(folder.children || []), newFolder]
                        };
                      }
                    }
                    return folder;
                  });
                };
                
                const updatedFolders = addFolderToTree(oldData.folders);
                
                return {
                  ...oldData,
                  folders: updatedFolders,
                  total_folders: oldData.total_folders + 1
                };
              });
              
              console.log('✅ Optimistically updated folder tree with new folder');
              
              // Auto-expand the parent folder if it exists
              if (update.folder.parent_folder_id) {
                setExpandedFolders(prev => {
                  const newExpanded = new Set(prev);
                  newExpanded.add(update.folder.parent_folder_id);
                  console.log(`📁 BULLY! Auto-expanding parent folder ${update.folder.parent_folder_id} for new subfolder`);
                  return newExpanded;
                });
              }
            } else if (update.action === 'deleted') {
              const folderName = update.folder?.name || 'Folder';
              showToast(`🗑️ ${folderName} deleted`, 'success');
              
              // Optimistic update - remove folder from tree immediately
              queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
                if (!oldData) return oldData;
                
                // Helper function to recursively remove folder from tree
                const removeFolderFromTree = (folders) => {
                  return folders
                    .filter(folder => folder.folder_id !== update.folder.folder_id) // Remove if this is the deleted folder
                    .map(folder => {
                      // If this folder has children, recursively clean them
                      if (folder.children && folder.children.length > 0) {
                        const updatedChildren = removeFolderFromTree(folder.children);
                        // Update subfolder count if any children were removed
                        if (updatedChildren.length !== folder.children.length) {
                          return {
                            ...folder,
                            children: updatedChildren,
                            subfolder_count: Math.max(0, (folder.subfolder_count || 0) - (folder.children.length - updatedChildren.length))
                          };
                        } else {
                          return { ...folder, children: updatedChildren };
                        }
                      }
                      return folder;
                    });
                };
                
                const updatedFolders = removeFolderFromTree(oldData.folders);
                
                return {
                  ...oldData,
                  folders: updatedFolders,
                  total_folders: Math.max(0, oldData.total_folders - 1)
                };
              });
              
              console.log('✅ Optimistically updated folder tree - removed deleted folder');
            } else if (update.action === 'renamed') {
              const oldName = update.old_name || 'Folder';
              const newName = update.folder?.name || 'Folder';
              showToast(`📁 "${oldName}" renamed to "${newName}"`, 'success');
              
              // Optimistic update - rename folder in tree
              queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
                if (!oldData) return oldData;
                
                // Helper function to recursively find and rename folder
                const renameFolderInTree = (folders) => {
                  return folders.map(folder => {
                    // If this is the renamed folder, update its name
                    if (folder.folder_id === update.folder.folder_id) {
                      return {
                        ...folder,
                        name: update.folder.name,
                        updated_at: update.folder.updated_at || new Date().toISOString()
                      };
                    }
                    // If this folder has children, recursively search them
                    else if (folder.children && folder.children.length > 0) {
                      return { ...folder, children: renameFolderInTree(folder.children) };
                    }
                    return folder;
                  });
                };
                
                const updatedFolders = renameFolderInTree(oldData.folders);
                
                return {
                  ...oldData,
                  folders: updatedFolders
                };
              });
              
              console.log('✅ Optimistically updated folder tree - renamed folder');
            } else if (update.action === 'moved') {
              // Update tree to reflect new parent
              const movedFolder = update.folder;
              const oldParentId = update.old_parent_id;
              const newParentId = update.new_parent_id;
              showToast(`📁 Moved "${movedFolder?.name || 'Folder'}"`, 'success');
              
              queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
                if (!oldData) return oldData;
                
                const removeFromParent = (folders) => folders.map(f => ({
                  ...f,
                  children: (f.children ? removeFromParent(f.children) : f.children)
                })).filter(f => f.folder_id !== movedFolder.folder_id);
                
                const addToNewParent = (folders) => {
                  return folders.map(f => {
                    if (f.folder_id === newParentId) {
                      return {
                        ...f,
                        children: [...(f.children || []), {
                          folder_id: movedFolder.folder_id,
                          name: movedFolder.name,
                          parent_folder_id: newParentId,
                          user_id: movedFolder.user_id,
                          collection_type: movedFolder.collection_type,
                          created_at: movedFolder.created_at,
                          updated_at: movedFolder.updated_at,
                          document_count: movedFolder.document_count || 0,
                          subfolder_count: movedFolder.subfolder_count || 0,
                          children: []
                        }]
                      };
                    } else if (f.children && f.children.length > 0) {
                      return { ...f, children: addToNewParent(f.children) };
                    }
                    // Root-level placement when no parent
                    if (!newParentId && ((f.folder_id === 'my_documents_root' && movedFolder.collection_type === 'user') || (f.folder_id === 'global_documents_root' && movedFolder.collection_type === 'global'))) {
                      return {
                        ...f,
                        children: [...(f.children || []), {
                          folder_id: movedFolder.folder_id,
                          name: movedFolder.name,
                          parent_folder_id: null,
                          user_id: movedFolder.user_id,
                          collection_type: movedFolder.collection_type,
                          created_at: movedFolder.created_at,
                          updated_at: movedFolder.updated_at,
                          document_count: movedFolder.document_count || 0,
                          subfolder_count: movedFolder.subfolder_count || 0,
                          children: []
                        }]
                      };
                    }
                    return f;
                  });
                };
                
                let folders = removeFromParent(oldData.folders);
                folders = addToNewParent(folders);
                return { ...oldData, folders };
              });
              
              // Expand new parent so user sees the moved folder
              if (newParentId) {
                setExpandedFolders(prev => new Set(prev).add(newParentId));
              }
              // Ensure absolute consistency with server
              queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
            } else if (update.action === 'file_uploading') {
              const filename = update.filename || 'File';
              showToast(`📤 ${filename} uploading...`, 'info');
              
              // Optimistic update - add file to folder immediately
              queryClient.setQueryData(['folders', 'contents', update.folder_id], (oldData) => {
                if (!oldData) return oldData;
                
                const optimisticFile = {
                  document_id: update.document_id,
                  filename: update.filename,
                  title: update.filename,
                  status: 'uploading',
                  upload_date: update.timestamp,
                  user_id: update.user_id,
                  folder_id: update.folder_id
                };
                
                return {
                  ...oldData,
                  documents: [...(oldData.documents || []), optimisticFile]
                };
              });
              
              console.log('✅ Optimistically added uploading file to folder');
            } else if (update.action === 'file_processed') {
              const filename = update.filename || 'File';
              showToast(`✅ ${filename} processed successfully`, 'success');
              
              // Update file status in folder
              queryClient.setQueryData(['folders', 'contents', update.folder_id], (oldData) => {
                if (!oldData) return oldData;
                
                const updatedDocuments = (oldData.documents || []).map(doc => 
                  doc.document_id === update.document_id 
                    ? { ...doc, status: update.status }
                    : doc
                );
                
                return {
                  ...oldData,
                  documents: updatedDocuments
                };
              });
              
              console.log('✅ Updated file status to processed');
            } else if (update.action === 'file_failed') {
              const filename = update.filename || 'File';
              showToast(`❌ ${filename} processing failed`, 'error');
              
              // Update file status in folder
              queryClient.setQueryData(['folders', 'contents', update.folder_id], (oldData) => {
                if (!oldData) return oldData;
                
                const updatedDocuments = (oldData.documents || []).map(doc => 
                  doc.document_id === update.document_id 
                    ? { ...doc, status: 'failed' }
                    : doc
                );
                
                return {
                  ...oldData,
                  documents: updatedDocuments
                };
              });
              
              console.log('✅ Updated file status to failed');
            } else if (update.action === 'file_created') {
              const filename = update.filename || 'File';
              showToast(`📄 ${filename} uploaded`, 'success');
              
              // Refresh folder contents to show the new file
              queryClient.invalidateQueries(['folders', 'contents', update.folder_id]);
              console.log('✅ Refreshed folder contents due to file creation');
            }
            
          } else if (update.type === 'folder_update') {
            console.log('🔄 Received folder structure update:', update);
            
            // If this is a file creation or processing update, expand the folder and refresh contents
            if (update.action === 'file_created' || update.action === 'file_processed') {
              console.log(`📁 Auto-expanding folder ${update.folder_id} due to ${update.action}`);
              
              // Show toast notification for new file
              if (update.action === 'file_created') {
                const fileName = update.metadata?.filename || 'New file';
                showToast(`📄 ${fileName} added to folder`, 'success');
              }
              
              // Auto-expand the folder
              setExpandedFolders(prev => {
                const newExpanded = new Set(prev);
                newExpanded.add(update.folder_id);
                return newExpanded;
              });
              
              // **ROOSEVELT FIX**: Refresh BOTH folder contents AND the folder tree!
              // This ensures document counts update and new items appear immediately
              Promise.all([
                apiService.getFolderContents(update.folder_id),
                queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role])
              ])
                .then(([contents]) => {
                  setFolderContents(prev => ({ ...prev, [update.folder_id]: contents }));
                  console.log(`✅ Updated folder ${update.folder_id} contents AND tree via WebSocket for ${update.action}`);
                  
                  // **BULLY!** Also re-check for org files if this might be a new org file
                  if (update.metadata?.filename?.toLowerCase().endsWith('.org')) {
                    setHasOrgFiles(true);
                    console.log('📋 Detected new org file, enabling Org Tools');
                  }
                })
                .catch(error => {
                  console.error(`❌ Failed to refresh folder ${update.folder_id}:`, error);
                });
            } else {
              // For other folder updates, refresh the entire folder tree
              queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
            }
          }
        } catch (error) {
          console.error('❌ Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log(`📡 Disconnected from folder updates WebSocket. Code: ${event.code}, Reason: ${event.reason}`);
        
        // If closed due to auth error, log it clearly
        if (event.code === 1008) {
          console.error(`❌ WebSocket authentication failed: ${event.reason}`);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ Folder WebSocket error:', error);
      };

    } catch (error) {
      console.error('❌ Failed to connect to folder updates WebSocket:', error);
    }

    // Cleanup on unmount
    return () => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [user?.user_id, apiService, showToast]); // Add showToast to dependencies

  // Queries
  const { data: folderTree, isLoading, error, refetch } = useQuery(
    ['folders', 'tree', user?.user_id, user?.role],
    () => apiService.get('/api/folders/tree'),
    {
      refetchOnWindowFocus: false,
      staleTime: 30000, // 30 seconds
      // Preserve expanded state during refetches
      keepPreviousData: true,
    }
  );

  // Load contents for expanded folders on mount/tree change
  useEffect(() => {
    if (!folderTree || !expandedFolders.size) return;
    
    console.log('🔄 Loading contents for expanded folders:', Array.from(expandedFolders));
    
    // Create a set of valid folder IDs from the current folder tree
    const validFolderIds = new Set();
    const collectValidIds = (folders) => {
      // Ensure folders is an array
      if (!Array.isArray(folders)) {
        console.warn('⚠️ folders is not an array:', folders);
        return;
      }
      
      folders.forEach(folder => {
        if (folder && folder.folder_id) {
          validFolderIds.add(folder.folder_id);
          if (folder.children && Array.isArray(folder.children) && folder.children.length > 0) {
            collectValidIds(folder.children);
          }
        }
      });
    };
    
    // Normalize folder tree to an array of nodes across possible shapes
    const rawNodes = Array.isArray(folderTree)
      ? folderTree
      : (folderTree?.folders || folderTree?.data?.folders || folderTree?.result?.folders || []);
    const treeNodes = Array.isArray(rawNodes) ? rawNodes : [];
    collectValidIds(treeNodes);
    
    // Load contents for all expanded folders that aren't virtual roots AND exist in the current tree
    // Fetch on refresh to ensure files appear without manual toggle (avoid gating on existing cache)
    const realFolders = Array.from(expandedFolders).filter(folderId => 
      folderId !== 'my_documents_root' && 
      folderId !== 'global_documents_root' &&
      validFolderIds.has(folderId)
    );
    
    // Remove any expanded folders that no longer exist in the tree
    const invalidExpandedFolders = Array.from(expandedFolders).filter(folderId => 
      folderId !== 'my_documents_root' && 
      folderId !== 'global_documents_root' &&
      !validFolderIds.has(folderId)
    );
    
    if (invalidExpandedFolders.length > 0) {
      console.log('🧹 Removing invalid expanded folders:', invalidExpandedFolders);
      setExpandedFolders(prev => {
        const newExpanded = new Set(prev);
        invalidExpandedFolders.forEach(folderId => newExpanded.delete(folderId));
        return newExpanded;
      });
      
      // Update localStorage
      try {
        const validExpanded = Array.from(expandedFolders).filter(folderId => 
          folderId === 'my_documents_root' || 
          folderId === 'global_documents_root' ||
          validFolderIds.has(folderId)
        );
        localStorage.setItem('expandedFolders', JSON.stringify(validExpanded));
      } catch (error) {
        console.error('Failed to update expanded folders in localStorage:', error);
      }
    }
    
    if (realFolders.length > 0) {
      console.log(`📁 Auto-loading contents for ${realFolders.length} expanded folders`);
      
      // Load all folder contents in parallel
      Promise.all(
        realFolders.map(async (folderId) => {
          try {
            const contents = await apiService.getFolderContents(folderId);
            return { folderId, contents };
          } catch (error) {
            console.error(`Failed to auto-load contents for folder ${folderId}:`, error);
            return { folderId, contents: null };
          }
        })
      ).then(results => {
        // Update folder contents in batch
        setFolderContents(prev => {
          const newContents = { ...prev };
          results.forEach(({ folderId, contents }) => {
            if (contents) {
              newContents[folderId] = contents;
            }
          });
          return newContents;
        });
      });
    }
  }, [folderTree, expandedFolders]); // Trigger when tree loads or expanded set changes (avoid loops on contents update)

  // Get documents not in any folder (root documents)
  const { data: rootDocuments, error: documentsError, isLoading: documentsLoading } = useQuery(
    ['documents', 'root', user?.user_id],
    () => apiService.getUserDocuments(0, 100),
    {
      refetchOnWindowFocus: false,
      staleTime: 30000,
      enabled: !!user?.user_id,
      onSuccess: (data) => {
        console.log('📄 User documents loaded:', data);
        // Check if any org files exist
        // **TRUST BUST!** The API returns {documents: [...], total: N}, not a direct array!
        const documents = data?.documents || [];
        if (Array.isArray(documents)) {
          const hasOrg = documents.some(doc => doc.filename?.toLowerCase().endsWith('.org'));
          setHasOrgFiles(hasOrg);
        }
      },
      onError: (error) => {
        console.error('❌ Failed to load user documents:', error);
      }
    }
  );

  // Build a flat map for quick folder lookups (used for inheritance indicators)
  const folderMap = useMemo(() => {
    const map = {};
    const walk = (folders) => {
      if (!folders) return;
      folders.forEach((folder) => {
        map[folder.folder_id] = folder;
        if (folder.children && folder.children.length > 0) {
          walk(folder.children);
        }
      });
    };
    if (folderTree?.folders) {
      walk(folderTree.folders);
    }
    return map;
  }, [folderTree]);

  // Effective exemption for a folder (walk ancestors until explicit setting)
  const getEffectiveFolderExemption = useCallback((folderId) => {
    let current = folderMap[folderId];
    while (current) {
      const val = current.exempt_from_vectorization;
      if (val === true || val === false) {
        return {
          status: val,
          sourceFolderId: current.folder_id,
          sourceExplicit: current.folder_id === folderId
        };
      }
      current = current.parent_folder_id ? folderMap[current.parent_folder_id] : null;
    }
    return { status: false, sourceFolderId: null, sourceExplicit: false };
  }, [folderMap]);

  // Effective exemption for a file (document override first, else folder inheritance)
  const getEffectiveFileExemption = useCallback((file) => {
    if (file.exempt_from_vectorization === true) {
      return { status: true, source: 'document' };
    }
    if (file.exempt_from_vectorization === false) {
      return { status: false, source: 'document' };
    }
    if (file.folder_id) {
      const folderResult = getEffectiveFolderExemption(file.folder_id);
      return {
        status: folderResult.status,
        source: folderResult.sourceExplicit ? 'folder' : 'ancestor'
      };
    }
    return { status: false, source: 'default' };
  }, [getEffectiveFolderExemption]);

  // **ROOSEVELT FIX**: Check for org files ANYWHERE, not just in loaded folders!
  // This makes Org Tools appear immediately if ANY .org files exist
  useEffect(() => {
    const checkForOrgFilesAnywhere = async () => {
      try {
        // Quick check: do ANY of our documents have .org extension?
        // This queries the backend directly instead of relying on loaded folder contents
        const response = await apiService.getUserDocuments(0, 1000); // Get all user docs
        
        console.log('🔍 Checking for org files, response:', response);
        
        // **TRUST BUST!** The API returns {documents: [...], total: N}, not a direct array!
        const documents = response?.documents || [];
        
        if (Array.isArray(documents)) {
          const hasOrg = documents.some(doc => doc.filename?.toLowerCase().endsWith('.org'));
          setHasOrgFiles(hasOrg);
          console.log(`📋 Found org files? ${hasOrg} (checked ${documents.length} documents)`);
        } else {
          console.warn('⚠️ ROOSEVELT: Response.documents is not an array:', documents);
          setHasOrgFiles(false);
        }
      } catch (error) {
        console.error('❌ Failed to check for org files:', error);
        // Fallback to old logic if API fails
        checkForOrgFilesLocal();
      }
    };
    
    const checkForOrgFilesLocal = () => {
      // Fallback: Check root documents
      if (rootDocuments && Array.isArray(rootDocuments)) {
        const hasOrg = rootDocuments.some(doc => doc.filename?.toLowerCase().endsWith('.org'));
        if (hasOrg) {
          setHasOrgFiles(true);
          return;
        }
      }
      
      // Fallback: Check folder contents (only loaded folders)
      for (const contents of Object.values(folderContents)) {
        if (contents?.documents && Array.isArray(contents.documents)) {
          const hasOrg = contents.documents.some(doc => doc.filename?.toLowerCase().endsWith('.org'));
          if (hasOrg) {
            setHasOrgFiles(true);
            return;
          }
        }
      }
      
      setHasOrgFiles(false);
    };
    
    // Check immediately on mount and when user changes
    if (user?.user_id) {
      checkForOrgFilesAnywhere();
    }
  }, [user?.user_id]); // Only re-check when user changes, not on every folder load
  
  // **ROOSEVELT**: Persist orgToolsExpanded to localStorage
  useEffect(() => {
    localStorage.setItem('orgToolsExpanded', JSON.stringify(orgToolsExpanded));
  }, [orgToolsExpanded]);
  
  // **ROOSEVELT**: Persist expandedFolders to localStorage ALWAYS when it changes
  useEffect(() => {
    try {
      const expandedArray = Array.from(expandedFolders).filter(id => 
        // Filter out virtual roots - they're always expanded by default
        id !== 'my_documents_root' && id !== 'global_documents_root'
      );
      localStorage.setItem('expandedFolders', JSON.stringify(expandedArray));
      console.log('💾 Saved expanded folders:', expandedArray);
    } catch (error) {
      console.error('❌ Failed to save expanded folders:', error);
    }
  }, [expandedFolders]);

  // Get note categories and tags
  const { data: noteCategories } = useQuery(
    ['notes', 'categories'],
    () => apiService.getNoteTags(),
    {
      refetchOnWindowFocus: false,
      staleTime: 60000, // 1 minute
    }
  );

  // Get RSS feeds
  const { data: rssFeedsData, isLoading: rssFeedsLoading, error: rssFeedsError, refetch: refetchRSSFeeds } = useQuery(
    ['rss', 'feeds', 'categorized'],
    () => rssService.getCategorizedFeeds(),
    {
      refetchOnWindowFocus: false,
      staleTime: 30000, // 30 seconds
      onSuccess: (data) => {
        console.log('📰 RSS categorized feeds loaded:', data);
        console.log('📰 User feeds:', data?.user_feeds?.length || 0);
        console.log('📰 Global feeds:', data?.global_feeds?.length || 0);
      },
      onError: (error) => {
        console.error('❌ Failed to load categorized RSS feeds:', error);
      }
    }
  );

  // Extract user and global feeds from categorized data
  const userRSSFeeds = rssFeedsData?.user_feeds || [];
  const globalRSSFeeds = rssFeedsData?.global_feeds || [];
  
  // For backward compatibility, also provide all feeds combined
  const rssFeeds = [...userRSSFeeds, ...globalRSSFeeds];

  // Get RSS unread counts
  const { data: rssUnreadCounts } = useQuery(
    ['rss', 'unread-counts'],
    () => rssService.getUnreadCounts(),
    {
      refetchOnWindowFocus: false,
      staleTime: 30000, // 30 seconds
    }
  );

  // Mutations
  const createProjectMutation = useMutation(
    (data) => apiService.createProject(data.parent_folder_id, data.project_name, data.project_type),
    {
      onSuccess: (response) => {
        setNewProjectDialog(false);
        setProjectName('');
        setProjectType('electronics');
        setProjectParentFolder(null);
        showToast(`✅ Project "${response.project_name || projectName}" created successfully!`, 'success');
        // Invalidate folder tree to refresh
        queryClient.invalidateQueries(['folders', 'tree']);
        queryClient.invalidateQueries(['folders', 'contents']);
      },
      onError: (error) => {
        console.error('❌ Project creation failed:', error);
        alert(`Failed to create project: ${error.message || 'Unknown error'}`);
      }
    }
  );

  const createFolderMutation = useMutation(
    (data) => apiService.post('/api/folders', data),
    {
      onMutate: async (newFolderData) => {
        // Cancel any outgoing refetches
        await queryClient.cancelQueries(['folders', 'tree']);
        
        // Snapshot the previous value
        const previousFolders = queryClient.getQueryData(['folders', 'tree']);
        
        // Generate a temporary ID for optimistic update
        const tempId = `temp-${Date.now()}`;
        const optimisticFolder = {
          folder_id: tempId,
          name: newFolderData.name,
          parent_folder_id: newFolderData.parent_folder_id || null,
          user_id: user?.user_id,
          collection_type: newFolderData.collection_type || 'user',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          document_count: 0,
          subfolder_count: 0,
          children: [],
          is_virtual_source: false
        };
        
        // Optimistically update the folder tree
        queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
          if (!oldData) return oldData;
          
          const updatedFolders = oldData.folders.map(virtualRoot => {
            if (virtualRoot.folder_id === 'my_documents_root' && newFolderData.collection_type === 'user') {
              return {
                ...virtualRoot,
                children: [...(virtualRoot.children || []), optimisticFolder]
              };
            }
            return virtualRoot;
          });
          
          return {
            ...oldData,
            folders: updatedFolders,
            total_folders: oldData.total_folders + 1
          };
        });
        
        // Return context with the optimistic folder and previous data
        return { optimisticFolder, previousFolders };
      },
      onSuccess: (response, variables, context) => {
        // Update the optimistic folder with the real data
        if (context?.optimisticFolder) {
          queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
            if (!oldData) return oldData;
            
            const updatedFolders = oldData.folders.map(virtualRoot => {
              if (virtualRoot.folder_id === 'my_documents_root') {
                return {
                  ...virtualRoot,
                  children: virtualRoot.children.map(folder => 
                    folder.folder_id === context.optimisticFolder.folder_id 
                      ? { ...folder, folder_id: response.folder_id }
                      : folder
                  )
                };
              }
              return virtualRoot;
            });
            
            return {
              ...oldData,
              folders: updatedFolders
            };
          });
        }
        
        setCreateFolderDialog(false);
        setNewFolderName('');
        setNewFolderParent(null);
        setNewFolderCollectionType('user');
      },
      onError: (error, variables, context) => {
        // Rollback on error
        if (context?.previousFolders) {
          queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], context.previousFolders);
        }
        console.error('❌ Folder creation failed:', error);
        alert(`Failed to create folder: ${error.message || 'Unknown error'}`);
      },
      onSettled: () => {
        // Always refetch to ensure consistency
        queryClient.invalidateQueries(['folders', 'tree']);
      }
    }
  );

  const createDefaultFoldersMutation = useMutation(
    () => apiService.createDefaultFolders(),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['folders', 'tree']);
      },
    }
  );

  const deleteFolderMutation = useMutation(
    (folderId) => apiService.delete(`/api/folders/${folderId}?recursive=true`),
    {
      onMutate: async (folderId) => {
        // Cancel any outgoing refetches
        await queryClient.cancelQueries(['folders', 'tree']);
        
        // Snapshot the previous value
        const previousFolders = queryClient.getQueryData(['folders', 'tree']);
        
        // Optimistically remove the folder from the tree
        queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
          if (!oldData) return oldData;
          
          const updatedFolders = oldData.folders.map(virtualRoot => {
            if (virtualRoot.folder_id === 'my_documents_root') {
              return {
                ...virtualRoot,
                children: (virtualRoot.children || []).filter(folder => 
                  folder.folder_id !== folderId
                )
              };
            }
            return virtualRoot;
          });
          
          return {
            ...oldData,
            folders: updatedFolders,
            total_folders: Math.max(0, oldData.total_folders - 1)
          };
        });
        
        // Return context with the previous data
        return { previousFolders };
      },
      onSuccess: () => {
        setContextMenu(null);
      },
      onError: (error, variables, context) => {
        // Rollback on error
        if (context?.previousFolders) {
          queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], context.previousFolders);
        }
        console.error('❌ Folder deletion failed:', error);
        alert(`Failed to delete folder: ${error.message || 'Unknown error'}`);
      },
      onSettled: () => {
        // Always refetch to ensure consistency
        queryClient.invalidateQueries(['folders', 'tree']);
      }
    }
  );

  const uploadMutation = useMutation(
    ({ formData, isGlobal, filename }) => {
      const endpoint = isGlobal ? '/api/documents/upload' : '/api/user/documents/upload';
      console.log(`📤 Using ${isGlobal ? 'admin' : 'user'} upload endpoint: ${endpoint}`);
      
      return apiService.request(endpoint, {
        method: 'POST',
        body: formData,
        headers: {} // Let browser set Content-Type for FormData
      });
    },
    {
      onSuccess: (result, variables) => {
        console.log('✅ File uploaded successfully:', result);
        
        // **ROOSEVELT STATUS-AWARE PROCESSING INDICATOR!**
        // Check if file is already completed (e.g., org files are instant)
        const isAlreadyComplete = result.status === 'completed';
        
        if (isAlreadyComplete) {
          // File processed instantly - no spinner needed!
          console.log(`⚡ BULLY! "${variables.filename}" processed instantly (${result.status})`);
          showToast(`✅ "${variables.filename}" uploaded and ready!`, 'success');
        } else {
          // File still processing - show spinner
          console.log(`🔄 "${variables.filename}" processing in background (${result.status})`);
          showToast(`✅ "${variables.filename}" uploaded successfully! Processing in background...`, 'success');
          
          // Add file to processing list
          setProcessingFiles(prev => [...prev, {
            filename: variables.filename,
            documentId: result.document_id,
            startTime: new Date(),
            status: result.status
          }]);
          
          // **ROOSEVELT FALLBACK POLLING!** If no WebSocket update in 30 seconds, check status
          setTimeout(() => {
            apiService.get(`/api/user/documents/${result.document_id}`)
              .then(doc => {
                if (doc.processing_status === 'completed' || doc.processing_status === 'failed') {
                  console.log(`⏰ ROOSEVELT FALLBACK: Manually removing "${variables.filename}" from processing (status: ${doc.processing_status})`);
                  setProcessingFiles(prev => prev.filter(f => f.documentId !== result.document_id));
                }
              })
              .catch(err => console.error('❌ Fallback status check failed:', err));
          }, 30000); // 30 second safety net
        }
        
        // Close upload dialog immediately for better UX
        setUploadDialog(false);
        setUploadFiles([]);
        setUploadTargetFolder(null);
        
        // Utility: find path of ancestor IDs from root to target folder
        const getFolderPathIds = (treeData, targetId) => {
          const stack = [];
          const nodes = Array.isArray(treeData) ? treeData : (treeData?.folders || []);
          const dfs = (arr, path) => {
            if (!Array.isArray(arr)) return null;
            for (const node of arr) {
              const nextPath = [...path, node.folder_id];
              if (node.folder_id === targetId) return nextPath;
              const found = dfs(node.children, nextPath);
              if (found) return found;
            }
            return null;
          };
          return dfs(nodes, []) || [];
        };
        
        // Get the target folder ID
        const targetFolderId = contextMenuTarget?.folder_id || uploadTargetFolder?.folder_id;
        console.log('🎯 Upload success - target folder:', targetFolderId);
        
        if (targetFolderId) {
          // Expand ancestor chain including target
          const pathIds = getFolderPathIds(folderTree, targetFolderId);
          setExpandedFolders(prev => {
            const newSet = new Set(prev);
            pathIds.forEach(id => newSet.add(id));
            try {
              localStorage.setItem('expandedFolders', JSON.stringify(Array.from(newSet)));
            } catch {}
            return newSet;
          });

          // Ensure folder selected so its files render
          try { onFolderSelect?.(targetFolderId); } catch {}
          
          // Refresh the specific folder contents to show the uploaded file
          apiService.getFolderContents(targetFolderId)
            .then(contents => {
              setFolderContents(prev => ({
                ...prev,
                [targetFolderId]: contents
              }));
              
              // Scroll folder into view after layout updates
              setTimeout(() => {
                const el = document.querySelector(`[data-folder-id="${targetFolderId}"]`);
                if (el && typeof el.scrollIntoView === 'function') {
                  el.scrollIntoView({ block: 'center', behavior: 'smooth' });
                }
              }, 50);
            })
            .catch(error => {
              console.error('Failed to reload folder contents:', error);
            });
        } else {
          // No target folder, refresh the document tree minimally
          queryClient.invalidateQueries(['documents', 'root']);
        }
      },
      onError: (error, variables) => {
        console.error('❌ File upload failed:', error);
        showToast(`❌ Failed to upload "${variables.filename}": ${error.message || 'Unknown error'}`, 'error');
      },
    }
  );

  const reprocessDocumentMutation = useMutation(
    (documentId) => apiService.reprocessUserDocument(documentId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['folders', 'tree']);
        queryClient.invalidateQueries(['documents', 'root']);
      },
    }
  );

  // Delete a document (file)
  const deleteDocumentMutation = useMutation(
    (documentId) => apiService.deleteDocument(documentId),
    {
      onSuccess: (_data, documentId) => {
        // Optimistically remove from current folder contents if present
        const folderId = contextMenuTarget?.folder_id;
        if (folderId && folderContents[folderId]?.documents) {
          setFolderContents(prev => ({
            ...prev,
            [folderId]: {
              ...prev[folderId],
              documents: prev[folderId].documents.filter(d => d.document_id !== documentId)
            }
          }));
          // Invalidate the specific folder contents to confirm deletion with server
          queryClient.invalidateQueries(['folders', 'contents', folderId]);
        }
        setContextMenu(null);
        // Keep counts in sync
        queryClient.invalidateQueries(['folders', 'tree']);
      },
      onError: (error) => {
        console.error('❌ File deletion failed:', error);
        alert(`Failed to delete file: ${error?.message || 'Unknown error'}`);
      }
    }
  );

  // Rename a document using FileManager rename (normalizes extension and disk file)
  const renameDocumentMutation = useMutation(
    ({ documentId, newFilename }) => apiService.renameDocument(documentId, newFilename),
    {
      onSuccess: (resp, variables) => {
        const { documentId } = variables || {};
        const newFilename = resp?.new_filename || variables?.newFilename;
        const newTitle = newFilename;
        // Update folder contents in place for immediate UI feedback
        const folderId = contextMenuTarget?.folder_id;
        if (folderId && folderContents[folderId]?.documents) {
          setFolderContents(prev => ({
            ...prev,
            [folderId]: {
              ...prev[folderId],
              documents: prev[folderId].documents.map(d =>
                d.document_id === documentId ? { ...d, title: newTitle, filename: newFilename } : d
              )
            }
          }));
        }
        setContextMenu(null);
      },
      onError: (error) => {
        console.error('❌ Rename failed:', error);
        alert(`Failed to rename: ${error?.message || 'Unknown error'}`);
      }
    }
  );

  const renameFolderMutation = useMutation(
    ({ folderId, newName }) => folderService.updateFolder(folderId, { name: newName }),
    {
      onSuccess: () => {
        // Invalidate folder tree to refresh with new name
        queryClient.invalidateQueries(['folders', 'tree']);
        setContextMenu(null);
      },
      onError: (error) => {
        console.error('❌ Folder rename failed:', error);
        alert(`Failed to rename folder: ${error?.message || 'Unknown error'}`);
      }
    }
  );

  const createNoteMutation = useMutation(
    (noteData) => apiService.createNote(noteData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['notes']);
        queryClient.invalidateQueries(['folders', 'tree']);
      },
    }
  );

  // RSS mutations
  const refreshRSSFeedMutation = useMutation(
    (feedId) => rssService.refreshFeed(feedId),
    {
      onSuccess: (data, feedId) => {
        console.log(`✅ RSS feed ${feedId} refreshed:`, data);
        queryClient.invalidateQueries(['rss', 'feeds']);
        queryClient.invalidateQueries(['rss', 'unread-counts']);
        // Show toast notification
        // TODO: Add toast notification system
      },
      onError: (error, feedId) => {
        console.error(`❌ Failed to refresh RSS feed ${feedId}:`, error);
        alert(`Failed to refresh feed: ${error.message || 'Unknown error'}`);
      },
    }
  );

  const deleteRSSFeedMutation = useMutation(
    ({ feedId, deleteArticles }) => rssService.deleteFeed(feedId, deleteArticles),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['rss', 'feeds']);
        queryClient.invalidateQueries(['rss', 'unread-counts']);
        setContextMenu(null);
      },
      onError: (error) => {
        console.error('❌ Failed to delete RSS feed:', error);
        alert(`Failed to delete feed: ${error.message || 'Unknown error'}`);
      },
    }
  );

  // Handlers
  const handleFolderToggle = useCallback(async (folderId) => {
    setExpandedFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) {
        newSet.delete(folderId);
      } else {
        newSet.add(folderId);
        // Always load folder contents when expanding to ensure we have the latest data
        apiService.getFolderContents(folderId)
          .then(contents => {
            console.log(`📁 Loading contents for folder ${folderId}:`, contents);
            setFolderContents(prev => ({
              ...prev,
              [folderId]: contents
            }));
          })
          .catch(error => {
            console.error('Failed to load folder contents:', error);
          });
      }
      return newSet;
    });
  }, []);

  const handleFolderClick = useCallback((folderId) => {
    // Handle RSS virtual directories - open them in the tabbed content area
    if (folderId === 'rss_feeds_virtual' || folderId === 'global_rss_feeds_virtual') {
      // For RSS virtual directories, we want to show the RSS feeds
      // This will be handled by the parent component through onFolderSelect
      onFolderSelect?.(folderId);
    } else if (folderId === 'news_virtual') {
      try {
        // Open the headlines tab via TabbedContentManager exposed method
        const event = new CustomEvent('openNewsHeadlines');
        window.dispatchEvent(event);
      } catch {}
    } else {
      onFolderSelect?.(folderId);
    }
  }, [onFolderSelect]);

  const handleContextMenu = useCallback((event, target) => {
    event.preventDefault();
    setContextMenu({
      mouseX: event.clientX + 2,
      mouseY: event.clientY - 6,
    });
    setContextMenuTarget(target);
  }, []);

  const handleContextMenuClose = useCallback(() => {
    setContextMenu(null);
    setContextMenuTarget(null);
    setVectorizationSubmenu(null);
  }, []);

  const handleCreateFolder = useCallback(() => {
    if (!newFolderName.trim()) return;
    
    createFolderMutation.mutate({
      name: newFolderName.trim(),
      parent_folder_id: newFolderParent,
      collection_type: newFolderCollectionType
    });
  }, [newFolderName, newFolderParent, newFolderCollectionType, createFolderMutation]);

  const handleCreateProject = useCallback(() => {
    if (!projectName.trim()) return;
    
    createProjectMutation.mutate({
      parent_folder_id: projectParentFolder,
      project_name: projectName.trim(),
      project_type: projectType
    });
  }, [projectName, projectParentFolder, projectType, createProjectMutation]);

  const handleCreateDefaultFolders = useCallback(() => {
    createDefaultFoldersMutation.mutate();
  }, [createDefaultFoldersMutation]);

  const handleDeleteFolder = useCallback(() => {
    if (!contextMenuTarget?.folder_id) return;
    
    // Prevent deletion of virtual root folders
    if (contextMenuTarget.folder_id.includes('_root')) {
      alert('Cannot delete system folders (My Documents, Global Documents)');
      return;
    }
    
    // Prevent deletion of team root folders (must delete team instead)
    if (contextMenuTarget.collection_type === 'team' && !contextMenuTarget.parent_folder_id) {
      alert('Cannot delete team root folder. Delete the team instead to remove the folder.');
      return;
    }
    
    if (window.confirm('Are you sure you want to delete this folder and all its contents?')) {
      deleteFolderMutation.mutate(contextMenuTarget.folder_id);
    }
  }, [contextMenuTarget, deleteFolderMutation]);

  const handleUploadFiles = useCallback(() => {
    if (uploadFiles.length === 0) return;
    
    // Get the target folder ID from context menu or upload dialog
    const targetFolderId = contextMenuTarget?.folder_id || uploadTargetFolder?.folder_id || null;
    
    console.log('🚀 Uploading files to folder:', targetFolderId, 'Context target:', contextMenuTarget, 'Upload target:', uploadTargetFolder);
    console.log('📁 Upload target folder details:', {
      contextMenuTarget: contextMenuTarget ? { id: contextMenuTarget.folder_id, name: contextMenuTarget.name } : null,
      uploadTargetFolder: uploadTargetFolder ? { id: uploadTargetFolder.folder_id, name: uploadTargetFolder.name } : null,
      finalTargetId: targetFolderId
    });
    
    // Handle file uploads
    uploadFiles.forEach(file => {
      const formData = new FormData();
      formData.append('file', file);
      
      // Add folder_id if we have a target folder
      if (targetFolderId) {
        formData.append('folder_id', targetFolderId);
      }
      
      // **ROOSEVELT METADATA**: Add category and tags if provided
      if (uploadCategory) {
        formData.append('category', uploadCategory);
      }
      if (uploadTags.length > 0) {
        formData.append('tags', uploadTags.join(','));
      }
      
      // Determine if this is a global folder upload
      const targetFolder = contextMenuTarget || uploadTargetFolder;
      const isGlobal = targetFolder?.collection_type === 'global';
      
      console.log(`📁 Upload to ${isGlobal ? 'global' : 'user'} folder:`, targetFolder?.name, `(${targetFolder?.collection_type})`);
      console.log(`📋 Metadata - category: ${uploadCategory}, tags: ${uploadTags.join(', ')}`);
      
      uploadMutation.mutate({ formData, isGlobal, filename: file.name });
    });
    
    // Reset metadata fields after upload
    setUploadCategory('');
    setUploadTags([]);
    setUploadTagInput('');
  }, [uploadFiles, uploadMutation, contextMenuTarget, uploadTargetFolder, uploadCategory, uploadTags]);

  const handleReprocessDocument = useCallback((documentId) => {
    if (window.confirm('Are you sure you want to reprocess this document? This will update its embeddings and metadata.')) {
      reprocessDocumentMutation.mutate(documentId);
    }
  }, [reprocessDocumentMutation]);

  const handleEditMetadata = useCallback((document, event) => {
    setMetadataPane({
      open: true,
      document,
      position: { x: event.clientX, y: event.clientY }
    });
    handleContextMenuClose();
  }, [handleContextMenuClose]);
  
  const handleEditFolderMetadata = useCallback((folder, event) => {
    setFolderMetadataPane({
      open: true,
      folder,
      position: { x: event.clientX, y: event.clientY }
    });
    handleContextMenuClose();
  }, [handleContextMenuClose]);

  const handleDeleteFile = useCallback(() => {
    if (!contextMenuTarget?.document_id) return;
    if (window.confirm('Are you sure you want to delete this file?')) {
      deleteDocumentMutation.mutate(contextMenuTarget.document_id);
    }
  }, [contextMenuTarget, deleteDocumentMutation]);

  const handleRenameFile = useCallback(() => {
    if (!contextMenuTarget?.document_id) return;
    const currentName = contextMenuTarget.filename || contextMenuTarget.title || '';
    const newName = prompt('Enter new filename (include extension if desired)', currentName);
    if (newName && newName.trim() && newName.trim() !== currentName) {
      renameDocumentMutation.mutate({ documentId: contextMenuTarget.document_id, newFilename: newName.trim() });
    }
  }, [contextMenuTarget, renameDocumentMutation]);

  const handleRenameFolder = useCallback(() => {
    if (!contextMenuTarget?.folder_id) return;
    const currentName = contextMenuTarget.name || '';
    const newName = prompt('Enter new folder name:', currentName);
    if (newName && newName.trim() && newName.trim() !== currentName) {
      renameFolderMutation.mutate({ folderId: contextMenuTarget.folder_id, newName: newName.trim() });
    }
  }, [contextMenuTarget, renameFolderMutation]);

  const handleCreateMarkdown = useCallback((folderId) => {
    const input = prompt('Enter file name (optionally with .md or .org extension):');
    if (input) {
      const raw = input.trim();
      if (!raw) { handleContextMenuClose(); return; }
      const hasDot = raw.lastIndexOf('.') > 0;
      const ext = hasDot ? raw.slice(raw.lastIndexOf('.') + 1).toLowerCase() : '';
      let finalFilename = raw;
      let finalDocType = 'md';
      let content;

      if (ext) {
        if (ext === 'md') {
          finalDocType = 'md';
          // Start with empty Markdown body; avoid auto-inserting a title to keep offsets stable
          content = '';
        } else if (ext === 'org') {
          finalDocType = 'org';
          const base = raw.replace(/\.[^.]+$/, '');
          const today = new Date().toISOString().split('T')[0];
          content = `#+TITLE: ${base}\n#+DATE: ${today}\n\n* ${base}\n\n`;
        } else {
          alert('Unsupported extension. Use .md or .org');
          handleContextMenuClose();
          return;
        }
      } else {
        finalFilename = `${raw}.md`;
        finalDocType = 'md';
        // Start with empty Markdown body; avoid auto-inserting a title to keep offsets stable
        content = '';
      }

      apiService.createDocumentFromContent({
        content,
        title: raw.replace(/\.[^.]+$/, ''),
        filename: finalFilename,
        userId: user?.user_id,
        folderId,
        docType: finalDocType
      }).then(() => {
        // Auto-expand the target folder so the user sees the new file
        if (folderId) {
          setExpandedFolders(prev => {
            const next = new Set(prev);
            next.add(folderId);
            try {
              localStorage.setItem('expandedFolders', JSON.stringify(Array.from(next)));
            } catch {}
            return next;
          });
        }
        // Invalidate the precise query key so react-query refetches the tree
        queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
      }).catch((err) => {
        console.error('❌ Failed to create markdown document:', err);
        alert('Failed to create markdown document');
      });
    }
    handleContextMenuClose();
  }, [user, queryClient, handleContextMenuClose]);

  const handleCreateOrgMode = useCallback((folderId) => {
    const input = prompt('Enter file name (optionally with .org or .md extension):');
    if (input) {
      const raw = input.trim();
      if (!raw) { handleContextMenuClose(); return; }
      const hasDot = raw.lastIndexOf('.') > 0;
      const ext = hasDot ? raw.slice(raw.lastIndexOf('.') + 1).toLowerCase() : '';
      let finalFilename = raw;
      let finalDocType = 'org';
      let content;

      if (ext) {
        if (ext === 'org') {
          finalDocType = 'org';
          const base = raw.replace(/\.[^.]+$/, '');
          const today = new Date().toISOString().split('T')[0];
          content = `#+TITLE: ${base}\n#+DATE: ${today}\n\n* ${base}\n\n`;
        } else if (ext === 'md') {
          finalDocType = 'md';
          content = `# ${raw.replace(/\.[^.]+$/, '')}\n\n`;
        } else {
          alert('Unsupported extension. Use .org or .md');
          handleContextMenuClose();
          return;
        }
      } else {
        finalFilename = `${raw}.org`;
        finalDocType = 'org';
        const today = new Date().toISOString().split('T')[0];
        content = `#+TITLE: ${raw}\n#+DATE: ${today}\n\n* ${raw}\n\n`;
      }

      apiService.createDocumentFromContent({
        content,
        title: raw.replace(/\.[^.]+$/, ''),
        filename: finalFilename,
        userId: user?.user_id,
        folderId,
        docType: finalDocType
      }).then(() => {
        // Auto-expand the target folder so the user sees the new file
        if (folderId) {
          setExpandedFolders(prev => {
            const next = new Set(prev);
            next.add(folderId);
            try {
              localStorage.setItem('expandedFolders', JSON.stringify(Array.from(next)));
            } catch {}
            return next;
          });
        }
        // Invalidate the precise query key so react-query refetches the tree
        queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
      }).catch((err) => {
        console.error('❌ Failed to create org document:', err);
        alert('Failed to create org document');
      });
    }
    handleContextMenuClose();
  }, [user, queryClient, handleContextMenuClose]);

  const handleToggleCollapse = useCallback(() => {
    if (typeof onToggleCollapse === 'function') {
      onToggleCollapse(!isCollapsed);
    }
    if (!isControlledCollapse) {
      setInternalCollapsed(!isCollapsed);
    }
  }, [isCollapsed, onToggleCollapse, isControlledCollapse]);



  // RSS handlers
  const handleRSSFeedClick = useCallback((feedId, feedName) => {
    onRSSFeedClick?.(feedId, feedName);
  }, [onRSSFeedClick]);

  const handleRSSFeedToggle = useCallback(() => {
    setRssFeedsExpanded(prev => !prev);
  }, []);

  const handleRefreshRSSFeed = useCallback((feedId) => {
    refreshRSSFeedMutation.mutate(feedId);
  }, [refreshRSSFeedMutation]);

  const handleDeleteRSSFeed = useCallback((feedId, deleteArticles = false) => {
    const message = deleteArticles 
      ? 'Are you sure you want to delete this feed and all its imported articles?' 
      : 'Are you sure you want to delete this feed? (Imported articles will be kept)';
    
    if (window.confirm(message)) {
      deleteRSSFeedMutation.mutate({ feedId, deleteArticles });
    }
  }, [deleteRSSFeedMutation]);

  const handleFileDrop = useCallback((event, targetFolderId) => {
    event.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(event.dataTransfer.files);
    if (files.length > 0) {
      setUploadFiles(files);
      // Set the target folder for the upload
      if (targetFolderId) {
        // Find the folder object from the folder tree
        const findFolder = (folders, folderId) => {
          for (const folder of folders) {
            if (folder.folder_id === folderId) {
              return folder;
            }
            if (folder.children) {
              const found = findFolder(folder.children, folderId);
              if (found) return found;
            }
          }
          return null;
        };
        
        const targetFolder = findFolder(folderTree || [], targetFolderId);
        if (targetFolder) {
          setUploadTargetFolder(targetFolder);
        }
      }
      setUploadDialog(true);
    }
  }, [folderTree]);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Render folder item
  const renderFolderItem = useCallback((folder, level = 0) => {
    const isExpanded = expandedFolders.has(folder.folder_id);
    const isSelected = selectedFolderId === folder.folder_id;
    const hasSubfolders = folder.children && folder.children.length > 0;
    const hasDocuments = folderContents[folder.folder_id]?.documents && folderContents[folder.folder_id].documents.length > 0;
    
    // Check if this is a virtual root folder first
    const isVirtualRoot = folder.folder_id === 'my_documents_root' || folder.folder_id === 'global_documents_root';
    
    // Check if this is an RSS virtual directory and has RSS feeds
    const isRSSVirtual = folder.folder_id === 'rss_feeds_virtual' || folder.folder_id === 'global_rss_feeds_virtual';
    const hasRSSFeeds = isRSSVirtual && rssFeeds && rssFeeds.length > 0;
    
    // For RSS virtual directories, count RSS feeds as children for expand/collapse
    // For real folders, also show expand/collapse if they might have documents (even if not loaded yet)
    const isRealFolder = !isVirtualRoot && !folder.is_virtual_source && !isRSSVirtual;
    const mightHaveDocuments = isRealFolder && (folder.document_count > 0 || folder.subfolder_count > 0);
    const hasChildren = hasSubfolders || hasDocuments || hasRSSFeeds || mightHaveDocuments;

    return (
      <Box key={folder.folder_id}>
        <ListItem
          disablePadding
          data-folder-id={folder.folder_id}
          sx={{
            pl: level * 2,
            backgroundColor: isSelected 
              ? (theme.palette.mode === 'dark' 
                  ? 'rgba(25, 118, 210, 0.4)' 
                  : 'rgba(25, 118, 210, 0.12)')
              : 'transparent',
            '&:hover': {
              backgroundColor: 'action.hover',
            },
            ...(isVirtualRoot && {
              backgroundColor: theme.palette.mode === 'dark' 
                ? 'rgba(25, 118, 210, 0.35)' 
                : 'primary.light',
              '&:hover': {
                backgroundColor: 'primary.main',
              },
              '& .MuiListItemText-primary': {
                fontWeight: 'bold',
                color: 'primary.contrastText',
              },
            }),
          }}
          onContextMenu={(e) => handleContextMenu(e, folder)}
          onDrop={(e) => {
            // Don't allow dropping on virtual roots or virtual sources
            if (folder.folder_id === 'global_documents_root' || folder.folder_id === 'my_documents_root' || folder.is_virtual_source) {
              alert('Cannot drop files directly on root folders or virtual sources. Please drop on specific folders.');
              return;
            }
            handleFileDrop(e, folder.folder_id);
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <ListItemButton
            onClick={() => handleFolderClick(folder.folder_id)}
            sx={{ minHeight: 32 }}
          >
            <ListItemIcon sx={{ minWidth: 36 }}>
              {hasChildren ? (
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleFolderToggle(folder.folder_id);
                  }}
                >
                  {isExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                </IconButton>
              ) : (
                <Box sx={{ width: 24 }} />
              )}
            </ListItemIcon>
            
            <ListItemIcon sx={{ minWidth: 24 }}>
              {folder.folder_id === 'news_virtual' ? (
                <Newspaper color="primary" />
              ) : isVirtualRoot ? (
                isExpanded ? <FolderOpen color="inherit" /> : <Folder color="inherit" />
              ) : folder.is_virtual_source ? (
                isExpanded ? <FolderOpen color="success" /> : <Folder color="success" />
              ) : (
                isExpanded ? <FolderOpen color="primary" /> : <Folder color="primary" />
              )}
            </ListItemIcon>
            
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" noWrap>
                    {folder.name}
                  </Typography>
                  {(() => {
                    const effective = getEffectiveFolderExemption(folder.folder_id);
                    if (effective.status === true) {
                      return (
                        <Tooltip title="Folder is not being vectorized" placement="top">
                          <Box
                            sx={{
                              display: 'inline-flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              width: 16,
                              height: 16,
                              borderRadius: '50%',
                              border: theme.palette.mode === 'dark' 
                                ? '1px solid rgba(255, 255, 255, 0.3)' 
                                : '1px solid rgba(0, 0, 0, 0.23)',
                              backgroundColor: 'transparent'
                            }}
                          >
                            <CheckCircle sx={{ 
                              color: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.38)', 
                              fontSize: 14,
                            }} />
                          </Box>
                        </Tooltip>
                      );
                    }
                    return null;
                  })()}
                  {folder.is_virtual_source && (
                    <Chip
                      size="small"
                      label="Virtual"
                      color="success"
                      variant="outlined"
                      sx={{ height: 16, fontSize: '0.7rem' }}
                    />
                  )}
                  {(folder.document_count > 0 || folder.subfolder_count > 0) && (
                    <Chip
                      size="small"
                      label={`${folder.document_count + folder.subfolder_count}`}
                      variant="outlined"
                      sx={{ height: 16, fontSize: '0.7rem' }}
                    />
                  )}
                </Box>
              }
            />
          </ListItemButton>
        </ListItem>

        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {folder.children.map(child => renderFolderItem(child, level + 1))}
              
              {/* Render RSS feeds for RSS virtual directories */}
              {isRSSVirtual && (
                <>
                  {/* Show user feeds in rss_feeds_virtual, global feeds in global_rss_feeds_virtual */}
                  {(folder.folder_id === 'rss_feeds_virtual' ? userRSSFeeds : globalRSSFeeds).map(feed => {
                    console.log('🔍 Processing RSS feed:', feed);
                    console.log('🔍 Feed ID:', feed?.feed_id);
                    console.log('🔍 Feed name:', feed?.feed_name);
                    console.log('🔍 Feed location:', folder.folder_id === 'rss_feeds_virtual' ? 'User' : 'Global');
                    const unreadCount = rssUnreadCounts?.[feed.feed_id] || 0;
                    return (
                      <ListItem
                        key={feed.feed_id}
                        disablePadding
                        sx={{ pl: (level + 1) * 2 + 2 }}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setContextMenu({
                            mouseX: e.clientX + 2,
                            mouseY: e.clientY - 6,
                          });
                          setContextMenuTarget({ ...feed, type: 'rss_feed' });
                        }}
                      >
                        <ListItemButton
                          onClick={() => handleRSSFeedClick(feed.feed_id, feed.feed_name)}
                          sx={{ minHeight: 30 }}
                        >
                          <ListItemIcon sx={{ minWidth: 24 }}>
                            <RssFeed color="primary" />
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                                  {feed.feed_name}
                                </Typography>
                                {unreadCount > 0 && (
                                  <Chip
                                    size="small"
                                    label={unreadCount}
                                    color="primary"
                                    sx={{ height: 16, fontSize: '0.7rem', minWidth: 20 }}
                                  />
                                )}
                              </Box>
                            }
                            secondary={
                              <Typography variant="caption" color="text.secondary">
                                Last updated: {feed.last_updated ? new Date(feed.last_updated).toLocaleDateString() : 'Never'}
                              </Typography>
                            }
                          />
                        </ListItemButton>
                      </ListItem>
                    );
                  })}
                </>
              )}
            </List>
          </Collapse>
        )}
        
        {/* Show documents in folder when expanded */}
        {isExpanded && !isVirtualRoot && folderContents[folder.folder_id]?.documents && folderContents[folder.folder_id].documents.length > 0 && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {folderContents[folder.folder_id].documents.map(doc => renderFileItem(doc, level + 1))}
            </List>
          </Collapse>
        )}
        

      </Box>
    );
  }, [
    expandedFolders,
    selectedFolderId,
    handleFolderToggle,
    handleFolderClick,
    handleContextMenu,
    handleFileDrop,
    handleDragOver,
    handleDragLeave,
    folderContents,
    rssFeeds,
    rssUnreadCounts,
    handleRSSFeedClick,
    user,
    getEffectiveFolderExemption,
    theme
  ]);

  // Render file item
  const renderFileItem = useCallback((file, level = 0) => {
    const isSelected = false; // TODO: Implement file selection
    
    const getFileIcon = (filename, status) => {
      const ext = filename.split('.').pop()?.toLowerCase();
      let baseIcon;
      switch (ext) {
        case 'md':
          baseIcon = <Description color="action" />;
          break;
        case 'org':
          baseIcon = <Article color="action" />;
          break;
        case 'mp3':
        case 'aac':
        case 'wav':
        case 'flac':
        case 'ogg':
        case 'm4a':
        case 'wma':
        case 'opus':
          baseIcon = <Audiotrack color="action" />;
          break;
        default:
          baseIcon = <InsertDriveFile color="action" />;
      }
      return baseIcon;
    };

    const getStatusIcon = (status, isExemptFromVectorization = false) => {
      const getStatusIconWithTooltip = (icon, title) => (
        <Tooltip title={title} placement="top">
          {icon}
        </Tooltip>
      );

      // If exempt from vectorization and file is completed/okay, show white CheckCircle
      if (isExemptFromVectorization && (status === 'completed' || !status || status === 'pending')) {
        return getStatusIconWithTooltip(
          <Box
            sx={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: 16,
              height: 16,
              borderRadius: '50%',
              border: theme.palette.mode === 'dark' 
                ? '1px solid rgba(255, 255, 255, 0.3)' 
                : '1px solid rgba(0, 0, 0, 0.23)',
              backgroundColor: 'transparent'
            }}
          >
            <CheckCircle sx={{ 
              color: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.7)' : 'rgba(0, 0, 0, 0.38)', 
              fontSize: 14,
            }} />
          </Box>,
          'File is ready but not being vectorized'
        );
      }

      switch (status) {
        case 'completed':
          return getStatusIconWithTooltip(
            <CheckCircle sx={{ color: '#4caf50', fontSize: 16 }} />,
            'Processing complete - ready to search'
          );
        case 'processing':
          return getStatusIconWithTooltip(
            <CloudSync sx={{ color: '#ff9800', fontSize: 16 }} />,
            'Processing document...'
          );
        case 'embedding':
          return getStatusIconWithTooltip(
            <CloudSync sx={{ color: '#ff9800', fontSize: 16 }} />,
            'Creating embeddings...'
          );
        case 'uploading':
          return getStatusIconWithTooltip(
            <HourglassEmpty sx={{ color: '#2196f3', fontSize: 16 }} />,
            'Uploading...'
          );
        case 'failed':
          return getStatusIconWithTooltip(
            <Error sx={{ color: '#f44336', fontSize: 16 }} />,
            'Processing failed'
          );
        default:
          return null;
      }
    };

    return (
      <ListItem
        key={file.document_id}
        disablePadding
        sx={{
          pl: level * 2 + 4, // ROOSEVELT: Files get extra indentation beyond their folder level
          backgroundColor: isSelected 
            ? (theme.palette.mode === 'dark' 
                ? 'rgba(25, 118, 210, 0.4)' 
                : 'rgba(25, 118, 210, 0.12)')
            : 'transparent',
          '&:hover': {
            backgroundColor: 'action.hover',
          },
        }}
        onContextMenu={(e) => handleContextMenu(e, file)}
      >
        <ListItemButton
          onClick={() => onFileSelect?.(file)}
          sx={{ minHeight: 30 }}
        >
          <ListItemIcon sx={{ minWidth: 24 }}>
            {getFileIcon(file.filename, file.status)}
          </ListItemIcon>
          
          <ListItemText
            primary={
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                  {file.filename}
                </Typography>
                {(() => {
                  const effective = getEffectiveFileExemption(file);
                  return getStatusIcon(file.status || file.processing_status, effective.status === true);
                })()}
              </Box>
            }
          />
        </ListItemButton>
      </ListItem>
    );
  }, [getEffectiveFileExemption, handleContextMenu, onFileSelect, theme]);

  // Render destination item for Move dialog
  const renderDestinationItem = useCallback((folder, level = 0) => {
    const disabled = folder.folder_id.includes('_root') || folder.is_virtual_source;
    const isSelected = moveDestinationId === folder.folder_id;
    return (
      <>
        <ListItem
          key={`dest-${folder.folder_id}`}
          disablePadding
          sx={{ pl: level * 2 }}
        >
          <ListItemButton
            disabled={disabled}
            selected={isSelected}
            onClick={() => setMoveDestinationId(folder.folder_id)}
            sx={{ minHeight: 28 }}
          >
            <ListItemIcon sx={{ minWidth: 24 }}>
              <Folder fontSize="small" />
            </ListItemIcon>
            <ListItemText primary={<Typography variant="body2" noWrap>{folder.name}</Typography>} />
          </ListItemButton>
        </ListItem>
        {Array.isArray(folder.children) && folder.children.length > 0 && (
          <>{folder.children.map(child => renderDestinationItem(child, level + 1))}</>
        )}
      </>
    );
  }, [moveDestinationId]);

  if (isLoading) {
    return (
      <Box sx={{ width, p: 2, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ width, p: 2 }}>
        <Alert severity="error">
          Failed to load folder tree
        </Alert>
      </Box>
    );
  }

  // Handle collapsed state
  if (isCollapsed) {
    return (
      <Box
        sx={{
          width: '48px',
          borderRight: 1,
          borderColor: 'divider',
          backgroundColor: 'background.paper',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          height: '100%',
        }}
      >
        <IconButton onClick={handleToggleCollapse} size="small" sx={{ mt: 1 }}>
          <ChevronRight />
        </IconButton>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        width,
        borderRight: 1,
        borderColor: 'divider',
        backgroundColor: 'background.paper',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="h6" sx={{ fontWeight: 600, fontSize: 'var(--font-size-lg)' }}>
            Documents
          </Typography>
          <Box sx={{ display: 'flex', gap: 0.5 }}>
            <Tooltip title="Refresh File List">
              <IconButton size="small" onClick={() => refetch()}>
                <Refresh fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Rescan & Recover Orphaned Files">
              <IconButton 
                size="small" 
                onClick={async () => {
                  try {
                    const confirmed = window.confirm(
                      'Scan filesystem for files not in database?\n\n' +
                      'This will:\n' +
                      '• Find files that exist on disk but not in the database\n' +
                      '• Re-add them without re-vectorizing if vectors exist\n' +
                      '• Create "Recovered Files" folder for orphans\n\n' +
                      'Continue?'
                    );
                    if (!confirmed) return;
                    
                    console.log('🔍 Starting file rescan...');
                    const response = await apiService.post('/api/user/documents/rescan');
                    
                    if (response.success) {
                      alert(
                        `File Recovery Complete!\n\n` +
                        `Recovered: ${response.recovered_count} files\n` +
                        `Skipped (already in DB): ${response.skipped_count}\n` +
                        `Errors: ${response.error_count}`
                      );
                      refetch(); // Refresh the file tree
                    } else {
                      alert(`Recovery failed: ${response.error}`);
                    }
                  } catch (error) {
                    console.error('❌ Rescan failed:', error);
                    alert(`Recovery failed: ${error.message}`);
                  }
                }}
              >
                <FindInPage fontSize="small" />
              </IconButton>
            </Tooltip>
            <Tooltip title="Collapse Documents">
              <IconButton onClick={handleToggleCollapse} size="small">
                <ChevronLeft />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </Box>

      {/* Folder Tree */}
      <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
        {folderTree?.folders?.length > 0 ? (
          <List dense>
            {/* Inject a virtual News node at the top */}
            <ListItem disablePadding onContextMenu={(e) => e.preventDefault()}>
              <ListItemButton onClick={() => handleFolderClick('news_virtual')} sx={{ minHeight: 32 }}>
                <ListItemIcon sx={{ minWidth: 24 }}>
                  <Newspaper color="primary" />
                </ListItemIcon>
                <ListItemText primary={<Typography variant="body2" noWrap>News</Typography>} />
              </ListItemButton>
            </ListItem>
            <AnimatePresence>
              {folderTree.folders.map(folder => (
                <motion.div
                  key={folder.folder_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.2 }}
                >
                  {renderFolderItem(folder)}
                </motion.div>
              ))}
            </AnimatePresence>
          </List>
        ) : (
          // Empty state - show message and create default folders button
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              No folders found
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 2 }}>
              Create default folders to get started
            </Typography>
            <Button
              variant="outlined"
              size="small"
              onClick={handleCreateDefaultFolders}
              disabled={createDefaultFoldersMutation.isLoading}
              startIcon={<CreateNewFolder />}
            >
              {createDefaultFoldersMutation.isLoading ? 'Creating...' : 'Create Default Folders'}
            </Button>
          </Box>
        )}

        {/* ROOSEVELT'S ORG TOOLS SECTION */}
        {hasOrgFiles && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box sx={{ px: 2 }}>
              <Box 
                sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 1, 
                  cursor: 'pointer',
                  '&:hover': { backgroundColor: 'action.hover' },
                  borderRadius: 1,
                  p: 0.5
                }}
                onClick={() => setOrgToolsExpanded(!orgToolsExpanded)}
              >
                {orgToolsExpanded ? <ExpandLess fontSize="small" /> : <ExpandMore fontSize="small" />}
                <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: '0.75rem', textTransform: 'uppercase', ml: 0.5 }}>
                  🔧 Org Tools
                </Typography>
              </Box>
              <Collapse in={orgToolsExpanded} timeout="auto">
                <List dense>
            <ListItem disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton 
                onClick={() => {
                  // Open org-agenda tab
                  if (window.tabbedContentManagerRef?.openOrgView) {
                    window.tabbedContentManagerRef.openOrgView('agenda');
                  }
                }}
                sx={{ 
                  borderRadius: 1,
                  '&:hover': { backgroundColor: 'action.hover' }
                }}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Typography sx={{ fontSize: '18px' }}>📅</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography variant="body2">Agenda View</Typography>}
                  secondary={<Typography variant="caption" color="text.secondary">Scheduled & deadlines</Typography>}
                />
              </ListItemButton>
            </ListItem>
            
            <ListItem disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton 
                onClick={() => {
                  if (window.tabbedContentManagerRef?.openOrgView) {
                    window.tabbedContentManagerRef.openOrgView('search');
                  }
                }}
                sx={{ 
                  borderRadius: 1,
                  '&:hover': { backgroundColor: 'action.hover' }
                }}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Typography sx={{ fontSize: '18px' }}>🔍</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography variant="body2">Search Org Files</Typography>}
                  secondary={<Typography variant="caption" color="text.secondary">Full-text search</Typography>}
                />
              </ListItemButton>
            </ListItem>
            
            <ListItem disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton 
                onClick={() => {
                  if (window.tabbedContentManagerRef?.openOrgView) {
                    window.tabbedContentManagerRef.openOrgView('todos');
                  }
                }}
                sx={{ 
                  borderRadius: 1,
                  '&:hover': { backgroundColor: 'action.hover' }
                }}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Typography sx={{ fontSize: '18px' }}>✅</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography variant="body2">All TODOs</Typography>}
                  secondary={<Typography variant="caption" color="text.secondary">Across all files</Typography>}
                />
              </ListItemButton>
            </ListItem>
            
            <ListItem disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton 
                onClick={() => {
                  if (window.tabbedContentManagerRef?.openOrgView) {
                    window.tabbedContentManagerRef.openOrgView('contacts');
                  }
                }}
                sx={{ 
                  borderRadius: 1,
                  '&:hover': { backgroundColor: 'action.hover' }
                }}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Typography sx={{ fontSize: '18px' }}>👤</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography variant="body2">Contacts</Typography>}
                  secondary={<Typography variant="caption" color="text.secondary">View all contacts</Typography>}
                />
              </ListItemButton>
            </ListItem>
            
            <ListItem disablePadding>
              <ListItemButton 
                onClick={() => {
                  if (window.tabbedContentManagerRef?.openOrgView) {
                    window.tabbedContentManagerRef.openOrgView('tags');
                  }
                }}
                sx={{ 
                  borderRadius: 1,
                  '&:hover': { backgroundColor: 'action.hover' }
                }}
              >
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Typography sx={{ fontSize: '18px' }}>🏷️</Typography>
                </ListItemIcon>
                <ListItemText 
                  primary={<Typography variant="body2">Tags Browser</Typography>}
                  secondary={<Typography variant="caption" color="text.secondary">Browse by tags</Typography>}
                />
              </ListItemButton>
            </ListItem>
                </List>
              </Collapse>
            </Box>
          </>
        )}

        {/* DATA WORKSPACES SECTION */}
        <DataWorkspacesSection onWorkspaceClick={(workspace) => {
          if (window.tabbedContentManagerRef?.openDataWorkspace) {
            window.tabbedContentManagerRef.openDataWorkspace(workspace.workspace_id);
          }
        }} />

      </Box>

      {/* Context Menu */}
      <Menu
        open={Boolean(contextMenu)}
        onClose={handleContextMenuClose}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        {contextMenuTarget?.folder_id && !contextMenuTarget?.document_id && (
          <>
            {/* Show different options for virtual roots vs regular folders */}
            {(contextMenuTarget.folder_id === 'global_documents_root' || contextMenuTarget.folder_id === 'my_documents_root') ? (
              // Virtual root options
              <>
                {contextMenuTarget.folder_id === 'global_documents_root' && user?.role === 'admin' && (
                  <MenuItem
                    onClick={() => {
                      setNewFolderParent(null); // No parent for root-level global folders
                      setNewFolderCollectionType('global');
                      setCreateFolderDialog(true);
                      handleContextMenuClose();
                    }}
                  >
                    <ListItemIcon>
                      <CreateNewFolder fontSize="small" />
                    </ListItemIcon>
                    New Global Folder
                  </MenuItem>
                )}
                {contextMenuTarget.folder_id === 'my_documents_root' && (
                  <MenuItem
                    onClick={() => {
                      setNewFolderParent(null); // No parent for root-level user folders
                      setNewFolderCollectionType('user');
                      setCreateFolderDialog(true);
                      handleContextMenuClose();
                    }}
                  >
                    <ListItemIcon>
                      <CreateNewFolder fontSize="small" />
                    </ListItemIcon>
                    New User Folder
                  </MenuItem>
                )}
              </>
            ) : (
              // Regular folder options
              <MenuItem
                onClick={() => {
                  setNewFolderParent(contextMenuTarget.folder_id);
                  // Auto-set collection type based on parent
                  setNewFolderCollectionType(contextMenuTarget.collection_type || 'user');
                  setCreateFolderDialog(true);
                  handleContextMenuClose();
                }}
              >
                <ListItemIcon>
                  <CreateNewFolder fontSize="small" />
                </ListItemIcon>
                New Folder
                {contextMenuTarget.collection_type === 'global' && (
                  <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                    (Global)
                  </Typography>
                )}
              </MenuItem>
            )}
            
            <MenuItem
              onClick={() => {
                setProjectParentFolder(contextMenuTarget.folder_id);
                setNewProjectDialog(true);
                handleContextMenuClose();
              }}
            >
              <ListItemIcon>
                <Add fontSize="small" />
              </ListItemIcon>
              New Project
            </MenuItem>
            
            <MenuItem
              onClick={() => {
                // Don't allow uploading to virtual roots
                if (contextMenuTarget.folder_id === 'global_documents_root' || contextMenuTarget.folder_id === 'my_documents_root') {
                  alert('Cannot upload files directly to root folders. Please upload to specific folders.');
                  handleContextMenuClose();
                  return;
                }
                
                setUploadTargetFolder(contextMenuTarget);
                setUploadDialog(true);
                handleContextMenuClose();
              }}
            >
              <ListItemIcon>
                <Upload fontSize="small" />
              </ListItemIcon>
              Upload Files
            </MenuItem>
            
            {/* ROOSEVELT FOLDER TAGGING - Edit Metadata for non-virtual folders */}
            {!contextMenuTarget.folder_id.includes('_root') && !contextMenuTarget.is_virtual_source && (
              <MenuItem
                onClick={(e) => handleEditFolderMetadata(contextMenuTarget, e)}
              >
                <ListItemIcon>
                  <Edit fontSize="small" />
                </ListItemIcon>
                Edit Folder Metadata
              </MenuItem>
            )}
            
            <MenuItem
              onClick={() => handleCreateMarkdown(contextMenuTarget.folder_id)}
            >
              <ListItemIcon>
                <Description fontSize="small" />
              </ListItemIcon>
              New Markdown
            </MenuItem>
            
            <MenuItem
              onClick={() => handleCreateOrgMode(contextMenuTarget.folder_id)}
            >
              <ListItemIcon>
                <Article fontSize="small" />
              </ListItemIcon>
              New Org-Mode
            </MenuItem>
            
            {/* RSS-specific options for RSS virtual directories */}
            {(contextMenuTarget.folder_id === 'rss_feeds_virtual' || contextMenuTarget.folder_id === 'global_rss_feeds_virtual') && (
              <MenuItem
                onClick={() => {
                  // Pass the context menu target to determine scope
                  const isGlobal = contextMenuTarget.folder_id === 'global_rss_feeds_virtual';
                  onAddRSSFeed({ isGlobal, folderContext: contextMenuTarget });
                  handleContextMenuClose();
                }}
              >
                <ListItemIcon>
                  <RssFeed fontSize="small" />
                </ListItemIcon>
                Add RSS Feed
              </MenuItem>
            )}
            
            {/* Only show rename/move/delete for non-virtual folders */}
            {!contextMenuTarget.folder_id.includes('_root') && !contextMenuTarget.is_virtual_source && (
              <>
                <Divider />
                
                <MenuItem
                  onClick={handleRenameFolder}
                >
                  <ListItemIcon>
                    <Edit fontSize="small" />
                  </ListItemIcon>
                  Rename
                </MenuItem>
                
                <MenuItem
                  onClick={() => {
                    setMoveTarget({ type: 'folder', data: contextMenuTarget });
                    setMoveDialogOpen(true);
                    handleContextMenuClose();
                  }}
                >
                  <ListItemIcon>
                    <DriveFileMove fontSize="small" />
                  </ListItemIcon>
                  Move
                </MenuItem>
                
                <Divider />
                
                {/* Vectorization Settings - Submenu */}
                <MenuItem
                  onMouseEnter={(e) => {
                    setVectorizationSubmenu(e.currentTarget);
                  }}
                >
                  <ListItemIcon>
                    <FindInPage fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Vectorization" />
                  <ChevronRight fontSize="small" />
                </MenuItem>
                
                <Menu
                  anchorEl={vectorizationSubmenu}
                  open={Boolean(vectorizationSubmenu)}
                  onClose={() => setVectorizationSubmenu(null)}
                  anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
                  transformOrigin={{ vertical: 'top', horizontal: 'left' }}
                  MenuListProps={{ 
                    onMouseLeave: () => setVectorizationSubmenu(null),
                    onMouseEnter: () => {
                      // Keep menu open when mouse enters submenu
                    }
                  }}
                >
                  {/* Option 1: Don't vectorize (exempt) */}
                  <MenuItem
                    onClick={async () => {
                      try {
                        await apiService.exemptFolder(contextMenuTarget.folder_id);
                        alert('Folder set to not vectorize. All files in this folder and subfolders will be skipped.');
                        setVectorizationSubmenu(null);
                        handleContextMenuClose();
                        queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
                        queryClient.invalidateQueries(['folders', 'contents']);
                        const expandedArray = Array.from(expandedFolders);
                        for (const folderId of expandedArray) {
                          try {
                            const contents = await apiService.getFolderContents(folderId);
                            setFolderContents(prev => ({ ...prev, [folderId]: contents }));
                          } catch (err) {
                            console.error(`Failed to refresh folder contents for ${folderId}:`, err);
                          }
                        }
                        if (onFolderSelect) {
                          onFolderSelect(selectedFolderId);
                        }
                      } catch (error) {
                        alert(`Failed to update setting: ${error.message}`);
                      }
                    }}
                    selected={contextMenuTarget.exempt_from_vectorization === true}
                  >
                    <ListItemIcon>
                      {contextMenuTarget.exempt_from_vectorization === true ? (
                        <CheckCircle fontSize="small" color="primary" />
                      ) : (
                        <Block fontSize="small" />
                      )}
                    </ListItemIcon>
                    <ListItemText 
                      primary="Don't vectorize"
                      secondary="Files won't be indexed"
                    />
                  </MenuItem>
                  
                  {/* Option 2: Vectorize (override parent) */}
                  <MenuItem
                    onClick={async () => {
                      try {
                        await apiService.overrideFolderExemption(contextMenuTarget.folder_id);
                        alert('Folder set to vectorize. Files will be indexed even if parent folder is exempt.');
                        setVectorizationSubmenu(null);
                        handleContextMenuClose();
                        queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
                        queryClient.invalidateQueries(['folders', 'contents']);
                        const expandedArray = Array.from(expandedFolders);
                        for (const folderId of expandedArray) {
                          try {
                            const contents = await apiService.getFolderContents(folderId);
                            setFolderContents(prev => ({ ...prev, [folderId]: contents }));
                          } catch (err) {
                            console.error(`Failed to refresh folder contents for ${folderId}:`, err);
                          }
                        }
                        if (onFolderSelect) {
                          onFolderSelect(selectedFolderId);
                        }
                      } catch (error) {
                        alert(`Failed to update setting: ${error.message}`);
                      }
                    }}
                    selected={contextMenuTarget.exempt_from_vectorization === false}
                  >
                    <ListItemIcon>
                      {contextMenuTarget.exempt_from_vectorization === false ? (
                        <CheckCircle fontSize="small" color="primary" />
                      ) : (
                        <FindInPage fontSize="small" />
                      )}
                    </ListItemIcon>
                    <ListItemText 
                      primary="Vectorize"
                      secondary="Files will be indexed"
                    />
                  </MenuItem>
                  
                  {/* Option 3: Use parent setting (inherit) */}
                  <MenuItem
                    onClick={async () => {
                      try {
                        await apiService.removeFolderExemption(contextMenuTarget.folder_id);
                        alert('Folder will use parent folder setting.');
                        setVectorizationSubmenu(null);
                        handleContextMenuClose();
                        queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
                        queryClient.invalidateQueries(['folders', 'contents']);
                        const expandedArray = Array.from(expandedFolders);
                        for (const folderId of expandedArray) {
                          try {
                            const contents = await apiService.getFolderContents(folderId);
                            setFolderContents(prev => ({ ...prev, [folderId]: contents }));
                          } catch (err) {
                            console.error(`Failed to refresh folder contents for ${folderId}:`, err);
                          }
                        }
                        if (onFolderSelect) {
                          onFolderSelect(selectedFolderId);
                        }
                      } catch (error) {
                        alert(`Failed to update setting: ${error.message}`);
                      }
                    }}
                    selected={contextMenuTarget.exempt_from_vectorization === null || contextMenuTarget.exempt_from_vectorization === undefined}
                  >
                    <ListItemIcon>
                      {(contextMenuTarget.exempt_from_vectorization === null || contextMenuTarget.exempt_from_vectorization === undefined) ? (
                        <CheckCircle fontSize="small" color="primary" />
                      ) : (
                        <Folder fontSize="small" />
                      )}
                    </ListItemIcon>
                    <ListItemText 
                      primary="Use parent setting"
                      secondary="Follows parent folder"
                    />
                  </MenuItem>
                </Menu>
                
                <Divider />
                
                {/* Hide delete option for team root folders */}
                {!(contextMenuTarget.collection_type === 'team' && !contextMenuTarget.parent_folder_id) && (
                  <MenuItem
                    onClick={handleDeleteFolder}
                    sx={{ color: 'error.main' }}
                  >
                    <ListItemIcon>
                      <Delete fontSize="small" color="error" />
                    </ListItemIcon>
                    Delete
                  </MenuItem>
                )}
              </>
            )}
          </>
        )}

                 {contextMenuTarget?.document_id && (
           <>
             <MenuItem
               onClick={() => {
                 handleReprocessDocument(contextMenuTarget.document_id);
                 handleContextMenuClose();
               }}
             >
               <ListItemIcon>
                 <Refresh fontSize="small" />
               </ListItemIcon>
               Re-process Document
             </MenuItem>
             
             <MenuItem
               onClick={(e) => handleEditMetadata(contextMenuTarget, e)}
             >
               <ListItemIcon>
                 <Edit fontSize="small" />
               </ListItemIcon>
               Edit Metadata
             </MenuItem>
            
            <Divider />
            
            <MenuItem
              onClick={handleRenameFile}
            >
              <ListItemIcon>
                <Edit fontSize="small" />
              </ListItemIcon>
              Rename
            </MenuItem>
            
            <MenuItem
              onClick={() => {
                setMoveTarget({ type: 'file', data: contextMenuTarget });
                setMoveDialogOpen(true);
                handleContextMenuClose();
              }}
            >
              <ListItemIcon>
                <DriveFileMove fontSize="small" />
              </ListItemIcon>
              Move
            </MenuItem>
            
            <Divider />
            
            {/* Vectorization Settings - Submenu for Documents */}
            <MenuItem
              onMouseEnter={(e) => {
                setVectorizationSubmenu(e.currentTarget);
              }}
            >
              <ListItemIcon>
                <FindInPage fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Vectorization" />
              <ChevronRight fontSize="small" />
            </MenuItem>
            
            <Menu
              anchorEl={vectorizationSubmenu}
              open={Boolean(vectorizationSubmenu) && contextMenuTarget?.document_id}
              onClose={() => setVectorizationSubmenu(null)}
              anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
              transformOrigin={{ vertical: 'top', horizontal: 'left' }}
              MenuListProps={{ onMouseLeave: () => setVectorizationSubmenu(null) }}
            >
              {/* Option 1: Don't vectorize (exempt) */}
              <MenuItem
                onClick={async () => {
                  try {
                    await apiService.exemptDocument(contextMenuTarget.document_id);
                    alert('Document set to not vectorize. This file will not be indexed for search.');
                    setVectorizationSubmenu(null);
                    handleContextMenuClose();
                    const folderId = contextMenuTarget.folder_id || 
                      Object.keys(folderContents).find(fId => 
                        folderContents[fId]?.documents?.some(d => d.document_id === contextMenuTarget.document_id)
                      );
                    if (folderId) {
                      queryClient.invalidateQueries(['folders', 'contents', folderId]);
                      try {
                        const contents = await apiService.getFolderContents(folderId);
                        setFolderContents(prev => ({ ...prev, [folderId]: contents }));
                      } catch (err) {
                        console.error('Failed to refresh folder contents:', err);
                      }
                    } else {
                      queryClient.invalidateQueries(['folders', 'contents']);
                    }
                  } catch (error) {
                    alert(`Failed to update setting: ${error.message}`);
                  }
                }}
                selected={contextMenuTarget.exempt_from_vectorization === true}
              >
                <ListItemIcon>
                  {contextMenuTarget.exempt_from_vectorization === true ? (
                    <CheckCircle fontSize="small" color="primary" />
                  ) : (
                    <Block fontSize="small" />
                  )}
                </ListItemIcon>
                <ListItemText 
                  primary="Don't vectorize"
                  secondary="File won't be indexed"
                />
              </MenuItem>
              
              {/* Option 2: Vectorize (explicit) */}
              <MenuItem
                onClick={async () => {
                  try {
                    await apiService.removeDocumentExemption(contextMenuTarget.document_id, false);
                    alert('Document set to vectorize. This file will be indexed for search.');
                    setVectorizationSubmenu(null);
                    handleContextMenuClose();
                    const folderId = contextMenuTarget.folder_id || 
                      Object.keys(folderContents).find(fId => 
                        folderContents[fId]?.documents?.some(d => d.document_id === contextMenuTarget.document_id)
                      );
                    if (folderId) {
                      queryClient.invalidateQueries(['folders', 'contents', folderId]);
                      try {
                        const contents = await apiService.getFolderContents(folderId);
                        setFolderContents(prev => ({ ...prev, [folderId]: contents }));
                      } catch (err) {
                        console.error('Failed to refresh folder contents:', err);
                      }
                    } else {
                      queryClient.invalidateQueries(['folders', 'contents']);
                    }
                  } catch (error) {
                    alert(`Failed to update setting: ${error.message}`);
                  }
                }}
                selected={contextMenuTarget.exempt_from_vectorization === false}
              >
                <ListItemIcon>
                  {contextMenuTarget.exempt_from_vectorization === false ? (
                    <CheckCircle fontSize="small" color="primary" />
                  ) : (
                    <FindInPage fontSize="small" />
                  )}
                </ListItemIcon>
                <ListItemText 
                  primary="Vectorize"
                  secondary="File will be indexed"
                />
              </MenuItem>
              
              {/* Option 3: Use folder setting (inherit) */}
              <MenuItem
                onClick={async () => {
                  try {
                    await apiService.removeDocumentExemption(contextMenuTarget.document_id, true);
                    alert('Document will use folder setting.');
                    setVectorizationSubmenu(null);
                    handleContextMenuClose();
                    const folderId = contextMenuTarget.folder_id || 
                      Object.keys(folderContents).find(fId => 
                        folderContents[fId]?.documents?.some(d => d.document_id === contextMenuTarget.document_id)
                      );
                    if (folderId) {
                      queryClient.invalidateQueries(['folders', 'contents', folderId]);
                      try {
                        const contents = await apiService.getFolderContents(folderId);
                        setFolderContents(prev => ({ ...prev, [folderId]: contents }));
                      } catch (err) {
                        console.error('Failed to refresh folder contents:', err);
                      }
                    } else {
                      queryClient.invalidateQueries(['folders', 'contents']);
                    }
                  } catch (error) {
                    alert(`Failed to update setting: ${error.message}`);
                  }
                }}
                selected={contextMenuTarget.exempt_from_vectorization === null || contextMenuTarget.exempt_from_vectorization === undefined}
              >
                <ListItemIcon>
                  {(contextMenuTarget.exempt_from_vectorization === null || contextMenuTarget.exempt_from_vectorization === undefined) ? (
                    <CheckCircle fontSize="small" color="primary" />
                  ) : (
                    <Folder fontSize="small" />
                  )}
                </ListItemIcon>
                <ListItemText 
                  primary="Use folder setting"
                  secondary="Follows folder setting"
                />
              </MenuItem>
            </Menu>
            
            <Divider />
            
            <MenuItem
              onClick={handleDeleteFile}
              sx={{ color: 'error.main' }}
            >
              <ListItemIcon>
                <Delete fontSize="small" color="error" />
              </ListItemIcon>
              Delete
            </MenuItem>
          </>
        )}

        {contextMenuTarget?.type === 'rss_feed' && (
          <>
            <MenuItem
              onClick={() => {
                handleRefreshRSSFeed(contextMenuTarget.feed_id);
                handleContextMenuClose();
              }}
              disabled={refreshRSSFeedMutation.isLoading}
            >
              <ListItemIcon>
                <Refresh fontSize="small" />
              </ListItemIcon>
              Refresh Feed
            </MenuItem>
            
            <Divider />
            
            <MenuItem
              onClick={() => {
                handleDeleteRSSFeed(contextMenuTarget.feed_id, false);
                handleContextMenuClose();
              }}
              sx={{ color: 'warning.main' }}
            >
              <ListItemIcon>
                <Delete fontSize="small" color="warning" />
              </ListItemIcon>
              Delete Feed Only
            </MenuItem>
            
            <MenuItem
              onClick={() => {
                handleDeleteRSSFeed(contextMenuTarget.feed_id, true);
                handleContextMenuClose();
              }}
              sx={{ color: 'error.main' }}
            >
              <ListItemIcon>
                <Delete fontSize="small" color="error" />
              </ListItemIcon>
              Delete Feed & Articles
            </MenuItem>
          </>
        )}
      </Menu>

      {/* Create Folder Dialog */}
      <Dialog open={createFolderDialog} onClose={() => setCreateFolderDialog(false)}>
        <DialogTitle>Create New Folder</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {newFolderParent ? 
                `Create folder in: ${contextMenuTarget?.name || 'Selected folder'}` :
                newFolderCollectionType === 'global' ? 
                  'Create folder in: Global Documents (root level)' :
                  'Create folder in: My Documents (root level)'
              }
            </Typography>
            {newFolderCollectionType === 'global' && (
              <Typography variant="caption" color="info.main" sx={{ display: 'block', mt: 1 }}>
                ⚠️ Global folders are shared with all users
              </Typography>
            )}
          </Box>
          
          <TextField
            autoFocus
            margin="dense"
            label="Folder Name"
            fullWidth
            variant="outlined"
            value={newFolderName}
            onChange={(e) => setNewFolderName(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleCreateFolder()}
            sx={{ mb: 2 }}
          />
          
          {user?.role === 'admin' && (
            <FormControl fullWidth margin="dense">
              <InputLabel>Collection Type</InputLabel>
              <Select
                value={newFolderCollectionType}
                onChange={(e) => setNewFolderCollectionType(e.target.value)}
                label="Collection Type"
                disabled={newFolderParent !== null} // Disable when creating under a specific folder
              >
                <MenuItem value="user">User Folder (Private)</MenuItem>
                <MenuItem value="global">Global Folder (Shared)</MenuItem>
              </Select>
              {newFolderParent !== null && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Collection type is determined by the parent folder
                </Typography>
              )}
            </FormControl>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateFolderDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateFolder}
            disabled={!newFolderName.trim() || createFolderMutation.isLoading}
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* New Project Dialog */}
      <Dialog open={newProjectDialog} onClose={() => setNewProjectDialog(false)}>
        <DialogTitle>Create New Project</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {projectParentFolder ? 
                `Create project in: ${contextMenuTarget?.name || 'Selected folder'}` :
                'Create project at root level'
              }
            </Typography>
          </Box>
          
          <TextField
            autoFocus
            margin="dense"
            label="Project Name"
            fullWidth
            variant="outlined"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleCreateProject()}
            sx={{ mb: 2 }}
          />
          
          <FormControl fullWidth margin="dense">
            <InputLabel>Project Type</InputLabel>
            <Select
              value={projectType}
              onChange={(e) => setProjectType(e.target.value)}
              label="Project Type"
            >
              <MenuItem value="electronics">Electronics</MenuItem>
              <MenuItem value="general">General</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setNewProjectDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleCreateProject}
            disabled={!projectName.trim() || createProjectMutation.isLoading}
            variant="contained"
          >
            Create
          </Button>
        </DialogActions>
      </Dialog>

      {/* Move Dialog */}
      <Dialog open={moveDialogOpen} onClose={() => setMoveDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Move {moveTarget?.type === 'folder' ? 'Folder' : 'File'}</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Select a destination folder:
          </Typography>
          <Box sx={{ maxHeight: 360, overflow: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
            <List dense>
              {/* Simple destination picker using current tree */}
              {(folderTree?.folders || []).map(root => (
                <React.Fragment key={`root-${root.folder_id}`}>
                  {renderDestinationItem(root, 0)}
                </React.Fragment>
              ))}
            </List>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMoveDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={async () => {
              try {
                if (!moveDestinationId) return;
                if (!moveTarget) return;
                if (moveTarget.type === 'folder') {
                  await apiService.moveFolder(moveTarget.data.folder_id, moveDestinationId);
                  showToast('📁 Folder moved', 'success');
                  // Optimistically update tree immediately
                  queryClient.setQueryData(['folders', 'tree', user?.user_id, user?.role], (oldData) => {
                    if (!oldData) return oldData;
                    const movedFolder = moveTarget.data;
                    const newParentId = moveDestinationId;
                    const removeFromParent = (folders) => folders.map(f => ({
                      ...f,
                      children: (f.children ? removeFromParent(f.children) : f.children)
                    })).filter(f => f.folder_id !== movedFolder.folder_id);
                    const addToNewParent = (folders) => folders.map(f => {
                      if (f.folder_id === newParentId) {
                        return {
                          ...f,
                          children: [...(f.children || []), {
                            folder_id: movedFolder.folder_id,
                            name: movedFolder.name,
                            parent_folder_id: newParentId,
                            user_id: movedFolder.user_id,
                            collection_type: movedFolder.collection_type,
                            created_at: movedFolder.created_at,
                            updated_at: movedFolder.updated_at,
                            document_count: movedFolder.document_count || 0,
                            subfolder_count: movedFolder.subfolder_count || 0,
                            children: []
                          }]
                        };
                      } else if (f.children && f.children.length > 0) {
                        return { ...f, children: addToNewParent(f.children) };
                      }
                      return f;
                    });
                    let folders = removeFromParent(oldData.folders);
                    folders = addToNewParent(folders);
                    return { ...oldData, folders };
                  });
                  // Expand destination and refresh tree for consistency
                  setExpandedFolders(prev => new Set(prev).add(moveDestinationId));
                  queryClient.invalidateQueries(['folders', 'tree', user?.user_id, user?.role]);
                } else if (moveTarget.type === 'file') {
                  await apiService.moveDocument(moveTarget.data.document_id, moveDestinationId, user?.user_id);
                  showToast('📄 File moved', 'success');
                  // Optimistically update folder contents: remove from old, add to new
                  const oldFolderId = moveTarget.data.folder_id;
                  setFolderContents(prev => {
                    const updated = { ...prev };
                    if (updated[oldFolderId]?.documents) {
                      updated[oldFolderId] = {
                        ...updated[oldFolderId],
                        documents: updated[oldFolderId].documents.filter(d => d.document_id !== moveTarget.data.document_id)
                      };
                    }
                    return updated;
                  });
                  // Invalidate both folders for server truth
                  queryClient.invalidateQueries(['folders', 'contents', moveDestinationId]);
                  queryClient.invalidateQueries(['folders', 'contents', oldFolderId]);
                }
              } catch (e) {
                console.error('Move failed:', e);
                showToast('❌ Move failed', 'error');
              } finally {
                setMoveDialogOpen(false);
                setMoveTarget(null);
                setMoveDestinationId(null);
              }
            }}
            disabled={!moveDestinationId}
          >
            Move Here
          </Button>
        </DialogActions>
      </Dialog>

      {/* Upload Dialog */}
      <Dialog open={uploadDialog} onClose={() => {
        setUploadDialog(false);
        setUploadCategory('');
        setUploadTags([]);
        setUploadTagInput('');
      }} maxWidth="sm" fullWidth>
        <DialogTitle>Upload Files</DialogTitle>
        <DialogContent>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Upload files to: {(contextMenuTarget?.name || uploadTargetFolder?.name) || 'Root'}
            </Typography>
            {((contextMenuTarget?.collection_type === 'global' || uploadTargetFolder?.collection_type === 'global')) && (
              <Typography variant="caption" color="info.main" sx={{ display: 'block', mt: 1 }}>
                📁 Global folder - files will be shared with all users
              </Typography>
            )}
            <Typography variant="caption" color="success.main" sx={{ display: 'block', mt: 1 }}>
              ⚡ Files will be processed in the background after upload
            </Typography>
          </Box>
          
          <input
            type="file"
            multiple
            onChange={(e) => setUploadFiles(Array.from(e.target.files))}
            style={{ width: '100%', marginBottom: '16px' }}
          />
          
          {uploadFiles.length > 0 && (
            <Box sx={{ mt: 2, mb: 2 }}>
              <Typography variant="body2" gutterBottom>
                Selected files ({uploadFiles.length}):
              </Typography>
              {uploadFiles.map((file, index) => (
                <Typography key={index} variant="body2" color="text.secondary" sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1,
                  mb: 0.5 
                }}>
                  <InsertDriveFile fontSize="small" />
                  {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </Typography>
              ))}
            </Box>
          )}
          
          {/* ROOSEVELT METADATA FIELDS */}
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            📋 Document Metadata (Optional)
          </Typography>
          
          {/* Category Selection */}
          <FormControl fullWidth size="small" sx={{ mt: 2 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={uploadCategory}
              onChange={(e) => setUploadCategory(e.target.value)}
              label="Category"
            >
              <MenuItem value="">
                <em>None</em>
              </MenuItem>
              <MenuItem value="technical">Technical</MenuItem>
              <MenuItem value="academic">Academic</MenuItem>
              <MenuItem value="business">Business</MenuItem>
              <MenuItem value="legal">Legal</MenuItem>
              <MenuItem value="medical">Medical</MenuItem>
              <MenuItem value="literature">Literature</MenuItem>
              <MenuItem value="manual">Manual</MenuItem>
              <MenuItem value="reference">Reference</MenuItem>
              <MenuItem value="research">Research</MenuItem>
              <MenuItem value="personal">Personal</MenuItem>
              <MenuItem value="news">News</MenuItem>
              <MenuItem value="education">Education</MenuItem>
              <MenuItem value="other">Other</MenuItem>
            </Select>
          </FormControl>
          
          {/* Tags Input */}
          <Box sx={{ mt: 2 }}>
            <TextField
              fullWidth
              size="small"
              label="Tags (press Enter to add)"
              value={uploadTagInput}
              onChange={(e) => setUploadTagInput(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && uploadTagInput.trim()) {
                  e.preventDefault();
                  if (!uploadTags.includes(uploadTagInput.trim())) {
                    setUploadTags([...uploadTags, uploadTagInput.trim()]);
                  }
                  setUploadTagInput('');
                }
              }}
              placeholder="machine-learning, tutorial, notes..."
              helperText="Press Enter to add each tag"
            />
            {uploadTags.length > 0 && (
              <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {uploadTags.map((tag) => (
                  <Chip
                    key={tag}
                    label={tag}
                    size="small"
                    onDelete={() => setUploadTags(uploadTags.filter(t => t !== tag))}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setUploadDialog(false);
            setUploadTargetFolder(null);
            setUploadCategory('');
            setUploadTags([]);
            setUploadTagInput('');
          }}>Cancel</Button>
          <Button 
            onClick={handleUploadFiles}
            disabled={uploadFiles.length === 0 || uploadMutation.isLoading}
            startIcon={uploadMutation.isLoading ? <CircularProgress size={16} /> : <Upload />}
            variant="contained"
            color="primary"
          >
            {uploadMutation.isLoading ? 'Uploading...' : `Upload ${uploadFiles.length} file${uploadFiles.length !== 1 ? 's' : ''}`}
          </Button>
        </DialogActions>
      </Dialog>
       
               {/* Document Metadata Pane */}
        <DocumentMetadataPane
          document={metadataPane.document}
          open={metadataPane.open}
          onClose={() => setMetadataPane({ open: false, document: null, position: { x: 0, y: 0 } })}
          position={metadataPane.position}
        />
        
        {/* Folder Metadata Pane - ROOSEVELT FOLDER TAGGING */}
        <FolderMetadataPane
          folder={folderMetadataPane.folder}
          open={folderMetadataPane.open}
          onClose={() => setFolderMetadataPane({ open: false, folder: null, position: { x: 0, y: 0 } })}
          position={folderMetadataPane.position}
        />

        {/* Processing Files Indicator */}
        {processingFiles.length > 0 && (
          <Box
            sx={{
              position: 'fixed',
              bottom: 20,
              right: 20,
              zIndex: 1200,
              maxWidth: 400,
              backgroundColor: 'background.paper',
              border: '1px solid',
              borderColor: 'divider',
              borderRadius: 2,
              boxShadow: 3,
              p: 2
            }}
          >
            <Typography variant="subtitle2" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <CloudSync fontSize="small" />
              Processing Files ({processingFiles.length})
            </Typography>
            {processingFiles.map((file, index) => (
              <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <CircularProgress size={16} />
                <Typography variant="body2" sx={{ flex: 1, fontSize: '0.875rem' }}>
                  {file.filename}
                </Typography>
                <IconButton
                  size="small"
                  onClick={() => setProcessingFiles(prev => prev.filter((_, i) => i !== index))}
                >
                  <Delete fontSize="small" />
                </IconButton>
              </Box>
            ))}
          </Box>
        )}

      </Box>
    );
  };

export default FileTreeSidebar; 