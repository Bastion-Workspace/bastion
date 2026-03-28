import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  Divider,
  IconButton,
  Tooltip,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Menu,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  Description,
  CalendarToday,
  Person,
  Category,
  Tag,
  Download,
  Edit,
  Save,
  Visibility,
  VisibilityOff,
  ViewAgenda,
  FileDownload,
  Schedule,
  Fullscreen,
  FullscreenExit,
  InfoOutlined,
  Subject,
  History,
  Restore,
  VolumeUp,
  Stop,
  AudioFile
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize from 'rehype-sanitize';
import apiService from '../services/apiService';
import exportService from '../services/exportService';
import settingsService from '../services/settings/SettingsService';
import { Dialog, DialogTitle, DialogContent, DialogActions, FormGroup, FormControlLabel, Checkbox, Button, Stack, Switch, Popover, Drawer, FormLabel, RadioGroup, Radio } from '@mui/material';
import OrgRenderer from './OrgRenderer';
import OrgCMEditor from './OrgCMEditor';
import MarkdownCMEditor from './MarkdownCMEditor';
import OrgRefileDialog from './OrgRefileDialog';
import OrgArchiveDialog from './OrgArchiveDialog';
import OrgTagDialog from './OrgTagDialog';
import PDFDocumentViewer from './PDFDocumentViewer';
import AudioPlayer from './AudioPlayer';
import DocxViewer from './DocxViewer';
import PptxViewer from './PptxViewer';
import EMLViewer from './EMLViewer';
import { useEditor } from '../contexts/EditorContext';
import { useChatSidebar } from '../contexts/ChatSidebarContext';
import { parseFrontmatter } from '../utils/frontmatterUtils';
import { useTheme } from '../contexts/ThemeContext';
// Correct import path for documentDiffStore
import { documentDiffStore } from '../services/documentDiffStore';
import { useImageLightbox } from './common/ImageLightbox';
import FaceTagSuggestions from './images/FaceTagSuggestions';
import ImageMetadataModal from './images/ImageMetadataModal';
import { useAuth } from '../contexts/AuthContext';
import { stripTextForSpeech } from '../utils/textForSpeech';
import { useTTS } from '../hooks/useTTS';
import AudioExportDialog from './AudioExportDialog';

// Normalize bbox from face or object (supports bbox_x/bbox_y/bbox_width/bbox_height or left/top/right/bottom)
const getBbox = (item) => {
  const x = item.bbox_x ?? item.left ?? item.x ?? 0;
  const y = item.bbox_y ?? item.top ?? item.y ?? 0;
  const w = item.bbox_width ?? (item.right != null && item.left != null ? item.right - item.left : 0) ?? item.width ?? 0;
  const h = item.bbox_height ?? (item.bottom != null && item.top != null ? item.bottom - item.top : 0) ?? item.height ?? 0;
  return { x, y, w, h };
};

// Image viewer component with authentication
const ImageViewer = ({ documentId, filename, title, onDownload }) => {
  const [imageData, setImageData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [showMetadataModal, setShowMetadataModal] = React.useState(false);
  const [visionEnabled, setVisionEnabled] = React.useState(false);
  const [visionServiceAvailable, setVisionServiceAvailable] = React.useState(false);
  const [visionStatusChecked, setVisionStatusChecked] = React.useState(false);
  const [showBoundingBoxes, setShowBoundingBoxes] = React.useState(false);
  const [boxesData, setBoxesData] = React.useState({ faces: [], objects: [] });
  const [boxesLoading, setBoxesLoading] = React.useState(false);
  const [imageDisplaySize, setImageDisplaySize] = React.useState({ w: 0, h: 0 });
  const imageRef = React.useRef(null);
  const [descriptionAnchorEl, setDescriptionAnchorEl] = React.useState(null);
  const [descriptionContent, setDescriptionContent] = React.useState(null);
  const [descriptionLoading, setDescriptionLoading] = React.useState(false);
  const { openLightbox } = useImageLightbox();
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';

  // Fetch image with authentication
  React.useEffect(() => {
    const fetchImageWithAuth = async () => {
      try {
        const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
        if (!token) {
          setError('Authentication token not found');
          setLoading(false);
          return;
        }

        const imageUrl = `/api/images/${encodeURIComponent(filename)}`;
        const response = await fetch(imageUrl, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setImageData(url);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch image with auth:', err);
        setError('Failed to load image. Authentication may have failed.');
        setLoading(false);
      }
    };

    fetchImageWithAuth();
  }, [filename]);

  // Cleanup: revoke the blob URL when component unmounts or imageData changes
  React.useEffect(() => {
    return () => {
      if (imageData) {
        URL.revokeObjectURL(imageData);
      }
    };
  }, [imageData]);

  // Check vision features enabled and service availability
  React.useEffect(() => {
    const checkVisionStatus = async () => {
      try {
        // Check if user has vision features enabled
        const visionEnabledResponse = await settingsService.getVisionFeaturesEnabled();
        const enabled = visionEnabledResponse?.enabled === true;
        console.log('Vision features enabled:', enabled, visionEnabledResponse);
        setVisionEnabled(enabled);

        // Check if vision service is available
        const serviceStatusResponse = await settingsService.getVisionServiceStatus();
        const available = serviceStatusResponse?.available === true;
        console.log('Vision service available:', available, serviceStatusResponse);
        setVisionServiceAvailable(available);
        setVisionStatusChecked(true);
        
        console.log('Button visibility check:', {
          visionServiceAvailable: available,
          isAdmin,
          visionEnabled: enabled,
          willShow: available && (isAdmin || enabled)
        });
      } catch (err) {
        console.error('Failed to check vision status:', err);
        // Default to false on error, but mark as checked
        setVisionEnabled(false);
        setVisionServiceAvailable(false);
        setVisionStatusChecked(true);
      }
    };

    checkVisionStatus();
  }, [documentId, isAdmin]);

  // Load metadata (faces + objects) when "Show bounding boxes" is toggled on.
  // Fetch objects from /objects (detected_objects table) and fallback to image-metadata.objects so viewer matches Overlay.
  React.useEffect(() => {
    if (!showBoundingBoxes || !documentId) {
      setBoxesData({ faces: [], objects: [] });
      return;
    }
    let cancelled = false;
    setBoxesLoading(true);
    const load = async () => {
      try {
        const [metaRes, facesRes, objectsRes] = await Promise.all([
          apiService.get(`/api/documents/${documentId}/image-metadata`).catch(() => ({ exists: false })),
          apiService.get(`/api/documents/${documentId}/faces`).catch(() => ({ success: false })),
          apiService.getDetectedObjects(documentId).catch(() => ({ success: false }))
        ]);
        if (cancelled) return;
        const faces = (facesRes?.success && Array.isArray(facesRes.faces) && facesRes.faces.length > 0)
          ? facesRes.faces
          : (metaRes?.exists && metaRes?.metadata?.faces && Array.isArray(metaRes.metadata.faces))
            ? metaRes.metadata.faces
            : [];
        const objects = (objectsRes?.success && Array.isArray(objectsRes.objects) && objectsRes.objects.length > 0)
          ? objectsRes.objects
          : (metaRes?.exists && metaRes?.metadata?.objects && Array.isArray(metaRes.metadata.objects))
            ? metaRes.metadata.objects
            : [];
        setBoxesData({ faces, objects });
      } catch {
        if (!cancelled) setBoxesData({ faces: [], objects: [] });
      } finally {
        if (!cancelled) setBoxesLoading(false);
      }
    };
    load();
    return () => { cancelled = true; };
  }, [showBoundingBoxes, documentId]);

  const hasBoxes = boxesData.faces.length > 0 || boxesData.objects.length > 0;

  const measureImage = React.useCallback(() => {
    if (imageRef.current) {
      const rect = imageRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setImageDisplaySize({ w: rect.width, h: rect.height });
      }
    }
  }, []);

  React.useEffect(() => {
    if (showBoundingBoxes && hasBoxes) measureImage();
  }, [showBoundingBoxes, hasBoxes, boxesData.faces.length, boxesData.objects.length, measureImage]);

  const handleDescriptionClick = React.useCallback(async (e) => {
    setDescriptionAnchorEl(e.currentTarget);
    setDescriptionLoading(true);
    setDescriptionContent(null);
    try {
      const res = await apiService.get(`/api/documents/${documentId}/image-metadata`);
      const content = res?.exists && res?.metadata?.content ? res.metadata.content : null;
      setDescriptionContent(content);
    } catch {
      setDescriptionContent(null);
    } finally {
      setDescriptionLoading(false);
    }
  }, [documentId]);

  const handleDescriptionClose = React.useCallback(() => {
    setDescriptionAnchorEl(null);
  }, []);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box>
          <Typography variant="h6">{filename || title || 'Image File'}</Typography>
          {title && title !== filename && (
            <Typography variant="body2" color="text.secondary">{title}</Typography>
          )}
        </Box>
        <Box display="flex" gap={1}>
          {/* Edit metadata: opens unified ImageMetadataModal */}
          {((isAdmin) || (visionStatusChecked && visionServiceAvailable && visionEnabled)) && (
            <Tooltip title={visionServiceAvailable || isAdmin ? "Edit metadata" : "Checking vision service..."}>
              <IconButton 
                size="small" 
                onClick={() => setShowMetadataModal(true)}
              >
                <InfoOutlined fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          {imageData && !loading && !error && (
            <FormControlLabel
              control={
                <Switch
                  size="small"
                  checked={showBoundingBoxes}
                  onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                />
              }
              label="Show bounding boxes"
            />
          )}
          {imageData && !loading && !error && (
            <Tooltip title="Show description from metadata">
              <IconButton size="small" onClick={handleDescriptionClick}>
                <Subject fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          <Tooltip title="Download">
            <IconButton size="small" onClick={onDownload}>
              <Download fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      <Box sx={{ flex: 1, p: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'auto', bgcolor: 'background.default' }}>
        {loading && (
          <CircularProgress />
        )}
        {error && (
          <Alert severity="error">{error}</Alert>
        )}
        {imageData && !loading && !error && (
          <Box
            sx={{
              position: 'relative',
              display: 'inline-block',
              maxWidth: '100%',
              maxHeight: '100%'
            }}
          >
            <Box
              ref={imageRef}
              component="img"
              src={imageData}
              alt={filename}
              onClick={() => openLightbox(imageData, { filename, alt: title, documentId })}
              onLoad={measureImage}
              sx={{
                display: 'block',
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
                boxShadow: 3,
                borderRadius: 1,
                bgcolor: 'white',
                cursor: 'pointer'
              }}
            />
            {showBoundingBoxes && hasBoxes && imageDisplaySize.w > 0 && imageRef.current && (
              <svg
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  width: imageDisplaySize.w,
                  height: imageDisplaySize.h,
                  pointerEvents: 'none'
                }}
                width={imageDisplaySize.w}
                height={imageDisplaySize.h}
                viewBox={`0 0 ${imageDisplaySize.w} ${imageDisplaySize.h}`}
              >
                <>
                  {boxesData.faces.map((face, idx) => {
                      const nw = imageRef.current?.naturalWidth || 1;
                      const nh = imageRef.current?.naturalHeight || 1;
                      const sx = imageDisplaySize.w / nw;
                      const sy = imageDisplaySize.h / nh;
                      const { x, y, w, h } = getBbox(face);
                      const rx = x * sx;
                      const ry = y * sy;
                      const rw = w * sx;
                      const rh = h * sy;
                      const label = face.identity_name || face.suggested_identity || null;
                      return (
                        <g key={face.id ?? `face-${idx}`}>
                          <rect
                            x={rx}
                            y={ry}
                            width={rw}
                            height={rh}
                            fill="none"
                            stroke="lime"
                            strokeWidth={2}
                          />
                          {label && (
                            <text
                              x={rx}
                              y={Math.max(ry - 4, 12)}
                              fill="white"
                              fontSize={12}
                              fontWeight="bold"
                              style={{ paintOrder: 'stroke', stroke: 'black', strokeWidth: 2 }}
                            >
                              {face.identity_name ? label : `${label}?`}
                            </text>
                          )}
                        </g>
                      );
                    })}
                    {boxesData.objects.map((obj, idx) => {
                      const nw = imageRef.current?.naturalWidth || 1;
                      const nh = imageRef.current?.naturalHeight || 1;
                      const sx = imageDisplaySize.w / nw;
                      const sy = imageDisplaySize.h / nh;
                      const { x, y, w, h } = getBbox(obj);
                      const rx = x * sx;
                      const ry = y * sy;
                      const rw = w * sx;
                      const rh = h * sy;
                      const label = obj.user_tag ?? obj.matched_description ?? obj.class_name ?? obj.label ?? obj.name ?? null;
                      const strokeColor = obj.detection_method === 'user_defined' ? '#9333ea' : '#f97316';
                      return (
                        <g key={obj.id ?? `obj-${idx}`}>
                          <rect
                            x={rx}
                            y={ry}
                            width={rw}
                            height={rh}
                            fill="none"
                            stroke={strokeColor}
                            strokeWidth={2}
                          />
                          {label && (
                            <text
                              x={rx}
                              y={Math.max(ry - 4, 12)}
                              fill="white"
                              fontSize={12}
                              fontWeight="bold"
                              style={{ paintOrder: 'stroke', stroke: 'black', strokeWidth: 2 }}
                            >
                              {label}
                            </text>
                          )}
                        </g>
                      );
                    })}
                </>
              </svg>
            )}
          </Box>
        )}
      </Box>
      <Popover
        open={Boolean(descriptionAnchorEl)}
        anchorEl={descriptionAnchorEl}
        onClose={handleDescriptionClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
      >
        <Box sx={{ p: 2, maxWidth: 420, maxHeight: 360, overflow: 'auto', minWidth: 200 }}>
          {descriptionLoading ? (
            <CircularProgress size={24} />
          ) : (
            <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
              {descriptionContent || 'No description in metadata.'}
            </Typography>
          )}
        </Box>
      </Popover>
      <Box sx={{ p: 2 }}>
        <FaceTagSuggestions
          documentId={documentId}
          onOpenFaceTagger={() => setShowMetadataModal(true)}
        />
      </Box>
      <ImageMetadataModal
        open={showMetadataModal}
        onClose={(success) => {
          setShowMetadataModal(false);
          if (success) setImageData((prev) => prev);
        }}
        documentId={documentId}
        filename={filename}
        imageUrl={imageData}
      />
    </Box>
  );
};

// Memoize DocumentViewer to prevent re-renders from parent context updates
const DocumentViewer = React.memo(({ documentId, onClose, scrollToLine = null, scrollToHeading = null, initialScrollPosition = 0, onScrollChange }) => {
  const { user } = useAuth();
  const isAdmin = user?.role === 'admin';
  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  // Track actual server content to detect REAL changes
  const [serverContent, setServerContent] = useState('');
  // Ref to track latest editContent for cleanup (avoids stale closure issues)
  const editContentRef = React.useRef('');
  // Ref to track if the component is unmounting
  const isUnmountingRef = React.useRef(false);
  // Track if user manually entered edit mode (vs auto-entered)
  const manuallyEditingRef = React.useRef(false);
  const [saving, setSaving] = useState(false);
  // View mode for editable docs: 'edit' (editor only) | 'split' (editor + preview) | 'preview' (preview only)
  const [viewMode, setViewMode] = useState('edit');
  const { setEditorState } = useEditor();
  const { messages: sidebarMessages = [] } = useChatSidebar();
  const { darkMode, accentId } = useTheme();
  const [exportOpen, setExportOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [pdfExportDialogOpen, setPdfExportDialogOpen] = useState(false);
  const [audioExportOpen, setAudioExportOpen] = useState(false);
  const [pdfExportOrientation, setPdfExportOrientation] = useState('portrait');
  const [pdfIncludeToc, setPdfIncludeToc] = useState(false);
  const [pdfTocDepth, setPdfTocDepth] = useState(3);
  const [pdfPageNumbers, setPdfPageNumbers] = useState(true);
  const [pdfWatermarkText, setPdfWatermarkText] = useState('');
  const [pdfWatermarkAllPages, setPdfWatermarkAllPages] = useState(true);
  const [pdfPageBreakHeadings, setPdfPageBreakHeadings] = useState([]);
  const [pdfLayout, setPdfLayout] = useState('article');
  const [pdfFontPreset, setPdfFontPreset] = useState('liberation');
  const [pdfTypefaceStyle, setPdfTypefaceStyle] = useState('mixed');
  const [pdfHeadingOutline, setPdfHeadingOutline] = useState([]);
  const [pdfOutlineLoading, setPdfOutlineLoading] = useState(false);
  const [pdfBookMarginTopMm, setPdfBookMarginTopMm] = useState(20);
  const [pdfBookMarginRightMm, setPdfBookMarginRightMm] = useState(20);
  const [pdfBookMarginBottomMm, setPdfBookMarginBottomMm] = useState(28);
  const [pdfBookMarginLeftMm, setPdfBookMarginLeftMm] = useState(20);
  const [pdfBookIndentBody, setPdfBookIndentBody] = useState(true);
  const [pdfBookNoIndentAfterHeading, setPdfBookNoIndentAfterHeading] = useState(true);
  const [pdfBookPageNumFormat, setPdfBookPageNumFormat] = useState('n_of_total');
  const [pdfBookPageNumVertical, setPdfBookPageNumVertical] = useState('bottom');
  const [pdfBookPageNumHorizontal, setPdfBookPageNumHorizontal] = useState('center');
  const [pdfBookSuppressFirstPage, setPdfBookSuppressFirstPage] = useState(false);
  const [pdfSectionOverrides, setPdfSectionOverrides] = useState({});
  const pdfOutlineTimerRef = useRef(null);
  const pdfSourceFormat = useMemo(() => {
    const n = (document?.filename || '').toLowerCase();
    return n.endsWith('.org') ? 'org' : 'markdown';
  }, [document?.filename]);
  const [epubTitle, setEpubTitle] = useState('');
  const [epubAuthorFirst, setEpubAuthorFirst] = useState('');
  const [epubAuthorLast, setEpubAuthorLast] = useState('');
  const [epubLanguage, setEpubLanguage] = useState('en');
  const [includeToc, setIncludeToc] = useState(true);
  const [includeCover, setIncludeCover] = useState(true);
  const [splitOnHeadings, setSplitOnHeadings] = useState(true);
  const [splitLevels, setSplitLevels] = useState([1, 2]);
  const [centerLevels, setCenterLevels] = useState([]);
  const [indentParagraphs, setIndentParagraphs] = useState(true);
  const [noIndentFirstParagraph, setNoIndentFirstParagraph] = useState(true);
  const contentBoxRef = React.useRef(null);
  const [backlinks, setBacklinks] = useState([]);
  const [loadingBacklinks, setLoadingBacklinks] = useState(false);
  const [refileDialogOpen, setRefileDialogOpen] = useState(false);
  const [refileSourceFile, setRefileSourceFile] = useState('');
  const [refileSourceLine, setRefileSourceLine] = useState(null);
  const [refileSourceHeading, setRefileSourceHeading] = useState('');
  const [archiveDialogOpen, setArchiveDialogOpen] = useState(false);
  const [archiveSourceFile, setArchiveSourceFile] = useState('');
  const [archiveSourceLine, setArchiveSourceLine] = useState(null);
  const [archiveSourceHeading, setArchiveSourceHeading] = useState('');
  const [activeClock, setActiveClock] = useState(null);
  const [checkingClock, setCheckingClock] = useState(true);
  const [tagDialogOpen, setTagDialogOpen] = useState(false);
  const [tagSourceLine, setTagSourceLine] = useState(null);
  const [tagSourceHeading, setTagSourceHeading] = useState('');
  const orgEditorRef = React.useRef(null);
  const markdownEditorRef = React.useRef(null);
  // Scroll position to restore after a refresh (editor remount). Set before fetchDocument so we have it when the new editor mounts.
  const pendingScrollAfterRefreshRef = React.useRef(null);
  // Proposal IDs recently accepted/rejected/terminal-failed; skip re-injecting from DB for TTL window.
  const resolvedProposalIdsRef = React.useRef(new Map());
  const [versionHistoryOpen, setVersionHistoryOpen] = useState(false);
  const [versionsList, setVersionsList] = useState([]);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [versionViewContent, setVersionViewContent] = useState(null);
  const [versionViewMeta, setVersionViewMeta] = useState(null);
  const [versionDiffResult, setVersionDiffResult] = useState(null);
  const [rollbackConfirmVersion, setRollbackConfirmVersion] = useState(null);
  const [rollbacking, setRollbacking] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editingFilename, setEditingFilename] = useState(false);
  const [editedTitle, setEditedTitle] = useState('');
  const [editedFilename, setEditedFilename] = useState('');
  const [updatingMetadata, setUpdatingMetadata] = useState(false);
  const [externalUpdateNotification, setExternalUpdateNotification] = useState(null);
  const [downloadMenuAnchor, setDownloadMenuAnchor] = useState(null);
  // Scroll position to restore after a full-document refresh (e.g. Accept All). Cleared after editor applies it.
  const [restoreScrollAfterRefresh, setRestoreScrollAfterRefresh] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const fullscreenContainerRef = React.useRef(null);
  const [currentSection, setCurrentSection] = useState(null);
  const { isSpeaking: isReadingAloud, canSpeak, speak } = useTTS({
    stopEventName: 'document-read-aloud-stop',
  });

  // Helper functions for unsaved content persistence
  const getUnsavedContentKey = (docId) => `unsaved_content_${docId}`;
  
  const getUnsavedContent = (docId) => {
    try {
      const key = getUnsavedContentKey(docId);
      const saved = localStorage.getItem(key);
      return saved ? saved : null;
    } catch (e) {
      console.error('Failed to get unsaved content:', e);
      return null;
    }
  };

  // Helper functions for edit mode preference
  const getEditModePreferenceKey = () => `prefer_edit_mode`;
  
  const getEditModePreference = () => {
    try {
      const saved = localStorage.getItem(getEditModePreferenceKey());
      return saved === 'true';
    } catch (e) {
      return true; // Default to true - always prefer edit mode
    }
  };
  
  const setEditModePreference = (preferEdit) => {
    try {
      localStorage.setItem(getEditModePreferenceKey(), preferEdit ? 'true' : 'false');
    } catch (e) {
      console.error('Failed to save edit mode preference:', e);
    }
  };

  // Check if user can edit this document
  const canUserEditDocument = (docData) => {
    if (!docData) return false;
    const fname = (docData.filename || '').toLowerCase();
    const isEditableType = fname.endsWith('.md') || fname.endsWith('.txt') || fname.endsWith('.org');
    if (!isEditableType) return false;
    
    const collectionType = docData.collection_type || 'user';
    const docUserId = docData.user_id;
    const currentUserId = user?.user_id;
    
    // Admin can edit global documents
    if (isAdmin && collectionType === 'global') return true;
    
    // Users can edit their own documents
    if (collectionType === 'user' && docUserId === currentUserId) return true;
    
    return false;
  };

  const readAloudFilenameLower = (document?.filename || '').toLowerCase();
  const isReadableTextDocument =
    !!document?.filename &&
    (
      readAloudFilenameLower.endsWith('.md') ||
      readAloudFilenameLower.endsWith('.txt') ||
      readAloudFilenameLower.endsWith('.org')
    );

  const getPreferredTextToRead = () => {
    const selectedFromOrg = orgEditorRef.current?.getSelectedText?.();
    const selectedFromMarkdown = markdownEditorRef.current?.getSelectedText?.();
    const selectedText = selectedFromOrg || selectedFromMarkdown;
    if (selectedText && String(selectedText).trim()) {
      return String(selectedText);
    }
    if (isEditing) {
      return editContent || '';
    }
    return document?.content || '';
  };

  const handleToggleReadAloud = async () => {
    if (!canSpeak || !isReadableTextDocument) return;

    if (isReadingAloud) {
      window.dispatchEvent(new CustomEvent('document-read-aloud-stop'));
      return;
    }

    const sourceText = getPreferredTextToRead();
    const mode = readAloudFilenameLower.endsWith('.org') ? 'org' : 'markdown';
    const speechText = stripTextForSpeech(sourceText, mode);
    if (!speechText) return;

    try {
      await speak(speechText);
    } catch (error) {
      console.error('Document read aloud failed:', error);
    }
  };
  
  const saveUnsavedContent = (docId, content) => {
    try {
      const key = getUnsavedContentKey(docId);
      if (content !== null && content !== undefined) {
        localStorage.setItem(key, content);
      }
    } catch (e) {
      console.error('Failed to save unsaved content:', e);
    }
  };
  
  const clearUnsavedContent = (docId) => {
    try {
      const key = getUnsavedContentKey(docId);
      localStorage.removeItem(key);
    } catch (e) {
      console.error('Failed to clear unsaved content:', e);
    }
  };

  // Fetch document content (can be called multiple times for refresh)
  const fetchDocument = React.useCallback(async (forceRefresh = false, preserveEditMode = false) => {
    try {
      setLoading(true);
      setError(null);
      
      // Check for unsaved content first (unless forcing refresh)
      if (!forceRefresh) {
        const unsavedContent = getUnsavedContent(documentId);
        if (unsavedContent !== null) {
          // We have unsaved content, but we still need document metadata
          // Fetch metadata only, then use unsaved content
          const response = await apiService.getDocumentContent(documentId);
          const docData = {
            ...response.metadata,
            content: unsavedContent, // Use unsaved content for display
            chunk_count: response.chunk_count,
            total_length: response.total_length
          };
          setDocument(docData);
          setEditContent(unsavedContent);
          // Store the actual server content separately
          setServerContent(response.content || '');
          
          // Determine if we should enter edit mode (for unsaved content path)
          const canEdit = canUserEditDocument(docData);
          
          if (preserveEditMode && isEditing) {
            // Preserve current edit mode
            setIsEditing(true);
          } else if (manuallyEditingRef.current && isEditing) {
            // User manually entered edit mode - preserve it
            setIsEditing(true);
          } else if (canEdit) {
            // User can edit - always open in edit mode
            setIsEditing(true);
            setViewMode('edit');
            manuallyEditingRef.current = false;
            setEditModePreference(true); // Remember preference
          } else {
            // User cannot edit - preview mode
            setIsEditing(false);
            manuallyEditingRef.current = false;
          }
          
          setLoading(false);
          return;
        }
      }
      
      // No unsaved content or forcing refresh - fetch from API
      const response = await apiService.getDocumentContent(documentId);
      
      const docData = {
        ...response.metadata,
        content: response.content,
        chunk_count: response.chunk_count,
        total_length: response.total_length
      };
      setDocument(docData);
      setEditContent(response.content || '');
      setServerContent(response.content || '');
      
      // Clear any stale unsaved content when we fetch fresh content
      if (forceRefresh) {
        clearUnsavedContent(documentId);
      }
      
      // Determine if we should enter edit mode
      const canEdit = canUserEditDocument(docData);
      
      if (preserveEditMode && isEditing) {
        // Preserve current edit mode when force refreshing (don't change isEditing state)
        console.log('🔄 Preserving edit mode during force refresh');
        // isEditing state will remain unchanged
      } else if (manuallyEditingRef.current && isEditing) {
        // User manually entered edit mode - preserve it
        console.log('🔄 Preserving manually entered edit mode');
        // isEditing state will remain unchanged
      } else if (canEdit) {
        // User can edit - always open in edit mode
        console.log('✅ Opening document in edit mode (user has permission)');
        setIsEditing(true);
        setViewMode('edit');
        manuallyEditingRef.current = false; // Auto-entered
        setEditModePreference(true); // Remember preference
      } else {
        // User cannot edit this document - always preview mode
        console.log('📖 Opening document in preview mode (user cannot edit)');
        setIsEditing(false);
        manuallyEditingRef.current = false;
      }
      
    } catch (err) {
      console.error('Failed to fetch document:', err);
      setError('Failed to load document content');
    } finally {
      setLoading(false);
    }
  }, [documentId, isEditing]);

  // Initial document load
  useEffect(() => {
    if (documentId) {
      manuallyEditingRef.current = false; // Reset manual edit flag when switching documents
      setViewMode('edit'); // Start each document in edit-only view
      fetchDocument(false);
    }
  }, [documentId, fetchDocument]);

  // Refresh when journal quick capture adds an entry (so open journal tab shows new content)
  useEffect(() => {
    const handler = () => {
      if (documentId) {
        fetchDocument(true, true);
      }
    };
    window.addEventListener('journalDocumentUpdated', handler);
    return () => window.removeEventListener('journalDocumentUpdated', handler);
  }, [documentId, fetchDocument]);

  // Clear pending scroll ref after editor has restored (so we don't keep forcing the same scroll)
  useEffect(() => {
    if (!document || loading || pendingScrollAfterRefreshRef.current == null) return;
    const t = setTimeout(() => {
      pendingScrollAfterRefreshRef.current = null;
      setRestoreScrollAfterRefresh(null);
    }, 250);
    return () => clearTimeout(t);
  }, [document?.document_id, loading]);

  // Save unsaved content whenever editContent changes (debounced)
  useEffect(() => {
    if (!documentId || !isEditing || !editContent) return;
    
    // Update ref with latest content
    editContentRef.current = editContent;

    // Compare against serverContent (disk) rather than document.content.
    // This prevents premature clearing when restoring from localStorage on mount.
    if (editContent === serverContent) {
      // Content matches server version, clear unsaved content
      clearUnsavedContent(documentId);
      return;
    }
    
    // Debounce saving unsaved content
    const timeoutId = setTimeout(() => {
      saveUnsavedContent(documentId, editContent);
    }, 500);
    
    return () => {
      clearTimeout(timeoutId);
      
      // Only save in the cleanup if we are actually unmounting.
      // This prevents saving on every keystroke (which was causing race conditions).
      // We rely on the unmount tracker effect (defined below) to set this flag.
      if (!isUnmountingRef.current) return;

      // CRITICAL FIX: Save immediately on unmount using ref to get latest content
      // This prevents loss when switching tabs quickly (avoids stale closure)
      const currentSavedContent = serverContent; // Use serverContent for final check
      const latestContent = editContentRef.current;
      
      // Only save if content differs from what's on disk
      if (latestContent && latestContent !== currentSavedContent) {
        // EXTRA PROTECTION: Don't overwrite unsaved changes with the original content 
        // if the original content is significantly shorter (prevents accidental wipes during crashes)
        const existingUnsaved = getUnsavedContent(documentId);
        if (existingUnsaved && latestContent.length < existingUnsaved.length * 0.5 && latestContent.length < 500) {
          console.warn('⚠️ Refusing to overwrite longer unsaved content with much shorter content during unmount (possible race condition)');
          return;
        }
        
        console.log('Saving unsaved content on unmount for document:', documentId);
        saveUnsavedContent(documentId, latestContent);
      }
    };
  }, [documentId, isEditing, editContent, serverContent]);

  // Track unmounting state. Define this AFTER the save effect
  // so its cleanup runs BEFORE the save effect's cleanup!
  useEffect(() => {
    return () => {
      isUnmountingRef.current = true;
    };
  }, []);

  const docViewerWsRef = useRef(null);
  const docViewerReconnectAttemptsRef = useRef(0);
  const docViewerReconnectTimeoutRef = useRef(null);
  const acceptShieldUntilRef = useRef(0);
  const refreshCooldownUntilRef = useRef(0);
  const acceptAllInProgressRef = useRef(false);
  const rejectAllInProgressRef = useRef(false);
  const syncPendingProposalsAbortRef = useRef(null);
  const documentIdRef = useRef(documentId);
  documentIdRef.current = documentId;

  // Centralized: capture scroll and do a single refresh (guards against duplicate refresh bursts).
  const refreshWithScrollCapture = React.useCallback(() => {
    if (Date.now() < refreshCooldownUntilRef.current) return;
    const scrollPos = orgEditorRef.current?.getScrollPosition?.() ?? markdownEditorRef.current?.getScrollPosition?.() ?? contentBoxRef?.current?.scrollTop ?? 0;
    if (typeof onScrollChange === 'function') onScrollChange(scrollPos);
    pendingScrollAfterRefreshRef.current = scrollPos;
    setRestoreScrollAfterRefresh(scrollPos);
    refreshCooldownUntilRef.current = Date.now() + 3000;
    fetchDocument(true, true);
  }, [fetchDocument, onScrollChange]);

  const handleScrollRestored = React.useCallback(() => {
    pendingScrollAfterRefreshRef.current = null;
    setRestoreScrollAfterRefresh(null);
  }, []);

  // Handle edit proposals - route through live-diff path (same as writing assistant)
  const handleEditProposal = React.useCallback(async (proposal) => {
    if (!proposal || !documentId) return;
    if (proposal.document_id != null && proposal.document_id !== documentId) return;

    try {
      const currentContent = editContentRef.current || editContent || document?.content || '';
      const proposalId = proposal.proposal_id ?? proposal.id ?? `proposal-${Date.now()}`;
      let ops = [];

      const parseOperations = (raw) => {
        if (Array.isArray(raw)) return raw;
        if (typeof raw === 'string') {
          try {
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
          } catch (_) {
            return [];
          }
        }
        return [];
      };

      if (proposal.edit_type === 'content' && proposal.content_edit) {
        const contentEdit = proposal.content_edit;
        const contentLength = currentContent.length;
        if (contentEdit.edit_mode === 'append') {
          ops = [{
            op_type: 'insert_after',
            start: contentLength,
            end: contentLength,
            text: contentEdit.content || '',
          }];
        } else if (contentEdit.edit_mode === 'replace') {
          ops = [{
            op_type: 'replace_range',
            start: 0,
            end: contentLength,
            text: contentEdit.content || '',
            original_text: currentContent,
          }];
        }
      } else {
        const rawOps = proposal.operations ?? proposal.editor_operations;
        ops = parseOperations(rawOps);
      }

      if (ops.length > 0) {
        const taggedOps = ops.map((op, idx) => {
          const base = { ...op };
          if (proposal.edit_type === 'operations') {
            base.proposal_operation_index =
              op.proposal_operation_index !== undefined && op.proposal_operation_index !== null
                ? Number(op.proposal_operation_index)
                : idx;
          }
          return base;
        });
        documentDiffStore.mergeProposalDiff(documentId, taggedOps, proposalId, currentContent);
        if (isEditing) {
          window.dispatchEvent(
            new CustomEvent('editorOperationsLive', {
              detail: {
                operations: taggedOps,
                messageId: proposalId,
                documentId,
                filename: document?.filename,
                preserveExisting: true,
              },
            })
          );
        }
      } else if (proposal.edit_type === 'operations' || proposal.operations != null || proposal.editor_operations != null) {
        console.warn('handleEditProposal: no ops after parse', {
          documentId,
          proposal_id: proposalId,
          edit_type: proposal.edit_type,
          rawOperationsType: typeof (proposal.operations ?? proposal.editor_operations),
          rawOperationsPreview: JSON.stringify((proposal.operations ?? proposal.editor_operations)?.slice?.(0, 1) ?? proposal.operations ?? proposal.editor_operations).slice(0, 120),
        });
      }
    } catch (err) {
      console.error('Error handling edit proposal:', err);
    }
  }, [documentId, isEditing, editContent, document]);

  const handleEditProposalRef = React.useRef(handleEditProposal);
  handleEditProposalRef.current = handleEditProposal;

  const syncPendingProposalsFromDb = React.useCallback((reason) => {
    if (!documentId) return;
    const now = Date.now();
    const resolved = resolvedProposalIdsRef.current;
    for (const [id, exp] of resolved.entries()) {
      if (exp <= now) resolved.delete(id);
    }
    const fetchedForDocId = documentId;
    syncPendingProposalsAbortRef.current?.abort();
    const controller = new AbortController();
    syncPendingProposalsAbortRef.current = controller;

    apiService.getPendingProposals(documentId, { signal: controller.signal })
      .then((proposals) => {
        if (controller.signal.aborted) return;
        if (documentIdRef.current !== fetchedForDocId) return;
        if (!Array.isArray(proposals)) return;
        const handler = handleEditProposalRef.current;
        if (!handler) return;
        for (const p of proposals) {
          const pid = p.proposal_id ?? p.id;
          if (pid && resolved.has(pid) && resolved.get(pid) > now) continue;
          handler(p);
        }
      })
      .catch((err) => {
        if (err?.name === 'AbortError') return;
      });
  }, [documentId]);

  // WebSocket listener for real-time document updates
  useEffect(() => {
    if (!documentId) return;

    const token = apiService.getToken();
    if (!token) {
      console.error('❌ No authentication token available for document WebSocket');
      return;
    }

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws/folders?token=${encodeURIComponent(token)}`;
    const MAX_RECONNECT_ATTEMPTS = 5;

    const connect = () => {
      let ws;
      try {
        ws = new WebSocket(wsUrl);
        docViewerWsRef.current = ws;
      } catch (err) {
        console.error('❌ Failed to create document WebSocket:', err);
        return;
      }

      ws.onopen = () => {
        console.log('📡 DocumentViewer: Connected to updates WebSocket');
        docViewerReconnectAttemptsRef.current = 0;
        syncPendingProposalsFromDb('ws_reconnect');
      };

      ws.onmessage = (event) => {
        try {
          const update = JSON.parse(event.data);
          
          // Listen for updates to THIS document
          if (update.type === 'document_status_update' && update.document_id === documentId) {
            console.log('🔄 Document updated, refreshing content:', update);
            
            // Always refresh when status is 'completed' (agent finished updating)
            // This ensures users see agent changes in real-time, even if they have unsaved content
            if (update.status === 'completed') {
              console.log('🔄 Auto-refreshing document content (status: completed - agent finished work)...');
              const hasUnsaved = getUnsavedContent(documentId) !== null;
              if (hasUnsaved) {
                console.log('⚠️ Document has unsaved changes, but refreshing due to agent completion');
                setExternalUpdateNotification({
                  message: 'File was updated by agent. Your unsaved changes were overwritten.',
                  timestamp: Date.now()
                });
                setTimeout(() => setExternalUpdateNotification(null), 5000);
              }
              clearUnsavedContent(documentId);
              if (!acceptAllInProgressRef.current && !rejectAllInProgressRef.current) {
                refreshWithScrollCapture();
              }
            } else if (update.status === 'processing') {
              console.log('⏳ Document is being processed, waiting for completion...');
            } else if (update.status === 'edit_proposal_applied') {
              if (!acceptAllInProgressRef.current && !rejectAllInProgressRef.current) {
                refreshWithScrollCapture();
              }
            } else if (update.status === 'edit_proposal_rejected') {
              // Skip refresh during bulk reject - single refresh happens in finally block
              if (!rejectAllInProgressRef.current && !isEditing) {
                // Single reject when not editing: allow one refresh
                if (Date.now() < acceptShieldUntilRef.current) {
                  console.log('⏸️ Within accept shield window, skipping refresh');
                } else {
                  console.log('🔄 Auto-refreshing document content (proposal rejected, not editing)...');
                  fetchDocument(true);
                }
              }
            } else if (!isEditing) {
              // If not editing and status changed, refresh (unless within accept shield or bulk operations)
              if (rejectAllInProgressRef.current || acceptAllInProgressRef.current) {
                console.log('⏸️ Bulk operation in progress, skipping refresh');
              } else if (Date.now() < acceptShieldUntilRef.current) {
                console.log('⏸️ Within accept shield window, skipping refresh');
              } else {
                console.log('🔄 Auto-refreshing document content (not editing, status changed)...');
                fetchDocument(true);
              }
            } else {
              // User is editing and status is not completed - don't interrupt them
              console.log('⏸️ User is editing and status is not completed, skipping auto-refresh');
            }
          }
          
          // Listen for document edit proposals (trigger-only: sync from DB as single source of truth)
          if (update.type === 'document_edit_proposal' && update.document_id === documentId) {
            console.log('📝 Edit proposal signal for this document, syncing from DB');
            syncPendingProposalsFromDb('ws_edit_proposal');
          }
        } catch (err) {
          console.error('❌ Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('❌ DocumentViewer WebSocket error:', err);
      };

      ws.onclose = () => {
        console.log('📡 DocumentViewer: WebSocket connection closed');
        docViewerWsRef.current = null;

        const attempts = docViewerReconnectAttemptsRef.current;
        if (attempts >= MAX_RECONNECT_ATTEMPTS) {
          console.warn('📡 DocumentViewer WebSocket: max reconnect attempts reached, giving up');
          return;
        }
        docViewerReconnectAttemptsRef.current = attempts + 1;
        const delay = Math.min(1000 * Math.pow(2, attempts), 30000);
        console.log(`📡 DocumentViewer WebSocket: reconnecting in ${delay}ms (attempt ${attempts + 1}/${MAX_RECONNECT_ATTEMPTS})`);
        docViewerReconnectTimeoutRef.current = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      syncPendingProposalsAbortRef.current?.abort();
      if (docViewerReconnectTimeoutRef.current) {
        clearTimeout(docViewerReconnectTimeoutRef.current);
        docViewerReconnectTimeoutRef.current = null;
      }
      if (docViewerWsRef.current && docViewerWsRef.current.readyState === WebSocket.OPEN) {
        docViewerWsRef.current.close();
        docViewerWsRef.current = null;
      }
    };
  }, [documentId, isEditing, fetchDocument, syncPendingProposalsFromDb, refreshWithScrollCapture]);

  // Listen for accept/reject from liveEditDiffExtension and call backend; dispatch Done so plugin can remove ops (proposal-level clear)
  useEffect(() => {
    if (!documentId) return;
    const RESOLVED_TTL_MS = 60000;
    const markResolved = (proposalId) => {
      if (proposalId) {
        resolvedProposalIdsRef.current.set(proposalId, Date.now() + RESOLVED_TTL_MS);
      }
    };
    const onProposalAccept = async (e) => {
      const { documentId: evDocId, proposalId, operationId, proposalOperationIndex } = e.detail || {};
      if (evDocId !== documentId || !proposalId) return;
      const hasOpIndex =
        proposalOperationIndex !== undefined &&
        proposalOperationIndex !== null &&
        !Number.isNaN(Number(proposalOperationIndex));
      const selectedIndices = hasOpIndex ? [Number(proposalOperationIndex)] : null;
      try {
        await apiService.applyDocumentEditProposal(proposalId, selectedIndices);
        if (hasOpIndex && operationId) {
          window.dispatchEvent(new CustomEvent('proposalAcceptDone', {
            detail: { documentId, operationIds: [operationId] },
          }));
          setTimeout(() => syncPendingProposalsFromDb('proposal_partial_apply'), 350);
        } else {
          markResolved(proposalId);
          window.dispatchEvent(new CustomEvent('proposalAcceptDone', {
            detail: { documentId, proposalId, clearAllForProposal: true },
          }));
        }
        refreshWithScrollCapture();
      } catch (err) {
        console.error('Apply document edit proposal failed:', err);
      }
    };
    const onProposalReject = async (e) => {
      const { documentId: evDocId, proposalId, operationId, operationIds } = e.detail || {};
      if (evDocId !== documentId || !proposalId) return;
      // Clear diff decoration and store immediately (same race fix as accept)
      markResolved(proposalId);
      window.dispatchEvent(new CustomEvent('proposalRejectDone', {
        detail: { documentId, proposalId, clearAllForProposal: true }
      }));
      try {
        await apiService.rejectDocumentEditProposal(proposalId);
      } catch (err) {
        console.error('Reject document edit proposal failed:', err);
        const status = err?.response?.status;
        const isTerminal = status >= 400 && status < 500;
        if (isTerminal) {
          markResolved(proposalId);
          window.dispatchEvent(new CustomEvent('proposalRejectDone', {
            detail: { documentId, proposalId, clearAllForProposal: true }
          }));
        }
      }
    };
    const onProposalAcceptAll = async (e) => {
      const { documentId: evDocId, proposalIds } = e.detail || {};
      if (evDocId !== documentId || !Array.isArray(proposalIds)) return;
      acceptAllInProgressRef.current = true;
      // Clear diff decorations immediately so UI updates without waiting for backend
      proposalIds.forEach((proposalId) => {
        markResolved(proposalId);
        window.dispatchEvent(new CustomEvent('proposalAcceptDone', {
          detail: { documentId, proposalId, clearAllForProposal: true }
        }));
      });
      try {
        for (const proposalId of proposalIds) {
          try {
            await apiService.applyDocumentEditProposal(proposalId);
          } catch (err) {
            const status = err?.response?.status;
            if (status >= 400 && status < 500) {
              markResolved(proposalId);
              window.dispatchEvent(new CustomEvent('proposalAcceptDone', {
                detail: { documentId, proposalId, clearAllForProposal: true }
              }));
            }
          }
        }
      } finally {
        acceptAllInProgressRef.current = false;
        refreshWithScrollCapture();
      }
    };
    const onProposalRejectAll = async (e) => {
      const { documentId: evDocId, proposalIds } = e.detail || {};
      if (evDocId !== documentId || !Array.isArray(proposalIds)) return;
      rejectAllInProgressRef.current = true;
      // Clear diff decorations immediately so UI updates without waiting for backend
      proposalIds.forEach((proposalId) => {
        markResolved(proposalId);
        window.dispatchEvent(new CustomEvent('proposalRejectDone', {
          detail: { documentId, proposalId, clearAllForProposal: true }
        }));
      });
      try {
        await Promise.all(
          proposalIds.map(async (proposalId) => {
            try {
              await apiService.rejectDocumentEditProposal(proposalId);
            } catch (err) {
              const status = err?.response?.status;
              // 404 means proposal already deleted - treat as success (idempotent)
              if (status === 404) {
                return; // Already rejected, no-op
              }
              // Other 4xx errors: already cleared upfront, just log
              if (status >= 400 && status < 500) {
                console.warn(`Reject proposal ${proposalId} failed with ${status}, but UI already cleared`);
                return;
              }
              // Re-throw unexpected errors
              throw err;
            }
          })
        );
      } finally {
        rejectAllInProgressRef.current = false;
        refreshWithScrollCapture();
      }
    };
    window.addEventListener('proposalAccept', onProposalAccept);
    window.addEventListener('proposalReject', onProposalReject);
    window.addEventListener('proposalAcceptAll', onProposalAcceptAll);
    window.addEventListener('proposalRejectAll', onProposalRejectAll);
    return () => {
      window.removeEventListener('proposalAccept', onProposalAccept);
      window.removeEventListener('proposalReject', onProposalReject);
      window.removeEventListener('proposalAcceptAll', onProposalAcceptAll);
      window.removeEventListener('proposalRejectAll', onProposalRejectAll);
    };
  }, [documentId, fetchDocument, refreshWithScrollCapture, syncPendingProposalsFromDb]);

  // Load pending proposals from DB when document opens or when entering edit mode (single path).
  useEffect(() => {
    if (!documentId || !isEditing) return;
    const timer = setTimeout(() => {
      syncPendingProposalsFromDb('initial');
    }, 200);
    return () => clearTimeout(timer);
  }, [documentId, isEditing, syncPendingProposalsFromDb]);

  // Re-sync pending proposals when agent stream completes
  useEffect(() => {
    if (!documentId || !isEditing) return;
    const refetch = () => syncPendingProposalsFromDb('agent_stream_complete');
    window.addEventListener('agentStreamComplete', refetch);
    return () => window.removeEventListener('agentStreamComplete', refetch);
  }, [documentId, isEditing, syncPendingProposalsFromDb]);

  // Auto-save content to backend immediately when a proposal is accepted (avoids race with WebSocket refresh)
  useEffect(() => {
    const handleApplied = async (e) => {
      const { documentId: docId, content } = e.detail || {};
      if (!docId || docId !== documentId || !content) return;

      acceptShieldUntilRef.current = Date.now() + 2000;

      try {
        await apiService.updateDocumentContent(docId, content);
        setServerContent(content);
        clearUnsavedContent(docId);
      } catch (err) {
        console.error('Failed to auto-save after accept:', err);
        saveUnsavedContent(docId, content);
      }
    };
    window.addEventListener('liveEditApplied', handleApplied);
    return () => window.removeEventListener('liveEditApplied', handleApplied);
  }, [documentId]);

  // Extract headers from document content for navigation dropdown
  const headers = React.useMemo(() => {
    if (!isEditing || !editContent || !document?.filename) return [];
    
    const fnameLower = document.filename.toLowerCase();
    const lines = editContent.split('\n');
    const extractedHeaders = [];
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      if (fnameLower.endsWith('.md')) {
        // Markdown headers: #, ##, ###
        const match = line.match(/^(#{1,3})\s+(.+)$/);
        if (match) {
          const level = match[1].length;
          const text = match[2].trim();
          extractedHeaders.push({
            level,
            text,
            lineNumber: i + 1
          });
        }
      } else if (fnameLower.endsWith('.org')) {
        // Org headers: *, **, ***
        const match = line.match(/^(\*{1,3})\s+(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)?\s*(.+)$/i);
        if (match) {
          const level = match[1].length;
          const text = match[3]?.trim() || match[2]?.trim() || line.trim();
          extractedHeaders.push({
            level,
            text,
            lineNumber: i + 1
          });
        }
      }
    }
    
    return extractedHeaders;
  }, [isEditing, editContent, document?.filename]);

  // Handle header navigation
  const handleHeaderNavigation = (lineNumber) => {
    const fnameLower = (document?.filename || '').toLowerCase();
    
    if (fnameLower.endsWith('.md')) {
      // Use markdown editor ref
      if (markdownEditorRef.current?.scrollToLine) {
        markdownEditorRef.current.scrollToLine(lineNumber);
      }
    } else if (fnameLower.endsWith('.org')) {
      // Use org editor ref
      if (orgEditorRef.current?.scrollToLine) {
        orgEditorRef.current.scrollToLine(lineNumber);
      }
    }
  };

  // Find current section based on cursor position
  const findCurrentSection = React.useCallback((content, cursorOffset, filename) => {
    if (!content || cursorOffset === null || cursorOffset === undefined || !filename) {
      return null;
    }

    const fnameLower = filename.toLowerCase();
    const lines = content.split('\n');
    
    // Find the line number where cursor is
    let currentLine = 1;
    let charCount = 0;
    for (let i = 0; i < lines.length; i++) {
      const lineLength = lines[i].length + 1; // +1 for newline
      if (charCount + lineLength > cursorOffset) {
        currentLine = i + 1;
        break;
      }
      charCount += lineLength;
    }

    // Search backwards from current line to find the most recent heading
    let foundHeading = null;
    for (let i = currentLine - 1; i >= 0; i--) {
      const line = lines[i];
      
      if (fnameLower.endsWith('.md')) {
        // Markdown headers: #, ##, ###
        const match = line.match(/^(#{1,3})\s+(.+)$/);
        if (match) {
          foundHeading = match[2].trim();
          break;
        }
      } else if (fnameLower.endsWith('.org')) {
        // Org headers: *, **, ***
        const match = line.match(/^(\*{1,3})\s+(TODO|NEXT|STARTED|WAITING|HOLD|DONE|CANCELED|CANCELLED)?\s*(.+)$/i);
        if (match) {
          foundHeading = match[3]?.trim() || match[2]?.trim() || line.trim();
          break;
        }
      }
    }

    return foundHeading;
  }, []);

  // Callback to update current section from editors
  const handleCurrentSectionChange = React.useCallback((content, cursorOffset) => {
    if (!document?.filename || !isEditing) {
      setCurrentSection(null);
      return;
    }
    const section = findCurrentSection(content, cursorOffset, document.filename);
    setCurrentSection(section);
  }, [document?.filename, isEditing, findCurrentSection]);

  // Reset current section when switching documents or exiting edit mode
  useEffect(() => {
    if (!isEditing || !document?.filename) {
      setCurrentSection(null);
    }
  }, [isEditing, document?.filename]);

  // Handle scrolling to specific line or heading
  useEffect(() => {
    if (!document || loading) return;

    // Give the DOM time to render
    const scrollTimeout = setTimeout(() => {
      if (scrollToHeading && contentBoxRef.current) {
        // Scroll to heading in preview pane (for non-edit mode)
        // Edit mode scrolling is handled by OrgCMEditor
        console.log('📍 Scrolling preview to heading:', scrollToHeading);
        try {
          const headingText = scrollToHeading.toLowerCase().trim();
          const allHeadings = contentBoxRef.current.querySelectorAll('[id^="org-heading-"]');
          
          for (const heading of allHeadings) {
            const titleElement = heading.querySelector('span[style*="fontWeight: 600"]');
            if (titleElement && titleElement.textContent.toLowerCase().trim() === headingText) {
              heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
              // Highlight briefly
              const originalBg = heading.style.backgroundColor;
              heading.style.backgroundColor = '#fff3cd';
              setTimeout(() => {
                heading.style.backgroundColor = originalBg;
              }, 1500);
              return;
            }
          }
          console.warn('⚠️ Heading not found in preview:', scrollToHeading);
        } catch (err) {
          console.error('❌ Error scrolling preview:', err);
        }
      } else if (scrollToLine !== null && contentBoxRef.current) {
        // Scroll to specific line number (approximate)
        console.log('📍 Scrolling to line:', scrollToLine);
        const contentBox = contentBoxRef.current;
        const lineHeight = 20; // Approximate line height
        const targetY = scrollToLine * lineHeight;
        contentBox.scrollTo({ top: targetY, behavior: 'smooth' });
      }
    }, 300);

    return () => clearTimeout(scrollTimeout);
  }, [document, loading, scrollToLine, scrollToHeading]);

  // Fetch backlinks for org files
  useEffect(() => {
    const fetchBacklinks = async () => {
      if (!document || !document.filename) return;
      
      const fname = document.filename.toLowerCase();
      if (!fname.endsWith('.org')) return;
      
      try {
        setLoadingBacklinks(true);
        console.log('🔗 Fetching backlinks for', document.filename);
        
        const response = await apiService.get(`/api/org/backlinks?filename=${encodeURIComponent(document.filename)}`);
        
        if (response.success && response.backlinks) {
          setBacklinks(response.backlinks);
          console.log(`✅ Found ${response.backlinks.length} backlinks`);
        }
      } catch (err) {
        console.error('Failed to fetch backlinks:', err);
        // Fail silently - backlinks are supplementary information
      } finally {
        setLoadingBacklinks(false);
      }
    };
    
    fetchBacklinks();
  }, [document]);

  // Refile hotkey: Ctrl+Shift+M to refile current heading
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check for Ctrl+Shift+M (or Cmd+Shift+M on Mac)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'M') {
        event.preventDefault();
        
        const fname = (document?.filename || '').toLowerCase();
        if (!fname.endsWith('.org')) return;
        
        // Open refile dialog with current position
        console.log('📦 Refile hotkey triggered');
        
        // Get current cursor position from editor (dynamic!)
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        console.log('📦 Current cursor at line:', currentLine, 'heading:', currentHeading);
        
        // Get relative file path from document
        // Try to construct proper path with folder
        let filePath = document.filename;
        
        // If we have folder info, include it
        if (document.folder_id && document.folder_name) {
          filePath = `${document.folder_name}/${document.filename}`;
        } else {
          // Default to OrgMode folder for org files
          filePath = `OrgMode/${document.filename}`;
        }
        
        console.log('📦 Refile source file path:', filePath);
        
        setRefileSourceFile(filePath);
        setRefileSourceLine(currentLine);
        setRefileSourceHeading(currentHeading);
        setRefileDialogOpen(true);
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading]);

  // Archive hotkey: Ctrl+Shift+A to archive current heading
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check for Ctrl+Shift+A (or Cmd+Shift+A on Mac)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'A') {
        event.preventDefault();
        
        const fname = (document?.filename || '').toLowerCase();
        if (!fname.endsWith('.org')) return;
        
        // Open archive dialog with current position
        console.log('📦 Archive hotkey triggered');
        
        // Get current cursor position from editor (dynamic!)
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        console.log('📦 Current cursor at line:', currentLine, 'heading:', currentHeading);
        
        // Get relative file path from document
        let filePath = document.filename;
        
        // If we have folder info, include it
        if (document.folder_id && document.folder_name) {
          filePath = `${document.folder_name}/${document.filename}`;
        } else {
          // Default to OrgMode folder for org files
          filePath = `OrgMode/${document.filename}`;
        }
        
        console.log('📦 Archive source file path:', filePath);
        
        setArchiveSourceFile(filePath);
        setArchiveSourceLine(currentLine);
        setArchiveSourceHeading(currentHeading);
        setArchiveDialogOpen(true);
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading]);

  // Clocking hotkeys: Ctrl+Shift+I (clock in) and Ctrl+Shift+O (clock out)
  useEffect(() => {
    const handleKeyDown = async (event) => {
      const fname = (document?.filename || '').toLowerCase();
      if (!fname.endsWith('.org')) return;
      
      // Clock In: Ctrl+Shift+I
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'I') {
        event.preventDefault();
        
        console.log('⏰ Clock in hotkey triggered');
        
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        let filePath = document.filename;
        if (document.folder_id && document.folder_name) {
          filePath = `${document.folder_name}/${document.filename}`;
        } else {
          filePath = `OrgMode/${document.filename}`;
        }
        
        try {
          const response = await apiService.post('/api/org/clock/in', {
            file_path: filePath,
            line_number: currentLine,
            heading: currentHeading
          });
          
          if (response.success) {
            console.log('✅ Clocked in:', currentHeading);
            alert(`⏰ Clocked in to:\n${currentHeading}`);
            setActiveClock(response);
          } else {
            alert(`⚠️ ${response.message}`);
          }
        } catch (err) {
          console.error('❌ Clock in failed:', err);
          alert(`❌ Clock in failed: ${err.message}`);
        }
      }
      
      // Clock Out: Ctrl+Shift+O
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'O') {
        event.preventDefault();
        
        console.log('⏰ Clock out hotkey triggered');
        
        try {
          const response = await apiService.post('/api/org/clock/out');
          
          if (response.success) {
            console.log('✅ Clocked out:', response.duration_display);
            alert(`⏰ Clocked out!\n\nDuration: ${response.duration_display}\nTask: ${response.heading}`);
            setActiveClock(null);
            // Refresh file to show LOGBOOK entry
            fetchDocument();
          } else {
            alert(`⚠️ ${response.message}`);
          }
        } catch (err) {
          console.error('❌ Clock out failed:', err);
          alert(`❌ Clock out failed: ${err.message}`);
        }
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading, fetchDocument]);

  // Check for active clock on mount
  useEffect(() => {
    const checkActiveClock = async () => {
      try {
        const response = await apiService.get('/api/org/clock/active');
        if (response.success && response.active_clock) {
          setActiveClock(response.active_clock);
        }
      } catch (err) {
        console.error('❌ Failed to check active clock:', err);
      } finally {
        setCheckingClock(false);
      }
    };
    
    checkActiveClock();
  }, []);

  // Tag hotkey - Ctrl+Shift+E to add tags to current heading
  useEffect(() => {
    const handleKeyDown = (event) => {
      // Check for Ctrl+Shift+E (or Cmd+Shift+E on Mac)
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key === 'E') {
        event.preventDefault();
        
        const fname = (document?.filename || '').toLowerCase();
        if (!fname.endsWith('.org')) return;
        
        console.log('🏷️ Tag hotkey triggered!');
        
        // Get current cursor position from editor
        const currentLine = orgEditorRef.current?.getCurrentLine() || scrollToLine || 1;
        const currentHeading = orgEditorRef.current?.getCurrentHeading() || scrollToHeading || 'Current entry';
        
        console.log('🏷️ Current cursor at line:', currentLine, 'heading:', currentHeading);
        
        setTagSourceLine(currentLine);
        setTagSourceHeading(currentHeading);
        setTagDialogOpen(true);
      }
    };
    
    if (isEditing && document) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isEditing, document, scrollToLine, scrollToHeading]);
  
  // Handle refile dialog close
  const handleRefileClose = (result) => {
    setRefileDialogOpen(false);
    
    if (result?.success) {
      // Refresh the document after successful refile
      console.log('✅ Refile completed, refreshing...');
      fetchDocument();
    }
  };

  // Handle archive dialog close
  const handleArchiveClose = (result) => {
    setArchiveDialogOpen(false);
    
    if (result?.success) {
      // Refresh the document after successful archive
      console.log('✅ Archive completed, refreshing...');
      fetchDocument();
    }
  };

  // Handle tag dialog close
  const handleTagClose = () => {
    setTagDialogOpen(false);
    // Note: Tag dialog handles refresh internally via window.location.reload()
  };

  // Publish editor context (must be declared before any early returns)
  useEffect(() => {
    const fname = (document?.filename || '').toLowerCase();
    if (isEditing && (fname.endsWith('.md') || fname.endsWith('.org'))) {
      // Parse frontmatter for proper editor context (org files may have frontmatter too)
      const parsed = parseFrontmatter(editContent || '');
      
      // Merge data and lists so array fields (files, components, etc.) are included!
      const mergedFrontmatter = { ...(parsed.data || {}), ...(parsed.lists || {}) };
      
      const editorStatePayload = {
        isEditable: true,
        filename: document?.filename || null,
        language: fname.endsWith('.org') ? 'org' : 'markdown',
        content: editContent,
        contentLength: (editContent || '').length,
        frontmatter: mergedFrontmatter,
        canonicalPath: document?.canonical_path || null,
        documentId: document?.document_id || null,
        folderId: document?.folder_id || null,
        cursorOffset: -1,
        selectionStart: -1,
        selectionEnd: -1,
      };
      
      setEditorState((prev) => ({
        ...prev,
        ...editorStatePayload
      }));
      
      // Also write to localStorage for ChatSidebar to read (backup to MarkdownCMEditor)
      // This ensures frontmatter is correct even if MarkdownCMEditor hasn't mounted yet
      try {
        localStorage.setItem('editor_ctx_cache', JSON.stringify(editorStatePayload));
      } catch (e) {
        console.error('Failed to update editor_ctx_cache from DocumentViewer:', e);
      }
    } else {
      setEditorState({
        isEditable: false,
        filename: null,
        language: null,
        content: null,
        contentLength: 0,
        frontmatter: null,
        cursorOffset: -1,
        selectionStart: -1,
        selectionEnd: -1,
        canonicalPath: null,
        documentId: null,
        folderId: null
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isEditing, editContent, document?.filename, document?.document_id]);

  // Clear React editor state when this tab unmounts (e.g. user switched to another tab).
  // Do NOT clear localStorage editor_ctx_cache here: chat sends messages with the last
  // open document context, and we only have one cache. Clearing on unmount caused
  // has_active_editor=false after switching tabs (e.g. outline open, switch to RSS, send
  // from chat). Cache is cleared when leaving /documents (App.js) so Chat page doesn't
  // send stale context. Diffs persist in documentDiffStore for when the tab reopens.
  useEffect(() => {
    return () => {
      setEditorState({
        isEditable: false,
        filename: null,
        language: null,
        content: null,
        contentLength: 0,
        frontmatter: null,
        cursorOffset: -1,
        selectionStart: -1,
        selectionEnd: -1,
        canonicalPath: null,
        documentId: null,
        folderId: null,
      });
    };
  }, [setEditorState]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Handle title click to start editing
  const handleTitleClick = () => {
    setEditedTitle(document.title || document.filename);
    setEditingTitle(true);
  };

  // Handle filename click to start editing
  const handleFilenameClick = () => {
    setEditedFilename(document.filename);
    setEditingFilename(true);
  };

  // Save title changes
  const handleSaveTitle = async () => {
    if (!editedTitle.trim() || editedTitle === (document.title || document.filename)) {
      setEditingTitle(false);
      return;
    }

    try {
      setUpdatingMetadata(true);
      await apiService.updateDocumentMetadata(documentId, { title: editedTitle.trim() });
      console.log('Updated document title');
      
      // Refresh document to get updated data
      await fetchDocument();
      setEditingTitle(false);
    } catch (err) {
      console.error('Failed to update title:', err);
      alert(`Failed to update title: ${err.message}`);
    } finally {
      setUpdatingMetadata(false);
    }
  };

  // Save filename changes
  const handleSaveFilename = async () => {
    if (!editedFilename.trim() || editedFilename === document.filename) {
      setEditingFilename(false);
      return;
    }

    try {
      setUpdatingMetadata(true);
      const resp = await apiService.renameDocument(documentId, editedFilename.trim());
      const displayName = resp?.new_filename ?? editedFilename.trim();
      window.tabbedContentManagerRef?.updateDocumentTabTitle?.(documentId, displayName);
      await fetchDocument();
      setEditingFilename(false);
    } catch (err) {
      console.error('Failed to rename file:', err);
      alert(`Failed to rename file: ${err.message}`);
    } finally {
      setUpdatingMetadata(false);
    }
  };

  // Handle escape key to cancel editing
  const handleKeyDown = (e, type) => {
    if (e.key === 'Escape') {
      if (type === 'title') {
        setEditingTitle(false);
      } else {
        setEditingFilename(false);
      }
    } else if (e.key === 'Enter') {
      if (type === 'title') {
        handleSaveTitle();
      } else {
        handleSaveFilename();
      }
    }
  };

  // Download menu handlers
  const handleDownloadMenuOpen = (event) => {
    setDownloadMenuAnchor(event.currentTarget);
  };

  const handleDownloadMenuClose = () => {
    setDownloadMenuAnchor(null);
  };

  // Download document handler (use window['document'] - state variable 'document' shadows global)
  const handleDownload = async () => {
    handleDownloadMenuClose();
    const doc = typeof window !== 'undefined' ? window['document'] : null;
    if (!doc) return;
    try {
      const token = localStorage.getItem('auth_token') || localStorage.getItem('token');
      const fnameLower = (document?.filename || '').toLowerCase();
      
      // For PDFs, use the PDF endpoint
      if (fnameLower.endsWith('.pdf')) {
        const response = await fetch(`/api/documents/${documentId}/pdf`, {
          headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        
        if (!response.ok) {
          throw new Error('Failed to download PDF');
        }
        
        const blob = await response.blob();
        const blobUrl = URL.createObjectURL(blob);
        const link = doc.createElement('a');
        link.href = blobUrl;
        link.download = document.filename || 'document.pdf';
        doc.body.appendChild(link);
        link.click();
        doc.body.removeChild(link);
        URL.revokeObjectURL(blobUrl);
      } else {
        // For other files, try the file endpoint first
        let response = await fetch(`/api/documents/${documentId}/file`, {
          headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });
        
        // If file endpoint fails, create blob from content
        if (!response.ok) {
          const content = isEditing ? editContent : (document.content || '');
          const blob = new Blob([content], { type: 'text/plain' });
          const blobUrl = URL.createObjectURL(blob);
          const link = doc.createElement('a');
          link.href = blobUrl;
          link.download = document.filename || 'document';
          doc.body.appendChild(link);
          link.click();
          doc.body.removeChild(link);
          URL.revokeObjectURL(blobUrl);
        } else {
          const blob = await response.blob();
          const blobUrl = URL.createObjectURL(blob);
          const link = doc.createElement('a');
          link.href = blobUrl;
          link.download = document.filename || 'document';
          doc.body.appendChild(link);
          link.click();
          doc.body.removeChild(link);
          URL.revokeObjectURL(blobUrl);
        }
      }
    } catch (err) {
      console.error('Failed to download document:', err);
      alert('Failed to download document');
    }
  };

  const openPdfExportDialog = () => {
    handleDownloadMenuClose();
    setPdfExportOrientation('portrait');
    setPdfIncludeToc(false);
    setPdfTocDepth(3);
    setPdfPageNumbers(true);
    setPdfWatermarkText('');
    setPdfWatermarkAllPages(true);
    setPdfPageBreakHeadings([]);
    setPdfLayout('article');
    setPdfFontPreset('liberation');
    setPdfTypefaceStyle('mixed');
    setPdfHeadingOutline([]);
    setPdfSectionOverrides({});
    setPdfBookMarginTopMm(20);
    setPdfBookMarginRightMm(20);
    setPdfBookMarginBottomMm(28);
    setPdfBookMarginLeftMm(20);
    setPdfBookIndentBody(true);
    setPdfBookNoIndentAfterHeading(true);
    setPdfBookPageNumFormat('n_of_total');
    setPdfBookPageNumVertical('bottom');
    setPdfBookPageNumHorizontal('center');
    setPdfBookSuppressFirstPage(false);
    setPdfExportDialogOpen(true);
  };

  useEffect(() => {
    if (!pdfExportDialogOpen || pdfLayout !== 'book' || pdfPageBreakHeadings.length === 0) {
      setPdfHeadingOutline([]);
      setPdfOutlineLoading(false);
      return undefined;
    }
    const md = editContent || document?.content || '';
    if (pdfOutlineTimerRef.current) {
      clearTimeout(pdfOutlineTimerRef.current);
    }
    pdfOutlineTimerRef.current = setTimeout(async () => {
      try {
        setPdfOutlineLoading(true);
        const res = await exportService.fetchPdfHeadingOutline(md, {
          sourceFormat: pdfSourceFormat,
        });
        setPdfHeadingOutline(Array.isArray(res.headings) ? res.headings : []);
      } catch (e) {
        console.error('PDF heading outline failed:', e);
        setPdfHeadingOutline([]);
      } finally {
        setPdfOutlineLoading(false);
      }
    }, 400);
    return () => {
      if (pdfOutlineTimerRef.current) {
        clearTimeout(pdfOutlineTimerRef.current);
      }
    };
  }, [
    pdfExportDialogOpen,
    pdfLayout,
    pdfPageBreakHeadings,
    editContent,
    document?.content,
    pdfSourceFormat,
  ]);

  const handleConfirmPdfExport = async () => {
    try {
      setExporting(true);
      const content = editContent || document.content || '';
      const bookSectionOverrides = Object.entries(pdfSectionOverrides)
        .filter(([, v]) => v?.enabled)
        .map(([headingId, v]) => ({
          heading_id: headingId,
          page_numbers: v.pageNumbers === 'inherit' ? null : v.pageNumbers === 'on',
          plain_first_page: !!v.plainFirstPage,
        }));
      await exportService.exportMarkdownAsPDF(content, {
        filename: document.filename || 'document',
        pageOrientation: pdfExportOrientation,
        includeToc: pdfIncludeToc,
        tocDepth: pdfTocDepth,
        pageNumbers: pdfPageNumbers,
        watermarkText: pdfWatermarkText.trim() || null,
        watermarkOnAllPages: pdfWatermarkAllPages,
        pageBreakBeforeHeadings: pdfPageBreakHeadings,
        pdfLayout,
        bookMarginTopMm: pdfBookMarginTopMm,
        bookMarginRightMm: pdfBookMarginRightMm,
        bookMarginBottomMm: pdfBookMarginBottomMm,
        bookMarginLeftMm: pdfBookMarginLeftMm,
        bookIndentBodyParagraphs: pdfBookIndentBody,
        bookNoIndentAfterSectionHeading: pdfBookNoIndentAfterHeading,
        bookPageNumberFormat: pdfBookPageNumFormat,
        bookPageNumberVertical: pdfBookPageNumVertical,
        bookPageNumberHorizontal: pdfBookPageNumHorizontal,
        bookSuppressPageNumberOnFirstPage: pdfBookSuppressFirstPage,
        bookSectionOverrides,
        pdfSourceFormat,
        pdfFontPreset,
        pdfTypefaceStyle,
      });
      setPdfExportDialogOpen(false);
    } catch (error) {
      console.error('PDF export failed:', error);
      alert(`PDF export failed: ${error.message}`);
    } finally {
      setExporting(false);
    }
  };

  // Export as EPUB handler
  const handleExportEpub = () => {
    handleDownloadMenuClose();
    // Parse frontmatter to pre-fill metadata
    const parsed = parseFrontmatter(editContent || document.content || '');
    const mergedFrontmatter = { ...(parsed.data || {}), ...(parsed.lists || {}) };
    
    // Pre-fill title from frontmatter or document
    setEpubTitle(mergedFrontmatter.title || mergedFrontmatter.Title || (document.title || document.filename || '').replace(/\.[^.]+$/, ''));
    
    // Pre-fill author fields from frontmatter
    if (mergedFrontmatter.author_first || mergedFrontmatter['Author First']) {
      setEpubAuthorFirst(mergedFrontmatter.author_first || mergedFrontmatter['Author First'] || '');
    } else {
      setEpubAuthorFirst('');
    }
    
    if (mergedFrontmatter.author_last || mergedFrontmatter['Author Last']) {
      setEpubAuthorLast(mergedFrontmatter.author_last || mergedFrontmatter['Author Last'] || '');
    } else {
      // Try to split existing author field if present
      const author = mergedFrontmatter.author || mergedFrontmatter.Author || document.author || '';
      if (author) {
        const parts = author.trim().split(/\s+/);
        if (parts.length >= 2) {
          setEpubAuthorFirst(parts[0]);
          setEpubAuthorLast(parts.slice(1).join(' '));
        } else {
          setEpubAuthorFirst(author);
          setEpubAuthorLast('');
        }
      } else {
        setEpubAuthorLast('');
      }
    }
    
    setEpubLanguage(mergedFrontmatter.language || mergedFrontmatter.Language || 'en');
    setExportOpen(true);
  };

  // Fullscreen handlers
  const handleToggleFullscreen = async () => {
    if (!fullscreenContainerRef.current) return;
    
    // Guard: ensure we're in a browser environment
    if (typeof document === 'undefined' || !document) {
      console.warn('Fullscreen API not available - not in browser environment');
      return;
    }

    const element = fullscreenContainerRef.current;
    const isCurrentlyFullscreen = !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement
    );

    try {
      if (!isCurrentlyFullscreen) {
        // Enter fullscreen - try different vendor prefixes
        if (element.requestFullscreen) {
          await element.requestFullscreen();
        } else if (element.webkitRequestFullscreen) {
          await element.webkitRequestFullscreen();
        } else if (element.mozRequestFullScreen) {
          await element.mozRequestFullScreen();
        } else if (element.msRequestFullscreen) {
          await element.msRequestFullscreen();
        } else {
          console.warn('Fullscreen API not supported');
          return;
        }
        setIsFullscreen(true);
      } else {
        // Exit fullscreen - try different vendor prefixes
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
          await document.webkitExitFullscreen();
        } else if (document.mozCancelFullScreen) {
          await document.mozCancelFullScreen();
        } else if (document.msExitFullscreen) {
          await document.msExitFullscreen();
        }
        setIsFullscreen(false);
      }
    } catch (err) {
      console.error('Fullscreen error:', err);
      setIsFullscreen(false);
    }
  };

  // Listen for fullscreen changes (user might exit via ESC key)
  useEffect(() => {
    // Guard: ensure we're in a browser environment
    if (typeof document === 'undefined' || !document) {
      return;
    }

    const handleFullscreenChange = () => {
      const isCurrentlyFullscreen = !!(
        document.fullscreenElement ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement ||
        document.msFullscreenElement
      );
      setIsFullscreen(isCurrentlyFullscreen);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      if (typeof document !== 'undefined' && document) {
        document.removeEventListener('fullscreenchange', handleFullscreenChange);
        document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
        document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
        document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
      }
    };
  }, []);

  const canShowVersionHistory = document && canUserEditDocument(document);

  const openVersionHistory = async () => {
    if (!documentId || !document) return;
    setVersionHistoryOpen(true);
    setVersionsLoading(true);
    setVersionsList([]);
    setVersionViewContent(null);
    setVersionDiffResult(null);
    try {
      const res = await apiService.getDocumentVersions(documentId);
      setVersionsList(res.versions || []);
    } catch (e) {
      console.error('Failed to load versions', e);
    } finally {
      setVersionsLoading(false);
    }
  };

  const handleViewVersion = async (version) => {
    try {
      const res = await apiService.getVersionContent(documentId, version.version_id);
      setVersionViewContent(res.content);
      setVersionViewMeta(version);
      setVersionDiffResult(null);
    } catch (e) {
      console.error('Failed to load version content', e);
    }
  };

  const handleDiffWithCurrent = async (version) => {
    const currentVersion = versionsList.find((v) => v.is_current) || versionsList[0];
    if (!currentVersion || currentVersion.version_id === version.version_id) return;
    try {
      const res = await apiService.diffVersions(documentId, version.version_id, currentVersion.version_id);
      setVersionDiffResult(res);
      setVersionViewContent(null);
      setVersionViewMeta({ from: version, to: currentVersion });
    } catch (e) {
      console.error('Failed to load diff', e);
    }
  };

  const handleRollbackClick = (version) => {
    setRollbackConfirmVersion(version);
  };

  const handleRollbackConfirm = async () => {
    if (!rollbackConfirmVersion || !documentId) return;
    setRollbacking(true);
    try {
      await apiService.rollbackToVersion(documentId, rollbackConfirmVersion.version_id);
      setRollbackConfirmVersion(null);
      setVersionHistoryOpen(false);
      fetchDocument(true);
    } catch (e) {
      console.error('Rollback failed', e);
      alert('Rollback failed: ' + (e.message || e));
    } finally {
      setRollbacking(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!document) {
    return (
      <Box sx={{ p: 2 }}>
        <Alert severity="info">No document selected</Alert>
      </Box>
    );
  }

  const fnameLower = (document.filename || '').toLowerCase();

  // Use pending scroll after refresh so remounted editor restores position (avoids parent state timing)
  const effectiveInitialScroll = pendingScrollAfterRefreshRef.current ?? initialScrollPosition;

  // DocX files get specialized viewer with formatting preserved
  if (fnameLower.endsWith('.docx')) {
    return (
      <DocxViewer 
        documentId={documentId}
        filename={document.filename}
      />
    );
  }

  // PowerPoint files get specialized slide viewer
  if (fnameLower.endsWith('.pptx') || fnameLower.endsWith('.ppt')) {
    return (
      <PptxViewer
        documentId={documentId}
        filename={document.filename}
      />
    );
  }

  // EML email files get specialized viewer with headers and body
  if (fnameLower.endsWith('.eml')) {
    return (
      <EMLViewer 
        documentId={documentId}
        filename={document.filename}
      />
    );
  }

  // PDF files get special full-screen viewer treatment
  if (fnameLower.endsWith('.pdf')) {
    return (
      <PDFDocumentViewer 
        documentId={documentId}
        filename={document.filename}
      />
    );
  }

  // Image files get image viewer treatment
  const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg'];
  const isImageFile = imageExtensions.some(ext => fnameLower.endsWith(ext));
  
  if (isImageFile && document) {
    return <ImageViewer documentId={documentId} filename={document.filename} title={document.title} onDownload={handleDownload} />;
  }

  // Audio files get audio player treatment
  const audioExtensions = ['.mp3', '.aac', '.wav', '.flac', '.ogg', '.m4a', '.wma', '.opus'];
  const isAudioFile = audioExtensions.some(ext => fnameLower.endsWith(ext));
  
  if (isAudioFile && document) {
    const audioUrl = `/api/documents/${documentId}/file`;
    return (
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
          <Typography variant="h6">{document.filename || document.title || 'Audio File'}</Typography>
          {document.title && document.title !== document.filename && (
            <Typography variant="body2" color="text.secondary">{document.title}</Typography>
          )}
        </Box>
        <Box sx={{ flex: 1, p: 3, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Box sx={{ width: '100%', maxWidth: '800px' }}>
            <AudioPlayer
              src={audioUrl}
              filename={document.filename}
            />
          </Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box ref={fullscreenContainerRef} sx={{ height: '100%', overflow: 'hidden' }}>
      {/* Single scroll area inside the viewer */}
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* External Update Notification */}
        {externalUpdateNotification && (
          <Alert 
            severity="info" 
            onClose={() => setExternalUpdateNotification(null)}
            sx={{ 
              mx: 2, 
              mt: 1, 
              mb: 0,
              '& .MuiAlert-message': {
                fontSize: '0.875rem'
              }
            }}
          >
            {externalUpdateNotification.message}
          </Alert>
        )}
        
        {/* Compact Header */}
        <Box sx={{ px: 2, py: 1, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 1 }}>
          <Box sx={{ minWidth: 0, overflow: 'hidden', flex: 1, display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
            {/* Editable Filename */}
            {editingFilename ? (
              <TextField
                value={editedFilename}
                onChange={(e) => setEditedFilename(e.target.value)}
                onBlur={handleSaveFilename}
                onKeyDown={(e) => handleKeyDown(e, 'filename')}
                autoFocus
                size="small"
                disabled={updatingMetadata}
                placeholder="Enter filename..."
                sx={{ minWidth: 200 }}
              />
            ) : (
              <>
                {isEditing && currentSection ? (
                  <Typography 
                    variant="caption" 
                    color="text.secondary" 
                    sx={{ 
                      whiteSpace: 'nowrap', 
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis',
                      fontWeight: 500,
                      fontStyle: 'italic'
                    }}
                    title={`Current section: ${currentSection}`}
                  >
                    {currentSection}
                  </Typography>
                ) : (
                  <Typography 
                    variant="caption" 
                    color="text.secondary" 
                    onClick={handleFilenameClick}
                    sx={{ 
                      whiteSpace: 'nowrap', 
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis',
                      cursor: 'pointer',
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: 'action.hover',
                        borderRadius: 1,
                        px: 0.5
                      }
                    }}
                    title="Click to rename file"
                  >
                    {document.filename}
                  </Typography>
                )}
                
                {/* Header Navigation Dropdown */}
                {isEditing && headers.length > 0 && (
                  <FormControl 
                    size="small" 
                    sx={{ 
                      minWidth: 180,
                      '& .MuiOutlinedInput-root': {
                        fontSize: '0.75rem',
                        height: '24px',
                        '& .MuiSelect-select': {
                          padding: '4px 32px 4px 8px',
                          fontSize: '0.75rem'
                        }
                      }
                    }}
                  >
                    <Select
                      value=""
                      displayEmpty
                      renderValue={() => (
                        <Typography variant="caption" color="text.secondary">
                          Jump to...
                        </Typography>
                      )}
                      onChange={(e) => {
                        const lineNumber = parseInt(e.target.value);
                        if (lineNumber) {
                          handleHeaderNavigation(lineNumber);
                        }
                      }}
                      sx={{
                        fontSize: '0.75rem',
                        color: 'text.secondary',
                        '& .MuiSelect-icon': {
                          fontSize: '1rem'
                        }
                      }}
                    >
                      <MenuItem value="" disabled>
                        <Typography variant="caption" color="text.secondary">
                          Jump to...
                        </Typography>
                      </MenuItem>
                      {headers.map((header, idx) => {
                        const fnameLower = (document?.filename || '').toLowerCase();
                        const prefix = fnameLower.endsWith('.org') ? '*'.repeat(header.level) : '#'.repeat(header.level);
                        return (
                          <MenuItem key={idx} value={header.lineNumber}>
                            <Typography 
                              variant="caption" 
                              sx={{ 
                                pl: (header.level - 1) * 1.5,
                                fontWeight: header.level === 1 ? 600 : header.level === 2 ? 500 : 400
                              }}
                            >
                              {prefix} {header.text}
                            </Typography>
                          </MenuItem>
                        );
                      })}
                    </Select>
                  </FormControl>
                )}
                
                {/* Word Count and Reading Time */}
                {isEditing && editContent && (
                  <Typography 
                    variant="caption" 
                    color="text.secondary"
                    sx={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 0.75,
                      opacity: 0.8,
                      whiteSpace: 'nowrap'
                    }}
                  >
                    {(() => {
                      // Calculate word count (excluding frontmatter for markdown files)
                      let textForCount = editContent;
                      if (document.filename?.toLowerCase().endsWith('.md')) {
                        // Remove frontmatter (---...--- at start)
                        textForCount = editContent.replace(/^---\n[\s\S]*?\n---\n/, '');
                      }
                      const wordCount = textForCount.trim().split(/\s+/).filter(word => word.length > 0).length;
                      const readingTime = Math.max(1, Math.ceil(wordCount / 200)); // Average reading speed: 200 words/min
                      
                      // Format last saved timestamp
                      let lastSavedText = null;
                      if (document.updated_at) {
                        try {
                          const updatedDate = new Date(document.updated_at);
                          if (!isNaN(updatedDate.getTime())) {
                            // Format as MM/DD/YYYY HH:MM:SS AM/PM
                            const month = String(updatedDate.getMonth() + 1).padStart(2, '0');
                            const day = String(updatedDate.getDate()).padStart(2, '0');
                            const year = updatedDate.getFullYear();
                            let hours = updatedDate.getHours();
                            const minutes = String(updatedDate.getMinutes()).padStart(2, '0');
                            const seconds = String(updatedDate.getSeconds()).padStart(2, '0');
                            const ampm = hours >= 12 ? 'PM' : 'AM';
                            hours = hours % 12;
                            hours = hours ? hours : 12; // the hour '0' should be '12'
                            const hoursStr = String(hours).padStart(2, '0');
                            lastSavedText = `Saved: ${month}/${day}/${year} ${hoursStr}:${minutes}:${seconds} ${ampm}`;
                          }
                        } catch (e) {
                          console.error('Failed to format updated_at:', e);
                        }
                      }
                      
                      return (
                        <>
                          <span>{wordCount.toLocaleString()} words</span>
                          <span>•</span>
                          <span>{readingTime} min read</span>
                          {lastSavedText && (
                            <>
                              <span>•</span>
                              <span>{lastSavedText}</span>
                            </>
                          )}
                        </>
                      );
                    })()}
                  </Typography>
                )}
              </>
            )}
          </Box>

          {/* Active clock indicator */}
          {activeClock && (
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 1, 
              px: 2, 
              py: 0.5, 
              bgcolor: 'primary.main', 
              color: 'primary.contrastText',
              borderRadius: 1,
              fontSize: '0.875rem'
            }}>
              <Schedule fontSize="small" />
              <Box>
                <Typography variant="caption" sx={{ display: 'block', fontWeight: 600, color: 'inherit' }}>
                  ⏰ Clocked In
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', fontSize: '0.75rem', color: 'inherit', opacity: 0.9 }}>
                  {activeClock.heading} ({activeClock.elapsed_display || '0:00'})
                </Typography>
              </Box>
            </Box>
          )}

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexShrink: 0 }}>
            {(document.filename && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org'))) && canUserEditDocument(document) && (
              !isEditing ? (
                <Tooltip title="Edit">
                  <IconButton size="small" onClick={() => {
                    manuallyEditingRef.current = true; // Mark as manually entered edit mode
                    setIsEditing(true);
                    setEditModePreference(true); // Remember user prefers edit mode
                    const fname = (document.filename || '').toLowerCase();
                    if (fname.endsWith('.md') || fname.endsWith('.org')) {
                      setViewMode('edit');
                    }
                  }}>
                    <Edit fontSize="small" />
                  </IconButton>
                </Tooltip>
              ) : (
                <Tooltip title={saving ? 'Saving...' : 'Save'}>
                  <span>
                    <IconButton size="small" onClick={async () => {
                      if (saving) return;
                      try {
                        setSaving(true);
                        
                        // Skip no-op saves (prevents unnecessary re-indexing)
                        if (editContent === serverContent) {
                          clearUnsavedContent(documentId);
                          return;
                        }

                        // Check for TODO->DONE changes (recurring tasks)
                        const isOrgFile = (document?.filename || '').toLowerCase().endsWith('.org');
                        let recurringHandled = false;
                        
                        if (isOrgFile && document.content) {
                          // Find lines that changed from TODO to DONE
                          const oldLines = document.content.split('\n');
                          const newLines = editContent.split('\n');
                          
                          for (let i = 0; i < Math.min(oldLines.length, newLines.length); i++) {
                            const oldLine = oldLines[i];
                            const newLine = newLines[i];
                            
                            // Check if line changed from TODO to DONE
                            const wasTodo = /^\*+\s+(TODO|NEXT|STARTED|WAITING)\s+/.test(oldLine);
                            const isDone = /^\*+\s+(DONE|CANCELED|CANCELLED)\s+/.test(newLine);
                            
                            if (wasTodo && isDone) {
                              console.log('Detected TODO->DONE at line', i + 1);
                              
                              // Build file path
                              let filePath = document.filename;
                              if (document.folder_id && document.folder_name) {
                                filePath = `${document.folder_name}/${document.filename}`;
                              } else {
                                filePath = `OrgMode/${document.filename}`;
                              }
                              
                              try {
                                // Call recurring API
                                const response = await apiService.post('/api/org/recurring/complete', {
                                  file_path: filePath,
                                  line_number: i + 1
                                });
                                
                                if (response.success && response.is_recurring) {
                                  console.log('✅ Recurring task handled:', response.message);
                                  recurringHandled = true;
                                  
                                  // Show notification
                                  alert(`🔁 Recurring Task!\n\n${response.message}\n\nFile will be refreshed.`);
                                }
                              } catch (err) {
                                console.error('❌ Failed to handle recurring task:', err);
                              }
                            }
                          }
                        }
                        
                        // Save content
                        await apiService.updateDocumentContent(document.document_id, editContent);
                        // Update document state with new content and current timestamp
                        const now = new Date().toISOString();
                        setDocument((prev) => prev ? { ...prev, content: editContent, updated_at: now } : prev);
                        
                        // Update serverContent as it's now on disk
                        setServerContent(editContent);
                        
                        // Clear unsaved content after successful save
                        clearUnsavedContent(documentId);
                        
                        // If recurring task was handled, refresh to get updated content
                        if (recurringHandled) {
                          setTimeout(() => {
                            fetchDocument(true); // Force refresh to get updated content
                          }, 500);
                        }
                        
                        // Keep the user in edit mode after saving
                      } catch (e) {
                        console.error('Save failed', e);
                        alert('Save failed');
                      } finally {
                        setSaving(false);
                      }
                    }} disabled={saving}>
                      <Save fontSize="small" />
                    </IconButton>
                  </span>
                </Tooltip>
              )
            )}
            {canShowVersionHistory && (
              <Tooltip title="Version history">
                <IconButton size="small" onClick={openVersionHistory}>
                  <History fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            {isEditing && (
              <Tooltip title={
                viewMode === 'edit' ? 'Show preview' :
                viewMode === 'split' ? 'Preview only' : 'Edit only'
              }>
                <IconButton
                  size="small"
                  onClick={() => setViewMode((m) => (m === 'edit' ? 'split' : m === 'split' ? 'preview' : 'edit'))}
                  color={viewMode === 'edit' ? 'default' : 'primary'}
                >
                  {viewMode === 'edit' && <VisibilityOff fontSize="small" />}
                  {viewMode === 'split' && <ViewAgenda fontSize="small" />}
                  {viewMode === 'preview' && <Visibility fontSize="small" />}
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}>
              <IconButton size="small" onClick={handleToggleFullscreen}>
                {isFullscreen ? <FullscreenExit fontSize="small" /> : <Fullscreen fontSize="small" />}
              </IconButton>
            </Tooltip>
            {isReadableTextDocument && (
              <Tooltip title={isReadingAloud ? 'Stop reading' : 'Read aloud'}>
                <span>
                  <IconButton
                    size="small"
                    onClick={handleToggleReadAloud}
                    disabled={!canSpeak}
                  >
                    {isReadingAloud ? <Stop fontSize="small" /> : <VolumeUp fontSize="small" />}
                  </IconButton>
                </span>
              </Tooltip>
            )}
            <Tooltip title="Download or Export">
              <IconButton size="small" onClick={handleDownloadMenuOpen}>
                <Download fontSize="small" />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={downloadMenuAnchor}
              open={Boolean(downloadMenuAnchor)}
              onClose={handleDownloadMenuClose}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
            >
              <MenuItem onClick={handleDownload}>
                <ListItemIcon>
                  <Download fontSize="small" />
                </ListItemIcon>
                <ListItemText>Download</ListItemText>
              </MenuItem>
              {isReadableTextDocument && (
                <MenuItem
                  onClick={() => {
                    handleDownloadMenuClose();
                    setAudioExportOpen(true);
                  }}
                >
                  <ListItemIcon>
                    <AudioFile fontSize="small" />
                  </ListItemIcon>
                  <ListItemText>Export as Audio…</ListItemText>
                </MenuItem>
              )}
              {isEditing && document.filename && fnameLower.endsWith('.md') && (
                <>
                  <MenuItem onClick={openPdfExportDialog}>
                    <ListItemIcon>
                      <FileDownload fontSize="small" />
                    </ListItemIcon>
                    <ListItemText>Export to PDF…</ListItemText>
                  </MenuItem>
                  <MenuItem onClick={handleExportEpub}>
                    <ListItemIcon>
                      <FileDownload fontSize="small" />
                    </ListItemIcon>
                    <ListItemText>Export as EPUB</ListItemText>
                  </MenuItem>
                </>
              )}
            </Menu>
            <Drawer
              anchor="right"
              open={versionHistoryOpen}
              onClose={() => { setVersionHistoryOpen(false); setVersionViewContent(null); setVersionDiffResult(null); setVersionViewMeta(null); }}
              PaperProps={{ sx: { width: 420, maxWidth: '100%' } }}
            >
              <Box sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                <Typography variant="h6" sx={{ mb: 2 }}>Version history</Typography>
                {versionsLoading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}><CircularProgress size={24} /></Box>
                ) : (
                  <Box sx={{ flex: 1, overflow: 'auto' }}>
                    {versionViewContent != null ? (
                      <Box>
                        <Button size="small" onClick={() => { setVersionViewContent(null); setVersionViewMeta(null); }} sx={{ mb: 1 }}>Back to list</Button>
                        <Typography variant="subtitle2" color="text.secondary">Version {versionViewMeta?.version_number} — {versionViewMeta?.change_source}</Typography>
                        <Paper variant="outlined" sx={{ p: 1, mt: 1, maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.85rem' }}>
                          {versionViewContent}
                        </Paper>
                      </Box>
                    ) : versionDiffResult ? (
                      <Box>
                        <Button size="small" onClick={() => setVersionDiffResult(null)} sx={{ mb: 1 }}>Back to list</Button>
                        <Typography variant="subtitle2" color="text.secondary">
                          Diff v{versionViewMeta?.from?.version_number} → v{versionViewMeta?.to?.version_number} (+{versionDiffResult.additions} / -{versionDiffResult.deletions})
                        </Typography>
                        <Paper variant="outlined" sx={{ p: 1, mt: 1, maxHeight: 400, overflow: 'auto', whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {versionDiffResult.diff || 'No changes'}
                        </Paper>
                      </Box>
                    ) : (
                      <Stack spacing={1}>
                        {versionsList.length === 0 && <Typography color="text.secondary">No versions yet. Edits will create versions.</Typography>}
                        {versionsList.map((v) => (
                          <Paper key={v.version_id} variant="outlined" sx={{ p: 1.5 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 0.5 }}>
                              <Box>
                                <Typography variant="subtitle2">v{v.version_number}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {v.created_at ? new Date(v.created_at).toLocaleString() : ''} · {v.change_source || '—'}
                                  {v.is_current && <Chip size="small" label="Current" sx={{ ml: 0.5 }} />}
                                </Typography>
                              </Box>
                              <Box sx={{ display: 'flex', gap: 0.5 }}>
                                <Button size="small" variant="outlined" onClick={() => handleViewVersion(v)}>View</Button>
                                {!v.is_current && versionsList[0] && (
                                  <Button size="small" variant="outlined" onClick={() => handleDiffWithCurrent(v)}>Diff</Button>
                                )}
                                {!v.is_current && (
                                  <Button size="small" color="primary" startIcon={<Restore fontSize="small" />} onClick={() => handleRollbackClick(v)}>Rollback</Button>
                                )}
                              </Box>
                            </Box>
                            {v.change_summary && <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>{v.change_summary}</Typography>}
                          </Paper>
                        ))}
                      </Stack>
                    )}
                  </Box>
                )}
              </Box>
            </Drawer>
            <Dialog open={Boolean(rollbackConfirmVersion)} onClose={() => setRollbackConfirmVersion(null)}>
              <DialogTitle>Rollback to this version?</DialogTitle>
              <DialogContent>
                {rollbackConfirmVersion && (
                  <Typography>Current content will be saved as a new version first, then the document will be restored to version {rollbackConfirmVersion.version_number}. You can undo by rolling back again.</Typography>
                )}
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setRollbackConfirmVersion(null)}>Cancel</Button>
                <Button variant="contained" color="primary" onClick={handleRollbackConfirm} disabled={rollbacking}>
                  {rollbacking ? 'Rolling back…' : 'Rollback'}
                </Button>
              </DialogActions>
            </Dialog>
          </Box>
        </Box>

        {/* Content area (single scroll) */}
        <Box
          ref={contentBoxRef}
          sx={{
            flex: 1,
            minHeight: 0,
            overflow: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 'hidden' : 'auto',
            p: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 1 : 2,
            backgroundColor: 'background.default',
          }}
        >
          <Paper
            variant="outlined"
            sx={{
              p: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 1 : 2,
              backgroundColor: 'background.paper',
              height: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? '100%' : undefined,
              display: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 'flex' : undefined,
              flexDirection: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 'column' : undefined,
              minHeight: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 0 : undefined,
              overflow: isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? 'hidden' : undefined,
            }}
          >
            {isEditing && (fnameLower.endsWith('.md') || fnameLower.endsWith('.txt') || fnameLower.endsWith('.org')) ? (
              fnameLower.endsWith('.org') ? (
                viewMode === 'preview' ? (
                  // Preview only for Org
                  <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                      Preview
                    </Typography>
                    <Box sx={{ p: 1, flex: 1, overflow: 'auto' }}>
                      <OrgRenderer 
                        content={editContent}
                        onNavigate={async (navInfo) => {
                          if (navInfo.type === 'file') {
                            console.log('🔗 Navigating to file:', navInfo.path);
                            alert(`File navigation coming soon!\nTarget: ${navInfo.path}`);
                          } else if (navInfo.type === 'id') {
                            console.log('🔗 Navigating to ID:', navInfo.id);
                            alert(`File navigation coming soon!\nTarget ID: ${navInfo.id}`);
                          }
                        }}
                      />
                    </Box>
                  </Box>
                ) : viewMode === 'split' ? (
                  // Split view for OrgMode: Editor + Preview
                  <Box sx={{ display: 'flex', gap: 2, height: '70vh' }}>
                    <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                        Edit Mode
                      </Typography>
                      <Box sx={{ p: 1, height: 'calc(100% - 40px)', overflow: 'auto' }}>
                        <OrgCMEditor 
                          key={`org-${documentId}-${darkMode ? 'dark' : 'light'}-${accentId}`}
                          ref={orgEditorRef}
                          value={editContent} 
                          onChange={setEditContent}
                          scrollToLine={scrollToLine}
                          scrollToHeading={scrollToHeading}
                          initialScrollPosition={effectiveInitialScroll}
                          onScrollChange={onScrollChange}
                          canonicalPath={document?.canonical_path}
                          filename={document?.filename}
                          documentId={documentId}
                          folderId={document?.folder_id}
                          onCurrentSectionChange={handleCurrentSectionChange}
                          darkMode={darkMode}
                        />
                      </Box>
                    </Box>
                    <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                        Preview
                      </Typography>
                      <Box sx={{ p: 1, height: 'calc(100% - 40px)', overflow: 'auto' }}>
                        <OrgRenderer 
                          content={editContent}
                          onNavigate={async (navInfo) => {
                            if (navInfo.type === 'file') {
                              console.log('🔗 Navigating to file:', navInfo.path);
                              alert(`File navigation coming soon!\nTarget: ${navInfo.path}`);
                            } else if (navInfo.type === 'id') {
                              console.log('🔗 Navigating to ID:', navInfo.id);
                              alert(`ID navigation coming soon!\nTarget ID: ${navInfo.id}`);
                            }
                          }}
                        />
                      </Box>
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    <OrgCMEditor 
                      key={`org-${documentId}-${darkMode ? 'dark' : 'light'}-${accentId}`}
                      ref={orgEditorRef}
                      value={editContent} 
                      onChange={setEditContent}
                      scrollToLine={scrollToLine}
                      scrollToHeading={scrollToHeading}
                      initialScrollPosition={effectiveInitialScroll}
                      onScrollChange={onScrollChange}
                      canonicalPath={document?.canonical_path}
                      filename={document?.filename}
                      documentId={documentId}
                      folderId={document?.folder_id}
                      onCurrentSectionChange={handleCurrentSectionChange}
                      darkMode={darkMode}
                    />
                  </Box>
                )
              ) : fnameLower.endsWith('.md') ? (
                viewMode === 'preview' ? (
                  // Preview only for Markdown
                  <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                      Preview
                    </Typography>
                    <Box sx={{ 
                      p: 2, 
                      flex: 1,
                      overflow: 'auto',
                      '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1, fontWeight: 'bold' },
                      '& p': { mb: 1.5, lineHeight: 1.6 },
                      '& img': { maxWidth: '100%', height: 'auto', borderRadius: 1, my: 2 },
                      '& a': { color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } },
                      '& blockquote': { borderLeft: 3, borderColor: 'primary.main', pl: 2, ml: 0, my: 2, backgroundColor: 'grey.100', py: 1, pr: 2 },
                      '& code': { backgroundColor: 'grey.200', px: 0.5, py: 0.25, borderRadius: 0.5, fontFamily: 'monospace', fontSize: '0.875em' },
                      '& pre': { backgroundColor: 'grey.200', p: 2, borderRadius: 1, overflow: 'auto', '& code': { backgroundColor: 'transparent', p: 0 } },
                      '& ul, & ol': { pl: 3, mb: 1.5 },
                      '& li': { mb: 0.5 },
                      '& strong': { fontWeight: 'bold' },
                      '& em': { fontStyle: 'italic' },
                      '& details': { mb: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1 },
                      '& summary': { cursor: 'pointer', fontWeight: 'medium', py: 1, '&:hover': { opacity: 0.8 } }
                    }}>
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[
                          rehypeRaw,
                          [
                            rehypeSanitize,
                            {
                              tagNames: [
                                'details', 'summary', 'div', 'span',
                                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's',
                                'ul', 'ol', 'li',
                                'blockquote', 'pre', 'code',
                                'table', 'thead', 'tbody', 'tr', 'th', 'td',
                                'a', 'img',
                                'hr'
                              ],
                              attributes: {
                                '*': ['class', 'id'],
                                'a': ['href', 'title'],
                                'img': ['src', 'alt', 'title', 'width', 'height'],
                                'div': ['style'],
                                'span': ['style'],
                                'details': ['open'],
                                'summary': []
                              },
                              protocols: {
                                href: ['http', 'https', 'mailto'],
                                src: ['http', 'https', 'data', '']
                              }
                            }
                          ]
                        ]}
                      >
                        {editContent}
                      </ReactMarkdown>
                    </Box>
                  </Box>
                ) : viewMode === 'split' ? (
                  // Split view for Markdown: Editor + Preview
                  <Box sx={{ display: 'flex', gap: 2, height: '70vh' }}>
                    <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                        Edit Mode
                      </Typography>
                      <Box sx={{ p: 1, height: 'calc(100% - 40px)', overflow: 'auto' }}>
                        <MarkdownCMEditor 
                          key={`md-${documentId}-${darkMode ? 'dark' : 'light'}-${accentId}`}
                          ref={markdownEditorRef}
                          value={editContent} 
                          onChange={setEditContent} 
                          filename={document.filename}
                          canonicalPath={document.canonical_path}
                          documentId={documentId} 
                          initialScrollPosition={effectiveInitialScroll}
                          restoreScrollAfterRefresh={restoreScrollAfterRefresh}
                          onScrollRestored={handleScrollRestored}
                          onScrollChange={onScrollChange}
                          onCurrentSectionChange={handleCurrentSectionChange}
                        />
                      </Box>
                    </Box>
                    <Box sx={{ flex: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                      <Typography variant="subtitle2" sx={{ p: 1, backgroundColor: 'grey.100', borderBottom: '1px solid #e0e0e0', fontWeight: 'bold' }}>
                        Preview
                      </Typography>
                      <Box sx={{ 
                        p: 2, 
                        height: 'calc(100% - 40px)', 
                        overflow: 'auto',
                        '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1, fontWeight: 'bold' },
                        '& p': { mb: 1.5, lineHeight: 1.6 },
                        '& img': { maxWidth: '100%', height: 'auto', borderRadius: 1, my: 2 },
                        '& a': { color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } },
                        '& blockquote': { borderLeft: 3, borderColor: 'primary.main', pl: 2, ml: 0, my: 2, backgroundColor: 'grey.100', py: 1, pr: 2 },
                        '& code': { backgroundColor: 'grey.200', px: 0.5, py: 0.25, borderRadius: 0.5, fontFamily: 'monospace', fontSize: '0.875em' },
                        '& pre': { backgroundColor: 'grey.200', p: 2, borderRadius: 1, overflow: 'auto', '& code': { backgroundColor: 'transparent', p: 0 } },
                        '& ul, & ol': { pl: 3, mb: 1.5 },
                        '& li': { mb: 0.5 },
                        '& strong': { fontWeight: 'bold' },
                        '& em': { fontStyle: 'italic' },
                        '& details': { mb: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1 },
                        '& summary': { cursor: 'pointer', fontWeight: 'medium', py: 1, '&:hover': { opacity: 0.8 } }
                      }}>
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[
                            rehypeRaw,
                            [
                              rehypeSanitize,
                              {
                                tagNames: [
                                  'details', 'summary', 'div', 'span',
                                  'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                  'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's',
                                  'ul', 'ol', 'li',
                                  'blockquote', 'pre', 'code',
                                  'table', 'thead', 'tbody', 'tr', 'th', 'td',
                                  'a', 'img',
                                  'hr'
                                ],
                                attributes: {
                                  '*': ['class', 'id'],
                                  'a': ['href', 'title'],
                                  'img': ['src', 'alt', 'title', 'width', 'height'],
                                  'div': ['style'],
                                  'span': ['style'],
                                  'details': ['open'],
                                  'summary': []
                                },
                                protocols: {
                                  href: ['http', 'https', 'mailto'],
                                  src: ['http', 'https', 'data', '']
                                }
                              }
                            ]
                          ]}
                        >
                          {editContent}
                        </ReactMarkdown>
                      </Box>
                    </Box>
                  </Box>
                ) : (
                  <Box sx={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
                    <MarkdownCMEditor 
                      key={`md-${documentId}-${darkMode ? 'dark' : 'light'}-${accentId}`}
                      ref={markdownEditorRef}
                      value={editContent} 
                      onChange={setEditContent} 
                      filename={document.filename}
                      canonicalPath={document.canonical_path}
                      documentId={documentId} 
                      initialScrollPosition={effectiveInitialScroll}
                      restoreScrollAfterRefresh={restoreScrollAfterRefresh}
                      onScrollRestored={handleScrollRestored}
                      onScrollChange={onScrollChange}
                      onCurrentSectionChange={handleCurrentSectionChange}
                    />
                  </Box>
                )
              ) : (
                <TextField
                  multiline
                  fullWidth
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  sx={{ 
                    '& .MuiInputBase-root': { 
                      fontFamily: 'monospace', 
                      fontSize: 14, 
                      lineHeight: 1.5,
                      minHeight: '60vh',
                      alignItems: 'flex-start'
                    }
                  }}
                />
              )
            ) : document.filename && fnameLower.endsWith('.md') ? (
              <Box sx={{ 
                '& h1, & h2, & h3, & h4, & h5, & h6': { mt: 2, mb: 1, fontWeight: 'bold' },
                '& p': { mb: 1.5, lineHeight: 1.6 },
                '& img': { maxWidth: '100%', height: 'auto', borderRadius: 1, my: 2 },
                '& a': { color: 'primary.main', textDecoration: 'none', '&:hover': { textDecoration: 'underline' } },
                '& blockquote': { borderLeft: 3, borderColor: 'primary.main', pl: 2, ml: 0, my: 2, backgroundColor: 'grey.100', py: 1, pr: 2 },
                '& code': { backgroundColor: 'grey.200', px: 0.5, py: 0.25, borderRadius: 0.5, fontFamily: 'monospace', fontSize: '0.875em' },
                '& pre': { backgroundColor: 'grey.200', p: 2, borderRadius: 1, overflow: 'auto', '& code': { backgroundColor: 'transparent', p: 0 } },
                '& ul, & ol': { pl: 3, mb: 1.5 },
                '& li': { mb: 0.5 },
                '& strong': { fontWeight: 'bold' },
                '& em': { fontStyle: 'italic' },
                '& details': { mb: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1 },
                '& summary': { cursor: 'pointer', fontWeight: 'medium', py: 1, '&:hover': { opacity: 0.8 } }
              }}>
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[
                    rehypeRaw,
                    [
                      rehypeSanitize,
                      {
                        tagNames: [
                          'details', 'summary', 'div', 'span',
                          'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                          'p', 'br', 'strong', 'em', 'b', 'i', 'u', 's',
                          'ul', 'ol', 'li',
                          'blockquote', 'pre', 'code',
                          'table', 'thead', 'tbody', 'tr', 'th', 'td',
                          'a', 'img',
                          'hr'
                        ],
                        attributes: {
                          '*': ['class', 'id'],
                          'a': ['href', 'title'],
                          'img': ['src', 'alt', 'title', 'width', 'height'],
                          'div': ['style'],
                          'span': ['style'],
                          'details': ['open'],
                          'summary': []
                        },
                        protocols: {
                          href: ['http', 'https', 'mailto'],
                          src: ['http', 'https', 'data', '']
                        }
                      }
                    ]
                  ]}
                >
                  {document.content}
                </ReactMarkdown>
              </Box>
            ) : document.filename && document.filename.endsWith('.org') ? (
              <>
                <Box sx={{ p: 1 }}>
                  <OrgRenderer 
                    content={document.content}
                    onNavigate={async (navInfo) => {
                      if (navInfo.type === 'file') {
                        // Handle file links - search for document by filename
                        try {
                          console.log('🔗 Navigating to file:', navInfo.path);
                          // TODO: Implement file navigation by searching for document
                          // For now, just log the intent
                          alert(`File navigation coming soon!\nTarget: ${navInfo.path}\n\nThis will search for and open the document.`);
                        } catch (err) {
                          console.error('Failed to navigate to file:', err);
                        }
                      } else if (navInfo.type === 'id') {
                        // Handle ID-based links
                        console.log('🔗 Navigating to ID:', navInfo.id);
                        alert(`ID navigation coming soon!\nTarget ID: ${navInfo.id}\n\nThis will navigate to the heading with this ID property.`);
                      }
                    }}
                  />
                </Box>
                
                {/* Backlinks Section */}
                {!isEditing && backlinks.length > 0 && (
                  <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' }}>
                    <Typography variant="h6" sx={{ mb: 1, fontSize: '1rem', fontWeight: 600, color: 'primary.main' }}>
                      🔗 Backlinks ({backlinks.length})
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary', fontStyle: 'italic' }}>
                      Files that link to this document
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {backlinks.map((backlink, index) => (
                        <Paper 
                          key={index}
                          elevation={1}
                          sx={{ 
                            p: 1.5, 
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            '&:hover': { 
                              backgroundColor: '#e3f2fd',
                              transform: 'translateX(4px)',
                              boxShadow: 2
                            }
                          }}
                          onClick={async () => {
                            console.log('🔗 Navigating to backlink:', backlink.filename);
                            try {
                              // Look up document by filename
                              const response = await apiService.get(`/api/org/lookup-document?filename=${encodeURIComponent(backlink.filename)}`);
                              if (response.success && response.document) {
                                // Open the document (this will require parent component support)
                                alert(`Navigation coming soon!\nWill open: ${backlink.filename}\n\nDocument ID: ${response.document.document_id}`);
                              }
                            } catch (err) {
                              console.error('Failed to navigate to backlink:', err);
                              alert(`Failed to navigate to ${backlink.filename}`);
                            }
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                            <Description fontSize="small" sx={{ color: 'primary.main' }} />
                            <Typography variant="body1" sx={{ fontWeight: 500, color: 'primary.main' }}>
                              {backlink.filename}
                            </Typography>
                            {backlink.link_count > 1 && (
                              <Chip 
                                label={`${backlink.link_count} links`}
                                size="small"
                                sx={{ height: 20, fontSize: '0.7rem' }}
                              />
                            )}
                          </Box>
                          <Typography variant="body2" sx={{ color: 'text.secondary', fontSize: '0.85rem', pl: 3.5 }}>
                            {backlink.context}
                          </Typography>
                        </Paper>
                      ))}
                    </Box>
                  </Box>
                )}
              </>
            ) : (
              <Typography 
                variant="body1" 
                sx={{ 
                  whiteSpace: 'pre-wrap', 
                  lineHeight: 1.6,
                  color: 'text.primary',
                  fontFamily: 'monospace',
                  fontSize: '14px'
                }}
              >
                {document.content}
              </Typography>
            )}
          </Paper>
        </Box>
      </Box>
      <AudioExportDialog
        open={audioExportOpen}
        onClose={() => setAudioExportOpen(false)}
        documentId={documentId}
        documentContent={getPreferredTextToRead()}
        documentTitle={document?.title}
        documentFilename={document?.filename}
        speechMode={readAloudFilenameLower.endsWith('.org') ? 'org' : 'markdown'}
      />
      {/* PDF Export Dialog */}
      <Dialog open={pdfExportDialogOpen} onClose={() => !exporting && setPdfExportDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Export to PDF</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2} sx={{ mt: 0.5 }}>
            <FormControl component="fieldset" variant="standard">
              <FormLabel component="legend">Layout</FormLabel>
              <RadioGroup
                value={pdfLayout}
                onChange={(e) => setPdfLayout(e.target.value)}
              >
                <FormControlLabel value="article" control={<Radio size="small" />} label="Article" />
                <FormControlLabel value="book" control={<Radio size="small" />} label="Book" />
              </RadioGroup>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              Article uses the standard single-column layout. Book adds margins, paragraph indents, configurable page numbers, and optional per-section overrides.
            </Typography>
            <FormControl component="fieldset" variant="standard">
              <FormLabel component="legend">Orientation</FormLabel>
              <RadioGroup
                value={pdfExportOrientation}
                onChange={(e) => setPdfExportOrientation(e.target.value)}
              >
                <FormControlLabel value="portrait" control={<Radio />} label="Portrait" />
                <FormControlLabel value="landscape" control={<Radio />} label="Landscape" />
              </RadioGroup>
            </FormControl>
            <Typography variant="body2" color="text.secondary">
              Landscape swaps width and height for the same paper size.
            </Typography>
            <FormControl fullWidth size="small">
              <InputLabel id="pdf-font-preset-label">Font (PDF)</InputLabel>
              <Select
                labelId="pdf-font-preset-label"
                label="Font (PDF)"
                value={pdfFontPreset}
                onChange={(e) => setPdfFontPreset(e.target.value)}
              >
                <MenuItem value="liberation">Liberation Serif / Sans</MenuItem>
                <MenuItem value="dejavu">DejaVu Serif / Sans</MenuItem>
                <MenuItem value="noto">Noto Serif / Sans</MenuItem>
                <MenuItem value="times_helvetica">Times / Helvetica</MenuItem>
              </Select>
            </FormControl>
            <FormControl component="fieldset" variant="standard">
              <FormLabel component="legend">Typeface</FormLabel>
              <RadioGroup
                value={pdfTypefaceStyle}
                onChange={(e) => setPdfTypefaceStyle(e.target.value)}
              >
                <FormControlLabel value="mixed" control={<Radio size="small" />} label="Mixed (serif body, sans headings)" />
                <FormControlLabel value="serif" control={<Radio size="small" />} label="Serif (body and headings)" />
                <FormControlLabel value="sans" control={<Radio size="small" />} label="Sans-serif (body and headings)" />
              </RadioGroup>
            </FormControl>
            {pdfSourceFormat === 'org' && (
              <Typography variant="caption" color="text.secondary" display="block">
                Org headlines are converted for PDF so heading levels use distinct sizes.
              </Typography>
            )}
            <FormControlLabel
              control={<Checkbox checked={pdfPageNumbers} onChange={(e) => setPdfPageNumbers(e.target.checked)} />}
              label={pdfLayout === 'book' ? 'Page numbers (default)' : 'Page numbers in footer'}
            />
            <FormGroup>
              <FormControlLabel
                control={<Checkbox checked={pdfIncludeToc} onChange={(e) => setPdfIncludeToc(e.target.checked)} />}
                label="Include table of contents"
              />
            </FormGroup>
            <FormControl fullWidth size="small" disabled={!pdfIncludeToc}>
              <InputLabel id="pdf-toc-depth-label">TOC depth (max heading level)</InputLabel>
              <Select
                labelId="pdf-toc-depth-label"
                label="TOC depth (max heading level)"
                value={pdfTocDepth}
                onChange={(e) => setPdfTocDepth(Number(e.target.value))}
              >
                {[1, 2, 3, 4, 5, 6].map((n) => (
                  <MenuItem key={n} value={n}>
                    Through H{n}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Box>
              <FormLabel component="legend">Page break before headings</FormLabel>
              <FormGroup row sx={{ flexWrap: 'wrap' }}>
                {[1, 2, 3, 4, 5, 6].map((lvl) => (
                  <FormControlLabel
                    key={`pdf-pb-${lvl}`}
                    control={(
                      <Checkbox
                        size="small"
                        checked={pdfPageBreakHeadings.includes(lvl)}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setPdfPageBreakHeadings((prev) =>
                            checked
                              ? Array.from(new Set([...prev, lvl])).sort((a, b) => a - b)
                              : prev.filter((v) => v !== lvl)
                          );
                        }}
                      />
                    )}
                    label={`H${lvl}`}
                  />
                ))}
              </FormGroup>
            </Box>
            {pdfLayout === 'book' && (
              <>
                <Divider />
                <Typography variant="subtitle2">Book margins (mm)</Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  <TextField
                    label="Top"
                    type="number"
                    size="small"
                    inputProps={{ min: 5, max: 80, step: 1 }}
                    value={pdfBookMarginTopMm}
                    onChange={(e) => setPdfBookMarginTopMm(Number(e.target.value))}
                    sx={{ width: 100 }}
                  />
                  <TextField
                    label="Right"
                    type="number"
                    size="small"
                    inputProps={{ min: 5, max: 80, step: 1 }}
                    value={pdfBookMarginRightMm}
                    onChange={(e) => setPdfBookMarginRightMm(Number(e.target.value))}
                    sx={{ width: 100 }}
                  />
                  <TextField
                    label="Bottom"
                    type="number"
                    size="small"
                    inputProps={{ min: 5, max: 80, step: 1 }}
                    value={pdfBookMarginBottomMm}
                    onChange={(e) => setPdfBookMarginBottomMm(Number(e.target.value))}
                    sx={{ width: 100 }}
                  />
                  <TextField
                    label="Left"
                    type="number"
                    size="small"
                    inputProps={{ min: 5, max: 80, step: 1 }}
                    value={pdfBookMarginLeftMm}
                    onChange={(e) => setPdfBookMarginLeftMm(Number(e.target.value))}
                    sx={{ width: 100 }}
                  />
                </Stack>
                <FormGroup>
                  <FormControlLabel
                    control={<Checkbox checked={pdfBookIndentBody} onChange={(e) => setPdfBookIndentBody(e.target.checked)} />}
                    label="Indent body paragraphs"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={pdfBookNoIndentAfterHeading} onChange={(e) => setPdfBookNoIndentAfterHeading(e.target.checked)} />}
                    label="No indent on first paragraph after a heading"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={pdfBookSuppressFirstPage} onChange={(e) => setPdfBookSuppressFirstPage(e.target.checked)} />}
                    label="Suppress page number on first page of document"
                  />
                </FormGroup>
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
                  <FormControl size="small" sx={{ minWidth: 160 }}>
                    <InputLabel id="pdf-pnf-label">Page # format</InputLabel>
                    <Select
                      labelId="pdf-pnf-label"
                      label="Page # format"
                      value={pdfBookPageNumFormat}
                      onChange={(e) => setPdfBookPageNumFormat(e.target.value)}
                    >
                      <MenuItem value="n_of_total">N of total</MenuItem>
                      <MenuItem value="n_only">N only</MenuItem>
                    </Select>
                  </FormControl>
                  <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel id="pdf-pnv-label">Number band</InputLabel>
                    <Select
                      labelId="pdf-pnv-label"
                      label="Number band"
                      value={pdfBookPageNumVertical}
                      onChange={(e) => setPdfBookPageNumVertical(e.target.value)}
                    >
                      <MenuItem value="bottom">Bottom</MenuItem>
                      <MenuItem value="top">Top</MenuItem>
                    </Select>
                  </FormControl>
                  <FormControl size="small" sx={{ minWidth: 140 }}>
                    <InputLabel id="pdf-pnh-label">Number align</InputLabel>
                    <Select
                      labelId="pdf-pnh-label"
                      label="Number align"
                      value={pdfBookPageNumHorizontal}
                      onChange={(e) => setPdfBookPageNumHorizontal(e.target.value)}
                    >
                      <MenuItem value="left">Left</MenuItem>
                      <MenuItem value="center">Center</MenuItem>
                      <MenuItem value="right">Right</MenuItem>
                    </Select>
                  </FormControl>
                </Stack>
                {pdfPageBreakHeadings.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    Select at least one heading level above for page breaks to list sections and apply per-section rules.
                  </Typography>
                ) : (
                  <>
                    <Typography variant="subtitle2">Sections (headings at page-break levels)</Typography>
                    {pdfOutlineLoading ? (
                      <Typography variant="body2" color="text.secondary">Loading headings…</Typography>
                    ) : (
                      <Box sx={{ maxHeight: 280, overflow: 'auto', border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
                        {pdfHeadingOutline
                          .filter((h) => pdfPageBreakHeadings.includes(h.level))
                          .map((h) => {
                            const ov = pdfSectionOverrides[h.id];
                            const enabled = !!ov?.enabled;
                            return (
                              <Box key={h.id} sx={{ py: 1, borderBottom: 1, borderColor: 'divider' }}>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>{h.text || '(empty)'}</Typography>
                                <Typography variant="caption" color="text.secondary">H{h.level} · {h.id}</Typography>
                                <FormControlLabel
                                  control={(
                                    <Checkbox
                                      size="small"
                                      checked={enabled}
                                      onChange={(e) => {
                                        const on = e.target.checked;
                                        if (on) {
                                          setPdfSectionOverrides((prev) => ({
                                            ...prev,
                                            [h.id]: { enabled: true, pageNumbers: 'inherit', plainFirstPage: false },
                                          }));
                                        } else {
                                          setPdfSectionOverrides((prev) => {
                                            const next = { ...prev };
                                            delete next[h.id];
                                            return next;
                                          });
                                        }
                                      }}
                                    />
                                  )}
                                  label="Custom rules for this section"
                                />
                                {enabled && (
                                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems="flex-start" sx={{ pl: 1, mt: 0.5 }}>
                                    <FormControl size="small" sx={{ minWidth: 160 }}>
                                      <InputLabel id={`pn-${h.id}`}>Page numbers</InputLabel>
                                      <Select
                                        labelId={`pn-${h.id}`}
                                        label="Page numbers"
                                        value={ov?.pageNumbers ?? 'inherit'}
                                        onChange={(e) => setPdfSectionOverrides((prev) => ({
                                          ...prev,
                                          [h.id]: { ...prev[h.id], enabled: true, pageNumbers: e.target.value, plainFirstPage: !!prev[h.id]?.plainFirstPage },
                                        }))}
                                      >
                                        <MenuItem value="inherit">Inherit default</MenuItem>
                                        <MenuItem value="on">Show</MenuItem>
                                        <MenuItem value="off">Hide</MenuItem>
                                      </Select>
                                    </FormControl>
                                    <FormControlLabel
                                      control={(
                                        <Checkbox
                                          size="small"
                                          checked={!!ov?.plainFirstPage}
                                          onChange={(e) => setPdfSectionOverrides((prev) => ({
                                            ...prev,
                                            [h.id]: {
                                              ...prev[h.id],
                                              enabled: true,
                                              pageNumbers: prev[h.id]?.pageNumbers ?? 'inherit',
                                              plainFirstPage: e.target.checked,
                                            },
                                          }))}
                                        />
                                      )}
                                      label="Plain first page of section (no page number on opening page)"
                                    />
                                  </Stack>
                                )}
                              </Box>
                            );
                          })}
                        {pdfHeadingOutline.filter((x) => pdfPageBreakHeadings.includes(x.level)).length === 0 && !pdfOutlineLoading ? (
                          <Typography variant="body2" color="text.secondary">No matching headings in the document.</Typography>
                        ) : null}
                      </Box>
                    )}
                  </>
                )}
              </>
            )}
            <TextField
              label="Watermark text"
              placeholder="e.g. DRAFT, CONFIDENTIAL"
              value={pdfWatermarkText}
              onChange={(e) => setPdfWatermarkText(e.target.value)}
              fullWidth
              size="small"
            />
            {(pdfWatermarkText || '').trim() !== '' && (
              <FormControl component="fieldset" variant="standard">
                <FormLabel component="legend">Watermark pages</FormLabel>
                <RadioGroup
                  row
                  value={pdfWatermarkAllPages ? 'all' : 'first'}
                  onChange={(e) => setPdfWatermarkAllPages(e.target.value === 'all')}
                >
                  <FormControlLabel value="all" control={<Radio size="small" />} label="All pages" />
                  <FormControlLabel value="first" control={<Radio size="small" />} label="First page only" />
                </RadioGroup>
              </FormControl>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPdfExportDialogOpen(false)} disabled={exporting}>Cancel</Button>
          <Button variant="contained" onClick={handleConfirmPdfExport} disabled={exporting}>
            {exporting ? 'Exporting…' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>
      {/* EPUB Export Dialog */}
      <Dialog open={exportOpen} onClose={() => setExportOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Export as EPUB</DialogTitle>
        <DialogContent dividers>
          <Stack spacing={2} sx={{ mt: 1 }}>
            <TextField label="Title" value={epubTitle} onChange={(e) => setEpubTitle(e.target.value)} fullWidth />
            <Stack direction="row" spacing={1}>
              <TextField label="Author First" value={epubAuthorFirst} onChange={(e) => setEpubAuthorFirst(e.target.value)} fullWidth />
              <TextField label="Author Last" value={epubAuthorLast} onChange={(e) => setEpubAuthorLast(e.target.value)} fullWidth />
            </Stack>
            <TextField label="Language" value={epubLanguage} onChange={(e) => setEpubLanguage(e.target.value)} fullWidth />
            <FormGroup>
              <FormControlLabel control={<Checkbox checked={includeToc} onChange={(e) => setIncludeToc(e.target.checked)} />} label="Include Table of Contents" />
              <FormControlLabel control={<Checkbox checked={includeCover} onChange={(e) => setIncludeCover(e.target.checked)} />} label="Include Cover (from frontmatter 'cover' field)" />
              <FormControlLabel control={<Checkbox checked={splitOnHeadings} onChange={(e) => setSplitOnHeadings(e.target.checked)} />} label="Split on Headings" />
            </FormGroup>
            {/* Split level selection H1-H6 */}
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Split on Headers (starts new section):</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {[1,2,3,4,5,6].map((lvl) => (
                  <FormControlLabel 
                    key={lvl} 
                    control={<Checkbox checked={splitLevels.includes(lvl)} onChange={(e) => {
                      const checked = e.target.checked;
                      setSplitLevels((prev) => checked ? Array.from(new Set([...prev, lvl])).sort((a,b)=>a-b) : prev.filter(v => v !== lvl));
                    }} />} 
                    label={`H${lvl}`} 
                  />
                ))}
              </Stack>
            </Box>
            {/* Center level selection H1-H6 */}
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>Center Headers:</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {[1,2,3,4,5,6].map((lvl) => (
                  <FormControlLabel 
                    key={lvl} 
                    control={<Checkbox checked={centerLevels.includes(lvl)} onChange={(e) => {
                      const checked = e.target.checked;
                      setCenterLevels((prev) => checked ? Array.from(new Set([...prev, lvl])).sort((a,b)=>a-b) : prev.filter(v => v !== lvl));
                    }} />} 
                    label={`H${lvl}`} 
                  />
                ))}
              </Stack>
            </Box>
            {/* Paragraph formatting options */}
            <FormGroup>
              <FormControlLabel 
                control={<Checkbox checked={indentParagraphs} onChange={(e) => setIndentParagraphs(e.target.checked)} />} 
                label="Indent Paragraphs (traditional book style)" 
              />
              {indentParagraphs && (
                <FormControlLabel 
                  control={<Checkbox checked={noIndentFirstParagraph} onChange={(e) => setNoIndentFirstParagraph(e.target.checked)} />} 
                  label="Don't indent first paragraph in each section" 
                  sx={{ ml: 3 }}
                />
              )}
            </FormGroup>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setExportOpen(false)} disabled={exporting}>Cancel</Button>
          <Button variant="contained" onClick={async () => {
            try {
              setExporting(true);
              // Build heading alignments dict from centerLevels
              const headingAlignments = {};
              centerLevels.forEach(level => {
                headingAlignments[level] = 'center';
              });
              
              // Build author string from first/last
              const authorFull = [epubAuthorFirst, epubAuthorLast].filter(Boolean).join(' ') || 'Unknown Author';
              
              await exportService.exportMarkdownAsEpub(editContent || document.content || '', {
                documentId: document.document_id,
                folderId: document.folder_id,
                includeToc,
                includeCover,
                splitOnHeadings,
                splitOnHeadingLevels: splitLevels,
                metadata: { 
                  title: epubTitle || 'Untitled', 
                  author: authorFull,
                  author_first: epubAuthorFirst || '',
                  author_last: epubAuthorLast || '',
                  language: epubLanguage || 'en' 
                },
                headingAlignments: headingAlignments,
                indentParagraphs: indentParagraphs,
                noIndentFirstParagraph: noIndentFirstParagraph,
              });
              setExportOpen(false);
            } catch (e) {
              alert('EPUB export failed');
            } finally {
              setExporting(false);
            }
          }} disabled={exporting}>{exporting ? 'Exporting...' : 'Export EPUB'}</Button>
        </DialogActions>
      </Dialog>

      {/* Org Refile Dialog */}
      <OrgRefileDialog
        open={refileDialogOpen}
        onClose={handleRefileClose}
        sourceFile={refileSourceFile}
        sourceLine={refileSourceLine}
        sourceHeading={refileSourceHeading}
      />

      {/* Org Archive Dialog */}
      <OrgArchiveDialog
        open={archiveDialogOpen}
        onClose={handleArchiveClose}
        sourceFile={archiveSourceFile}
        sourceLine={archiveSourceLine}
        sourceHeading={archiveSourceHeading}
        onArchiveComplete={handleArchiveClose}
      />

      {/* Org Tag Dialog */}
      <OrgTagDialog
        open={tagDialogOpen}
        onClose={handleTagClose}
        document={document}
        lineNumber={tagSourceLine}
        currentHeading={tagSourceHeading}
      />
    </Box>
  );
}, (prevProps, nextProps) => {
  // Custom comparison: only re-render if these specific props change
  return (
    prevProps.documentId === nextProps.documentId &&
    prevProps.scrollToLine === nextProps.scrollToLine &&
    prevProps.scrollToHeading === nextProps.scrollToHeading &&
    prevProps.initialScrollPosition === nextProps.initialScrollPosition
  );
  // Note: onClose and onScrollChange are callback functions - we assume they're stable
});

export default DocumentViewer;
