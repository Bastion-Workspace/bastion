import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Collapse,
  ToggleButton,
  ToggleButtonGroup,
  Popover,
  Menu,
  ListSubheader,
  Badge
} from '@mui/material';
import {
  CheckCircle,
  Schedule,
  Error as ErrorIcon,
  LocalOffer,
  Description,
  DriveFileMove,
  Archive,
  ExpandMore,
  ChevronRight,
  UnfoldMore,
  UnfoldLess,
  Add,
  FilterList,
  ArrowDropDown
} from '@mui/icons-material';
import apiService from '../services/apiService';
import orgService from '../services/org/OrgService';
import OrgRefileDialog from './OrgRefileDialog';
import OrgArchiveDialog from './OrgArchiveDialog';

/**
 * Org TODOs view: all TODO items across org files (hierarchy, filters, inline state).
 */
const ORG_TODOS_PREFS_KEY = 'orgTodosView.filters';

/** Org keyword values unchanged; labels match sentence case in the UI */
const TODO_STATE_OPTIONS = [
  { value: 'TODO', label: 'Todo' },
  { value: 'NEXT', label: 'Next' },
  { value: 'STARTED', label: 'Started' },
  { value: 'WAITING', label: 'Waiting' },
  { value: 'HOLD', label: 'On hold' },
  { value: 'DONE', label: 'Done' },
  { value: 'CANCELED', label: 'Canceled' },
];

const todoStateDisplayLabel = (value) =>
  TODO_STATE_OPTIONS.find((o) => o.value === value)?.label ?? value;

const todoRowStateSelectSx = {
  minWidth: 108,
  maxWidth: 128,
  '& .MuiSelect-select': {
    display: 'flex',
    alignItems: 'center',
    py: 0.5,
    pl: 1,
    pr: 0.5,
  },
  '& .MuiSelect-icon': { fontSize: '1.125rem', right: 4 },
};

/** Row state menu: same Typography as list rows (body2), compact list */
const todosStateRowMenuProps = {
  MenuListProps: { dense: true },
  PaperProps: {
    sx: {
      '& .MuiMenuItem-root': {
        minHeight: 34,
        py: 0.5,
        px: 1.25,
      },
    },
  },
};

const todosSelectMenuProps = {
  PaperProps: {
    sx: { '& .MuiMenuItem-root': { fontSize: '0.875rem', lineHeight: 1.43 } },
  },
};

/** Status view tabs (server-side / client filter bucket) */
const STATUS_VIEW_OPTIONS = [
  { value: 'active', label: 'Active', tooltip: 'Due or overdue' },
  { value: 'upcoming', label: 'Upcoming', tooltip: 'Scheduled ahead' },
  { value: 'done', label: 'Done', tooltip: 'Completed items' },
  { value: 'all', label: 'All', tooltip: 'Every TODO' },
];

const SORT_LABELS = {
  file: 'By file',
  state: 'By state',
  date: 'By date',
  priority: 'By priority',
};

const todosPopoverFieldSx = {
  width: '100%',
  minWidth: 260,
  '& .MuiInputLabel-root': { fontSize: '0.875rem' },
  '& .MuiOutlinedInput-root': { fontSize: '0.875rem' },
  '& .MuiSelect-select': { fontSize: '0.875rem', lineHeight: 1.43 },
};

const readFilterPrefs = () => {
  try {
    const raw = localStorage.getItem(ORG_TODOS_PREFS_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw);
    if (p && typeof p === 'object') return p;
  } catch (_) {}
  return null;
};

const OrgTodosView = ({ onOpenDocument }) => {
  const prefs = readFilterPrefs();
  const [filterState, setFilterState] = useState(prefs?.filterState ?? 'active');
  const [sortBy, setSortBy] = useState(prefs?.sortBy ?? 'file');
  const [tagFilter, setTagFilter] = useState(prefs?.tagFilter ?? '');
  const [priorityFilter, setPriorityFilter] = useState(prefs?.priorityFilter ?? '');
  const [categoryFilter, setCategoryFilter] = useState(prefs?.categoryFilter ?? '');
  const [loading, setLoading] = useState(true);
  const [todosData, setTodosData] = useState(null);
  const [error, setError] = useState(null);
  const [refileDialogOpen, setRefileDialogOpen] = useState(false);
  const [refileItem, setRefileItem] = useState(null);
  const [archiveDialogOpen, setArchiveDialogOpen] = useState(false);
  const [archiveItem, setArchiveItem] = useState(null);
  const [moveMenuAnchor, setMoveMenuAnchor] = useState(null);
  const [moveMenuItem, setMoveMenuItem] = useState(null);
  const [bulkArchiveDialogOpen, setBulkArchiveDialogOpen] = useState(false);
  const [bulkArchiveFile, setBulkArchiveFile] = useState('');
  const [bulkArchiving, setBulkArchiving] = useState(false);
  const [filtersPopoverAnchor, setFiltersPopoverAnchor] = useState(null);
  const [sortMenuAnchor, setSortMenuAnchor] = useState(null);
  const [collapsedSections, setCollapsedSections] = useState({});
  
  // Collapsible section state: expanded paths as Set of stringified keys for fast lookup
  const [expandedPaths, setExpandedPaths] = useState(new Set());
  
  // Track which files are expanded
  const [expandedFiles, setExpandedFiles] = useState(new Set());

  // Persist filter/sort preferences to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(ORG_TODOS_PREFS_KEY, JSON.stringify({
        filterState,
        sortBy,
        tagFilter: tagFilter || '',
        priorityFilter: priorityFilter || '',
        categoryFilter: categoryFilter || '',
      }));
    } catch (_) {}
  }, [filterState, sortBy, tagFilter, priorityFilter, categoryFilter]);

  // Toggle expand/collapse for a specific path
  const togglePath = useCallback((path) => {
    setExpandedPaths(prev => {
      const newSet = new Set(prev);
      const pathKey = JSON.stringify(path);
      if (newSet.has(pathKey)) {
        newSet.delete(pathKey);
      } else {
        newSet.add(pathKey);
      }
      return newSet;
    });
  }, []);

  // Check if a path is expanded
  const isPathExpanded = useCallback((path) => {
    return expandedPaths.has(JSON.stringify(path));
  }, [expandedPaths]);
  
  // Toggle file expansion
  const toggleFile = useCallback((filename) => {
    setExpandedFiles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(filename)) {
        newSet.delete(filename);
      } else {
        newSet.add(filename);
      }
      return newSet;
    });
  }, []);
  
  // Check if a file is expanded
  const isFileExpanded = useCallback((filename) => {
    if (!filename) return false;
    return expandedFiles.has(filename);
  }, [expandedFiles]);

  // Group todos by filename, then by parent path
  const groupByFile = useCallback((todos) => {
    const grouped = {};
    todos.forEach(todo => {
      if (!grouped[todo.filename]) {
        grouped[todo.filename] = [];
      }
      grouped[todo.filename].push(todo);
    });
    return grouped;
  }, []);

  // Map filter to TODO states (API request)
  const getStatesForFilter = useCallback((filter) => {
    switch (filter) {
      case 'active':
      case 'upcoming':
        return 'TODO,NEXT,STARTED,WAITING,HOLD';
      case 'done':
        return 'DONE,CANCELED,CANCELLED,WONTFIX,FIXED';
      case 'all':
        return null; // No filter
      default:
        return 'TODO,NEXT,STARTED,WAITING,HOLD';
    }
  }, []);

  // Parse date string (YYYY-MM-DD or YYYY-MM-DD HH:mm:ss) to date-only for comparison
  // Extract YYYY-MM-DD from org timestamps (e.g. "<2026-10-01 Thu>", "2026-10-01 14:00", "2026-10-01")
  const getDatePart = useCallback((str) => {
    if (!str || typeof str !== 'string') return null;
    const m = String(str).trim().match(/(\d{4}-\d{2}-\d{2})/);
    return m ? m[1] : null;
  }, []);

  // True if item has at least one of scheduled/deadline in the future (after today)
  const isScheduledInFuture = useCallback((item) => {
    const today = new Date().toISOString().slice(0, 10);
    const scheduled = getDatePart(item.scheduled);
    const deadline = getDatePart(item.deadline);
    return (scheduled && scheduled > today) || (deadline && deadline > today);
  }, [getDatePart]);

  // True if item is "actionable now": no dates, or all dates are today or in the past
  const isActionableNow = useCallback((item) => {
    const today = new Date().toISOString().slice(0, 10);
    const scheduled = getDatePart(item.scheduled);
    const deadline = getDatePart(item.deadline);
    const scheduledOk = !scheduled || scheduled <= today;
    const deadlineOk = !deadline || deadline <= today;
    return scheduledOk && deadlineOk;
  }, [getDatePart]);

  // Load TODO data (universal todo API; returns 0-based line_number for inline toggle/update)
  const loadTodos = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const states = getStatesForFilter(filterState);
      const response = await orgService.listTodos({
        scope: 'all',
        states: states ? states.split(',') : undefined,
        limit: 500,
      });
      if (response.success) {
        setTodosData({
          success: true,
          results: response.results || [],
          count: response.count || 0,
          files_searched: response.files_searched || 0,
        });
      } else {
        setError(response.error || 'Failed to load TODOs');
      }
    } catch (err) {
      console.error('TODOs load error:', err);
      setError(err.message || 'Failed to load TODOs');
    } finally {
      setLoading(false);
    }
  }, [filterState, getStatesForFilter]);

  // Load on mount and when filter changes
  useEffect(() => {
    loadTodos();
  }, [loadTodos]);
  
  // Expand all files by default when data loads
  useEffect(() => {
    if (todosData?.results && sortBy === 'file') {
      // Use groupByFile directly on results to get unique filenames
      // Don't use getSortedTodos here to avoid dependency on filters
      const grouped = groupByFile(todosData.results);
      const filenames = Object.keys(grouped);
      setExpandedFiles(new Set(filenames));
    }
  }, [todosData, sortBy, groupByFile]);

  const [actionItemId, setActionItemId] = useState(null);

  const handleToggleTodo = useCallback(async (item, e) => {
    e.stopPropagation();
    if (item.file_path == null || item.line_number == null) return;
    setActionItemId(`${item.file_path}:${item.line_number}`);
    try {
      const res = await orgService.toggleTodo(item.file_path, item.line_number, item.heading);
      if (res.success) {
        loadTodos();
      } else {
        alert(res.error || 'Toggle failed');
      }
    } catch (err) {
      console.error('Toggle todo error:', err);
      alert(err.message || 'Toggle failed');
    } finally {
      setActionItemId(null);
    }
  }, [loadTodos]);

  const handleStateChange = useCallback(async (item, newState, e) => {
    e.stopPropagation();
    if (item.file_path == null || item.line_number == null) return;
    setActionItemId(`${item.file_path}:${item.line_number}`);
    try {
      const res = await orgService.updateTodo(item.file_path, item.line_number, { new_state: newState }, item.heading);
      if (res.success) {
        loadTodos();
      } else {
        alert(res.error || 'Update failed');
      }
    } catch (err) {
      console.error('Update todo error:', err);
      alert(err.message || 'Update failed');
    } finally {
      setActionItemId(null);
    }
  }, [loadTodos]);

  // Handle clicking a TODO item
  const handleItemClick = async (item) => {
    if (!onOpenDocument) return;

    let documentId = item.document_id;

    // If document_id is missing, try to look it up from filename
    if (!documentId && item.filename) {
      try {
        console.log(`🔍 Looking up document_id for: ${item.filename}`);
        const lookupResult = await apiService.get(`/api/org/lookup-document?filename=${encodeURIComponent(item.filename)}`);
        
        if (lookupResult.success && lookupResult.document_id) {
          documentId = lookupResult.document_id;
          console.log(`✅ Found document_id: ${documentId}`);
        } else {
          console.error('❌ Could not find document ID for:', item.filename);
          alert(`❌ Could not find document ID for: ${item.filename}`);
          return;
        }
      } catch (err) {
        console.error('❌ Failed to lookup document:', err);
        alert(`❌ Failed to find document: ${item.filename}`);
        return;
      }
    }

    if (!documentId) {
      console.error('❌ TODO item missing document_id:', item);
      alert(`❌ Could not find document ID for: ${item.filename}`);
      return;
    }

    console.log('✅ Opening org file:', documentId);
    
    // Open document with scroll parameters
    onOpenDocument({
      documentId: documentId,
      documentName: item.filename,
      scrollToLine: item.line_number,
      scrollToHeading: item.heading
    });
  };

  // Sort and filter todos
  const getSortedTodos = useCallback((todos) => {
    if (!todos) return [];

    let filtered = [...todos];
    
    // Apply tag filter
    if (tagFilter) {
      filtered = filtered.filter(todo => 
        todo.tags && todo.tags.includes(tagFilter)
      );
    }
    if (priorityFilter) {
      filtered = filtered.filter(todo => todo.priority === priorityFilter);
    }
    if (categoryFilter) {
      filtered = filtered.filter(todo => todo.category === categoryFilter);
    }
    
    // Apply state filter
    const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
    if (filterState === 'active') {
      filtered = filtered.filter(todo => !doneStates.includes(todo.todo_state) && isActionableNow(todo));
    } else if (filterState === 'upcoming') {
      filtered = filtered.filter(todo => !doneStates.includes(todo.todo_state) && isScheduledInFuture(todo));
    } else if (filterState === 'done') {
      filtered = filtered.filter(todo => doneStates.includes(todo.todo_state));
    }
    // 'all' shows everything
    
    // Sort
    switch (sortBy) {
      case 'file':
        filtered.sort((a, b) => a.filename.localeCompare(b.filename));
        break;
      case 'state':
        filtered.sort((a, b) => (a.todo_state || '').localeCompare(b.todo_state || ''));
        break;
      case 'date':
        filtered.sort((a, b) => {
          const aDate = a.scheduled || a.deadline || '';
          const bDate = b.scheduled || b.deadline || '';
          return bDate.localeCompare(aDate);
        });
        break;
      case 'priority': {
        const order = { A: 0, B: 1, C: 2 };
        filtered.sort((a, b) =>
          (order[a.priority] ?? 3) - (order[b.priority] ?? 3)
        );
        break;
      }
      default:
        break;
    }

    return filtered;
  }, [sortBy, filterState, tagFilter, priorityFilter, categoryFilter, isActionableNow, isScheduledInFuture]);

  // Get badge color for TODO state
  const getTodoStateColor = (state) => {
    const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
    return doneStates.includes(state) ? 'success' : 'error';
  };

  // Group todos within a file by their parent heading path
  const groupByParentPath = useCallback((todos) => {
    // Separate into root-level (no parents) and nested (has parents)
    const rootTodos = todos.filter(todo => !todo.parent_path || todo.parent_path.length === 0);
    const nestedTodos = todos.filter(todo => todo.parent_path && todo.parent_path.length > 0);
    
    // Group nested todos by their parent path
    const parentGroups = {};
    nestedTodos.forEach(todo => {
      const pathKey = todo.parent_path.join(' > ');
      if (!parentGroups[pathKey]) {
        parentGroups[pathKey] = {
          path: todo.parent_path,
          levels: todo.parent_levels || [],
          todos: []
        };
      }
      parentGroups[pathKey].todos.push(todo);
    });
    
    return {
      rootTodos,
      parentGroups: Object.values(parentGroups).sort((a, b) => {
        // Sort by first parent level, then by path string
        if (a.levels[0] !== b.levels[0]) {
          return a.levels[0] - b.levels[0];
        }
        return a.path.join(' > ').localeCompare(b.path.join(' > '));
      })
    };
  }, []);

  // Build hierarchical tree from flat TODO list with parent paths
  const buildHierarchicalTree = useCallback((todos) => {
    /**
     * Builds a tree structure for TODOs based on their parent_path hierarchy
     * 
     * Tree node structure:
     * {
     *   heading: "Parent Heading",
     *   level: 1,
     *   todos: [todo1, todo2],  // Org TODO entries directly under this heading
     *   children: [childNode1, childNode2],  // Sub-headings
     *   path: ["Parent", "Child"],  // Full path to this node
     *   isOrphan: false  // True if this is the "File Root" section
     * }
     */
    const tree = {
      heading: null,
      level: 0,
      todos: [],
      children: [],
      path: [],
      isOrphan: true  // Root node represents file root
    };

    // Helper to find or create a node in the tree
    const findOrCreateNode = (parentNode, pathSegment, level, fullPath) => {
      let node = parentNode.children.find(child => child.heading === pathSegment);
      if (!node) {
        node = {
          heading: pathSegment,
          level: level,
          todos: [],
          children: [],
          path: fullPath,
          isOrphan: false
        };
        parentNode.children.push(node);
      }
      return node;
    };

    // Process each org TODO row
    todos.forEach(todo => {
      const parentPath = todo.parent_path || [];
      const parentLevels = todo.parent_levels || [];

      if (parentPath.length === 0) {
        // Orphan TODO - directly at file root
        tree.todos.push(todo);
      } else {
        // Navigate/build the tree to the correct parent node
        let currentNode = tree;
        
        for (let i = 0; i < parentPath.length; i++) {
          const pathSegment = parentPath[i];
          const level = parentLevels[i] || (i + 1);
          const fullPath = parentPath.slice(0, i + 1);
          
          currentNode = findOrCreateNode(currentNode, pathSegment, level, fullPath);
        }

        // Add TODO to its parent node
        currentNode.todos.push(todo);
      }
    });

    return tree;
  }, []);

  // Expand all sections
  const expandAll = useCallback(() => {
    const allPaths = new Set();
    const allFiles = new Set();
    
    const collectPaths = (node) => {
      if (node.path && node.path.length > 0) {
        allPaths.add(JSON.stringify(node.path));
      }
      if (node.children) {
        node.children.forEach(child => collectPaths(child));
      }
    };
    
    // Collect from all files
    if (sortBy === 'file' && todosData?.results) {
      const grouped = groupByFile(getSortedTodos(todosData.results));
      Object.keys(grouped).forEach(filename => {
        allFiles.add(filename);
        const tree = buildHierarchicalTree(grouped[filename]);
        collectPaths(tree);
      });
    }
    
    setExpandedPaths(allPaths);
    setExpandedFiles(allFiles);
  }, [sortBy, todosData, groupByFile, getSortedTodos, buildHierarchicalTree]);

  // Collapse all sections
  const collapseAll = useCallback(() => {
    setExpandedPaths(new Set());
    setExpandedFiles(new Set());
  }, []);

  // Extract all unique tags from todos
  const getAllTags = useCallback((todos) => {
    if (!todos) return [];
    
    const tagSet = new Set();
    todos.forEach(todo => {
      if (todo.tags && Array.isArray(todo.tags)) {
        todo.tags.forEach(tag => tagSet.add(tag));
      }
    });
    
    return Array.from(tagSet).sort();
  }, []);

  // Extract all unique categories from todos
  const getAllCategories = useCallback((todos) => {
    if (!todos) return [];
    const categorySet = new Set();
    todos.forEach(todo => {
      if (todo.category && String(todo.category).trim()) {
        categorySet.add(String(todo.category).trim());
      }
    });
    return Array.from(categorySet).sort();
  }, []);

  const clearRefinements = useCallback(() => {
    setTagFilter('');
    setPriorityFilter('');
    setCategoryFilter('');
  }, []);

  // Handle bulk archive for a specific file
  const handleBulkArchive = async (filename) => {
    try {
      setBulkArchiving(true);
      console.log('Bulk archiving DONE items from:', filename);
      
      // Construct file path (assuming OrgMode folder)
      const filePath = `OrgMode/${filename}`;
      
      const response = await apiService.post('/api/org/archive-bulk', {
        source_file: filePath
      });
      
      if (response.success) {
        console.log(`✅ Bulk archive successful: ${response.archived_count} items archived`);
        alert(`✅ Archived ${response.archived_count} DONE items from ${filename}`);
        
        // Reload TODOs to reflect changes
        loadTodos();
      } else {
        throw new Error(response.error || 'Bulk archive failed');
      }
    } catch (err) {
      console.error('❌ Bulk archive failed:', err);
      alert(`❌ Bulk archive failed: ${err.message}`);
    } finally {
      setBulkArchiving(false);
      setBulkArchiveDialogOpen(false);
    }
  };

  const openMoveMenu = useCallback((e, item) => {
    e.stopPropagation();
    setMoveMenuAnchor(e.currentTarget);
    setMoveMenuItem(item);
  }, []);

  const closeMoveMenu = useCallback(() => {
    setMoveMenuAnchor(null);
    setMoveMenuItem(null);
  }, []);

  const handleArchiveDialogComplete = useCallback(
    (result) => {
      setArchiveDialogOpen(false);
      setArchiveItem(null);
      if (result?.success) {
        loadTodos();
      }
    },
    [loadTodos]
  );

  const sortedTodos = getSortedTodos(todosData?.results || []);
  const groupedTodos = sortBy === 'file' ? groupByFile(sortedTodos) : null;

  // Render hierarchical tree with indentation and collapsible sections
  const renderHierarchicalNode = useCallback((node, depth = 0, baseIndent = 0, filename = null) => {
    /**
     * Recursively render a tree node with:
     * - Collapsible heading (if not orphan root)
     * - TODOs directly under this heading
     * - Child headings (recursively)
     * 
     * Indentation logic:
     * - baseIndent: starting indent level (0 for file root)
     * - Orphan TODOs and level-1 headings: baseIndent
     * - Level-2 headings: baseIndent + 1
     * - Level-3+ headings: baseIndent + (level - 1)
     */
    
    // Calculate total TODOs in this subtree
    const countTodos = (n) => {
      let count = n.todos.length;
      if (n.children) {
        n.children.forEach(child => count += countTodos(child));
      }
      return count;
    };
    
    const totalTodos = countTodos(node);
    if (totalTodos === 0) return null;  // Don't render empty sections
    
    const isExpanded = node.path && node.path.length > 0 ? isPathExpanded(node.path) : true;
    
    // For orphan nodes, check if the file is expanded
    const shouldShowOrphanTodos = node.isOrphan ? (filename ? isFileExpanded(filename) : true) : false;
    
    // Calculate indent for this node
    // Root and level-1 headings have same indent
    const nodeIndent = node.isOrphan ? baseIndent : 
                       node.level === 1 ? baseIndent : 
                       baseIndent + (node.level - 1);
    
    return (
      <Box key={node.path ? JSON.stringify(node.path) : 'root'} sx={{ mb: 2 }}>
        {/* Section heading (if not root) */}
        {node.heading && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              ml: nodeIndent * 3,  // 3 = spacing multiplier
              mb: 1,
              cursor: 'pointer',
              '&:hover': { backgroundColor: 'action.hover' },
              borderRadius: 1,
              p: 0.5
            }}
            onClick={() => togglePath(node.path)}
          >
            {isExpanded ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
            <Typography
              variant="subtitle2"
              sx={{
                fontWeight: 600,
                color: 'primary.main',
                flex: 1
              }}
            >
              {node.heading}
            </Typography>
            <Chip label={totalTodos} size="small" />
          </Box>
        )}
        
        {/* Org TODO entries directly under this heading (when expanded) */}
        {/* For orphan nodes, show org TODO entries when file is expanded (no "File Root" wrapper) */}
        {/* For regular nodes, show when heading is expanded */}
        {((node.isOrphan && shouldShowOrphanTodos) || (!node.isOrphan && isExpanded)) && node.todos.length > 0 && (
          <List disablePadding sx={{ ml: (nodeIndent + 1) * 3 }}>
            {node.todos.map((item, idx) => {
              const hasStateSelect = item.file_path != null && item.line_number != null;
              return (
              <React.Fragment key={idx}>
                {idx > 0 && <Divider />}
                <ListItem
                  disablePadding
                  alignItems="stretch"
                  sx={{ display: 'flex', flexDirection: 'row', gap: 0.5 }}
                >
                  <ListItemButton
                    onClick={() => handleItemClick(item)}
                    sx={{
                      flex: 1,
                      minWidth: 0,
                      py: 0.75,
                      alignItems: 'flex-start',
                      overflow: 'hidden',
                    }}
                  >
                    <Box sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflow: 'hidden' }}>
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 0.75,
                          minWidth: 0,
                          maxWidth: '100%',
                          overflow: 'hidden'
                        }}
                      >
                        {item.priority && (
                          <Chip
                            label={`#${item.priority}`}
                            size="small"
                            color={item.priority === 'A' ? 'error' : item.priority === 'B' ? 'warning' : 'default'}
                            sx={{ fontWeight: 700, fontSize: '0.65rem', height: 18, minWidth: 'auto', flexShrink: 0, '& .MuiChip-label': { px: 0.5 } }}
                          />
                        )}
                        <Typography
                          variant="body2"
                          component="span"
                          sx={{
                            fontWeight: 500,
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            flex: '1 1 0%',
                            minWidth: 0,
                            maxWidth: '100%'
                          }}
                        >
                          {item.heading}
                        </Typography>
                        {item.tags && item.tags.length > 0 && (
                          <Box
                            sx={{
                              display: 'flex',
                              gap: 0.5,
                              flexShrink: 1,
                              minWidth: 0,
                              maxWidth: '42%',
                              overflow: 'hidden',
                              flexWrap: 'nowrap'
                            }}
                          >
                            {item.tags.map(tag => (
                              <Chip
                                key={tag}
                                label={tag}
                                size="small"
                                color="primary"
                                variant="outlined"
                                sx={{
                                  fontSize: '0.65rem',
                                  height: 18,
                                  flexShrink: 1,
                                  minWidth: 0,
                                  maxWidth: '100%',
                                  '& .MuiChip-label': { px: 0.5, overflow: 'hidden', textOverflow: 'ellipsis' }
                                }}
                              />
                            ))}
                          </Box>
                        )}
                        {item.effort && (
                          <Typography variant="caption" sx={{ color: 'text.secondary', flexShrink: 0 }}>
                            ~{item.effort}
                          </Typography>
                        )}
                        <Box sx={{ flexGrow: 1, flexShrink: 1, minWidth: 0 }} />
                        {item.scheduled && (
                          <Typography variant="caption" sx={{ color: 'info.main', whiteSpace: 'nowrap', flexShrink: 0 }}>
                            <Schedule sx={{ fontSize: 12, verticalAlign: 'middle', mr: 0.25 }} />
                            {item.scheduled.split(' ')[0]}
                          </Typography>
                        )}
                        {item.deadline && (
                          <Typography variant="caption" sx={{ color: 'warning.main', fontWeight: 600, whiteSpace: 'nowrap', flexShrink: 0 }}>
                            <ErrorIcon sx={{ fontSize: 12, verticalAlign: 'middle', mr: 0.25 }} />
                            {item.deadline.split(' ')[0]}
                          </Typography>
                        )}
                      </Box>
                      {item.body_preview && (
                        <Typography
                          variant="caption"
                          component="div"
                          sx={{
                            color: 'text.secondary',
                            mt: 0.25,
                            display: 'block',
                            minWidth: 0,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                            opacity: 0.7,
                            width: '100%',
                            maxWidth: '100%',
                          }}
                        >
                          {item.body_preview}
                        </Typography>
                      )}
                    </Box>
                  </ListItemButton>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      flexShrink: 0,
                      gap: 0.5,
                      pr: 0.5,
                    }}
                  >
                    {hasStateSelect && (
                      <Select
                        size="small"
                        value={item.todo_state || 'TODO'}
                        onChange={(e) => handleStateChange(item, e.target.value, e)}
                        onClick={(e) => e.stopPropagation()}
                        sx={todoRowStateSelectSx}
                        MenuProps={todosStateRowMenuProps}
                        disabled={!!actionItemId}
                        renderValue={(v) => (
                          <Typography variant="body2" component="span" sx={{ fontWeight: 500, lineHeight: 1.43 }}>
                            {todoStateDisplayLabel(v)}
                          </Typography>
                        )}
                      >
                        {TODO_STATE_OPTIONS.map(({ value, label }) => (
                          <MenuItem key={value} value={value} dense>
                            <Typography variant="body2" component="span" sx={{ fontWeight: 400, lineHeight: 1.43 }}>
                              {label}
                            </Typography>
                          </MenuItem>
                        ))}
                      </Select>
                    )}
                    <Tooltip title="Refile or archive…">
                      <IconButton
                        edge="end"
                        size="small"
                        color="primary"
                        onClick={(e) => openMoveMenu(e, item)}
                        aria-label="Refile or archive"
                        sx={{
                          flexShrink: 0,
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', lineHeight: 0 }}>
                          <DriveFileMove sx={{ fontSize: '1.125rem' }} />
                          <ArrowDropDown sx={{ fontSize: '1.125rem', ml: -0.5 }} />
                        </Box>
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItem>
              </React.Fragment>
              );
            })}
          </List>
        )}
        
        {/* Child headings (recursive, when expanded) */}
        {isExpanded && node.children && node.children.length > 0 && (
          <Box>
            {node.children.map(child => renderHierarchicalNode(child, depth + 1, baseIndent, filename))}
          </Box>
        )}
      </Box>
    );
  }, [
    handleItemClick,
    handleStateChange,
    actionItemId,
    getTodoStateColor,
    isPathExpanded,
    togglePath,
    isFileExpanded,
    openMoveMenu,
  ]);

  const refineFilterCount = [tagFilter, priorityFilter, categoryFilter].filter(Boolean).length;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderBottom: '1px solid',
          borderColor: 'divider',
          backgroundColor: 'background.paper',
          flexShrink: 0,
        }}
      >
        <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircle /> All TODOs
        </Typography>
      </Box>

      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <Box
          sx={{
            px: 2,
            pt: 2,
            pb: 1.5,
            borderBottom: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'background.paper',
            flexShrink: 0,
          }}
        >
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 1.5 }}>
            <Tooltip title="Opens Quick Capture (Ctrl+Shift+C). Captures to your org inbox as a TODO by default.">
              <Button
                size="small"
                variant="contained"
                startIcon={<Add />}
                onClick={() => {
                  try {
                    window.dispatchEvent(new CustomEvent('openQuickCapture'));
                  } catch (_) {}
                }}
              >
                New TODO
              </Button>
            </Tooltip>
            {sortBy === 'file' && sortedTodos.length > 0 && (
              <>
                <Tooltip title="Expand all sections">
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<UnfoldMore />}
                    onClick={expandAll}
                  >
                    Expand All
                  </Button>
                </Tooltip>
                <Tooltip title="Collapse all sections">
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<UnfoldLess />}
                    onClick={collapseAll}
                  >
                    Collapse All
                  </Button>
                </Tooltip>
              </>
            )}
            {filterState === 'done' && sortBy === 'file' && groupedTodos && Object.keys(groupedTodos).length > 0 && (
              <Tooltip title="Archive all DONE items per file">
                <Button
                  size="small"
                  variant="outlined"
                  color="primary"
                  startIcon={<Archive />}
                  onClick={() => {
                    const files = Object.keys(groupedTodos);
                    if (files.length === 1) {
                      setBulkArchiveFile(files[0]);
                      setBulkArchiveDialogOpen(true);
                    } else {
                      setBulkArchiveFile(files[0]);
                      setBulkArchiveDialogOpen(true);
                    }
                  }}
                >
                  Bulk Archive
                </Button>
              </Tooltip>
            )}
          </Box>

          <Box
            sx={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              gap: 1.5,
              columnGap: 2,
              rowGap: 1.5,
            }}
          >
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'flex-start',
                justifyContent: 'center',
                minWidth: 0,
                flex: '1 1 180px',
                maxWidth: '100%',
                pr: { xs: 0, sm: 1 },
              }}
            >
              {todosData && !loading ? (
                <Typography variant="body2" color="text.secondary" component="div">
                  {sortedTodos.length} of {todosData.count} TODO items · {todosData.files_searched} files searched
                </Typography>
              ) : loading ? (
                <Typography variant="body2" color="text.secondary">
                  Updating…
                </Typography>
              ) : null}
            </Box>

            <ToggleButtonGroup
              exclusive
              value={filterState}
              onChange={(e, next) => {
                if (next !== null) setFilterState(next);
              }}
              size="small"
              color="primary"
              sx={{
                flexShrink: 0,
                '& .MuiToggleButton-root': {
                  typography: 'body2',
                  textTransform: 'none',
                  px: { xs: 1, sm: 1.25 },
                  py: 0.5,
                },
              }}
            >
              {STATUS_VIEW_OPTIONS.map((opt) => (
                <ToggleButton key={opt.value} value={opt.value} title={opt.tooltip}>
                  {opt.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>

            <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
              <Badge badgeContent={refineFilterCount} color="primary" invisible={refineFilterCount === 0}>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<FilterList />}
                  onClick={(e) => setFiltersPopoverAnchor(e.currentTarget)}
                  color={refineFilterCount > 0 ? 'primary' : 'inherit'}
                  sx={{ textTransform: 'none', typography: 'body2' }}
                >
                  Filters
                </Button>
              </Badge>
              <Button
                size="small"
                variant="text"
                color="inherit"
                endIcon={<ArrowDropDown />}
                onClick={(e) => setSortMenuAnchor(e.currentTarget)}
                sx={{ textTransform: 'none', typography: 'body2', fontWeight: 500 }}
              >
                Sort: {SORT_LABELS[sortBy] ?? sortBy}
              </Button>
            </Box>
          </Box>

          {(tagFilter || priorityFilter || categoryFilter) && (
            <Box
              sx={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: 0.75,
                mt: 1.5,
                alignItems: 'center',
              }}
            >
              <Typography variant="caption" color="text.secondary" sx={{ mr: 0.25 }}>
                Refining:
              </Typography>
              {tagFilter && (
                <Chip
                  size="small"
                  label={`Tag: ${tagFilter}`}
                  onDelete={() => setTagFilter('')}
                  icon={<LocalOffer sx={{ fontSize: 16 }} />}
                />
              )}
              {priorityFilter && (
                <Chip
                  size="small"
                  label={`Priority ${priorityFilter}`}
                  onDelete={() => setPriorityFilter('')}
                />
              )}
              {categoryFilter && (
                <Chip size="small" label={`Category: ${categoryFilter}`} onDelete={() => setCategoryFilter('')} />
              )}
              <Button size="small" variant="text" onClick={clearRefinements} sx={{ minWidth: 'auto', ml: 0.5 }}>
                Clear all
              </Button>
            </Box>
          )}

          <Popover
            open={Boolean(filtersPopoverAnchor)}
            anchorEl={filtersPopoverAnchor}
            onClose={() => setFiltersPopoverAnchor(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
            transformOrigin={{ vertical: 'top', horizontal: 'left' }}
            PaperProps={{ sx: { mt: 0.5, p: 0 } }}
          >
            <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', gap: 2, maxWidth: 320 }}>
              <Typography variant="subtitle2" color="text.primary">
                Refine list
              </Typography>
              <FormControl size="small" sx={todosPopoverFieldSx}>
                <InputLabel id="org-todos-popover-tag-label">Tag</InputLabel>
                <Select
                  labelId="org-todos-popover-tag-label"
                  value={tagFilter}
                  label="Tag"
                  onChange={(e) => setTagFilter(e.target.value)}
                  MenuProps={todosSelectMenuProps}
                >
                  <MenuItem value="">
                    <em>Any tag</em>
                  </MenuItem>
                  {getAllTags(todosData?.results || []).map((tag) => (
                    <MenuItem key={tag} value={tag}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.875rem' }}>
                        <LocalOffer sx={{ fontSize: 16 }} />
                        {tag}
                      </Box>
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <FormControl size="small" sx={todosPopoverFieldSx}>
                <InputLabel id="org-todos-popover-priority-label">Priority</InputLabel>
                <Select
                  labelId="org-todos-popover-priority-label"
                  value={priorityFilter}
                  label="Priority"
                  onChange={(e) => setPriorityFilter(e.target.value)}
                  MenuProps={todosSelectMenuProps}
                >
                  <MenuItem value="">
                    <em>Any priority</em>
                  </MenuItem>
                  <MenuItem value="A">A — Critical</MenuItem>
                  <MenuItem value="B">B — Important</MenuItem>
                  <MenuItem value="C">C — Normal</MenuItem>
                </Select>
              </FormControl>
              <Box>
                <FormControl size="small" sx={todosPopoverFieldSx}>
                  <InputLabel id="org-todos-popover-category-label">Category</InputLabel>
                  <Select
                    labelId="org-todos-popover-category-label"
                    value={categoryFilter}
                    label="Category"
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    MenuProps={todosSelectMenuProps}
                  >
                    <MenuItem value="">
                      <em>Any category</em>
                    </MenuItem>
                    {getAllCategories(todosData?.results || []).map((cat) => (
                      <MenuItem key={cat} value={cat}>
                        {cat}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.75 }}>
                  From #+CATEGORY: or :CATEGORY: in org files.
                </Typography>
              </Box>
              <Divider />
              <Button size="small" variant="text" color="primary" onClick={clearRefinements} sx={{ alignSelf: 'flex-start' }}>
                Clear refinements
              </Button>
            </Box>
          </Popover>

          <Menu
            anchorEl={sortMenuAnchor}
            open={Boolean(sortMenuAnchor)}
            onClose={() => setSortMenuAnchor(null)}
            MenuListProps={{ dense: true }}
            PaperProps={{ sx: { minWidth: 200 } }}
          >
            <ListSubheader disableSticky sx={{ typography: 'overline', lineHeight: 2.5, color: 'text.secondary' }}>
              Organization
            </ListSubheader>
            <MenuItem
              selected={sortBy === 'file'}
              onClick={() => {
                setSortBy('file');
                setSortMenuAnchor(null);
              }}
            >
              {SORT_LABELS.file}
            </MenuItem>
            <ListSubheader disableSticky sx={{ typography: 'overline', lineHeight: 2.5, color: 'text.secondary' }}>
              Task
            </ListSubheader>
            <MenuItem
              selected={sortBy === 'state'}
              onClick={() => {
                setSortBy('state');
                setSortMenuAnchor(null);
              }}
            >
              {SORT_LABELS.state}
            </MenuItem>
            <MenuItem
              selected={sortBy === 'date'}
              onClick={() => {
                setSortBy('date');
                setSortMenuAnchor(null);
              }}
            >
              {SORT_LABELS.date}
            </MenuItem>
            <MenuItem
              selected={sortBy === 'priority'}
              onClick={() => {
                setSortBy('priority');
                setSortMenuAnchor(null);
              }}
            >
              {SORT_LABELS.priority}
            </MenuItem>
          </Menu>
        </Box>

        <Box sx={{ flex: 1, overflow: 'auto', minHeight: 0, p: 2 }}>
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Alert severity="error" icon={<ErrorIcon />}>
            {error}
          </Alert>
        )}

        {!loading && todosData && (
          <>
            {sortedTodos.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <CheckCircle sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  {todosData.count === 0 ? 'No TODO Items' : 'No Matching TODOs'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {todosData.count === 0 ? (
                    <>
                      {filterState === 'active' && 'No active TODOs due or overdue'}
                      {filterState === 'upcoming' && 'No upcoming TODOs'}
                      {filterState === 'done' && 'No completed items found'}
                      {filterState === 'all' && 'No TODO items found in your org files'}
                    </>
                  ) : (
                    <>
                      {tagFilter && `No TODOs found with tag "${tagFilter}"`}
                      {!tagFilter && filterState === 'active' && 'No active TODOs due or overdue'}
                      {!tagFilter && filterState === 'upcoming' && 'No upcoming TODOs'}
                      {!tagFilter && filterState === 'done' && 'No completed items found'}
                      {!tagFilter && filterState === 'all' && 'No TODO items match your filters'}
                    </>
                  )}
                </Typography>
                {tagFilter && (
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={() => setTagFilter('')}
                    sx={{ mt: 2 }}
                  >
                    Clear Tag Filter
                  </Button>
                )}
              </Box>
            ) : (
              <>
                {/* Grouped by file view - HIERARCHICAL! */}
                {groupedTodos ? (
                  Object.entries(groupedTodos).map(([filename, items]) => {
                    // Build hierarchical tree for this file
                    const tree = buildHierarchicalTree(items);
                    
                    const fileExpanded = isFileExpanded(filename);
                    
                    return (
                      <Box key={filename} sx={{ mb: 4 }}>
                        {/* File header - now collapsible */}
                        <Box
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            mb: 2,
                            borderBottom: '2px solid',
                            borderColor: 'primary.main',
                            pb: 1,
                            cursor: 'pointer',
                            '&:hover': { backgroundColor: 'action.hover' },
                            borderRadius: 1,
                            px: 1,
                            py: 0.5
                          }}
                          onClick={() => toggleFile(filename)}
                        >
                          {fileExpanded ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
                          <Description fontSize="small" />
                          <Typography
                            variant="subtitle2"
                            sx={{
                              fontWeight: 600,
                              color: 'primary.main',
                              flex: 1
                            }}
                          >
                            {filename}
                          </Typography>
                          <Chip label={items.length} size="small" />
                          {tagFilter && (
                            <Chip 
                              label={`Tag: ${tagFilter}`} 
                              size="small" 
                              color="primary" 
                              variant="outlined"
                              sx={{ ml: 1 }}
                            />
                          )}
                        </Box>

                        {/* Render hierarchical tree - collapsible */}
                        <Collapse in={fileExpanded}>
                          <Paper variant="outlined" sx={{ p: 2 }}>
                            {renderHierarchicalNode(tree, 0, 0, filename)}
                          </Paper>
                        </Collapse>
                      </Box>
                    );
                  })
                ) : (
                  /* Flat list view */
                  <Paper variant="outlined">
                    <List disablePadding>
                      {sortedTodos.map((item, idx) => {
                        const hasStateSelectFlat = item.file_path != null && item.line_number != null;
                        return (
                        <React.Fragment key={idx}>
                          {idx > 0 && <Divider />}
                          <ListItem
                            disablePadding
                            alignItems="stretch"
                            sx={{ display: 'flex', flexDirection: 'row', gap: 0.5 }}
                          >
                            <ListItemButton
                              onClick={() => handleItemClick(item)}
                              sx={{ flex: 1, minWidth: 0, py: 0.75, overflow: 'hidden' }}
                            >
                              <Box sx={{ width: '100%', minWidth: 0, maxWidth: '100%', overflow: 'hidden' }}>
                                <Box
                                  sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 0.75,
                                    minWidth: 0,
                                    maxWidth: '100%',
                                    overflow: 'hidden'
                                  }}
                                >
                                  <Chip
                                    label={item.todo_state}
                                    size="small"
                                    color={getTodoStateColor(item.todo_state)}
                                    sx={{ fontWeight: 600, fontSize: '0.65rem', height: 18, flexShrink: 0, '& .MuiChip-label': { px: 0.5 } }}
                                  />
                                  {item.priority && (
                                    <Chip
                                      label={`#${item.priority}`}
                                      size="small"
                                      color={item.priority === 'A' ? 'error' : item.priority === 'B' ? 'warning' : 'default'}
                                      sx={{ fontWeight: 700, fontSize: '0.65rem', height: 18, minWidth: 'auto', flexShrink: 0, '& .MuiChip-label': { px: 0.5 } }}
                                    />
                                  )}
                                  <Typography
                                    variant="body2"
                                    component="span"
                                    sx={{
                                      fontWeight: 500,
                                      whiteSpace: 'nowrap',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      flex: '1 1 0%',
                                      minWidth: 0,
                                      maxWidth: '100%'
                                    }}
                                  >
                                    {item.heading}
                                  </Typography>
                                  {item.tags && item.tags.length > 0 && (
                                    <Box
                                      sx={{
                                        display: 'flex',
                                        gap: 0.5,
                                        flexShrink: 1,
                                        minWidth: 0,
                                        maxWidth: '36%',
                                        overflow: 'hidden',
                                        flexWrap: 'nowrap'
                                      }}
                                    >
                                      {item.tags.map(tag => (
                                        <Chip
                                          key={tag}
                                          label={tag}
                                          size="small"
                                          color="primary"
                                          variant="outlined"
                                          sx={{
                                            fontSize: '0.65rem',
                                            height: 18,
                                            flexShrink: 1,
                                            minWidth: 0,
                                            maxWidth: '100%',
                                            '& .MuiChip-label': { px: 0.5, overflow: 'hidden', textOverflow: 'ellipsis' }
                                          }}
                                        />
                                      ))}
                                    </Box>
                                  )}
                                  {item.effort && (
                                    <Typography variant="caption" sx={{ color: 'text.secondary', flexShrink: 0 }}>
                                      ~{item.effort}
                                    </Typography>
                                  )}
                                  <Box sx={{ flexGrow: 1, flexShrink: 1, minWidth: 0 }} />
                                  <Typography variant="caption" sx={{ color: 'text.secondary', whiteSpace: 'nowrap', flexShrink: 0, maxWidth: '28%', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                    {item.filename}
                                  </Typography>
                                  {item.scheduled && (
                                    <Typography variant="caption" sx={{ color: 'info.main', whiteSpace: 'nowrap', flexShrink: 0 }}>
                                      <Schedule sx={{ fontSize: 12, verticalAlign: 'middle', mr: 0.25 }} />
                                      {item.scheduled.split(' ')[0]}
                                    </Typography>
                                  )}
                                  {item.deadline && (
                                    <Typography variant="caption" sx={{ color: 'warning.main', fontWeight: 600, whiteSpace: 'nowrap', flexShrink: 0 }}>
                                      <ErrorIcon sx={{ fontSize: 12, verticalAlign: 'middle', mr: 0.25 }} />
                                      {item.deadline.split(' ')[0]}
                                    </Typography>
                                  )}
                                </Box>
                                {item.body_preview && (
                                  <Typography
                                    variant="caption"
                                    component="div"
                                    sx={{
                                      color: 'text.secondary',
                                      mt: 0.25,
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                      whiteSpace: 'nowrap',
                                      opacity: 0.7,
                                      width: '100%',
                                      maxWidth: '100%'
                                    }}
                                  >
                                    {item.body_preview}
                                  </Typography>
                                )}
                              </Box>
                            </ListItemButton>
                            <Box
                              sx={{
                                display: 'flex',
                                alignItems: 'center',
                                flexShrink: 0,
                                gap: 0.5,
                                pr: 0.5,
                              }}
                            >
                              {hasStateSelectFlat && (
                                <Select
                                  size="small"
                                  value={item.todo_state || 'TODO'}
                                  onChange={(e) => handleStateChange(item, e.target.value, e)}
                                  onClick={(e) => e.stopPropagation()}
                                  sx={todoRowStateSelectSx}
                                  MenuProps={todosStateRowMenuProps}
                                  disabled={!!actionItemId}
                                  renderValue={(v) => (
                                    <Typography variant="body2" component="span" sx={{ fontWeight: 500, lineHeight: 1.43 }}>
                                      {todoStateDisplayLabel(v)}
                                    </Typography>
                                  )}
                                >
                                  {TODO_STATE_OPTIONS.map(({ value, label }) => (
                                    <MenuItem key={value} value={value} dense>
                                      <Typography variant="body2" component="span" sx={{ fontWeight: 400, lineHeight: 1.43 }}>
                                        {label}
                                      </Typography>
                                    </MenuItem>
                                  ))}
                                </Select>
                              )}
                              <Tooltip title="Refile or archive…">
                                <IconButton
                                  edge="end"
                                  size="small"
                                  color="primary"
                                  onClick={(e) => openMoveMenu(e, item)}
                                  aria-label="Refile or archive"
                                  sx={{
                                    flexShrink: 0,
                                    border: '1px solid',
                                    borderColor: 'divider',
                                    borderRadius: 1,
                                  }}
                                >
                                  <Box sx={{ display: 'flex', alignItems: 'center', lineHeight: 0 }}>
                                    <DriveFileMove sx={{ fontSize: '1.125rem' }} />
                                    <ArrowDropDown sx={{ fontSize: '1.125rem', ml: -0.5 }} />
                                  </Box>
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </ListItem>
                        </React.Fragment>
                        );
                      })}
                    </List>
                  </Paper>
                )}
              </>
            )}
          </>
        )}
        </Box>
      </Box>

      {/* Org Refile Dialog */}
      {refileItem && (
        <OrgRefileDialog
          open={refileDialogOpen}
          onClose={(result) => {
            setRefileDialogOpen(false);
            if (result?.success) {
              loadTodos();
            }
          }}
          sourceFile={`OrgMode/${refileItem.filename}`}
          sourceLine={refileItem.line_number}
          sourceHeading={refileItem.heading}
        />
      )}

      <Menu
        anchorEl={moveMenuAnchor}
        open={Boolean(moveMenuAnchor)}
        onClose={closeMoveMenu}
        onClick={(e) => e.stopPropagation()}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem
          onClick={(e) => {
            e.stopPropagation();
            const it = moveMenuItem;
            closeMoveMenu();
            if (it) {
              setRefileItem(it);
              setRefileDialogOpen(true);
            }
          }}
        >
          <DriveFileMove fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
          Refile…
        </MenuItem>
        <MenuItem
          onClick={(e) => {
            e.stopPropagation();
            const it = moveMenuItem;
            closeMoveMenu();
            if (it) {
              setArchiveItem(it);
              setArchiveDialogOpen(true);
            }
          }}
        >
          <Archive fontSize="small" sx={{ mr: 1, verticalAlign: 'middle' }} />
          Archive…
        </MenuItem>
      </Menu>

      {archiveItem && (
        <OrgArchiveDialog
          open={archiveDialogOpen}
          onClose={() => {
            setArchiveDialogOpen(false);
            setArchiveItem(null);
          }}
          sourceFile={`OrgMode/${archiveItem.filename}`}
          sourceLine={archiveItem.line_number}
          sourceHeading={archiveItem.heading}
          onArchiveComplete={handleArchiveDialogComplete}
        />
      )}

      {/* Bulk Archive Confirmation Dialog */}
      <Dialog
        open={bulkArchiveDialogOpen}
        onClose={() => !bulkArchiving && setBulkArchiveDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Archive />
          Bulk Archive DONE Items
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" paragraph>
            Archive all DONE items from <strong>{bulkArchiveFile}</strong>?
          </Typography>
          <Typography variant="body2" color="text.secondary">
            DONE items will be moved to <code>{bulkArchiveFile.replace('.org', '_archive.org')}</code>
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setBulkArchiveDialogOpen(false)} disabled={bulkArchiving}>
            Cancel
          </Button>
          <Button
            onClick={() => handleBulkArchive(bulkArchiveFile)}
            variant="contained"
            color="primary"
            disabled={bulkArchiving}
            startIcon={bulkArchiving ? <CircularProgress size={16} /> : <Archive />}
          >
            {bulkArchiving ? 'Archiving...' : 'Archive All DONE'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default OrgTodosView;

