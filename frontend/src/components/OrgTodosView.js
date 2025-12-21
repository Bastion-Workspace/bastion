import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  ToggleButtonGroup,
  ToggleButton,
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
  Collapse
} from '@mui/material';
import {
  CheckCircle,
  Schedule,
  Error as ErrorIcon,
  LocalOffer,
  Description,
  DriveFileMove,
  Archive,
  DeleteSweep,
  ExpandMore,
  ChevronRight,
  UnfoldMore,
  UnfoldLess
} from '@mui/icons-material';
import apiService from '../services/apiService';
import OrgRefileDialog from './OrgRefileDialog';

/**
 * ROOSEVELT'S ORG TODOS VIEW
 * View and manage all TODO items across org files
 */
const OrgTodosView = ({ onOpenDocument }) => {
  const [filterState, setFilterState] = useState('active'); // active, done, all
  const [sortBy, setSortBy] = useState('file'); // file, state, date
  const [tagFilter, setTagFilter] = useState(''); // tag filter
  const [loading, setLoading] = useState(true);
  const [todosData, setTodosData] = useState(null);
  const [error, setError] = useState(null);
  const [refileDialogOpen, setRefileDialogOpen] = useState(false);
  const [refileItem, setRefileItem] = useState(null);
  const [bulkArchiveDialogOpen, setBulkArchiveDialogOpen] = useState(false);
  const [bulkArchiveFile, setBulkArchiveFile] = useState('');
  const [bulkArchiving, setBulkArchiving] = useState(false);
  const [collapsedSections, setCollapsedSections] = useState({});
  
  // **BULLY!** Collapsible section state management
  // Store expanded paths as Set of stringified paths for fast lookup
  const [expandedPaths, setExpandedPaths] = useState(new Set());
  
  // Track which files are expanded
  const [expandedFiles, setExpandedFiles] = useState(new Set());

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
    return expandedFiles.has(filename);
  }, [expandedFiles]);

  // Map filter to TODO states
  const getStatesForFilter = useCallback((filter) => {
    switch (filter) {
      case 'active':
        return 'TODO,NEXT,STARTED,WAITING,HOLD';
      case 'done':
        return 'DONE,CANCELED,CANCELLED,WONTFIX,FIXED';
      case 'all':
        return null; // No filter
      default:
        return 'TODO,NEXT,STARTED,WAITING,HOLD';
    }
  }, []);

  // Load TODO data
  const loadTodos = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const states = getStatesForFilter(filterState);
      const url = states 
        ? `/api/org/todos?states=${encodeURIComponent(states)}`
        : '/api/org/todos';

      const response = await apiService.get(url);

      if (response.success) {
        setTodosData(response);
      } else {
        setError(response.error || 'Failed to load TODOs');
      }
    } catch (err) {
      console.error('❌ TODOs error:', err);
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
      const grouped = groupByFile(getSortedTodos(todosData.results));
      const filenames = Object.keys(grouped);
      setExpandedFiles(new Set(filenames));
    }
  }, [todosData, sortBy, groupByFile, getSortedTodos]);

  // Handle clicking a TODO item
  const handleItemClick = (item) => {
    if (!onOpenDocument) return;

    // document_id is already in the search results!
    if (!item.document_id) {
      console.error('❌ TODO item missing document_id:', item);
      alert(`❌ Could not find document ID for: ${item.filename}`);
      return;
    }

    console.log('✅ ROOSEVELT: Opening org file:', item.document_id);
    
    // Open document with scroll parameters
    onOpenDocument({
      documentId: item.document_id,
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
    
    // Apply state filter
    const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
    if (filterState === 'active') {
      filtered = filtered.filter(todo => !doneStates.includes(todo.todo_state));
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
      default:
        break;
    }

    return filtered;
  }, [sortBy, filterState, tagFilter]);

  // Get badge color for TODO state
  const getTodoStateColor = (state) => {
    const doneStates = ['DONE', 'CANCELED', 'CANCELLED', 'WONTFIX', 'FIXED'];
    return doneStates.includes(state) ? 'success' : 'error';
  };

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

  // **BULLY!** Build hierarchical tree structure from flat TODO list with parent paths
  const buildHierarchicalTree = useCallback((todos) => {
    /**
     * Builds a tree structure for TODOs based on their parent_path hierarchy
     * 
     * Tree node structure:
     * {
     *   heading: "Parent Heading",
     *   level: 1,
     *   todos: [todo1, todo2],  // TODOs directly under this heading
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

    // Process each TODO
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

  // Handle bulk archive for a specific file
  const handleBulkArchive = async (filename) => {
    try {
      setBulkArchiving(true);
      console.log('📦 ROOSEVELT: Bulk archiving DONE items from:', filename);
      
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

  const sortedTodos = getSortedTodos(todosData?.results || []);
  const groupedTodos = sortBy === 'file' ? groupByFile(sortedTodos) : null;

  // **BULLY!** Render hierarchical tree with proper indentation and collapsibility
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
        
        {/* TODOs directly under this heading (when expanded) */}
        {/* For orphan nodes, show TODOs when file is expanded (no "File Root" wrapper) */}
        {/* For regular nodes, show when heading is expanded */}
        {((node.isOrphan && shouldShowOrphanTodos) || (!node.isOrphan && isExpanded)) && node.todos.length > 0 && (
          <List disablePadding sx={{ ml: (nodeIndent + 1) * 3 }}>
            {node.todos.map((item, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && <Divider />}
                <ListItem 
                  disablePadding
                  secondaryAction={
                    <Tooltip title="Refile (Ctrl+Shift+M)">
                      <IconButton 
                        edge="end" 
                        size="small"
                        onClick={(e) => {
                          e.stopPropagation();
                          setRefileItem(item);
                          setRefileDialogOpen(true);
                        }}
                      >
                        <DriveFileMove fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  }
                >
                  <ListItemButton onClick={() => handleItemClick(item)}>
                    <Box sx={{ width: '100%' }}>
                      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.5 }}>
                        <Typography variant="body1" sx={{ flex: 1, fontWeight: 500 }}>
                          {'•'.repeat(item.level)} {item.heading}
                        </Typography>

                        <Chip
                          label={item.todo_state}
                          size="small"
                          color={getTodoStateColor(item.todo_state)}
                          sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                        />
                      </Box>

                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        {item.tags && item.tags.length > 0 && (
                          <Box sx={{ display: 'flex', gap: 0.5 }}>
                            {item.tags.map(tag => (
                              <Chip
                                key={tag}
                                icon={<LocalOffer sx={{ fontSize: 12 }} />}
                                label={tag}
                                size="small"
                                color="primary"
                                variant="outlined"
                                sx={{ fontSize: '0.7rem', height: 20 }}
                              />
                            ))}
                          </Box>
                        )}

                        {item.scheduled && (
                          <Chip
                            icon={<Schedule sx={{ fontSize: 12 }} />}
                            label={`SCHED: ${item.scheduled.split(' ')[0]}`}
                            size="small"
                            color="info"
                            variant="outlined"
                            sx={{ fontSize: '0.7rem', height: 20 }}
                          />
                        )}

                        {item.deadline && (
                          <Chip
                            icon={<ErrorIcon sx={{ fontSize: 12 }} />}
                            label={`DUE: ${item.deadline.split(' ')[0]}`}
                            size="small"
                            color="warning"
                            variant="outlined"
                            sx={{ fontSize: '0.7rem', height: 20 }}
                          />
                        )}
                      </Box>
                    </Box>
                  </ListItemButton>
                </ListItem>
              </React.Fragment>
            ))}
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
  }, [handleItemClick, setRefileItem, setRefileDialogOpen, getTodoStateColor, isPathExpanded, togglePath, isFileExpanded]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <CheckCircle /> All TODOs
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, mb: 1, flexWrap: 'wrap' }}>
          {/* Filter Toggle */}
          <ToggleButtonGroup
            value={filterState}
            exclusive
            onChange={(e, newFilter) => newFilter && setFilterState(newFilter)}
            size="small"
          >
            <ToggleButton value="active">Active</ToggleButton>
            <ToggleButton value="done">Done</ToggleButton>
            <ToggleButton value="all">All</ToggleButton>
          </ToggleButtonGroup>

          {/* Sort Select */}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              label="Sort By"
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="file">By File</MenuItem>
              <MenuItem value="state">By State</MenuItem>
              <MenuItem value="date">By Date</MenuItem>
            </Select>
          </FormControl>

          {/* Tag Filter */}
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Filter by Tag</InputLabel>
            <Select
              value={tagFilter}
              label="Filter by Tag"
              onChange={(e) => setTagFilter(e.target.value)}
            >
              <MenuItem value="">All Tags</MenuItem>
              {getAllTags(todosData?.todos || []).map(tag => (
                <MenuItem key={tag} value={tag}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LocalOffer sx={{ fontSize: 14 }} />
                    {tag}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Expand/Collapse All Buttons - Only for hierarchical view */}
          {sortBy === 'file' && sortedTodos.length > 0 && (
            <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
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
            </Box>
          )}

          {/* Bulk Archive Button - Only show for DONE filter and when grouped by file */}
          {filterState === 'done' && sortBy === 'file' && groupedTodos && Object.keys(groupedTodos).length > 0 && (
            <Tooltip title="Archive all DONE items per file">
              <Button
                size="small"
                variant="outlined"
                color="primary"
                startIcon={<Archive />}
                onClick={() => {
                  // Show dropdown or dialog to select which file to bulk archive
                  const files = Object.keys(groupedTodos);
                  if (files.length === 1) {
                    setBulkArchiveFile(files[0]);
                    setBulkArchiveDialogOpen(true);
                  } else {
                    // For multiple files, could show a menu - for now just archive first file as example
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

        {todosData && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mt: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {sortedTodos.length} of {todosData.count} TODO items • {todosData.files_searched} files searched
              {tagFilter && ` • Filtered by tag: ${tagFilter}`}
            </Typography>
            {tagFilter && (
              <Button
                size="small"
                variant="text"
                color="primary"
                onClick={() => setTagFilter('')}
                sx={{ minWidth: 'auto', px: 1 }}
              >
                Clear Tag Filter
              </Button>
            )}
          </Box>
        )}
      </Box>

      {/* Content Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
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
                      {filterState === 'active' && 'No active TODOs found'}
                      {filterState === 'done' && 'No completed items found'}
                      {filterState === 'all' && 'No TODO items found in your org files'}
                    </>
                  ) : (
                    <>
                      {tagFilter && `No TODOs found with tag "${tagFilter}"`}
                      {!tagFilter && filterState === 'active' && 'No active TODOs found'}
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
                      {sortedTodos.map((item, idx) => (
                        <React.Fragment key={idx}>
                          {idx > 0 && <Divider />}
                          <ListItem disablePadding>
                            <ListItemButton onClick={() => handleItemClick(item)}>
                              <Box sx={{ width: '100%' }}>
                                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, mb: 0.5 }}>
                                  <Typography variant="body1" sx={{ flex: 1, fontWeight: 500 }}>
                                    {'•'.repeat(item.level)} {item.heading}
                                  </Typography>

                                  <Chip
                                    label={item.todo_state}
                                    size="small"
                                    color={getTodoStateColor(item.todo_state)}
                                    sx={{ fontWeight: 600, fontSize: '0.7rem' }}
                                  />
                                </Box>

                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                                  <Chip
                                    icon={<Description fontSize="small" />}
                                    label={item.filename}
                                    size="small"
                                    variant="outlined"
                                    sx={{ fontSize: '0.7rem' }}
                                  />

                                  {item.tags && item.tags.length > 0 && (
                                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                                      {item.tags.map(tag => (
                                        <Chip
                                          key={tag}
                                          icon={<LocalOffer sx={{ fontSize: 12 }} />}
                                          label={tag}
                                          size="small"
                                          color="primary"
                                          variant="outlined"
                                          sx={{ fontSize: '0.7rem', height: 20 }}
                                        />
                                      ))}
                                    </Box>
                                  )}

                                  {item.scheduled && (
                                    <Chip
                                      icon={<Schedule sx={{ fontSize: 12 }} />}
                                      label={`SCHED: ${item.scheduled.split(' ')[0]}`}
                                      size="small"
                                      color="info"
                                      variant="outlined"
                                      sx={{ fontSize: '0.7rem', height: 20 }}
                                    />
                                  )}

                                  {item.deadline && (
                                    <Chip
                                      icon={<ErrorIcon sx={{ fontSize: 12 }} />}
                                      label={`DUE: ${item.deadline.split(' ')[0]}`}
                                      size="small"
                                      color="warning"
                                      variant="outlined"
                                      sx={{ fontSize: '0.7rem', height: 20 }}
                                    />
                                  )}
                                </Box>
                              </Box>
                            </ListItemButton>
                          </ListItem>
                        </React.Fragment>
                      ))}
                    </List>
                  </Paper>
                )}
              </>
            )}
          </>
        )}
      </Box>

      {/* Org Refile Dialog */}
      {refileItem && (
        <OrgRefileDialog
          open={refileDialogOpen}
          onClose={(result) => {
            setRefileDialogOpen(false);
            if (result?.success) {
              console.log('✅ ROOSEVELT: Refile completed, refreshing TODOs...');
              loadTodos(); // Refresh the TODO list
            }
          }}
          sourceFile={`OrgMode/${refileItem.filename}`}
          sourceLine={refileItem.line_number}
          sourceHeading={refileItem.heading}
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

