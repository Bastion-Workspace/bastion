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
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Tooltip,
  Button,
  Avatar
} from '@mui/material';
import {
  Contacts,
  Email,
  Phone,
  Business,
  Cake,
  Description,
  DriveFileMove,
  Person,
  ExpandMore,
  ChevronRight,
  UnfoldMore,
  UnfoldLess,
  Folder
} from '@mui/icons-material';
import apiService from '../services/apiService';
import OrgRefileDialog from './OrgRefileDialog';

/**
 * ROOSEVELT'S ORG CONTACTS VIEW
 * View and manage all contacts across org files
 */
const OrgContactsView = ({ onOpenDocument }) => {
  const [sortBy, setSortBy] = useState('file'); // file, name, category
  const [categoryFilter, setCategoryFilter] = useState(''); // personal, work, family, etc.
  const [loading, setLoading] = useState(true);
  const [contactsData, setContactsData] = useState(null);
  const [error, setError] = useState(null);
  const [refileDialogOpen, setRefileDialogOpen] = useState(false);
  const [refileItem, setRefileItem] = useState(null);
  const [collapsedSections, setCollapsedSections] = useState({});
  const [expandedPaths, setExpandedPaths] = useState(new Set());

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

  // Load contact data
  const loadContacts = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiService.get('/api/org/contacts');

      if (response.success) {
        setContactsData(response);
      } else {
        setError(response.error || 'Failed to load contacts');
      }
    } catch (err) {
      console.error('❌ Contacts error:', err);
      setError(err.message || 'Failed to load contacts');
    } finally {
      setLoading(false);
    }
  }, []);

  // Load on mount
  useEffect(() => {
    loadContacts();
  }, [loadContacts]);

  // Handle clicking a contact item
  const handleItemClick = (item) => {
    if (!onOpenDocument) return;

    if (!item.document_id) {
      console.error('❌ Contact item missing document_id:', item);
      alert(`❌ Could not find document ID for: ${item.filename}`);
      return;
    }

    console.log('✅ ROOSEVELT: Opening org file:', item.document_id);
    
    onOpenDocument({
      documentId: item.document_id,
      documentName: item.filename,
      scrollToLine: item.line_number,
      scrollToHeading: item.heading
    });
  };

  // Sort and filter contacts
  const getSortedContacts = useCallback((contacts) => {
    if (!contacts) return [];

    let filtered = [...contacts];
    
    // Apply category filter (based on tags or parent path)
    if (categoryFilter) {
      filtered = filtered.filter(contact => {
        const tags = contact.tags || [];
        const parentPath = contact.parent_path || [];
        return tags.includes(categoryFilter) || 
               parentPath.some(p => p.toLowerCase().includes(categoryFilter.toLowerCase()));
      });
    }
    
    // Sort
    switch (sortBy) {
      case 'file':
        filtered.sort((a, b) => a.filename.localeCompare(b.filename));
        break;
      case 'name':
        filtered.sort((a, b) => a.heading.localeCompare(b.heading));
        break;
      case 'category':
        filtered.sort((a, b) => {
          const aCat = a.parent_path?.[0] || '';
          const bCat = b.parent_path?.[0] || '';
          return aCat.localeCompare(bCat);
        });
        break;
      default:
        break;
    }

    return filtered;
  }, [sortBy, categoryFilter]);

  // Group contacts by filename
  const groupByFile = useCallback((contacts) => {
    const grouped = {};
    contacts.forEach(contact => {
      if (!grouped[contact.filename]) {
        grouped[contact.filename] = [];
      }
      grouped[contact.filename].push(contact);
    });
    return grouped;
  }, []);

  // Build hierarchical tree structure
  const buildHierarchicalTree = useCallback((contacts) => {
    const tree = {
      heading: null,
      level: 0,
      contacts: [],
      children: [],
      path: [],
      isOrphan: true
    };

    const findOrCreateNode = (parentNode, pathSegment, level, fullPath) => {
      let node = parentNode.children.find(child => child.heading === pathSegment);
      if (!node) {
        node = {
          heading: pathSegment,
          level: level,
          contacts: [],
          children: [],
          path: fullPath,
          isOrphan: false
        };
        parentNode.children.push(node);
      }
      return node;
    };

    contacts.forEach(contact => {
      const parentPath = contact.parent_path || [];
      const parentLevels = contact.parent_levels || [];

      if (parentPath.length === 0) {
        tree.contacts.push(contact);
      } else {
        let currentNode = tree;
        
        for (let i = 0; i < parentPath.length; i++) {
          const pathSegment = parentPath[i];
          const level = parentLevels[i] || (i + 1);
          const fullPath = parentPath.slice(0, i + 1);
          
          currentNode = findOrCreateNode(currentNode, pathSegment, level, fullPath);
        }

        currentNode.contacts.push(contact);
      }
    });

    return tree;
  }, []);

  // Expand all sections
  const expandAll = useCallback(() => {
    const allPaths = new Set();
    
    if (sortBy === 'file' && contactsData?.results) {
      const grouped = groupByFile(contactsData.results);
      Object.values(grouped).forEach(fileContacts => {
        const tree = buildHierarchicalTree(fileContacts);
        const collectPaths = (node) => {
          if (node.path && node.path.length > 0) {
            allPaths.add(JSON.stringify(node.path));
          }
          if (node.children) {
            node.children.forEach(child => collectPaths(child));
          }
        };
        collectPaths(tree);
      });
    }
    
    setExpandedPaths(allPaths);
  }, [sortBy, contactsData, groupByFile, buildHierarchicalTree]);

  // Collapse all sections
  const collapseAll = useCallback(() => {
    setExpandedPaths(new Set());
  }, []);

  // Extract all unique categories from contacts
  const getAllCategories = useCallback((contacts) => {
    if (!contacts) return [];
    
    const categorySet = new Set();
    contacts.forEach(contact => {
      // Add tags
      if (contact.tags && Array.isArray(contact.tags)) {
        contact.tags.forEach(tag => categorySet.add(tag));
      }
      // Add top-level parent as category
      if (contact.parent_path && contact.parent_path.length > 0) {
        categorySet.add(contact.parent_path[0]);
      }
    });
    
    return Array.from(categorySet).sort();
  }, []);

  // Render hierarchical tree
  const renderHierarchicalNode = useCallback((node, depth = 0, baseIndent = 0) => {
    const countContacts = (n) => {
      let count = n.contacts.length;
      if (n.children) {
        n.children.forEach(child => count += countContacts(child));
      }
      return count;
    };
    
    const totalContacts = countContacts(node);
    if (totalContacts === 0) return null;
    
    const isExpanded = node.path && node.path.length > 0 ? isPathExpanded(node.path) : true;
    const nodeIndent = node.isOrphan ? baseIndent : 
                       node.level === 1 ? baseIndent : 
                       baseIndent + (node.level - 1);
    
    return (
      <Box key={node.path ? JSON.stringify(node.path) : 'root'} sx={{ mb: 2 }}>
        {/* Section heading */}
        {node.heading && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              ml: nodeIndent * 3,
              mb: 1,
              cursor: 'pointer',
              '&:hover': { backgroundColor: 'action.hover' },
              borderRadius: 1,
              p: 0.5
            }}
            onClick={() => togglePath(node.path)}
          >
            {isExpanded ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
            <Folder fontSize="small" color="primary" />
            <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'primary.main', flex: 1 }}>
              {node.heading}
            </Typography>
            <Chip label={totalContacts} size="small" />
          </Box>
        )}
        
        {/* Orphan section heading */}
        {node.isOrphan && node.contacts.length > 0 && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              ml: nodeIndent * 3,
              mb: 1,
              cursor: 'pointer',
              '&:hover': { backgroundColor: 'action.hover' },
              borderRadius: 1,
              p: 0.5
            }}
            onClick={() => togglePath(['__orphan__'])}
          >
            {isPathExpanded(['__orphan__']) ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
            <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary', fontStyle: 'italic', flex: 1 }}>
              File Root
            </Typography>
            <Chip label={node.contacts.length} size="small" />
          </Box>
        )}
        
        {/* Contacts directly under this heading */}
        {((node.isOrphan && isPathExpanded(['__orphan__'])) || (!node.isOrphan && isExpanded)) && node.contacts.length > 0 && (
          <List disablePadding sx={{ ml: (nodeIndent + 1) * 3 }}>
            {node.contacts.map((item, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && <Divider />}
                <ListItem 
                  disablePadding
                  secondaryAction={
                    <Tooltip title="Refile Contact">
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
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                      <Avatar sx={{ bgcolor: 'primary.main' }}>
                        <Person />
                      </Avatar>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="body1" sx={{ fontWeight: 500 }}>
                          {item.heading}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
                          {item.properties?.EMAIL && (
                            <Chip
                              icon={<Email sx={{ fontSize: 12 }} />}
                              label={item.properties.EMAIL}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          )}
                          {item.properties?.PHONE && (
                            <Chip
                              icon={<Phone sx={{ fontSize: 12 }} />}
                              label={item.properties.PHONE}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          )}
                          {item.properties?.COMPANY && (
                            <Chip
                              icon={<Business sx={{ fontSize: 12 }} />}
                              label={item.properties.COMPANY}
                              size="small"
                              color="info"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          )}
                          {item.properties?.BIRTHDAY && (
                            <Chip
                              icon={<Cake sx={{ fontSize: 12 }} />}
                              label={item.properties.BIRTHDAY}
                              size="small"
                              color="secondary"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          )}
                          {item.tags && item.tags.length > 0 && item.tags.map(tag => (
                            <Chip
                              key={tag}
                              label={tag}
                              size="small"
                              color="primary"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          ))}
                        </Box>
                      </Box>
                    </Box>
                  </ListItemButton>
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        )}
        
        {/* Child headings */}
        {isExpanded && node.children && node.children.length > 0 && (
          <Box>
            {node.children.map(child => renderHierarchicalNode(child, depth + 1, baseIndent))}
          </Box>
        )}
      </Box>
    );
  }, [handleItemClick, setRefileItem, setRefileDialogOpen, isPathExpanded, togglePath]);

  const sortedContacts = getSortedContacts(contactsData?.results || []);
  const groupedContacts = sortBy === 'file' ? groupByFile(sortedContacts) : null;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Contacts /> All Contacts
        </Typography>

        <Box sx={{ display: 'flex', gap: 2, mb: 1, flexWrap: 'wrap' }}>
          {/* Sort Select */}
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Sort By</InputLabel>
            <Select
              value={sortBy}
              label="Sort By"
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="file">By File</MenuItem>
              <MenuItem value="name">By Name</MenuItem>
              <MenuItem value="category">By Category</MenuItem>
            </Select>
          </FormControl>

          {/* Category Filter */}
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Filter by Category</InputLabel>
            <Select
              value={categoryFilter}
              label="Filter by Category"
              onChange={(e) => setCategoryFilter(e.target.value)}
            >
              <MenuItem value="">All Categories</MenuItem>
              {getAllCategories(contactsData?.results || []).map(cat => (
                <MenuItem key={cat} value={cat}>{cat}</MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Expand/Collapse All Buttons */}
          {sortBy === 'file' && sortedContacts.length > 0 && (
            <Box sx={{ display: 'flex', gap: 1, ml: 'auto' }}>
              <Tooltip title="Expand all sections">
                <Button size="small" variant="outlined" startIcon={<UnfoldMore />} onClick={expandAll}>
                  Expand All
                </Button>
              </Tooltip>
              <Tooltip title="Collapse all sections">
                <Button size="small" variant="outlined" startIcon={<UnfoldLess />} onClick={collapseAll}>
                  Collapse All
                </Button>
              </Tooltip>
            </Box>
          )}
        </Box>

        {contactsData && (
          <Typography variant="caption" color="text.secondary">
            {sortedContacts.length} of {contactsData.count} contacts • {contactsData.files_searched} files searched
            {categoryFilter && ` • Filtered by: ${categoryFilter}`}
          </Typography>
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
          <Alert severity="error">{error}</Alert>
        )}

        {!loading && contactsData && (
          <>
            {sortedContacts.length === 0 ? (
              <Box sx={{ textAlign: 'center', py: 8 }}>
                <Contacts sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" gutterBottom>
                  {contactsData.count === 0 ? 'No Contacts Found' : 'No Matching Contacts'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {categoryFilter ? `No contacts found in category "${categoryFilter}"` : 'No contacts found in your org files'}
                </Typography>
              </Box>
            ) : (
              <>
                {/* Grouped by file view */}
                {groupedContacts ? (
                  Object.entries(groupedContacts).map(([filename, items]) => {
                    const tree = buildHierarchicalTree(items);
                    
                    return (
                      <Box key={filename} sx={{ mb: 4 }}>
                        <Typography
                          variant="subtitle2"
                          sx={{
                            fontWeight: 600,
                            mb: 2,
                            color: 'primary.main',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            borderBottom: '2px solid',
                            borderColor: 'primary.main',
                            pb: 1
                          }}
                        >
                          <Description fontSize="small" />
                          {filename}
                          <Chip label={items.length} size="small" sx={{ ml: 'auto' }} />
                        </Typography>

                        <Paper variant="outlined" sx={{ p: 2 }}>
                          {renderHierarchicalNode(tree, 0, 0)}
                        </Paper>
                      </Box>
                    );
                  })
                ) : (
                  /* Flat list view */
                  <Paper variant="outlined">
                    <List disablePadding>
                      {sortedContacts.map((item, idx) => (
                        <React.Fragment key={idx}>
                          {idx > 0 && <Divider />}
                          <ListItem disablePadding>
                            <ListItemButton onClick={() => handleItemClick(item)}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                                <Avatar sx={{ bgcolor: 'primary.main' }}>
                                  <Person />
                                </Avatar>
                                <Box sx={{ flex: 1 }}>
                                  <Typography variant="body1" sx={{ fontWeight: 500 }}>
                                    {item.heading}
                                  </Typography>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mt: 0.5 }}>
                                    <Chip
                                      icon={<Description fontSize="small" />}
                                      label={item.filename}
                                      size="small"
                                      variant="outlined"
                                      sx={{ fontSize: '0.7rem' }}
                                    />
                                    {item.properties?.EMAIL && (
                                      <Chip
                                        icon={<Email sx={{ fontSize: 12 }} />}
                                        label={item.properties.EMAIL}
                                        size="small"
                                        variant="outlined"
                                        sx={{ fontSize: '0.7rem', height: 20 }}
                                      />
                                    )}
                                    {item.properties?.PHONE && (
                                      <Chip
                                        icon={<Phone sx={{ fontSize: 12 }} />}
                                        label={item.properties.PHONE}
                                        size="small"
                                        variant="outlined"
                                        sx={{ fontSize: '0.7rem', height: 20 }}
                                      />
                                    )}
                                  </Box>
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
              console.log('✅ ROOSEVELT: Refile completed, refreshing contacts...');
              loadContacts();
            }
          }}
          sourceFile={`OrgMode/${refileItem.filename}`}
          sourceLine={refileItem.line_number}
          sourceHeading={refileItem.heading}
        />
      )}
    </Box>
  );
};

export default OrgContactsView;

