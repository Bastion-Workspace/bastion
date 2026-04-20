import React, { useState, useRef, useLayoutEffect, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Chip,
  Divider,
  Tooltip,
  Popover,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Badge,
} from '@mui/material';
import {
  Description,
  Settings,
  Logout,
  PersonAdd,
  LightMode,
  DarkMode,
  Menu as MenuIcon,
  Mail,
  MailOutline,
  Group,
  MusicNote,
  HelpOutline,
  Map,
  Home,
  Dashboard,
  Build,
  Code,
  Notifications,
  NotificationsNone,
  PushPin,
  PushPinOutlined,
  SportsEsports,
  Info,
  Tune,
  Apps,
  DragIndicator,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { useCapabilities } from '../contexts/CapabilitiesContext';
import { useMessaging } from '../contexts/MessagingContext';
import { useNotifications } from '../contexts/NotificationContext';
import { useTeam } from '../contexts/TeamContext';
import apiService from '../services/apiService';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import HelpOverlay from './HelpOverlay';
import NotificationDropdown from './NotificationDropdown';
import AboutDialog from './AboutDialog';

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const { darkMode, toggleDarkMode, themePreference } = useTheme();
  const { isAdmin, has } = useCapabilities();
  const { toggleDrawer, totalUnreadCount } = useMessaging();
  const { unreadCount: notificationUnreadCount } = useNotifications();
  const { pendingInvitations, unreadCounts } = useTeam();
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationAnchorEl, setNotificationAnchorEl] = useState(null);
  const notificationBellRef = useRef(null);
  const [logoError, setLogoError] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [helpInitialTopicId, setHelpInitialTopicId] = useState(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [pagesAnchorEl, setPagesAnchorEl] = useState(null);

  useEffect(() => {
    const onOpenHelp = (ev) => {
      const topicId = ev.detail?.topicId;
      setHelpInitialTopicId(topicId || null);
      setHelpOpen(true);
    };
    window.addEventListener('bastion-open-help', onOpenHelp);
    return () => window.removeEventListener('bastion-open-help', onOpenHelp);
  }, []);
  const appBarRef = useRef(null);

  const publishNavHeight = useCallback(() => {
    const el = appBarRef.current;
    if (!el) return;
    const h = Math.round(el.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--app-nav-height', `${h}px`);
  }, []);

  useLayoutEffect(() => {
    publishNavHeight();
    window.addEventListener('resize', publishNavHeight);
    const el = appBarRef.current;
    const ro =
      typeof ResizeObserver !== 'undefined' && el
        ? new ResizeObserver(() => publishNavHeight())
        : null;
    if (ro && el) ro.observe(el);
    return () => {
      window.removeEventListener('resize', publishNavHeight);
      if (ro) ro.disconnect();
    };
  }, [publishNavHeight]);

  const STORAGE_KEY = 'bastion_pinned_nav';
  const DEFAULT_PINS = ['/documents', '/teams'];
  const [pinnedPaths, setPinnedPaths] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : DEFAULT_PINS;
    } catch { return DEFAULT_PINS; }
  });
  const togglePin = (path) => {
    setPinnedPaths(prev => {
      const next = prev.includes(path)
        ? prev.filter(p => p !== path)
        : [...prev, path];
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const { data: pendingApprovals = [] } = useQuery(
    'agentPendingApprovals',
    () => apiService.agentFactory.listPendingApprovals(),
    { retry: false, refetchInterval: 60000 }
  );
  const pendingApprovalsCount = pendingApprovals.length || 0;

  // Check if user has any media source configured
  const { data: mediaSources } = useQuery(
    'mediaSources',
    () => apiService.music.getSources(),
    {
      retry: false,
      refetchOnWindowFocus: false,
    }
  );

  const hasMediaConfig = mediaSources?.sources && mediaSources.sources.length > 0;

  // Calculate total unread count (pending invitations + unread posts)
  const totalUnreadPosts = unreadCounts ? Object.values(unreadCounts).reduce((sum, count) => sum + count, 0) : 0;
  const totalTeamNotifications = (pendingInvitations?.length || 0) + totalUnreadPosts;

  const navItems = [
      { label: 'Home', path: '/home', icon: <Home /> },
      { label: 'Documents', path: '/documents', icon: <Description /> },
      { label: 'Code Spaces', path: '/code-spaces', icon: <Code /> },
      { label: 'Teams', path: '/teams', icon: <Group />, badge: totalTeamNotifications },
      ...(isAdmin || has('feature.games.view') ? [{ label: 'Games', path: '/games', icon: <SportsEsports /> }] : []),
      ...(isAdmin || has('feature.maps.view') ? [{ label: 'Map', path: '/map', icon: <Map /> }] : []),
      ...(hasMediaConfig ? [{ label: 'Media', path: '/media', icon: <MusicNote /> }] : []),
      { label: 'Agent Factory', path: '/agent-factory', icon: <Build /> },
      { label: 'Operations', path: '/agent-dashboard', icon: <Dashboard />, badge: pendingApprovalsCount },
  ];
  const navPathKey = navItems.map((i) => i.path).join('|');
  // eslint-disable-next-line react-hooks/exhaustive-deps -- navPathKey encodes which routes exist; re-run when that set changes
  useEffect(() => {
    const valid = new Set(navItems.map((i) => i.path));
    setPinnedPaths((prev) => {
      const next = prev.filter((p) => valid.has(p));
      if (next.length === prev.length && next.every((p, i) => p === prev[i])) return prev;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, [navPathKey]);

  const handlePinnedDragEnd = (result) => {
    if (!result.destination) return;
    if (result.source.droppableId !== result.destination.droppableId) return;
    const from = result.source.index;
    const to = result.destination.index;
    if (from === to) return;
    setPinnedPaths((prev) => {
      const validSet = new Set(navItems.map((i) => i.path));
      const ordered = prev.filter((p) => validSet.has(p));
      const next = [...ordered];
      const [removed] = next.splice(from, 1);
      next.splice(to, 0, removed);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const pinnedItems = pinnedPaths
    .filter((p) => navItems.some((i) => i.path === p))
    .map((p) => navItems.find((i) => i.path === p))
    .filter(Boolean);
  const unpinnedItems = navItems.filter((i) => !pinnedPaths.includes(i.path));
  const pagesNavBadgeCount = navItems.reduce((sum, item) => sum + (item.badge || 0), 0);

  const isActive = (path) => location.pathname === path;

  const handleUserMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleUserMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = async () => {
    handleUserMenuClose();
    await logout();
    navigate('/login');
  };

  const getUserDisplayName = () => {
    return user?.display_name || user?.username || 'User';
  };

  const getUserInitials = () => {
    const name = getUserDisplayName();
    return name.split(' ').map(word => word[0]).join('').toUpperCase().slice(0, 2);
  };

  return (
    <AppBar 
      ref={appBarRef}
      position="static" 
      elevation={1} 
      sx={{ 
        zIndex: 1201, 
        paddingTop: 'env(safe-area-inset-top)',
        backgroundColor: darkMode ? 'background.default' : 'primary.main',
        '& .MuiToolbar-root': {
          minHeight: 59, // Force standard height (reduced by 5px)
        }
      }}
    >
      <Toolbar sx={{ minHeight: 59, py: 0.5 }}>
        <Box
          onClick={() => navigate('/documents')}
          sx={{ mr: 2, display: 'flex', alignItems: 'center', cursor: 'pointer' }}
        >
          {!logoError ? (
            <Box
              component="img"
              src={darkMode ? '/images/bastion-dark.png' : '/images/bastion.png'}
              alt="Bastion"
              sx={{ height: 48 }}
              onError={() => setLogoError(true)}
            />
          ) : (
            <Typography variant="h6" component="div">Bastion</Typography>
          )}
        </Box>

        <Box sx={{ flexGrow: 1 }} />

        <Box sx={{ display: { xs: 'none', md: 'flex' } }}>
          {pinnedItems.map((item) => (
            <Button
              key={item.path}
              color="inherit"
              startIcon={
                item.badge > 0 ? (
                  <Badge badgeContent={item.badge} color="error" max={99}>
                    {item.icon}
                  </Badge>
                ) : (
                  item.icon
                )
              }
              onClick={() => navigate(item.path)}
              sx={{
                mx: 1,
                backgroundColor: isActive(item.path) ? 'rgba(255,255,255,0.1)' : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>

        {/* Mobile hamburger */}
        <Box sx={{ display: { xs: 'flex', md: 'none' }, alignItems: 'center' }}>
          <IconButton color="inherit" onClick={() => setMobileOpen(true)} aria-label="open navigation menu">
            <MenuIcon />
          </IconButton>
        </Box>

        {/* Right cluster: pages menu, notifications, messages, account — pronounced gap from pinned pages */}
        <Box sx={{ display: 'flex', alignItems: 'center', ml: { xs: 0, md: 4 }, gap: 0.5 }}>
        <Tooltip title="Pages">
          <IconButton
            color="inherit"
            size="small"
            onClick={(e) => setPagesAnchorEl(e.currentTarget)}
            aria-label="Open pages menu"
          >
            <Badge badgeContent={pagesNavBadgeCount} color="error" max={99} invisible={pagesNavBadgeCount === 0}>
              <Apps />
            </Badge>
          </IconButton>
        </Tooltip>
        <Popover
          open={Boolean(pagesAnchorEl)}
          anchorEl={pagesAnchorEl}
          onClose={() => setPagesAnchorEl(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          PaperProps={{
            sx: {
              minWidth: 280,
              maxWidth: 400,
              mt: 1,
              maxHeight: 'min(70vh, 480px)',
              overflow: 'auto',
            },
          }}
        >
          <Box sx={{ px: 1.5, pt: 1, pb: 0.5 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              Pinned
            </Typography>
          </Box>
          {pinnedItems.length === 0 ? (
            <Box sx={{ px: 1.5, pb: 1 }}>
              <Typography variant="body2" color="text.secondary">
                No pinned pages. Pin from More Pages below.
              </Typography>
            </Box>
          ) : (
            <DragDropContext onDragEnd={handlePinnedDragEnd}>
              <Droppable droppableId="pinned-nav-pages">
                {(dropProvided) => (
                  <List
                    ref={dropProvided.innerRef}
                    {...dropProvided.droppableProps}
                    dense
                    disablePadding
                    sx={{ pb: 0.5 }}
                  >
                    {pinnedItems.map((item, idx) => (
                      <Draggable key={item.path} draggableId={item.path} index={idx}>
                        {(dragProvided, snapshot) => (
                          <Box
                            ref={dragProvided.innerRef}
                            {...dragProvided.draggableProps}
                            sx={{
                              display: 'flex',
                              alignItems: 'stretch',
                              px: 0.5,
                              bgcolor: snapshot.isDragging ? 'action.hover' : 'transparent',
                              borderRadius: 1,
                            }}
                          >
                            <Tooltip title="Drag to reorder">
                              <IconButton
                                {...dragProvided.dragHandleProps}
                                size="small"
                                aria-label="Drag to reorder pinned page"
                                sx={{
                                  alignSelf: 'center',
                                  flexShrink: 0,
                                  cursor: 'grab',
                                  '&:active': { cursor: 'grabbing' },
                                }}
                              >
                                <DragIndicator fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            <ListItemButton
                              sx={{ flex: 1, borderRadius: 1, py: 0.5 }}
                              onClick={() => {
                                navigate(item.path);
                                setPagesAnchorEl(null);
                              }}
                            >
                              <ListItemIcon sx={{ minWidth: 36 }}>
                                {item.badge > 0 ? (
                                  <Badge badgeContent={item.badge} color="error" max={99}>
                                    {item.icon}
                                  </Badge>
                                ) : (
                                  item.icon
                                )}
                              </ListItemIcon>
                              <ListItemText primary={item.label} primaryTypographyProps={{ fontSize: '0.875rem' }} />
                            </ListItemButton>
                            <Box sx={{ display: 'flex', alignItems: 'center', pr: 0.5 }}>
                              <Tooltip title="Unpin from toolbar">
                                <IconButton
                                  size="small"
                                  aria-label="Unpin from toolbar"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    togglePin(item.path);
                                  }}
                                >
                                  <PushPin fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </Box>
                        )}
                      </Draggable>
                    ))}
                    {dropProvided.placeholder}
                  </List>
                )}
              </Droppable>
            </DragDropContext>
          )}
          <Divider />
          <Box sx={{ px: 1.5, pt: 1, pb: 0.5 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, textTransform: 'uppercase' }}>
              More Pages
            </Typography>
          </Box>
          <List dense disablePadding sx={{ pb: 1 }}>
            {unpinnedItems.map((item) => (
              <Box key={item.path} sx={{ display: 'flex', alignItems: 'center', px: 0.5 }}>
                <ListItemButton
                  sx={{ flex: 1, borderRadius: 1, py: 0.5 }}
                  onClick={() => {
                    navigate(item.path);
                    setPagesAnchorEl(null);
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 36 }}>
                    {item.badge > 0 ? (
                      <Badge badgeContent={item.badge} color="error" max={99}>
                        {item.icon}
                      </Badge>
                    ) : (
                      item.icon
                    )}
                  </ListItemIcon>
                  <ListItemText primary={item.label} primaryTypographyProps={{ fontSize: '0.875rem' }} />
                </ListItemButton>
                <Box sx={{ display: 'flex', alignItems: 'center', pr: 0.5 }}>
                  <Tooltip title="Pin to toolbar">
                    <IconButton
                      size="small"
                      aria-label="Pin to toolbar"
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePin(item.path);
                      }}
                    >
                      <PushPinOutlined fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
            ))}
          </List>
        </Popover>

        {/* Agent Notifications */}
        <Tooltip title="Agent Notifications">
          <IconButton
            ref={notificationBellRef}
            color="inherit"
            onClick={(e) => setNotificationAnchorEl(e.currentTarget)}
            size="small"
            sx={{ mr: 0 }}
          >
            <Badge badgeContent={notificationUnreadCount} color="error" max={99}>
              {notificationUnreadCount > 0 ? <Notifications /> : <NotificationsNone />}
            </Badge>
          </IconButton>
        </Tooltip>
        <NotificationDropdown
          anchorEl={notificationAnchorEl}
          open={Boolean(notificationAnchorEl)}
          onClose={() => setNotificationAnchorEl(null)}
        />

        {/* Messaging Toggle */}
        <Tooltip title="Messages">
          <IconButton
            color="inherit"
            onClick={toggleDrawer}
            size="small"
            sx={{ mr: 0 }}
          >
            <Badge badgeContent={totalUnreadCount} color="error" max={99}>
              {totalUnreadCount > 0 ? <Mail /> : <MailOutline />}
            </Badge>
          </IconButton>
        </Tooltip>

        {/* User Menu */}
        <Box sx={{ ml: 0.5 }}>
          {user?.role === 'admin' && (
            <Chip
              label="Admin"
              size="small"
              color="secondary"
              sx={{ mr: 2, color: 'white', backgroundColor: 'rgba(255,255,255,0.2)' }}
            />
          )}
          
          <IconButton
            onClick={handleUserMenuOpen}
            color="inherit"
            sx={{ p: 0 }}
          >
            <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)' }}>
              {getUserInitials()}
            </Avatar>
          </IconButton>
          
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleUserMenuClose}
            PaperProps={{
              sx: {
                minWidth: 200,
                '& .MuiListItemIcon-root': { minWidth: 36 },
                '& .MuiMenuItem-root': { py: 0.75 },
                '& .MuiDivider-root': { my: 0.75 },
              }
            }}
          >
            <MenuItem disabled sx={{ py: 0.75, opacity: 1 }}>
              <Box>
                <Typography variant="subtitle2" fontSize="0.8125rem">{getUserDisplayName()}</Typography>
                <Typography variant="body2" color="text.secondary" fontSize="0.75rem">
                  {user?.email}
                </Typography>
              </Box>
            </MenuItem>
            
            <Divider />

            {themePreference !== 'system' && (
              <MenuItem onClick={() => toggleDarkMode()}>
                <ListItemIcon sx={{ minWidth: 36 }}>
                  {darkMode ? <LightMode /> : <DarkMode />}
                </ListItemIcon>
                <ListItemText primary={darkMode ? 'Light mode' : 'Dark mode'} primaryTypographyProps={{ fontSize: '0.875rem' }} />
              </MenuItem>
            )}
            
            <MenuItem onClick={() => { handleUserMenuClose(); navigate('/settings'); }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Settings fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Settings" primaryTypographyProps={{ fontSize: '0.875rem' }} />
            </MenuItem>
            
            <MenuItem onClick={() => { handleUserMenuClose(); navigate('/control-panes'); }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Tune fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Control Panes" primaryTypographyProps={{ fontSize: '0.875rem' }} />
            </MenuItem>
            
            {user?.role === 'admin' && (
              <MenuItem onClick={() => { handleUserMenuClose(); navigate('/settings?tab=users'); }}>
                <ListItemIcon sx={{ minWidth: 36 }}>
                  <PersonAdd fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="User Management" primaryTypographyProps={{ fontSize: '0.875rem' }} />
              </MenuItem>
            )}
            
            <MenuItem onClick={() => { handleUserMenuClose(); setHelpOpen(true); }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <HelpOutline fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Help" primaryTypographyProps={{ fontSize: '0.875rem' }} />
            </MenuItem>
            
            <MenuItem onClick={() => { handleUserMenuClose(); setAboutOpen(true); }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Info fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="About" primaryTypographyProps={{ fontSize: '0.875rem' }} />
            </MenuItem>
            
            <Divider />
            
            <MenuItem onClick={handleLogout}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Logout fontSize="small" />
              </ListItemIcon>
              <ListItemText primary="Logout" primaryTypographyProps={{ fontSize: '0.875rem' }} />
            </MenuItem>
          </Menu>
          
          <HelpOverlay
            open={helpOpen}
            initialTopicId={helpInitialTopicId}
            onClose={() => {
              setHelpOpen(false);
              setHelpInitialTopicId(null);
            }}
          />
          <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} />
        </Box>
        </Box>
      </Toolbar>
      {/* Mobile Drawer */}
      <Drawer
        anchor="left"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        ModalProps={{ keepMounted: true }}
        PaperProps={{ sx: { width: '80vw', maxWidth: 360, paddingTop: 'env(safe-area-inset-top)', paddingBottom: 'env(safe-area-inset-bottom)' } }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 1 }}>Menu</Typography>
        </Box>
        <Divider />
        <List>
          {navItems.map((item) => (
            <ListItemButton key={item.path} onClick={() => { setMobileOpen(false); navigate(item.path); }} selected={isActive(item.path)}>
              <ListItemIcon>
                {item.badge > 0 ? (
                  <Badge badgeContent={item.badge} color="error" max={99}>
                    {item.icon}
                  </Badge>
                ) : (
                  item.icon
                )}
              </ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          ))}
        </List>
        <Divider />
        <List>
          <ListItemButton onClick={() => { setMobileOpen(false); setNotificationAnchorEl(notificationBellRef.current); }}>
            <ListItemIcon>
              <Badge badgeContent={notificationUnreadCount} color="error" max={99}>
                {notificationUnreadCount > 0 ? <Notifications /> : <NotificationsNone />}
              </Badge>
            </ListItemIcon>
            <ListItemText primary="Agent Notifications" />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); toggleDrawer(); }}>
            <ListItemIcon>
              <Badge badgeContent={totalUnreadCount} color="error" max={99}>
                {totalUnreadCount > 0 ? <Mail /> : <MailOutline />}
              </Badge>
            </ListItemIcon>
            <ListItemText primary="Messages" />
          </ListItemButton>
          {themePreference !== 'system' && (
            <ListItemButton onClick={() => { setMobileOpen(false); toggleDarkMode(); }}>
              <ListItemIcon>{darkMode ? <LightMode /> : <DarkMode />}</ListItemIcon>
              <ListItemText primary={darkMode ? 'Light Mode' : 'Dark Mode'} />
            </ListItemButton>
          )}
          <ListItemButton onClick={() => { setMobileOpen(false); navigate('/settings'); }}>
            <ListItemIcon><Settings /></ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); navigate('/control-panes'); }}>
            <ListItemIcon><Tune /></ListItemIcon>
            <ListItemText primary="Control Panes" />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); setHelpOpen(true); }}>
            <ListItemIcon><HelpOutline /></ListItemIcon>
            <ListItemText primary="Help" />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); setAboutOpen(true); }}>
            <ListItemIcon><Info /></ListItemIcon>
            <ListItemText primary="About" />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); handleLogout(); }}>
            <ListItemIcon><Logout /></ListItemIcon>
            <ListItemText primary="Logout" />
          </ListItemButton>
        </List>
      </Drawer>
    </AppBar>
  );
};

export default Navigation;
