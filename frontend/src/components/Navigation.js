import React, { useState, useRef, useLayoutEffect, useCallback } from 'react';
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
  Notifications,
  NotificationsNone,
  PushPin,
  SportsEsports,
  Info,
  Tune,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { useCapabilities } from '../contexts/CapabilitiesContext';
import { useMessaging } from '../contexts/MessagingContext';
import { useNotifications } from '../contexts/NotificationContext';
import { useTeam } from '../contexts/TeamContext';
import apiService from '../services/apiService';
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
  const [aboutOpen, setAboutOpen] = useState(false);
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
      { label: 'Teams', path: '/teams', icon: <Group />, badge: totalTeamNotifications },
      ...(isAdmin || has('feature.games.view') ? [{ label: 'Games', path: '/games', icon: <SportsEsports /> }] : []),
      ...(isAdmin || has('feature.news.view') ? [{ label: 'News', path: '/news', icon: <Description /> }] : []),
      ...(isAdmin || has('feature.maps.view') ? [{ label: 'Map', path: '/map', icon: <Map /> }] : []),
      ...(hasMediaConfig ? [{ label: 'Media', path: '/media', icon: <MusicNote /> }] : []),
      { label: 'Agent Factory', path: '/agent-factory', icon: <Build /> },
      { label: 'Operations', path: '/agent-dashboard', icon: <Dashboard />, badge: pendingApprovalsCount },
  ];
  const pinnedItems = navItems.filter(i => pinnedPaths.includes(i.path));
  const unpinnedItems = navItems.filter(i => !pinnedPaths.includes(i.path));
  const unpinnedBadgeCount = unpinnedItems.reduce((sum, item) => sum + (item.badge || 0), 0);

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

        {/* Right cluster: notifications, messages, account — pronounced gap from pinned pages */}
        <Box sx={{ display: 'flex', alignItems: 'center', ml: { xs: 0, md: 4 }, gap: 0.5 }}>
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
            <Badge badgeContent={unpinnedBadgeCount} color="error" max={99}>
              <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)' }}>
                {getUserInitials()}
              </Avatar>
            </Badge>
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
            
            {unpinnedItems.length > 0 && unpinnedItems.map((item) => (
              <MenuItem
                key={item.path}
                onClick={() => { handleUserMenuClose(); navigate(item.path); }}
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
                <IconButton
                  size="small"
                  onClick={(e) => { e.stopPropagation(); togglePin(item.path); }}
                  sx={{ ml: 0.5 }}
                  aria-label="Pin to toolbar"
                >
                  <PushPin fontSize="small" />
                </IconButton>
              </MenuItem>
            ))}
            
            {unpinnedItems.length > 0 && <Divider />}
            
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
          
          <HelpOverlay open={helpOpen} onClose={() => setHelpOpen(false)} />
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
