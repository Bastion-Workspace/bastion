import React, { useState } from 'react';
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
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { useCapabilities } from '../contexts/CapabilitiesContext';
import { useMessaging } from '../contexts/MessagingContext';
import { useTeam } from '../contexts/TeamContext';
import apiService from '../services/apiService';

const Navigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuth();
  const { darkMode, toggleDarkMode } = useTheme();
  const { isAdmin, has } = useCapabilities();
  const { toggleDrawer, totalUnreadCount } = useMessaging();
  const { pendingInvitations, unreadCounts } = useTeam();
  const [anchorEl, setAnchorEl] = useState(null);
  const [logoError, setLogoError] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

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
      { label: 'Documents', path: '/documents', icon: <Description /> },
      { label: 'Teams', path: '/teams', icon: <Group />, badge: totalTeamNotifications },
      ...(isAdmin || has('feature.news.view') ? [{ label: 'News', path: '/news', icon: <Description /> }] : []),
      ...(hasMediaConfig ? [{ label: 'Media', path: '/media', icon: <MusicNote /> }] : []),
  ];

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
      position="static" 
      elevation={1} 
      sx={{ 
        zIndex: 1201, 
        paddingTop: 'env(safe-area-inset-top)',
        backgroundColor: darkMode ? '#121212' : 'primary.main',
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
          {navItems.map((item) => (
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

        {/* Theme Toggle */}
        <Tooltip title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
          <IconButton
            color="inherit"
            onClick={toggleDarkMode}
            sx={{ mr: 2 }}
          >
            {darkMode ? <LightMode /> : <DarkMode />}
          </IconButton>
        </Tooltip>

        {/* Messaging Toggle - BULLY! Roosevelt's Messaging Cavalry! */}
        <Tooltip title="Messages">
          <IconButton
            color="inherit"
            onClick={toggleDrawer}
            sx={{ mr: 2 }}
          >
            <Badge badgeContent={totalUnreadCount} color="error" max={99}>
              {totalUnreadCount > 0 ? <Mail /> : <MailOutline />}
            </Badge>
          </IconButton>
        </Tooltip>

        {/* User Menu */}
        <Box sx={{ ml: 2 }}>
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
              sx: { minWidth: 200 }
            }}
          >
            <MenuItem disabled>
              <Box>
                <Typography variant="subtitle2">{getUserDisplayName()}</Typography>
                <Typography variant="body2" color="text.secondary">
                  {user?.email}
                </Typography>
              </Box>
            </MenuItem>
            
            <Divider />
            
            <MenuItem onClick={() => { handleUserMenuClose(); navigate('/settings'); }}>
              <Settings sx={{ mr: 1 }} />
              Settings
            </MenuItem>
            
            {user?.role === 'admin' && (
              <MenuItem onClick={() => { handleUserMenuClose(); navigate('/settings?tab=users'); }}>
                <PersonAdd sx={{ mr: 1 }} />
                User Management
              </MenuItem>
            )}
            
            <Divider />
            
            <MenuItem onClick={handleLogout}>
              <Logout sx={{ mr: 1 }} />
              Logout
            </MenuItem>
          </Menu>
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
          <ListItemButton onClick={() => { setMobileOpen(false); toggleDrawer(); }}>
            <ListItemIcon>
              <Badge badgeContent={totalUnreadCount} color="error" max={99}>
                {totalUnreadCount > 0 ? <Mail /> : <MailOutline />}
              </Badge>
            </ListItemIcon>
            <ListItemText primary="Messages" />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); toggleDarkMode(); }}>
            <ListItemIcon>{darkMode ? <LightMode /> : <DarkMode />}</ListItemIcon>
            <ListItemText primary={darkMode ? 'Light Mode' : 'Dark Mode'} />
          </ListItemButton>
          <ListItemButton onClick={() => { setMobileOpen(false); navigate('/settings'); }}>
            <ListItemIcon><Settings /></ListItemIcon>
            <ListItemText primary="Settings" />
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
