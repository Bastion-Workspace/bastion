import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Chip,
  IconButton,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  PersonAdd,
  Edit,
  Delete,
  Security,
  VpnKey,
  Visibility,
  VisibilityOff,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/apiService';
import { useAuth } from '../contexts/AuthContext';

const UserManagement = () => {
  const { user: currentUser } = useAuth();
  const queryClient = useQueryClient();
  
  // State for dialogs
  const [createUserDialog, setCreateUserDialog] = useState(false);
  const [editUserDialog, setEditUserDialog] = useState(false);
  const [changePasswordDialog, setChangePasswordDialog] = useState(false);
  const [selectedUser, setSelectedUser] = useState(null);
  const [capabilities, setCapabilities] = useState({});
  
  // State for forms
  const [newUser, setNewUser] = useState({
    username: '',
    email: '',
    password: '',
    display_name: '',
    role: 'user'
  });
  const [editUser, setEditUser] = useState({});
  const [passwordChange, setPasswordChange] = useState({
    current_password: '',
    new_password: '',
    confirm_password: ''
  });
  const [showPasswords, setShowPasswords] = useState({
    password: false,
    current: false,
    new: false,
    confirm: false
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Fetch users
  const { data: usersData } = useQuery(
    'users',
    () => apiService.getUsers(),
    {
      enabled: currentUser?.role === 'admin'
    }
  );

  // Create user mutation
  const createUserMutation = useMutation(
    (userData) => apiService.createUser(userData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('users');
        setCreateUserDialog(false);
        setNewUser({ username: '', email: '', password: '', display_name: '', role: 'user' });
        setSuccess('User created successfully');
        setTimeout(() => setSuccess(''), 3000);
      },
      onError: (error) => {
        setError(error.response?.data?.detail || 'Failed to create user');
      }
    }
  );

  // Update user mutation
  const updateUserMutation = useMutation(
    ({ userId, userData }) => apiService.updateUser(userId, userData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('users');
        setEditUserDialog(false);
        setSuccess('User updated successfully');
        setTimeout(() => setSuccess(''), 3000);
      },
      onError: (error) => {
        setError(error.response?.data?.detail || 'Failed to update user');
      }
    }
  );

  // Change password mutation
  const changePasswordMutation = useMutation(
    ({ userId, passwordData }) => apiService.adminChangePassword(userId, passwordData),
    {
      onSuccess: () => {
        setChangePasswordDialog(false);
        setPasswordChange({ current_password: '', new_password: '', confirm_password: '' });
        setSuccess('Password changed successfully');
        setTimeout(() => setSuccess(''), 3000);
      },
      onError: (error) => {
        setError(error.response?.data?.detail || 'Failed to change password');
      }
    }
  );

  // Delete user mutation
  const deleteUserMutation = useMutation(
    (userId) => apiService.deleteUser(userId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('users');
        setSuccess('User deleted successfully');
        setTimeout(() => setSuccess(''), 3000);
      },
      onError: (error) => {
        setError(error.response?.data?.detail || 'Failed to delete user');
      }
    }
  );

  const handleCreateUser = () => {
    setError('');
    if (newUser.password.length < 8) {
      setError('Password must be at least 8 characters long');
      return;
    }
    createUserMutation.mutate(newUser);
  };

  const handleEditUser = (user) => {
    setSelectedUser(user);
    setEditUser({
      email: user.email,
      display_name: user.display_name,
      role: user.role,
      is_active: user.is_active
    });
    setEditUserDialog(true);
  };

  // Load capabilities for selected user when edit dialog opens
  React.useEffect(() => {
    const loadCaps = async () => {
      try {
        if (editUserDialog && selectedUser) {
          const res = await apiService.get(`/api/admin/users/${selectedUser.user_id}/capabilities`);
          setCapabilities(res?.capabilities || {});
        }
      } catch {}
    };
    loadCaps();
  }, [editUserDialog, selectedUser]);

  const handleUpdateUser = () => {
    setError('');
    updateUserMutation.mutate({
      userId: selectedUser.user_id,
      userData: editUser
    });
  };

  const handleChangePassword = (user) => {
    setSelectedUser(user);
    setPasswordChange({ current_password: '', new_password: '', confirm_password: '' });
    setChangePasswordDialog(true);
  };

  const handlePasswordSubmit = () => {
    setError('');
    
    if (passwordChange.new_password !== passwordChange.confirm_password) {
      setError('New passwords do not match');
      return;
    }
    
    if (passwordChange.new_password.length < 8) {
      setError('Password must be at least 8 characters long');
      return;
    }

    // For admin changing other user's password, current_password is not required
    const passwordData = currentUser?.user_id === selectedUser?.user_id
      ? passwordChange
      : { current_password: '', new_password: passwordChange.new_password };

    changePasswordMutation.mutate({
      userId: selectedUser.user_id,
      passwordData
    });
  };

  const handleDeleteUser = (user) => {
    if (window.confirm(`Are you sure you want to delete user "${user.username}"?`)) {
      deleteUserMutation.mutate(user.user_id);
    }
  };

  const togglePasswordVisibility = (field) => {
    setShowPasswords(prev => ({ ...prev, [field]: !prev[field] }));
  };

  if (currentUser?.role !== 'admin') {
    return (
      <Card>
        <CardContent>
          <Alert severity="warning">
            You don't have permission to access user management.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h5" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Security color="primary" />
          User Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<PersonAdd />}
          onClick={() => setCreateUserDialog(true)}
        >
          Add User
        </Button>
      </Box>

      {/* Alerts */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      {success && (
        <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess('')}>
          {success}
        </Alert>
      )}

      {/* Users Table */}
      <Card>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Username</TableCell>
                <TableCell>Email</TableCell>
                <TableCell>Display Name</TableCell>
                <TableCell>Role</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Last Login</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {usersData?.users?.map((user) => (
                <TableRow key={user.user_id}>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {user.username}
                      {user.user_id === currentUser.user_id && (
                        <Chip label="You" size="small" color="primary" />
                      )}
                    </Box>
                  </TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>{user.display_name}</TableCell>
                  <TableCell>
                    <Chip
                      label={user.role}
                      color={user.role === 'admin' ? 'secondary' : 'default'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={user.is_active ? 'Active' : 'Inactive'}
                      color={user.is_active ? 'success' : 'error'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    {new Date(user.created_at).toLocaleDateString()}
                  </TableCell>
                  <TableCell>
                    {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
                  </TableCell>
                  <TableCell align="center">
                    <IconButton
                      size="small"
                      onClick={() => handleEditUser(user)}
                      color="primary"
                    >
                      <Edit />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => handleChangePassword(user)}
                      color="secondary"
                    >
                      <VpnKey />
                    </IconButton>
                    {user.user_id !== currentUser.user_id && (
                      <IconButton
                        size="small"
                        onClick={() => handleDeleteUser(user)}
                        color="error"
                      >
                        <Delete />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* Create User Dialog */}
      <Dialog open={createUserDialog} onClose={() => setCreateUserDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New User</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Username"
            fullWidth
            variant="outlined"
            value={newUser.username}
            onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Email"
            type="email"
            fullWidth
            variant="outlined"
            value={newUser.email}
            onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Display Name"
            fullWidth
            variant="outlined"
            value={newUser.display_name}
            onChange={(e) => setNewUser({ ...newUser, display_name: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Password"
            type={showPasswords.password ? 'text' : 'password'}
            fullWidth
            variant="outlined"
            value={newUser.password}
            onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
            InputProps={{
              endAdornment: (
                <IconButton onClick={() => togglePasswordVisibility('password')}>
                  {showPasswords.password ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              )
            }}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth variant="outlined">
            <InputLabel>Role</InputLabel>
            <Select
              value={newUser.role}
              onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
              label="Role"
            >
              <MenuItem value="user">User</MenuItem>
              <MenuItem value="admin">Admin</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateUserDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateUser} variant="contained">Create User</Button>
        </DialogActions>
      </Dialog>

      {/* Edit User Dialog */}
      <Dialog open={editUserDialog} onClose={() => setEditUserDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit User: {selectedUser?.username}</DialogTitle>
        <DialogContent>
          <TextField
            margin="dense"
            label="Email"
            type="email"
            fullWidth
            variant="outlined"
            value={editUser.email || ''}
            onChange={(e) => setEditUser({ ...editUser, email: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Display Name"
            fullWidth
            variant="outlined"
            value={editUser.display_name || ''}
            onChange={(e) => setEditUser({ ...editUser, display_name: e.target.value })}
            sx={{ mb: 2 }}
          />
          <FormControl fullWidth variant="outlined" sx={{ mb: 2 }}>
            <InputLabel>Role</InputLabel>
            <Select
              value={editUser.role || ''}
              onChange={(e) => setEditUser({ ...editUser, role: e.target.value })}
              label="Role"
            >
              <MenuItem value="user">User</MenuItem>
              <MenuItem value="admin">Admin</MenuItem>
            </Select>
          </FormControl>
          <FormControlLabel
            control={
              <Switch
                checked={editUser.is_active || false}
                onChange={(e) => setEditUser({ ...editUser, is_active: e.target.checked })}
              />
            }
            label="Account Active"
          />

          {selectedUser?.role !== 'admin' && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>Features</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 1 }}>
                <FormControlLabel
                  control={<Switch checked={!!capabilities['feature.news.view']} onChange={(e) => setCapabilities(prev => ({ ...prev, 'feature.news.view': e.target.checked }))} />}
                  label="News: View"
                />
                <FormControlLabel
                  control={<Switch checked={!!capabilities['feature.news.agent']} onChange={(e) => setCapabilities(prev => ({ ...prev, 'feature.news.agent': e.target.checked }))} />}
                  label="News: Agent Access"
                />
                <FormControlLabel
                  control={<Switch checked={!!capabilities['feature.news.notifications']} onChange={(e) => setCapabilities(prev => ({ ...prev, 'feature.news.notifications': e.target.checked }))} />}
                  label="News: Notifications"
                />
                <FormControlLabel
                  control={<Switch checked={!!capabilities['feature.maps.view']} onChange={(e) => setCapabilities(prev => ({ ...prev, 'feature.maps.view': e.target.checked }))} />}
                  label="Maps: View"
                />
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditUserDialog(false)}>Cancel</Button>
          <Button onClick={async () => {
            handleUpdateUser();
            try {
              if (selectedUser?.role !== 'admin') {
                await apiService.post(`/api/admin/users/${selectedUser.user_id}/capabilities`, capabilities);
              }
            } catch {}
          }} variant="contained">Update User</Button>
        </DialogActions>
      </Dialog>

      {/* Change Password Dialog */}
      <Dialog open={changePasswordDialog} onClose={() => setChangePasswordDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Change Password: {selectedUser?.username}</DialogTitle>
        <DialogContent>
          {currentUser?.user_id === selectedUser?.user_id && (
            <TextField
              margin="dense"
              label="Current Password"
              type={showPasswords.current ? 'text' : 'password'}
              fullWidth
              variant="outlined"
              value={passwordChange.current_password}
              onChange={(e) => setPasswordChange({ ...passwordChange, current_password: e.target.value })}
              InputProps={{
                endAdornment: (
                  <IconButton onClick={() => togglePasswordVisibility('current')}>
                    {showPasswords.current ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                )
              }}
              sx={{ mb: 2 }}
            />
          )}
          <TextField
            margin="dense"
            label="New Password"
            type={showPasswords.new ? 'text' : 'password'}
            fullWidth
            variant="outlined"
            value={passwordChange.new_password}
            onChange={(e) => setPasswordChange({ ...passwordChange, new_password: e.target.value })}
            InputProps={{
              endAdornment: (
                <IconButton onClick={() => togglePasswordVisibility('new')}>
                  {showPasswords.new ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              )
            }}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Confirm New Password"
            type={showPasswords.confirm ? 'text' : 'password'}
            fullWidth
            variant="outlined"
            value={passwordChange.confirm_password}
            onChange={(e) => setPasswordChange({ ...passwordChange, confirm_password: e.target.value })}
            InputProps={{
              endAdornment: (
                <IconButton onClick={() => togglePasswordVisibility('confirm')}>
                  {showPasswords.confirm ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              )
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setChangePasswordDialog(false)}>Cancel</Button>
          <Button onClick={handlePasswordSubmit} variant="contained">Change Password</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default UserManagement;
