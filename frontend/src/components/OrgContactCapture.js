/**
 * Org Contact Capture Component
 * Structured form for capturing contact information with progressive disclosure
 * 
 * **BULLY!** Four-tier professional contact management!
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  IconButton,
  Tooltip,
  Collapse,
  Chip,
  Stack,
  Autocomplete,
  CircularProgress
} from '@mui/material';
import {
  Close,
  Send,
  ExpandMore,
  ExpandLess,
  Add,
  RemoveCircleOutline,
  Person,
  Cake,
  Favorite
} from '@mui/icons-material';
import apiService from '../services/apiService';

const OrgContactCapture = ({ open, onClose }) => {
  // Tier 1 - Always Visible
  const [firstName, setFirstName] = useState('');
  const [middleName, setMiddleName] = useState('');
  const [lastName, setLastName] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [email1, setEmail1] = useState('');
  const [email1Type, setEmail1Type] = useState('Work');
  const [email2, setEmail2] = useState('');
  const [email2Type, setEmail2Type] = useState('Home');
  const [phone1, setPhone1] = useState('');
  const [phone1Type, setPhone1Type] = useState('Mobile');
  const [phone2, setPhone2] = useState('');
  const [phone2Type, setPhone2Type] = useState('Work');
  const [company, setCompany] = useState('');
  const [title, setTitle] = useState('');

  // Tier 2 - Advanced Details
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [address1, setAddress1] = useState('');
  const [address1Type, setAddress1Type] = useState('Home');
  const [address2, setAddress2] = useState('');
  const [address2Type, setAddress2Type] = useState('Work');
  const [website1, setWebsite1] = useState('');
  const [website1Type, setWebsite1Type] = useState('Personal');
  const [website2, setWebsite2] = useState('');
  const [website2Type, setWebsite2Type] = useState('Business');
  const [social1, setSocial1] = useState('');
  const [social1Type, setSocial1Type] = useState('LinkedIn');
  const [social2, setSocial2] = useState('');
  const [social2Type, setSocial2Type] = useState('Twitter');
  const [social3, setSocial3] = useState('');
  const [social3Type, setSocial3Type] = useState('Facebook');
  const [notes, setNotes] = useState('');

  // Tier 3 - Personal & Family
  const [showPersonal, setShowPersonal] = useState(false);
  const [relationship, setRelationship] = useState('');
  const [spouse, setSpouse] = useState(null);
  const [birthday, setBirthday] = useState('');
  const [anniversary, setAnniversary] = useState('');
  const [children, setChildren] = useState([]);

  // Contact search/autocomplete
  const [contactOptions, setContactOptions] = useState([]);
  const [loadingContacts, setLoadingContacts] = useState(false);

  // Form state
  const [capturing, setCapturing] = useState(false);
  const [error, setError] = useState(null);

  const firstNameRef = useRef(null);

  // Auto-focus first name field when dialog opens
  useEffect(() => {
    if (open && firstNameRef.current) {
      setTimeout(() => firstNameRef.current?.focus(), 100);
    }
  }, [open]);

  // Helper to generate display name
  const generateDisplayName = (first, middle, last) => {
    const parts = [first.trim(), middle.trim(), last.trim()].filter(Boolean);
    return parts.join(' ');
  };

  // Auto-update display name when leaving name fields
  const handleNameBlur = () => {
    const generated = generateDisplayName(firstName, middleName, lastName);
    // Only auto-update if display name is empty or matches the generated value
    // (meaning user hasn't manually edited it)
    if (!displayName || displayName === generated) {
      setDisplayName(generated);
    }
  };

  // Load existing contacts for autocomplete
  useEffect(() => {
    if (open) {
      loadContacts();
    }
  }, [open]);

  const loadContacts = async () => {
    try {
      setLoadingContacts(true);
      const response = await apiService.get('/api/org/contacts');
      
      if (response.success) {
        // Format contacts for autocomplete
        const options = response.results.map(contact => ({
          label: contact.heading,
          value: contact.heading,
          filename: contact.filename
        }));
        setContactOptions(options);
      }
    } catch (err) {
      console.error('❌ Failed to load contacts:', err);
    } finally {
      setLoadingContacts(false);
    }
  };

  // Reset form when dialog closes
  useEffect(() => {
    if (!open) {
      setTimeout(() => {
        // Tier 1
        setFirstName('');
        setMiddleName('');
        setLastName('');
        setDisplayName('');
        setEmail1('');
        setEmail1Type('Work');
        setEmail2('');
        setEmail2Type('Home');
        setPhone1('');
        setPhone1Type('Mobile');
        setPhone2('');
        setPhone2Type('Work');
        setCompany('');
        setTitle('');
        
        // Tier 2
        setShowAdvanced(false);
        setAddress1('');
        setAddress1Type('Home');
        setAddress2('');
        setAddress2Type('Work');
        setWebsite1('');
        setWebsite1Type('Personal');
        setWebsite2('');
        setWebsite2Type('Business');
        setSocial1('');
        setSocial1Type('LinkedIn');
        setSocial2('');
        setSocial2Type('Twitter');
        setSocial3('');
        setSocial3Type('Facebook');
        setNotes('');
        
        // Tier 3
        setShowPersonal(false);
        setRelationship('');
        setSpouse(null);
        setBirthday('');
        setAnniversary('');
        setChildren([]);
        
        setError(null);
      }, 300);
    }
  }, [open]);

  const handleAddChild = () => {
    setChildren([...children, { id: Date.now(), value: null }]);
  };

  const handleRemoveChild = (id) => {
    setChildren(children.filter(child => child.id !== id));
  };

  const handleUpdateChild = (id, value) => {
    setChildren(children.map(child => 
      child.id === id ? { ...child, value } : child
    ));
  };

  const handleCapture = async () => {
    // Validate required field
    if (!firstName.trim()) {
      setError('First name is required');
      return;
    }

    // Use display name if set, otherwise generate it
    const finalDisplayName = displayName.trim() || generateDisplayName(firstName, middleName, lastName);

    try {
      setCapturing(true);
      setError(null);

      // Build structured message with all contact details
      const contactParts = [`Add contact ${finalDisplayName}`];
      
      // Include name components as properties
      if (firstName) contactParts.push(`first name ${firstName}`);
      if (middleName) contactParts.push(`middle name ${middleName}`);
      if (lastName) contactParts.push(`last name ${lastName}`);
      
      if (email1) contactParts.push(`email ${email1Type.toLowerCase()} ${email1}`);
      if (email2) contactParts.push(`email ${email2Type.toLowerCase()} ${email2}`);
      if (phone1) contactParts.push(`phone ${phone1Type.toLowerCase()} ${phone1}`);
      if (phone2) contactParts.push(`phone ${phone2Type.toLowerCase()} ${phone2}`);
      if (company) contactParts.push(`company ${company}`);
      if (title) contactParts.push(`title ${title}`);
      if (address1) contactParts.push(`address ${address1Type.toLowerCase()} ${address1}`);
      if (address2) contactParts.push(`address ${address2Type.toLowerCase()} ${address2}`);
      if (website1) contactParts.push(`website ${website1Type.toLowerCase()} ${website1}`);
      if (website2) contactParts.push(`website ${website2Type.toLowerCase()} ${website2}`);
      if (social1) contactParts.push(`social ${social1Type.toLowerCase()} ${social1}`);
      if (social2) contactParts.push(`social ${social2Type.toLowerCase()} ${social2}`);
      if (social3) contactParts.push(`social ${social3Type.toLowerCase()} ${social3}`);
      if (birthday) contactParts.push(`birthday ${birthday}`);
      if (anniversary) contactParts.push(`anniversary ${anniversary}`);
      if (relationship) contactParts.push(`relationship ${relationship}`);
      if (spouse) contactParts.push(`spouse [[Contact: ${spouse.label}]]`);
      
      children.forEach((child, idx) => {
        if (child.value) {
          contactParts.push(`child ${idx + 1} [[Contact: ${child.value.label}]]`);
        }
      });
      
      if (notes) contactParts.push(`notes: ${notes}`);

      const message = contactParts.join(', ');

      // Send to chat API which will route to org inbox agent
      const response = await apiService.post('/api/chat/send', {
        message: message,
        conversation_id: null
      });

      if (response.error) {
        throw new Error(response.error);
      }

      // Success - close dialog
      onClose({ success: true, contact: finalDisplayName });
      
    } catch (err) {
      console.error('❌ Contact capture failed:', err);
      setError(err.message || 'Failed to capture contact');
    } finally {
      setCapturing(false);
    }
  };

  return (
    <Dialog 
      open={open} 
      onClose={() => onClose({ success: false })}
      maxWidth="md"
      fullWidth
    >
      <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Person />
          <Typography variant="h6">Add Contact</Typography>
        </Box>
        <IconButton size="small" onClick={() => onClose({ success: false })}>
          <Close />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers>
        {error && (
          <Box sx={{ mb: 2, p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
            <Typography color="error.contrastText">{error}</Typography>
          </Box>
        )}

        {/* TIER 1 - Always Visible */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 600 }}>
            Contact Information
          </Typography>

          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              inputRef={firstNameRef}
              label="First Name"
              value={firstName}
              onChange={(e) => setFirstName(e.target.value)}
              onBlur={handleNameBlur}
              fullWidth
              required
              placeholder="John"
            />
            <TextField
              label="Last Name"
              value={lastName}
              onChange={(e) => setLastName(e.target.value)}
              onBlur={handleNameBlur}
              fullWidth
              placeholder="Smith"
            />
          </Box>

          {/* Middle Name - Progressive */}
          {firstName && (
            <TextField
              label="Middle Name"
              value={middleName}
              onChange={(e) => setMiddleName(e.target.value)}
              fullWidth
              sx={{ mb: 2 }}
              placeholder="Anderson (optional)"
            />
          )}

          <TextField
            label="Display As"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            fullWidth
            sx={{ mb: 2 }}
            placeholder="Auto-generated from name fields"
            helperText="How this contact appears in lists (auto-generated, but editable)"
          />

          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              label="Email"
              value={email1}
              onChange={(e) => setEmail1(e.target.value)}
              fullWidth
              type="email"
              placeholder="john@example.com"
            />
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel>Type</InputLabel>
              <Select value={email1Type} onChange={(e) => setEmail1Type(e.target.value)} label="Type">
                <MenuItem value="Work">Work</MenuItem>
                <MenuItem value="Home">Home</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Email 2 - Progressive */}
          {email1 && (
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <TextField
                label="Email 2"
                value={email2}
                onChange={(e) => setEmail2(e.target.value)}
                fullWidth
                type="email"
                placeholder="john.personal@example.com"
              />
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel>Type</InputLabel>
                <Select value={email2Type} onChange={(e) => setEmail2Type(e.target.value)} label="Type">
                  <MenuItem value="Work">Work</MenuItem>
                  <MenuItem value="Home">Home</MenuItem>
                </Select>
              </FormControl>
            </Box>
          )}

          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              label="Phone"
              value={phone1}
              onChange={(e) => setPhone1(e.target.value)}
              fullWidth
              placeholder="555-1234"
            />
            <FormControl sx={{ minWidth: 120 }}>
              <InputLabel>Type</InputLabel>
              <Select value={phone1Type} onChange={(e) => setPhone1Type(e.target.value)} label="Type">
                <MenuItem value="Mobile">Mobile</MenuItem>
                <MenuItem value="Work">Work</MenuItem>
                <MenuItem value="Home">Home</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Phone 2 - Progressive */}
          {phone1 && (
            <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
              <TextField
                label="Phone 2"
                value={phone2}
                onChange={(e) => setPhone2(e.target.value)}
                fullWidth
                placeholder="555-5678"
              />
              <FormControl sx={{ minWidth: 120 }}>
                <InputLabel>Type</InputLabel>
                <Select value={phone2Type} onChange={(e) => setPhone2Type(e.target.value)} label="Type">
                  <MenuItem value="Mobile">Mobile</MenuItem>
                  <MenuItem value="Work">Work</MenuItem>
                  <MenuItem value="Home">Home</MenuItem>
                </Select>
              </FormControl>
            </Box>
          )}

          <TextField
            label="Company"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            fullWidth
            sx={{ mb: 2 }}
            placeholder="Acme Corporation"
          />

          <TextField
            label="Title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            fullWidth
            placeholder="Senior Engineer"
          />
        </Box>

        {/* TIER 2 - Advanced Details */}
        <Box sx={{ mb: 2 }}>
          <Button
            onClick={() => setShowAdvanced(!showAdvanced)}
            endIcon={showAdvanced ? <ExpandLess /> : <ExpandMore />}
            sx={{ mb: 1 }}
          >
            Advanced Details
          </Button>
          
          <Collapse in={showAdvanced}>
            <Box sx={{ pl: 2, borderLeft: '3px solid', borderColor: 'primary.main' }}>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  label="Address"
                  value={address1}
                  onChange={(e) => setAddress1(e.target.value)}
                  fullWidth
                  multiline
                  rows={2}
                  placeholder="123 Main St, City, State 12345"
                />
                <FormControl sx={{ minWidth: 120 }}>
                  <InputLabel>Type</InputLabel>
                  <Select value={address1Type} onChange={(e) => setAddress1Type(e.target.value)} label="Type">
                    <MenuItem value="Home">Home</MenuItem>
                    <MenuItem value="Work">Work</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              {address1 && (
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <TextField
                    label="Address 2"
                    value={address2}
                    onChange={(e) => setAddress2(e.target.value)}
                    fullWidth
                    multiline
                    rows={2}
                    placeholder="456 Business Ave, City, State 12345"
                  />
                  <FormControl sx={{ minWidth: 120 }}>
                    <InputLabel>Type</InputLabel>
                    <Select value={address2Type} onChange={(e) => setAddress2Type(e.target.value)} label="Type">
                      <MenuItem value="Home">Home</MenuItem>
                      <MenuItem value="Work">Work</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              )}

              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  label="Website"
                  value={website1}
                  onChange={(e) => setWebsite1(e.target.value)}
                  fullWidth
                  placeholder="https://example.com"
                />
                <FormControl sx={{ minWidth: 120 }}>
                  <InputLabel>Type</InputLabel>
                  <Select value={website1Type} onChange={(e) => setWebsite1Type(e.target.value)} label="Type">
                    <MenuItem value="Personal">Personal</MenuItem>
                    <MenuItem value="Business">Business</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              {website1 && (
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <TextField
                    label="Website 2"
                    value={website2}
                    onChange={(e) => setWebsite2(e.target.value)}
                    fullWidth
                    placeholder="https://company.com"
                  />
                  <FormControl sx={{ minWidth: 120 }}>
                    <InputLabel>Type</InputLabel>
                    <Select value={website2Type} onChange={(e) => setWebsite2Type(e.target.value)} label="Type">
                      <MenuItem value="Personal">Personal</MenuItem>
                      <MenuItem value="Business">Business</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              )}

              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  label="Social Media"
                  value={social1}
                  onChange={(e) => setSocial1(e.target.value)}
                  fullWidth
                  placeholder="username or URL"
                />
                <FormControl sx={{ minWidth: 140 }}>
                  <InputLabel>Platform</InputLabel>
                  <Select value={social1Type} onChange={(e) => setSocial1Type(e.target.value)} label="Platform">
                    <MenuItem value="LinkedIn">LinkedIn</MenuItem>
                    <MenuItem value="Twitter">Twitter</MenuItem>
                    <MenuItem value="Facebook">Facebook</MenuItem>
                    <MenuItem value="Instagram">Instagram</MenuItem>
                    <MenuItem value="GitHub">GitHub</MenuItem>
                    <MenuItem value="Other">Other</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              {social1 && (
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <TextField
                    label="Social Media 2"
                    value={social2}
                    onChange={(e) => setSocial2(e.target.value)}
                    fullWidth
                    placeholder="username or URL"
                  />
                  <FormControl sx={{ minWidth: 140 }}>
                    <InputLabel>Platform</InputLabel>
                    <Select value={social2Type} onChange={(e) => setSocial2Type(e.target.value)} label="Platform">
                      <MenuItem value="LinkedIn">LinkedIn</MenuItem>
                      <MenuItem value="Twitter">Twitter</MenuItem>
                      <MenuItem value="Facebook">Facebook</MenuItem>
                      <MenuItem value="Instagram">Instagram</MenuItem>
                      <MenuItem value="GitHub">GitHub</MenuItem>
                      <MenuItem value="Other">Other</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              )}

              {social2 && (
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <TextField
                    label="Social Media 3"
                    value={social3}
                    onChange={(e) => setSocial3(e.target.value)}
                    fullWidth
                    placeholder="username or URL"
                  />
                  <FormControl sx={{ minWidth: 140 }}>
                    <InputLabel>Platform</InputLabel>
                    <Select value={social3Type} onChange={(e) => setSocial3Type(e.target.value)} label="Platform">
                      <MenuItem value="LinkedIn">LinkedIn</MenuItem>
                      <MenuItem value="Twitter">Twitter</MenuItem>
                      <MenuItem value="Facebook">Facebook</MenuItem>
                      <MenuItem value="Instagram">Instagram</MenuItem>
                      <MenuItem value="GitHub">GitHub</MenuItem>
                      <MenuItem value="Other">Other</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              )}

              <TextField
                label="Notes"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                fullWidth
                multiline
                rows={3}
                placeholder="Additional notes about this contact..."
              />
            </Box>
          </Collapse>
        </Box>

        {/* TIER 3 - Personal & Family */}
        <Box>
          <Button
            onClick={() => setShowPersonal(!showPersonal)}
            endIcon={showPersonal ? <ExpandLess /> : <ExpandMore />}
            startIcon={<Favorite />}
            sx={{ mb: 1 }}
          >
            Personal & Family
          </Button>
          
          <Collapse in={showPersonal}>
            <Box sx={{ pl: 2, borderLeft: '3px solid', borderColor: 'secondary.main' }}>
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Relationship</InputLabel>
                <Select value={relationship} onChange={(e) => setRelationship(e.target.value)} label="Relationship">
                  <MenuItem value="">None</MenuItem>
                  <MenuItem value="Friend">Friend</MenuItem>
                  <MenuItem value="Colleague">Colleague</MenuItem>
                  <MenuItem value="Client">Client</MenuItem>
                  <MenuItem value="Vendor">Vendor</MenuItem>
                  <MenuItem value="Family">Family</MenuItem>
                  <MenuItem value="Mentor">Mentor</MenuItem>
                  <MenuItem value="Student">Student</MenuItem>
                  <MenuItem value="Other">Other</MenuItem>
                </Select>
              </FormControl>

              <Autocomplete
                value={spouse}
                onChange={(e, newValue) => setSpouse(newValue)}
                options={contactOptions}
                loading={loadingContacts}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Spouse"
                    placeholder="Search existing contacts..."
                    InputProps={{
                      ...params.InputProps,
                      endAdornment: (
                        <>
                          {loadingContacts ? <CircularProgress size={20} /> : null}
                          {params.InputProps.endAdornment}
                        </>
                      ),
                    }}
                  />
                )}
                sx={{ mb: 2 }}
              />

              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <TextField
                  label="Birthday"
                  type="date"
                  value={birthday}
                  onChange={(e) => setBirthday(e.target.value)}
                  fullWidth
                  InputLabelProps={{ shrink: true }}
                  InputProps={{
                    endAdornment: <Cake sx={{ color: 'action.disabled' }} />
                  }}
                />
                <TextField
                  label="Anniversary"
                  type="date"
                  value={anniversary}
                  onChange={(e) => setAnniversary(e.target.value)}
                  fullWidth
                  InputLabelProps={{ shrink: true }}
                  InputProps={{
                    endAdornment: <Favorite sx={{ color: 'action.disabled' }} />
                  }}
                />
              </Box>

              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                Children
              </Typography>

              <Stack spacing={1} sx={{ mb: 2 }}>
                {children.map((child) => (
                  <Box key={child.id} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                    <Autocomplete
                      value={child.value}
                      onChange={(e, newValue) => handleUpdateChild(child.id, newValue)}
                      options={contactOptions}
                      fullWidth
                      renderInput={(params) => (
                        <TextField
                          {...params}
                          placeholder="Search existing contacts..."
                          size="small"
                        />
                      )}
                    />
                    <IconButton 
                      size="small" 
                      onClick={() => handleRemoveChild(child.id)}
                      color="error"
                    >
                      <RemoveCircleOutline />
                    </IconButton>
                  </Box>
                ))}
                
                <Button
                  startIcon={<Add />}
                  onClick={handleAddChild}
                  variant="outlined"
                  size="small"
                  sx={{ alignSelf: 'flex-start' }}
                >
                  Add Child
                </Button>
              </Stack>
            </Box>
          </Collapse>
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={() => onClose({ success: false })} disabled={capturing}>
          Cancel
        </Button>
        <Button
          onClick={handleCapture}
          variant="contained"
          disabled={capturing || !firstName.trim()}
          startIcon={capturing ? <CircularProgress size={20} /> : <Send />}
        >
          {capturing ? 'Capturing...' : 'Capture Contact'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default OrgContactCapture;

