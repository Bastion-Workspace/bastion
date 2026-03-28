/**
 * Contacts View: merged O365 + org-mode contacts.
 * Source selector, search, and contact cards with source badge.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  InputAdornment,
  Chip,
  CircularProgress,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Contacts, Search, Email, Phone, Business, Cake } from '@mui/icons-material';
import apiService from '../services/apiService';

const normalizeEmail = (e) => (e && String(e).trim().toLowerCase()) || '';

const ContactsView = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connections, setConnections] = useState([]);
  const [connectionId, setConnectionId] = useState(null);
  const [includeOrg, setIncludeOrg] = useState(true);
  const [o365Contacts, setO365Contacts] = useState([]);
  const [orgContacts, setOrgContacts] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');

  const loadConnections = useCallback(async () => {
    if (!apiService.contacts) return;
    try {
      const res = await apiService.contacts.getConnections();
      const conns = res?.connections ?? [];
      setConnections(conns);
      if (conns.length > 0 && connectionId == null) {
        setConnectionId(conns[0].id);
      }
    } catch (err) {
      console.error('Contacts connections error:', err);
    }
  }, [connectionId]);

  const loadO365 = useCallback(async () => {
    if (!apiService.contacts || connectionId == null) {
      setO365Contacts([]);
      return;
    }
    try {
      const res = await apiService.contacts.getO365Contacts(connectionId, '', 500);
      setO365Contacts(res?.contacts ?? []);
      if (res?.error && !res?.contacts?.length) {
        setError(res.error);
      }
    } catch (err) {
      console.error('O365 contacts error:', err);
      setO365Contacts([]);
    }
  }, [connectionId]);

  const loadOrg = useCallback(async () => {
    if (!apiService.contacts || !includeOrg) {
      setOrgContacts([]);
      return;
    }
    try {
      const res = await apiService.contacts.getOrgContacts(null, 500);
      setOrgContacts(res?.results ?? []);
    } catch (err) {
      console.error('Org contacts error:', err);
      setOrgContacts([]);
    }
  }, [includeOrg]);

  useEffect(() => {
    loadConnections();
  }, [loadConnections]);

  useEffect(() => {
    if (connectionId != null) loadO365();
    else setO365Contacts([]);
  }, [connectionId, loadO365]);

  useEffect(() => {
    if (includeOrg) loadOrg();
    else setOrgContacts([]);
  }, [includeOrg, loadOrg]);

  useEffect(() => {
    setLoading(false);
  }, []);

  const normalizeO365 = (c) => {
    const emails = (c.email_addresses || []).map((e) => (e.address || e).trim()).filter(Boolean);
    const phones = (c.phone_numbers || []).map((p) => (typeof p === 'object' ? p.number : p)).filter(Boolean);
    return {
      id: c.id,
      source: 'o365',
      displayName: c.display_name || [c.given_name, c.surname].filter(Boolean).join(' ') || '(No name)',
      givenName: c.given_name || '',
      surname: c.surname || '',
      emails,
      phones,
      companyName: c.company_name || '',
      jobTitle: c.job_title || '',
      birthday: c.birthday || null,
      notes: c.notes || '',
    };
  };

  const normalizeOrg = (item) => {
    const props = item.properties || {};
    const email = props.EMAIL || props.email || '';
    const emails = email ? [email.trim()] : [];
    const phone = props.PHONE || props.phone || props.MOBILE || '';
    const phones = phone ? [phone.trim()] : [];
    return {
      id: item.document_id + '#' + (item.line_number ?? '') + '#' + (item.heading || ''),
      source: 'org',
      displayName: item.heading || item.title || '(No name)',
      givenName: '',
      surname: '',
      emails,
      phones,
      companyName: props.COMPANY || props.company || '',
      jobTitle: props.TITLE || props.title || '',
      birthday: props.BIRTHDAY || props.birthday || null,
      notes: '',
      file_path: item.file_path || item.filename,
      line_number: item.line_number,
    };
  };

  const merged = useMemo(() => {
    const byEmail = new Map();
    o365Contacts.forEach((c) => {
      const n = normalizeO365(c);
      n.emails.forEach((e) => {
        const key = normalizeEmail(e);
        if (key && !byEmail.has(key)) byEmail.set(key, n);
      });
      if (n.emails.length === 0) byEmail.set(`o365:${n.id}`, n);
    });
    orgContacts.forEach((item) => {
      const n = normalizeOrg(item);
      const added = n.emails.some((e) => {
        const key = normalizeEmail(e);
        if (key && !byEmail.has(key)) {
          byEmail.set(key, n);
          return true;
        }
        return false;
      });
      if (!added && n.emails.length === 0) byEmail.set(`org:${n.id}`, n);
    });
    return Array.from(byEmail.values());
  }, [o365Contacts, orgContacts]);

  const filtered = useMemo(() => {
    const q = (searchQuery || '').trim().toLowerCase();
    if (!q) return merged;
    return merged.filter((c) => {
      const name = (c.displayName || '').toLowerCase();
      const company = (c.companyName || '').toLowerCase();
      const emails = (c.emails || []).join(' ').toLowerCase();
      const phones = (c.phones || []).join(' ').toLowerCase();
      return name.includes(q) || company.includes(q) || emails.includes(q) || phones.includes(q);
    });
  }, [merged, searchQuery]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', backgroundColor: 'background.paper' }}>
        <Typography variant="h6" sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <Contacts /> Contacts
        </Typography>

        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 1 }}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Source</InputLabel>
            <Select
              value={connectionId ?? ''}
              label="Source"
              onChange={(e) => setConnectionId(e.target.value || null)}
            >
              {connections.map((conn) => (
                <MenuItem key={conn.id} value={conn.id}>
                  {conn.display_name || conn.provider}
                </MenuItem>
              ))}
              {connections.length === 0 && (
                <MenuItem value="">No O365 connection</MenuItem>
              )}
            </Select>
          </FormControl>
          <Chip
            label={includeOrg ? 'Org mode: On' : 'Org mode: Off'}
            color={includeOrg ? 'primary' : 'default'}
            variant={includeOrg ? 'filled' : 'outlined'}
            onClick={() => setIncludeOrg(!includeOrg)}
            sx={{ cursor: 'pointer' }}
          />
        </Box>

        <TextField
          size="small"
          placeholder="Search contacts..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search fontSize="small" />
              </InputAdornment>
            ),
          }}
          sx={{ width: '100%', maxWidth: 400 }}
        />
      </Box>

      {error && (
        <Alert severity="warning" onClose={() => setError(null)} sx={{ m: 1 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Paper variant="outlined" sx={{ overflow: 'hidden' }}>
            <List dense>
              {filtered.length === 0 ? (
                <ListItem>
                  <ListItemText primary="No contacts" secondary={searchQuery ? "Try a different search." : "Connect O365 in Settings or add org-mode contacts."} />
                </ListItem>
              ) : (
                filtered.map((c) => (
                  <ListItem key={c.id} divider>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                          <Typography variant="subtitle1">{c.displayName}</Typography>
                          <Chip size="small" label={c.source === 'o365' ? 'O365' : 'Org'} variant="outlined" />
                        </Box>
                      }
                      secondary={
                        <Box component="span" sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 0.5 }}>
                          {c.emails?.length > 0 && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Email fontSize="small" sx={{ color: 'text.secondary' }} />
                              <Typography variant="body2" color="text.secondary">{c.emails.join(', ')}</Typography>
                            </Box>
                          )}
                          {c.phones?.length > 0 && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Phone fontSize="small" sx={{ color: 'text.secondary' }} />
                              <Typography variant="body2" color="text.secondary">{c.phones.join(', ')}</Typography>
                            </Box>
                          )}
                          {(c.companyName || c.jobTitle) && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Business fontSize="small" sx={{ color: 'text.secondary' }} />
                              <Typography variant="body2" color="text.secondary">
                                {[c.companyName, c.jobTitle].filter(Boolean).join(' · ')}
                              </Typography>
                            </Box>
                          )}
                          {c.birthday && (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Cake fontSize="small" sx={{ color: 'text.secondary' }} />
                              <Typography variant="body2" color="text.secondary">{c.birthday}</Typography>
                            </Box>
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                ))
              )}
            </List>
          </Paper>
        )}
        {!loading && merged.length > 0 && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            {filtered.length} of {merged.length} contacts
            {includeOrg && ` (O365 + org-mode)`}
          </Typography>
        )}
      </Box>
    </Box>
  );
};

export default ContactsView;
