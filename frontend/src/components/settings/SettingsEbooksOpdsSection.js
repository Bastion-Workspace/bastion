import React, { useCallback, useMemo, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Divider,
  IconButton,
  Alert,
  Switch,
  FormControlLabel,
  Chip,
} from '@mui/material';
import { Add, Delete } from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import ebooksService from '../../services/ebooksService';

function newCatalogId() {
  return `cat_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function apiErrorDetail(err) {
  const d = err?.response?.data?.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d
      .map((x) => (typeof x === 'object' && x !== null ? x.msg || JSON.stringify(x) : String(x)))
      .join('; ');
  }
  return err?.message || 'Request failed';
}

export default function SettingsEbooksOpdsSection() {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery(['ebooks-settings'], () => ebooksService.getSettings(), {
    staleTime: 30_000,
  });

  const [catalogs, setCatalogs] = useState([]);
  const [dirty, setDirty] = useState(false);
  const [catalogAlert, setCatalogAlert] = useState(null);
  const [kosyncAlert, setKosyncAlert] = useState(null);

  React.useEffect(() => {
    if (data?.catalogs) {
      setCatalogs(data.catalogs.map((c) => ({ ...c })));
      setDirty(false);
    }
  }, [data]);

  const saveCatalogsMutation = useMutation(
    (payload) => ebooksService.putSettings(payload),
    {
      onMutate: () => setCatalogAlert(null),
      onSuccess: () => {
        queryClient.invalidateQueries(['ebooks-settings']);
        setDirty(false);
        setCatalogAlert({ severity: 'success', text: 'Catalogs saved.' });
      },
      onError: (err) => {
        setCatalogAlert({ severity: 'error', text: apiErrorDetail(err) });
      },
    }
  );

  const onAddCatalog = useCallback(() => {
    setCatalogs((prev) => [
      ...prev,
      { id: newCatalogId(), title: 'New catalog', root_url: '', verify_ssl: true },
    ]);
    setDirty(true);
  }, []);

  const onRemove = useCallback((id) => {
    setCatalogs((prev) => prev.filter((c) => c.id !== id));
    setDirty(true);
  }, []);

  const onField = useCallback((id, field, value) => {
    setCatalogs((prev) => prev.map((c) => (c.id === id ? { ...c, [field]: value } : c)));
    setDirty(true);
  }, []);

  const onSaveCatalogs = useCallback(() => {
    const trimmed = catalogs.map((c) => {
      const title = (c.title || '').trim();
      const root_url = (c.root_url || '').trim();
      const entry = {
        id: c.id,
        title,
        root_url,
        verify_ssl: c.verify_ssl !== false,
      };
      if (c.http_basic_b64) {
        entry.http_basic_b64 = c.http_basic_b64;
      } else if (c.http_basic_b64 === '') {
        entry.http_basic_b64 = '';
      }
      return entry;
    });
    const invalid = trimmed.find((c) => !c.root_url);
    if (invalid) {
      setCatalogAlert({ severity: 'warning', text: 'Each catalog needs a root URL (including hostname).' });
      return;
    }
    saveCatalogsMutation.mutate({ catalogs: trimmed });
  }, [catalogs, saveCatalogsMutation]);

  const [basicUser, setBasicUser] = useState({});
  const [basicPass, setBasicPass] = useState({});

  const applyBasicAuth = useCallback(
    (id) => {
      const u = (basicUser[id] || '').trim();
      const p = basicPass[id] || '';
      if (!u && !p) {
        onField(id, 'http_basic_b64', '');
        return;
      }
      const token = typeof btoa !== 'undefined' ? btoa(`${u}:${p}`) : '';
      onField(id, 'http_basic_b64', token || '');
    },
    [basicUser, basicPass, onField]
  );

  const kosync = data?.kosync || {};
  const [ksBase, setKsBase] = useState('');
  const [ksUser, setKsUser] = useState('');
  const [ksPass, setKsPass] = useState('');
  const [ksVerify, setKsVerify] = useState(true);

  React.useEffect(() => {
    setKsBase(kosync.base_url || '');
    setKsUser(kosync.username || '');
    setKsPass('');
    setKsVerify(kosync.verify_ssl !== false);
  }, [kosync.base_url, kosync.username, kosync.verify_ssl]);

  const saveKosyncMutation = useMutation((body) => ebooksService.putKosyncSettings(body), {
    onMutate: () => setKosyncAlert(null),
    onSuccess: () => {
      queryClient.invalidateQueries(['ebooks-settings']);
      setKosyncAlert({ severity: 'success', text: 'KoSync settings saved.' });
    },
    onError: (err) => setKosyncAlert({ severity: 'error', text: apiErrorDetail(err) }),
  });

  const onSaveKosync = useCallback(() => {
    saveKosyncMutation.mutate({
      base_url: ksBase.trim(),
      username: ksUser.trim(),
      password: ksPass || undefined,
      verify_ssl: ksVerify,
    });
  }, [ksBase, ksUser, ksPass, ksVerify, saveKosyncMutation]);

  const testMutation = useMutation((vars) => ebooksService.kosyncTest(vars), {
    onMutate: () => setKosyncAlert(null),
    onSuccess: (data) => {
      setKosyncAlert({
        severity: data?.ok ? 'success' : 'warning',
        text: data?.ok
          ? 'KoSync accepted these credentials (authorize OK).'
          : `Login test response: ${JSON.stringify(data)}`,
      });
    },
    onError: (err) => setKosyncAlert({ severity: 'error', text: apiErrorDetail(err) }),
  });

  const healthMutation = useMutation(() => ebooksService.kosyncHealth(), {
    onMutate: () => setKosyncAlert(null),
    onSuccess: (data) => {
      setKosyncAlert({
        severity: data?.ok ? 'success' : 'warning',
        text: data?.ok ? 'KoSync server reachable (saved base URL).' : JSON.stringify(data),
      });
    },
    onError: (err) => setKosyncAlert({ severity: 'error', text: apiErrorDetail(err) }),
  });

  const registerMutation = useMutation((vars) => ebooksService.kosyncRegister(vars), {
    onMutate: () => setKosyncAlert(null),
    onSuccess: () => {
      queryClient.invalidateQueries(['ebooks-settings']);
      setKsPass('');
      setKosyncAlert({ severity: 'success', text: 'Registered or updated user on KoSync.' });
    },
    onError: (err) => setKosyncAlert({ severity: 'error', text: apiErrorDetail(err) }),
  });

  const catalogRows = useMemo(() => catalogs, [catalogs]);

  if (isLoading && !data) {
    return (
      <Box p={2}>
        <Typography>Loading…</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: 900, mx: 'auto', p: 2 }}>
      <Typography variant="h5" gutterBottom>
        Ebooks (OPDS)
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        Add OPDS catalogs for the Documents sidebar. After you save at least one catalog, an <strong>Ebooks</strong>{' '}
        segment appears in Documents (below the folder tree); open <strong>OPDS catalogs</strong> there. Books are not
        imported into Bastion; reading progress can sync with a self-hosted{' '}
        <a href="https://github.com/koreader/koreader-sync-server" target="_blank" rel="noreferrer">
          KoSync
        </a>{' '}
        server (same protocol as KOReader).
      </Typography>
      <Typography variant="body2" color="text.secondary" paragraph>
        HTTP Basic credentials are stored for your account in the application database (Base64 encoding is not
        encryption). They are not returned by the API after save; use at-rest database encryption if your threat model
        requires it.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            OPDS catalogs
          </Typography>
          {catalogAlert && (
            <Alert severity={catalogAlert.severity} sx={{ mb: 2 }} onClose={() => setCatalogAlert(null)}>
              {catalogAlert.text}
            </Alert>
          )}
          {catalogRows.length === 0 && (
            <Alert severity="info" sx={{ mb: 2 }}>
              No catalogs yet. Add one with the root URL of an OPDS feed (often ending in <code>catalog.atom</code> or
              similar).
            </Alert>
          )}
          {catalogRows.map((c) => (
            <Box key={c.id} sx={{ mb: 2, p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" gap={1}>
                <TextField
                  label="Title"
                  value={c.title}
                  onChange={(e) => onField(c.id, 'title', e.target.value)}
                  size="small"
                  fullWidth
                  sx={{ mb: 1 }}
                />
                <IconButton aria-label="Remove catalog" onClick={() => onRemove(c.id)} color="error">
                  <Delete />
                </IconButton>
              </Box>
              <TextField
                label="Root URL"
                value={c.root_url}
                onChange={(e) => onField(c.id, 'root_url', e.target.value)}
                size="small"
                fullWidth
                sx={{ mb: 1 }}
                placeholder="https://example.com/opds/catalog.atom"
                helperText="Must include a hostname. You may omit the scheme for public hosts (https is assumed)."
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={c.verify_ssl !== false}
                    onChange={(e) => onField(c.id, 'verify_ssl', e.target.checked)}
                  />
                }
                label="Verify TLS for this catalog"
              />
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
                <Typography variant="caption" color="text.secondary">
                  Optional HTTP Basic (write-only; not shown again after save)
                </Typography>
                {c.http_basic_configured ? (
                  <Chip size="small" label="Basic auth on file" sx={{ height: 20 }} color="info" variant="outlined" />
                ) : null}
              </Box>
              <Box display="flex" gap={1} flexWrap="wrap" alignItems="center">
                <TextField
                  label="Basic user"
                  size="small"
                  value={basicUser[c.id] ?? ''}
                  onChange={(e) => setBasicUser((p) => ({ ...p, [c.id]: e.target.value }))}
                />
                <TextField
                  label="Basic password"
                  type="password"
                  size="small"
                  value={basicPass[c.id] ?? ''}
                  onChange={(e) => setBasicPass((p) => ({ ...p, [c.id]: e.target.value }))}
                />
                <Button variant="outlined" size="small" onClick={() => applyBasicAuth(c.id)}>
                  Apply basic auth
                </Button>
              </Box>
            </Box>
          ))}
          <Box display="flex" gap={1} mt={2}>
            <Button startIcon={<Add />} variant="outlined" onClick={onAddCatalog}>
              Add catalog
            </Button>
            <Button variant="contained" disabled={!dirty || saveCatalogsMutation.isLoading} onClick={onSaveCatalogs}>
              Save catalogs
            </Button>
          </Box>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            KoSync
          </Typography>
          <TextField
            label="Server base URL"
            value={ksBase}
            onChange={(e) => setKsBase(e.target.value)}
            fullWidth
            margin="normal"
            placeholder="https://kosync.example.com"
          />
          <TextField label="Username" value={ksUser} onChange={(e) => setKsUser(e.target.value)} fullWidth margin="normal" />
          <TextField
            label="Password"
            type="password"
            value={ksPass}
            onChange={(e) => setKsPass(e.target.value)}
            fullWidth
            margin="normal"
            helperText="Leave blank when saving to keep the existing derived key."
          />
          <FormControlLabel
            control={<Switch checked={ksVerify} onChange={(e) => setKsVerify(e.target.checked)} />}
            label="Verify TLS when talking to KoSync"
          />
          <Divider sx={{ my: 2 }} />
          {kosyncAlert && (
            <Alert severity={kosyncAlert.severity} sx={{ mb: 2 }} onClose={() => setKosyncAlert(null)}>
              {kosyncAlert.text}
            </Alert>
          )}
          <Box display="flex" flexWrap="wrap" gap={1}>
            <Button variant="contained" onClick={onSaveKosync} disabled={saveKosyncMutation.isLoading}>
              Save KoSync settings
            </Button>
            <Button
              variant="outlined"
              onClick={() =>
                testMutation.mutate({
                  base_url: ksBase.trim(),
                  username: ksUser.trim(),
                  password: ksPass,
                  verify_ssl: ksVerify,
                })
              }
              disabled={testMutation.isLoading || !ksPass}
            >
              Test login
            </Button>
            <Button variant="outlined" onClick={() => healthMutation.mutate()} disabled={healthMutation.isLoading}>
              Health from saved URL
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              onClick={() =>
                registerMutation.mutate({
                  username: ksUser.trim(),
                  password: ksPass,
                  base_url: ksBase.trim() || undefined,
                  verify_ssl: ksVerify,
                })
              }
              disabled={registerMutation.isLoading || !ksPass || !ksUser}
            >
              Register user
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
