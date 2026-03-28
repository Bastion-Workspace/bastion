/**
 * Inline configuration for external tool packs (email, calendar, MCP) on playbook steps.
 * Updates only external pack entries; parent merges with built-in tool_packs.
 */

import React, { useMemo, useCallback } from 'react';
import {
  Box,
  Typography,
  Checkbox,
  FormControlLabel,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
} from '@mui/material';

function normalizePackEntry(e) {
  if (typeof e === 'object' && e?.pack) {
    return {
      pack: e.pack,
      mode: e.mode === 'read' ? 'read' : 'full',
      connections: Array.isArray(e.connections) ? e.connections.map((x) => Number(x)) : [],
    };
  }
  return { pack: String(e).trim(), mode: 'full', connections: [] };
}

export default function ExternalPackConfigurator({ externalPacks, stepToolPacks, onCommit, readOnly }) {
  const extNames = useMemo(() => new Set((externalPacks || []).map((p) => p.name)), [externalPacks]);

  const externalOnlyEntries = useMemo(() => {
    const raw = Array.isArray(stepToolPacks) ? stepToolPacks : [];
    return raw.map(normalizePackEntry).filter((e) => extNames.has(e.pack));
  }, [stepToolPacks, extNames]);

  const commitList = useCallback(
    (nextExternalList) => {
      onCommit(nextExternalList);
    },
    [onCommit]
  );

  const getEntry = (packName) => externalOnlyEntries.find((e) => e.pack === packName);

  const applyPackUpdate = (packName, mutator) => {
    if (readOnly) return;
    const others = externalOnlyEntries.filter((e) => e.pack !== packName);
    const cur = getEntry(packName) || { pack: packName, mode: 'full', connections: [] };
    const draft = {
      ...cur,
      connections: [...(cur.connections || [])],
    };
    const updated = mutator(draft);
    if (!updated || !updated.connections || updated.connections.length === 0) {
      commitList(others);
    } else {
      commitList([...others, updated]);
    }
  };

  if (!externalPacks || externalPacks.length === 0) {
    return null;
  }

  return (
    <Box sx={{ mb: 2 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        External tool packs
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        Choose which connected accounts or MCP servers this step may use. Run discovery on MCP servers in Settings
        if tools are missing.
      </Typography>
      {(externalPacks || []).map((pack) => {
        const packName = pack.name;
        const entry = getEntry(packName);
        const selected = new Set((entry?.connections || []).map((x) => Number(x)));
        const conns = Array.isArray(pack.available_connections) ? pack.available_connections : [];
        const hasWrite = pack.has_write_tools === true;
        const mode = entry?.mode === 'read' ? 'read' : 'full';
        const displayTitle = packName.startsWith('mcp:') ? (pack.description || packName) : packName;

        const toggleConn = (id) => {
          applyPackUpdate(packName, (d) => {
            const n = Number(id);
            const set = new Set(d.connections.map(Number));
            if (set.has(n)) set.delete(n);
            else set.add(n);
            d.connections = Array.from(set);
            return d;
          });
        };

        const setMode = (v) => {
          if (readOnly || v == null) return;
          if (!entry || !entry.connections || entry.connections.length === 0) return;
          applyPackUpdate(packName, (d) => {
            d.mode = v;
            return d;
          });
        };

        const toggleSingleOrHeader = () => {
          if (conns.length !== 1) return;
          const id = Number(conns[0].id);
          if (selected.has(id)) {
            applyPackUpdate(packName, () => null);
          } else {
            commitList([
              ...externalOnlyEntries.filter((e) => e.pack !== packName),
              { pack: packName, mode: 'full', connections: [id] },
            ]);
          }
        };

        return (
          <Box key={packName} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5, mb: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
              <Typography variant="body2" fontWeight={600}>
                {displayTitle}
                {selected.size > 0 ? (
                  <Typography component="span" variant="caption" color="primary" sx={{ ml: 1 }}>
                    ({selected.size} selected)
                  </Typography>
                ) : null}
              </Typography>
              {hasWrite && selected.size > 0 && (
                <ToggleButtonGroup
                  size="small"
                  exclusive
                  value={mode}
                  onChange={(_, v) => setMode(v)}
                >
                  <ToggleButton value="read">Read</ToggleButton>
                  <ToggleButton value="full">Full</ToggleButton>
                </ToggleButtonGroup>
              )}
            </Box>
            {conns.length === 0 ? (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                No connections configured. Add accounts or MCP servers in Settings → External connections.
              </Typography>
            ) : conns.length === 1 ? (
              <FormControlLabel
                sx={{ mt: 0.5 }}
                control={
                  <Checkbox
                    checked={selected.has(Number(conns[0].id))}
                    onChange={toggleSingleOrHeader}
                    disabled={readOnly}
                  />
                }
                label={conns[0].label || `Connection ${conns[0].id}`}
              />
            ) : (
              <Box sx={{ mt: 1 }}>
                {conns.map((c) => (
                  <FormControlLabel
                    key={c.id}
                    control={
                      <Checkbox
                        checked={selected.has(Number(c.id))}
                        onChange={() => toggleConn(c.id)}
                        disabled={readOnly}
                      />
                    }
                    label={c.label || `Connection ${c.id}`}
                  />
                ))}
              </Box>
            )}
            {pack.tools && pack.tools.length > 0 && (
              <>
                <Divider sx={{ my: 1 }} />
                <Typography variant="caption" color="text.secondary">
                  Tools: {pack.tools.slice(0, 12).join(', ')}
                  {pack.tools.length > 12 ? '…' : ''}
                </Typography>
              </>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
