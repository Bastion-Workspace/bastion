/**
 * Collapsible tool picker: groups tools by category into accordion sections.
 * Each category header is clickable to expand/collapse, showing selected count.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  FormControlLabel,
  Checkbox,
  Collapse,
  ButtonBase,
} from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

export default function CollapsibleToolPicker({
  actions = [],
  selectedTools = [],
  onToggleTool,
}) {
  const [expandedCategories, setExpandedCategories] = useState({});

  const byCategory = useMemo(() => {
    const map = {};
    actions.forEach((a) => {
      const name = typeof a === 'string' ? a : a?.name;
      if (!name) return;
      const cat = (typeof a === 'object' && a?.category) ? a.category : 'General';
      if (!map[cat]) map[cat] = [];
      map[cat].push({
        name,
        description: typeof a === 'object'
          ? (a.short_description || a.description || a.name)
          : a,
      });
    });
    return map;
  }, [actions]);

  const categories = useMemo(() => Object.keys(byCategory).sort(), [byCategory]);

  const toggleCategory = (cat) => {
    setExpandedCategories((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  const toggleAllInCategory = (cat, shouldSelect) => {
    const names = (byCategory[cat] || []).map((t) => t.name);
    if (shouldSelect) {
      const toAdd = names.filter((n) => !selectedTools.includes(n));
      onToggleTool([...selectedTools, ...toAdd]);
    } else {
      const nameSet = new Set(names);
      onToggleTool(selectedTools.filter((t) => !nameSet.has(t)));
    }
  };

  return (
    <Box
      sx={{
        maxHeight: 280,
        overflowY: 'auto',
        overflowX: 'hidden',
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
      }}
    >
      {categories.map((cat) => {
        const tools = byCategory[cat] || [];
        const expanded = !!expandedCategories[cat];
        const selectedCount = tools.filter((t) =>
          selectedTools.includes(t.name)
        ).length;
        const allSelected =
          selectedCount === tools.length && tools.length > 0;
        const someSelected =
          selectedCount > 0 && selectedCount < tools.length;

        return (
          <Box key={cat}>
            <ButtonBase
              onClick={() => toggleCategory(cat)}
              sx={{
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                px: 1.5,
                py: 0.75,
                justifyContent: 'space-between',
                bgcolor: expanded ? 'action.selected' : 'transparent',
                '&:hover': { bgcolor: 'action.hover' },
                borderBottom: 1,
                borderColor: 'divider',
              }}
            >
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  minWidth: 0,
                }}
              >
                <Checkbox
                  size="small"
                  checked={allSelected}
                  indeterminate={someSelected}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleAllInCategory(cat, !allSelected);
                  }}
                  sx={{ p: 0.25 }}
                />
                <Typography variant="body2" fontWeight={600} noWrap>
                  {cat}
                </Typography>
                {selectedCount > 0 && (
                  <Typography
                    variant="caption"
                    sx={{
                      bgcolor: 'primary.main',
                      color: 'primary.contrastText',
                      borderRadius: '10px',
                      px: 0.75,
                      py: 0.1,
                      fontSize: '0.7rem',
                      lineHeight: 1.4,
                      minWidth: 18,
                      textAlign: 'center',
                    }}
                  >
                    {selectedCount}
                  </Typography>
                )}
              </Box>
              {expanded ? (
                <ExpandLess fontSize="small" />
              ) : (
                <ExpandMore fontSize="small" />
              )}
            </ButtonBase>
            <Collapse in={expanded}>
              <Box sx={{ pl: 1.5, pr: 1, py: 0.5 }}>
                {tools.map(({ name, description }) => {
                  const checked = selectedTools.includes(name);
                  return (
                    <FormControlLabel
                      key={name}
                      control={
                        <Checkbox
                          size="small"
                          checked={checked}
                          onChange={() => {
                            const next = checked
                              ? selectedTools.filter((t) => t !== name)
                              : [...selectedTools, name];
                            onToggleTool(next);
                          }}
                          sx={{ p: 0.25, mr: 0.5 }}
                        />
                      }
                      label={description || name}
                      sx={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        mx: 0,
                        my: 0.25,
                        '& .MuiFormControlLabel-label': {
                          fontSize: '0.875rem',
                          lineHeight: 1.4,
                          whiteSpace: 'normal',
                          wordBreak: 'break-word',
                        },
                      }}
                    />
                  );
                })}
              </Box>
            </Collapse>
          </Box>
        );
      })}
      {categories.length === 0 && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ p: 2, textAlign: 'center' }}
        >
          No tools available
        </Typography>
      )}
    </Box>
  );
}
