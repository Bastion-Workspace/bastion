# Org-Mode Refile UX Flow

**BULLY!** How users refile from the org editor - the complete experience!

## Scenario: User is Viewing inbox.org

### Step 1: User Opens inbox.org in DocumentViewer

```
┌─────────────────────────────────────────────────────────────┐
│ 📄 inbox.org                              [Edit] [Preview]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ * TODO Review quarterly reports                   :@work:  │
│   [2025-10-21 Mon 11:03]                                   │
│   ⋮ (actions)                                              │
│                                                             │
│ * TODO Buy chocolate                            :@errands: │
│   [2025-10-21 Mon 11:15]                                   │
│   ⋮ (actions)                                              │
│                                                             │
│ * TODO Research ML algorithms                  :@personal: │
│   [2025-10-21 Mon 11:20]                                   │
│   ⋮ (actions)                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 2: Hover Over TODO → Actions Menu Appears

```
┌─────────────────────────────────────────────────────────────┐
│ 📄 inbox.org                              [Edit] [Preview]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ * TODO Review quarterly reports                   :@work:  │
│   [2025-10-21 Mon 11:03]                                   │
│   [✓ Toggle Done] [📋 Refile] [✏️ Edit] [🗑️ Delete]        │
│                              ↑                              │
│                         Click this!                         │
│                                                             │
│ * TODO Buy chocolate                            :@errands: │
│   [2025-10-21 Mon 11:15]                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 3: Refile Dialog Opens

```
┌──────────────────────────────────────────────────────┐
│ 📋 Refile Entry                                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│ Entry to refile:                                     │
│ ┌────────────────────────────────────────────────┐  │
│ │ * TODO Review quarterly reports      :@work:   │  │
│ │   [2025-10-21 Mon 11:03]                       │  │
│ └────────────────────────────────────────────────┘  │
│                                                      │
│ 🤖 AI Suggestion: projects.org > Work Projects      │
│    (Based on @work tag and content analysis)        │
│    [Use Suggestion]                                  │
│                                                      │
│ ─────────────────────────────────────────────────   │
│                                                      │
│ Select destination:                                  │
│ ┌────────────────────────────────────────────────┐  │
│ │ 🔍 Search targets...                           │  │
│ └────────────────────────────────────────────────┘  │
│                                                      │
│ ⭐ Favorites                                         │
│ ○ projects.org > Work Projects                      │
│ ○ errands.org                                        │
│                                                      │
│ 📂 All Targets                                       │
│ ○ projects.org (root)                                │
│ ● projects.org > Work Projects         ← Selected   │
│ ○ projects.org > Personal Projects                   │
│ ○ learning.org (root)                                │
│ ○ learning.org > Technical                           │
│ ○ learning.org > Personal Growth                     │
│ ○ errands.org (root)                                 │
│                                                      │
│ Position: ○ First   ● Last                           │
│                                                      │
│ [Cancel]                          [Refile Entry] →  │
└──────────────────────────────────────────────────────┘
```

### Step 4: After Refile → Success Notification

```
┌─────────────────────────────────────────────────────────────┐
│ 📄 inbox.org                              [Edit] [Preview]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ✅ Refiled to projects.org > Work Projects                  │
│                                                             │
│ * TODO Buy chocolate                            :@errands: │
│   [2025-10-21 Mon 11:15]                                   │
│   ⋮ (actions)                                              │
│                                                             │
│ * TODO Research ML algorithms                  :@personal: │
│   [2025-10-21 Mon 11:20]                                   │
│   ⋮ (actions)                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Alternative: Keyboard Shortcut Flow

For power users working in **Edit Mode**:

### Step 1: Cursor on TODO Line
```
* TODO Review quarterly reports                   :@work:
  ↑ Cursor here
```

### Step 2: Press `Ctrl+R` (Refile Hotkey)
→ Opens refile dialog with that entry pre-selected

### Step 3: Type to Search
```
┌──────────────────────────────────────────┐
│ Select destination:                      │
│ ┌────────────────────────────────────┐   │
│ │ work proj▌                         │   │
│ └────────────────────────────────────┘   │
│                                          │
│ Filtered results:                        │
│ ● projects.org > Work Projects           │
│                                          │
└──────────────────────────────────────────┘
```

### Step 4: Press `Enter` to Refile
→ Entry moved instantly!

## Bulk Refile Mode (Inbox Processing)

For processing entire inbox:

### Inbox Review Mode Interface

```
┌──────────────────────────────────────────────────────────────┐
│ 📥 Inbox Review Mode                     [Exit Review Mode]  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ Item 1 of 3                                    [Skip] [Next] │
│                                                              │
│ ┌──────────────────────────────────────────────────────┐    │
│ │ * TODO Review quarterly reports          :@work:     │    │
│ │   [2025-10-21 Mon 11:03]                             │    │
│ └──────────────────────────────────────────────────────┘    │
│                                                              │
│ 🤖 Suggested: projects.org > Work Projects                   │
│                                                              │
│ Quick Actions:                                               │
│ [1] Refile to suggestion                                     │
│ [2] Choose different target                                  │
│ [3] Mark as done                                             │
│ [4] Skip (keep in inbox)                                     │
│ [5] Delete                                                   │
│                                                              │
│ ────────────────────────────────────────────────────────     │
│                                                              │
│ Keyboard shortcuts:                                          │
│ 1-5 = Quick actions  |  r = Refile  |  d = Done  |  n = Next│
└──────────────────────────────────────────────────────────────┘
```

## Implementation: Frontend Components

### 1. OrgRefileDialog Component

**File:** `frontend/src/components/OrgRefileDialog.js`

```javascript
const OrgRefileDialog = ({ 
  open, 
  onClose, 
  entry,           // The TODO/heading to refile
  sourceFile,      // Current file (e.g., "inbox.org")
  sourceLine,      // Line number in source
  onRefileSuccess 
}) => {
  const [targets, setTargets] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [aiSuggestion, setAiSuggestion] = useState(null);
  const [position, setPosition] = useState('last');

  // Fetch refile targets on mount
  useEffect(() => {
    if (open) {
      fetchRefileTargets();
      fetchAiSuggestion();
    }
  }, [open]);

  const fetchRefileTargets = async () => {
    const response = await apiService.get('/api/org/refile-targets');
    setTargets(response.targets);
  };

  const fetchAiSuggestion = async () => {
    const response = await apiService.post('/api/org/suggest-refile', {
      entry_text: entry,
      source_file: sourceFile
    });
    setAiSuggestion(response.suggestion);
  };

  const handleRefile = async () => {
    await apiService.post('/api/org/refile', {
      source_file: sourceFile,
      source_line: sourceLine,
      target_file: selectedTarget.path,
      target_heading: selectedTarget.heading,
      position: position
    });
    
    onRefileSuccess();
    onClose();
  };

  // Filter targets by search query
  const filteredTargets = targets.filter(t => 
    t.display.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>📋 Refile Entry</DialogTitle>
      <DialogContent>
        {/* Entry Preview */}
        <Box sx={{ mb: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
            {entry}
          </Typography>
        </Box>

        {/* AI Suggestion */}
        {aiSuggestion && (
          <Alert severity="info" sx={{ mb: 2 }}>
            🤖 AI Suggestion: <strong>{aiSuggestion.display}</strong>
            <br />
            <Typography variant="caption">{aiSuggestion.reason}</Typography>
            <Button size="small" onClick={() => setSelectedTarget(aiSuggestion)}>
              Use Suggestion
            </Button>
          </Alert>
        )}

        {/* Search */}
        <TextField
          fullWidth
          placeholder="🔍 Search targets..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ mb: 2 }}
        />

        {/* Target List */}
        <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
          {/* Favorites */}
          {filteredTargets.filter(t => t.favorite).length > 0 && (
            <>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>⭐ Favorites</Typography>
              <RadioGroup value={selectedTarget?.id} onChange={(e) => setSelectedTarget(targets.find(t => t.id === e.target.value))}>
                {filteredTargets.filter(t => t.favorite).map(target => (
                  <FormControlLabel
                    key={target.id}
                    value={target.id}
                    control={<Radio />}
                    label={target.display}
                  />
                ))}
              </RadioGroup>
              <Divider sx={{ my: 2 }} />
            </>
          )}

          {/* All Targets */}
          <Typography variant="subtitle2" sx={{ mb: 1 }}>📂 All Targets</Typography>
          <RadioGroup value={selectedTarget?.id}>
            {filteredTargets.map(target => (
              <FormControlLabel
                key={target.id}
                value={target.id}
                control={<Radio />}
                label={target.display}
                onClick={() => setSelectedTarget(target)}
              />
            ))}
          </RadioGroup>
        </Box>

        {/* Position */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2">Position:</Typography>
          <RadioGroup row value={position} onChange={(e) => setPosition(e.target.value)}>
            <FormControlLabel value="first" control={<Radio />} label="First" />
            <FormControlLabel value="last" control={<Radio />} label="Last" />
          </RadioGroup>
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button 
          variant="contained" 
          onClick={handleRefile}
          disabled={!selectedTarget}
        >
          Refile Entry →
        </Button>
      </DialogActions>
    </Dialog>
  );
};
```

### 2. DocumentViewer Integration

**File:** `frontend/src/components/DocumentViewer.js`

Add refile action to org-mode documents:

```javascript
// In DocumentViewer.js - for org files

const [refileDialogOpen, setRefileDialogOpen] = useState(false);
const [refileEntry, setRefileEntry] = useState(null);

// Parse org content to find TODOs and add refile buttons
const renderOrgContent = (content) => {
  const lines = content.split('\n');
  
  return lines.map((line, index) => {
    // Detect TODO lines
    if (line.match(/^\*+\s+(TODO|NEXT|WAITING)/)) {
      return (
        <Box key={index} sx={{ position: 'relative', '&:hover .actions': { opacity: 1 } }}>
          <Typography sx={{ fontFamily: 'monospace' }}>{line}</Typography>
          
          {/* Hover Actions */}
          <Box className="actions" sx={{ opacity: 0, transition: 'opacity 0.2s' }}>
            <IconButton size="small" onClick={() => handleToggleDone(index)}>
              <CheckCircle />
            </IconButton>
            <IconButton size="small" onClick={() => handleRefile(line, index)}>
              <MoveToInbox /> {/* Refile icon */}
            </IconButton>
            <IconButton size="small" onClick={() => handleEdit(index)}>
              <Edit />
            </IconButton>
          </Box>
        </Box>
      );
    }
    
    return <Typography key={index} sx={{ fontFamily: 'monospace' }}>{line}</Typography>;
  });
};

const handleRefile = (entryText, lineNumber) => {
  setRefileEntry({ text: entryText, line: lineNumber });
  setRefileDialogOpen(true);
};

// Render refile dialog
<OrgRefileDialog
  open={refileDialogOpen}
  onClose={() => setRefileDialogOpen(false)}
  entry={refileEntry?.text}
  sourceFile={document?.filename}
  sourceLine={refileEntry?.line}
  onRefileSuccess={() => {
    // Refresh document
    fetchDocument();
  }}
/>
```

## Settings Integration

**In Settings → Org-Mode Settings Tab:**

```javascript
<Box sx={{ mt: 3 }}>
  <Typography variant="h6">Refile Preferences</Typography>
  
  {/* Favorite Targets */}
  <Box sx={{ mt: 2 }}>
    <Typography variant="subtitle2">Favorite Refile Targets</Typography>
    <Typography variant="caption" color="text.secondary">
      Pin frequently-used targets to the top of the refile dialog
    </Typography>
    
    <List>
      {favoriteTargets.map((target, index) => (
        <ListItem key={index}>
          <ListItemText primary={target} />
          <IconButton onClick={() => removeFavorite(index)}>
            <Delete />
          </IconButton>
        </ListItem>
      ))}
    </List>
    
    <Button onClick={openAddFavoriteDialog}>
      Add Favorite Target
    </Button>
  </Box>

  {/* Auto-Refile Rules */}
  <Box sx={{ mt: 3 }}>
    <Typography variant="subtitle2">Auto-Refile Rules</Typography>
    <Typography variant="caption" color="text.secondary">
      Automatically suggest targets based on tags
    </Typography>
    
    {/* Rule editor */}
  </Box>

  {/* Excluded Files */}
  <Box sx={{ mt: 3 }}>
    <Typography variant="subtitle2">Excluded Files</Typography>
    <Typography variant="caption" color="text.secondary">
      Don't show these files as refile targets
    </Typography>
    
    <Autocomplete
      multiple
      options={allOrgFiles}
      value={excludedFiles}
      onChange={(e, newValue) => setExcludedFiles(newValue)}
      renderInput={(params) => <TextField {...params} placeholder="Select files to exclude" />}
    />
  </Box>
</Box>
```

## Complete User Flow Summary

1. **Targets are auto-discovered** from all .org files (no manual setup required)
2. **User views inbox.org** in DocumentViewer
3. **Hovers over TODO** → sees action buttons
4. **Clicks 📋 Refile button** → opens refile dialog
5. **Gets AI suggestion** based on tags/content
6. **Searches/selects target** from list
7. **Clicks "Refile Entry"** → entry moved!
8. **Document auto-refreshes** via WebSocket
9. **Settings** allow customization of favorites and rules

**By George!** The refile experience is smooth as a well-oiled cavalry charge! 🏇



