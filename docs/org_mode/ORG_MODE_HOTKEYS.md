# Org-Mode Keyboard Shortcuts

**BULLY!** Consistent hotkey patterns for org-mode workflows!

## Global Hotkeys (Work Anywhere)

### Capture
**Hotkey:** `Ctrl+Shift+C` (Windows/Linux) or `Cmd+Shift+C` (Mac)

**What it does:**
- Opens quick capture modal
- Add TODO, note, journal entry, or meeting notes
- Appends to inbox.org

**Usage:**
- Press hotkey from anywhere in the app
- Type your content
- Press `Ctrl+Enter` or click "Capture"
- Entry saved to inbox.org

---

## Org File Editing Hotkeys (In DocumentViewer)

### Refile
**Hotkey:** `Ctrl+Shift+M` (Windows/Linux) or `Cmd+Shift+M` (Mac)

**What it does:**
- Opens refile dialog for entry at cursor
- Move TODO/heading to proper project file
- AI suggests best location

**Usage:**
- Open any .org file in DocumentViewer
- Put cursor on a TODO line
- Press `Ctrl+Shift+M`
- Select target destination
- Press `Enter` to refile

**Note:** Only works when viewing .org files in the editor

---

### Archive
**Hotkey:** `Ctrl+Shift+A` (Windows/Linux) or `Cmd+Shift+A` (Mac)

**What it does:**
- Archives entry at cursor to `{filename}_archive.org`
- Removes from active file, preserves in archive
- Adds ARCHIVE_TIME and ARCHIVE_FILE properties

**Usage:**
- Open any .org file in DocumentViewer
- Put cursor on a DONE task
- Press `Ctrl+Shift+A`
- Confirm archive operation
- Entry moved to archive file

**Note:** Only works when viewing .org files in the editor

---

### Tag
**Hotkey:** `Ctrl+Shift+E` (Windows/Linux) or `Cmd+Shift+E` (Mac)

**What it does:**
- Opens tag dialog for entry at cursor
- Add or update tags on current heading
- Merge with existing tags or replace them

**Usage:**
- Open any .org file in DocumentViewer
- Put cursor on a TODO/heading line
- Press `Ctrl+Shift+E`
- Enter tags (e.g., @outside, urgent)
- Press `Enter` to add tags

**Note:** Only works when viewing .org files in the editor

---

### Insert File Link
**Hotkey:** `Ctrl+Alt+L` (Windows/Linux) or `Cmd+Alt+L` (Mac)

**What it does:**
- Opens document picker dialog
- Select a document from your document tree
- Inserts Org Mode file link at cursor position
- Automatically calculates relative path

**Usage:**
- Open any .org file in DocumentViewer
- Put cursor where you want the link
- Press `Ctrl+Alt+L`
- Search and select a document
- Press `Enter` or click "Insert Link"
- Link is inserted in format: `[[file:./path/to/file.md][Description]]`

**Note:** Only works when viewing .org files in the editor. Relative paths are calculated automatically based on current document location.

---

## Pattern: Ctrl+Shift + Letter

**Why this pattern?**
- ✅ **Consistent**: Both org-mode actions use `Ctrl+Shift`
- ✅ **Memorable**: C for Capture, R for Refile
- ✅ **Safe**: Shift modifier prevents accidental triggers
- ✅ **Cross-platform**: Works on Windows, Linux, and Mac (with Cmd instead of Ctrl)

**Mnemonic:**
- **Ctrl+Shift+C** = **C**apture (create new entry)
- **Ctrl+Shift+M** = **M**ove/refile (move existing entry)
- **Ctrl+Shift+A** = **A**rchive (archive DONE entries)
- **Ctrl+Shift+E** = tag **E**ntry (add tags to entry)
- **Ctrl+Alt+L** = **L**ink (insert file link)

**Note:** We use M instead of R for refile because Ctrl+Shift+R is the browser's hard refresh shortcut.

---

## In-Dialog Hotkeys

### Quick Capture Modal
- `Ctrl+Enter` - Submit capture
- `Esc` - Cancel and close
- `Enter` (in tag field) - Add tag

### Refile Dialog
- `Enter` - Execute refile (when target selected)
- `Esc` - Cancel and close
- Type to search/filter targets

---

## All TODOs Page

### Refile from List
**Hotkey:** None (use mouse)

**What it does:**
- Click 📋 refile icon next to any TODO
- Opens same refile dialog
- Works for TODOs from any org file

---

## Future Hotkeys (Planned)

### Clock In/Out
**Proposed:** `Ctrl+Shift+I` / `Ctrl+Shift+O`
- Start/stop time tracking on task

### Agenda View
**Proposed:** `Ctrl+Shift+G` (for aGenda)
- Open org-mode agenda calendar

**Note:** All hotkeys avoid browser conflicts like Ctrl+Shift+R (refresh), Ctrl+Shift+T (reopen tab), Ctrl+Shift+N (incognito), etc.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│ ORG-MODE HOTKEYS                                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ GLOBAL (Anywhere in app):                              │
│   Ctrl+Shift+C    Quick Capture to inbox.org          │
│                                                         │
│ IN ORG EDITOR:                                          │
│   Ctrl+Shift+M    Refile entry at cursor (M = Move)   │
│   Ctrl+Shift+A    Archive entry at cursor             │
│   Ctrl+Shift+E    Tag entry at cursor (E = Entry)     │
│   Ctrl+Alt+L     Insert file link at cursor (L = Link)│
│   Ctrl+Shift+I    Clock in to task                    │
│   Ctrl+Shift+O    Clock out from task                 │
│                                                         │
│ IN DIALOGS:                                             │
│   Enter           Submit/Confirm                        │
│   Esc             Cancel/Close                          │
│                                                         │
│ Mac Users: Use Cmd instead of Ctrl                     │
└─────────────────────────────────────────────────────────┘
```

---

## Design Principles

1. **Consistent Modifier**: All org-mode actions use `Ctrl+Shift`
2. **Mnemonic Letters**: First letter of action (Capture, Refile, etc.)
3. **Context-Aware**: Some hotkeys only work in appropriate contexts
4. **Non-Conflicting**: Avoid conflicts with browser/system shortcuts
5. **Platform-Aware**: Automatically use Cmd on Mac

---

## Implementation Notes

**Global hotkeys** (like Capture) are registered at the App.js level:
```javascript
// In App.js
useEffect(() => {
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
      e.preventDefault();
      setQuickCaptureOpen(true);
    }
  };
  
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);
```

**Context-specific hotkeys** (like Refile) are registered in their components:
```javascript
// In DocumentViewer.js - only for .org files
useEffect(() => {
  if (!document?.filename?.endsWith('.org')) return;
  
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'M') {
      e.preventDefault();
      handleRefileAtCursor();
    }
  };
  
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [document]);
```

**By George!** A well-organized keyboard cavalry makes for efficient workflows! 🏇

