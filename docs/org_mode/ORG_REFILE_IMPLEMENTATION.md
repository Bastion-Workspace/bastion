# Org-Mode Refile Implementation Plan

**BULLY!** Refile from OrgMode Editor and All TODOs page - Roosevelt's organized cavalry!

## Phase 1: Backend Implementation

### 1.1 Refile Service
**File:** `backend/services/org_refile_service.py`

```python
"""
Org-Mode Refile Service
Move entries between org files and headings
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

from config import settings

logger = logging.getLogger(__name__)


class OrgRefileService:
    """Service for refiling org-mode entries"""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
    
    async def get_refile_targets(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all possible refile targets (files + headings)
        
        Returns list of:
        - File roots: { type: "file", path: "...", display: "projects.org" }
        - Headings: { type: "heading", path: "...", heading: "Work Projects", display: "projects.org > Work Projects" }
        """
        from services.database_manager.database_helpers import fetch_one
        
        # Get username
        row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
        username = row['username'] if row else user_id
        
        # Find all org files
        user_base_dir = self.upload_dir / "Users" / username
        org_files = []
        
        if user_base_dir.exists():
            org_files = list(user_base_dir.rglob("*.org"))
            # Exclude archive files and hidden files
            org_files = [f for f in org_files if not f.name.startswith('.') and 'archive' not in f.name.lower()]
            org_files.sort()
        
        targets = []
        
        for org_file in org_files:
            relative_path = org_file.relative_to(user_base_dir)
            
            # Add file root as target
            targets.append({
                "id": f"file:{relative_path}",
                "type": "file",
                "path": str(relative_path),
                "display": f"{org_file.name}",
                "description": "Top level of file"
            })
            
            # Parse headings as targets
            try:
                headings = await self._parse_org_headings(org_file)
                for heading in headings:
                    targets.append({
                        "id": f"heading:{relative_path}:{heading['title']}",
                        "type": "heading",
                        "path": str(relative_path),
                        "heading": heading['title'],
                        "level": heading['level'],
                        "display": f"{org_file.name} > {heading['title']}",
                        "description": f"Level {heading['level']} heading"
                    })
            except Exception as e:
                logger.error(f"Failed to parse headings in {org_file}: {e}")
        
        logger.info(f"Found {len(targets)} refile targets for user {username}")
        return targets
    
    async def _parse_org_headings(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse org file to extract all headings"""
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        headings = []
        heading_pattern = re.compile(r'^(\*+)\s+(.+?)(?:\s+(:[a-zA-Z0-9_@:]+:))?\s*$')
        
        for line in lines:
            match = heading_pattern.match(line)
            if match:
                stars, title, tags = match.groups()
                # Remove TODO keywords from title
                title = re.sub(r'^(TODO|DONE|NEXT|WAITING|CANCELED)\s+', '', title).strip()
                
                headings.append({
                    "level": len(stars),
                    "title": title,
                    "tags": tags.strip(':').split(':') if tags else []
                })
        
        return headings
    
    async def refile_entry(
        self,
        user_id: str,
        source_file: str,
        source_line: int,
        target_file: str,
        target_heading: Optional[str] = None,
        position: str = "last"
    ) -> Dict[str, Any]:
        """
        Move an org entry from source to target
        
        Args:
            user_id: User ID
            source_file: Source file path (relative to user dir)
            source_line: Line number in source (0-indexed)
            target_file: Target file path (relative to user dir)
            target_heading: Optional heading to refile under
            position: "first" or "last" under heading
        
        Returns:
            { success: bool, message: str, entry_preview: str }
        """
        from services.database_manager.database_helpers import fetch_one
        
        try:
            # Get username
            row = await fetch_one("SELECT username FROM users WHERE user_id = $1", user_id)
            username = row['username'] if row else user_id
            
            user_base_dir = self.upload_dir / "Users" / username
            source_path = user_base_dir / source_file
            target_path = user_base_dir / target_file
            
            # Extract entry from source
            entry_lines, start_line, end_line = await self._extract_entry(source_path, source_line)
            entry_text = '\n'.join(entry_lines)
            
            logger.info(f"Refiling entry from {source_file}:{source_line} to {target_file}")
            logger.debug(f"Entry: {entry_text[:100]}...")
            
            # Remove from source
            await self._remove_entry(source_path, start_line, end_line)
            
            # Insert into target
            if target_heading:
                await self._insert_under_heading(target_path, target_heading, entry_text, position)
            else:
                await self._append_to_file(target_path, entry_text)
            
            # Get preview (first line)
            preview = entry_lines[0] if entry_lines else ""
            
            logger.info(f"✅ Successfully refiled entry to {target_file}")
            
            return {
                "success": True,
                "message": f"Refiled to {target_file}" + (f" > {target_heading}" if target_heading else ""),
                "entry_preview": preview,
                "source_file": source_file,
                "target_file": target_file
            }
            
        except Exception as e:
            logger.error(f"❌ Refile failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": f"Refile failed: {str(e)}",
                "entry_preview": None
            }
    
    async def _extract_entry(self, file_path: Path, line_number: int) -> tuple:
        """
        Extract an org entry starting at line_number
        Returns (entry_lines, start_line, end_line)
        """
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        if line_number >= len(lines):
            raise ValueError(f"Line number {line_number} out of range")
        
        # Find the heading level at start line
        start_line = line_number
        start_heading_match = re.match(r'^(\*+)\s+', lines[start_line])
        
        if not start_heading_match:
            raise ValueError(f"Line {line_number} is not a heading")
        
        start_level = len(start_heading_match.group(1))
        
        # Find end of entry (next heading at same or higher level)
        end_line = start_line + 1
        while end_line < len(lines):
            line = lines[end_line]
            heading_match = re.match(r'^(\*+)\s+', line)
            
            if heading_match:
                heading_level = len(heading_match.group(1))
                if heading_level <= start_level:
                    break
            
            end_line += 1
        
        # Extract entry lines
        entry_lines = lines[start_line:end_line]
        
        return entry_lines, start_line, end_line
    
    async def _remove_entry(self, file_path: Path, start_line: int, end_line: int):
        """Remove lines from start_line to end_line (exclusive)"""
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # Remove the lines
        new_lines = lines[:start_line] + lines[end_line:]
        
        # Write back
        file_path.write_text('\n'.join(new_lines), encoding='utf-8')
        logger.info(f"Removed lines {start_line}-{end_line} from {file_path.name}")
    
    async def _append_to_file(self, file_path: Path, entry: str):
        """Append entry to end of file"""
        content = file_path.read_text(encoding='utf-8')
        
        # Ensure file ends with newline
        if content and not content.endswith('\n'):
            content += '\n'
        
        # Add blank line before entry if file has content
        if content.strip():
            content += '\n'
        
        content += entry + '\n'
        
        file_path.write_text(content, encoding='utf-8')
        logger.info(f"Appended entry to {file_path.name}")
    
    async def _insert_under_heading(
        self,
        file_path: Path,
        heading: str,
        entry: str,
        position: str = "last"
    ):
        """Insert entry under a specific heading"""
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # Find the heading
        heading_line = None
        heading_level = None
        
        for i, line in enumerate(lines):
            # Match heading (with or without TODO keywords)
            if re.match(rf'^\*+\s+(TODO|DONE|NEXT|WAITING|CANCELED\s+)?{re.escape(heading)}\s*(:.*)?$', line):
                heading_line = i
                heading_match = re.match(r'^(\*+)\s+', line)
                heading_level = len(heading_match.group(1))
                break
        
        if heading_line is None:
            raise ValueError(f"Heading '{heading}' not found in {file_path.name}")
        
        # Find where to insert
        insert_line = heading_line + 1
        
        if position == "last":
            # Find end of section (next heading at same or higher level)
            while insert_line < len(lines):
                line = lines[insert_line]
                heading_match = re.match(r'^(\*+)\s+', line)
                
                if heading_match:
                    level = len(heading_match.group(1))
                    if level <= heading_level:
                        break
                
                insert_line += 1
        # else position == "first", insert right after heading
        
        # Increase indent level of entry to be one level deeper than heading
        entry_lines = entry.split('\n')
        indented_lines = []
        for line in entry_lines:
            if line.strip().startswith('*'):
                indented_lines.append('*' + line)
            else:
                indented_lines.append(line)
        
        # Insert entry
        new_lines = (
            lines[:insert_line] +
            [''] +  # Blank line before entry
            indented_lines +
            [''] +  # Blank line after entry
            lines[insert_line:]
        )
        
        file_path.write_text('\n'.join(new_lines), encoding='utf-8')
        logger.info(f"Inserted entry under '{heading}' at position {position} in {file_path.name}")
    
    async def suggest_refile_target(
        self,
        user_id: str,
        entry_text: str,
        source_file: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use AI to suggest best refile target based on entry content/tags
        """
        from services.chat_service import ChatService
        
        try:
            # Get available targets
            targets = await self.get_refile_targets(user_id)
            
            # Build target list for LLM
            target_list = '\n'.join([f"- {t['display']}" for t in targets])
            
            # Ask LLM to suggest
            chat_service = ChatService()
            await chat_service.initialize()
            
            prompt = f"""
Based on this org-mode entry, suggest the BEST refile target from the list below.

ENTRY:
{entry_text}

AVAILABLE TARGETS:
{target_list}

Analyze the entry's:
- TODO keywords and priority
- Tags (words starting with @ or :)
- Content and context

Respond with ONLY a JSON object:
{{
  "target": "exact target display name from list",
  "reason": "brief explanation why this target is best"
}}
"""
            
            response = await chat_service.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            suggestion_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            suggestion = json.loads(suggestion_text)
            
            # Find matching target
            matching_target = next(
                (t for t in targets if t['display'] == suggestion['target']),
                None
            )
            
            if matching_target:
                return {
                    **matching_target,
                    "reason": suggestion['reason']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to suggest refile target: {e}")
            return None


# Singleton
_org_refile_service: Optional[OrgRefileService] = None

async def get_org_refile_service() -> OrgRefileService:
    """Get or create org refile service instance"""
    global _org_refile_service
    
    if _org_refile_service is None:
        _org_refile_service = OrgRefileService()
        logger.info("🗂️ BULLY! Org Refile Service initialized!")
    
    return _org_refile_service
```

### 1.2 API Endpoints
**File:** `backend/api/org_refile_api.py`

```python
"""
Org-Mode Refile API
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException

from models.auth import AuthenticatedUserResponse
from services.auth_service import get_current_user
from services.org_refile_service import get_org_refile_service
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/org", tags=["org-refile"])


class RefileRequest(BaseModel):
    """Request to refile an org entry"""
    source_file: str = Field(..., description="Source file path (relative to user dir)")
    source_line: int = Field(..., description="Line number in source (0-indexed)")
    target_file: str = Field(..., description="Target file path (relative to user dir)")
    target_heading: str | None = Field(None, description="Optional heading to refile under")
    position: str = Field("last", description="Position under heading: 'first' or 'last'")


class SuggestRefileRequest(BaseModel):
    """Request AI suggestion for refile target"""
    entry_text: str = Field(..., description="The org entry to refile")
    source_file: str = Field(..., description="Source file path")


@router.get("/refile-targets")
async def get_refile_targets(
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all available refile targets (files + headings)
    
    **BULLY!** Auto-discovers all org files and their structure!
    """
    try:
        refile_service = await get_org_refile_service()
        targets = await refile_service.get_refile_targets(current_user.user_id)
        
        return {
            "success": True,
            "targets": targets,
            "count": len(targets)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get refile targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refile")
async def refile_entry(
    request: RefileRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Refile an org entry from source to target
    
    **By George!** Move that task to its proper location!
    """
    try:
        refile_service = await get_org_refile_service()
        result = await refile_service.refile_entry(
            user_id=current_user.user_id,
            source_file=request.source_file,
            source_line=request.source_line,
            target_file=request.target_file,
            target_heading=request.target_heading,
            position=request.position
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Refile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest-refile")
async def suggest_refile_target(
    request: SuggestRefileRequest,
    current_user: AuthenticatedUserResponse = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get AI suggestion for best refile target
    
    **BULLY!** AI analyzes content and suggests the best home!
    """
    try:
        refile_service = await get_org_refile_service()
        suggestion = await refile_service.suggest_refile_target(
            user_id=current_user.user_id,
            entry_text=request.entry_text,
            source_file=request.source_file
        )
        
        if suggestion:
            return {
                "success": True,
                "suggestion": suggestion
            }
        else:
            return {
                "success": False,
                "message": "No suggestion available"
            }
        
    except Exception as e:
        logger.error(f"❌ Failed to suggest refile target: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 1.3 Register Router
**File:** `backend/main.py`

```python
# Add to imports
from api.org_refile_api import router as org_refile_router

# Add to router registration
app.include_router(org_refile_router)
```

## Phase 2: Frontend - Refile Dialog Component

### 2.1 Refile Dialog
**File:** `frontend/src/components/OrgRefileDialog.js`

```javascript
import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  RadioGroup,
  FormControlLabel,
  Radio,
  Box,
  Typography,
  Alert,
  CircularProgress,
  Divider,
  Chip
} from '@mui/material';
import { MoveToInbox, AutoAwesome } from '@mui/icons-material';
import apiService from '../services/apiService';

const OrgRefileDialog = ({ 
  open, 
  onClose, 
  entry,           // The TODO/heading text to refile
  sourceFile,      // Current file (e.g., "OrgMode/inbox.org")
  sourceLine,      // Line number in source (0-indexed)
  onRefileSuccess 
}) => {
  const [targets, setTargets] = useState([]);
  const [filteredTargets, setFilteredTargets] = useState([]);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [aiSuggestion, setAiSuggestion] = useState(null);
  const [position, setPosition] = useState('last');
  const [loading, setLoading] = useState(false);
  const [refiling, setRefiling] = useState(false);
  const [error, setError] = useState(null);

  // Fetch targets and AI suggestion when dialog opens
  useEffect(() => {
    if (open) {
      fetchRefileTargets();
      fetchAiSuggestion();
      setSearchQuery('');
      setSelectedTarget(null);
      setError(null);
    }
  }, [open]);

  // Filter targets by search query
  useEffect(() => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      setFilteredTargets(
        targets.filter(t => t.display.toLowerCase().includes(query))
      );
    } else {
      setFilteredTargets(targets);
    }
  }, [searchQuery, targets]);

  const fetchRefileTargets = async () => {
    setLoading(true);
    try {
      const response = await apiService.get('/api/org/refile-targets');
      setTargets(response.targets || []);
    } catch (err) {
      console.error('Failed to fetch refile targets:', err);
      setError('Failed to load refile targets');
    } finally {
      setLoading(false);
    }
  };

  const fetchAiSuggestion = async () => {
    if (!entry) return;
    
    try {
      const response = await apiService.post('/api/org/suggest-refile', {
        entry_text: entry,
        source_file: sourceFile
      });
      
      if (response.success && response.suggestion) {
        setAiSuggestion(response.suggestion);
      }
    } catch (err) {
      console.error('Failed to get AI suggestion:', err);
      // Non-critical, don't show error to user
    }
  };

  const handleUseSuggestion = () => {
    if (aiSuggestion) {
      setSelectedTarget(aiSuggestion);
      setSearchQuery(''); // Clear search to show all targets
    }
  };

  const handleRefile = async () => {
    if (!selectedTarget) return;
    
    setRefiling(true);
    setError(null);
    
    try {
      const result = await apiService.post('/api/org/refile', {
        source_file: sourceFile,
        source_line: sourceLine,
        target_file: selectedTarget.path,
        target_heading: selectedTarget.heading || null,
        position: position
      });
      
      if (result.success) {
        // Success! Notify parent and close
        onRefileSuccess && onRefileSuccess(result);
        onClose();
      } else {
        setError(result.message || 'Refile failed');
      }
    } catch (err) {
      console.error('Refile failed:', err);
      setError(err.response?.data?.detail || 'Failed to refile entry');
    } finally {
      setRefiling(false);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    if (!open) return;
    
    const handleKeyDown = (e) => {
      // Enter to refile
      if (e.key === 'Enter' && !e.shiftKey && selectedTarget) {
        e.preventDefault();
        handleRefile();
      }
      // Escape to cancel
      if (e.key === 'Escape') {
        onClose();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, selectedTarget]);

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: { minHeight: '600px' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <MoveToInbox />
          <Typography variant="h6">Refile Entry</Typography>
        </Box>
      </DialogTitle>

      <DialogContent>
        {/* Entry Preview */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Entry to refile:
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              fontFamily: 'monospace', 
              whiteSpace: 'pre-wrap',
              mt: 1
            }}
          >
            {entry}
          </Typography>
        </Box>

        {/* AI Suggestion */}
        {aiSuggestion && (
          <Alert 
            severity="info" 
            icon={<AutoAwesome />}
            action={
              <Button 
                size="small" 
                onClick={handleUseSuggestion}
                disabled={selectedTarget?.id === aiSuggestion.id}
              >
                Use Suggestion
              </Button>
            }
            sx={{ mb: 2 }}
          >
            <Typography variant="subtitle2">
              AI suggests: <strong>{aiSuggestion.display}</strong>
            </Typography>
            <Typography variant="caption" display="block">
              {aiSuggestion.reason}
            </Typography>
          </Alert>
        )}

        {/* Error */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Search */}
        <TextField
          fullWidth
          placeholder="🔍 Search targets... (type to filter)"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          sx={{ mb: 2 }}
          autoFocus
        />

        {/* Loading */}
        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {/* Target List */}
        {!loading && (
          <Box sx={{ maxHeight: 300, overflow: 'auto', mb: 2 }}>
            {filteredTargets.length === 0 ? (
              <Typography variant="body2" color="text.secondary" align="center">
                No targets found matching "{searchQuery}"
              </Typography>
            ) : (
              <RadioGroup 
                value={selectedTarget?.id || ''} 
                onChange={(e) => {
                  const target = targets.find(t => t.id === e.target.value);
                  setSelectedTarget(target);
                }}
              >
                {filteredTargets.map(target => (
                  <FormControlLabel
                    key={target.id}
                    value={target.id}
                    control={<Radio />}
                    label={
                      <Box>
                        <Typography variant="body2">
                          {target.display}
                        </Typography>
                        {target.description && (
                          <Typography variant="caption" color="text.secondary">
                            {target.description}
                          </Typography>
                        )}
                      </Box>
                    }
                    sx={{ 
                      py: 1,
                      borderBottom: '1px solid',
                      borderColor: 'divider'
                    }}
                  />
                ))}
              </RadioGroup>
            )}
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        {/* Position Selection */}
        {selectedTarget?.type === 'heading' && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Position under heading:
            </Typography>
            <RadioGroup 
              row 
              value={position} 
              onChange={(e) => setPosition(e.target.value)}
            >
              <FormControlLabel value="first" control={<Radio />} label="First (top)" />
              <FormControlLabel value="last" control={<Radio />} label="Last (bottom)" />
            </RadioGroup>
          </Box>
        )}

        {/* Keyboard hints */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            💡 Press <Chip label="Enter" size="small" /> to refile • <Chip label="Esc" size="small" /> to cancel
          </Typography>
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={refiling}>
          Cancel
        </Button>
        <Button 
          variant="contained" 
          onClick={handleRefile}
          disabled={!selectedTarget || refiling}
          startIcon={refiling ? <CircularProgress size={20} /> : <MoveToInbox />}
        >
          {refiling ? 'Refiling...' : 'Refile Entry'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default OrgRefileDialog;
```

## Phase 3: Integration Points

### 3.1 DocumentViewer (OrgMode Editor)
**File:** `frontend/src/components/DocumentViewer.js`

Add refile support for .org files:

```javascript
import OrgRefileDialog from './OrgRefileDialog';

// Add state
const [refileDialogOpen, setRefileDialogOpen] = useState(false);
const [refileEntry, setRefileEntry] = useState(null);

// Add keyboard shortcut listener (Method 2!)
useEffect(() => {
  if (!document || !document.filename?.endsWith('.org')) return;
  
  const handleKeyDown = (e) => {
    // Ctrl+Shift+M or Cmd+Shift+M to refile (M = Move, avoids Ctrl+Shift+R browser refresh)
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'M') {
      e.preventDefault();
      handleRefileAtCursor();
    }
  };
  
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, [document]);

// Extract entry at cursor position
const handleRefileAtCursor = () => {
  if (!content) return;
  
  // Get cursor position from Monaco editor or textarea
  // For now, simplified version - in practice, get from editor API
  const lines = content.split('\n');
  
  // Find current line with cursor (this is simplified)
  // In real implementation, get from Monaco editor's getCursorPosition()
  const cursorLine = 0; // Placeholder - get from editor
  
  // Find the TODO/heading at or before cursor
  let entryLine = cursorLine;
  while (entryLine >= 0) {
    const line = lines[entryLine];
    if (line.match(/^\*+\s+(TODO|DONE|NEXT|WAITING|CANCELED)?/)) {
      // Found a heading!
      const entryText = extractEntryFromLine(lines, entryLine);
      setRefileEntry({
        text: entryText,
        line: entryLine
      });
      setRefileDialogOpen(true);
      break;
    }
    entryLine--;
  }
};

// Extract full entry (heading + content until next heading)
const extractEntryFromLine = (lines, startLine) => {
  const startMatch = lines[startLine].match(/^(\*+)\s+/);
  if (!startMatch) return lines[startLine];
  
  const startLevel = startMatch[1].length;
  let endLine = startLine + 1;
  
  // Find next heading at same or higher level
  while (endLine < lines.length) {
    const match = lines[endLine].match(/^(\*+)\s+/);
    if (match && match[1].length <= startLevel) {
      break;
    }
    endLine++;
  }
  
  return lines.slice(startLine, endLine).join('\n');
};

// Add refile dialog
<OrgRefileDialog
  open={refileDialogOpen}
  onClose={() => setRefileDialogOpen(false)}
  entry={refileEntry?.text}
  sourceFile={document?.filename}
  sourceLine={refileEntry?.line}
  onRefileSuccess={() => {
    // Refresh document
    fetchDocument();
    // Show success message
    setSnackbar({
      open: true,
      message: 'Entry refiled successfully!',
      severity: 'success'
    });
  }}
/>
```

### 3.2 All TODOs Page
**File:** `frontend/src/components/AllTodosPage.js`

Add refile button to each TODO:

```javascript
import OrgRefileDialog from './OrgRefileDialog';
import { MoveToInbox } from '@mui/icons-material';

// Add state
const [refileDialogOpen, setRefileDialogOpen] = useState(false);
const [refileItem, setRefileItem] = useState(null);

// In TODO item rendering
<ListItem>
  <ListItemText
    primary={todo.heading}
    secondary={`${todo.filename} • ${todo.todo_state}`}
  />
  <ListItemSecondaryAction>
    <IconButton 
      edge="end" 
      onClick={() => {
        setRefileItem({
          text: todo.entry_text, // Full entry text
          file: todo.file_path,
          line: todo.line_number
        });
        setRefileDialogOpen(true);
      }}
      title="Refile this TODO"
    >
      <MoveToInbox />
    </IconButton>
  </ListItemSecondaryAction>
</ListItem>

// Add refile dialog
<OrgRefileDialog
  open={refileDialogOpen}
  onClose={() => setRefileDialogOpen(false)}
  entry={refileItem?.text}
  sourceFile={refileItem?.file}
  sourceLine={refileItem?.line}
  onRefileSuccess={() => {
    // Refresh TODO list
    fetchTodos();
    setRefileDialogOpen(false);
    // Show success message
    setSnackbar({
      open: true,
      message: 'TODO refiled successfully!',
      severity: 'success'
    });
  }}
/>
```

## Summary: User Experience

### From OrgMode Editor (Method 2!)
1. User viewing `inbox.org` in DocumentViewer
2. Cursor on a TODO line
3. Press **`Ctrl+Shift+M`** (or `Cmd+Shift+M` on Mac) - M for Move!
4. Refile dialog opens with that TODO pre-loaded
5. 🤖 AI suggests best target
6. Type to search/filter targets
7. Select target, press **`Enter`** to refile
8. Entry moved! Document auto-refreshes

**Note:** We use M instead of R to avoid conflicting with the browser's hard refresh (Ctrl+Shift+R).

### From All TODOs Page
1. User viewing "All TODOs" page
2. See list of all TODOs across org files
3. Click **📋 Refile** icon next to any TODO
4. Same refile dialog experience
5. After refile, TODO list refreshes

## Implementation Phases

**Phase 1: Backend** (Day 1)
- ✅ OrgRefileService
- ✅ API endpoints
- ✅ Register router

**Phase 2: Frontend Dialog** (Day 2)
- ✅ OrgRefileDialog component
- ✅ AI suggestion integration
- ✅ Search and keyboard shortcuts

**Phase 3: Integration** (Day 3)
- ✅ DocumentViewer keyboard shortcut
- ✅ All TODOs page refile buttons
- ✅ Testing and refinement

**By George!** The refile cavalry is ready to charge! 🏇

