# Org Quick Capture - Emacs-Style Capture to inbox.org

**BULLY!** Quick capture functionality like Emacs org-capture, available anywhere in the app!

## Overview

Capture notes, TODOs, journal entries, or meeting notes to `inbox.org` with a single hotkey from anywhere in the application. No need to navigate to files or open editors - just press the hotkey, type, and capture!

## How It Works

### Global Hotkey
- **Ctrl+Shift+C** (Windows/Linux) or **Cmd+Shift+C** (Mac)
- Works from any page in the application
- Opens the Quick Capture modal instantly

### Capture Templates

1. **Note** - Quick note with timestamp
2. **TODO** - Task item with optional priority, scheduling, and deadline
3. **Journal** - Journal entry with automatic date and tags
4. **Meeting** - Meeting notes with structured sections (Attendees, Notes, Action Items)

### Workflow

1. Press **Ctrl+Shift+C** anywhere in the app
2. Select a template (Note, TODO, Journal, Meeting)
3. Type your content
4. Optional: Add tags, priority, scheduled/deadline dates
5. Press **Ctrl+Enter** to capture (or click Capture button)
6. Entry is instantly appended to `inbox.org`
7. Return to what you were doing!

## Technical Implementation

### Backend Components

**API Endpoint:** `backend/api/org_capture_api.py`
- POST `/api/org/capture` - Capture content to inbox.org
- Handles authentication and user-specific inbox locations

**Service:** `backend/services/org_capture_service.py`
- Formats entries based on template type (BeOrg-compatible style)
- Smart inbox discovery - finds your existing inbox.org automatically
- Clean org-mode formatting: single-level headings, right-aligned tags, simple timestamps

**Models:** `backend/models/org_capture_models.py`
- `OrgCaptureRequest` - Capture request with content, template, tags, etc.
- `OrgCaptureResponse` - Response with success status and preview
- `OrgCaptureTemplate` - Template definitions

### Frontend Components

**Component:** `frontend/src/components/OrgQuickCapture.js`
- Modal dialog with template selection
- Auto-focus content input
- Tag management
- Advanced options for TODO items (priority, scheduled, deadline)
- Keyboard shortcuts (Ctrl+Enter to capture, Esc to cancel)

**Integration:** `frontend/src/App.js`
- Global keyboard listener for Ctrl+Shift+C
- Renders Quick Capture modal at app level (available everywhere)

## Inbox.org Location - Smart Discovery!

**BULLY!** The system automatically finds your existing inbox.org!

### How It Works

On first capture, the system:
1. **Checks your org-mode settings** for a configured inbox location
2. **Searches your entire directory** for any existing `inbox.org`
3. **Uses your existing file** if found
4. **Creates a new one** only if none exists
5. **Saves the location** to your settings for future use

### Multiple Inbox Files?

If multiple `inbox.org` files are found, the system:
- Uses the **deepest path** (e.g., `OrgMode/inbox.org` over root `inbox.org`)
- Logs a warning about duplicates
- Shows a warning in the Quick Capture modal
- You can manually configure which one to use in **Settings → Org-Mode Settings**

**NOTE:** The system prevents creating duplicate inbox.org files:
- Upload of new `inbox.org` is blocked if one already exists
- Quick capture will always find and use the existing one

### Configuring Inbox Location

You can manually set your inbox location in **Settings → Org-Mode Settings**:
- Set the path relative to your user directory
- Example: `OrgMode/inbox.org` or just `inbox.org`
- Leave blank for auto-discovery

## Entry Formats

**BULLY!** Clean BeOrg-compatible formatting - works seamlessly with BeOrg, Emacs, and all org-mode tools!

### TODO Entry
```org
* TODO Review quarterly reports                                  :work:review:
SCHEDULED: <2025-10-21>
[2025-10-20 Mon 15:30]
```

**Features:**
- Single-level heading (`*`) like BeOrg
- Right-aligned tags at column 77
- Simple timestamp (no properties drawer) **in your configured timezone**
- Optional SCHEDULED/DEADLINE on separate lines
- Priority: `* TODO [#A] High priority task`

**Timezone Support:**
- Timestamps use your configured timezone from **Settings → Personal Settings → Timezone**
- Falls back to UTC if no timezone is configured
- Ensures accurate time tracking across different locations

### Note Entry
```org
* Quick meeting summary                                               :notes:
[2025-10-20 Mon 15:30]
```

### Journal Entry
```org
* Journal Entry                                                      :journal:
[2025-10-20 Mon 15:30]

Had a productive day working on the new feature...
```

### Meeting Entry
```org
* Meeting: Q4 Planning                                              :meeting:
[2025-10-20 Mon 15:30]

** Attendees

** Notes

** Action Items
```

**Note:** All formats support any valid org-mode syntax. The system generates clean BeOrg-style entries by default, but you can manually add properties drawers, subtasks, or any other org-mode elements as needed.

## Settings Integration

**Timezone Configuration:**
Quick capture respects your timezone setting from **Settings → Personal Settings → Timezone**. This ensures all timestamps match your local time, not the server's UTC time.

To configure your timezone:
1. Navigate to **Settings** (gear icon in navigation)
2. Scroll to **Personal Settings → Timezone**
3. Select your timezone from the dropdown
4. Click **Update Timezone**

## Keyboard Shortcuts

- **Ctrl+Shift+C** - Open Quick Capture modal (global)
- **Ctrl+Enter** - Capture and save
- **Esc** - Cancel and close
- **Enter** (in tag field) - Add tag

## Future Enhancements

- Custom capture templates from user settings
- Capture to different files (not just inbox.org)
- Refile captured items to other org files
- Capture from selected text on page
- Mobile-friendly touch interface
- Voice capture integration

