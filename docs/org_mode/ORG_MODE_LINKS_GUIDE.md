# Org-Mode Link Support - Roosevelt's Link Cavalry

**BULLY!** This guide covers org-mode link functionality in the Plato Knowledge Base!

## Link Syntax

Org-mode supports two link formats:

### Simple Links
```org
[[link-target]]
```

### Links with Description
```org
[[link-target][Friendly Description]]
```

## Supported Link Types

### 1. External URLs 🌐

**Syntax:**
- `[[https://example.com]]`
- `[[http://example.com]]`
- `[[https://example.com][Click Here]]`

**Behavior:**
- Opens in new browser tab
- Styled in blue (#1976d2)
- Underlined on hover

**Examples:**
```org
Visit [[https://orgmode.org][Org-Mode Documentation]]
Check out [[https://github.com][GitHub]]
```

### 2. File Links 📄

**Syntax:**
- `[[file:path/to/file.org]]`
- `[[file:another-document.md]]`
- `[[tasks.org]]` (plain filename)

**Behavior:**
- Styled in green (#2e7d32)
- Currently shows "coming soon" placeholder
- Will search and open matching documents (future)

**Examples:**
```org
See [[file:inbox.org][my inbox]]
Review [[file:projects/main-project.org]]
Check [[tasks.org]]
```

### 3. Internal Heading Links 🔗

**Syntax:**
- `[[#Heading Text]]`
- `[[*Heading Text]]`
- `[[Heading Text]]` (plain text)

**Behavior:**
- Styled in green (#2e7d32)
- Scrolls smoothly to matching heading
- Briefly highlights target heading
- Case-insensitive matching

**Examples:**
```org
Jump to [[#Internal Navigation]]
See [[*Important Section]]
Reference [[Project Overview]]
```

### 4. ID-Based Links 🆔

**Syntax:**
- `[[id:unique-identifier]]`
- `[[id:uuid][Description]]`

**Behavior:**
- Links to headings with matching `:ID:` property
- Styled in green (#2e7d32)
- Currently shows "coming soon" placeholder
- Full implementation requires ID indexing

**Example:**
```org
* Important Section
:PROPERTIES:
:ID: important-section-123
:END:

Later in document: See [[id:important-section-123][Important Section]]
```

## Editor Features

### Visual Decorations
Links are highlighted in the editor with:
- Blue color for easy identification
- Underline styling
- Applied to complete link syntax `[[...]]`

### Syntax Highlighting
The editor recognizes and highlights:
- Opening brackets `[[`
- Link target
- Description separator `][`
- Closing brackets `]]`

## Current Limitations

### File Navigation 📝
File links currently show a placeholder message. Full implementation requires:
- Document search by filename
- Relative path resolution
- Cross-document navigation API

### ID Navigation 🔍
ID-based links show a placeholder. Full implementation requires:
- Document ID indexing
- Cross-document ID lookup
- Property drawer parsing

## Best Practices

### 1. Use Descriptive Links
```org
✅ GOOD: Check the [[file:project-plan.org][project plan]]
❌ BAD:  Check [[file:project-plan.org]]
```

### 2. Internal Links for Navigation
```org
* Table of Contents

- [[#Introduction]]
- [[#Installation]]
- [[#Usage]]
- [[#Troubleshooting]]
```

### 3. External References
```org
* Research Sources

- [[https://arxiv.org/abs/1234][Smith et al. 2024]]
- [[https://nature.com/article][Nature Article]]
```

### 4. File Organization
```org
* Project Files

- [[file:specs.org][Technical Specifications]]
- [[file:timeline.org][Project Timeline]]
- [[file:budget.org][Budget Breakdown]]
```

## Technical Implementation

### Parsing
- Regex-based link detection: `/\[\[([^\]]+)\](?:\[([^\]]+)\])?\]/g`
- Handles nested brackets gracefully
- Preserves surrounding text

### Link Type Detection
```javascript
parseLinkTarget(target):
  - Checks for http:// or https:// → URL link
  - Checks for file: prefix → File link
  - Checks for id: prefix → ID link
  - Checks for # or * prefix → Heading link
  - Checks for file extensions (.org, .md) → File link
  - Defaults to heading link
```

### Navigation Handling
- **URL links**: `window.open(url, '_blank')`
- **Internal links**: Smooth scroll + highlight animation
- **File links**: Callback to parent component (DocumentViewer)
- **ID links**: Callback to parent component (future implementation)

## Testing

A comprehensive test file is available at:
- `/uploads/org-mode-link-test.org`

This file demonstrates:
- All link types
- Mixed content with links
- Links in lists and tables
- Edge cases and advanced examples

## Future Enhancements

**Phase 2 - File Navigation:**
- Backend API for document search by filename
- Relative path resolution based on current document
- Tab-based document navigation

**Phase 3 - Advanced Features:**
- Link completion in editor (Ctrl+Space)
- Backlinks view (what links to this document)
- Broken link detection
- Link refactoring tools

**BULLY!** Start using links today and enjoy interconnected org-mode knowledge management! 🏇


