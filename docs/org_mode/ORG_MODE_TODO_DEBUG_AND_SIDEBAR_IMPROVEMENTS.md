# Org-Mode TODO Debug & Sidebar Improvements

**Date:** October 20, 2025  
**BULLY!** Debugging in progress and sidebar enhanced! 🏇

---

## 🔍 **Issue 1: TODO List Debug Logging**

### **Problem:**
"All TODOs" view returns no results despite org files containing TODO items.

**Logs show:**
```
📂 Found 1 org files for user admin
🔍 ROOSEVELT: Searching 1 org files for ''
INFO: "GET /api/org/todos?states=TODO%2CNEXT%2CSTARTED%2CWAITING%2CHOLD HTTP/1.1" 200 OK
```

### **Debug Logging Added:**
**File:** `backend/services/org_search_service.py`

**New logging:**
1. **After parsing:** Shows how many headings were found
2. **TODO states found:** Lists all TODO states detected
3. **Filter stage:** Shows which items are filtered out and why
4. **Item inclusion:** Shows which items pass filters
5. **Final results:** Shows count of results returned

**Added lines:**
```python
# After parsing
logger.info(f"📋 Parsed {len(headings)} headings from {file_path.name}")

# Log TODO states found
todo_headings = [h for h in headings if h.get('todo_state')]
if todo_headings:
    logger.info(f"✅ Found {len(todo_headings)} headings with TODO states: {[h.get('todo_state') for h in todo_headings]}")

# During filtering
logger.debug(f"🔍 Filtered out '{heading['heading']}' - TODO state '{heading.get('todo_state')}' not in {todo_states}")

# When including
logger.debug(f"✅ Including '{heading['heading']}' with TODO state '{heading.get('todo_state')}'")

# Final results
logger.info(f"📊 Returning {len(results)} results from {file_path.name}")
```

### **Expected Debug Output:**
When you run `docker compose up --build` and click "All TODOs", you should now see:
```
📋 Parsed 4 headings from inbox.org
✅ Found 3 headings with TODO states: ['TODO', 'DONE', 'DONE']
🔍 Filtered out 'Tie up corn shocks' - TODO state 'DONE' not in ['TODO', 'NEXT', 'STARTED', 'WAITING', 'HOLD']
🔍 Filtered out 'Get Miranda another bag of food' - TODO state 'DONE' not in ['TODO', 'NEXT', 'STARTED', 'WAITING', 'HOLD']
✅ Including 'Put Swimming Pool away' with TODO state 'TODO'
📊 Returning 1 results from inbox.org
```

This will help us identify:
- Are headings being parsed correctly?
- Are TODO states being detected?
- Why are items being filtered out?
- What's the final result count?

---

## 🎨 **Issue 2: Org Tools Sidebar Improvements**

### **Requirements:**
1. **Collapsible** - Org Tools section should expand/collapse
2. **Conditional visibility** - Only show if `.org` files exist in the document tree
3. **Persistent state** - Remember collapsed/expanded state

### **Implementation:**

#### **1. Added State Management**
**File:** `frontend/src/components/FileTreeSidebar.js`

```javascript
const [hasOrgFiles, setHasOrgFiles] = useState(false);
const [orgToolsExpanded, setOrgToolsExpanded] = useState(() => {
  const saved = localStorage.getItem('orgToolsExpanded');
  return saved !== null ? JSON.parse(saved) : true;
});
```

#### **2. Check for Org Files**
Added `useEffect` to check all documents for `.org` files:

```javascript
useEffect(() => {
  const checkForOrgFiles = () => {
    // Check root documents
    if (rootDocuments && Array.isArray(rootDocuments)) {
      const hasOrg = rootDocuments.some(doc => doc.filename?.toLowerCase().endsWith('.org'));
      if (hasOrg) {
        setHasOrgFiles(true);
        return;
      }
    }
    
    // Check folder contents
    for (const contents of Object.values(folderContents)) {
      if (contents?.documents && Array.isArray(contents.documents)) {
        const hasOrg = contents.documents.some(doc => doc.filename?.toLowerCase().endsWith('.org'));
        if (hasOrg) {
          setHasOrgFiles(true);
          return;
        }
      }
    }
    
    setHasOrgFiles(false);
  };
  
  checkForOrgFiles();
}, [rootDocuments, folderContents]);
```

**What this does:**
- Checks root documents (files not in folders)
- Checks all folder contents
- Sets `hasOrgFiles` to `true` if ANY `.org` file is found
- Runs whenever documents or folders change

#### **3. Persist Expanded State**
```javascript
useEffect(() => {
  localStorage.setItem('orgToolsExpanded', JSON.stringify(orgToolsExpanded));
}, [orgToolsExpanded]);
```

#### **4. Updated UI Structure**
**Before:**
```jsx
<Divider sx={{ my: 2 }} />
<Box sx={{ px: 2 }}>
  <Typography>🔧 Org Tools</Typography>
  <List dense>
    {/* Org tools buttons */}
  </List>
</Box>
```

**After:**
```jsx
{hasOrgFiles && (
  <>
    <Divider sx={{ my: 2 }} />
    <Box sx={{ px: 2 }}>
      <Box 
        sx={{ /* clickable header styles */ }}
        onClick={() => setOrgToolsExpanded(!orgToolsExpanded)}
      >
        {orgToolsExpanded ? <ExpandLess /> : <ExpandMore />}
        <Typography>🔧 Org Tools</Typography>
      </Box>
      <Collapse in={orgToolsExpanded} timeout="auto">
        <List dense>
          {/* Org tools buttons */}
        </List>
      </Collapse>
    </Box>
  </>
)}
```

**Features:**
- ✅ **Conditional rendering:** Only shows if `hasOrgFiles` is true
- ✅ **Collapsible header:** Click to expand/collapse
- ✅ **Visual indicator:** Arrow icon shows state
- ✅ **Smooth animation:** Material-UI Collapse component
- ✅ **Hover effect:** Highlights header on hover
- ✅ **Persistent:** State saved to localStorage

---

## 🎯 **Testing**

### **Test 1: Debug Logging (TODO list)**
```bash
docker compose up --build
```

1. Watch the backend logs
2. Click "All TODOs" in Org Tools
3. ✅ Should see detailed parsing logs
4. ✅ Should see which TODO states were found
5. ✅ Should see which items were filtered out
6. ✅ Should see final result count

**If TODOs still don't appear**, the logs will tell us why!

### **Test 2: Org Tools Visibility**
```bash
# Without org files
1. Start with NO .org files
2. ✅ Org Tools section should be HIDDEN
3. Upload an .org file
4. ✅ Org Tools section should APPEAR

# With org files
1. Start with .org files present
2. ✅ Org Tools section should be VISIBLE
3. Delete all .org files
4. ✅ Org Tools section should DISAPPEAR
```

### **Test 3: Org Tools Collapse**
```bash
1. Click "All TODOs" (section is visible)
2. Click "🔧 Org Tools" header
3. ✅ Section should collapse (tools hide)
4. ✅ Arrow icon changes to ExpandMore
5. Click header again
6. ✅ Section should expand (tools show)
7. ✅ Arrow icon changes to ExpandLess
8. Refresh page
9. ✅ State should persist
```

---

## 📊 **Files Modified**

### **Backend**
1. `backend/services/org_search_service.py`
   - Added comprehensive debug logging
   - 5 new log statements for debugging
   - Shows parsing, filtering, and result details

### **Frontend**
2. `frontend/src/components/FileTreeSidebar.js`
   - Added `hasOrgFiles` state
   - Added `orgToolsExpanded` state with localStorage
   - Added `useEffect` to check for org files
   - Added `useEffect` to persist expanded state
   - Updated Org Tools UI to be collapsible
   - Wrapped Org Tools in conditional render

---

## 🎨 **UI Changes**

### **Org Tools Header**
**Before:** Static text
```
🔧 ORG TOOLS
```

**After:** Clickable with icon
```
▼ 🔧 ORG TOOLS  (expanded)
▶ 🔧 ORG TOOLS  (collapsed)
```

### **Visibility Logic**
```
No org files → Section hidden
Has org files → Section visible
  ├─ Collapsed → Only header visible
  └─ Expanded → Header + all tools visible
```

---

## 🚀 **Next Steps**

### **After Debug Logs:**
Once we see the debug output, we'll know:
1. Are headings being parsed? (Line count)
2. Are TODO states detected? (State list)
3. Why items are filtered? (Filter logs)
4. What's being returned? (Result count)

### **Possible Issues to Watch For:**
1. **No headings parsed** → File reading issue
2. **No TODO states detected** → Regex not matching
3. **Items filtered incorrectly** → Filter logic issue
4. **Results not returned** → API serialization issue

---

## ✅ **Status**

✅ **Debug logging added** - Ready to diagnose TODO list issue  
✅ **Sidebar improvements** - Org Tools now collapsible and conditional  
✅ **No linter errors**  
✅ **Ready for testing**

---

**Trust busting for mysterious bugs and cluttered sidebars!** The cavalry charges with better debugging and cleaner UX! 💪

