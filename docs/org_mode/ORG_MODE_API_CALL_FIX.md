# Org Mode API Call Bug Fix

**Date:** October 20, 2025  
**Issue:** API calls returning HTML instead of JSON  
**Status:** ✅ FIXED

---

## 🐛 **The Bug**

**Error in Browser Console:**
```
❌ Agenda error: SyntaxError: Unexpected token '<', "<!doctype "... is not valid JSON
```

**Root Cause:**
```javascript
// ❌ WRONG - Parameters reversed!
apiService.request('GET', '/api/org/agenda')
```

The API request was being made with the URL as `'GET'` and method as the second parameter. This resulted in:
- Request URL: `'GET'` (invalid)
- Server returned 404 HTML page
- Frontend tried to parse HTML as JSON
- Error: `"<!doctype "... is not valid JSON`

---

## ✅ **The Fix**

Changed all three Org view components to use the correct API call syntax:

```javascript
// ✅ CORRECT - Use apiService.get()
apiService.get('/api/org/agenda')
```

---

## 📝 **Files Modified**

1. **frontend/src/components/OrgAgendaView.js**
   ```javascript
   // Before:
   const response = await apiService.request('GET', `/api/org/agenda?days_ahead=${days}`);
   
   // After:
   const response = await apiService.get(`/api/org/agenda?days_ahead=${days}`);
   ```

2. **frontend/src/components/OrgSearchView.js**
   ```javascript
   // Before:
   const response = await apiService.request('GET', `/api/org/search?query=${encodeURIComponent(query)}`);
   
   // After:
   const response = await apiService.get(`/api/org/search?query=${encodeURIComponent(query)}`);
   ```

3. **frontend/src/components/OrgTodosView.js**
   ```javascript
   // Before:
   const response = await apiService.request('GET', url);
   
   // After:
   const response = await apiService.get(url);
   ```

---

## 🧪 **Testing**

After fix:
```bash
docker compose up --build
```

Then:
1. ✅ Click "Agenda View" - No error
2. ✅ Click "Search Org Files" - No error
3. ✅ Click "All TODOs" - No error
4. ✅ All views load data correctly

---

## 📚 **Lesson Learned**

**ApiService Method Signatures:**
```javascript
// Correct usage:
apiService.get(url, options)       // GET request
apiService.post(url, data, options) // POST request
apiService.put(url, data, options)  // PUT request
apiService.delete(url, options)     // DELETE request

// Generic request:
apiService.request(url, { method: 'GET', ...options })
```

**Never:**
```javascript
apiService.request('GET', url)  // ❌ WRONG ORDER!
```

---

**BULLY!** Bug squashed like a well-aimed cavalry charge! 🏇

