const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Config management
  getConfig: () => ipcRenderer.invoke('get-config'),
  saveConfig: (config) => ipcRenderer.invoke('save-config', config),
  closeConfigWindow: () => ipcRenderer.invoke('close-config-window'),
  
  // Auto-login functionality
  autoLogin: async (username, password) => {
    try {
      // Wait for React app to be ready
      await new Promise(resolve => {
        if (document.readyState === 'complete') {
          resolve();
        } else {
          window.addEventListener('load', resolve);
        }
      });

      // Wait a bit more for React to initialize
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Try to find and fill login form
      const usernameInput = document.querySelector('input[type="text"], input[name="username"], input[placeholder*="username" i], input[autocomplete="username"]');
      const passwordInput = document.querySelector('input[type="password"], input[name="password"], input[autocomplete="current-password"]');
      const submitButton = document.querySelector('button[type="submit"], button:has-text("Sign In"), button:has-text("Login")');

      if (usernameInput && passwordInput) {
        // Fill in credentials
        usernameInput.value = username;
        passwordInput.value = password;

        // Trigger React onChange events
        const usernameEvent = new Event('input', { bubbles: true });
        const passwordEvent = new Event('input', { bubbles: true });
        usernameInput.dispatchEvent(usernameEvent);
        passwordInput.dispatchEvent(passwordEvent);

        // Try to find the form and submit it
        const form = usernameInput.closest('form');
        if (form) {
          // Wait a moment for React state to update
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // Try clicking submit button first
          if (submitButton && !submitButton.disabled) {
            submitButton.click();
          } else {
            // Fallback: submit form directly
            form.requestSubmit();
          }
        } else if (submitButton && !submitButton.disabled) {
          // No form found, but button exists - click it
          await new Promise(resolve => setTimeout(resolve, 500));
          submitButton.click();
        }
      } else {
        // Alternative: Try direct API login
        const currentUrl = window.location.href;
        const baseUrl = currentUrl.split('/').slice(0, 3).join('/');
        
        try {
          const response = await fetch(`${baseUrl}/api/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, password })
          });

          if (response.ok) {
            const data = await response.json();
            if (data.access_token) {
              localStorage.setItem('auth_token', data.access_token);
              // Reload to trigger auth state update
              window.location.reload();
            }
          }
        } catch (error) {
          console.error('Auto-login API call failed:', error);
        }
      }
    } catch (error) {
      console.error('Auto-login error:', error);
    }
  }
});

