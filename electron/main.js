const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const keytar = require('keytar');

let mainWindow = null;
let configWindow = null;

const CONFIG_DIR = path.join(app.getPath('appData'), 'Bastion');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');
const KEYCHAIN_SERVICE = 'Bastion';

// Ensure config directory exists
function ensureConfigDir() {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
}

// Load configuration (server URL from file, credentials from keychain)
async function loadConfig() {
  try {
    let config = null;
    
    // Load server URL from config file
    if (fs.existsSync(CONFIG_FILE)) {
      const configData = fs.readFileSync(CONFIG_FILE, 'utf8');
      const fileConfig = JSON.parse(configData);
      config = {
        serverUrl: fileConfig.serverUrl
      };
    }
    
    // Load credentials from OS keychain
    if (config && config.serverUrl) {
      try {
        // Get all credentials for this service
        const credentials = await keytar.findCredentials(KEYCHAIN_SERVICE);
        
        // Use the first credential found (or most recent)
        if (credentials && credentials.length > 0) {
          // For now, use the first credential
          // In the future, could support multiple accounts
          const credential = credentials[0];
          config.username = credential.account;
          config.password = credential.password;
        }
      } catch (error) {
        console.error('Error loading credentials from keychain:', error);
        // Continue without credentials - user will need to re-enter
      }
    }
    
    return config;
  } catch (error) {
    console.error('Error loading config:', error);
    return null;
  }
}

// Save configuration (server URL to file, credentials to keychain)
async function saveConfig(config) {
  try {
    ensureConfigDir();
    
    // Save server URL to config file (not sensitive)
    const fileConfig = {
      serverUrl: config.serverUrl
    };
    fs.writeFileSync(CONFIG_FILE, JSON.stringify(fileConfig, null, 2), 'utf8');
    
    // Set file permissions to user-only (Windows)
    if (process.platform === 'win32') {
      fs.chmodSync(CONFIG_FILE, 0o600);
    }
    
    // Save credentials to OS keychain
    if (config.username && config.password) {
      try {
        // Delete existing credentials for this username (if any)
        const existingCredentials = await keytar.findCredentials(KEYCHAIN_SERVICE);
        for (const cred of existingCredentials) {
          if (cred.account === config.username) {
            await keytar.deletePassword(KEYCHAIN_SERVICE, config.username);
          }
        }
        
        // Store new credentials in keychain
        await keytar.setPassword(KEYCHAIN_SERVICE, config.username, config.password);
      } catch (error) {
        console.error('Error saving credentials to keychain:', error);
        // Continue - at least server URL is saved
      }
    }
    
    return true;
  } catch (error) {
    console.error('Error saving config:', error);
    return false;
  }
}

// Create main application window
async function createMainWindow(config) {
  if (mainWindow) {
    mainWindow.focus();
    return;
  }

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true
    },
    icon: path.join(__dirname, 'assets', 'icon.ico')
  });

  // Load the hosted frontend URL
  const serverUrl = config.serverUrl || 'http://localhost:3051';
  mainWindow.loadURL(serverUrl);

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle navigation to login page - auto-login if credentials available
  mainWindow.webContents.on('did-finish-load', async () => {
    const url = mainWindow.webContents.getURL();
    if (url.includes('/login')) {
      // Load credentials from keychain
      try {
        const credentials = await keytar.findCredentials(KEYCHAIN_SERVICE);
        if (credentials && credentials.length > 0) {
          const credential = credentials[0];
          
          // Escape credentials properly for JavaScript string injection
          // Replace backslashes first, then single quotes
          const escapedUsername = credential.account.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
          const escapedPassword = credential.password.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
          
          // Inject credentials via preload script
          mainWindow.webContents.executeJavaScript(`
            window.electronAPI.autoLogin('${escapedUsername}', '${escapedPassword}');
          `).catch(err => console.error('Auto-login error:', err));
        }
      } catch (error) {
        console.error('Error loading credentials for auto-login:', error);
      }
    }
  });
}

// Create configuration window
function createConfigWindow() {
  if (configWindow) {
    configWindow.focus();
    return;
  }

  configWindow = new BrowserWindow({
    width: 500,
    height: 400,
    resizable: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    parent: mainWindow,
    modal: true,
    icon: path.join(__dirname, 'assets', 'icon.ico')
  });

  configWindow.loadFile(path.join(__dirname, 'renderer', 'config.html'));

  configWindow.on('closed', () => {
    configWindow = null;
  });
}

// Create application menu
function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Settings',
          accelerator: 'CmdOrCtrl+,',
          click: () => {
            createConfigWindow();
          }
        },
        { type: 'separator' },
        {
          label: 'Exit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload', label: 'Reload' },
        { role: 'forceReload', label: 'Force Reload' },
        { role: 'toggleDevTools', label: 'Toggle Developer Tools' },
        { type: 'separator' },
        { role: 'resetZoom', label: 'Actual Size' },
        { role: 'zoomIn', label: 'Zoom In' },
        { role: 'zoomOut', label: 'Zoom Out' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: 'Toggle Fullscreen' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'About',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: 'About Bastion',
              message: 'Bastion Desktop Application',
              detail: `Version ${app.getVersion()}\n\nBastion AI Workspace Desktop Client`
            });
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// IPC Handlers
ipcMain.handle('get-config', async () => {
  return await loadConfig();
});

ipcMain.handle('save-config', async (event, config) => {
  const success = await saveConfig(config);
  if (success && mainWindow) {
    // Reload main window with new config
    const serverUrl = config.serverUrl || 'http://localhost:3051';
    mainWindow.loadURL(serverUrl);
  }
  return success;
});

ipcMain.handle('close-config-window', async () => {
  if (configWindow) {
    configWindow.close();
  }
  // After closing config window, check if we should open main window
  setTimeout(async () => {
    const config = await loadConfig();
    if (config && config.serverUrl && !mainWindow) {
      await createMainWindow(config);
    }
  }, 100);
});

// App event handlers
app.whenReady().then(async () => {
  createMenu();
  
  const config = await loadConfig();
  
  if (config && config.serverUrl) {
    await createMainWindow(config);
  } else {
    // Show config window first if no config exists
    createConfigWindow();
    
    // Listen for config saved - check periodically or use window close event
    const checkConfigInterval = setInterval(async () => {
      const newConfig = await loadConfig();
      if (newConfig && newConfig.serverUrl && !mainWindow) {
        clearInterval(checkConfigInterval);
        await createMainWindow(newConfig);
        if (configWindow) {
          configWindow.close();
        }
      }
    }, 500);
  }

  app.on('activate', async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      const config = await loadConfig();
      if (config && config.serverUrl) {
        await createMainWindow(config);
      } else {
        createConfigWindow();
      }
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (mainWindow) {
    mainWindow.removeAllListeners('close');
  }
});

