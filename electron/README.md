# Bastion Desktop Application

Electron-based desktop application for connecting to your hosted Bastion instance.

## Features

- Standalone desktop application (no browser required)
- **Secure credential storage** using Windows Credential Manager (OS Keychain)
- Auto-login functionality
- Settings management via application menu
- Windows installer and portable executable

## Development

### Prerequisites

- Node.js 20 or higher
- npm

### Setup

```bash
cd electron
npm install
```

### Running in Development

```bash
npm start
```

### Building

Build Windows executable:

```bash
npm run build:win
```

Build outputs will be in the `dist/` directory:
- `Bastion-Setup-{version}.exe` - Windows installer
- `Bastion-{version}-portable.exe` - Portable executable

## Configuration

On first launch, the application will prompt you to configure:

- **Server URL**: Full URL of your Bastion instance (e.g., `http://192.168.80.XXX:3051`)
- **Username**: Your Bastion username
- **Password**: Your Bastion password

### Storage

- **Server URL**: Stored in `%APPDATA%\Bastion\config.json` (not sensitive)
- **Credentials**: Stored securely in **Windows Credential Manager** (OS Keychain)

You can change settings at any time via **File > Settings** in the application menu.

### Security

Credentials are stored using Windows Credential Manager, which provides:
- **OS-level encryption** - Credentials are encrypted by Windows
- **User isolation** - Only accessible by the current Windows user
- **No plain text storage** - Passwords never stored in files
- **System integration** - Uses Windows security infrastructure

You can view/manage stored credentials in Windows Credential Manager:
- Open "Credential Manager" from Control Panel
- Look for entries under "Windows Credentials" with service name "Bastion"

## Auto-Login

The application automatically attempts to log in when:
1. The login page is detected
2. Valid credentials are stored in Windows Credential Manager

If auto-login fails, you can manually log in through the standard login form.

## Building Icons

The application expects an icon file at `assets/icon.png` (for development) and `assets/icon.ico` (for Windows builds).

To generate icons:
1. Create a 512x512 PNG icon
2. Place it as `assets/icon.png`
3. For Windows builds, convert to ICO format and place as `assets/icon.ico`

## GitHub Actions

The Electron app is automatically built via GitHub Actions on:
- Push to `main` branch
- Tagged releases (e.g., `v1.0.0`)

Build artifacts are:
- Uploaded as workflow artifacts for all builds
- Attached to GitHub Releases for tagged versions

## Security

### Credential Storage

- **Credentials (username/password)**: Stored in **Windows Credential Manager** (OS Keychain)
  - Encrypted by Windows OS
  - Only accessible by the current Windows user
  - Never stored in plain text files
  - Managed through Windows security infrastructure

- **Server URL**: Stored in config file (`%APPDATA%\Bastion\config.json`)
  - Not sensitive information
  - File permissions restricted to current user

### Keychain Access

The application uses the `keytar` package to interact with Windows Credential Manager. Credentials are stored with:
- **Service**: "Bastion"
- **Account**: Your username
- **Password**: Your password (encrypted by Windows)

### Viewing Stored Credentials

To view or manually manage stored credentials:
1. Open Windows Credential Manager (Control Panel > Credential Manager)
2. Click "Windows Credentials"
3. Look for entries with service name "Bastion"
4. You can view, edit, or remove credentials from here

