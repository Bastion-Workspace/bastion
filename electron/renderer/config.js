// Wait for electronAPI to be available
async function init() {
    // Wait for electronAPI to load
    while (!window.electronAPI) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    const form = document.getElementById('configForm');
    const serverUrlInput = document.getElementById('serverUrl');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const togglePasswordBtn = document.getElementById('togglePassword');
    const saveButton = document.getElementById('saveButton');
    const cancelButton = document.getElementById('cancelButton');
    const errorMessage = document.getElementById('errorMessage');
    const successMessage = document.getElementById('successMessage');

    // Load existing config if available
    try {
        const config = await window.electronAPI.getConfig();
        if (config) {
            if (config.serverUrl) serverUrlInput.value = config.serverUrl;
            if (config.username) usernameInput.value = config.username;
            if (config.password) passwordInput.value = config.password;
        }
    } catch (error) {
        console.error('Error loading config:', error);
    }

    // Toggle password visibility
    togglePasswordBtn.addEventListener('click', () => {
        const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
        passwordInput.setAttribute('type', type);
        togglePasswordBtn.querySelector('.eye-icon').textContent = type === 'password' ? '👁️' : '🙈';
    });

    // Form validation
    function validateForm() {
        const serverUrl = serverUrlInput.value.trim();
        const username = usernameInput.value.trim();
        const password = passwordInput.value;

        if (!serverUrl) {
            showError('Server URL is required');
            return false;
        }

        try {
            new URL(serverUrl);
        } catch (e) {
            showError('Please enter a valid URL (e.g., http://192.168.80.XXX:3051)');
            return false;
        }

        if (!username) {
            showError('Username is required');
            return false;
        }

        if (!password) {
            showError('Password is required');
            return false;
        }

        return true;
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        successMessage.style.display = 'none';
    }

    // Show success message
    function showSuccess(message) {
        successMessage.textContent = message;
        successMessage.style.display = 'block';
        errorMessage.style.display = 'none';
    }

    // Clear messages
    function clearMessages() {
        errorMessage.style.display = 'none';
        successMessage.style.display = 'none';
    }

    // Handle form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearMessages();

        if (!validateForm()) {
            return;
        }

        const config = {
            serverUrl: serverUrlInput.value.trim(),
            username: usernameInput.value.trim(),
            password: passwordInput.value
        };

        saveButton.disabled = true;
        saveButton.textContent = 'Saving...';

        try {
            const success = await window.electronAPI.saveConfig(config);
            
            if (success) {
                showSuccess('Configuration saved successfully!');
                
                // Close window after a brief delay
                setTimeout(() => {
                    window.electronAPI.closeConfigWindow();
                }, 1000);
            } else {
                showError('Failed to save configuration. Please try again.');
                saveButton.disabled = false;
                saveButton.textContent = 'Save Configuration';
            }
        } catch (error) {
            console.error('Error saving config:', error);
            showError('An error occurred while saving. Please try again.');
            saveButton.disabled = false;
            saveButton.textContent = 'Save Configuration';
        }
    });

    // Handle cancel button
    cancelButton.addEventListener('click', () => {
        window.electronAPI.closeConfigWindow();
    });

    // Clear messages when user starts typing
    [serverUrlInput, usernameInput, passwordInput].forEach(input => {
        input.addEventListener('input', clearMessages);
    });

    // Validate URL format on blur
    serverUrlInput.addEventListener('blur', () => {
        const url = serverUrlInput.value.trim();
        if (url) {
            try {
                new URL(url);
                clearMessages();
            } catch (e) {
                showError('Please enter a valid URL');
            }
        }
    });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

