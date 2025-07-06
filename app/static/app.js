alert('Testing if app.js is loaded');

console.log('app.js is loaded!');

// Global constants
const API_BASE_URL = '/'; // Use relative paths for uvicorn server

// File upload handlers
window.handleSummaryUpload = async function(event) {
    event.preventDefault();
    console.log('Handle summary upload called');
    const fileInput = document.getElementById('summaryFile');
    const file = fileInput.files[0];
    console.log('Selected file:', file);
    
    if (!file) {
        console.log('No file selected');
        showError('Please select a file to upload');
        return;
    }
    
    if (!file.name.endsWith('.json')) {
        console.log('Invalid file type:', file.name);
        showError('Only JSON files are accepted');
        return;
    }
    
    try {
        const formData = new FormData();
        formData.append('report', file);
        const submitUrl = `${API_BASE_URL}api/submit-report`;
        console.log('Sending request to:', submitUrl);
        
        const response = await fetch(submitUrl, {
            method: 'POST',
            body: formData
        });
        console.log('Response received:', response);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error response:', errorData);
            throw new Error(errorData.detail || 'Failed to upload summary');
        }
        
        const result = await response.json();
        console.log('Success response:', result);
        showSuccess(result.message || 'Summary uploaded successfully');
        
        // Clear the file input
        fileInput.value = '';
        
        // Update status with processing info
        updateStatus(`Processing summary: ${file.name}`, null, false);
        
    } catch (error) {
        console.error('Error uploading summary:', error);
        showError(error.message || 'Failed to upload summary');
    }
}

// Helper functions
window.showError = function(message) {
    const errorDisplay = document.getElementById('errorDisplay');
    if (errorDisplay) {
        errorDisplay.textContent = message;
        errorDisplay.classList.remove('hidden');
        setTimeout(() => {
            errorDisplay.classList.add('hidden');
        }, 5000);
    } else {
        console.error('Error:', message);
    }
}

window.showSuccess = function(message) {
    const successDisplay = document.getElementById('successDisplay') || createSuccessDisplay();
    successDisplay.textContent = message;
    successDisplay.classList.remove('hidden');
    setTimeout(() => {
        successDisplay.classList.add('hidden');
    }, 3000);
}

window.updateStatus = function(message, id = null, isError = false) {
    const statusList = document.getElementById('statusList');
    if (!statusList) return;

    const statusItem = document.createElement('div');
    statusItem.className = `p-2 mb-2 rounded ${isError ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'}`;
    statusItem.textContent = message;
    if (id) statusItem.id = `status-${id}`;
    
    statusList.appendChild(statusItem);
}

function createSuccessDisplay() {
    const successDisplay = document.createElement('div');
    successDisplay.id = 'successDisplay';
    successDisplay.className = 'bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4 hidden';
    const targetSection = document.getElementById('status') || document.body;
    targetSection.insertBefore(successDisplay, targetSection.firstChild);
    return successDisplay;
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded');
    
    // Initialize UI elements first
    const statusSection = document.getElementById('status');
    if (!statusSection) {
        const mainContainer = document.querySelector('.container');
        const newStatusSection = document.createElement('div');
        newStatusSection.id = 'status';
        newStatusSection.className = 'mt-4';
        mainContainer.appendChild(newStatusSection);
    }

    // Create error display element if it doesn't exist
    let errorDisplay = document.getElementById('errorDisplay');
    if (!errorDisplay) {
        errorDisplay = document.createElement('div');
        errorDisplay.id = 'errorDisplay';
        errorDisplay.className = 'bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 hidden';
        const targetSection = document.getElementById('status') || document.body;
        targetSection.insertBefore(errorDisplay, targetSection.firstChild);
    }

    // Add quota check button
    const quotaButton = document.createElement('button');
    quotaButton.id = 'quotaCheckButton';
    quotaButton.className = 'btn btn-info mb-3';
    quotaButton.textContent = 'Check API Quota';
    quotaButton.onclick = checkQuota;
    
    // Add quota status display div
    const quotaDiv = document.createElement('div');
    quotaDiv.id = 'quota-status';
    quotaDiv.className = 'mb-3';
    
    // Ensure the status section exists and add our elements
    const targetSection = document.getElementById('status') || document.body;
    targetSection.appendChild(quotaButton);
    targetSection.appendChild(quotaDiv);

    // Form submission handlers
    document.getElementById('singleVideoForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = document.getElementById('singleVideoUrl').value;
        await processVideo(url);
        document.getElementById('singleVideoUrl').value = '';
    });

    document.getElementById('multipleVideosForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const urls = document.getElementById('multipleVideosUrls').value
            .split('\n')
            .map(url => url.trim())
            .filter(url => url);
        
        for (const url of urls) {
            await processVideo(url);
        }
        document.getElementById('multipleVideosUrls').value = '';
    });

    document.getElementById('playlistForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const url = document.getElementById('playlistUrl').value;
        await processPlaylist(url);
        document.getElementById('playlistUrl').value = '';
    });

    document.getElementById('chatForm')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const input = document.getElementById('chatInput');
        const query = input.value.trim();
        
        if (!query) return;
        
        // Display user message
        appendMessage('user', query);
        input.value = '';
        
        try {
            const response = await fetch(`${API_BASE_URL}chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });
            
            if (!response.ok) {
                throw new Error('Failed to get response');
            }
            
            const data = await response.json();
            appendMessage('assistant', data.response);
            
        } catch (error) {
            console.error('Chat error:', error);
            appendMessage('error', 'Sorry, I encountered an error processing your request.');
        }
    });

    const summaryForm = document.getElementById('summaryUploadForm');
    console.log('Found summary form:', summaryForm);
    
    if (summaryForm) {
        summaryForm.addEventListener('submit', handleSummaryUpload);
    }

    // Initialize form handlers
    const uploadBtn = document.getElementById('uploadSummaryBtn');
    console.log('Found upload button:', uploadBtn);
    
    if (uploadBtn) {
        uploadBtn.addEventListener('click', async () => {
            console.log('Upload button clicked');
            
            const fileInput = document.getElementById('summaryFile');
            const file = fileInput.files[0];
            console.log('Selected file:', file);
            
            if (!file) {
                console.log('No file selected');
                showError('Please select a file to upload');
                return;
            }
            
            if (!file.name.endsWith('.json')) {
                console.log('Invalid file type:', file.name);
                showError('Only JSON files are accepted');
                return;
            }
            
            try {
                const formData = new FormData();
                formData.append('report', file);
                const submitUrl = `${API_BASE_URL}api/submit-report`;
                console.log('Sending request to:', submitUrl);
                
                const response = await fetch(submitUrl, {
                    method: 'POST',
                    body: formData
                });
                console.log('Response received:', response);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    console.error('Error response:', errorData);
                    throw new Error(errorData.detail || 'Failed to upload summary');
                }
                
                const result = await response.json();
                console.log('Success response:', result);
                showSuccess(result.message || 'Summary uploaded successfully');
                
                // Clear the file input
                fileInput.value = '';
                
                // Update status with processing info
                updateStatus(`Processing summary: ${file.name}`, null, false);
                
            } catch (error) {
                console.error('Error uploading summary:', error);
                showError(error.message || 'Failed to upload summary');
            }
        });
    }

    // API interaction functions
    async function processVideo(url) {
        try {
            // Validate URL isn't empty
            if (!url || url.trim() === '') {
                throw new Error('URL cannot be empty');
            }
            
            // Clean the URL
            url = url.trim();
            
            // Add protocol if missing
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                url = 'https://' + url;
            }
            
            console.log('Sending URL to backend:', url);
            logToConsole(`Starting video analysis for: ${url}`, 'info');
            
            // Use FormData to match the new API endpoint
            const formData = new FormData();
            formData.append('url', url);
            
            const response = await fetch(`${API_BASE_URL}api/analyze-url`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                if (response.status === 429) {
                    showError('API rate limit exceeded. Please try again later.');
                    logToConsole('Rate limit exceeded', 'error');
                    return null;
                }
                throw new Error(errorData.detail || 'Failed to process video');
            }
            
            const data = await response.json();
            logToConsole(`Video queued successfully: ${data.video_id}`, 'success');
            logToConsole(`Queue position: ${data.queue_info.position}`, 'info');
            updateStatus(`Video queued for processing: ${url}`, data.video_id);
            
            // Start monitoring the queue and video processing
            startQueueMonitoring();
            pollVideoStatus(data.video_id);
            
        } catch (error) {
            console.error('Error processing video:', error);
            logToConsole(`Error processing video: ${error.message}`, 'error');
            updateStatus(`Error processing video: ${error.message}`, null, true);
        }
    }

    async function processPlaylist(url) {
        try {
            updateStatus(`Starting playlist processing: ${url}`);
            
            const response = await fetch(`${API_BASE_URL}analyze-playlist`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url }),
            });
            
            if (!response.ok) {
                const error = await response.text();
                if (response.status === 429) {
                    showError('API rate limit exceeded. Please try again later.');
                    return null;
                }
                throw new Error(error);
            }
            
            const data = await response.json();
            updateStatus(`Processing playlist with ${data.total_videos} videos: ${url}`, data.playlist_id);
            
            // Start polling for each video
            for (const videoId of data.video_ids) {
                updateStatus(`Starting video processing`, videoId);
                pollVideoStatus(videoId);
            }
        } catch (error) {
            console.error('Error processing playlist:', error);
            updateStatus(`Error processing playlist: ${error.message}`, null, true);
        }
    }

    async function pollVideoStatus(videoId) {
        const maxAttempts = 60; // 10 minutes maximum (10s * 60)
        let attempts = 0;
        let lastProgress = {};
        let lastStatus = '';
        
        const interval = setInterval(async () => {
            try {
                if (attempts >= maxAttempts) {
                    console.error('Status polling timeout');
                    updateStatus(`Timeout waiting for video ${videoId}`, videoId, true);
                    clearInterval(interval);
                    return;
                }
                
                const response = await fetch(`${API_BASE_URL}status/${videoId}`);
                if (!response.ok) {
                    if (response.status === 429) {
                        showError('API rate limit exceeded. Please try again later.');
                        clearInterval(interval);
                        return;
                    }
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Only update if there's a change in progress or status
                const hasProgressChanged = JSON.stringify(data.progress) !== JSON.stringify(lastProgress);
                
                if (hasProgressChanged || data.status !== lastStatus) {
                    console.log('Status update:', data);  // Debug log
                    
                    if (data.error) {
                        console.error(`Video processing error:`, data.error);
                        updateStatus(`Error processing video: ${data.error}`, videoId, true);
                        clearInterval(interval);
                        return;
                    }
                    
                    // Update status with detailed information
                    let statusMessage = `Video ${videoId}: ${data.status}`;
                    if (data.progress) {
                        if (data.progress.downloading) {
                            statusMessage += ` | Downloading: ${data.progress.downloading}%`;
                        }
                        if (data.progress.transcribing) {
                            statusMessage += ` | Transcribing: ${data.progress.transcribing}%`;
                        }
                        if (data.progress.analyzing) {
                            statusMessage += ` | Analyzing: ${data.progress.analyzing}%`;
                        }
                        if (data.progress.current_chunk) {
                            statusMessage += ` | Processing chunk: ${data.progress.current_chunk}/${data.progress.total_chunks}`;
                        }
                        lastProgress = { ...data.progress };
                        lastStatus = data.status;
                        console.log('Progress update:', data.progress);
                    }
                    
                    updateStatus(statusMessage, videoId);
                }
                
                if (data.status === 'COMPLETED' || data.status === 'FAILED') {
                    if (data.status === 'COMPLETED') {
                        console.log('Video processing completed:', data);
                        if (data.transcript_path) {
                            console.log('Transcript saved at:', data.transcript_path);
                        }
                    }
                    clearInterval(interval);
                }
                
                attempts++;
            } catch (error) {
                console.error('Error polling status:', error);
                attempts++;
                
                if (attempts >= maxAttempts) {
                    updateStatus(`Failed to get video status: ${error.message}`, videoId, true);
                    clearInterval(interval);
                }
            }
        }, 10000); // Poll every 10 seconds instead of 5
        
        // Clean up interval when page unloads
        window.addEventListener('beforeunload', () => clearInterval(interval));
    }

    // UI update functions
    function updateStatus(message, id = null, isError = false) {
        const statusDiv = document.getElementById('status');
        const statusItem = document.createElement('div');
        statusItem.className = `status-item ${isError ? 'error' : ''}`;
        
        const timestamp = new Date().toLocaleTimeString();
        statusItem.innerHTML = `
            <span class="timestamp">[${timestamp}]</span>
            <span class="message">${message}</span>
            ${id ? `<span class="id">(ID: ${id})</span>` : ''}
        `;
        
        statusDiv.insertBefore(statusItem, statusDiv.firstChild);
        console.log(`[${timestamp}] ${message}${id ? ` (ID: ${id})` : ''}`);
        
        // Keep only the last 50 status messages
        while (statusDiv.children.length > 50) {
            statusDiv.removeChild(statusDiv.lastChild);
        }
    }

    function appendMessage(role, content) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        
        // Style based on message role
        let className = 'p-4 rounded-lg max-w-[80%] ';
        switch (role) {
            case 'user':
                className += 'bg-blue-100 ml-auto';
                break;
            case 'assistant':
                className += 'bg-gray-100';
                break;
            case 'error':
                className += 'bg-red-100 text-red-700';
                break;
        }
        
        messageDiv.className = className;
        
        // Format message content
        if (role === 'assistant') {
            // Convert markdown-style code blocks to HTML
            content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, 
                (_, lang, code) => `<pre class="bg-gray-800 text-white p-2 rounded"><code>${code.trim()}</code></pre>`
            );
            // Convert single backtick code to inline code
            content = content.replace(/`([^`]+)`/g, 
                (_, code) => `<code class="bg-gray-200 px-1 rounded">${code}</code>`
            );
            messageDiv.innerHTML = content;
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function checkQuota() {
        try {
            const quotaDiv = document.getElementById('quota-status');
            quotaDiv.innerHTML = 'Checking quota status...';
            
            const response = await fetch(`${API_BASE_URL}quota`);
            if (!response.ok) {
                throw new Error('Failed to check quota');
            }
            
            const data = await response.json();
            updateQuotaDisplay(data);
            
        } catch (error) {
            console.error('Error checking quota:', error);
            const quotaDiv = document.getElementById('quota-status');
            quotaDiv.innerHTML = `<div class="alert alert-danger">Error checking quota: ${error.message}</div>`;
        }
    }

    function updateQuotaDisplay(data) {
        const quotaDiv = document.getElementById('quota-status');
        if (data.status === 'error') {
            quotaDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
            return;
        }
        
        let html = '<div class="quota-info">';
        html += '<h5>Gemini Flash API Status:</h5>';
        
        for (const [modelName, info] of Object.entries(data.quotas)) {
            const statusClass = info.status === 'available' ? 'text-success' : 'text-danger';
            html += `
                <div class="model-quota mb-2">
                    <strong>${info.purpose}:</strong>
                    <span class="${statusClass}">${info.status}</span>
                    ${info.error ? `<div class="error-msg text-danger">${info.error}</div>` : ''}
                </div>`;
        }
        
        html += `<div class="text-muted small">Last checked: ${new Date(data.timestamp).toLocaleString()}</div>`;
        html += '</div>';
        
        quotaDiv.innerHTML = html;
    }

    // LangTrace settings functionality removed

    // Load settings on startup
    loadSettings();
    
    // Save settings when the save button is clicked
    document.getElementById('saveSettings').addEventListener('click', function() {
        const geminiApiKey = document.getElementById('geminiApiKey').value;
        const geminiRpm = parseInt(document.getElementById('geminiRpm').value);
        
        // Validate RPM
        if (geminiRpm < 1 || geminiRpm > 60) {
            alert('RPM must be between 1 and 60');
            return;
        }
        
        // Save settings
        fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                gemini_api_key: geminiApiKey || null,
                gemini_rpm_override: geminiRpm
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('settingsModal'));
                modal.hide();
                
                // Show success message
                showMessage('Settings saved successfully', 'success');
            } else {
                showMessage('Failed to save settings: ' + (data.detail || 'Unknown error'), 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showMessage('Failed to save settings: ' + error.message, 'error');
        });
    });

    function loadSettings() {
        fetch('/api/settings')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.gemini_rpm_override) {
                    document.getElementById('geminiRpm').value = data.gemini_rpm_override;
                }
                // Don't load API key for security
            })
            .catch(error => {
                console.error('Error loading settings:', error);
                showMessage('Failed to load settings: ' + error.message, 'error');
            });
    }

    function showMessage(message, type = 'info') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show fixed-top mx-auto mt-3`;
        alertDiv.style.maxWidth = '500px';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    const style = document.createElement('style');
    style.textContent = `
        .quota-info {
            padding: 10px;
            border-radius: 4px;
            background: #f5f5f5;
            margin: 10px 0;
        }
        .quota-item {
            margin: 8px 0;
            padding: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: white;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .model-info {
            display: flex;
            flex-direction: column;
        }
        .model-name {
            font-weight: bold;
            color: #333;
        }
        .model-id {
            font-size: 0.8em;
            color: #666;
        }
        .status {
            padding: 4px 8px;
            border-radius: 3px;
            font-weight: 500;
        }
        .status.available {
            background: #4caf50;
            color: white;
        }
        .status.exhausted {
            background: #f44336;
            color: white;
        }
        .status.error {
            background: #ff9800;
            color: white;
        }
        .quota-error {
            color: #f44336;
            padding: 10px;
            border: 1px solid #f44336;
            border-radius: 4px;
            background: #ffebee;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize enhanced console logging with proper timing
    setTimeout(() => {
        initializeEnhancedConsole();
        
        // Start queue status monitoring after console is ready
        setTimeout(() => {
            startQueueMonitoring();
        }, 500);
    }, 100);
});

// Enhanced Console Logging System
let autoScroll = true;
let consoleLog = null;
let logWebSocket = null;
let wsReconnectAttempts = 0;
const maxReconnectAttempts = 5;

function initializeEnhancedConsole() {
    consoleLog = document.getElementById('consoleLog');
    
    if (!consoleLog) {
        console.error('Console log element not found!');
        return;
    }
    
    // Clear any existing content
    consoleLog.innerHTML = '';
    
    // Clear console button
    document.getElementById('clearConsole')?.addEventListener('click', clearConsole);
    
    // Toggle auto-scroll button
    const toggleBtn = document.getElementById('toggleAutoScroll');
    toggleBtn?.addEventListener('click', function() {
        autoScroll = !autoScroll;
        toggleBtn.textContent = `Auto-scroll: ${autoScroll ? 'ON' : 'OFF'}`;
        toggleBtn.className = autoScroll ? 
            'px-3 py-1 text-sm bg-blue-500 hover:bg-blue-600 text-white rounded' :
            'px-3 py-1 text-sm bg-gray-500 hover:bg-gray-600 text-white rounded';
    });
    
    // Log initialization message
    logToConsole('Video analyzer console initialized', 'info');
    logToConsole('Enhanced console initialized', 'info');
    
    // Initialize WebSocket for log streaming
    initializeLogWebSocket();
}

function logToConsole(message, level = 'info', timestamp = null) {
    // Ensure console log element is available
    if (!consoleLog) {
        consoleLog = document.getElementById('consoleLog');
        if (!consoleLog) {
            console.warn('Console log element not found');
            return;
        }
    }
    
    const entry = document.createElement('div');
    entry.className = `console-entry ${level}`;
    
    const ts = timestamp || new Date().toISOString().substring(0, 19);
    const levelText = level.toUpperCase().padEnd(7);
    
    entry.innerHTML = `
        <span class="timestamp">[${ts}]</span>
        <span class="level">${levelText}</span>
        <span class="message">${escapeHtml(message)}</span>
    `;
    
    consoleLog.appendChild(entry);
    
    // Auto-scroll to bottom if enabled
    if (autoScroll) {
        setTimeout(() => {
            consoleLog.scrollTop = consoleLog.scrollHeight;
        }, 10);
    }
    
    // Limit console entries to prevent memory issues
    const maxEntries = 1000;
    const entries = consoleLog.querySelectorAll('.console-entry');
    if (entries.length > maxEntries) {
        entries[0].remove();
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function clearConsole() {
    if (consoleLog) {
        consoleLog.innerHTML = '';
        logToConsole('Console cleared', 'info');
    }
}

// WebSocket Log Streaming Functions
function initializeLogWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;
    
    try {
        logWebSocket = new WebSocket(wsUrl);
        
        logWebSocket.onopen = function(event) {
            logToConsole('Event streaming connected', 'success');
            wsReconnectAttempts = 0;
        };
        
        logWebSocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                
                // Handle different message types
                if (data.type === 'log') {
                    logToConsole(data.data.message, data.data.level, data.data.timestamp);
                } 
                else if (data.type === 'event') {
                    // Handle specific video processing events
                    const eventType = data.event_type;
                    const message = data.message;
                    const videoId = data.video_id;
                    
                    // Format message based on event type
                    let level = 'info';
                    let prefix = '';
                    
                    switch (eventType) {
                        case 'video_queued':
                            prefix = '🎬 ';
                            break;
                        case 'transcription_start':
                        case 'transcription_progress':
                        case 'transcription_complete':
                            prefix = '🗣️ ';
                            break;
                        case 'image_analysis_start':
                        case 'image_analysis_progress':
                        case 'image_analysis_complete':
                            prefix = '🖼️ ';
                            break;
                        case 'content_analysis_start':
                        case 'content_analysis_complete':
                            prefix = '🧠 ';
                            break;
                        case 'video_processing_complete':
                            prefix = '✅ ';
                            level = 'success';
                            break;
                        case 'error_occurred':
                            prefix = '❌ ';
                            level = 'error';
                            break;
                    }
                    
                    // Log the event with appropriate formatting
                    logToConsole(`${prefix}[${videoId}] ${message}`, level, data.timestamp);
                    
                    // If this is a completion event, refresh video status
                    if (eventType === 'video_processing_complete') {
                        refreshVideoStatus(videoId);
                    }
                }
                // Handle connection messages
                else if (data.type === 'connection') {
                    logToConsole(data.message, 'success', data.timestamp);
                }
                // Handle ping/pong for keepalive
                else if (data.type === 'ping') {
                    logWebSocket.send(JSON.stringify({type: 'pong'}));
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        logWebSocket.onclose = function(event) {
            logToConsole('Log streaming disconnected', 'warning');
            
            // Attempt to reconnect if not too many attempts
            if (wsReconnectAttempts < maxReconnectAttempts) {
                wsReconnectAttempts++;
                logToConsole(`Attempting to reconnect (${wsReconnectAttempts}/${maxReconnectAttempts})...`, 'info');
                setTimeout(() => {
                    initializeLogWebSocket();
                }, 2000 * wsReconnectAttempts); // Exponential backoff
            } else {
                logToConsole('Max reconnection attempts reached. Manual refresh may be needed.', 'error');
            }
        };
        
        logWebSocket.onerror = function(error) {
            logToConsole('WebSocket error occurred', 'error');
            console.error('WebSocket error:', error);
        };
        
    } catch (error) {
        logToConsole('Failed to initialize log streaming', 'error');
        console.error('WebSocket initialization error:', error);
    }
}

function closeLogWebSocket() {
    if (logWebSocket) {
        logWebSocket.close();
        logWebSocket = null;
    }
}

// Queue Status Monitoring
let queueMonitorInterval = null;

function startQueueMonitoring() {
    // Initial load
    updateQueueStatus();
    
    // Update every 5 seconds
    queueMonitorInterval = setInterval(updateQueueStatus, 5000);
}

async function updateQueueStatus() {
    try {
        const response = await fetch('/api/queue/status');
        if (response.ok) {
            const queueData = await response.json();
            displayQueueStatus(queueData);
        } else {
            throw new Error(`Queue status request failed: ${response.status}`);
        }
    } catch (error) {
        console.error('Error updating queue status:', error);
        document.getElementById('queueInfo').innerHTML = 
            '<span class="text-red-600">Error loading queue status</span>';
    }
}

function displayQueueStatus(queueData) {
    const queueInfo = document.getElementById('queueInfo');
    if (!queueInfo) return;
    
    // Handle both possible API response formats with fallbacks
    const queue_size = queueData.queue_size ?? queueData.size ?? 0;
    const total_processed = queueData.total_processed ?? queueData.processed ?? 0;
    const is_processing = queueData.is_processing ?? queueData.processing ?? false;
    const current_video = queueData.current_video ?? queueData.current ?? null;
    const status = queueData.status ?? (is_processing ? 'Processing' : 'Idle');
    
    let html = `
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
                <span class="font-medium">Queue Size:</span>
                <span class="ml-1 px-2 py-1 bg-blue-100 text-blue-800 rounded">${queue_size}</span>
            </div>
            <div>
                <span class="font-medium">Total Processed:</span>
                <span class="ml-1 px-2 py-1 bg-green-100 text-green-800 rounded">${total_processed}</span>
            </div>
            <div>
                <span class="font-medium">Status:</span>
                <span class="ml-1 px-2 py-1 rounded ${
                    is_processing ? 'bg-yellow-100 text-yellow-800' : 'bg-gray-100 text-gray-800'
                }">${status}</span>
            </div>
            <div>
                <span class="font-medium">Current:</span>
                <span class="ml-1 text-xs">${current_video || 'None'}</span>
            </div>
        </div>
    `;
    
    queueInfo.innerHTML = html;
    
    // Log queue status changes
    if (is_processing && current_video) {
        const shortId = current_video.substring(0, 8);
        logToConsole(`Processing video: ${shortId}...`, 'info');
    }
}
