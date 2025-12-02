/**
 * ROOSEVELT'S AGENT STATUS TYPES: Type definitions for out-of-band WebSocket agent status messages
 * 
 * This is the client-side companion to the backend's agent status streaming system.
 * Use these types to handle real-time agent tool execution updates.
 */

/**
 * Agent Status Message Types
 * @typedef {'agent_status_connected' | 'agent_status' | 'agent_status_echo'} AgentStatusMessageType
 */

/**
 * Agent Status Types (what's happening)
 * @typedef {'tool_start' | 'tool_complete' | 'tool_error' | 'iteration_start' | 'synthesis'} AgentStatusType
 */

/**
 * Agent Types (who's doing the work)
 * @typedef {'research_agent' | 'chat_agent' | 'weather_agent' | 'coding_agent' | 'direct_agent' | 'site_crawl_agent'} AgentType
 */

/**
 * Tool Names (what tool is being executed)
 * @typedef {'search_local' | 'get_document' | 'search_and_crawl' | 'crawl_web_content' | 'search_entities'} ToolName
 */

/**
 * Base Agent Status Message
 * @typedef {Object} AgentStatusMessage
 * @property {'agent_status'} type - Message type
 * @property {string} conversation_id - Target conversation ID
 * @property {AgentStatusType} status_type - What's happening (tool_start, tool_complete, etc.)
 * @property {string} message - Human-readable status message
 * @property {AgentType} [agent_type] - Which agent is working
 * @property {ToolName} [tool_name] - Which tool is being executed
 * @property {number} [iteration] - Current iteration number (1-8)
 * @property {number} [max_iterations] - Maximum iterations (8)
 * @property {Object} [metadata] - Additional context
 * @property {string} timestamp - ISO timestamp
 */

/**
 * Connection Confirmation Message
 * @typedef {Object} AgentStatusConnectedMessage
 * @property {'agent_status_connected'} type - Message type
 * @property {string} conversation_id - Conversation ID
 * @property {string} message - Confirmation message
 * @property {string} timestamp - ISO timestamp
 */

/**
 * Echo/Keepalive Message
 * @typedef {Object} AgentStatusEchoMessage
 * @property {'agent_status_echo'} type - Message type
 * @property {string} conversation_id - Conversation ID
 * @property {string} data - Echoed data
 * @property {string} timestamp - ISO timestamp
 */

/**
 * Union type for all agent status messages
 * @typedef {AgentStatusMessage | AgentStatusConnectedMessage | AgentStatusEchoMessage} AgentStatusWebSocketMessage
 */

/**
 * WebSocket Connection Options
 * @typedef {Object} AgentStatusWebSocketOptions
 * @property {string} conversationId - Conversation ID to subscribe to
 * @property {string} token - Authentication token
 * @property {function(AgentStatusWebSocketMessage): void} onMessage - Message handler
 * @property {function(): void} [onConnect] - Connection established handler
 * @property {function(): void} [onDisconnect] - Disconnection handler
 * @property {function(Error): void} [onError] - Error handler
 */

/**
 * ROOSEVELT'S AGENT STATUS WEBSOCKET CLIENT
 * 
 * Creates and manages WebSocket connection for agent status updates
 * 
 * @param {AgentStatusWebSocketOptions} options - Connection options
 * @returns {Object} WebSocket control object
 */
export function createAgentStatusWebSocket(options) {
    const { conversationId, token, onMessage, onConnect, onDisconnect, onError } = options;
    
    // Construct WebSocket URL
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.location.host;
    const wsUrl = `${wsProtocol}//${wsHost}/api/ws/agent-status/${conversationId}?token=${encodeURIComponent(token)}`;
    
    console.log(`🤖 AGENT STATUS: Connecting to ${wsUrl}`);
    
    let ws = null;
    let reconnectTimeout = null;
    let isIntentionallyClosed = false;
    
    const connect = () => {
        try {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log(`✅ AGENT STATUS: Connected to conversation ${conversationId}`);
                if (onConnect) onConnect();
            };
            
            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log(`🤖 AGENT STATUS: Received`, message);
                    onMessage(message);
                } catch (error) {
                    console.error(`❌ AGENT STATUS: Failed to parse message`, error);
                    if (onError) onError(error);
                }
            };
            
            ws.onclose = () => {
                console.log(`📡 AGENT STATUS: Disconnected from conversation ${conversationId}`);
                if (onDisconnect) onDisconnect();
                
                // Auto-reconnect unless intentionally closed
                if (!isIntentionallyClosed) {
                    console.log(`🔄 AGENT STATUS: Reconnecting in 3 seconds...`);
                    reconnectTimeout = setTimeout(connect, 3000);
                }
            };
            
            ws.onerror = (error) => {
                console.error(`❌ AGENT STATUS: WebSocket error`, error);
                if (onError) onError(error);
            };
            
        } catch (error) {
            console.error(`❌ AGENT STATUS: Failed to create WebSocket`, error);
            if (onError) onError(error);
        }
    };
    
    // Initial connection
    connect();
    
    // Return control object
    return {
        /**
         * Close the WebSocket connection
         */
        close: () => {
            isIntentionallyClosed = true;
            if (reconnectTimeout) {
                clearTimeout(reconnectTimeout);
            }
            if (ws) {
                ws.close();
            }
            console.log(`🔌 AGENT STATUS: Connection closed for conversation ${conversationId}`);
        },
        
        /**
         * Send a keepalive ping
         */
        ping: () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
            }
        },
        
        /**
         * Get connection state
         * @returns {number} WebSocket.readyState
         */
        getState: () => {
            return ws ? ws.readyState : WebSocket.CLOSED;
        }
    };
}

/**
 * ROOSEVELT'S STATUS MESSAGE FORMATTER
 * 
 * Format agent status messages for UI display
 * 
 * @param {AgentStatusMessage} message - Agent status message
 * @returns {string} Formatted display text
 */
export function formatAgentStatusMessage(message) {
    const { status_type, message: text, tool_name, iteration, max_iterations } = message;
    
    // Add iteration context if available
    let formatted = text;
    if (iteration && max_iterations) {
        formatted = `[${iteration}/${max_iterations}] ${text}`;
    }
    
    return formatted;
}

/**
 * ROOSEVELT'S STATUS INDICATOR STYLES
 * 
 * Get status indicator styles based on status type
 * 
 * @param {AgentStatusType} statusType - Status type
 * @returns {Object} Style object
 */
export function getAgentStatusStyle(statusType) {
    const styles = {
        tool_start: {
            color: '#2196F3', // Blue
            icon: '🔧',
            animation: 'pulse'
        },
        tool_complete: {
            color: '#4CAF50', // Green
            icon: '✅',
            animation: 'none'
        },
        tool_error: {
            color: '#F44336', // Red
            icon: '❌',
            animation: 'shake'
        },
        iteration_start: {
            color: '#FF9800', // Orange
            icon: '🔄',
            animation: 'spin'
        },
        synthesis: {
            color: '#9C27B0', // Purple
            icon: '✨',
            animation: 'fade'
        }
    };
    
    return styles[statusType] || {
        color: '#757575',
        icon: '📡',
        animation: 'none'
    };
}

/**
 * ROOSEVELT'S STATUS FILTER
 * 
 * Determine if a status update should be shown based on settings
 * 
 * @param {AgentStatusMessage} message - Agent status message
 * @param {Object} settings - User settings
 * @returns {boolean} Whether to show the status
 */
export function shouldShowAgentStatus(message, settings = {}) {
    const {
        showToolStart = true,
        showToolComplete = false,  // Hide successes by default to reduce noise
        showToolError = true,
        showIterationStart = true,
        showSynthesis = true
    } = settings;
    
    switch (message.status_type) {
        case 'tool_start':
            return showToolStart;
        case 'tool_complete':
            return showToolComplete;
        case 'tool_error':
            return showToolError;
        case 'iteration_start':
            return showIterationStart;
        case 'synthesis':
            return showSynthesis;
        default:
            return true;
    }
}

export default {
    createAgentStatusWebSocket,
    formatAgentStatusMessage,
    getAgentStatusStyle,
    shouldShowAgentStatus
};








