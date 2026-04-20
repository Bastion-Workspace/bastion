/**
 * Client-side types and WebSocket helper for out-of-band agent status messages.
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
 * Tool name from the orchestrator (open set; examples include search_documents, search_and_crawl).
 * @typedef {string} ToolName
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
 * Creates and manages WebSocket connection for agent status updates.
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
    
    console.log(`Agent status WebSocket connecting to ${wsUrl}`);
    
    let ws = null;
    let reconnectTimeout = null;
    let isIntentionallyClosed = false;
    
    const connect = () => {
        try {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log(`Agent status WebSocket connected for conversation ${conversationId}`);
                if (onConnect) onConnect();
            };
            
            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    console.log('Agent status WebSocket message', message);
                    onMessage(message);
                } catch (error) {
                    console.error('Agent status WebSocket failed to parse message', error);
                    if (onError) onError(error);
                }
            };
            
            ws.onclose = () => {
                console.log(`Agent status WebSocket disconnected from conversation ${conversationId}`);
                if (onDisconnect) onDisconnect();
                
                // Auto-reconnect unless intentionally closed
                if (!isIntentionallyClosed) {
                    console.log('Agent status WebSocket reconnecting in 3 seconds...');
                    reconnectTimeout = setTimeout(connect, 3000);
                }
            };
            
            ws.onerror = (error) => {
                console.error('Agent status WebSocket error', error);
                if (onError) onError(error);
            };
            
        } catch (error) {
            console.error('Agent status WebSocket failed to create', error);
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
            console.log(`Agent status WebSocket closed for conversation ${conversationId}`);
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
