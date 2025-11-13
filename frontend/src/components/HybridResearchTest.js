import React, { useState } from 'react';
import { Box, TextField, Button, Typography, Paper, Alert } from '@mui/material';

const HybridResearchTest = () => {
  const [query, setQuery] = useState('');
  const [conversationId, setConversationId] = useState('test-conv-123');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const testHybridResearch = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/research/hybrid/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query_or_clarification: query,
          conversation_id: conversationId,
          title: `Test: ${query.substring(0, 30)}...`
        })
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ success: false, error: error.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800 }}>
      <Typography variant="h4" gutterBottom>
        🎯 Roosevelt's Hybrid Research System Test
      </Typography>
      
      <Typography variant="body1" sx={{ mb: 3, fontStyle: 'italic' }}>
        Test the hybrid clarification system that handles both ambiguous queries and clarification responses.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        <TextField
          fullWidth
          label="Conversation ID"
          value={conversationId}
          onChange={(e) => setConversationId(e.target.value)}
          sx={{ mb: 2 }}
          helperText="Use the same conversation ID to test clarification resolution"
        />
        
        <TextField
          fullWidth
          multiline
          rows={3}
          label="Query or Clarification"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          sx={{ mb: 2 }}
          placeholder="Try: 'How did he die?' then 'Bernie Ebbers'"
        />
        
        <Button 
          variant="contained" 
          onClick={testHybridResearch}
          disabled={loading || !query.trim()}
          sx={{ mb: 2 }}
        >
          {loading ? 'Processing...' : 'Process with Hybrid System'}
        </Button>
      </Paper>

      {result && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Result:
          </Typography>
          
          {result.success ? (
            <Box>
              <Alert severity="success" sx={{ mb: 2 }}>
                Type: {result.type}
              </Alert>
              
              {result.type === 'query_resolved' && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" color="primary">
                    🎯 Query Successfully Resolved!
                  </Typography>
                  <Typography variant="body2">
                    <strong>Original:</strong> {result.original_query}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Resolved:</strong> {result.resolved_query}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Confidence:</strong> {(result.resolution_confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    <strong>Plan ID:</strong> {result.plan_id}
                  </Typography>
                </Box>
              )}
              
              {result.type === 'clarification_needed' && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" color="warning.main">
                    ❓ Clarification Requested
                  </Typography>
                  <Typography variant="body2">
                    {result.clarification_request?.message}
                  </Typography>
                  {result.clarification_request?.suggested_options && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2"><strong>Suggestions:</strong></Typography>
                      <ul>
                        {result.clarification_request.suggested_options.map((option, idx) => (
                          <li key={idx}>{option}</li>
                        ))}
                      </ul>
                    </Box>
                  )}
                </Box>
              )}
              
              {result.type === 'plan_created' && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" color="success.main">
                    ✅ Research Plan Created
                  </Typography>
                  <Typography variant="body2">
                    <strong>Plan ID:</strong> {result.plan_id}
                  </Typography>
                </Box>
              )}
              
              {result.type === 'research_cancelled' && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" color="info.main">
                    🚫 Research Planning Cancelled
                  </Typography>
                  <Typography variant="body2">
                    <strong>Graceful Exit:</strong> {result.graceful_exit ? 'Yes' : 'No'}
                  </Typography>
                  {result.cancelled_queries && result.cancelled_queries.length > 0 && (
                    <Typography variant="body2">
                      <strong>Cancelled Queries:</strong> {result.cancelled_queries.length}
                    </Typography>
                  )}
                  {result.suggested_alternative && (
                    <Typography variant="body2">
                      <strong>Alternative:</strong> {result.suggested_alternative}
                    </Typography>
                  )}
                </Box>
              )}
              
              <Typography variant="body2" sx={{ mt: 2 }}>
                <strong>Message:</strong> {result.message}
              </Typography>
            </Box>
          ) : (
            <Alert severity="error">
              Error: {result.error || 'Unknown error occurred'}
            </Alert>
          )}
          
          <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.100', borderRadius: 1 }}>
            <Typography variant="caption" component="pre">
              {JSON.stringify(result, null, 2)}
            </Typography>
          </Box>
        </Paper>
      )}
      
      <Paper sx={{ p: 2, mt: 3, backgroundColor: 'info.light' }}>
        <Typography variant="h6" gutterBottom>
          🎖️ Test Scenarios:
        </Typography>
        <Typography variant="body2" component="div">
          <strong>1. Ambiguous Query Test:</strong>
          <br />• Query: "How did he die?"
          <br />• Expected: Clarification request
          <br /><br />
          <strong>2. Clarification Response Test:</strong>
          <br />• Query: "Bernie Ebbers" (same conversation)
          <br />• Expected: Resolved to "How did Bernie Ebbers die?"
          <br /><br />
          <strong>3. Clear Query Test:</strong>
          <br />• Query: "How did Bernie Ebbers die?"
          <br />• Expected: Direct plan creation
          <br /><br />
          <strong>4. Cancellation Test:</strong>
          <br />• Query: "How did he die?" → "Never mind"
          <br />• Expected: Research cancelled, graceful return to chat
          <br /><br />
          <strong>5. Implicit Cancellation Test:</strong>
          <br />• Query: "How did he die?" → "Let's move on"
          <br />• Expected: LLM detects cancellation intent
        </Typography>
      </Paper>
    </Box>
  );
};

export default HybridResearchTest;
