import React, { useState } from 'react';
import { Box, Button, TextField, Typography, Paper, Alert, List, ListItem, ListItemText, Divider } from '@mui/material';

const ContextAwareResearchTest = () => {
  const [conversationId, setConversationId] = useState('');
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const testContextAwareResearch = async () => {
    if (!conversationId || !query) {
      setError('Please provide both conversation ID and query');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/research/context-aware', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          query: query
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const testClarityAssessment = async () => {
    if (!conversationId || !query) {
      setError('Please provide both conversation ID and query');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/research/assess-clarity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          query: query
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const testReferenceResolution = async () => {
    if (!conversationId || !query) {
      setError('Please provide both conversation ID and query');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/research/resolve-references', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          query: query
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const testGetContext = async () => {
    if (!conversationId) {
      setError('Please provide conversation ID');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`/api/research/context/${conversationId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;

    if (result.type === 'clarification_needed') {
      return (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" color="warning.main" gutterBottom>
            Clarification Needed
          </Typography>
          <Typography variant="body1" gutterBottom>
            Clarity Score: {result.assessment?.clarity_score}/100
          </Typography>
          <Typography variant="body1" gutterBottom>
            {result.clarification_request?.message}
          </Typography>
          
          {result.clarification_request?.questions && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Questions:
              </Typography>
              <List dense>
                {result.clarification_request.questions.map((question, index) => (
                  <ListItem key={index}>
                    <ListItemText primary={question} />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {result.clarification_request?.suggested_options && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Suggested Options:
              </Typography>
              <List dense>
                {result.clarification_request.suggested_options.map((option, index) => (
                  <ListItem key={index}>
                    <ListItemText primary={option} />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
        </Paper>
      );
    }

    if (result.type === 'plan_created') {
      return (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" color="success.main" gutterBottom>
            Research Plan Created
          </Typography>
          <Typography variant="body1" gutterBottom>
            Plan ID: {result.plan_id}
          </Typography>
          <Typography variant="body1" gutterBottom>
            Resolved Query: {result.resolved_query}
          </Typography>
          <Typography variant="body1" gutterBottom>
            Clarity Score: {result.assessment?.clarity_score}/100
          </Typography>
        </Paper>
      );
    }

    // Generic result display
    return (
      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Result
        </Typography>
        <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
          {JSON.stringify(result, null, 2)}
        </pre>
      </Paper>
    );
  };

  return (
    <Box sx={{ p: 3, maxWidth: 800, mx: 'auto' }}>
      <Typography variant="h4" gutterBottom>
        Context-Aware Research Test
      </Typography>
      
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Test Parameters
        </Typography>
        
        <TextField
          fullWidth
          label="Conversation ID"
          value={conversationId}
          onChange={(e) => setConversationId(e.target.value)}
          margin="normal"
          placeholder="Enter conversation ID to test with"
        />
        
        <TextField
          fullWidth
          label="Query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          margin="normal"
          placeholder="Enter a query (e.g., 'What about his foreign policy?')"
          multiline
          rows={3}
        />
      </Paper>

      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Test Actions
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            onClick={testContextAwareResearch}
            disabled={loading}
          >
            Test Context-Aware Research
          </Button>
          
          <Button
            variant="outlined"
            onClick={testClarityAssessment}
            disabled={loading}
          >
            Test Clarity Assessment
          </Button>
          
          <Button
            variant="outlined"
            onClick={testReferenceResolution}
            disabled={loading}
          >
            Test Reference Resolution
          </Button>
          
          <Button
            variant="outlined"
            onClick={testGetContext}
            disabled={loading}
          >
            Get Conversation Context
          </Button>
        </Box>
      </Box>

      {loading && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Processing...
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {renderResult()}

      <Paper sx={{ p: 2, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Test Examples
        </Typography>
        <Typography variant="body2" paragraph>
          Try these examples to test the context-aware features:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText 
              primary="Ambiguous Query: 'What about his foreign policy?'"
              secondary="This should trigger clarification if 'he' is unclear"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="Vague Reference: 'Tell me about the company'"
              secondary="This should ask which company you're referring to"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="Broad Scope: 'Research everything about AI'"
              secondary="This should suggest narrowing the scope"
            />
          </ListItem>
          <ListItem>
            <ListItemText 
              primary="Clear Query: 'Research renewable energy policy in the European Union'"
              secondary="This should create a plan directly"
            />
          </ListItem>
        </List>
      </Paper>
    </Box>
  );
};

export default ContextAwareResearchTest;
