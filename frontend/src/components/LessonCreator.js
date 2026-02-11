import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  IconButton,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Chip
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Save as SaveIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import apiService from '../services/apiService';

const LessonCreator = ({ open, onClose, initialLesson = null }) => {
  const [lessonData, setLessonData] = useState(initialLesson || {
    title: '',
    description: '',
    difficulty: 'medium',
    introduction: '',
    time_limit_default: 30,
    tags: ['lesson', 'education'],
    questions: []
  });

  const [currentQuestion, setCurrentQuestion] = useState({
    type: 'text',
    question: '',
    options: ['', '', '', ''],
    correct_answer: '',
    explanation: '',
    time_limit: 30,
    image_path: ''
  });

  const [editingIndex, setEditingIndex] = useState(null);
  const [saving, setSaving] = useState(false);

  const handleLessonFieldChange = (field, value) => {
    setLessonData(prev => ({ ...prev, [field]: value }));
  };

  const handleQuestionFieldChange = (field, value) => {
    setCurrentQuestion(prev => ({ ...prev, [field]: value }));
  };

  const handleOptionChange = (index, value) => {
    setCurrentQuestion(prev => {
      const newOptions = [...prev.options];
      newOptions[index] = value;
      return { ...prev, options: newOptions };
    });
  };

  const addQuestion = () => {
    if (!currentQuestion.question || !currentQuestion.correct_answer) {
      alert('Please fill in question and correct answer');
      return;
    }

    const newQuestion = {
      ...currentQuestion,
      id: editingIndex !== null ? editingIndex + 1 : lessonData.questions.length + 1
    };

    if (editingIndex !== null) {
      const updated = [...lessonData.questions];
      updated[editingIndex] = newQuestion;
      setLessonData(prev => ({ ...prev, questions: updated }));
      setEditingIndex(null);
    } else {
      setLessonData(prev => ({
        ...prev,
        questions: [...prev.questions, newQuestion]
      }));
    }

    // Reset form
    setCurrentQuestion({
      type: 'text',
      question: '',
      options: ['', '', '', ''],
      correct_answer: '',
      explanation: '',
      time_limit: 30,
      image_path: ''
    });
  };

  const editQuestion = (index) => {
    setCurrentQuestion(lessonData.questions[index]);
    setEditingIndex(index);
  };

  const deleteQuestion = (index) => {
    setLessonData(prev => ({
      ...prev,
      questions: prev.questions.filter((_, i) => i !== index)
    }));
  };

  const saveLesson = async () => {
    if (!lessonData.title || lessonData.questions.length === 0) {
      alert('Please add title and at least one question');
      return;
    }

    setSaving(true);
    try {
      const filename = `${lessonData.title.toLowerCase().replace(/\s+/g, '_')}_lesson.json`;
      const jsonContent = JSON.stringify(lessonData, null, 2);

      await apiService.uploadDocument({
        content: jsonContent,
        filename: filename,
        category: 'lesson',
        tags: lessonData.tags
      });

      alert('Lesson saved successfully!');
      onClose();
    } catch (error) {
      console.error('Failed to save lesson:', error);
      alert('Failed to save lesson');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">Lesson Creator</Typography>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {/* Lesson Info */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>Lesson Information</Typography>
          <TextField
            fullWidth
            label="Title"
            value={lessonData.title}
            onChange={(e) => handleLessonFieldChange('title', e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            label="Description"
            value={lessonData.description}
            onChange={(e) => handleLessonFieldChange('description', e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Introduction"
            value={lessonData.introduction}
            onChange={(e) => handleLessonFieldChange('introduction', e.target.value)}
            sx={{ mb: 2 }}
          />
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Difficulty</InputLabel>
              <Select
                value={lessonData.difficulty}
                onChange={(e) => handleLessonFieldChange('difficulty', e.target.value)}
                label="Difficulty"
              >
                <MenuItem value="easy">Easy</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="hard">Hard</MenuItem>
              </Select>
            </FormControl>
            <TextField
              type="number"
              label="Default Time Limit (seconds)"
              value={lessonData.time_limit_default}
              onChange={(e) => handleLessonFieldChange('time_limit_default', parseInt(e.target.value))}
              sx={{ minWidth: 200 }}
            />
          </Box>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Question Editor */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            {editingIndex !== null ? 'Edit Question' : 'Add Question'}
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Question Type</InputLabel>
            <Select
              value={currentQuestion.type}
              onChange={(e) => handleQuestionFieldChange('type', e.target.value)}
              label="Question Type"
            >
              <MenuItem value="text">Text</MenuItem>
              <MenuItem value="image">Image</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Question"
            value={currentQuestion.question}
            onChange={(e) => handleQuestionFieldChange('question', e.target.value)}
            sx={{ mb: 2 }}
          />
          {currentQuestion.type === 'image' && (
            <TextField
              fullWidth
              label="Image Path (relative, e.g., ./images/puffin.jpg)"
              value={currentQuestion.image_path}
              onChange={(e) => handleQuestionFieldChange('image_path', e.target.value)}
              sx={{ mb: 2 }}
            />
          )}
          <Typography variant="body2" gutterBottom>Options:</Typography>
          {currentQuestion.options.map((option, idx) => (
            <TextField
              key={idx}
              fullWidth
              label={`Option ${idx + 1}`}
              value={option}
              onChange={(e) => handleOptionChange(idx, e.target.value)}
              sx={{ mb: 1 }}
            />
          ))}
          <TextField
            fullWidth
            label="Correct Answer (must match one option exactly)"
            value={currentQuestion.correct_answer}
            onChange={(e) => handleQuestionFieldChange('correct_answer', e.target.value)}
            sx={{ mb: 2, mt: 1 }}
          />
          <TextField
            fullWidth
            multiline
            rows={2}
            label="Explanation"
            value={currentQuestion.explanation}
            onChange={(e) => handleQuestionFieldChange('explanation', e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            type="number"
            label="Time Limit (seconds)"
            value={currentQuestion.time_limit}
            onChange={(e) => handleQuestionFieldChange('time_limit', parseInt(e.target.value))}
            sx={{ mb: 2 }}
          />
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={addQuestion}
          >
            {editingIndex !== null ? 'Update Question' : 'Add Question'}
          </Button>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Questions List */}
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            Questions ({lessonData.questions.length})
          </Typography>
          <List>
            {lessonData.questions.map((q, idx) => (
              <Paper key={idx} sx={{ mb: 1, p: 2 }}>
                <ListItem disableGutters>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip label={q.type} size="small" />
                        <Typography variant="body1">{q.question}</Typography>
                      </Box>
                    }
                    secondary={`Answer: ${q.correct_answer}`}
                  />
                  <ListItemSecondaryAction>
                    <IconButton onClick={() => editQuestion(idx)}>
                      Edit
                    </IconButton>
                    <IconButton onClick={() => deleteQuestion(idx)}>
                      <DeleteIcon />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              </Paper>
            ))}
          </List>
        </Box>
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          startIcon={<SaveIcon />}
          onClick={saveLesson}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save Lesson'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default LessonCreator;
