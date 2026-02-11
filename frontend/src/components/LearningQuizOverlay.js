import React, { useState, useEffect, useCallback } from 'react';
import {
  Dialog,
  DialogContent,
  Box,
  Typography,
  LinearProgress,
  List,
  ListItemButton,
  ListItemText,
  Button,
  Alert,
  Chip,
  IconButton,
  Paper
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import { useLearning } from '../contexts/LearningContext';
import { useTheme } from '../contexts/ThemeContext';
import apiService from '../services/apiService';

const LearningQuizOverlay = () => {
  const { quizState, closeQuiz, nextQuestion, addScore, recordAnswer } = useLearning();
  const { isDarkMode } = useTheme();
  const [currentAttempts, setCurrentAttempts] = useState(0);
  const [timeRemaining, setTimeRemaining] = useState(30);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [isCorrect, setIsCorrect] = useState(null);
  const [quizComplete, setQuizComplete] = useState(false);

  const currentQuestion = quizState.lesson?.questions?.[quizState.currentQuestionIndex];
  const isLastQuestion = quizState.currentQuestionIndex === (quizState.lesson?.questions?.length - 1);
  const totalQuestions = quizState.lesson?.questions?.length || 0;
  const maxScore = totalQuestions * 3;

  // Reset state when question changes
  useEffect(() => {
    if (quizState.isOpen && currentQuestion) {
      setCurrentAttempts(0);
      setTimeRemaining(currentQuestion.time_limit || 30);
      setSelectedAnswer(null);
      setShowExplanation(false);
      setIsCorrect(null);
    }
  }, [quizState.currentQuestionIndex, quizState.isOpen, currentQuestion]);

  // Timer logic
  useEffect(() => {
    if (!quizState.isOpen || showExplanation || quizComplete) return;

    const timer = setInterval(() => {
      setTimeRemaining(prev => {
        if (prev <= 1) {
          handleTimeOut();
          return currentQuestion?.time_limit || 30;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [quizState.isOpen, showExplanation, quizState.currentQuestionIndex, quizComplete, currentQuestion]);

  const handleTimeOut = () => {
    setCurrentAttempts(prev => {
      const newAttempts = prev + 1;
      if (newAttempts >= 3) {
        setShowExplanation(true);
        setIsCorrect(false);
        recordAnswer(currentQuestion.id, null, false, newAttempts);
      }
      return newAttempts;
    });
  };

  const handleAnswerSelect = useCallback((answer) => {
    if (showExplanation || currentAttempts >= 3) return;

    setSelectedAnswer(answer);
    const correct = answer === currentQuestion.correct_answer;
    setIsCorrect(correct);

    if (correct) {
      const points = 3 - currentAttempts;
      addScore(points);
      setShowExplanation(true);
      recordAnswer(currentQuestion.id, answer, true, currentAttempts + 1);
      console.log(`Correct! Earned ${points} points (attempt ${currentAttempts + 1})`);
    } else {
      const newAttempts = currentAttempts + 1;
      setCurrentAttempts(newAttempts);
      
      if (newAttempts >= 3) {
        setShowExplanation(true);
        recordAnswer(currentQuestion.id, answer, false, newAttempts);
        console.log('Failed all 3 attempts');
      } else {
        console.log(`Incorrect. ${3 - newAttempts} attempts remaining`);
        setTimeout(() => {
          setSelectedAnswer(null);
          setIsCorrect(null);
        }, 800);
      }
    }
  }, [currentAttempts, showExplanation, currentQuestion, addScore, recordAnswer]);

  const handleNextQuestion = async () => {
    if (isLastQuestion) {
      setQuizComplete(true);
      
      // Record progress
      try {
        const timeTaken = Math.floor((Date.now() - quizState.startTime) / 1000);
        await apiService.post('/learning/progress', {
          lesson_document_id: quizState.lesson.document_id,
          score: quizState.score,
          total_questions: totalQuestions,
          max_score: maxScore,
          time_taken_seconds: timeTaken
        });
        console.log('Progress recorded successfully');
      } catch (error) {
        console.error('Failed to record progress:', error);
      }
    } else {
      nextQuestion();
    }
  };

  const handleCloseQuiz = () => {
    setQuizComplete(false);
    closeQuiz();
  };

  const renderCompletionScreen = () => {
    const timeTaken = Math.floor((Date.now() - quizState.startTime) / 1000);
    const minutes = Math.floor(timeTaken / 60);
    const seconds = timeTaken % 60;
    const percentage = ((quizState.score / maxScore) * 100).toFixed(1);

    return (
      <Box sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h4" gutterBottom>
          🎉 Quiz Complete!
        </Typography>
        <Typography variant="h2" sx={{ my: 3, fontWeight: 'bold', color: 'primary.main' }}>
          {quizState.score} / {maxScore}
        </Typography>
        <Typography variant="h6" gutterBottom>
          {percentage}% Correct
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
          Time: {minutes}:{seconds.toString().padStart(2, '0')}
        </Typography>
        <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button variant="contained" onClick={handleCloseQuiz}>
            Close
          </Button>
        </Box>
      </Box>
    );
  };

  if (!quizState.isOpen) return null;

  if (quizComplete) {
    return (
      <Dialog
        open={true}
        onClose={handleCloseQuiz}
        maxWidth="sm"
        fullWidth
      >
        <DialogContent>
          {renderCompletionScreen()}
        </DialogContent>
      </Dialog>
    );
  }

  if (!currentQuestion) return null;

  const progress = ((quizState.currentQuestionIndex + 1) / totalQuestions) * 100;
  const timerProgress = (timeRemaining / (currentQuestion.time_limit || 30)) * 100;

  return (
    <Dialog
      open={quizState.isOpen}
      onClose={handleCloseQuiz}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: {
          height: '90vh',
          bgcolor: isDarkMode ? 'background.paper' : 'background.default'
        }
      }}
    >
      {/* Header */}
      <Box sx={{
        p: 2,
        borderBottom: '1px solid',
        borderColor: 'divider',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h5" gutterBottom>
            {quizState.lesson?.title}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 1 }}>
            <Chip
              label={`Question ${quizState.currentQuestionIndex + 1} of ${totalQuestions}`}
              size="small"
              color="primary"
            />
            <Chip
              label={`Score: ${quizState.score}`}
              size="small"
              color="secondary"
            />
          </Box>
          <LinearProgress
            variant="determinate"
            value={progress}
            sx={{ height: 8, borderRadius: 1 }}
          />
        </Box>
        <IconButton onClick={handleCloseQuiz} sx={{ ml: 2 }}>
          <CloseIcon />
        </IconButton>
      </Box>

      {/* Main Content */}
      <DialogContent sx={{ display: 'flex', p: 0, height: 'calc(100% - 120px)' }}>
        {/* Left Side - Image or Text Content */}
        <Box sx={{
          flex: 1,
          p: 4,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRight: '1px solid',
          borderColor: 'divider',
          bgcolor: isDarkMode ? 'background.default' : 'grey.50'
        }}>
          {currentQuestion.type === 'image' && currentQuestion.image_url ? (
            <img
              src={currentQuestion.image_url}
              alt="Question"
              style={{
                maxWidth: '100%',
                maxHeight: '100%',
                objectFit: 'contain',
                borderRadius: '8px'
              }}
            />
          ) : (
            <Paper elevation={0} sx={{ p: 4, bgcolor: 'transparent' }}>
              <Typography variant="h4" align="center" color="text.secondary">
                {currentQuestion.question}
              </Typography>
            </Paper>
          )}
        </Box>

        {/* Right Side - Question and Options */}
        <Box sx={{ width: 450, p: 3, display: 'flex', flexDirection: 'column' }}>
          {/* Question Text (for image questions) */}
          {currentQuestion.type === 'image' && (
            <Typography variant="h6" sx={{ mb: 3 }}>
              {currentQuestion.question}
            </Typography>
          )}

          {/* Options */}
          <List sx={{ flex: 1, overflow: 'auto' }}>
            {currentQuestion.options?.map((option, idx) => {
              const isSelected = selectedAnswer === option;
              const isCorrectOption = option === currentQuestion.correct_answer;
              const showResult = showExplanation || isCorrect !== null;

              return (
                <ListItemButton
                  key={idx}
                  onClick={() => handleAnswerSelect(option)}
                  disabled={currentAttempts >= 3 || showExplanation}
                  selected={isSelected}
                  sx={{
                    mb: 1,
                    borderRadius: 1,
                    border: '2px solid',
                    borderColor: showResult && isSelected
                      ? (isCorrect ? 'success.main' : 'error.main')
                      : 'divider',
                    bgcolor: showResult && isSelected
                      ? (isCorrect ? 'success.light' : 'error.light')
                      : 'background.paper',
                    '&:hover': {
                      bgcolor: showResult && isSelected
                        ? (isCorrect ? 'success.light' : 'error.light')
                        : 'action.hover'
                    }
                  }}
                >
                  <ListItemText
                    primary={option}
                    primaryTypographyProps={{
                      fontWeight: isSelected ? 'bold' : 'normal'
                    }}
                  />
                  {showExplanation && isCorrectOption && (
                    <CheckCircleIcon color="success" />
                  )}
                  {showExplanation && isSelected && !isCorrect && (
                    <CancelIcon color="error" />
                  )}
                </ListItemButton>
              );
            })}
          </List>

          {/* Timer */}
          {!showExplanation && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress
                variant="determinate"
                value={timerProgress}
                color={timeRemaining < 10 ? 'error' : 'primary'}
                sx={{ height: 8, borderRadius: 1 }}
              />
              <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                Time: {timeRemaining}s
              </Typography>
            </Box>
          )}

          {/* Attempts */}
          {!showExplanation && (
            <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary' }}>
              Attempts remaining: {3 - currentAttempts}
            </Typography>
          )}

          {/* Explanation */}
          {showExplanation && (
            <Alert
              severity={isCorrect ? 'success' : 'info'}
              sx={{ mt: 2 }}
            >
              <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                {isCorrect ? '✓ Correct!' : '✗ Incorrect'}
              </Typography>
              <Typography variant="body2">
                {currentQuestion.explanation}
              </Typography>
            </Alert>
          )}

          {/* Next Button */}
          {showExplanation && (
            <Button
              fullWidth
              variant="contained"
              onClick={handleNextQuestion}
              sx={{ mt: 2 }}
              size="large"
            >
              {isLastQuestion ? 'See Results' : 'Next Question'}
            </Button>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
};

export default LearningQuizOverlay;
