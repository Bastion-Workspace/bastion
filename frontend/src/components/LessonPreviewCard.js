import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Chip
} from '@mui/material';
import SchoolIcon from '@mui/icons-material/School';
import TimerIcon from '@mui/icons-material/Timer';
import QuizIcon from '@mui/icons-material/Quiz';
import { useLearning } from '../contexts/LearningContext';

const LessonPreviewCard = ({ lesson }) => {
  const { startQuiz } = useLearning();

  if (!lesson) return null;

  const difficultyColor = {
    'easy': 'success',
    'medium': 'warning',
    'hard': 'error'
  }[lesson.difficulty?.toLowerCase()] || 'default';

  const questionCount = lesson.questions?.length || 0;
  const imageQuestions = lesson.questions?.filter(q => q.type === 'image').length || 0;
  const textQuestions = questionCount - imageQuestions;

  return (
    <Card
      sx={{
        mb: 2,
        borderRadius: 2,
        boxShadow: 3,
        '&:hover': {
          boxShadow: 6
        }
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <SchoolIcon color="primary" />
          <Typography variant="h6" component="div">
            {lesson.title}
          </Typography>
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {lesson.description}
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip
            label={lesson.difficulty || 'Medium'}
            size="small"
            color={difficultyColor}
          />
          <Chip
            icon={<QuizIcon />}
            label={`${questionCount} question${questionCount !== 1 ? 's' : ''}`}
            size="small"
            variant="outlined"
          />
          {imageQuestions > 0 && (
            <Chip
              label={`${imageQuestions} image${imageQuestions !== 1 ? 's' : ''}`}
              size="small"
              variant="outlined"
              color="primary"
            />
          )}
          {textQuestions > 0 && (
            <Chip
              label={`${textQuestions} text`}
              size="small"
              variant="outlined"
              color="secondary"
            />
          )}
          {lesson.time_limit_default && (
            <Chip
              icon={<TimerIcon />}
              label={`${lesson.time_limit_default}s per question`}
              size="small"
              variant="outlined"
            />
          )}
        </Box>

        {lesson.introduction && (
          <Typography
            variant="body2"
            sx={{
              mt: 2,
              p: 2,
              bgcolor: 'action.hover',
              borderRadius: 1,
              fontStyle: 'italic'
            }}
          >
            {lesson.introduction.length > 200
              ? lesson.introduction.substring(0, 200) + '...'
              : lesson.introduction}
          </Typography>
        )}
      </CardContent>

      <CardActions sx={{ px: 2, pb: 2 }}>
        <Button
          variant="contained"
          size="large"
          fullWidth
          onClick={() => startQuiz(lesson)}
          startIcon={<SchoolIcon />}
        >
          Start Quiz
        </Button>
      </CardActions>
    </Card>
  );
};

export default LessonPreviewCard;
