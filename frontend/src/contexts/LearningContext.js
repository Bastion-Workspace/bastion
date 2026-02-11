import React, { createContext, useContext, useState, useCallback } from 'react';

const LearningContext = createContext();

export const useLearning = () => {
  const context = useContext(LearningContext);
  if (!context) {
    throw new Error('useLearning must be used within a LearningProvider');
  }
  return context;
};

export const LearningProvider = ({ children }) => {
  const [quizState, setQuizState] = useState({
    isOpen: false,
    lesson: null,
    currentQuestionIndex: 0,
    score: 0,
    answers: [],
    startTime: null
  });

  const startQuiz = useCallback((lessonData) => {
    console.log('Starting quiz with lesson:', lessonData.title);
    setQuizState({
      isOpen: true,
      lesson: lessonData,
      currentQuestionIndex: 0,
      score: 0,
      answers: [],
      startTime: Date.now()
    });
  }, []);

  const closeQuiz = useCallback(() => {
    console.log('Closing quiz');
    setQuizState({
      isOpen: false,
      lesson: null,
      currentQuestionIndex: 0,
      score: 0,
      answers: [],
      startTime: null
    });
  }, []);

  const nextQuestion = useCallback(() => {
    setQuizState(prev => {
      const nextIndex = prev.currentQuestionIndex + 1;
      if (nextIndex >= prev.lesson.questions.length) {
        return prev;
      }
      return {
        ...prev,
        currentQuestionIndex: nextIndex
      };
    });
  }, []);

  const addScore = useCallback((points) => {
    setQuizState(prev => ({
      ...prev,
      score: prev.score + points
    }));
  }, []);

  const recordAnswer = useCallback((questionId, answer, isCorrect, attemptsUsed) => {
    setQuizState(prev => ({
      ...prev,
      answers: [...prev.answers, {
        questionId,
        answer,
        isCorrect,
        attemptsUsed,
        timestamp: Date.now()
      }]
    }));
  }, []);

  const value = {
    quizState,
    startQuiz,
    closeQuiz,
    nextQuestion,
    addScore,
    recordAnswer
  };

  return (
    <LearningContext.Provider value={value}>
      {children}
    </LearningContext.Provider>
  );
};
