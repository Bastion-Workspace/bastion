"""
Learning API - Endpoints for lesson progress tracking
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

from services.auth_service import get_current_user
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class LearningProgressCreate(BaseModel):
    """Model for creating a learning progress record"""
    lesson_document_id: str = Field(..., description="Document ID of the lesson")
    score: int = Field(..., ge=0, description="Score achieved (points earned)")
    total_questions: int = Field(..., gt=0, description="Total number of questions")
    max_score: int = Field(..., gt=0, description="Maximum possible score (total_questions * 3)")
    time_taken_seconds: int = Field(..., ge=0, description="Time taken in seconds")


class LearningProgressResponse(BaseModel):
    """Model for learning progress response"""
    id: int
    user_id: str
    lesson_document_id: str
    score: int
    total_questions: int
    max_score: int
    percentage: float
    completed_at: datetime
    time_taken_seconds: int


@router.post("/learning/progress", response_model=LearningProgressResponse)
async def record_learning_progress(
    progress: LearningProgressCreate,
    current_user: dict = Depends(get_current_user)
):
    """
    Record user's quiz completion and score
    
    Stores the completion data for analytics and progress tracking.
    """
    try:
        import asyncpg
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        try:
            # Set RLS context for Row-Level Security policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", current_user['user_id'])
            await conn.execute("SELECT set_config('app.current_user_role', $1, false)", current_user.get('role', 'user'))
            
            row = await conn.fetchrow(
                """
                INSERT INTO learning_progress 
                (user_id, lesson_document_id, score, total_questions, max_score, time_taken_seconds)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id, user_id, lesson_document_id, score, total_questions, max_score, 
                          percentage, completed_at, time_taken_seconds
                """,
                current_user['user_id'],
                progress.lesson_document_id,
                progress.score,
                progress.total_questions,
                progress.max_score,
                progress.time_taken_seconds
            )
            
            return {
                "id": row['id'],
                "user_id": row['user_id'],
                "lesson_document_id": row['lesson_document_id'],
                "score": row['score'],
                "total_questions": row['total_questions'],
                "max_score": row['max_score'],
                "percentage": float(row['percentage']),
                "completed_at": row['completed_at'],
                "time_taken_seconds": row['time_taken_seconds']
            }
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to record learning progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record progress: {str(e)}")


@router.get("/learning/progress", response_model=List[LearningProgressResponse])
async def get_user_learning_progress(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's learning progress history
    
    Returns recent quiz completions for the current user.
    """
    try:
        import asyncpg
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        try:
            # Set RLS context for Row-Level Security policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", current_user['user_id'])
            await conn.execute("SELECT set_config('app.current_user_role', $1, false)", current_user.get('role', 'user'))
            
            rows = await conn.fetch(
                """
                SELECT id, user_id, lesson_document_id, score, total_questions, max_score,
                       percentage, completed_at, time_taken_seconds
                FROM learning_progress
                WHERE user_id = $1
                ORDER BY completed_at DESC
                LIMIT $2
                """,
                current_user['user_id'], limit
            )
            
            return [
                {
                    "id": row['id'],
                    "user_id": row['user_id'],
                    "lesson_document_id": row['lesson_document_id'],
                    "score": row['score'],
                    "total_questions": row['total_questions'],
                    "max_score": row['max_score'],
                    "percentage": float(row['percentage']),
                    "completed_at": row['completed_at'],
                    "time_taken_seconds": row['time_taken_seconds']
                }
                for row in rows
            ]
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to get learning progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")


@router.get("/learning/progress/{lesson_document_id}", response_model=List[LearningProgressResponse])
async def get_lesson_progress(
    lesson_document_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's progress for a specific lesson
    
    Returns all attempts for a specific lesson by the current user.
    """
    try:
        import asyncpg
        connection_string = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        conn = await asyncpg.connect(connection_string)
        try:
            # Set RLS context for Row-Level Security policies
            await conn.execute("SELECT set_config('app.current_user_id', $1, false)", current_user['user_id'])
            await conn.execute("SELECT set_config('app.current_user_role', $1, false)", current_user.get('role', 'user'))
            
            rows = await conn.fetch(
                """
                SELECT id, user_id, lesson_document_id, score, total_questions, max_score,
                       percentage, completed_at, time_taken_seconds
                FROM learning_progress
                WHERE user_id = $1 AND lesson_document_id = $2
                ORDER BY completed_at DESC
                """,
                current_user['user_id'], lesson_document_id
            )
            
            return [
                {
                    "id": row['id'],
                    "user_id": row['user_id'],
                    "lesson_document_id": row['lesson_document_id'],
                    "score": row['score'],
                    "total_questions": row['total_questions'],
                    "max_score": row['max_score'],
                    "percentage": float(row['percentage']),
                    "completed_at": row['completed_at'],
                    "time_taken_seconds": row['time_taken_seconds']
                }
                for row in rows
            ]
        finally:
            await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to get lesson progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lesson progress: {str(e)}")
