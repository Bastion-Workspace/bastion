-- Learning Progress Tracking
-- Stores user quiz completion data for analytics and progress tracking

CREATE TABLE IF NOT EXISTS learning_progress (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    lesson_document_id VARCHAR(255) NOT NULL,
    score INTEGER NOT NULL CHECK (score >= 0),
    total_questions INTEGER NOT NULL CHECK (total_questions > 0),
    max_score INTEGER NOT NULL CHECK (max_score > 0),
    percentage DECIMAL(5,2) GENERATED ALWAYS AS ((score::DECIMAL / max_score::DECIMAL) * 100) STORED,
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    time_taken_seconds INTEGER CHECK (time_taken_seconds >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_user_lesson_completion UNIQUE(user_id, lesson_document_id, completed_at)
);

-- Indexes for efficient queries
CREATE INDEX idx_learning_progress_user ON learning_progress(user_id);
CREATE INDEX idx_learning_progress_lesson ON learning_progress(lesson_document_id);
CREATE INDEX idx_learning_progress_completed_at ON learning_progress(completed_at DESC);
CREATE INDEX idx_learning_progress_user_completed ON learning_progress(user_id, completed_at DESC);

-- Comment
COMMENT ON TABLE learning_progress IS 'Tracks user quiz completion and scores for learning analytics';
COMMENT ON COLUMN learning_progress.percentage IS 'Automatically calculated percentage score';
