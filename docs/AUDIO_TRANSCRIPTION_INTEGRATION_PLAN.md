# Audio Transcription Integration Plan

## Overview
Plan for integrating audio transcription capabilities to enable users to upload audio files/podcasts, get transcripts, and add user tagging and notes.

## Current State
- **No existing transcription capabilities** in the current system
- Document processor only handles text-based formats (PDF, DOCX, EPUB, TXT, HTML, email)
- System has robust document processing pipeline that could be extended

## External Transcription Services

### 1. OpenAI Whisper API (Recommended)
- **Best for**: High-quality transcription with speaker detection
- **Features**: 
  - Supports multiple audio formats (MP3, WAV, M4A, etc.)
  - Speaker diarization (identifies different speakers)
  - Multiple language support
  - Timestamp generation
  - Confidence scores
- **Integration**: Direct API calls to OpenAI's Whisper endpoint
- **Cost**: $0.006 per minute
- **Advantage**: Already have OpenAI integration in system

### 2. AssemblyAI
- **Best for**: Advanced features like sentiment analysis, topic detection
- **Features**:
  - Real-time transcription
  - Custom vocabulary training
  - Entity detection
  - Auto-chapters and highlights
  - Sentiment analysis
- **Cost**: $0.00025 per second

### 3. Rev.ai
- **Best for**: Professional-grade transcription with human review options
- **Features**:
  - 99% accuracy guarantee
  - Custom vocabulary
  - Speaker identification
  - Multiple output formats
- **Cost**: $0.25 per minute

### 4. Google Speech-to-Text
- **Best for**: Google ecosystem integration
- **Features**:
  - Real-time transcription
  - Multiple language support
  - Custom models
  - Automatic punctuation
- **Cost**: $0.006 per 15 seconds

## Implementation Architecture

### 1. Audio Upload Endpoint
```python
# New API endpoint in backend/api/
async def upload_audio_for_transcription(
    file: UploadFile,
    transcription_service: str = "whisper"
) -> TranscriptionResult
```

### 2. Transcription Service
```python
# New service in backend/services/
class TranscriptionService:
    async def transcribe_audio(self, audio_file: bytes, service: str) -> str
    async def process_transcript(self, transcript: str) -> List[Chunk]
```

### 3. Integration with Existing Pipeline
- Upload audio file
- Send to transcription service
- Receive transcript with timestamps
- Process transcript through existing document pipeline
- Store in vector database for search/querying

### 4. User Tagging Features
- Allow users to add tags to specific timestamp ranges
- Enable note-taking on transcript segments
- Support for highlighting important sections
- Integration with existing knowledge graph

## Frontend Components Needed

### 1. Audio Upload Interface
- Drag-and-drop audio file upload
- Progress indicator for transcription
- Format validation (MP3, WAV, M4A, etc.)

### 2. Transcript Viewer
- Timestamped transcript display
- Speaker identification highlighting
- Click-to-play audio segments
- Search within transcript

### 3. Tagging Interface
- Add tags to timestamp ranges
- Note-taking on specific segments
- Highlight important sections
- Export tagged segments

## Database Schema Extensions

### 1. Audio Documents Table
```sql
CREATE TABLE audio_documents (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    original_filename VARCHAR(255),
    audio_file_path VARCHAR(500),
    transcript_text TEXT,
    transcription_metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### 2. Audio Tags Table
```sql
CREATE TABLE audio_tags (
    id UUID PRIMARY KEY,
    audio_document_id UUID REFERENCES audio_documents(id),
    user_id UUID REFERENCES users(id),
    start_timestamp DECIMAL(10,3),
    end_timestamp DECIMAL(10,3),
    tag_text VARCHAR(255),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Implementation Priority

### Phase 1: Basic Transcription
1. Implement OpenAI Whisper API integration
2. Create audio upload endpoint
3. Basic transcript display
4. Integration with existing document pipeline

### Phase 2: User Tagging
1. Add tagging interface
2. Implement timestamp-based tagging
3. Note-taking capabilities
4. Tag search and filtering

### Phase 3: Advanced Features
1. Speaker identification UI
2. Audio playback integration
3. Export capabilities
4. Advanced search within transcripts

## Technical Considerations

### File Size Limits
- Current upload limit: 1500MB (sufficient for audio files)
- Consider chunked uploads for very large audio files

### Processing Time
- Transcription can take significant time for long audio files
- Implement background job processing
- Use existing WebSocket infrastructure for progress updates

### Storage
- Audio files may be large
- Consider cloud storage options for audio files
- Keep transcripts in database for fast search

### Cost Management
- Monitor transcription API usage
- Implement user quotas if needed
- Consider caching frequently accessed transcripts

## Future Enhancements

### 1. Real-time Transcription
- Live audio streaming
- Real-time transcript display
- Live tagging capabilities

### 2. Multi-language Support
- Automatic language detection
- Translation capabilities
- Multi-language search

### 3. Advanced Analytics
- Speaker analytics
- Topic modeling
- Sentiment analysis
- Conversation flow analysis

### 4. Integration with Existing Features
- Knowledge graph integration
- Research plan generation from audio content
- Chat integration for audio-based queries
- Document linking to audio segments

## Dependencies

### Backend Dependencies
- OpenAI API (already integrated)
- Audio file processing libraries
- Background job processing (already available)

### Frontend Dependencies
- Audio player components
- Timeline visualization
- Tag management interface

## Security Considerations

### File Upload Security
- Validate audio file formats
- Scan for malicious content
- Implement file size limits
- Secure file storage

### Privacy
- Audio content may contain sensitive information
- Implement proper access controls
- Consider encryption for audio files
- User consent for audio processing

## Success Metrics

### User Engagement
- Number of audio files uploaded
- Time spent with transcript interface
- Tag usage frequency
- Search queries on audio content

### Technical Performance
- Transcription accuracy
- Processing time
- API cost per transcription
- Storage efficiency

---

*This plan provides a comprehensive roadmap for implementing audio transcription capabilities while leveraging the existing robust document processing infrastructure.* 