# CASH (Compact Analysis System Hub)

A powerful tool for analyzing YouTube videos using AI, featuring intelligent chat capabilities and vector-based search.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)

## Features

- **Video Processing**
  - Single video upload and analysis
  - Multiple videos batch processing with sequential queue
  - YouTube playlist support
  - Automatic transcription and content extraction
  - File upload API for direct video uploads
  - Intelligent video processing queue management

- **AI-Powered Analysis**
  - Advanced language model integration
  - Vector-based similarity search
  - Intelligent content summarization
  - Context-aware chat interface

- **Enhanced User Interface**
  - Modern, responsive web interface
  - Real-time processing status updates
  - Enhanced console log with scroll, color coding, and timestamps
  - Interactive chat with processed content
  - Video processing queue status monitoring
  - Easy-to-use upload forms

- **System Management**
  - Automated file cleanup service (Janitor) with cron scheduling
  - Configurable retention policies and smart file filtering
  - Web UI controls for service management
  - Background maintenance tasks
  - Queue management and monitoring

- **API Integration**
  - RESTful API for video upload and analysis
  - Queue management endpoints
  - File cleanup management
  - Real-time status monitoring

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- YouTube Data API key (optional, for enhanced playlist features)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-analyzer.git
cd video-analyzer
```

2. Run the setup script:
```bash
# On Linux/Mac:
chmod +x setup.sh
./setup.sh

# On Windows (using Git Bash or WSL):
bash setup.sh
```

The setup script will:
- Verify Python installation
- Check project structure
- Create and activate a virtual environment
- Install all dependencies
- Verify Python imports
- Provide detailed diagnostic information if anything goes wrong

3. Start the server:
```bash
# Activate the virtual environment (if not already activated)
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\activate  # On Windows

# Start the server from the project root
python -m app.main
```

> ⚠️ **Important**: Always run the server from the project root directory, not from within the `app` directory.

## Usage

### Single Video Analysis
1. Paste a YouTube video URL in the single video form
2. Click "Upload Video"
3. Monitor processing status in real-time
4. Once complete, use the chat interface to interact with the video content

### Playlist Processing
1. Enter a YouTube playlist URL in the playlist form
2. Click "Upload Playlist"
3. Track progress of individual videos
4. Chat with content from all processed videos

### Multiple Videos
1. Enter multiple YouTube URLs (one per line)
2. Click "Upload Videos"
3. Monitor batch processing status in the enhanced console
4. Videos are processed sequentially to avoid rate limits
5. Interact with combined content through chat

### File Upload API
Upload video files directly via API:
```bash
curl -X POST "http://localhost:8000/api/upload-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video_file=@your_video.mp4" \
  -F "title=My Video" \
  -F "description=Video description" \
  -F "priority=1"
```

### URL Analysis API
Analyze videos from URLs:
```bash
curl -X POST "http://localhost:8000/api/analyze-url" \
  -H "accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "url=https://www.youtube.com/watch?v=VIDEO_ID&title=My Video&priority=1"
```

## Architecture

### Components

- **FastAPI Backend**
  - RESTful API endpoints
  - Async video processing
  - WebSocket support for real-time updates
  - Static file serving

- **Video Processing Pipeline**
  - YouTube video download
  - Audio extraction
  - Transcription processing
  - Content analysis

- **Vector Store**
  - Efficient similarity search
  - Content embedding storage
  - Fast query processing

- **LLM Service**
  - Context-aware responses
  - Content summarization
  - Natural language understanding

## Configuration

The system is configured via `config.yaml`. Key configuration sections:

### File Cleanup (Janitor) Settings
```yaml
janitor:
  enabled: true
  schedule: "0 1 * * *"  # Daily at 1:00 AM (cron syntax)
  cleanup_paths:
    - "downloads"
    - "temp"
    - "transcripts"
  file_patterns:
    - "*.mp4"
    - "*.wav"
    - "*.txt"
    - "*.json"
  retention_hours: 168  # 7 days
  dry_run: false
  max_file_size_mb: null  # Optional file size limit
  exclude_patterns: []  # Files to exclude from cleanup
  log_deletions: true  # Log each file deletion
  preserve_recent_files: true  # Keep files from last 24h regardless of retention
```

### Storage Settings
```yaml
storage:
  base_path: "./storage"
  downloads_path: "downloads"
  transcripts_path: "transcripts"
```

### API Settings
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"
```

## API Documentation

Access the interactive API documentation at:
```
http://localhost:8000/docs
```

### Core Video Processing Endpoints
- `POST /analyze`: Process single video
- `POST /analyze-playlist`: Process YouTube playlist
- `POST /api/upload-video`: Upload and analyze video file
- `POST /api/analyze-url`: Analyze video from URL
- `POST /chat`: Interact with processed content
- `GET /status/{video_id}`: Check processing status

### Queue Management Endpoints
- `GET /api/queue/status`: Get video processing queue status
- `GET /api/queue/video/{video_id}`: Get specific video queue info

### File Cleanup (Janitor) Endpoints
- `GET /api/janitor/status`: Get service status and configuration
- `POST /api/janitor/config`: Update runtime configuration
- `POST /api/janitor/cleanup/manual`: Trigger manual cleanup
- `GET /api/janitor/cleanup/preview`: Preview cleanup without deletion
- `POST /api/janitor/start`: Start the janitor service
- `POST /api/janitor/stop`: Stop the janitor service

## Enhanced Features

### Enhanced Console Features

The web interface now includes an enhanced console log with:
- **Scrollable output**: 300px height with smooth scrolling
- **Color-coded log levels**: INFO (blue), SUCCESS (green), WARNING (yellow), ERROR (red), DEBUG (purple)
- **Timestamps**: ISO format timestamps for all entries
- **Auto-scroll toggle**: Enable/disable auto-scrolling to latest entries
- **Clear console**: Button to clear all log entries
- **Queue monitoring**: Real-time queue status display
- **Memory management**: Automatic cleanup of old entries (max 1000)

### Video Processing Queue

Videos are now processed sequentially to avoid:
- Rate limiting issues
- System resource bottlenecks
- Parallel processing conflicts

Queue features:
- **Priority processing**: Higher priority videos processed first
- **Status tracking**: Real-time status updates for each video
- **Error handling**: Failed videos don't block the queue
- **Progress monitoring**: Track position and processing time

### Automated File Cleanup (Janitor)

The janitor service automatically manages storage with advanced features:
- **Cron-based scheduling**: Configurable cleanup schedules using cron syntax
- **Smart file filtering**: Exclusion patterns, size limits, and recent file preservation
- **Retention policies**: Keep files for configurable time periods
- **Web UI management**: Full control panel with real-time status and configuration
- **Preview mode**: Safe cleanup preview before actual deletion
- **Runtime configuration**: Update settings without restarting the service
- **Detailed logging**: Comprehensive deletion tracking and error reporting
- **Manual triggers**: API endpoints for on-demand cleanup

#### Janitor Web Interface Features
- **Service Controls**: Start/stop service with one click
- **Configuration Panel**: Adjust retention hours, file size limits, and cleanup schedule
- **Cleanup Options**: Preview what would be deleted or run manual cleanup
- **Status Display**: Real-time service status and last cleanup results
- **Results Visualization**: Detailed statistics showing files deleted, space freed, and errors

### File Upload Support

Direct file upload capabilities:
- **Multi-format support**: Accept various video file formats
- **API integration**: RESTful endpoints for programmatic uploads
- **Priority queuing**: Set processing priority for uploaded files
- **Metadata support**: Add titles and descriptions during upload

## Development

### Project Structure
```
video-analyzer/
├── app/
│   ├── static/
│   │   ├── index.html
│   │   └── app.js
│   ├── services/
│   │   ├── video_processor.py
│   │   ├── llm_service.py
│   │   ├── vector_store.py
│   │   └── model_manager.py
│   └── main.py
├── requirements.txt
├── config.yaml
└── README.md
```

### Adding New Features

1. Create feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Implement changes
3. Update tests
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- OpenAI for language model support
- FastAPI for the web framework
- CopilotKit for enhanced chat capabilities
