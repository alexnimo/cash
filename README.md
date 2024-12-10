# CASH (Compact Analysis System Hub)

A powerful tool for analyzing YouTube videos using AI, featuring intelligent chat capabilities and vector-based search.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green)

## Features

- **Video Processing**
  - Single video upload and analysis
  - Multiple videos batch processing
  - YouTube playlist support
  - Automatic transcription and content extraction

- **AI-Powered Analysis**
  - Advanced language model integration
  - Vector-based similarity search
  - Intelligent content summarization
  - Context-aware chat interface

- **User Interface**
  - Modern, responsive web interface
  - Real-time processing status updates
  - Interactive chat with processed content
  - Easy-to-use upload forms

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
3. Monitor batch processing status
4. Interact with combined content through chat

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

## API Documentation

Access the interactive API documentation at:
```
http://localhost:8000/docs
```

Key endpoints:
- `POST /analyze`: Process single video
- `POST /analyze-playlist`: Process YouTube playlist
- `POST /chat`: Interact with processed content
- `GET /status/{video_id}`: Check processing status

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
