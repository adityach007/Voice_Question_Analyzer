# Voice Question Analyzer 🎤

A sophisticated voice-driven application that transcribes speech, extracts questions, and provides intelligent answers using RAG (Retrieval Augmented Generation) and LLM technologies.

https://drive.google.com/file/d/1Prz6fATFqlFf-NENeSdahGgUiPKVnMi4/view?usp=sharing

## 🌟 Features

- **Real-time Voice Recording**: Capture audio input with high-quality sampling
- **Speech-to-Text**: Advanced transcription using Groq's Whisper model
- **Question Detection**: Automated extraction of both explicit and implicit questions
- **Intelligent Answering**: Combines document retrieval and web search for comprehensive responses
- **Beautiful UI**: Streamlit-based interface with custom styling
- **Fallback Mechanisms**: Local processing capabilities when cloud services are unavailable

## 🛠️ Technical Architecture

### Components

1. **Main Application (`main.py`)**
   - Streamlit-based user interface
   - Audio recording management
   - Real-time status updates
   - Results display with expandable sections

2. **Audio Handler (`audio_handler.py`)**
   - Audio processing and conversion
   - Primary transcription using Groq
   - Fallback to local Whisper model
   - Temporary file management

3. **RAG Agent (`rag_agent.py`)**
   - Document retrieval system
   - Vector store management
   - Multiple search strategies
   - Question processing pipeline

## 📋 Prerequisites

```bash
python >= 3.8
streamlit
sounddevice
numpy
groq
langchain
transformers
faiss-cpu
sentence-transformers
duckduckgo-search
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/adityach007/Stealth_Assignment/tree/main
cd voice-question-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GROQ_API_KEY="your_groq_api_key"
# or on Windows
set GROQ_API_KEY=your_groq_api_key
```

## 💻 Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the application:
   - Click "Start Recording" to begin voice capture
   - Speak your questions naturally
   - Click "Stop Recording" to process the audio
   - View transcription and answers in the results panel

## 🔧 Configuration

### Audio Settings
- Sample Rate: 16000 Hz
- Channels: 1 (Mono)
- Format: WAV

### RAG Configuration
- Model: Groq Mixtral-8x7b-32768
- Temperature: 0.7
- Vector Store: FAISS
- Embeddings: all-MiniLM-L6-v2

## 📚 Document Management

The system includes built-in knowledge about:
- Automotive specifications
- Vehicle features
- Safety information
- Maintenance guidelines

### Search Strategies

1. **Semantic Search**
   - Dense vector similarity matching
   - Used for document-specific queries

2. **Keyword Search**
   - Traditional text matching
   - Fallback for specific terms

3. **Hybrid Search**
   - Combines both approaches
   - Used for complex queries

## 🎯 Feature Details

### Voice Recording
- Real-time audio capture
- Visual feedback during recording
- Automatic sample rate adjustment

### Transcription
- Primary: Groq's Whisper model
- Fallback: Local Whisper model
- JSON response format

### Question Processing
- Explicit question detection
- Implicit question inference
- Context-aware processing

### Answer Generation
- Document retrieval prioritization
- Web search integration
- Source attribution

## 🎨 UI Components

- Recording controls
- Status indicators
- Expandable result cards
- Styled sections:
  - Transcription box
  - Question display
  - Answer presentation

## 🔍 Logging

- Detailed error tracking
- Performance monitoring
- File-based logging
- Console output

## ⚠️ Error Handling

- Transcription fallbacks
- File management recovery
- API rate limiting management
- Network error recovery

## 🔒 Security

- Temporary file cleanup
- API key protection
- Safe file operations
- Input validation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - *Initial work*

## 🙏 Acknowledgments

- Groq for AI services
- Streamlit for UI framework
- HuggingFace for models
- OpenAI for Whisper

## 📞 Support

For support, please open an issue in the repository or contact the maintainers.
