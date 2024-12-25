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


## Results:

### Question extraction:

![Screenshot 2024-12-25 220420](https://github.com/user-attachments/assets/31cd8aca-9bce-4874-a3c4-a940bdbdcb1d)

![Screenshot 2024-12-25 220429](https://github.com/user-attachments/assets/166ba7ca-2cd5-483e-a012-14f0dd8f932f)

![Screenshot 2024-12-25 220439](https://github.com/user-attachments/assets/f1d31656-ef06-4dc2-9605-86ce6750b48a)

![Screenshot 2024-12-25 220446](https://github.com/user-attachments/assets/6cecf28e-c49e-45de-8b80-4b9c364e7ede)

![Screenshot 2024-12-25 220452](https://github.com/user-attachments/assets/f4d2f636-a3a5-45d1-b847-de4df098f514)


### Agent Working:

![Screenshot 2024-12-25 221027](https://github.com/user-attachments/assets/de6b9e38-8d67-431d-8052-e9b17caa679d)

![Screenshot 2024-12-25 221038](https://github.com/user-attachments/assets/096e8ca0-037a-4de6-972c-2e8a779c3d04)

![Screenshot 2024-12-25 221055](https://github.com/user-attachments/assets/de54a68b-c8e7-4b3e-a16f-cd2e4e91aa7a)

![Screenshot 2024-12-25 221107](https://github.com/user-attachments/assets/91655733-3976-45c5-9704-73abe85529d1)

![Screenshot 2024-12-25 221117](https://github.com/user-attachments/assets/1f8b5131-e882-4171-afff-fda215e1bdf8)

![Screenshot 2024-12-25 221133](https://github.com/user-attachments/assets/fb3ecc1e-75a6-43af-bd79-c850e18d4cc8)


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
