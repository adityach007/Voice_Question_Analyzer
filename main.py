import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import io
from rag_agent import process_audio_input
import time
import threading

# Set page configuration - MUST BE FIRST
st.set_page_config(
    page_title="Voice Question Analyzer",
    page_icon="🎤",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 12px;
        border: none;
        transition-duration: 0.4s;
    }
    .stButton > button:hover { background-color: #45a049; }
    .recording { color: red; font-weight: bold; }
    
    /* New styles for enhanced UI */
    .result-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 5px solid #4CAF50;
    }
    .transcription-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 10px 0;
    }
    .question-box {
        background-color: #e7f3fe;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1976d2;
    }
    .answer-box {
        background-color: #f0f7f0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #43a047;
    }
    .section-header {
        color: #2c3e50;
        font-size: 1.2em;
        font-weight: bold;
        margin: 15px 0 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = False
        self.frames = []
        
    def callback(self, indata, frames, time, status):
        if self.recording:
            self.frames.append(indata.copy())
    
    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.callback,
            dtype=np.int16
        )
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        if self.frames:
            return np.concatenate(self.frames, axis=0)
        return None

def convert_to_wav(audio_data, sample_rate=16000):
    """Convert numpy array to WAV format"""
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    return byte_io.getvalue()

def main():
    st.title("🎤 Voice Question Analyzer")
    st.markdown("### Speak naturally and I'll identify and answer questions in your speech")

    # Initialize session state
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'results' not in st.session_state:
        st.session_state.results = []

    col1, col2 = st.columns([1, 2])

    with col1:
        if not st.session_state.recording:
            if st.button("🎙️ Start Recording"):
                st.session_state.recording = True
                st.session_state.recorder.start_recording()
                st.rerun()
        else:
            if st.button("⏹️ Stop Recording"):
                st.session_state.recording = False
                audio_data = st.session_state.recorder.stop_recording()
                
                if audio_data is not None:
                    status = st.empty()
                    status.markdown("🔄 Processing your speech...")
                    
                    # Convert and process audio
                    wav_data = convert_to_wav(audio_data)
                    result = process_audio_input(wav_data)
                    
                    if result:
                        st.session_state.results.append(result)
                    
                    status.markdown("✅ Processing complete!")
                st.rerun()
        
        # Show recording status
        if st.session_state.recording:
            st.markdown("""
                <div class="recording">
                    ⚪ Recording in progress...
                </div>
            """, unsafe_allow_html=True)

    with col2:
        # Display results
        for idx, result in enumerate(st.session_state.results):
            with st.expander(f"Recording {idx + 1}", expanded=True):
                # Transcription
                st.markdown('<div class="section-header">📝 Transcription</div>', 
                          unsafe_allow_html=True)
                st.markdown(f'<div class="transcription-box">{result["transcription"]}</div>', 
                          unsafe_allow_html=True)
                
                # Questions
                if result['questions']:
                    st.markdown('<div class="section-header">❓ Questions Detected</div>', 
                              unsafe_allow_html=True)
                    for i, question in enumerate(result['questions'], 1):
                        st.markdown(f'<div class="question-box">Q{i}: {question}</div>', 
                                  unsafe_allow_html=True)
                
                # Answers
                if result['answers']:
                    st.markdown('<div class="section-header">💡 Answers</div>', 
                              unsafe_allow_html=True)
                    # Split answers into individual QA pairs
                    answer_pairs = result['answers'].split('\n\n')
                    for answer in answer_pairs:
                        # Split each pair into question and answer
                        parts = answer.split('\nAnswer: ')
                        if len(parts) == 2:
                            question, answer_text = parts
                            st.markdown(f'<div class="answer-box">'
                                      f'<strong>{question}</strong><br><br>'
                                      f'{answer_text}</div>', 
                                      unsafe_allow_html=True)

    # Clear button
    if st.button("Clear All Results"):
        st.session_state.results = []
        st.rerun()

if __name__ == "__main__":
    main()
