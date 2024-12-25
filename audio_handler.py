import os
import io
import tempfile
import uuid
from groq import Groq
import librosa
import time
import numpy as np

def get_unique_temp_path():
    """Generate a unique temporary file path."""
    temp_dir = tempfile.gettempdir()
    unique_filename = f"audio_{uuid.uuid4().hex}.wav"
    return os.path.join(temp_dir, unique_filename)

def convert_bytes_to_array(audio_bytes):
    """Convert audio bytes to array for visualization or additional processing if needed."""
    audio_bytes = io.BytesIO(audio_bytes)
    audio, sample_rate = librosa.load(audio_bytes)
    return audio, sample_rate

def transcribe_audio(audio_input):
    """
    Transcribe audio using Groq's Whisper model with improved file handling.
    
    Args:
        audio_input: Audio data (dict or bytes)
        
    Returns:
        str: Transcribed text
    """
    temp_filename = None
    try:
        # Extract audio bytes from dictionary if needed
        audio_bytes = audio_input['bytes'] if isinstance(audio_input, dict) else audio_input
        
        # Initialize Groq client with API key
        client = Groq(api_key="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")
        
        # Create and write to temporary file
        temp_filename = get_unique_temp_path()
        with open(temp_filename, 'wb') as temp_file:
            temp_file.write(audio_bytes)
        
        time.sleep(0.1)  # Ensure file is written
        
        # Transcribe using Groq
        try:
            with open(temp_filename, 'rb') as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(temp_filename), audio_file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
                
                if hasattr(transcription, 'text'):
                    return transcription.text.strip()
                else:
                    return process_local_transcription(audio_bytes)
                    
        except Exception as e:
            print(f"Groq transcription failed: {e}")
            return process_local_transcription(audio_bytes)
                
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return process_local_transcription(audio_bytes)
        
    finally:
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.close(os.open(temp_filename, os.O_RDONLY))
                os.unlink(temp_filename)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")

def process_local_transcription(audio_bytes):
    """
    Process audio locally using Whisper when Groq fails.
    
    Args:
        audio_bytes: Binary audio data
        
    Returns:
        str: Transcribed text
    """
    try:
        print("Attempting local transcription...")
        audio_array, _ = convert_bytes_to_array(audio_bytes)
        
        import torch
        from transformers import pipeline
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=device,
        )
        
        result = pipe(audio_array, batch_size=1)["text"]
        return result.strip()
        
    except Exception as e:
        print(f"Local transcription error: {str(e)}")
        return "Error: Could not process audio using either Groq or local processing"

def safe_delete_file(filepath):
    """
    Safely delete a file with retry logic.
    
    Args:
        filepath: Path to the file to delete
    """
    max_attempts = 5
    delay = 0.1
    
    for attempt in range(max_attempts):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
            return True
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Could not delete file after {max_attempts} attempts: {str(e)}")
                return False
            time.sleep(delay)
            delay *= 2  # Exponential backoff
