"""
🟩 PHASE 3 — UNIFIED VOICE ASSISTANT PIPELINE
Wake → Record → Transcribe (All in One)
"""

import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime
import pvporcupine
import struct
import whisper
import soundfile as sf

# Configuration
ACCESS_KEY = "f+4dR9DLMc4/+1zjLleNyCHMg8WdqfFHE9QgRz9LJmN44gPUIbRItA=="
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
DTYPE = np.int16
RECORDING_DURATION = 5  # seconds
RECORDED_AUDIO_DIR = "recorded_audio"
AUDIO_DEVICE = 2  # Built-in Microphone Array (Intel)

def ensure_recorded_audio_dir():
    """Create recorded_audio directory if it doesn't exist."""
    if not os.path.exists(RECORDED_AUDIO_DIR):
        os.makedirs(RECORDED_AUDIO_DIR)

def listen_for_wake_word(porcupine, audio_stream):
    """Listen for wake word and return True when detected."""
    frame_count = 0
    while True:
        try:
            audio_frame, overflowed = audio_stream.read(porcupine.frame_length)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print("🔴 Listening...", end="\r")
            
            if overflowed:
                print("Warning: Audio buffer overflowed")
            
            pcm = struct.unpack_from("h" * porcupine.frame_length, audio_frame.tobytes())
            keyword_index = porcupine.process(pcm)
            
            if keyword_index >= 0:
                print("\n✅ 🎤 WAKE WORD DETECTED! 'hey siri'")
                return True
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False

def record_audio(duration=RECORDING_DURATION):
    """Record audio from the microphone."""
    print(f"🎙️  Recording for {duration} seconds...")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=AUDIO_DEVICE
    )
    sd.wait()
    print("✅ Recording finished!")
    
    max_amplitude = np.max(np.abs(audio_data))
    print(f"   Amplitude: {max_amplitude}")
    
    return audio_data

def save_audio(audio_data, filename=None):
    """Save audio data to a WAV file."""
    ensure_recorded_audio_dir()
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
    
    filepath = os.path.join(RECORDED_AUDIO_DIR, filename)
    
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    
    print(f"📁 Saved: {filepath}")
    return filepath

def transcribe_audio(audio_data, whisper_model):
    """Transcribe audio using Whisper."""
    print("🧠 Transcribing...")
    
    # Flatten to 1D if needed
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    # Convert to float32 and normalize to [-1, 1]
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    print(f"   Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}")
    
    # Transcribe
    result = whisper_model.transcribe(
        audio=audio_data,
        language="en",
        fp16=False
    )
    
    text = result["text"].strip()
    return text

def main():
    """Main pipeline: Wake → Record → Transcribe."""
    print("=" * 60)
    print("🎯 VOICE ASSISTANT - UNIFIED PIPELINE")
    print("=" * 60)
    print("\n📍 Loading models...")
    
    # Initialize Porcupine
    porcupine = pvporcupine.create(
        access_key=ACCESS_KEY,
        keywords=["hey siri"]
    )
    print("✅ Porcupine initialized")
    
    # Load Whisper model
    whisper_model = whisper.load_model("base")
    print("✅ Whisper loaded")
    
    # Start audio stream
    print("\n📍 Starting audio stream...")
    audio_stream = sd.InputStream(
        device=AUDIO_DEVICE,
        channels=CHANNELS,
        samplerate=porcupine.sample_rate,
        dtype=DTYPE,
        blocksize=porcupine.frame_length
    )
    audio_stream.start()
    print("✅ Ready! Say 'hey siri'\n")
    
    try:
        while True:
            # Wait for wake word
            if listen_for_wake_word(porcupine, audio_stream):
                # Record speech
                audio_data = record_audio()
                
                # Save audio file
                save_audio(audio_data)
                
                # Transcribe
                text = transcribe_audio(audio_data, whisper_model)
                
                # Print result
                print("\n" + "=" * 60)
                print("📝 TRANSCRIPTION:")
                print(f"   {text}")
                print("=" * 60 + "\n")
                
                print("🎤 Listening again...")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Exiting. Goodbye!")
    
    finally:
        audio_stream.stop()
        audio_stream.close()
        porcupine.delete()

if __name__ == "__main__":
    main()
