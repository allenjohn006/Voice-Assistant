import sounddevice as sd
import numpy as np
import wave
import os
from datetime import datetime
import pvporcupine
import struct

# Configuration
ACCESS_KEY = "f+4dR9DLMc4/+1zjLleNyCHMg8WdqfFHE9QgRz9LJmN44gPUIbRItA=="
SAMPLE_RATE = 16000  # Hz
CHANNELS = 1
DTYPE = np.int16
RECORDING_DURATION = 5  # seconds
RECORDED_AUDIO_DIR = "recorded_audio"
AUDIO_DEVICE = 2  # Built-in Microphone Array (Intel) - MME (most compatible)

def ensure_recorded_audio_dir():
    """Create recorded_audio directory if it doesn't exist."""
    if not os.path.exists(RECORDED_AUDIO_DIR):
        os.makedirs(RECORDED_AUDIO_DIR)
        print(f"Created directory: {RECORDED_AUDIO_DIR}")

def record_audio(duration=RECORDING_DURATION):
    """Record audio from the microphone."""
    print(f"Recording for {duration} seconds...")
    print(f"Using device: {AUDIO_DEVICE}")
    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=AUDIO_DEVICE  # Add device parameter
    )
    sd.wait()  # Wait until recording is finished
    print("Recording finished!")
    
    # Check if audio was captured
    max_amplitude = np.max(np.abs(audio_data))
    print(f"Max amplitude: {max_amplitude}")
    if max_amplitude < 100:
        print("⚠️  WARNING: Audio is very quiet or silent! Check microphone.")
    
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
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    
    print(f"Audio saved to: {filepath}")
    return filepath

def list_audio_devices():
    """List all available audio input devices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())

def listen_for_wake_word():
    """Listen for the wake word using Porcupine."""
    porcupine = None
    audio_stream = None
    
    try:
        # Initialize Porcupine with default wake word
        print("\n📍 Initializing Porcupine...")
        porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keywords=["hey siri"]
        )
        print("✅ Porcupine initialized successfully")
        
        print(f"\n🎤 Listening for wake word 'hey siri'... (Press Ctrl+C to stop)")
        print(f"   Sample rate: {porcupine.sample_rate} Hz")
        print(f"   Frame length: {porcupine.frame_length} samples")
        print(f"   Audio device: {AUDIO_DEVICE}")
        
        # Open audio stream with specific device
        print("\n📍 Starting audio stream...")
        try:
            audio_stream = sd.InputStream(
                device=AUDIO_DEVICE,
                channels=CHANNELS,
                samplerate=porcupine.sample_rate,
                dtype=DTYPE,
                blocksize=porcupine.frame_length
            )
            audio_stream.start()
            print("✅ Audio stream started. Listening now... Say 'hey siri'\n")
        except Exception as stream_error:
            print(f"\n❌ Failed to start audio stream: {stream_error}")
            print(f"   Tried to use device {AUDIO_DEVICE}")
            print("   Try changing AUDIO_DEVICE to a different input device number")
            raise
        
        frame_count = 0
        while True:
            try:
                # Read audio frame
                audio_frame, overflowed = audio_stream.read(porcupine.frame_length)
                frame_count += 1
                
                # Show listening indicator every 50 frames (~1.6 seconds)
                if frame_count % 50 == 0:
                    print("🔴 Listening...", end="\r")
                
                if overflowed:
                    print("Warning: Audio buffer overflowed")
                
                # Convert to int16 array
                pcm = struct.unpack_from("h" * porcupine.frame_length, audio_frame.tobytes())
                
                # Check for wake word
                keyword_index = porcupine.process(pcm)
                
                if keyword_index >= 0:
                    print("\n✅ 🎤 WAKE WORD DETECTED! 'hey siri'")
                    return True
            
            except Exception as inner_error:
                print(f"\n❌ Error reading audio frame: {inner_error}")
                raise
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped listening for wake word.")
        return False
    
    except Exception as e:
        print(f"\n❌ Error during wake word detection: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
        
        if porcupine is not None:
            porcupine.delete()

def main():
    """Main function to run the voice assistant."""
    print("=" * 50)
    print("Voice Assistant - Phase 1: Wake & Listen")
    print("=" * 50)
    
    # List available audio devices
    list_audio_devices()
    
    print("\nStarting wake word detection...")
    print("Say 'hey siri' to activate recording, or press Ctrl+C to quit.")
    
    try:
        while True:
            # Listen for wake word
            result = listen_for_wake_word()
            
            if result is False:
                # User pressed Ctrl+C or error occurred
                break
            
            if result is True:
                # Record audio after wake word detected
                audio_data = record_audio()
                
                # Save audio
                save_audio(audio_data)
                
                print("\n✅ Recording saved! Listening for wake word again...")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Exiting voice assistant. Goodbye!")


if __name__ == "__main__":
    main()
