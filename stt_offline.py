import whisper
import soundfile as sf
import numpy as np
import os

# Load Whisper Model (Offline)
model_size = "base"   # good balance (≈70–80% accuracy)

print("📍 Loading Whisper model...")
model = whisper.load_model(model_size)
print("✅ Model loaded")

# Select Audio File
AUDIO_FOLDER = "recorded_audio"

files = os.listdir(AUDIO_FOLDER)
files = [f for f in files if f.endswith(".wav")]

if not files:
    raise FileNotFoundError("No audio files found in recorded_audio/")

latest_audio = sorted(files)[-1]
audio_path = os.path.join(AUDIO_FOLDER, latest_audio)

print(f"\n📍 Transcribing: {audio_path}")

# Load audio using soundfile instead of ffmpeg
audio_data, sr = sf.read(audio_path)

# Convert to float32 (Whisper requirement)
audio_data = audio_data.astype(np.float32)

# Resample to 16kHz if needed
if sr != 16000:
    import librosa
    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

# Transcribe Audio (CORE STEP)
result = model.transcribe(
    audio=audio_data,
    language="en",
    fp16=False  # CPU compatibility
)

# Print Transcribed Text (NO CLEANING)
print("\n--- TRANSCRIPTION ---")
print(result["text"])
print("---------------------\n")
