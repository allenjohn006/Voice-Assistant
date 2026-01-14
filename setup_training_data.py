"""
Fine-tune emotion model on your own recordings
Collects real training data from your voice and microphone
"""
import os
import shutil
from pathlib import Path

print("=" * 70)
print("🎤 EMOTION MODEL FINE-TUNING SETUP")
print("=" * 70)

# Create directories for user recordings
base_dir = "my_training_data"
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

print(f"\n📁 Creating training data directories...\n")

for emotion in emotions:
    emotion_dir = os.path.join(base_dir, emotion)
    os.makedirs(emotion_dir, exist_ok=True)
    print(f"   ✅ {emotion_dir}")

print("\n" + "=" * 70)
print("📝 INSTRUCTIONS:")
print("=" * 70)
print("""
1. The pipeline will record 5-second clips when you say "hey siri"
2. After each recording, manually move it to the correct emotion folder:
   
   Windows Explorer:
   📂 C:\\Users\\allen\\Downloads\\voice-assistant\\voice_assistant_phase1\\
      └─ 📂 recorded_audio
         └─ recording_YYYYMMDD_HHMMSS.wav
      
      Copy/move to:
      📂 my_training_data
         ├─ 📂 angry
         ├─ 📂 calm
         ├─ 📂 happy
         ├─ 📂 sad
         ├─ 📂 neutral
         ├─ 📂 fearful
         ├─ 📂 disgusted
         └─ 📂 surprised

3. Collect AT LEAST 5-10 samples per emotion (80+ total recordings)

4. Once collected, run:
   python finetune_emotion.py

This will retrain the model on YOUR specific voice/microphone!
""")
print("=" * 70)
