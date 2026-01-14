"""
Diagnostic script to understand emotion detection issues
"""
import os
import numpy as np
import librosa
from emotion_detector import EmotionDetector
import soundfile as sf

detector = EmotionDetector()

print("=" * 70)
print("🔍 EMOTION DETECTION DIAGNOSTIC")
print("=" * 70)

# Get the 5 most recent recordings
recorded_dir = "recorded_audio"
wav_files = sorted([f for f in os.listdir(recorded_dir) if f.endswith('.wav')])[-5:]

print(f"\n📊 Analyzing {len(wav_files)} recent recordings:\n")

for wav_file in wav_files:
    audio_path = os.path.join(recorded_dir, wav_file)
    
    # Get audio info
    y, sr = librosa.load(audio_path, sr=16000, duration=3)
    
    # Get detection result
    result = detector.predict(audio_path)
    
    # Calculate audio metrics
    amplitude = np.max(np.abs(y))
    energy = np.mean(y ** 2)
    
    print(f"📁 {wav_file}")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Duration: {len(y) / sr:.2f} seconds")
    print(f"   Amplitude: {amplitude:.4f} (0.0-1.0 scale)")
    print(f"   Energy: {energy:.6f}")
    print(f"   🧠 Detected: {result['emotion'].upper():12} ({result['confidence']*100:5.1f}%)")
    
    # Show confidence distribution
    print(f"   📈 Confidence breakdown:")
    sorted_emotions = sorted(result['all_emotions'].items(), key=lambda x: x[1], reverse=True)
    for emotion, confidence in sorted_emotions[:3]:
        print(f"      {emotion:12}: {confidence*100:5.1f}%")
    print()

print("=" * 70)
print("\n💡 POSSIBLE ISSUES:\n")
print("1️⃣ LOW AMPLITUDE (< 0.1) → Recording too quiet")
print("   → Speak louder or move microphone closer")
print()
print("2️⃣ AMBIGUOUS CONFIDENCE → Model unsure (top 2 emotions similar)")
print("   → Try speaking with more emotion/expression")
print()
print("3️⃣ CONSISTENT WRONG DETECTION → Mismatch between training & real speech")
print("   → Training: Acted RAVDESS speech")
print("   → Real: Your natural speech (different microphone, background noise)")
print()
print("=" * 70)
