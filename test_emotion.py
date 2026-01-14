"""Test emotion detector on RAVDESS samples"""
from emotion_detector import EmotionDetector

detector = EmotionDetector()

# Test on known emotions
test_files = [
    ('ravdess_dataset/Actor_01/03-01-05-01-01-01-01.wav', 'angry'),      # emotion code 05
    ('ravdess_dataset/Actor_01/03-01-03-01-01-01-01.wav', 'happy'),      # emotion code 03
    ('ravdess_dataset/Actor_01/03-01-04-01-01-01-01.wav', 'sad'),        # emotion code 04
    ('ravdess_dataset/Actor_01/03-01-02-01-01-01-01.wav', 'calm'),       # emotion code 02
    ('ravdess_dataset/Actor_01/03-01-01-01-01-01-01.wav', 'neutral'),    # emotion code 01
]

print("=" * 70)
print("🧠 EMOTION DETECTOR - RAVDESS TEST")
print("=" * 70)

for audio_file, expected_emotion in test_files:
    try:
        result = detector.predict(audio_file)
        match = "✅" if result['emotion'].lower() == expected_emotion else "❌"
        print(f"\n{match} Expected: {expected_emotion:10} | Predicted: {result['emotion']:10} | Confidence: {result['confidence']*100:5.1f}%")
    except Exception as e:
        print(f"❌ Error processing {audio_file}: {e}")

print("\n" + "=" * 70)
