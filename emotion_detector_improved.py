"""
Improved Emotion Detector with Confidence Filtering
- Only makes predictions when confident
- Suggests re-speaking if ambiguous
"""
import numpy as np
import librosa
import joblib
import os

MODEL_PATH = "emotion_model.joblib"

class ImprovedEmotionDetector:
    def __init__(self, model_path=MODEL_PATH, confidence_threshold=0.60):
        """Load trained emotion model with confidence threshold."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict):
            self.model = model_data['classifier']
            self.scaler = model_data['scaler']
        else:
            self.model = model_data
            self.scaler = None
        
        self.confidence_threshold = confidence_threshold
        print(f"✅ Emotion model loaded (confidence threshold: {confidence_threshold*100:.0f}%)")
    
    def extract_mfcc(self, audio_path, n_mfcc=20):
        """Extract MFCC features from audio."""
        y, sr = librosa.load(audio_path, sr=16000, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        features = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid, spectral_rolloff, zero_crossing_rate]])
        return features
    
    def predict(self, audio_path):
        """Predict emotion from audio file with confidence check."""
        features = self.extract_mfcc(audio_path)
        features = features.reshape(1, -1)
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        emotion = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        # Check if confident enough
        is_confident = confidence >= self.confidence_threshold
        
        # Find top 3 emotions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_emotions = [(self.model.classes_[i], probabilities[i]) for i in top_indices]
        
        return {
            "emotion": emotion,
            "confidence": float(confidence),
            "is_confident": is_confident,
            "confidence_status": "✅ HIGH" if is_confident else "⚠️ LOW",
            "top_3": [(e, float(c*100)) for e, c in top_emotions],
            "all_emotions": {
                self.model.classes_[i]: float(probabilities[i])
                for i in range(len(self.model.classes_))
            }
        }

# Test
if __name__ == "__main__":
    detector = ImprovedEmotionDetector(confidence_threshold=0.65)
    
    test_file = "recorded_audio/recording_20260114_224158.wav"
    if os.path.exists(test_file):
        result = detector.predict(test_file)
        
        print("\n" + "=" * 60)
        print(f"📊 EMOTION: {result['emotion'].upper()}")
        print(f"   Confidence: {result['confidence']*100:.1f}%")
        print(f"   Status: {result['confidence_status']}")
        print(f"\n   Top 3 predictions:")
        for emotion, conf in result['top_3']:
            print(f"      {emotion:12} {conf:5.1f}%")
        
        if not result['is_confident']:
            print(f"\n   💡 TIP: Model unsure. Try speaking with more expression!")
        
        print("=" * 60)
