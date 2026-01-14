"""
Emotion Detector - Inference Module
Loads trained model and predicts emotion from audio
"""

import numpy as np
import librosa
import joblib
import os

MODEL_PATH = "emotion_model.joblib"

class EmotionDetector:
    def __init__(self, model_path=MODEL_PATH):
        """Load trained emotion model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            self.model = model_data['classifier']
            self.scaler = model_data['scaler']
        else:
            self.model = model_data
            self.scaler = None
        print(f"✅ Emotion model loaded")
    
    def extract_mfcc(self, audio_path, n_mfcc=20):
        """Extract MFCC features from audio."""
        y, sr = librosa.load(audio_path, sr=16000, duration=3)
        # MFCC features (mean and std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        # Combine features
        features = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid, spectral_rolloff, zero_crossing_rate]])
        return features
    
    def predict(self, audio_path):
        """Predict emotion from audio file."""
        features = self.extract_mfcc(audio_path)
        features = features.reshape(1, -1)
        
        # Apply scaling if available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Get prediction and confidence
        emotion = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        return {
            "emotion": emotion,
            "confidence": float(confidence),
            "all_emotions": {
                self.model.classes_[i]: float(probabilities[i])
                for i in range(len(self.model.classes_))
            }
        }

def main():
    """Test emotion detection."""
    print("=" * 60)
    print("🧠 EMOTION DETECTOR - TEST")
    print("=" * 60)
    
    try:
        detector = EmotionDetector()
        
        # Find a test audio file
        test_audio_dir = "recorded_audio"
        if os.path.exists(test_audio_dir):
            wav_files = [f for f in os.listdir(test_audio_dir) if f.endswith('.wav')]
            
            if wav_files:
                test_file = os.path.join(test_audio_dir, wav_files[-1])
                print(f"\n📍 Testing on: {test_file}")
                
                result = detector.predict(test_file)
                
                print("\n" + "=" * 60)
                print(f"📊 EMOTION: {result['emotion'].upper()}")
                print(f"   Confidence: {result['confidence']:.2%}")
                print("\n   All emotions:")
                for emotion, prob in sorted(result['all_emotions'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    print(f"      {emotion:12} {prob:.2%}")
                print("=" * 60)
            else:
                print("❌ No audio files in recorded_audio/")
                print("   Record using: python pipeline.py")
        else:
            print("❌ recorded_audio directory not found")
    
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Train first: python train_emotion.py")

if __name__ == "__main__":
    main()
