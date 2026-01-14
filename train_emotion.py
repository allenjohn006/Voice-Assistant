"""
🟩 PHASE 4 — EMOTION DETECTION (CLASSICAL ML)
Train Random Forest on RAVDESS dataset using MFCC features
"""

import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import urllib.request
import zipfile

# Configuration
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DATASET_DIR = "ravdess_dataset"
MODEL_PATH = "emotion_model.joblib"

# Emotion mapping (RAVDESS format)
# Format: Actor_Modality-Vocal channel-Emotion-Intensity-Statement-Repetition-Actor.mp3
# Emotions: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgusted, 08=surprised
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgusted',
    '08': 'surprised'
}

def extract_mfcc(audio_path, n_mfcc=20):
    """Extract MFCC features from audio file."""
    try:
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
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def load_ravdess_data(data_dir):
    """Load RAVDESS data and extract features."""
    X = []
    y = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found at {data_dir}")
        print("📍 Using sample data instead...")
        return None, None
    
    actor_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"📍 Found {len(actor_dirs)} actor directories")
    
    for actor_dir in actor_dirs:
        actor_path = os.path.join(data_dir, actor_dir)
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for audio_file in audio_files:
            try:
                # Extract emotion from filename
                parts = audio_file.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in EMOTION_MAP:
                        emotion = EMOTION_MAP[emotion_code]
                        audio_path = os.path.join(actor_path, audio_file)
                        
                        # Extract MFCC features
                        features = extract_mfcc(audio_path)
                        if features is not None:
                            X.append(features)
                            y.append(emotion)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
    
    if len(X) == 0:
        print("❌ No audio files loaded")
        return None, None
    
    return np.array(X), np.array(y)

def train_emotion_model():
    """Train Random Forest emotion classifier."""
    print("=" * 60)
    print("🧠 EMOTION DETECTION - TRAINING")
    print("=" * 60)
    
    print("\n📍 Loading RAVDESS dataset...")
    print("   Note: Download from https://zenodo.org/record/1188976")
    print("   Extract to 'ravdess_dataset' folder\n")
    
    X, y = load_ravdess_data(DATASET_DIR)
    
    if X is None:
        print("❌ Cannot train without RAVDESS dataset")
        print("   Please download and extract the dataset first")
        return False
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"   Classes: {np.unique(y)}")
    print(f"   Feature shape: {X.shape}")
    
    # Split data
    print("\n📍 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Standardize features
    print("\n📍 Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try both Random Forest and SVM
    print("\n📍 Training classifiers...")
    
    # Random Forest
    print("   🌲 Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"      Accuracy: {rf_accuracy * 100:.2f}%")
    
    # SVM
    print("   🎯 SVM...")
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print(f"      Accuracy: {svm_accuracy * 100:.2f}%")
    
    # Choose best model
    if svm_accuracy > rf_accuracy:
        model = svm_model
        y_pred = svm_pred
        print(f"\n✅ Selected SVM (best accuracy)")
    else:
        model = rf_model
        y_pred = rf_pred
        print(f"\n✅ Selected Random Forest (best accuracy)")
    
    # Save both model and scaler
    final_model = {'classifier': model, 'scaler': scaler}
    print("✅ Training complete")
    
    # Evaluate
    print("\n📍 Evaluating model...")
    accuracy = max(rf_accuracy, svm_accuracy)
    print(f"✅ Accuracy: {accuracy:.2%}")
    
    if accuracy < 0.70:
        print("⚠️  Warning: Accuracy < 70%")
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\n📍 Saving model...")
    joblib.dump(final_model, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")
    
    return True

if __name__ == "__main__":
    train_emotion_model()
