"""
Fine-tune the emotion model on your own voice/microphone recordings
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

EMOTION_DIR = "my_training_data"
RAVDESS_DIR = "ravdess_dataset"
MODEL_PATH = "emotion_model_finetuned.joblib"

def extract_mfcc(audio_path, n_mfcc=20):
    """Extract MFCC features from audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        features = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid, spectral_rolloff, zero_crossing_rate]])
        return features
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def load_user_data(data_dir):
    """Load user's own recorded emotions."""
    X = []
    y = []
    
    if not os.path.exists(data_dir):
        print(f"❌ User training data not found at {data_dir}")
        return None, None
    
    emotion_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"\n📍 Loading user training data...")
    print(f"   Found {len(emotion_dirs)} emotion folders")
    
    for emotion in emotion_dirs:
        emotion_path = os.path.join(data_dir, emotion)
        audio_files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        
        print(f"   {emotion:12}: {len(audio_files):2} files", end="")
        
        for audio_file in audio_files:
            audio_path = os.path.join(emotion_path, audio_file)
            features = extract_mfcc(audio_path)
            if features is not None:
                X.append(features)
                y.append(emotion)
        
        print(" ✅")
    
    return np.array(X) if X else None, y

def load_ravdess_data(data_dir):
    """Load RAVDESS data (for comparison/background training)."""
    X = []
    y = []
    
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised'
    }
    
    if not os.path.exists(data_dir):
        return None, None
    
    actor_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for actor_dir in actor_dirs:
        actor_path = os.path.join(data_dir, actor_dir)
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for audio_file in audio_files:
            try:
                parts = audio_file.split('-')
                if len(parts) >= 3 and parts[2] in emotion_map:
                    emotion = emotion_map[parts[2]]
                    audio_path = os.path.join(actor_path, audio_file)
                    features = extract_mfcc(audio_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
            except:
                pass
    
    return np.array(X) if X else None, y

def main():
    print("=" * 70)
    print("🧠 EMOTION MODEL - FINE-TUNING")
    print("=" * 70)
    
    # Load user data
    X_user, y_user = load_user_data(EMOTION_DIR)
    
    if X_user is None or len(X_user) < 10:
        print("\n⚠️  Not enough user data collected yet!")
        print(f"   Found: {len(X_user) if X_user is not None else 0} samples")
        print(f"   Need: At least 10 samples (5-10 per emotion)\n")
        print("📍 Steps:")
        print("   1. Run: python setup_training_data.py")
        print("   2. Collect recordings using: python pipeline.py")
        print("   3. Move them to my_training_data/[emotion]/ folders")
        print("   4. Run this script again")
        return False
    
    print(f"✅ Loaded {len(X_user)} user samples")
    
    # Load RAVDESS as base
    X_ravdess, y_ravdess = load_ravdess_data(RAVDESS_DIR)
    print(f"✅ Loaded {len(X_ravdess) if X_ravdess is not None else 0} RAVDESS samples")
    
    # Combine datasets (70% RAVDESS, 30% user for fine-tuning effect)
    if X_ravdess is not None and len(X_ravdess) > 0:
        # Weight user data more heavily (appears 2x)
        X = np.vstack([X_ravdess, X_user, X_user])
        y = y_ravdess + y_user + y_user
        print(f"✅ Combined: {len(X)} total samples")
    else:
        X = X_user
        y = y_user
        print(f"⚠️  Using only user data: {len(X)} samples")
    
    # Convert to numpy array if needed
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    print(f"\n📍 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Standardize
    print(f"\n📍 Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print(f"\n📍 Training classifiers...")
    
    print("   🌲 Random Forest...", end="", flush=True)
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_split=3,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test_scaled))
    print(f" {rf_accuracy*100:.1f}%")
    
    print("   🎯 SVM...", end="", flush=True)
    svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test_scaled))
    print(f" {svm_accuracy*100:.1f}%")
    
    # Select best
    if svm_accuracy > rf_accuracy:
        best_model = svm_model
        best_accuracy = svm_accuracy
        best_name = "SVM"
    else:
        best_model = rf_model
        best_accuracy = rf_accuracy
        best_name = "Random Forest"
    
    print(f"\n✅ Selected {best_name} (best accuracy: {best_accuracy*100:.1f}%)")
    
    # Save
    print(f"\n📍 Saving fine-tuned model...")
    final_model = {'classifier': best_model, 'scaler': scaler}
    joblib.dump(final_model, MODEL_PATH)
    print(f"✅ Saved to {MODEL_PATH}")
    
    # Report
    print(f"\n📊 Classification Report:")
    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    # Update pipeline
    print("\n" + "=" * 70)
    print("⚠️  NEXT STEP:")
    print("=" * 70)
    print("Update pipeline.py to use the fine-tuned model:")
    print("   Change: emotion_model.joblib")
    print("   To:     emotion_model_finetuned.joblib")
    print("\nOr manually replace the model file:")
    print("   copy emotion_model_finetuned.joblib emotion_model.joblib")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    main()
