import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# Define dataset paths
DATASET_PATH = "C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH Hackathon/dataset/converted_new"  # Root folder containing 'songs/', 'fights/', 'dialogues/'
CATEGORIES = ["songs", "fights", "dialogues"]

# Define the target output CSV file
OUTPUT_CSV = "new_audio_features.csv"

# Function to extract advanced audio features
def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=13), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

        # Extract Tempo (BPM)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

        # Combine all features into a single array
        features = np.hstack([
            mfccs, zcr, rmse, spectral_centroid, spectral_bandwidth, spectral_rolloff, 
            spectral_contrast, chroma_stft, mel_spectrogram, tonnetz, tempo
        ])
        
        return features
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare feature extraction
data = []

# Define column names for the CSV file
columns = (
    [f"mfcc_{i}" for i in range(13)] +
    ["zcr", "rmse", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff"] +
    [f"spectral_contrast_{i}" for i in range(7)] +
    [f"chroma_{i}" for i in range(12)] +
    [f"mel_{i}" for i in range(13)] +
    [f"tonnetz_{i}" for i in range(6)] +
    ["tempo", "label"]
)

# Process each category
for category in CATEGORIES:
    category_path = os.path.join(DATASET_PATH, category)
    label = category  # Use folder name as label

    print(f"Processing category: {category}")

    for filename in tqdm(os.listdir(category_path)):
        file_path = os.path.join(category_path, filename)
        
        # Ensure it's an audio file
        if not filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue

        # Extract features
        features = extract_features(file_path)
        if features is not None:
            data.append(np.append(features, label))

# Save extracted features to CSV
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Feature extraction complete! Data saved to {OUTPUT_CSV}")
