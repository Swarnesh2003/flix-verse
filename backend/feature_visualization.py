'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analyze_feature_means(csv_file):
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Get feature columns (excluding 'label')
    feature_cols = [col for col in df.columns if col != 'label']
    
    # Calculate means for each feature by category
    means_by_category = df.groupby('label')[feature_cols].mean()
    
    # Create a heatmap visualization
    plt.figure(figsize=(15, 8))
    sns.heatmap(means_by_category, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, cbar_kws={'label': 'Mean Value'})
    plt.title('Mean Feature Values by Category')
    plt.tight_layout()
    plt.show()
    
    # Print detailed means with formatting
    print("\nDetailed Mean Values for Each Feature by Category:")
    print("=" * 100)
    
    # Convert means to a formatted DataFrame with rounded values
    means_formatted = means_by_category.round(4)
    print(means_formatted)
    
    # Calculate and print feature ranges to show the scale of differences
    print("\nFeature Value Ranges:")
    print("=" * 100)
    for feature in feature_cols:
        min_val = df[feature].min()
        max_val = df[feature].max()
        range_val = max_val - min_val
        print(f"{feature:15} Min: {min_val:8.4f} | Max: {max_val:8.4f} | Range: {range_val:8.4f}")
    
    # Create bar plots for each feature
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, feature in enumerate(feature_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        means_by_category[feature].plot(kind='bar')
        plt.title(f'Mean {feature} by Category')
        plt.xticks(rotation=45)
        plt.ylabel('Mean Value')
    
    plt.tight_layout()
    plt.show()

# Example usage
analyze_feature_means('audio_features.csv')


'''
'''
import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# Define dataset paths
DATASET_PATH = "C:/Users/Swarneshwar S/Desktop/FILES/PROJECTS/DISH Hackathon/dataset/converted"  # Root folder containing 'songs/', 'fights/', 'dialogues/'
CATEGORIES = ["songs", "fights", "dialogues"]

# Define the target output CSV file
OUTPUT_CSV = "audio_features.csv"

# Function to extract audio features
def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rmse = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Return all features as a flat array
        return np.hstack([mfccs, zcr, rmse, spectral_centroid])
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare feature extraction
data = []
columns = [f"mfcc_{i}" for i in range(13)] + ["zcr", "rmse", "spectral_centroid", "label"]

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
'''

'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio_features(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Set up the plotting style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Box plots for MFCC features
    plt.subplot(2, 2, 1)
    mfcc_cols = [col for col in df.columns if 'mfcc' in col]
    mfcc_data = pd.melt(df[mfcc_cols + ['label']], id_vars=['label'],
                        var_name='MFCC', value_name='Value')
    sns.boxplot(x='MFCC', y='Value', hue='label', data=mfcc_data)
    plt.xticks(rotation=45)
    plt.title('MFCC Features Distribution by Category')
    
    # 2. Violin plots for ZCR, RMSE, and Spectral Centroid
    plt.subplot(2, 2, 2)
    features = ['zcr', 'rmse', 'spectral_centroid']
    feature_data = pd.melt(df[features + ['label']], id_vars=['label'],
                          var_name='Feature', value_name='Value')
    sns.violinplot(x='Feature', y='Value', hue='label', data=feature_data)
    plt.title('Audio Features Distribution by Category')
    
    # 3. 2D scatter plot using first two MFCCs
    plt.subplot(2, 2, 3)
    for label in df['label'].unique():
        mask = df['label'] == label
        plt.scatter(df.loc[mask, 'mfcc_0'], df.loc[mask, 'mfcc_1'], 
                   label=label, alpha=0.6)
    plt.xlabel('MFCC_0')
    plt.ylabel('MFCC_1')
    plt.title('2D Scatter Plot: MFCC_0 vs MFCC_1')
    plt.legend()
    
    # 4. Feature correlation heatmap
    plt.subplot(2, 2, 4)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, 
                fmt='.2f', square=True)
    plt.title('Feature Correlation Heatmap')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Calculate and print summary statistics
    print("\nSummary Statistics by Category:")
    for feature in ['zcr', 'rmse', 'spectral_centroid']:
        print(f"\n{feature.upper()} Statistics:")
        print(df.groupby('label')[feature].describe())

# Example usage
visualize_audio_features('audio_features.csv')
'''
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_audio_features_two_categories(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter for only fight and songs categories
    df = df[df['label'].isin(['fights', 'songs'])]
    
    # Set up the plotting style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Box plots for MFCC features
    plt.subplot(2, 2, 1)
    mfcc_cols = [col for col in df.columns if 'mfcc' in col]
    mfcc_data = pd.melt(df[mfcc_cols + ['label']], id_vars=['label'],
                        var_name='MFCC', value_name='Value')
    sns.boxplot(x='MFCC', y='Value', hue='label', data=mfcc_data)
    plt.xticks(rotation=45)
    plt.title('MFCC Features Distribution: Fights vs Songs')
    
    # 2. Violin plots for ZCR, RMSE, and Spectral Centroid
    plt.subplot(2, 2, 2)
    features = ['zcr', 'rmse', 'spectral_centroid']
    feature_data = pd.melt(df[features + ['label']], id_vars=['label'],
                          var_name='Feature', value_name='Value')
    sns.violinplot(x='Feature', y='Value', hue='label', data=feature_data)
    plt.title('Audio Features Distribution: Fights vs Songs')
    
    # 3. 2D scatter plot using first two MFCCs
    plt.subplot(2, 2, 3)
    for label in ['fights', 'songs']:
        mask = df['label'] == label
        plt.scatter(df.loc[mask, 'mfcc_0'], df.loc[mask, 'mfcc_1'], 
                   label=label, alpha=0.6)
    plt.xlabel('MFCC_0')
    plt.ylabel('MFCC_1')
    plt.title('2D Scatter Plot: MFCC_0 vs MFCC_1 (Fights vs Songs)')
    plt.legend()
    
    # 4. Feature correlation heatmap
    plt.subplot(2, 2, 4)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, 
                fmt='.2f', square=True)
    plt.title('Feature Correlation Heatmap')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Calculate and print summary statistics
    print("\nSummary Statistics - Fights vs Songs:")
    print("=" * 50)
    
    # Print statistics for each feature
    for feature in ['zcr', 'rmse', 'spectral_centroid'] + mfcc_cols:
        print(f"\n{feature.upper()} Statistics:")
        stats = df.groupby('label')[feature].describe()
        print(stats)
        
        # Calculate and print the difference between categories
        fight_mean = stats.loc['fights', 'mean']
        songs_mean = stats.loc['songs', 'mean']
        difference = abs(fight_mean - songs_mean)
        print(f"\nDifference between fight and songs: {difference:.4f}")
        print("-" * 50)
    
    # Additional comparison visualization
    plt.figure(figsize=(15, 6))
    feature_means = df.groupby('label')[features].mean()
    feature_means.plot(kind='bar')
    plt.title('Mean Feature Values: Fight vs Songs')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)
    plt.legend(title='Features')
    plt.tight_layout()
    plt.show()

# Example usage
visualize_audio_features_two_categories('audio_features.csv')'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset
df = pd.read_csv("audio_features.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 1. Plot histograms for some key features
plt.figure(figsize=(15, 6))
for i, feature in enumerate(["mfcc_0", "mfcc_1", "zcr", "rmse", "spectral_centroid", "tempo"]):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data=df, x=feature, hue=y, kde=True, bins=30)
    plt.title(f"Distribution of {feature}")

plt.tight_layout()
plt.show()

# 🔹 2. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pd.DataFrame(X_scaled, columns=X.columns).corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 🔹 3. PCA Visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette="deep", alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA - 2D Projection")
plt.show()

# 🔹 4. t-SNE Visualization (2D)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="deep", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE - 2D Projection")
plt.show()
