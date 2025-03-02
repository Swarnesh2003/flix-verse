import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load extracted features
df = pd.read_csv("single_audio_features.csv")

# Separate features and labels
X = df.drop(columns=["label"]).values  # Feature values
y = df["label"].values  # Labels

# Apply normalization
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Convert back to DataFrame
columns = df.columns[:-1]  # Feature names (excluding "label")
df_normalized = pd.DataFrame(X_normalized, columns=columns)
df_normalized["label"] = y  # Reattach labels

# Save normalized features
df_normalized.to_csv("normalized_single_audio_features.csv", index=False)

print("Feature normalization complete! Data saved to 'normalized_audio_features.csv'.")
