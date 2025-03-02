import pandas as pd

# Load dataset
df = pd.read_csv("audio_features.csv")

# Group by category and compute mean for each feature
mean_features = df.groupby("label").mean()

# Display the mean values
print(mean_features)

# Save to CSV if needed
mean_features.to_csv("feature_means_per_category.csv")

print("Mean feature values saved to feature_means_per_category.csv")
