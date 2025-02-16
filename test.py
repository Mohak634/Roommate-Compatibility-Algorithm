import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import os


# Enable GPU usage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is enabled")
else:
    print("No GPU found, using CPU")

# Load the encoded dataset
df = pd.read_csv("encoded.csv")
question_mapping = pd.read_csv("question_mapping.csv")

# Extract valid feature columns from question_mapping
valid_features = question_mapping["Question"].tolist()
feature_columns = [col for col in df.columns if col in valid_features]

# Convert only numeric features
X = df[feature_columns].values.astype(float)

# Extract weights dynamically from mapping file
weights = dict(zip(question_mapping["Question"], question_mapping["Normalized Weight"]))
feature_weights = np.array([weights.get(col, 1.0) for col in feature_columns])

# Apply weights only to relevant features
weighted_X = X * feature_weights

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(weighted_X)

# Load new user (first record in encoded.csv)
new_user = X_scaled[0]

# Check if model or clustering file exists

if os.path.exists("roommate_model.h5"):
    # Load trained neural network model
    model = keras.models.load_model("roommate_model.h5", compile=False)
    predictions = model.predict(X_scaled).flatten()
    df["Compatibility Score"] = predictions
else:
    # Load clustering results
    clustered_df = pd.read_csv("clustered_roommates.csv")
    new_user_cluster = clustered_df.loc[0, "Cluster"]
    same_cluster = clustered_df[clustered_df["Cluster"] == new_user_cluster].copy()

    # Compute weighted similarity scores
    def weighted_similarity(user, others):
        return 1 - np.sum(feature_weights * np.abs(user - others), axis=1) / np.sum(feature_weights)

    same_cluster_features = X_scaled[clustered_df["Cluster"] == new_user_cluster]
    same_cluster["Compatibility Score"] = weighted_similarity(new_user, same_cluster_features)
    df = same_cluster

# Sort and get top 10 matches
top_matches = df.sort_values(by="Compatibility Score", ascending=False).head(10)
top_matches.to_csv("top_matches.csv", index=False)
print("Top 10 roommates:")
print(top_matches[["Full Name", "Compatibility Score"]])
