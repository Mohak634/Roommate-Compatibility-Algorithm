import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load datasets
df = pd.read_csv("encoded.csv")
question_mapping = pd.read_csv("question_mapping.csv")

# Extract valid feature columns from question_mapping
valid_features = question_mapping["Question"].tolist()
feature_columns = [col for col in df.columns if col in valid_features]

# Convert only numeric features
X = df[feature_columns].values.astype(float)

# Extract weights dynamically
weights = dict(zip(question_mapping["Question"], question_mapping["Normalized Weight"]))
feature_weights = np.array([weights.get(col, 1.0) for col in feature_columns])

# Apply weights
weighted_X = X * feature_weights
scaler = StandardScaler()
X_scaled = scaler.fit_transform(weighted_X)

# Take first record as new user
new_user = X_scaled[0].reshape(1, -1)

if os.path.exists("roommate_model.h5"):
    print("Using trained neural network model.")
    model = keras.models.load_model("roommate_model.h5", compile=False)
    predictions = model.predict(X_scaled).flatten()
    df["Compatibility Score"] = predictions
    df_sorted = df.sort_values(by="Compatibility Score", ascending=False)
    df_sorted.to_csv("top_matches.csv", index=False)
    print("Top matches saved to 'top_matches.csv'.")
else:
    print("No trained model found. Using clustering approach.")
    clustered_df = pd.read_csv("clustered_roommates.csv")
    new_user_cluster = clustered_df.loc[0, "Cluster"]
    cluster_mates = clustered_df[clustered_df["Cluster"] == new_user_cluster].copy()

    # Compute similarity scores within the cluster
    cluster_mates_features = scaler.transform(cluster_mates[feature_columns].values.astype(float))
    similarity_scores = cosine_similarity(new_user, cluster_mates_features).flatten()
    cluster_mates["Compatibility Score"] = similarity_scores
    df_sorted = cluster_mates.sort_values(by="Compatibility Score", ascending=False)
    df_sorted.to_csv("top_matches.csv", index=False)
    print("Top matches saved to 'top_matches.csv'.")

# Show top 10 matches
print("Top 10 roommates:")
print(df_sorted[["Full Name", "Compatibility Score"]].head(10))
