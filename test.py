import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

df["Full Name"] = df["Full Name"].astype(str)

# Ask for user input
reference_name = input("Enter the full name of the reference user: ")
if reference_name not in df["Full Name"].values:
    print("User not found in dataset.")
    exit()

# Get index of reference user
user_index = df[df["Full Name"] == reference_name].index[0]
new_user = X_scaled[user_index]

# Check if model or clustering file exists
if os.path.exists("roommate_model.h5"):
    # Load trained neural network model
    model = keras.models.load_model("roommate_model.h5", compile=False)
    feature_representations = model.predict(X_scaled)

    # Compute similarity between selected user and others
    user_representation = feature_representations[user_index]
    df["Compatibility Score"] = 1 - np.linalg.norm(feature_representations - user_representation, axis=1)
else:
    # Load clustering results
    clustered_df = pd.read_csv("clustered_roommates.csv")
    new_user_cluster = clustered_df.loc[user_index, "Cluster"]
    same_cluster = clustered_df[clustered_df["Cluster"] == new_user_cluster].copy()

    # Compute weighted similarity scores
    def weighted_similarity(user, others):
        return 1 - np.sum(feature_weights * np.abs(user - others), axis=1) / np.sum(feature_weights)

    same_cluster_features = X_scaled[clustered_df["Cluster"] == new_user_cluster]
    same_cluster["Compatibility Score"] = weighted_similarity(new_user, same_cluster_features)
    df = same_cluster

# Normalize compatibility scores
min_max_scaler = MinMaxScaler()
df["Compatibility Score"] = min_max_scaler.fit_transform(df[["Compatibility Score"]])

# Sort and get top 20 matches
top_matches = df.sort_values(by="Compatibility Score", ascending=False).head(20)

# Display top matches one by one
def plot_radar_chart(reference, match, match_name):
    """ Generate and display radar plot comparing the reference user and a match. """
    categories = np.array(feature_columns)  # Feature names
    values_ref = reference
    values_match = match

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_ref = np.concatenate((values_ref, [values_ref[0]]))
    values_match = np.concatenate((values_match, [values_match[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    ax.fill(angles, values_ref, color='blue', alpha=0.3, label='You')
    ax.fill(angles, values_match, color='red', alpha=0.3, label=match_name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, rotation=45, ha="right")

    plt.title(f"Comparison: You vs. {match_name}", fontsize=12)
    plt.legend()
    plt.show()

# Loop through top matches
for _, row in top_matches.iterrows():
    match_name = row["Full Name"]
    match_index = df[df["Full Name"] == match_name].index[0]

    # Extract relevant data
    match_features = X_scaled[match_index]

    # Find the 5-7 most similar factors
    similarity_diffs = np.abs(new_user - match_features)
    top_factor_indices = np.argsort(similarity_diffs)[:7]  # 7 smallest differences
    top_factors = [feature_columns[i] for i in top_factor_indices]

    # Display information
    print(f"\nMatch: {match_name}")
    print(f"Compatibility Score: {row['Compatibility Score']:.4f}")
    print("Most similar factors:", ", ".join(top_factors))

    # Show radar plot
    plot_radar_chart(new_user, match_features, match_name)

    # Wait for user input before showing the next match
    input("Press Enter to see the next match...")

print("Top 20 matches shown.")
