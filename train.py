import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# Define features (only numeric, excluding name and any non-relevant columns)
feature_columns = [col for col in df.columns if col in valid_features]

# Convert only numeric features
X = df[feature_columns].values.astype(float)  # Convert only relevant columns

# Extract weights dynamically from mapping file
weights = dict(zip(question_mapping["Question"], question_mapping["Normalized Weight"]))
feature_weights = np.array([weights.get(col, 1.0) for col in feature_columns])

# Apply weights only to relevant features
weighted_X = X * feature_weights

# If compatibility ratings exist, use them as labels; otherwise, use clustering
y = df["Roommate Compatibility Rating"].values if "Roommate Compatibility Rating" in df.columns else None
supervised = y is not None

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(weighted_X)

# Check dataset size
if len(df) < 499:
    print("Dataset too small (<499 records). Running K-Means clustering instead of training the neural network.")
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    df.to_csv("clustered_roommates.csv", index=False)
    print("Clustering complete. Results saved to 'clustered_roommates.csv'.")
else:
    if supervised:
        # Split data for supervised learning
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build Neural Network Model
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')  # Regression output (compatibility score)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

        # Save model
        model.save("roommate_model.h5")
        print("Neural network training complete. Model saved.")
    else:
        # Clustering approach for unsupervised learning
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        df["Cluster"] = clusters
        df.to_csv("clustered_roommates.csv", index=False)
        print("Clustering complete. Results saved to 'clustered_roommates.csv'.")
