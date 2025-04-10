import gspread
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import seaborn as sns
from app.core.utils import get_data_path, get_model_path


#--------------------------------------Miscellaneous-----------------------------------------------------------------
# Enable GPU usage
def enablegpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is enabled")
    else:
        print("No GPU found, using CPU")


def preprocess_data(encoded_file=get_data_path("encoded.csv"), mapping_file=get_data_path("question_mapping.csv")):
    # Load datasets
    df = pd.read_csv(encoded_file)
    question_mapping = pd.read_csv(mapping_file)

    # Extract valid feature columns
    valid_features = question_mapping["Question"].tolist()
    feature_columns = [col for col in df.columns if col in valid_features]

    # Convert to numeric values
    X = df[feature_columns].values.astype(float)

    # Apply feature weights
    weights = dict(zip(question_mapping["Question"], question_mapping["Normalized Weight"]))
    feature_weights = np.array([weights.get(col, 1.0) for col in feature_columns])
    weighted_X = X * feature_weights

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(weighted_X)

    # Check if compatibility ratings exist
    y = df["Roommate Compatibility Rating"].values if "Roommate Compatibility Rating" in df.columns else None

    return df, X_scaled, y  # y is None if no labels exist


#--------------------------------------Training-----------------------------------------------------------------

def train_cluster():
    enablegpu()
    df, X_scaled, _ = preprocess_data()

    # Check dataset size
    print("Dataset too small (<499 records). Running K-Means clustering instead of training the neural network.")
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["Cluster"] = clusters
    df.to_csv(get_data_path("clustered_roommates.csv"), index=False)
    print("Clustering complete. Results saved to 'clustered_roommates.csv'.")


def train_model():
    enablegpu()
    df, X_scaled, y = preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build Neural Network Model - increase layers after new data 
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=8)

    # Save model
    model.save(get_model_path("roommate_model.h5"))
    print("Neural network training complete. Model saved.")


#--------------------------------------Testing-----------------------------------------------------------------

def preprocess_testdata():
    """Load and preprocess data (feature extraction, scaling, and weighting)."""
    df = pd.read_csv(get_data_path("encoded.csv"))
    question_mapping = pd.read_csv(get_data_path("question_mapping.csv"))

    # Extract relevant feature columns
    valid_features = question_mapping["Question"].tolist()
    feature_columns = [col for col in df.columns if col in valid_features]

    # Convert only numeric features
    X = df[feature_columns].values.astype(float)

    # Extract weights dynamically
    weights = dict(zip(question_mapping["Question"], question_mapping["Normalized Weight"]))
    feature_weights = np.array([weights.get(col, 1.0) for col in feature_columns])

    # Apply weights and standardize features
    weighted_X = X * feature_weights
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(weighted_X)

    return df, X_scaled, feature_columns, feature_weights


def plot_radar_chart(reference, match, match_name, feature_columns):
    """Generate and display radar plot comparing the reference user and a match."""
    categories = np.array(feature_columns)
    values_ref = np.concatenate((reference, [reference[0]]))
    values_match = np.concatenate((match, [match[0]]))
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values_ref, color='blue', alpha=0.3, label='You')
    ax.fill(angles, values_match, color='red', alpha=0.3, label=match_name)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8, rotation=45, ha="right")
    plt.title(f"Comparison: You vs. {match_name}", fontsize=12)
    plt.legend()
    plt.show()


def display_top_matches(df, X_scaled, feature_columns, reference_name):
    """Display top matches using radar plots and similarity factors."""
    df["Compatibility Score"] = MinMaxScaler().fit_transform(df[["Compatibility Score"]])

    # Sort and get top 20 matches
    top_matches = df.sort_values(by="Compatibility Score", ascending=False).head(20)
    reference_index = df[df["Full Name"] == reference_name].index[0]
    reference_features = X_scaled[reference_index]

    for _, row in top_matches.iterrows():
        match_name = row["Full Name"]
        match_index = df[df["Full Name"] == match_name].index[0]
        match_features = X_scaled[match_index]

        # Find the 5-7 most similar factors
        similarity_diffs = np.abs(reference_features - match_features)
        top_factor_indices = np.argsort(similarity_diffs)[:7]  # 7 smallest differences
        top_factors = [feature_columns[i] for i in top_factor_indices]

        # Show radar plot
        plot_radar_chart(reference_features, match_features, match_name, feature_columns)

        # Display information
        print(f"\nMatch: {match_name}")
        print(f"Compatibility Score: {row['Compatibility Score']:.4f}")
        print("Most similar factors:", ", ".join(top_factors))

        input("Press Enter to see the next match...")

    print("Top 20 matches shown.")


def test_with_model():
    """Test matching using the trained neural network model."""
    df, X_scaled, feature_columns, _ = preprocess_testdata()

    reference_name = test_matching()  # ✅ Fix: No argument needed

    if reference_name is None:
        return  # ✅ Handle missing user case

    model = keras.models.load_model(get_model_path("roommate_model.h5"), compile=False)
    feature_representations = model.predict(X_scaled)

    # Get user index and representation
    user_index = df[df["Full Name"] == reference_name].index[0]
    user_representation = feature_representations[user_index]

    # Compute similarity scores
    df["Compatibility Score"] = 1 / (1 + np.linalg.norm(feature_representations - user_representation, axis=1))

    # Display results
    display_top_matches(df, X_scaled, feature_columns, reference_name)


def test_with_clusters():
    """Test matching using clustering instead of a trained model."""
    df, X_scaled, feature_columns, feature_weights = preprocess_testdata()

    reference_name = test_matching()  # ✅ Fix: No argument needed

    if reference_name is None:
        return  # ✅ Handle missing user case

    clustered_df = pd.read_csv(get_data_path("clustered_roommates.csv"))

    if reference_name not in clustered_df["Full Name"].values:
        print("User not found in clustered data.")
        return

    user_index = clustered_df[clustered_df["Full Name"] == reference_name].index[0]
    new_user = X_scaled[user_index]

    # Identify user's cluster
    new_user_cluster = clustered_df.loc[user_index, "Cluster"]
    same_cluster = clustered_df[clustered_df["Cluster"] == new_user_cluster].copy()

    # Compute weighted similarity scores
    def weighted_similarity(user, others):
        return 1 - np.sum(feature_weights * np.abs(user - others), axis=1) / np.sum(feature_weights)

    same_cluster_features = X_scaled[clustered_df["Cluster"] == new_user_cluster]
    same_cluster["Compatibility Score"] = weighted_similarity(new_user, same_cluster_features)

    # Display results
    display_top_matches(same_cluster, X_scaled, feature_columns, reference_name)


def test_matching():
    """Ask for user input and return the reference user's name."""
    df, _, _, _ = preprocess_testdata()
    df["Full Name"] = df["Full Name"].astype(str)

    # Ask for user input
    reference_name = input("Enter the full name of the reference user: ").strip()

    if reference_name not in df["Full Name"].values:
        print("User not found in dataset.")
        return None  # Fix: Return None instead of breaking the program

    return reference_name  # Fix: Return the input for function calls

def compare_two_users(user1_name, user2_name, use_model=True):
    """Compare two specific users using radar plot and similarity score."""
    df, X_scaled, feature_columns, feature_weights = preprocess_testdata()

    # Check users exist
    if user1_name not in df["Full Name"].values or user2_name not in df["Full Name"].values:
        print("One or both users not found in dataset.")
        return

    index1 = df[df["Full Name"] == user1_name].index[0]
    index2 = df[df["Full Name"] == user2_name].index[0]

    user1_features = X_scaled[index1]
    user2_features = X_scaled[index2]

    # Optionally load model and compare embeddings
    if use_model:
        model = keras.models.load_model(get_model_path("roommate_model.h5"), compile=False)
        user1_rep = model.predict(X_scaled[[index1]])[0]
        user2_rep = model.predict(X_scaled[[index2]])[0]
        compatibility_score = 1 / (1 + np.linalg.norm(user1_rep - user2_rep))
    else:
        # Use weighted similarity (cluster-style)
        diff = np.abs(user1_features - user2_features)
        compatibility_score = 1 - np.sum(feature_weights * diff) / np.sum(feature_weights)

    # Top similar factors
    similarity_diffs = np.abs(user1_features - user2_features)
    top_indices = np.argsort(similarity_diffs)[:7]
    top_factors = [feature_columns[i] for i in top_indices]

    # Plot radar chart
    plot_radar_chart(user1_features, user2_features, user2_name, feature_columns)

    # Show results
    print(f"\nComparison: {user1_name} vs. {user2_name}")
    print(f"Compatibility Score: {compatibility_score:.4f}")
    print("Most similar factors:", ", ".join(top_factors))


#--------------------------------------Fetch Sheet responses-----------------------------------------------------------------


def fetch_google_sheet(sheet_id="1DETGS8rhcTcfWpxLzr56v_GTQccDfxWD9S-qFpCPnz0", raw_csv=get_data_path("rawData.csv")):
    """Fetches data from a Google Sheet and saves it as raw CSV."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(get_data_path("credentials.json"), scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Raw data saved to {raw_csv}")


#--------------------------------------Cleaning-----------------------------------------------------------------

def clean_data(input_csv=get_data_path("rawData.csv"), output_csv=get_data_path("Cleaned.csv")):
    """Cleans the raw data and saves it as a processed CSV."""
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    column_mappings = {
        "Full Name": "Full Name",
        "Age (For example: 20)": "Age",
        "Gender": "Gender",
        "Course of Study (For Example: BTech)": "Course of Study",
        "Year of Study:": "Year of Study",
        "Please enter your roommate's full name if they're also filling the form": "Roommate Full Name",
        "On a scale of 1–10, how important is cleanliness to you?": "Cleanliness Scale",
        "How often do you clean your personal space?": "Cleaning Frequency",
        "What is your typical bedtime?": "Bedtime",
        "On a scale of 1–10, how much does noise during your sleep bother you?": "Noise Tolerance Sleep",
        "How often do you study": "Study Frequency",
        "Where do you prefer to study?": "Study Location",
        "On a scale of 1–10, how tolerant are you of noise or activity while studying?": "Noise Tolerance Study",
        "How social are you?": "Social Preference",
        "How much personal space do you need?": "Personal Space",
        "How often do you attend or host parties?": "Party Frequency",
        "Are you comfortable with your roommate hosting friends or gatherings?": "Roommate Host",
        "What are your top 3 hobbies?": "Hobbies",
        "On a scale of 1-10, how would you rate your overall compatibility as a roommate?": "Roommate Compatibility Rating",
        "How many mutual friends do you and your roommate share?": "Mutual Friends"
    }
    df = df.rename(columns={k: v for k, v in column_mappings.items() if k in df.columns})
    selected_columns = [v for v in column_mappings.values() if v in df.columns]
    df = df[selected_columns]
    df.fillna({"Roommate Full Name": "Unknown", "Hobbies": "Empty", "Roommate Compatibility Rating": 0}, inplace=True)
    df["Hobbies"] = df["Hobbies"].str.replace(r"Sports \(.*?\)", "Sports", regex=True)
    df.replace({"1-3": "1 to 3", "4-5": "4 to 5"}, inplace=True)
    if "Hobbies" in df.columns:
        df[['Hobby 1', 'Hobby 2', 'Hobby 3']] = df["Hobbies"].str.split(",", expand=True).reindex(
            columns=[0, 1, 2]).fillna("Empty")
        df.drop(columns=["Hobbies"], inplace=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Cleaned data saved to {output_csv}")


#--------------------------------------Encoding-----------------------------------------------------------------

def encode_data(input_csv=get_data_path("Cleaned.csv"), output_csv=get_data_path("encoded.csv")):
    """Encodes categorical and ordinal variables from a cleaned dataset and saves the encoded version."""
    df = pd.read_csv(input_csv)
    if "Full Name" in df.columns:
        df["Full Name"] = (
            df["Full Name"]
            .astype(str)
            .str.strip()  # Remove leading/trailing spaces
            .str.replace(r"\s+", " ", regex=True)  # Replace multiple spaces with a single space
            .str.title()  # Capitalize first letter of each word
        )
    df.columns = df.columns.str.strip()
    ordinal_mappings = {
        "Gender": {"Male": 1, "Female": 2, "Other": 3},
        "Bedtime": {"Early (Before 10 PM)": 0, "Moderate (10PM - 12AM)": 1, "Late (After 12AM)": 2},
        "Study Frequency": {"Rarely": 0, "Sometimes": 1, "Frequently": 2, "Constantly": 3},
        "Personal Space": {"Minimal": 0, "Moderate": 1, "A lot": 2},
        "Mutual Friends": {"None": 0, "1 to 3": 1, "4 to 5": 2, "More than 5": 3},
        "Cleaning Frequency": {"Rarely": 0, "Occasionally": 1, "Weekly": 2, "Daily": 3},
        "Social Preference": {"Introverted (Prefer minimal social interaction)": 0, "Ambiverted (Balanced)": 1,
                              "Extroverted (Very social and outgoing)": 2},
        "Party Frequency": {"Never": 0, "Occasionally": 1, "Often": 2},
        "Year of Study": {"First Year": 1, "Second Year": 2, "Third Year": 3, "Fourth Year": 4, "Fifth Year": 5}
    }
    for col, mapping in ordinal_mappings.items():
        df[col] = df[col].map(mapping).fillna(0).astype(int, errors='ignore')
    categorical_columns = ["Study Location", "Roommate Host", "Course of Study", "Hobby 1", "Hobby 2", "Hobby 3"]
    category_options = {
        "Study Location": ["In the room", "In the library or in other public spaces"],
        "Roommate Host": ["Yes", "No"],
        "Course of Study": ["BTech", "BSc", "BA", "BCom", "Psychology"],
        "Hobby 1": ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                    "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"],
        "Hobby 2": ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                    "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"],
        "Hobby 3": ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                    "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"]
    }
    encoder = OneHotEncoder(handle_unknown="ignore",
                            categories=[category_options.get(col, None) for col in categorical_columns])
    encoded_array = encoder.fit_transform(df[categorical_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns))
    df = df.drop(columns=categorical_columns).reset_index(drop=True)
    df_encoded = pd.concat([df, encoded_df], axis=1)
    df_encoded.to_csv(output_csv, index=False, na_rep="0")
    print(f"✅ Encoding completed successfully. Saved as '{output_csv}'")


"""List of functions defined in this script for DormMate
fetch_google_sheet
clean_data
encode_data
train_cluster
train_model
test_with_clusters
test_with_models
test_matching (accepts user input and called in test functions) 
preprocess_data (called in training functions) 
preprocess_datatest (called in testing functions) 
display_top_matches
plot_radar_chart
enable_gpu
"""
