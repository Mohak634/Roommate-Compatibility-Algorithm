import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the cleaned CSV file
df = pd.read_csv("Cleaned.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
print(df.columns.tolist())

# Define ordinal mappings
ordinal_mappings = {
    "Gender": {"Male": 1, "Female": 2, "Other": 3},
    "Bedtime": {"Early (Before 10 PM)": 0, "Moderate (10PM - 12AM)": 1, "Late (After 12AM)": 2},
    "Study Frequency": {"Rarely": 0, "Sometimes": 1, "Frequently": 2, "Constantly": 3},
    "Personal Space": {"Minimal": 0, "Moderate": 1, "A lot": 2},
    "Mutual Friends": {"None": 0, "1 to 3": 1, "4 to 5": 2, "More than 5": 3},
    "Cleaning Frequency": {"Rarely": 0, "Occasionally": 1, "Weekly": 2, "Daily": 3},
    "Social Preference": {"Introverted (Prefer minimal social interaction)": 0, "Ambiverted (Balanced)": 1, "Extroverted (Very social and outgoing)": 2},
    "Party Frequency": {"Never": 0, "Occasionally": 1, "Often": 2},
    "Year of Study": {"First Year": 1, "Second Year": 2, "Third Year": 3, "Fourth Year": 4, "Fifth Year": 5}
}


# Ensure integer type for numeric scales and preserve 'Unknown' for roommate names
df["Cleanliness Scale"] = df["Cleanliness Scale"].astype(int, errors='ignore')
df["Noise Tolerance Sleep"] = df["Noise Tolerance Sleep"].astype(int, errors='ignore')
df["Noise Tolerance Study"] = df["Noise Tolerance Study"].astype(int, errors='ignore')
df["Roommate Full Name"] = df["Roommate Full Name"].fillna("Unknown")
df["Roommate Compatibility Rating"] = df["Roommate Compatibility Rating"].fillna(0).astype(int, errors='ignore')

# Apply ordinal encoding
for col, mapping in ordinal_mappings.items():
    df[col] = df[col].map(mapping).fillna(0).astype(int, errors='ignore')  # Ensure integer type, fill missing with 0

# Define categorical columns for One-Hot Encoding (excluding ordinal ones)
categorical_columns = [
    "Study Location", "Roommate Host", "Course of Study", "Hobby 1", "Hobby 2", "Hobby 3"
]

# Define predefined category options for One-Hot Encoding
category_options = {
    "Study Location": ["In the room", "In the library or in other public spaces"],
    "Roommate Host": ["Yes", "No"],
    "Course of Study": ["BTech", "BSc", "BA", "BCom", "Psychology"],  # Add more courses here from possible responses
    "Hobby 1": ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"],
    "Hobby 2": ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"],
    "Hobby 3": ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"]
}

# One-Hot Encoding with predefined categories
encoder = OneHotEncoder(handle_unknown="ignore",
                        categories=[category_options.get(col, None) for col in categorical_columns])

# Transform categorical data and convert back to DataFrame
encoded_array = encoder.fit_transform(df[categorical_columns]).toarray()
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns))

# Merge with original DataFrame (drop old categorical columns)
df = df.drop(columns=categorical_columns).reset_index(drop=True)
df_encoded = pd.concat([df, encoded_df], axis=1)

# Save to new CSV
df_encoded.to_csv("encoded.csv", index=False, na_rep="0")  # Preserve 0 values

print("Encoding completed successfully. Saved as 'encoded.csv'.")
