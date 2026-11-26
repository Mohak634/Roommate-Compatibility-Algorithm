"""Survey vocabulary: the answer options for every question, in order.

Single source of truth for the survey schema (app/services/survey.py), the
matching engine (app/core/matching.py) and the offline data pipeline
(scripts/pipeline.py). Kept free of heavy imports so the API can load it
without TensorFlow or plotting libraries.
"""

HOBBY_OPTIONS = ["Sports", "Listening to Music", "Reading", "Gaming", "Watching Movies/TV", "Painting/Drawing",
                 "Playing musical instruments", "Fitness", "Traveling", "Socializing/Parties", "Empty"]

ORDINAL_MAPPINGS = {
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

CATEGORY_OPTIONS = {
    "Study Location": ["In the room", "In the library or in other public spaces"],
    "Roommate Host": ["Yes", "No"],
    "Course of Study": ["BTech", "BSc", "BA", "BCom", "Psychology"],
    "Hobby 1": HOBBY_OPTIONS,
    "Hobby 2": HOBBY_OPTIONS,
    "Hobby 3": HOBBY_OPTIONS,
}
