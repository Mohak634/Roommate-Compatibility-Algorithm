"""Web-safe wrappers around the matching logic in main_script.

The original test_* functions are CLI-oriented: they call input() for the
reference name and plt.show() / input("Press Enter") to page through matches.
Neither works behind an HTTP API. These functions take the name as an
argument and return JSON-serialisable dicts instead.
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.core.main_script import preprocess_testdata
from app.core.utils import get_data_path, get_model_path

# Cache the Keras model so we don't reload the .h5 on every request.
_model = None


def _load_model():
    global _model
    if _model is None:
        from tensorflow import keras  # imported lazily to speed cold start

        _model = keras.models.load_model(get_model_path("roommate_model.h5"), compile=False)
    return _model


def list_users():
    """Return every Full Name in the dataset (for the picker in the UI)."""
    df, _, _, _ = preprocess_testdata()
    names = sorted(df["Full Name"].astype(str).unique().tolist())
    return names


def _top_factors(ref_features, match_features, feature_columns, k=7):
    diffs = np.abs(ref_features - match_features)
    idx = np.argsort(diffs)[:k]
    return [feature_columns[i] for i in idx]


def _row_extras(row):
    """Pull a few human-readable fields off the encoded row for the card."""
    extras = {}
    for key in ("Age", "Year of Study", "Cleanliness Scale"):
        if key in row and not pd.isna(row[key]):
            try:
                extras[key] = int(row[key])
            except (ValueError, TypeError):
                extras[key] = None
    return extras


# --- Radar axes: per-question survey values, mirroring the original ---------
# plot_radar_chart axes (the numeric survey columns, one axis per question).
# Each question is MinMax-normalised across the cohort to 0..100 so the UI
# radar is bounded; axis order matches the old matplotlib plot.
_RADAR_QUESTIONS = [
    "Bedtime",
    "Cleaning Frequency",
    "Cleanliness Scale",
    "Year of Study",
    "Mutual Friends",
    "Roommate Compatibility Rating",
    "Party Frequency",
    "Personal Space",
    "Social Preference",
    "Noise Tolerance Study",
    "Study Frequency",
    "Noise Tolerance Sleep",
]

_radar_cache = None


def _radar_frame():
    """Per-person, per-question values in 0..1, built once from encoded.csv."""
    global _radar_cache
    if _radar_cache is None:
        df = pd.read_csv(get_data_path("encoded.csv"))
        cols = [q for q in _RADAR_QUESTIONS if q in df.columns]
        normed = pd.DataFrame(
            MinMaxScaler().fit_transform(df[cols].astype(float)), columns=cols
        )
        _radar_cache = (normed, cols)
    return _radar_cache


def _radar_for(ref_index, match_index):
    normed, cols = _radar_frame()
    return [
        {
            "label": c,
            "you": round(float(normed[c].iloc[ref_index]) * 100, 1),
            "them": round(float(normed[c].iloc[match_index]) * 100, 1),
        }
        for c in cols
    ]


def _course_of(row):
    """Decode the one-hot Course of Study_* columns back to a label."""
    for col in row.index:
        if col.startswith("Course of Study_") and row[col] == 1:
            return col.replace("Course of Study_", "")
    return None


def _student_email(name):
    """MOCK contact detail — derived from the name until a real student
    directory exists. The dataset has no email column."""
    slug = re.sub(r"[^a-z0-9]+", ".", str(name).lower()).strip(".")
    return f"{slug}@dormmate.edu"


def get_matches(reference_name, use_model=True, top_n=10):
    """Rank everyone against reference_name and return the top matches.

    use_model=True  -> distance between neural-net embeddings
    use_model=False -> weighted feature similarity (cluster-style)
    """
    df, X_scaled, feature_columns, feature_weights = preprocess_testdata()
    df["Full Name"] = df["Full Name"].astype(str)

    if reference_name not in df["Full Name"].values:
        return {"success": False, "message": "User not found in dataset.", "matches": []}

    ref_index = df[df["Full Name"] == reference_name].index[0]
    ref_features = X_scaled[ref_index]

    if use_model:
        model = _load_model()
        reps = model.predict(X_scaled, verbose=0)
        ref_rep = reps[ref_index]
        df["Compatibility Score"] = 1 / (1 + np.linalg.norm(reps - ref_rep, axis=1))
    else:
        denom = np.sum(feature_weights)
        df["Compatibility Score"] = 1 - np.sum(
            feature_weights * np.abs(ref_features - X_scaled), axis=1
        ) / denom

    # Normalise to 0..1 across the cohort, then drop the reference themselves.
    df["Compatibility Score"] = MinMaxScaler().fit_transform(df[["Compatibility Score"]])
    ranked = df.sort_values("Compatibility Score", ascending=False)
    ranked = ranked[ranked["Full Name"] != reference_name].head(top_n)

    matches = []
    for _, row in ranked.iterrows():
        m_index = df[df["Full Name"] == row["Full Name"]].index[0]
        matches.append(
            {
                "name": row["Full Name"],
                "score": round(float(row["Compatibility Score"]) * 100, 1),
                "top_factors": _top_factors(ref_features, X_scaled[m_index], feature_columns),
                "course": _course_of(row),
                "radar": _radar_for(ref_index, m_index),
                **_row_extras(row),
            }
        )

    return {
        "success": True,
        "reference": reference_name,
        "method": "neural-net" if use_model else "weighted-similarity",
        "matches": matches,
    }


def compare_users_json(user1_name, user2_name, use_model=True):
    """Compare exactly two users; returns score + shared factors (no plot)."""
    df, X_scaled, feature_columns, feature_weights = preprocess_testdata()
    df["Full Name"] = df["Full Name"].astype(str)

    for name in (user1_name, user2_name):
        if name not in df["Full Name"].values:
            return {"success": False, "message": f"'{name}' not found in dataset."}

    i1 = df[df["Full Name"] == user1_name].index[0]
    i2 = df[df["Full Name"] == user2_name].index[0]
    f1, f2 = X_scaled[i1], X_scaled[i2]

    if use_model:
        model = _load_model()
        r1 = model.predict(X_scaled[[i1]], verbose=0)[0]
        r2 = model.predict(X_scaled[[i2]], verbose=0)[0]
        score = 1 / (1 + np.linalg.norm(r1 - r2))
    else:
        score = 1 - np.sum(feature_weights * np.abs(f1 - f2)) / np.sum(feature_weights)

    row2 = df.iloc[i2]
    return {
        "success": True,
        "user1": user1_name,
        "user2": user2_name,
        "method": "neural-net" if use_model else "weighted-similarity",
        "compatibility_score": round(float(score) * 100, 1),
        "top_factors": _top_factors(f1, f2, feature_columns),
        "radar": _radar_for(i1, i2),
        "course": _course_of(row2),
        "student_email": _student_email(user2_name),
        **_row_extras(row2),
    }
