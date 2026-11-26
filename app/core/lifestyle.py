"""Lifestyle types: K-means clusters over lifestyle answers.

Centroids are fitted offline on the survey cohort (scripts/generate_profiles.py)
and stored in app/core/lifestyle.json as aggregates only. At runtime a profile
is assigned to its nearest centroid with plain numpy, so the API needs no ML
library.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core import matching

# Features used for clustering, each scaled to 0..1.
FEATURES = [
    ("Cleanliness Scale", "scale"),
    ("Cleaning Frequency", "ordinal"),
    ("Bedtime", "ordinal"),
    ("Noise Tolerance Sleep", "scale"),
    ("Study Frequency", "ordinal"),
    ("Noise Tolerance Study", "scale"),
    ("Social Preference", "ordinal"),
    ("Party Frequency", "ordinal"),
    ("Personal Space", "ordinal"),
]

# Words for a cluster that sits clearly above (+) or below (-) the cohort average.
TRAIT_WORDS = {
    "Cleanliness Scale": ("Neat", "Relaxed about mess"),
    "Cleaning Frequency": ("Cleans often", "Cleans now and then"),
    "Bedtime": ("Night owl", "Early bird"),
    "Noise Tolerance Sleep": ("Light sleeper", "Sleeps through anything"),
    "Study Frequency": ("Studious", "Studies light"),
    "Noise Tolerance Study": ("Fine with noise", "Needs quiet to study"),
    "Social Preference": ("Social", "Quiet"),
    "Party Frequency": ("Party-goer", "Homebody"),
    "Personal Space": ("Needs space", "Easygoing about space"),
}


def vector(profile: dict) -> np.ndarray | None:
    """Lifestyle answers as a 0..1 vector; None if any are missing."""
    out = []
    for key, kind in FEATURES:
        if kind == "scale":
            v = matching._scale(profile.get(key))
            v = None if v is None else (v - 1) / 9
        else:
            code = matching._ordinal(key, profile.get(key))
            values = matching.ORDINAL_MAPPINGS[key].values()
            v = None if code is None else (code - min(values)) / (max(values) - min(values))
        if v is None:
            return None
        out.append(v)
    return np.array(out)


@lru_cache(maxsize=1)
def _model() -> dict | None:
    path = Path(__file__).with_name("lifestyle.json")
    try:
        data = json.loads(path.read_text())
        data["centroids"] = np.array(data["centroids"])
        return data
    except (OSError, ValueError, KeyError):
        return None


def lifestyle_type(profile: dict) -> str | None:
    model, v = _model(), vector(profile)
    if model is None or v is None:
        return None
    i = int(np.argmin(np.linalg.norm(model["centroids"] - v, axis=1)))
    return model["labels"][i]


def label_centroids(centroids: np.ndarray, overall_mean: np.ndarray, words: int = 2) -> list[str]:
    """Name each cluster by the traits where it differs most from the average."""
    labels = []
    for c in centroids:
        dev = c - overall_mean
        top = np.argsort(-np.abs(dev))[:words]
        parts = [TRAIT_WORDS[FEATURES[i][0]][0 if dev[i] > 0 else 1] for i in top]
        labels.append(" · ".join(parts))
    return labels
