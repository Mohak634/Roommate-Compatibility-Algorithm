"""Generate sample student profiles for the live app, and fit lifestyle types.

The live app never holds real students' data. Instead it shows sample
profiles whose answers follow the real survey's patterns:

1. Pick a real response as a template.
2. Redraw each answer from that question's overall distribution with
   probability RESAMPLE, so answers keep some of their real correlations but
   no profile copies a real person.
3. Reject any profile that matches a real response on all but a few answers.
4. Add an invented name, a short bio built from the answers and a generated
   avatar. Every profile is marked `is_sample: true`.

Also fits K-means lifestyle types on the real responses and writes only the
centroids and labels to app/core/lifestyle.json.

    python -m scripts.generate_profiles --data /path/to/Cleaned.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from app.core import lifestyle
from app.core.bio import bio_from_answers
from app.core.matching import hobbies_of, normalise_course

OUT_PROFILES = Path("app/data/sample_profiles.json")
OUT_LIFESTYLE = Path("app/core/lifestyle.json")
RESAMPLE = 0.4
MIN_DIFFERENT_ANSWERS = 4

QUESTIONS = [
    "Gender", "Year of Study", "Course of Study", "Cleanliness Scale", "Cleaning Frequency", "Bedtime",
    "Noise Tolerance Sleep", "Study Frequency", "Study Location", "Noise Tolerance Study",
    "Social Preference", "Personal Space", "Party Frequency", "Roommate Host",
]

FIRST_NAMES = {
    "Male": ["Aarav", "Vihaan", "Arjun", "Kabir", "Rohan", "Ishaan", "Aditya", "Dhruv", "Karan", "Siddharth",
             "Yash", "Aryan", "Nikhil", "Rahul", "Varun", "Pranav", "Harsh", "Kunal", "Manav", "Ritvik",
             "Om", "Tanmay", "Ayaan", "Neel", "Shaurya", "Parth", "Dev", "Rudra", "Atharv", "Krish"],
    "Female": ["Ananya", "Diya", "Ishita", "Kavya", "Myra", "Saanvi", "Aditi", "Riya", "Tanya", "Meera",
               "Nisha", "Pooja", "Sneha", "Aisha", "Kiara", "Tara", "Anika", "Shreya", "Naina", "Isha",
               "Avni", "Zara", "Mahira", "Prisha", "Radhika", "Sana", "Trisha", "Vanya", "Jiya", "Navya"],
}
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Mehta", "Iyer", "Nair", "Reddy", "Kapoor", "Malhotra", "Bose",
              "Chatterjee", "Joshi", "Kulkarni", "Desai", "Patel", "Shah", "Singh", "Rao", "Menon", "Pillai",
              "Agarwal", "Bansal", "Khanna", "Saxena", "Mishra", "Pandey", "Chopra", "Arora", "Bhatt", "Das"]

def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Course of Study"] = df["Course of Study"].map(normalise_course)
    df["Year of Study"] = df["Year of Study"].astype(str).str.strip().str.title()
    df["Hobbies"] = [sorted(hobbies_of(r)) for r in df.to_dict("records")]
    return df


def generate(df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    real_rows = [tuple(str(r[q]) for q in QUESTIONS) for r in df.to_dict("records")]
    marginals = {q: [v for v in df[q].tolist() if pd.notna(v)] for q in QUESTIONS}
    all_hobbies = [h for hs in df["Hobbies"] for h in hs]
    used_names, profiles = set(), []
    while len(profiles) < n:
        t = df.iloc[rng.randrange(len(df))]
        answers = {q: (rng.choice(marginals[q]) if rng.random() < RESAMPLE or pd.isna(t[q]) else t[q])
                   for q in QUESTIONS}
        row = tuple(str(answers[q]) for q in QUESTIONS)
        if any(sum(a != b for a, b in zip(row, real)) < MIN_DIFFERENT_ANSWERS for real in real_rows):
            continue
        hobbies = list(t["Hobbies"]) if rng.random() > RESAMPLE else []
        while len(set(hobbies)) < 3:
            hobbies.append(rng.choice(all_hobbies))
        answers["Hobbies"] = sorted(set(hobbies))[:3]
        for k in ("Cleanliness Scale", "Noise Tolerance Sleep", "Noise Tolerance Study"):
            answers[k] = int(answers[k])
        gender = answers["Gender"] if answers["Gender"] in FIRST_NAMES else rng.choice(list(FIRST_NAMES))
        name = None
        while name is None or name in used_names:
            name = f"{rng.choice(FIRST_NAMES[gender])} {rng.choice(LAST_NAMES)}"
        used_names.add(name)
        answers["Full Name"] = name
        answers["Age"] = {"First Year": 18, "Second Year": 19, "Third Year": 20, "Fourth Year": 21}.get(
            answers["Year of Study"], 20) + rng.choice([0, 0, 1])
        pid = f"sample-{len(profiles) + 1:03d}"
        bio = bio_from_answers(answers)
        profiles.append({
            "id": pid,
            "name": name,
            "is_sample": True,
            "answers": answers,
            "bio": bio,
            "photo_url": f"https://api.dicebear.com/9.x/notionists/png?seed={pid}&size=256",
        })
    return profiles


def fit_lifestyle(df: pd.DataFrame, k: int, seed: int) -> dict:
    from sklearn.cluster import KMeans

    X = np.array([v for v in (lifestyle.vector(r) for r in df.to_dict("records")) if v is not None])
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
    labels = lifestyle.label_centroids(km.cluster_centers_, X.mean(axis=0))
    return {"source": f"K-means (k={k}) on {len(X)} survey responses; centroids only",
            "features": [f for f, _ in lifestyle.FEATURES],
            "centroids": np.round(km.cluster_centers_, 4).tolist(), "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("DORMMATE_SURVEY_CSV"))
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("-k", type=int, default=4, help="number of lifestyle types")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.data:
        ap.error("pass --data /path/to/Cleaned.csv or set DORMMATE_SURVEY_CSV")

    df = load_real(args.data)
    model = fit_lifestyle(df, args.k, args.seed)
    OUT_LIFESTYLE.write_text(json.dumps(model, indent=2))
    lifestyle._model.cache_clear()
    print("Lifestyle types:", model["labels"])

    profiles = generate(df, args.n, args.seed)
    for p in profiles:
        p["lifestyle_type"] = lifestyle.lifestyle_type(p["answers"])
    OUT_PROFILES.write_text(json.dumps(profiles, indent=1, ensure_ascii=False))
    print(f"Wrote {len(profiles)} sample profiles to {OUT_PROFILES}")


if __name__ == "__main__":
    main()
