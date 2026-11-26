"""Load the sample profiles and the demo account into MongoDB. Safe to re-run.

    python -m scripts.seed

Uses the same database settings as the API (MONGODB_URI, or MONGO_USER /
MONGO_PASS / MONGO_CLUSTER). The demo account's password comes from
DEMO_PASSWORD (default: dormmate-demo); the app's "Try the demo" button uses it.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from app.services import profile_store
from app.services.auth import hash_password
from app.services.db import get_db

DEMO_USERNAME = "demo"
DEMO_ANSWERS = {
    "Full Name": "Demo Student", "Age": 20, "Gender": "Male", "Course of Study": "BTech",
    "Year of Study": "Third Year", "Bedtime": "Late (After 12AM)", "Noise Tolerance Sleep": 4,
    "Cleanliness Scale": 8, "Cleaning Frequency": "Weekly", "Study Frequency": "Sometimes",
    "Study Location": "In the room", "Noise Tolerance Study": 5, "Social Preference": "Ambiverted (Balanced)",
    "Party Frequency": "Occasionally", "Roommate Host": "Yes", "Personal Space": "Moderate",
    "Hobbies": ["Watching Movies/TV", "Listening to Music", "Sports"],
}


def main() -> int:
    db = get_db()
    if db is None:
        print("No database configured. Set MONGODB_URI (or MONGO_USER/MONGO_PASS/MONGO_CLUSTER).")
        return 1
    samples = profile_store.load_sample_file()
    for p in samples:
        db.profiles.replace_one({"id": p["id"]}, p, upsert=True)
    print(f"Upserted {len(samples)} sample profiles.")

    password = os.getenv("DEMO_PASSWORD", "dormmate-demo")
    db.users.update_one(
        {"username": DEMO_USERNAME},
        {"$set": {"email": "demo@dormmate.example", "password_hash": hash_password(password)},
         "$setOnInsert": {"created_at": datetime.now(timezone.utc), "profile_complete": False}},
        upsert=True,
    )
    profile_store.save_user_profile(DEMO_USERNAME, DEMO_ANSWERS)
    print(f"Demo account ready: username '{DEMO_USERNAME}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
