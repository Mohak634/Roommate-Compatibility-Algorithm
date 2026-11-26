"""Candidate profiles: generated sample students plus real sign-ups.

With a database, profiles live in the `profiles` collection (seeded from
app/data/sample_profiles.json by scripts/seed.py). Without one, the app runs
read-only on the bundled sample file.

Profile document:
  { id, name, is_sample, answers: {<question>: <answer>}, bio, photo_url,
    lifestyle_type, updated_at }
For real users, id == username.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.bio import bio_from_answers
from app.core.lifestyle import lifestyle_type
from app.services.db import get_db

SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_profiles.json"
CACHE_SECONDS = 60

_cache: dict = {"at": 0.0, "profiles": None}


def load_sample_file() -> list[dict]:
    return json.loads(SAMPLE_FILE.read_text())


def all_profiles() -> list[dict]:
    now = time.monotonic()
    if _cache["profiles"] is None or now - _cache["at"] > CACHE_SECONDS:
        db = get_db()
        _cache["profiles"] = (list(db.profiles.find({}, {"_id": 0})) if db is not None
                              else load_sample_file())
        _cache["at"] = now
    return _cache["profiles"]


def get_profile(profile_id: str) -> dict | None:
    db = get_db()
    if db is not None:
        return db.profiles.find_one({"id": profile_id}, {"_id": 0})
    return next((p for p in load_sample_file() if p["id"] == profile_id), None)


def save_user_profile(username: str, answers: dict) -> dict:
    db = get_db()
    doc = {
        "id": username,
        "name": (answers.get("Full Name") or username).strip(),
        "is_sample": False,
        "answers": answers,
        "bio": bio_from_answers(answers),
        "photo_url": f"https://api.dicebear.com/9.x/notionists/png?seed={username}&size=256",
        "lifestyle_type": lifestyle_type(answers),
        "updated_at": datetime.now(timezone.utc),
    }
    db.profiles.replace_one({"id": username}, doc, upsert=True)
    db.users.update_one({"username": username}, {"$set": {"profile_complete": True}})
    refresh()
    doc.pop("_id", None)
    return doc


def refresh() -> None:
    _cache["profiles"] = None
