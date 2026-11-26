import os

import pytest

os.environ["MONGODB_URI"] = "mongomock://"
os.environ["JWT_SECRET"] = "test-secret-not-for-production"


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.app import app
    from app.services import db, profile_store

    db.reset()
    profile_store.refresh()
    from scripts.seed import main as seed

    assert seed() == 0
    with TestClient(app) as c:
        yield c
    db.reset()
    profile_store.refresh()


ANSWERS = {
    "Full Name": "Test Person", "Age": 19, "Gender": "Female", "Course of Study": "BSc",
    "Year of Study": "Second Year", "Bedtime": "Moderate (10PM - 12AM)", "Noise Tolerance Sleep": 6,
    "Cleanliness Scale": 9, "Cleaning Frequency": "Daily", "Study Frequency": "Frequently",
    "Study Location": "In the room", "Noise Tolerance Study": 3,
    "Social Preference": "Introverted (Prefer minimal social interaction)", "Party Frequency": "Never",
    "Roommate Host": "No", "Personal Space": "A lot", "Hobbies": ["Reading", "Painting/Drawing"],
}


def signup(client, username="tester", email="t@example.com", password="password123"):
    r = client.post("/api/auth/signup", json={"username": username, "email": email, "password": password})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}
