from tests.conftest import ANSWERS, signup


def test_health_and_schema(client):
    assert client.get("/health").json() == {"status": "ok"}
    keys = [q["key"] for q in client.get("/api/survey/schema").json()["questions"]]
    assert "Bedtime" in keys
    assert "Roommate Compatibility Rating" not in keys and "Mutual Friends" not in keys


def test_signup_login_and_me(client):
    headers = signup(client)
    assert client.get("/api/me", headers=headers).json()["profile_complete"] is False
    r = client.post("/api/auth/login", json={"username_or_email": "T@example.com", "password": "password123"})
    assert r.status_code == 200 and r.json()["user"]["username"] == "tester"
    assert client.post("/api/auth/login", json={"username_or_email": "tester", "password": "nope"}).status_code == 401


def test_duplicate_and_weak_signups_are_refused(client):
    signup(client)
    dup = client.post("/api/auth/signup", json={"username": "tester", "email": "x@example.com", "password": "password123"})
    assert dup.status_code == 409
    weak = client.post("/api/auth/signup", json={"username": "ab", "email": "nope", "password": "short"})
    assert weak.status_code == 422


def test_protected_routes_need_a_valid_token(client):
    for path in ("/api/me", "/api/profile", "/api/matches", "/api/interests/sent", "/api/matches/mutual"):
        assert client.get(path).status_code == 401
    assert client.get("/api/me", headers={"Authorization": "Bearer forged"}).status_code == 401


def test_new_user_is_matched_right_after_the_survey(client):
    headers = signup(client)
    assert client.get("/api/matches", headers=headers).status_code == 409  # survey not done yet
    saved = client.put("/api/profile", json={"answers": ANSWERS}, headers=headers)
    assert saved.status_code == 200, saved.text
    assert saved.json()["profile"]["lifestyle_type"]
    assert client.get("/api/me", headers=headers).json()["profile_complete"] is True

    body = client.get("/api/matches?top_n=10", headers=headers).json()
    scores = [m["score"] for m in body["matches"]]
    assert len(scores) == 10 and scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)
    assert all(m["id"] != "tester" for m in body["matches"])
    assert body["matches"][0]["closest"]


def test_invalid_answers_are_rejected(client):
    headers = signup(client)
    bad = dict(ANSWERS, **{"Bedtime": "Whenever", "Unknown": 1})
    r = client.put("/api/profile", json={"answers": bad}, headers=headers)
    assert r.status_code == 422 and "Bedtime" in r.json()["detail"]
    missing = {k: v for k, v in ANSWERS.items() if k != "Hobbies"}
    assert client.put("/api/profile", json={"answers": missing}, headers=headers).status_code == 422


def test_contact_details_only_after_a_mutual_like(client):
    headers = signup(client)
    client.put("/api/profile", json={"answers": ANSWERS}, headers=headers)
    top = client.get("/api/matches", headers=headers).json()["matches"][0]

    detail = client.get(f"/api/matches/{top['id']}", headers=headers).json()
    assert detail["contact_email"] is None and detail["radar"] and detail["breakdown"]

    r = client.post("/api/interests", json={"profile_id": top["id"], "action": "like"}, headers=headers).json()
    assert r["score"] >= 70 and r["mutual"] is True  # sample profiles answer likes from good matches
    detail = client.get(f"/api/matches/{top['id']}", headers=headers).json()
    assert detail["contact_email"].endswith("@dormmate.example")
    assert client.get("/api/matches/mutual", headers=headers).json()["mutual"][0]["id"] == top["id"]

    low = client.get("/api/matches?top_n=50", headers=headers).json()["matches"][-1]
    if low["score"] < 70:
        assert client.post("/api/interests", json={"profile_id": low["id"], "action": "like"},
                           headers=headers).json()["mutual"] is False
    assert top["id"] in {e["id"] for e in client.get("/api/interests/sent", headers=headers).json()["sent"]}


def test_real_users_need_each_other_to_match(client):
    a = signup(client, "alice", "a@example.com")
    b = signup(client, "bobby", "b@example.com")
    client.put("/api/profile", json={"answers": ANSWERS}, headers=a)
    client.put("/api/profile", json={"answers": dict(ANSWERS, **{"Full Name": "Bob"})}, headers=b)
    assert client.post("/api/interests", json={"profile_id": "bobby", "action": "like"}, headers=a).json()["mutual"] is False
    assert client.get("/api/matches/bobby", headers=a).json()["contact_email"] is None
    assert client.post("/api/interests", json={"profile_id": "alice", "action": "like"}, headers=b).json()["mutual"] is True
    assert client.get("/api/matches/bobby", headers=a).json()["contact_email"] == "b@example.com"


def test_demo_account_works(client):
    r = client.post("/api/auth/login", json={"username_or_email": "demo", "password": "dormmate-demo"})
    headers = {"Authorization": f"Bearer {r.json()['token']}"}
    assert len(client.get("/api/matches", headers=headers).json()["matches"]) == 10
