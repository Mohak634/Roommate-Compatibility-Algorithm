# DormMate API

Base path `/api`. Interactive docs at `/docs` when the server is running.
Routes marked 🔒 need `Authorization: Bearer <token>`; get the token from signup or login.

## Auth

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/auth/signup` | `{username, email, password}` | `201 {token, user}` (409 if taken, 422 if weak) |
| POST | `/auth/login` | `{username_or_email, password}` | `{token, user}` (401 if wrong) |
| GET 🔒 | `/me` | | `{username, email, profile_complete}` |

`user` = `{username, email, profile_complete}`. Tokens last 7 days. Demo account: `demo` / `dormmate-demo`.

## Survey and profile

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/survey/schema` | | `{sections, questions}` (build the form from this) |
| GET 🔒 | `/profile` | | `{profile: Card or null, answers or null}` |
| PUT 🔒 | `/profile` | `{answers: {<question key>: <answer>}}` | `{profile: Card, answers}`; 422 lists problems |

Every question in the schema must be answered. `Hobbies` is a list (max 3).

## Matches

| Method | Path | Returns |
| --- | --- | --- |
| GET 🔒 | `/matches?top_n=10` | `{you: Card, method, matches: [Card + {score, closest, differences}]}`; 409 until the survey is saved |
| GET 🔒 | `/matches/{profile_id}` | Card + `{score, closest, differences, radar, breakdown, mutual, contact_email}` |
| GET 🔒 | `/matches/mutual` | `{mutual: [{id, name, photo_url, score, mutual}]}` |

- `score`: 0-100, absolute and symmetric (the same pair always scores the same).
- `closest` / `differences`: up to 3 factor labels where the two people agree most / differ most.
- `radar`: `[{label, you, them}]`, values 0-100, one axis per lifestyle question.
- `breakdown`: `[{factor, similarity}]`, similarity 0-100 per factor.
- `contact_email`: only after a mutual like, otherwise `null`.

## Interests

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST 🔒 | `/interests` | `{profile_id, action: "like" or "skip"}` | `{mutual, score}` |
| GET 🔒 | `/interests/sent` | | `{sent: [{id, name, photo_url, score, mutual}]}` (newest first) |

Sample profiles can't respond, so a like on a sample profile scoring 70 or more is
answered automatically. Likes between real users need both people.

## Card

```json
{
  "id": "sample-074", "name": "Ayaan Agarwal", "is_sample": true,
  "photo_url": "https://api.dicebear.com/9.x/notionists/png?seed=sample-074&size=256",
  "bio": "Second Year BTech student. Night owl, ...",
  "lifestyle_type": "Sleeps through anything · Homebody",
  "course": "BTech", "year": "Second Year", "age": 19,
  "hobbies": ["Painting/Drawing", "Socializing/Parties"]
}
```

## Changes from the old API (for the Flutter app)

| Old | New |
| --- | --- |
| `POST /Sign Up New User` | `POST /auth/signup` (now returns a token) |
| `POST /Login User` | `POST /auth/login` |
| `GET/POST /profile/{username}` | `GET/PUT /profile` (user from token) |
| `GET /users` + `POST /matches {name}` | `GET /matches` (matches for the logged-in user) |
| `POST /compare {user1, user2}` | `GET /matches/{profile_id}` |
| `POST /interest {from_user, to_user, action, score}` | `POST /interests {profile_id, action}` (score computed by the server) |
| `GET /interests/sent/{user}` | `GET /interests/sent` |
| `GET /matches/mutual/{user}` | `GET /matches/mutual` |
| `student_email` on every compare | `contact_email`, only after a mutual like |
| Training and pipeline routes | Removed from the API; see `scripts/pipeline.py` |
