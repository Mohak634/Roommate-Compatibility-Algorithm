# DormMate backend

FastAPI backend for DormMate, a roommate-matching app. Students answer a lifestyle
survey; the API ranks other students by compatibility and explains each match.
The Flutter app lives in a separate repo (`dormmate_app`).

## Run

```
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # set MONGODB_URI; mongomock:// works for a quick look
python -m scripts.seed          # sample profiles + demo account (demo / dormmate-demo)
uvicorn app.app:app --reload    # http://127.0.0.1:8000/docs
pytest -q                       # must pass before every commit
```

Windows: `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Layout

- `app/core/matching.py`: the matching engine (weighted similarity over 14 lifestyle factors)
- `app/core/weights.json`: production weights, written by `scripts/evaluate_matching.py`
- `app/core/lifestyle.py` + `lifestyle.json`: K-means lifestyle types (centroids only)
- `app/core/vocab.py`: survey answer options, the single source of truth
- `app/services/`: auth (JWT), db (shared Mongo client), profiles, matches, interests, survey schema
- `app/endpoints/endpoints.py`: all routes; see `docs/API.md`
- `app/data/sample_profiles.json`: 400 generated sample students (no real people)
- `scripts/`: `seed.py`, `generate_profiles.py`, `evaluate_matching.py`, `pipeline.py` (original offline pipeline)

## Rules

- Never commit real survey data. The real CSV lives outside the repo; scripts take
  `--data` or `DORMMATE_SURVEY_CSV`. Only aggregates (weights, centroids) and
  generated sample profiles are committed.
- "Roommate Compatibility Rating" and "Mutual Friends" describe an existing
  roommate pair. They are the evaluation ground truth, never matching inputs.
- Change matching weights only by re-running `scripts/evaluate_matching.py --write-weights`
  and reporting the validation table in the PR. Evaluation rows must never use
  weights fitted on the pairs they score.
- Protected routes take the user from the JWT (`auth.current_user`), never from the URL or body.
- The API must not import TensorFlow, scikit-learn or plotting libraries; those are
  for offline scripts only (`requirements-dev.txt`, `requirements-pipeline.txt`).
- Scores are absolute 0-100 and symmetric; never rescale them per list.
