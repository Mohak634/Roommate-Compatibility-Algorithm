# DormMate: progress, changes and decisions

Last updated: 26 Nov 2026. Goal: finish DormMate once, deploy it, and use it on the resume.

- Pull request: [#1 (draft)](https://github.com/Mohak634/Roommate-Compatibility-Algorithm/pull/1), branch `deploy-prep-and-anonymise`, CI green
- Full plan: "DormMate: Finish and Deploy Plan" (Claude doc, linked from the PR conversation)
- API reference: [`docs/API.md`](API.md) · Repo rules and run commands: [`CLAUDE.md`](../CLAUDE.md)

## Status

| Phase | What | Status |
| --- | --- | --- |
| 0 | Cleanup: anonymise data, remove caches, deploy basics | Done |
| 1 | Matching engine: weighted similarity, no TensorFlow | Done (K-means label moved to phase 3) |
| 2 | Validation against real roommate pairs | Done |
| 3 | Sample profiles, lifestyle types, one `profiles` collection, seed script | Done |
| 4 | Token login, new routes, contact details gated, shared DB client | Done |
| 5 | Tests (18) and GitHub Actions | Done |
| 6 | UI rework in the Flutter app | **Blocked:** app not in git (see below) |
| 7 | Deploy (Railway + MongoDB Atlas + Flutter web) and README | Not started |

## Decisions

| Date | Decision | Why |
| --- | --- | --- |
| 26 Nov | Replace the neural network with weighted similarity; keep the network only as a baseline in the validation table | It used the roommate rating as both input and target, and ranked real roommates no better than random |
| 26 Nov | The roommate rating is pair-level ground truth for evaluation, never a matching input (same for mutual friends) | It describes an existing pair, as originally intended |
| 26 Nov | Production weights = 50% fitted on real pairs + 50% hand-set (fixed in advance, not tuned) | Fitted weights alone zero out cleanliness, bedtime and space with only 24 pairs |
| 26 Nov | The live app shows only generated sample profiles | No real student's data reaches the server; visitors see full match lists |
| 26 Nov | Real survey CSVs removed from the repo; scripts read a private copy (`--data` / `DORMMATE_SURVEY_CSV`) | Classmates' answers shouldn't be public |
| 26 Nov | Old git history left as is for now (still contains names, emails and the old DB password) | Rewriting history can't be undone; Mohak's call |
| 26 Nov | Likes on sample profiles scoring 70+ are answered automatically (flagged `demo_auto`); real users need each other | Otherwise a visitor can never see a mutual match |
| 26 Nov | Cut for good: chat, notifications, request approvals, university verification, settings beyond logout, the old Next.js attempt, retraining from the live server | None of it changes what a recruiter sees |
| 26 Nov | Survey no longer asks the two pair-level questions | They only make sense for existing roommates |

## Validation (24 real roommate pairs, 97 candidates each)

| Method | Real roommate in top 10 | Median rank |
| --- | --- | --- |
| Production (50% fitted, 50% hand-set) | 25% | 23.5 |
| Fitted weights (leave-one-pair-out) | 34% | 20.5 |
| Hand-set weights | 16% | 35.5 |
| Hand-set, lifestyle only (no year/course) | 18% | 39 |
| Original neural network | 9% | 45.5 |
| Random | 10% | 49 |

Year of study is the strongest single signal, likely because hostels group students by year.
Reproduce: `python -m scripts.evaluate_matching --data /private/path/Cleaned.csv`.

## What changed (backend)

- `app/core/matching.py`: weighted similarity over 14 factors; absolute, symmetric 0-100 scores; agreements, differences and a per-factor breakdown. Restores hobbies, course, study location and hosting friends, which the old encoding dropped, and fixes "First year" being read as year 0.
- `scripts/evaluate_matching.py`, `app/core/weights.json`: validation and production weights.
- `scripts/generate_profiles.py`, `app/data/sample_profiles.json`: 400 sample students from the real survey's answer patterns; `app/core/lifestyle.json`: K-means centroids and labels only.
- `app/services/`: shared DB client, JWT auth, profiles, matches, interests. Routes renamed (old -> new table in `docs/API.md`).
- TensorFlow, plotting and Google Sheets code moved to `scripts/pipeline.py` and out of the API. Install 2.0 GB -> 179 MB; startup about 25 s -> 4 s.
- 18 tests, GitHub Actions, `CLAUDE.md`, `.env.example`, Procfile, Python 3.11 pin.

## Open items

1. **Flutter app:** `dormmate_app` exists only on Mohak Laptop and was never committed. Either push it to a new private repo (`Mohak634/dormmate-app`), or open the Claude chat in the desktop app on Mohak Laptop so Claude can reach the folder. The current app won't work against the new routes until phase 6 updates it.
2. **Phase 6 UI:** design the new screens first (login, onboarding survey, matches, match detail, profile), get approval, then build. Add "Try the demo", match cards with score and reasons, sample-profile badge; remove unfinished screens.
3. **Phase 7 deploy:** Atlas M0 cluster and Railway project (Mohak); env vars `MONGODB_URI`, `JWT_SECRET`, `ALLOWED_ORIGINS`, `DEMO_PASSWORD`; run `python -m scripts.seed` once; Flutter web on Vercel or Netlify; README with screenshots, architecture, the validation table and the live link.
4. **Lifestyle-type labels** read oddly in places ("Early bird · Party-goer"); tune the wording during phase 6.
5. **Housekeeping (Mohak):** keep the original survey CSV somewhere safe off the repo; delete the old Atlas cluster or user if it still exists; decide later whether to purge old git history; update the resume bullet once deployed.

## Resume from here

```
git checkout deploy-prep-and-anonymise
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
MONGODB_URI=mongomock:// python -m scripts.seed   # or a real MONGODB_URI in .env
uvicorn app.app:app --reload                      # demo login: demo / dormmate-demo
pytest -q
```
