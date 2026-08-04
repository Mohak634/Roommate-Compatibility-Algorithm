# DormMate — Project Checklist

Last updated: 2026-07-08 (after Phase B + match profile screen)

## Quick reference — how to run

```
# Backend (from Roommate-Compatibility-Algorithm/, needs .venv311 = Python 3.11 for TensorFlow)
.venv311\Scripts\python.exe -m uvicorn app.app:app --host 127.0.0.1 --port 8000

# Flutter web app (from Projects/dormmate_app/)
C:\src\flutter\bin\flutter.bat run -d web-server --web-port 5000 --web-hostname localhost
```

- Offline mode: set `kUseLocalData = true` in `dormmate_app/lib/repositories/providers.dart` (runs off `assets/fixtures/`).
- Regenerate fixtures after backend response changes: curl the endpoint into `assets/fixtures/*.json`.

---

## ✅ Done

### Backend (FastAPI — this repo)
- [x] Web-safe matching module (`app/core/web_matching.py`) — no `input()`/`plt.show()`
- [x] `GET /api/users`, `POST /api/matches` (neural-net ranking, per-question radar
      values [12 axes matching the original plot], decoded course), `POST /api/compare`
- [x] Auth against MongoDB (`/api/Login User`, `/api/Sign Up New User`),
      Windows cp1252 emoji-crash fix, CORS middleware
- [x] Mongo credentials as parts in `.env` (URL-encoded via `quote_plus`)
- [x] Interests (Phase A): `POST /api/interest` (like/skip, unique index,
      auto-reciprocate ≥85% flagged `demo_auto`), `GET /api/interests/sent/{user}`,
      `GET /api/matches/mutual/{user}`
- [x] Survey schema (Phase B): `GET /api/survey/schema` generated from the
      encoder's own `ORDINAL_MAPPINGS`/`CATEGORY_OPTIONS` (single source of truth)
- [x] Profiles (Phase B): `GET/POST /api/profile/{username}` with validation,
      stored in `preferences` collection, flips `users.profile_complete`
- [x] `/api/compare` extended: radar axes, age/year/course, mock
      `student_email` (derived from name — flagged)
- [x] `requirements.txt`; `.venv311` (Python 3.11 — TF has no 3.14 build)

### Flutter app (`Projects/dormmate_app`)
- [x] Production architecture: Riverpod + go_router + feature-first folders,
      repository layer (Api/Mock/Local per service), design-system tokens
      (`core/theme/`), reusable core widgets — UI never imports `http`
- [x] Splash (SVG brand mark `assets/icons/icon.svg`, loading spinner)
- [x] Login + Signup (validation, show/hide password, forgot-password UI,
      University-ID button)
- [x] Discover: `flutter_card_swiper` deck, real recommendations, fl_chart
      radar (You vs Them), top-factor chips, reference picker, match snackbar
- [x] Matches tab: Mutual / Requests-sent (real interests data), MatchCard,
      refresh, loading/empty/error states
- [x] My preferences (Phase B): schema-driven survey form (sliders, choice
      chips, hobby multi-pick max 3), grouped by section, save to backend
- [x] Session persistence (`shared_preferences`) + router auth guard
      (fixes swipes being attributed to "guest")
- [x] Match profile screen (`/student/:id`): big radar, "most aligned on"
      factors, profile details, and student email revealed **only on mutual
      match** (locked card otherwise) — no in-app messaging by design
- [x] Offline fixtures + `kUseLocalData` switch
- [x] `flutter analyze` clean; widget tests (splash, StudentCard radar)

### Infra
- [x] Flutter SDK 3.44.4 installed at `C:\src\flutter`
- [x] `.mcp.json` for MongoDB MCP (gitignored; connection string pasted by Mohak)
- [x] `.env` untracked + gitignored

---

## 🔜 To do (agreed roadmap)

- [ ] **Phase C — Onboarding**: survey as step-by-step pages after signup
      (progress bar, back/next); login redirects to onboarding while
      `profile_complete: false`. Reuses `SurveyQuestionTile` + `surveyAnswersProvider`.
- [ ] **Phase D — Profile upgrade**: real stats (likes sent / mutual counts from
      interests), radar of own answers, richer profile card.
- [ ] **Phase E — Mongo matching migration**: encode `preferences` docs into the
      feature space and add them to the cohort → new signups become matchable;
      `train_model`/`train_cluster`/`preprocess_*` read from Mongo
      (`encoded`, `question_mapping` collections) instead of CSVs; retire CSV files.
      Only the data-loading step changes — TF/KMeans logic stays.
- [ ] University verification screen (MockUniversityRepository already stubbed)
- [ ] Settings screen: campus info, matching-logic explainer, privacy,
      notifications, theme toggle (dark ThemeData already built), logout
- [ ] Messaging / chat (mock first), notification service, request approval flow
- [ ] Deployment story (host FastAPI + Flutter web build)

---

## 🔧 Improvements / tech debt

**Security (do before anything public):**
- [ ] Rotate the MongoDB password — the old one is in pushed git history
      (commit `be5d7c6`); consider purging history with git-filter-repo
- [ ] Auth returns raw user JSON — add token/JWT sessions before production
- [ ] CORS is wildcard (`*`) for dev — restrict origins for production

**Demo scaffolding to remove later:**
- [ ] Auto-reciprocated likes (`demo_auto: true` docs) — delete behavior + docs
      when real users exist
- [ ] Clean up the 12 pre-fix `from_user: "guest"` docs in `interests`
- [ ] `bio` / `photoUrl` are mocked (dataset has neither)
- [ ] `student_email` is derived from the name (mock) — replace with real
      campus directory / verified emails; also gate server-side (currently the
      API returns it to any caller; only the UI enforces the mutual-match lock)
- [ ] Discover's "viewing as" reference picker — replace with the logged-in
      user's own encoded profile once Phase E lands

**Architecture / data:**
- [ ] Identity model: account usernames (`users`) and dataset people
      (`Full Name`) are separate namespaces — unify in Phase E
      (likes currently bridge them: from=username, to=dataset name)
- [ ] Backend opens/closes a Mongo connection per request — pool for production
- [ ] `dormmate_app` has no git repo — `git init` it (or fold into a monorepo)
- [ ] MongoDB MCP hasn't connected in-session yet — verify `.mcp.json` string
      on next session start
- [ ] Old artifacts to prune: `dormmate-web/` (superseded Next.js attempt),
      `app/static/`, `dormmate_walkthrough_script.md`, unused `matches` collection

**Polish:**
- [ ] Radar card: 12 axis labels are dense on small screens — consider
      abbreviating labels or a "top 6 axes" compact mode
- [ ] Test coverage: only 2 widget tests, no backend tests (pytest for
      interests/profiles services would be cheap wins)
- [ ] Dev ergonomics: run uvicorn with `--reload` during development
