"""Validate the matching engine against real roommate pairs.

The original survey asked each student to name their current roommate. Where
both roommates answered, we know a real pair that chose to live together. For
each such person, rank everyone else with the matching engine and record where
their actual roommate lands. A useful engine puts real roommates near the top.

Compared:
- hand-set weights (the production engine)
- weights fitted with logistic regression on real pairs vs random non-pairs,
  scored leave-one-pair-out so a pair is never ranked by weights learned from it
- random ranking (analytic baseline)
- the original neural network, if TensorFlow and the model file are available

The survey data holds real students' answers, so it is NOT in the repo. Point
the script at a private copy of the cleaned survey CSV:

    python -m scripts.evaluate_matching --data /path/to/Cleaned.csv
"""

from __future__ import annotations

import argparse
import os
import random
from statistics import median

import numpy as np
import pandas as pd

from app.core.matching import FACTORS, FACTOR_KEYS, factor_similarities, score


def load_profiles(path: str) -> tuple[list[dict], dict[str, str]]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Full Name"] = df["Full Name"].astype(str).str.strip()
    profiles = df.to_dict("records")
    for p in profiles:
        p["id"] = p["Full Name"]
    names = {p["id"] for p in profiles}
    roommate = {
        p["id"]: str(p.get("Roommate Full Name", "")).strip()
        for p in profiles
        if str(p.get("Roommate Full Name", "")).strip() in names
    }
    return profiles, roommate


def pairs_of(roommate: dict[str, str]) -> set[tuple[str, str]]:
    return {tuple(sorted((a, b))) for a, b in roommate.items()}


def rank_of_roommate(person: dict, roommate_id: str, profiles: list[dict], weights=None) -> int:
    scored = sorted(
        ((score(person, c, weights), c["id"]) for c in profiles if c["id"] != person["id"]),
        key=lambda t: (-t[0], t[1]),
    )
    return 1 + [cid for _, cid in scored].index(roommate_id)


def summarise(name: str, ranks: list[int], n_candidates: int) -> dict:
    r = np.array(ranks)
    return {
        "method": name,
        "queries": len(r),
        "median_rank": float(median(r)),
        "top10_hit_rate": round(float((r <= 10).mean()), 3),
        "candidates": n_candidates,
    }


def feature_vector(a: dict, b: dict) -> list[float]:
    sims = factor_similarities(a, b)
    return [sims.get(k, 0.5) for k in FACTOR_KEYS]  # neutral 0.5 where an answer is missing


BLEND = 0.5  # production weights = BLEND * fitted + (1 - BLEND) * hand-set; fixed, not tuned


def _hand_set() -> np.ndarray:
    w = np.array([f.weight for f in FACTORS])
    return w / w.sum()


def fitted_weight_ranks(profiles, roommate, negatives_per_pair=20, seed=7, blend=1.0):
    from sklearn.linear_model import LogisticRegression

    by_id = {p["id"]: p for p in profiles}
    pairs = sorted(pairs_of(roommate))
    rng = random.Random(seed)
    ids = list(by_id)
    ranks = []
    for held_out in pairs:
        X, y = [], []
        for a, b in pairs:
            if (a, b) == held_out:
                continue
            X.append(feature_vector(by_id[a], by_id[b])); y.append(1)
            for _ in range(negatives_per_pair):
                c = rng.choice(ids)
                if c in (a, b) or tuple(sorted((a, c))) in pairs:
                    continue
                X.append(feature_vector(by_id[a], by_id[c])); y.append(0)
        model = LogisticRegression(C=0.5, max_iter=1000).fit(np.array(X), np.array(y))
        coef = np.clip(model.coef_[0], 0, None)
        coef = coef / coef.sum() if coef.sum() else np.ones_like(coef) / len(coef)
        weights = dict(zip(FACTOR_KEYS, blend * coef + (1 - blend) * _hand_set()))
        for person, mate in ((held_out[0], held_out[1]), (held_out[1], held_out[0])):
            if roommate.get(person) == mate:
                ranks.append(rank_of_roommate(by_id[person], mate, profiles, weights))
    return ranks


def fit_all_pairs(profiles, roommate, negatives_per_pair=20, seed=7) -> dict[str, float]:
    """Weights fitted on every real pair, for production. Negative weights are clipped to 0
    and the rest scaled to sum to 1."""
    from sklearn.linear_model import LogisticRegression

    by_id = {p["id"]: p for p in profiles}
    pairs = sorted(pairs_of(roommate))
    rng = random.Random(seed)
    ids = list(by_id)
    X, y = [], []
    for a, b in pairs:
        X.append(feature_vector(by_id[a], by_id[b])); y.append(1)
        for _ in range(negatives_per_pair):
            c = rng.choice(ids)
            if c in (a, b) or tuple(sorted((a, c))) in pairs:
                continue
            X.append(feature_vector(by_id[a], by_id[c])); y.append(0)
    coef = np.clip(LogisticRegression(C=0.5, max_iter=1000).fit(np.array(X), np.array(y)).coef_[0], 0, None)
    coef = coef / coef.sum()
    return {k: round(float(v), 4) for k, v in zip(FACTOR_KEYS, coef)}


def neural_network_ranks(roommate):
    """Ranks under the original network (distance between its 1-D outputs)."""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        from scripts.pipeline import preprocess_testdata
        from tensorflow import keras
        from app.core.utils import get_model_path
    except Exception:
        return None
    try:
        df, X, _, _ = preprocess_testdata()
        reps = keras.models.load_model(get_model_path("roommate_model.h5"), compile=False).predict(X, verbose=0)
    except Exception:
        return None
    names = list(df["Full Name"].astype(str).str.strip())
    idx = {n: i for i, n in enumerate(names)}
    ranks = []
    for a, b in roommate.items():
        if a in idx and b in idx:
            i = idx[a]
            d = np.linalg.norm(reps - reps[i], axis=1)
            d[i] = np.inf
            order = np.argsort(d, kind="stable")
            ranks.append(1 + int(np.where(order == idx[b])[0][0]))
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.getenv("DORMMATE_SURVEY_CSV"), required=False,
                    help="private path to the cleaned survey CSV (or set DORMMATE_SURVEY_CSV)")
    ap.add_argument("--skip-nn", action="store_true", help="skip the neural-network baseline")
    ap.add_argument("--write-weights", metavar="PATH",
                    help="fit weights on all pairs and write them as JSON (e.g. app/core/weights.json)")
    args = ap.parse_args()
    if not args.data:
        ap.error("pass --data /path/to/Cleaned.csv or set DORMMATE_SURVEY_CSV")

    profiles, roommate = load_profiles(args.data)
    by_id = {p["id"]: p for p in profiles}
    n = len(profiles) - 1
    print(f"{len(profiles)} people, {len(pairs_of(roommate))} real roommate pairs, "
          f"{len(roommate)} people whose roommate is in the data\n")

    hand_set = {f.key: f.weight for f in FACTORS}  # explicit: never the saved production weights,
    # which were fitted on these same pairs and would leak the answer
    rows = [summarise("Hand-set weights",
                      [rank_of_roommate(by_id[a], b, profiles, hand_set) for a, b in roommate.items()], n)]
    lifestyle = {f.key: (0.0 if f.group == "Academics" else f.weight) for f in FACTORS}
    rows.append(summarise("Hand-set, lifestyle only (no year/course)",
                          [rank_of_roommate(by_id[a], b, profiles, lifestyle) for a, b in roommate.items()], n))
    rows.append(summarise("Fitted weights (leave-one-pair-out)", fitted_weight_ranks(profiles, roommate), n))
    rows.append(summarise(f"Blend {BLEND:.0%} fitted (production, leave-one-pair-out)",
                          fitted_weight_ranks(profiles, roommate, blend=BLEND), n))
    rows.append({"method": "Random", "queries": len(roommate), "median_rank": (n + 1) / 2,
                 "top10_hit_rate": round(10 / n, 3), "candidates": n})
    if not args.skip_nn:
        nn = neural_network_ranks(roommate)
        if nn:
            rows.append(summarise("Original neural network", nn, n))

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    fitted = fit_all_pairs(profiles, roommate)
    hand = dict(zip(FACTOR_KEYS, _hand_set()))
    production = {k: round(BLEND * fitted[k] + (1 - BLEND) * hand[k], 4) for k in FACTOR_KEYS}
    print("\nWeights                hand-set  fitted  production")
    for f in FACTORS:
        print(f"  {f.label:<22} {hand[f.key]:.3f}    {fitted[f.key]:.3f}   {production[f.key]:.3f}")
    if args.write_weights:
        import json
        with open(args.write_weights, "w") as fh:
            json.dump({"source": f"{BLEND:.0%} logistic regression fitted on real roommate pairs vs random "
                                 f"non-pairs, {1 - BLEND:.0%} hand-set weights (scripts/evaluate_matching.py)",
                       "weights": production, "fitted": fitted, "hand_set": {k: round(v, 4) for k, v in hand.items()}},
                      fh, indent=2)
        print(f"\nWrote fitted weights to {args.write_weights}")


if __name__ == "__main__":
    main()
