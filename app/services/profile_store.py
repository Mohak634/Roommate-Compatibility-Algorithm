"""Where candidate profiles come from.

Profiles are dicts of survey answers with an "id". For now they load from the
bundled cohort CSV; Phase 3 of the finish plan replaces this with the MongoDB
`profiles` collection (generated sample students plus real sign-ups).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from app.core.utils import get_data_path


@lru_cache(maxsize=1)
def _cohort() -> tuple[dict, ...]:
    df = pd.read_csv(get_data_path("Cleaned.csv"), encoding="utf-8-sig")
    df = df.where(pd.notna(df), None)
    out = []
    for row in df.to_dict("records"):
        row["id"] = str(row["Full Name"]).strip()
        out.append(row)
    return tuple(out)


def all_profiles() -> list[dict]:
    return list(_cohort())


def get_profile_by_id(profile_id: str) -> dict | None:
    for p in _cohort():
        if p["id"] == profile_id:
            return p
    return None


def refresh() -> None:
    """Drop the cache after profiles change."""
    _cohort.cache_clear()
