from app.core import matching

A = {
    "id": "a", "Cleanliness Scale": 9, "Cleaning Frequency": "Daily", "Bedtime": "Late (After 12AM)",
    "Noise Tolerance Sleep": 3, "Study Frequency": "Frequently", "Study Location": "In the room",
    "Noise Tolerance Study": 4, "Social Preference": "Ambiverted (Balanced)", "Roommate Host": "Yes",
    "Party Frequency": "Occasionally", "Hobbies": ["Gaming", "Reading", "Fitness"],
    "Personal Space": "Moderate", "Year of Study": "Third Year", "Course of Study": "BTech",
    "Roommate Compatibility Rating": 10, "Mutual Friends": "More than 5",
}
B = {
    "id": "b", "Cleanliness Scale": 4, "Cleaning Frequency": "Rarely", "Bedtime": "Early (Before 10 PM)",
    "Noise Tolerance Sleep": 9, "Study Frequency": "Rarely", "Study Location": "In the library or in other public spaces",
    "Noise Tolerance Study": 9, "Social Preference": "Extroverted (Very social and outgoing)", "Roommate Host": "No",
    "Party Frequency": "Often", "Hobby 1": "Sports", "Hobby 2": " Socializing/Parties", "Hobby 3": "Traveling",
    "Personal Space": "A lot", "Year of Study": "first year", "Course of Study": "B.Tech",
    "Roommate Compatibility Rating": 2, "Mutual Friends": "None",
}


def test_identical_profiles_score_100():
    assert matching.score(A, dict(A, id="a2")) == 100


def test_score_is_symmetric_and_bounded():
    s = matching.score(A, B)
    assert s == matching.score(B, A)
    assert 0 <= s < 50


def test_score_is_deterministic():
    assert matching.score(A, B) == matching.score(A, B)


def test_pair_level_fields_are_not_inputs():
    """Changing the roommate rating or mutual friends must not change the score."""
    changed = dict(B, **{"Roommate Compatibility Rating": 10, "Mutual Friends": "More than 5"})
    assert matching.score(A, B) == matching.score(A, changed)


def test_missing_answers_are_skipped_not_penalised():
    partial = {"id": "p", "Cleanliness Scale": 9, "Bedtime": "Late (After 12AM)"}
    assert matching.score(A, partial) == 100


def test_answer_normalisation():
    assert matching.normalise_course(" btech ") == "BTech"
    assert matching.normalise_course("B.Tech") == "BTech"
    assert matching.hobbies_of(B) == {"Sports", "Socializing/Parties", "Traveling"}
    assert matching._ordinal("Year of Study", "first year") == 1


def test_explain_lists_agreements_and_differences():
    why = matching.explain(A, B)
    labels = {f.key: f.label for f in matching.FACTORS}
    sims = matching.factor_similarities(A, B)
    lowest = {labels[k] for k, s in sims.items() if s == min(sims.values())}
    assert len(why["differences"]) == 3 and set(why["differences"]) <= lowest
    assert matching.explain(A, A)["differences"] == []
    assert len(matching.explain(A, A)["closest"]) == 3


def test_rank_excludes_reference_and_sorts_best_first():
    C = dict(A, id="c", **{"Cleanliness Scale": 8})
    ranked = matching.rank(A, [A, B, C], top_n=5)
    assert [m["id"] for m in ranked] == ["c", "b"]
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_production_weights_cover_every_factor():
    assert set(matching.WEIGHTS) == set(matching.FACTOR_KEYS)
    assert all(w >= 0 for w in matching.WEIGHTS.values())
