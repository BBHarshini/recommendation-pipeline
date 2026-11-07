import pandas as pd

from src.validation.checks import apply_checks, interaction_checks, user_checks

RULES = {"rating_min": 1, "rating_max": 5, "age_min": 1, "age_max": 100,
         "allowed_genders": ["M", "F"], "earliest_timestamp": "1995-01-01"}


def test_user_checks_flag_bad_rows():
    users = pd.DataFrame({
        "user_id": ["1", "1", "2", "3", "4"],
        "age": ["30", "31", "250", "x", "40"],
        "gender": ["M", "M", "F", "F", "Q"],
        "occupation": ["writer"] * 5,
        "zip_code": ["00000"] * 5,
    })
    clean, errors, _ = apply_checks(users, user_checks(RULES))
    assert list(clean["user_id"]) == ["1"]
    reasons = dict(zip(errors.index, errors["failed_checks"]))
    assert "duplicate_id" in reasons[1]
    assert "age_out_of_range" in reasons[2]
    assert "invalid_type" in reasons[3]
    assert "invalid_gender" in reasons[4]


def test_interaction_checks_cover_ratings_timestamps_and_references():
    events = pd.DataFrame({
        "user_id": ["1", "1", "1", "1", "9", "1"],
        "item_id": ["1", "1", "2", "3", "1", "4"],
        "rating": ["4", "4", "6", "3", "5", "2"],
        "timestamp": ["880000000", "880000000", "880000000", "99", "880000000", "880000000"],
    })
    clean, errors, _ = apply_checks(events, interaction_checks(RULES, {1}, {1, 2, 3}))
    assert len(clean) == 1
    reasons = errors["failed_checks"].tolist()
    assert reasons == ["duplicate_record", "invalid_rating", "invalid_timestamp", "unknown_user", "unknown_item"]
