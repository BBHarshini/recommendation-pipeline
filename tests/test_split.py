import pandas as pd

from src.preparation.prepare import clean_interactions, time_based_split


def test_split_has_no_future_leakage():
    events = pd.DataFrame({"user_id": [1, 1, 2, 2, 3] * 4, "item_id": range(20),
                           "rating": [4] * 20, "timestamp": range(1000, 1020)})
    train, test, cutoff = time_based_split(events, 0.25)
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()
    assert len(train) + len(test) == len(events)


def test_repeated_ratings_keep_latest():
    events = pd.DataFrame({"user_id": ["1", "1"], "item_id": ["5", "5"],
                           "rating": ["2", "5"], "timestamp": ["100", "200"]})
    cleaned, removed = clean_interactions(events)
    assert removed == 1
    assert cleaned.iloc[0]["rating"] == 5
