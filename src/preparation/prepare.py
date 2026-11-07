"""Stage 3: Data preparation.

Cleans the validated tables (types, formats, missing values, repeated ratings) and
creates the train/test datasets with a time-based split: everything before the
cutoff time is training data, everything after it is test data.
"""

import pandas as pd

from src.common import get_logger, project_path, write_json


def clean_users(users):
    users = users.drop(columns=["record_source"], errors="ignore").copy()
    users["user_id"] = users["user_id"].astype(int)
    users["age"] = users["age"].astype(int)
    users["gender"] = users["gender"].str.strip().str.upper()
    users["occupation"] = users["occupation"].str.strip().str.lower()
    users["zip_code"] = users["zip_code"].fillna("unknown").astype(str).str.strip().str.upper()
    users["age_group"] = pd.cut(users["age"], bins=[0, 17, 24, 34, 44, 54, 120],
                                labels=["<18", "18-24", "25-34", "35-44", "45-54", "55+"]).astype(str)
    return users


def clean_items(items):
    items = items.drop(columns=["record_source"], errors="ignore").copy()
    items["item_id"] = items["item_id"].astype(int)
    items["title"] = items["title"].str.strip()
    items["genres"] = items["genres"].fillna("").replace("", "unknown")

    items["release_date"] = pd.to_datetime(items["release_date"], format="%d-%b-%Y", errors="coerce")
    items["release_year"] = items["release_date"].dt.year
    items["release_year_imputed"] = items["release_year"].isna()
    items["release_year"] = items["release_year"].fillna(items["release_year"].median()).astype(int)
    items["release_date"] = items["release_date"].dt.strftime("%Y-%m-%d")

    # video_release_date is empty for every MovieLens 100K movie, so it carries no information
    if items["video_release_date"].isna().all():
        items = items.drop(columns=["video_release_date"])
    items["imdb_url"] = items["imdb_url"].fillna("")
    return items


def clean_interactions(events):
    events = events.drop(columns=["record_source"], errors="ignore").copy()
    for col in ["user_id", "item_id", "rating", "timestamp"]:
        events[col] = pd.to_numeric(events[col]).astype(int)
    events["event_time"] = pd.to_datetime(events["timestamp"], unit="s", utc=True)

    # if a user rated the same item more than once, keep only the latest rating
    before = len(events)
    events = (events.sort_values("timestamp")
              .drop_duplicates(subset=["user_id", "item_id"], keep="last")
              .reset_index(drop=True))
    return events, before - len(events)


def time_based_split(events, test_fraction):
    """Global time cutoff: the newest `test_fraction` of interactions become the test set."""
    cutoff = int(events["timestamp"].quantile(1 - test_fraction))
    train = events[events["timestamp"] < cutoff].copy()
    test = events[events["timestamp"] >= cutoff].copy()
    return train, test, cutoff


def run(cfg):
    logger = get_logger(cfg)
    in_dir = project_path(cfg["data"]["validated_dir"])
    out_dir = project_path(cfg["data"]["processed_dir"])

    users = clean_users(pd.read_csv(in_dir / "users.csv", dtype=str))
    items = clean_items(pd.read_csv(in_dir / "items.csv", dtype=str))
    events, repeated = clean_interactions(pd.read_csv(in_dir / "interactions.csv", dtype=str))
    logger.info("Cleaned data: %d users, %d items, %d interactions (%d repeated ratings collapsed)",
                len(users), len(items), len(events), repeated)

    train, test, cutoff = time_based_split(events, cfg["split"]["test_fraction"])
    train_users, train_items = set(train["user_id"]), set(train["item_id"])
    test_users = set(test["user_id"])

    split_info = {
        "method": "global time cutoff",
        "cutoff_timestamp": cutoff,
        "cutoff_datetime": pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat(),
        "train_interactions": len(train),
        "test_interactions": len(test),
        "train_period": [str(train["event_time"].min()), str(train["event_time"].max())],
        "test_period": [str(test["event_time"].min()), str(test["event_time"].max())],
        "test_users": len(test_users),
        "test_users_with_train_history": len(test_users & train_users),
        "test_users_new": len(test_users - train_users),
        "test_interactions_on_items_unseen_in_train": int((~test["item_id"].isin(train_items)).sum()),
        "repeated_ratings_collapsed": repeated,
        "items_with_imputed_release_year": int(items["release_year_imputed"].sum()),
    }

    users.to_csv(out_dir / "users.csv", index=False)
    items.to_csv(out_dir / "items.csv", index=False)
    events.to_csv(out_dir / "interactions.csv", index=False)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    write_json(split_info, out_dir / "split_info.json")

    logger.info("Time split at %s: %d train / %d test interactions; %d of %d test users have training history",
                split_info["cutoff_datetime"], len(train), len(test),
                split_info["test_users_with_train_history"], len(test_users))
    return split_info
