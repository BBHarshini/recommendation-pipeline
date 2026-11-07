"""Stage 5: Feature management.

Feature logic lives in functions that take the interactions and a reference time
(`as_of`). The same functions are used for training (as_of = split cutoff, using
training data only) and for prediction (as_of = now, using all data), so training
and serving features are always computed the same way.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.common import get_logger, markdown_table, project_path, read_json, write_json
from src.features.registry import FEATURES, names

SECONDS_PER_DAY = 86400


def _explode_genres(events, items):
    """One row per (interaction, genre)."""
    g = items[["item_id", "genres"]].assign(genre=items["genres"].str.split("|")).explode("genre")
    return events[["user_id", "item_id"]].merge(g[["item_id", "genre"]], on="item_id")


def user_genre_shares(events, items):
    """Share of each user's ratings that fall in each genre (rows: users, columns: genres)."""
    eg = _explode_genres(events, items)
    counts = eg.groupby(["user_id", "genre"]).size().unstack(fill_value=0)
    n_events = events.groupby("user_id").size()
    return counts.div(n_events, axis=0)


def build_user_features(events, users, items, as_of):
    g = events.groupby("user_id")
    f = pd.DataFrame({
        "user_activity_count": g.size(),
        "user_avg_rating": g["rating"].mean(),
        "user_rating_std": g["rating"].std().fillna(0.0),
        "user_days_since_last_interaction": ((as_of - g["timestamp"].max()) // SECONDS_PER_DAY).astype(int),
    })
    eg = _explode_genres(events, items)
    genre_counts = eg.groupby(["user_id", "genre"]).size().reset_index(name="n")
    top = genre_counts.sort_values(["user_id", "n", "genre"], ascending=[True, False, True]) \
                      .drop_duplicates("user_id").set_index("user_id")["genre"]
    f["user_preferred_genre"] = top
    f["user_genre_count"] = genre_counts.groupby("user_id").size()
    profile = users.set_index("user_id")[["age_group", "gender", "occupation"]] \
                   .rename(columns=lambda c: f"user_{c}")
    f = f.join(profile, how="left")
    return f.reset_index()[["user_id"] + names("user")]


def build_item_features(events, items, as_of, prior_weight):
    g = events.groupby("item_id")
    global_mean = events["rating"].mean()
    count = g.size()
    avg = g["rating"].mean()
    f = pd.DataFrame({
        "item_popularity": count,
        "item_avg_rating": avg,
        "item_bayes_rating": (count * avg + prior_weight * global_mean) / (count + prior_weight),
        "item_days_since_last_interaction": ((as_of - g["timestamp"].max()) // SECONDS_PER_DAY),
    })
    f = items.set_index("item_id")[["release_year", "genres"]].join(f, how="left")
    f["item_popularity"] = f["item_popularity"].fillna(0).astype(int)
    f["item_bayes_rating"] = f["item_bayes_rating"].fillna(global_mean)
    f["item_release_year"] = f["release_year"]
    f["item_primary_genre"] = f["genres"].str.split("|").str[0]
    f["item_genre_count"] = f["genres"].str.split("|").str.len()
    return f.reset_index()[["item_id"] + names("item")]


def genre_affinity(user_ids, item_ids, shares, items):
    """genre_affinity for arbitrary (user, item) pairs; usable at prediction time."""
    item_genres = items.set_index("item_id")["genres"].str.split("|")
    out = np.zeros(len(user_ids))
    for k, (u, i) in enumerate(zip(user_ids, item_ids)):
        if u in shares.index and i in item_genres.index:
            out[k] = np.mean([shares.at[u, gname] if gname in shares.columns else 0.0
                              for gname in item_genres[i]])
    return out


def build_interaction_features(events, items, as_of, shares):
    f = events[["user_id", "item_id", "rating", "timestamp"]].copy()
    eg = _explode_genres(events, items)
    long_shares = shares.stack().rename("share").reset_index().rename(columns={"level_1": "genre"})
    aff = eg.merge(long_shares, on=["user_id", "genre"], how="left") \
            .groupby(["user_id", "item_id"])["share"].mean().rename("genre_affinity")
    f = f.merge(aff.reset_index(), on=["user_id", "item_id"], how="left")
    f["rating_deviation"] = f["rating"] - f.groupby("user_id")["rating"].transform("mean")
    f["recency_weight"] = np.exp(-((as_of - f["timestamp"]) / SECONDS_PER_DAY) / 30.0)
    return f[["user_id", "item_id"] + names("user-item")]


def build_all(events, users, items, as_of, prior_weight):
    shares = user_genre_shares(events, items)
    return {
        "user_features": build_user_features(events, users, items, as_of),
        "item_features": build_item_features(events, items, as_of, prior_weight),
        "interaction_features": build_interaction_features(events, items, as_of, shares),
        "user_genre_shares": shares,
    }


def write_catalog(path, feature_tables):
    rows = []
    for f in FEATURES:
        table = {"user": "user_features", "item": "item_features"}.get(f.entity, "interaction_features")
        col = feature_tables[table][f.name]
        rows.append({"feature": f.name, "entity": f.entity, "type": f.dtype, "description": f.description,
                     "source": f.source, "available at prediction": "yes" if f.at_prediction else "training only",
                     "missing %": round(100 * col.isna().mean(), 2)})
    text = ["# Feature catalog", "",
            "Generated from `src/features/registry.py`. Training features are computed from training "
            "interactions only, with the reference time set to the train/test cutoff.", "",
            markdown_table(pd.DataFrame(rows), float_fmt="{:.2f}"), ""]
    path.write_text("\n".join(text), encoding="utf-8")


def run(cfg):
    logger = get_logger(cfg)
    proc = project_path(cfg["data"]["processed_dir"])
    feat_dir = project_path(cfg["data"]["features_dir"])
    reports_dir = project_path(cfg["paths"]["reports_dir"])

    train = pd.read_csv(proc / "train.csv")
    users = pd.read_csv(proc / "users.csv")
    items = pd.read_csv(proc / "items.csv")
    cutoff = read_json(proc / "split_info.json")["cutoff_timestamp"]

    tables = build_all(train, users, items, as_of=int(cutoff),
                       prior_weight=cfg["features"]["bayes_prior_weight"])
    for name in ["user_features", "item_features", "interaction_features"]:
        tables[name].to_csv(feat_dir / f"{name}.csv", index=False)
    tables["user_genre_shares"].to_csv(feat_dir / "user_genre_shares.csv")

    write_json({"as_of_timestamp": int(cutoff),
                "as_of_datetime": pd.Timestamp(int(cutoff), unit="s", tz="UTC").isoformat(),
                "built_from": "train.csv", "built_at": datetime.now(timezone.utc).isoformat(),
                "feature_count": len(FEATURES)}, feat_dir / "feature_metadata.json")
    write_catalog(reports_dir / "feature_catalog.md", tables)

    logger.info("Built %d features: %d users, %d items, %d user-item rows",
                len(FEATURES), len(tables["user_features"]), len(tables["item_features"]),
                len(tables["interaction_features"]))
    return {"features": len(FEATURES), "users": len(tables["user_features"]), "items": len(tables["item_features"])}
