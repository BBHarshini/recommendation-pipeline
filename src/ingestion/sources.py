"""Readers for the two supported data sources: MovieLens 100K and a synthetic generator."""

import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]

USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]
ITEM_COLUMNS = ["item_id", "title", "release_date", "video_release_date", "imdb_url"]
INTERACTION_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def download_movielens(url, raw_dir, logger):
    """Download the MovieLens 100K zip once; later runs reuse the local copy."""
    zip_path = Path(raw_dir) / "ml-100k.zip"
    if zip_path.exists():
        logger.info("Using cached download %s", zip_path)
    else:
        logger.info("Downloading %s", url)
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def read_movielens(zip_path):
    """Read users (u.user), items (u.item) and ratings (u.data) from the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("ml-100k/u.user") as f:
            users = pd.read_csv(f, sep="|", header=None, names=USER_COLUMNS, dtype=str)
        with zf.open("ml-100k/u.item") as f:
            items = pd.read_csv(f, sep="|", header=None, names=ITEM_COLUMNS + GENRES,
                                dtype=str, encoding="latin-1")
        with zf.open("ml-100k/u.data") as f:
            interactions = pd.read_csv(f, sep="\t", header=None, names=INTERACTION_COLUMNS, dtype=str)

    # u.item stores genres as 19 one-hot columns; keep them as one readable field
    flags = items[GENRES].astype(int)
    items["genres"] = flags.apply(lambda row: "|".join(g for g, v in zip(GENRES, row) if v == 1), axis=1)
    items = items[ITEM_COLUMNS + ["genres"]]
    return users, items, interactions


def generate_synthetic(n_users, n_items, n_ratings, seed):
    """Synthetic data with the MovieLens layout.

    Ratings come from a small latent-factor model plus noise, item popularity follows
    a long-tail distribution, and users join over time, so the time-based split and
    the models behave in a realistic way.
    """
    rng = np.random.default_rng(seed)
    occupations = ["student", "engineer", "educator", "programmer", "writer", "artist",
                   "administrator", "librarian", "technician", "other"]
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(15, 70, n_users),
        "gender": rng.choice(["M", "F"], n_users, p=[0.7, 0.3]),
        "occupation": rng.choice(occupations, n_users),
        "zip_code": [f"{z:05d}" for z in rng.integers(1000, 99999, n_users)],
    })

    real_genres = GENRES[1:]
    item_genres = []
    for _ in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        item_genres.append("|".join(sorted(rng.choice(real_genres, k, replace=False))))
    release = pd.Timestamp("1930-01-01") + pd.to_timedelta(rng.integers(0, 68 * 365, n_items), unit="D")
    items = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Movie {i} ({d.year})" for i, d in zip(range(1, n_items + 1), release)],
        "release_date": release.strftime("%d-%b-%Y"),
        "video_release_date": "",
        "imdb_url": [f"http://example.org/movie/{i}" for i in range(1, n_items + 1)],
        "genres": item_genres,
    })

    # latent tastes and item qualities
    n_factors = 6
    p = rng.normal(0, 0.6, (n_users, n_factors))
    q = rng.normal(0, 0.6, (n_items, n_factors))
    user_bias = rng.normal(0, 0.4, n_users)
    item_bias = rng.normal(0, 0.5, n_items)

    popularity = 1.0 / np.arange(1, n_items + 1) ** 0.9
    popularity = popularity[rng.permutation(n_items)]
    activity = rng.pareto(1.3, n_users) + 1

    # users join during a 7-month window and rate over the following weeks
    start = pd.Timestamp("1997-09-20").value // 10**9
    window = 210 * 86400
    join_time = start + rng.integers(0, window, n_users)

    # each user picks movies that are popular AND match their taste
    per_user = np.maximum(20, np.round(n_ratings * activity / activity.sum())).astype(int)
    per_user = np.minimum(per_user, n_items // 2)
    u_list, i_list = [], []
    for user in range(n_users):
        weight = popularity * np.exp(1.5 * (q @ p[user]))
        chosen = rng.choice(n_items, per_user[user], replace=False, p=weight / weight.sum())
        u_list.append(np.full(len(chosen), user))
        i_list.append(chosen)
    u, i = np.concatenate(u_list), np.concatenate(i_list)

    score = 3.6 + user_bias[u] + item_bias[i] + (p[u] * q[i]).sum(axis=1) + rng.normal(0, 0.7, len(u))
    rating = np.clip(np.rint(score), 1, 5).astype(int)
    ts = np.minimum(join_time[u] + rng.exponential(20 * 86400, len(u)).astype(int), start + window)

    interactions = pd.DataFrame({"user_id": u + 1, "item_id": i + 1, "rating": rating, "timestamp": ts})
    return users.astype(str), items.astype(str), interactions.astype(str)
