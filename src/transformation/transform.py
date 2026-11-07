"""Stage 4: Data transformation.

Builds derived datasets from the TRAINING interactions only (so nothing from the
test period leaks into them):
  * user_activity_summary.csv  - one row per user
  * item_popularity.csv        - one row per item
  * user_item_matrix.npz       - sparse user x item rating matrix (+ index files)
"""

import numpy as np
import pandas as pd
from scipy import sparse

from src.common import get_logger, project_path


def user_activity_summary(train):
    g = train.groupby("user_id")
    summary = pd.DataFrame({
        "n_interactions": g.size(),
        "mean_rating": g["rating"].mean().round(3),
        "rating_std": g["rating"].std().fillna(0).round(3),
        "share_high_ratings": g["rating"].apply(lambda r: (r >= 4).mean()).round(3),
        "first_interaction": pd.to_datetime(g["timestamp"].min(), unit="s", utc=True),
        "last_interaction": pd.to_datetime(g["timestamp"].max(), unit="s", utc=True),
    })
    summary["active_days"] = (summary["last_interaction"] - summary["first_interaction"]).dt.days + 1
    return summary.reset_index()


def item_popularity(train, items):
    g = train.groupby("item_id")
    pop = pd.DataFrame({"n_ratings": g.size(), "mean_rating": g["rating"].mean().round(3),
                        "n_unique_users": g["user_id"].nunique()}).reset_index()
    pop["share_of_interactions"] = (pop["n_ratings"] / len(train)).round(6)
    pop["popularity_rank"] = pop["n_ratings"].rank(ascending=False, method="min").astype(int)
    pop = items[["item_id", "title", "genres"]].merge(pop, on="item_id", how="left")
    pop[["n_ratings", "n_unique_users"]] = pop[["n_ratings", "n_unique_users"]].fillna(0).astype(int)
    return pop.sort_values("n_ratings", ascending=False)


def user_item_matrix(train):
    user_ids = np.sort(train["user_id"].unique())
    item_ids = np.sort(train["item_id"].unique())
    u_index = {u: i for i, u in enumerate(user_ids)}
    i_index = {it: j for j, it in enumerate(item_ids)}
    rows = train["user_id"].map(u_index).to_numpy()
    cols = train["item_id"].map(i_index).to_numpy()
    matrix = sparse.csr_matrix((train["rating"].to_numpy(dtype=np.float32), (rows, cols)),
                               shape=(len(user_ids), len(item_ids)))
    return matrix, user_ids, item_ids


def run(cfg):
    logger = get_logger(cfg)
    proc = project_path(cfg["data"]["processed_dir"])
    train = pd.read_csv(proc / "train.csv")
    items = pd.read_csv(proc / "items.csv")

    activity = user_activity_summary(train)
    popularity = item_popularity(train, items)
    matrix, user_ids, item_ids = user_item_matrix(train)

    activity.to_csv(proc / "user_activity_summary.csv", index=False)
    popularity.to_csv(proc / "item_popularity.csv", index=False)
    sparse.save_npz(proc / "user_item_matrix.npz", matrix)
    pd.DataFrame({"row": range(len(user_ids)), "user_id": user_ids}).to_csv(proc / "matrix_user_index.csv", index=False)
    pd.DataFrame({"col": range(len(item_ids)), "item_id": item_ids}).to_csv(proc / "matrix_item_index.csv", index=False)

    density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    logger.info("User-item matrix %d x %d, %d ratings, density %.2f%%",
                matrix.shape[0], matrix.shape[1], matrix.nnz, 100 * density)
    return {"matrix_shape": list(matrix.shape), "density": round(density, 4),
            "users_summarised": len(activity), "items_summarised": len(popularity)}
