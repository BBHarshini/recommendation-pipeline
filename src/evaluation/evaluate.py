"""Stage 7: Model evaluation.

Evaluation protocol
  * Users: test-period users who also appear in training and have at least one
    relevant test item (rating >= relevance_threshold on an item known from training).
  * Each model recommends K items per user, excluding items the user rated in training.
  * Ranking metrics: Precision@K, Recall@K, NDCG@K, catalogue Coverage and
    intra-list Diversity (genre-based).
  * Rating metrics (models that predict stars): RMSE and MAE on every test rating
    whose user and item both appear in training.
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import PROJECT_ROOT, get_logger, markdown_table, project_path, read_json, write_json
from src.evaluation import metrics as M
from src.tracking import Tracker


def evaluation_users(train, test, threshold):
    known_items = set(train["item_id"])
    liked = test[(test["rating"] >= threshold) & test["item_id"].isin(known_items) & test["user_id"].isin(set(train["user_id"]))]
    relevant = liked.groupby("user_id")["item_id"].apply(set).to_dict()
    return sorted(relevant), relevant


def evaluate_model(model, users, relevant, seen, k, item_genres, catalog_size, rating_pairs):
    recs = model.recommend(users, k, seen)
    lists = recs.groupby("user_id")["item_id"].apply(list).to_dict()
    per_user = [(M.precision_at_k(lists[u], relevant[u], k), M.recall_at_k(lists[u], relevant[u], k),
                 M.ndcg_at_k(lists[u], relevant[u], k), M.intra_list_diversity(lists[u], item_genres))
                for u in users]
    p, r, n, d = (float(np.mean(col)) for col in zip(*per_user))
    result = {f"precision_at_{k}": p, f"recall_at_{k}": r, f"ndcg_at_{k}": n,
              "coverage": M.catalog_coverage(lists.values(), catalog_size), "diversity": d,
              "rmse": None, "mae": None}
    if model.predicts_ratings and len(rating_pairs):
        pred = model.predict_ratings(rating_pairs["user_id"].tolist(), rating_pairs["item_id"].tolist())
        result["rmse"] = M.rmse(rating_pairs["rating"], pred)
        result["mae"] = M.mae(rating_pairs["rating"], pred)
    return result, recs


def plot_comparison(results, k, path):
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity"]
    labels = [f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", "Coverage", "Diversity"]
    x = np.arange(len(cols))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, (_, row) in enumerate(results.iterrows()):
        ax.bar(x + n * width - 0.4 + width / 2, [row[c] for c in cols], width, label=row["model"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score")
    ax.set_title(f"Model comparison on the time-based test split (K = {k})")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(cfg):
    logger = get_logger(cfg)
    proc = project_path(cfg["data"]["processed_dir"])
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    rec_dir = project_path(f"{cfg['paths']['reports_dir']}/recommendations")
    fig_dir = project_path(f"{cfg['paths']['reports_dir']}/figures")
    k = cfg["evaluation"]["k"]

    train = pd.read_csv(proc / "train.csv")
    test = pd.read_csv(proc / "test.csv")
    items = pd.read_csv(proc / "items.csv")
    registry = read_json(project_path(cfg["paths"]["models_dir"]) / "registry.json")
    tracker = Tracker(cfg)

    users, relevant = evaluation_users(train, test, cfg["split"]["relevance_threshold"])
    seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
    item_genres = {i: set(g.split("|")) for i, g in zip(items["item_id"], items["genres"])}
    catalog_size = train["item_id"].nunique()
    rating_pairs = test[test["user_id"].isin(set(train["user_id"])) & test["item_id"].isin(set(train["item_id"]))]
    logger.info("Evaluating on %d users (K = %d), %d test ratings for RMSE/MAE",
                len(users), k, len(rating_pairs))

    rows = []
    for name, entry in registry["models"].items():
        with open(PROJECT_ROOT / entry["path"], "rb") as f:
            model = pickle.load(f)
        result, recs = evaluate_model(model, users, relevant, seen, k, item_genres, catalog_size, rating_pairs)
        rec_path = rec_dir / f"{name}_top{k}.csv"
        recs.merge(items[["item_id", "title", "genres"]], on="item_id").sort_values(["user_id", "rank"]) \
            .to_csv(rec_path, index=False)
        tracker.log_metrics(entry["run_id"], {**result, "eval_users": len(users)})
        tracker.log_artifact(entry["run_id"], rec_path)
        rows.append({"model": name, **result})
        logger.info("%-13s P@%d=%.4f R@%d=%.4f NDCG@%d=%.4f cov=%.3f div=%.3f",
                    name, k, result[f"precision_at_{k}"], k, result[f"recall_at_{k}"],
                    k, result[f"ndcg_at_{k}"], result["coverage"], result["diversity"])

    results = pd.DataFrame(rows)
    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    plot_comparison(results, k, fig_dir / "model_comparison.png")

    display = results.rename(columns={f"precision_at_{k}": f"Precision@{k}", f"recall_at_{k}": f"Recall@{k}",
                                      f"ndcg_at_{k}": f"NDCG@{k}", "coverage": "Coverage",
                                      "diversity": "Diversity", "rmse": "RMSE", "mae": "MAE", "model": "Model"})
    text = ["# Model evaluation results", "",
            f"Time-based split, K = {k}. Evaluated users: {len(users)} (test users with training history and at "
            f"least one test rating >= {cfg['split']['relevance_threshold']}). RMSE/MAE use {len(rating_pairs)} "
            "test ratings; n/a means the model does not predict star ratings.", "",
            markdown_table(display), "",
            "![Model comparison](figures/model_comparison.png)", ""]
    (reports_dir / "model_results.md").write_text("\n".join(text), encoding="utf-8")
    write_json({"k": k, "eval_users": len(users), "rating_pairs": len(rating_pairs), "results": rows},
               reports_dir / "model_results.json")
    return {r["model"]: round(r[f"ndcg_at_{k}"], 4) for r in rows}
