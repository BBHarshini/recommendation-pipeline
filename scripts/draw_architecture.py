"""Draws reports/figures/architecture.png (run: python scripts/draw_architecture.py)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"

STAGES = [
    ("1. Ingestion", "MovieLens 100K\n(users, items,\nratings)", "data/raw/"),
    ("2. Validation", "missing, duplicates,\ntypes, ratings,\ntimestamps, ref. integrity", "data/validated/\ndata/errors/"),
    ("3. Preparation", "clean, standardise,\nimpute, time-based\ntrain/test split", "data/processed/\ntrain.csv, test.csv"),
    ("4. Transformation", "user activity,\nitem popularity,\nuser-item matrix", "data/processed/\n*.csv, *.npz"),
    ("5. Features", "user, item and\nuser-item features\n+ feature catalog", "data/features/"),
    ("6. Training", "Popularity baseline\nImplicit ALS (MF)\nExplicit MF (SGD)", "models/*.pkl"),
    ("7. Evaluation", "P@K, R@K, NDCG@K,\nCoverage, Diversity,\nRMSE, MAE", "reports/"),
]

fig, ax = plt.subplots(figsize=(16, 6.2))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6.2)
ax.axis("off")

box_w, box_h, gap, x0, y_stage = 1.95, 1.75, 0.27, 0.25, 2.6
for n, (title, body, store) in enumerate(STAGES):
    x = x0 + n * (box_w + gap)
    ax.add_patch(FancyBboxPatch((x, y_stage), box_w, box_h, boxstyle="round,pad=0.03",
                                fc="#e8f0fb", ec="#2a5d9f", lw=1.4))
    ax.text(x + box_w / 2, y_stage + box_h - 0.25, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + box_w / 2, y_stage + 0.72, body, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyBboxPatch((x + 0.12, 1.0), box_w - 0.24, 0.85, boxstyle="round,pad=0.02",
                                fc="#f4f4f1", ec="#8a8a85", lw=1))
    ax.text(x + box_w / 2, 1.43, store, ha="center", va="center", fontsize=8, family="monospace")
    ax.add_patch(FancyArrowPatch((x + box_w / 2, y_stage), (x + box_w / 2, 1.88),
                                 arrowstyle="-|>", mutation_scale=10, color="#8a8a85", lw=1))
    if n < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch((x + box_w + 0.02, y_stage + box_h / 2), (x + box_w + gap - 0.02, y_stage + box_h / 2),
                                     arrowstyle="-|>", mutation_scale=14, color="#2a5d9f", lw=1.6))

ax.add_patch(FancyBboxPatch((0.25, 4.85), 15.5, 0.75, boxstyle="round,pad=0.03", fc="#fdf1e6", ec="#c0682b", lw=1.4))
ax.text(8, 5.22, "Orchestration: Prefect flow  (dags/recommendation_flow.py)  -  tasks run in order, retries, "
        "daily schedule, run history in the Prefect UI", ha="center", va="center", fontsize=10)

ax.add_patch(FancyBboxPatch((10.95, 0.08), 4.8, 0.65, boxstyle="round,pad=0.03", fc="#eaf5ea", ec="#3c8a3c", lw=1.4))
ax.text(13.35, 0.4, "MLflow: parameters, metrics, model files", ha="center", va="center", fontsize=9.5)
ax.add_patch(FancyBboxPatch((0.25, 0.08), 6.2, 0.65, boxstyle="round,pad=0.03", fc="#ffffff", ec="#8a8a85", lw=1))
ax.text(3.35, 0.4, "configs/config.yaml  -  one config + fixed seed for every stage", ha="center", va="center", fontsize=9.5)

ax.set_title("End-to-end data management pipeline for a movie recommender", fontsize=13, weight="bold", pad=4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")
