"""Prefect flow that orchestrates the full recommendation pipeline.

Each stage is a Prefect task. Every task receives the result of the previous one,
so Prefect runs them strictly in order and stops if a stage fails.

Run once (shows up in the Prefect UI if a server is running):
    python dags/recommendation_flow.py

Run on a schedule (every day at 02:00) until stopped with Ctrl+C:
    python dags/recommendation_flow.py --serve
"""

import os
import sys
from pathlib import Path

# make `src` importable when this file is run directly, and let Prefect capture our logs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "recsys")

from prefect import flow, task  # noqa: E402

from src.common import load_config, set_seed  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.features import build as features  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.preparation import prepare  # noqa: E402
from src.training import train  # noqa: E402
from src.transformation import transform  # noqa: E402
from src.validation import validate  # noqa: E402

TASK_OPTIONS = dict(retries=1, retry_delay_seconds=10, persist_result=False)


@task(name="1-ingest-data", **TASK_OPTIONS)
def ingest_task(cfg):
    return ingest.run(cfg)


@task(name="2-validate-data", **TASK_OPTIONS)
def validate_task(cfg, upstream):
    return validate.run(cfg)


@task(name="3-prepare-data", **TASK_OPTIONS)
def prepare_task(cfg, upstream):
    return prepare.run(cfg)


@task(name="4-transform-data", **TASK_OPTIONS)
def transform_task(cfg, upstream):
    return transform.run(cfg)


@task(name="5-build-features", **TASK_OPTIONS)
def features_task(cfg, upstream):
    return features.run(cfg)


@task(name="6-train-models", **TASK_OPTIONS)
def train_task(cfg, upstream):
    return train.run(cfg)


@task(name="7-evaluate-models", **TASK_OPTIONS)
def evaluate_task(cfg, upstream):
    return evaluate.run(cfg)


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str = "configs/config.yaml"):
    cfg = load_config(Path(__file__).resolve().parents[1] / config_path)
    set_seed(cfg["seed"])

    raw = ingest_task(cfg)
    validated = validate_task(cfg, raw)
    prepared = prepare_task(cfg, validated)
    transformed = transform_task(cfg, prepared)
    feats = features_task(cfg, transformed)
    trained = train_task(cfg, feats)
    results = evaluate_task(cfg, trained)

    print(f"Pipeline complete. NDCG@{cfg['evaluation']['k']} by model: {results}")
    return results


if __name__ == "__main__":
    if "--serve" in sys.argv:
        recommendation_pipeline.serve(name="daily-recommendation-pipeline", cron="0 2 * * *")
    else:
        recommendation_pipeline()
