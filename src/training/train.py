"""Stage 6: Model training.

Trains every model enabled in the config on the training split, saves it to
models/<name>.pkl and logs parameters and the model file to the experiment tracker.
models/registry.json records which file and tracking run belong to each model.
"""

import pickle
import time
from datetime import datetime, timezone

import pandas as pd

from src.common import get_logger, project_path, read_json, set_seed, write_json
from src.features.store import FeatureStore
from src.tracking import Tracker
from src.training.models import MODEL_CLASSES


def run(cfg):
    logger = get_logger(cfg)
    set_seed(cfg["seed"])
    proc = project_path(cfg["data"]["processed_dir"])
    models_dir = project_path(cfg["paths"]["models_dir"])

    train = pd.read_csv(proc / "train.csv")
    items = pd.read_csv(proc / "items.csv")
    store = FeatureStore(project_path(cfg["data"]["features_dir"]), items)
    split = read_json(proc / "split_info.json")
    tracker = Tracker(cfg)

    registry = {"trained_at": datetime.now(timezone.utc).isoformat(),
                "train_cutoff": split["cutoff_datetime"], "tracking_backend": tracker.backend, "models": {}}
    min_ratings = cfg["training"]["min_item_ratings"]

    for name, params in cfg["training"]["models"].items():
        if not params.get("enabled", True):
            continue
        params = {k: v for k, v in params.items() if k != "enabled"}
        model = MODEL_CLASSES[name](min_item_ratings=min_ratings, seed=cfg["seed"], **params)

        start = time.time()
        model.fit(train, store)
        seconds = round(time.time() - start, 2)

        path = models_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

        run_id = tracker.start(run_name=name, tags={"stage": "training", "data_source": cfg["data"]["source"]})
        tracker.log_params(run_id, {**params, "model": name, "min_item_ratings": min_ratings,
                                    "seed": cfg["seed"], "train_rows": len(train),
                                    "train_cutoff": split["cutoff_datetime"]})
        tracker.log_metrics(run_id, {"training_seconds": seconds,
                                     **({"train_rmse_last_epoch": model.train_rmse[-1]}
                                        if hasattr(model, "train_rmse") else {})})
        tracker.log_artifact(run_id, path)

        registry["models"][name] = {"path": str(path.relative_to(models_dir.parent)), "run_id": run_id,
                                    "params": params, "training_seconds": seconds}
        logger.info("Trained %-13s in %6.1f s (run %s)", name, seconds, run_id)

    write_json(registry, models_dir / "registry.json")
    return {name: m["training_seconds"] for name, m in registry["models"].items()}
