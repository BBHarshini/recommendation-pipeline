"""Stage 1: Data ingestion.

Pulls users, items and user-item interactions from the configured source and stores
them unchanged (as text) in data/raw, together with a manifest describing the load.
"""

import hashlib
from datetime import datetime, timezone

from src.common import get_logger, project_path, write_json
from src.ingestion.sources import download_movielens, generate_synthetic, read_movielens
from src.ingestion.test_errors import inject_test_errors


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cfg):
    logger = get_logger(cfg)
    data_cfg = cfg["data"]
    raw_dir = project_path(data_cfg["raw_dir"])

    if data_cfg["source"] == "movielens_100k":
        zip_path = download_movielens(data_cfg["movielens_url"], raw_dir, logger)
        users, items, interactions = read_movielens(zip_path)
    elif data_cfg["source"] == "synthetic":
        s = cfg["synthetic"]
        users, items, interactions = generate_synthetic(s["n_users"], s["n_items"], s["n_ratings"], cfg["seed"])
    else:
        raise ValueError(f"Unknown data source: {data_cfg['source']}")

    logger.info("Loaded %d users, %d items, %d interactions from '%s'",
                len(users), len(items), len(interactions), data_cfg["source"])

    if data_cfg.get("inject_test_errors"):
        users, items, interactions = inject_test_errors(users, items, interactions)
        logger.info("Appended labelled test records for validation (inject_test_errors = true)")

    manifest = {"source": data_cfg["source"], "ingested_at": datetime.now(timezone.utc).isoformat(),
                "inject_test_errors": bool(data_cfg.get("inject_test_errors")), "tables": {}}
    for name, df in [("users", users), ("items", items), ("interactions", interactions)]:
        path = raw_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        manifest["tables"][name] = {"path": str(path.relative_to(raw_dir.parents[1])),
                                    "rows": len(df), "columns": list(df.columns), "sha256": _sha256(path)}

    write_json(manifest, raw_dir / "ingestion_manifest.json")
    logger.info("Raw tables written to %s", raw_dir)
    return {name: t["rows"] for name, t in manifest["tables"].items()}
