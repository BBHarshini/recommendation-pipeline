"""Shared helpers: configuration, paths, logging and small I/O utilities."""

import json
import logging
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
LOGGER_NAME = "recsys"


def load_config(path=None):
    """Read the YAML config and return it as a dictionary."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative):
    """Resolve a path from the config relative to the project root and make sure it exists."""
    path = PROJECT_ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_logger(cfg=None):
    """Logger that writes to the console and to logs/pipeline.log."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logs_dir = project_path((cfg or {}).get("paths", {}).get("logs_dir", "logs"))
    file_handler = logging.FileHandler(logs_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def markdown_table(df, float_fmt="{:.4f}"):
    """Render a DataFrame as a GitHub markdown table (avoids the optional 'tabulate' dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(float_fmt.format(value) if value == value else "n/a")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
