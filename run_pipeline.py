"""Run the whole pipeline (or selected stages) without an orchestrator.

    python run_pipeline.py                         # all stages
    python run_pipeline.py --from training         # resume from a stage
    python run_pipeline.py --config configs/config.yaml
"""

import argparse
import time

from src.common import get_logger, load_config, set_seed
from src.pipeline import STAGES


def main():
    parser = argparse.ArgumentParser(description="Recommendation data pipeline")
    parser.add_argument("--config", default=None, help="path to the YAML config")
    parser.add_argument("--from", dest="start", default=None, choices=[n for n, _ in STAGES],
                        help="start from this stage (earlier outputs must exist)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger(cfg)

    names = [n for n, _ in STAGES]
    first = names.index(args.start) if args.start else 0
    total = time.time()
    for name, stage in STAGES[first:]:
        start = time.time()
        logger.info("=== %s ===", name.upper())
        result = stage(cfg)
        logger.info("%s finished in %.1f s -> %s", name, time.time() - start, result)
    logger.info("Pipeline finished in %.1f s", time.time() - total)


if __name__ == "__main__":
    main()
